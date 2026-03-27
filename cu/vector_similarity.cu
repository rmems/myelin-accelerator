// ════════════════════════════════════════════════════════════════════
//  vector_similarity.cu — GPU cosine-similarity kernels
//
//  Kernels exported:
//    cosine_similarity_batched  — compute similarity of Q query vectors
//                                 against K key vectors (Q×K matrix)
//    cosine_similarity_top_k    — like above but also write top-k indices
//
//  Target: sm_120 (RTX 5080 Blackwell) · CUDA 13.x
// ════════════════════════════════════════════════════════════════════

#include "common.cuh"

// ════════════════════════════════════════════════════════════════════
//  cosine_similarity_batched
//
//  Computes the cosine similarity between every (query, key) pair:
//    out[q * n_keys + k] = dot(Q[q], K[k]) / (|Q[q]| * |K[k]| + eps)
//
//  Grid: (n_keys, n_queries) blocks of (dim,) threads — one block per pair.
//        For large dim use TILE_DIM threads and loop.
//
//  Params
//    queries    [n_queries × dim]  — query matrix (row-major)
//    keys       [n_keys   × dim]  — key matrix
//    out        [n_queries × n_keys] — output similarity matrix
//    n_queries, n_keys, dim
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void cosine_similarity_batched(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    float*       __restrict__ out,
    int n_queries,
    int n_keys,
    int dim)
{
    // Block index identifies (query, key) pair
    int q = blockIdx.y;
    int k = blockIdx.x;
    if (q >= n_queries || k >= n_keys) return;

    const float* qv = queries + (long)q * dim;
    const float* kv = keys    + (long)k * dim;

    // Each thread accumulates a partial dot product and L2 norms
    float dot   = 0.0f;
    float norm_q = 0.0f;
    float norm_k = 0.0f;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float qi = qv[i];
        float ki = kv[i];
        dot    = fmaf(qi, ki, dot);
        norm_q = fmaf(qi, qi, norm_q);
        norm_k = fmaf(ki, ki, norm_k);
    }

    // Warp-level reduction
    dot    = warp_reduce_sum(dot);
    norm_q = warp_reduce_sum(norm_q);
    norm_k = warp_reduce_sum(norm_k);

    // Block-level reduction via shared memory
    __shared__ float s_dot[WARP_SIZE];
    __shared__ float s_nq [WARP_SIZE];
    __shared__ float s_nk [WARP_SIZE];

    int lane  = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int nWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    if (lane == 0) {
        s_dot[warpId] = dot;
        s_nq [warpId] = norm_q;
        s_nk [warpId] = norm_k;
    }
    __syncthreads();

    if (threadIdx.x < nWarps) {
        dot    = s_dot[threadIdx.x];
        norm_q = s_nq [threadIdx.x];
        norm_k = s_nk [threadIdx.x];
    } else {
        dot    = 0.0f;
        norm_q = 0.0f;
        norm_k = 0.0f;
    }

    if (warpId == 0) {
        dot    = warp_reduce_sum(dot);
        norm_q = warp_reduce_sum(norm_q);
        norm_k = warp_reduce_sum(norm_k);
    }

    if (threadIdx.x == 0) {
        float denom = sqrtf(norm_q) * sqrtf(norm_k) + SHIP_EPS;
        out[(long)q * n_keys + k] = dot / denom;
    }
}

// ════════════════════════════════════════════════════════════════════
//  cosine_similarity_top_k
//
//  Like cosine_similarity_batched but for each query also writes
//  the top-K key indices (by similarity score) into `top_k_indices`.
//
//  Implementation: compute full row of similarities, then one warp
//  does a simple selection sort for small K (K ≤ 32).
//
//  Params
//    queries         [n_queries × dim]
//    keys            [n_keys   × dim]
//    similarities    [n_queries × n_keys]  — full similarity matrix output
//    top_k_indices   [n_queries × top_k]   — indices of top-k keys per query
//    n_queries, n_keys, dim, top_k
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void cosine_similarity_top_k(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    float*       __restrict__ similarities,
    int*         __restrict__ top_k_indices,
    int n_queries,
    int n_keys,
    int dim,
    int top_k)
{
    // One block per query; threads cooperate over keys
    int q = blockIdx.x;
    if (q >= n_queries) return;

    const float* qv = queries + (long)q * dim;

    extern __shared__ float s_mem[];  // n_keys floats for similarity row
    float* s_sim = s_mem;

    // Compute similarity for all keys
    for (int k = threadIdx.x; k < n_keys; k += blockDim.x) {
        const float* kv = keys + (long)k * dim;

        float dot   = 0.0f;
        float norm_q = 0.0f;
        float norm_k = 0.0f;

        for (int i = 0; i < dim; ++i) {
            float qi = qv[i];
            float ki = kv[i];
            dot    = fmaf(qi, ki, dot);
            norm_q = fmaf(qi, qi, norm_q);
            norm_k = fmaf(ki, ki, norm_k);
        }

        float denom = sqrtf(norm_q) * sqrtf(norm_k) + SHIP_EPS;
        s_sim[k] = dot / denom;
        similarities[(long)q * n_keys + k] = s_sim[k];
    }
    __syncthreads();

    // Thread 0 selects top-k indices (simple selection, K ≤ WARP_SIZE)
    if (threadIdx.x == 0) {
        int actual_k = (top_k < n_keys) ? top_k : n_keys;
        int* out_idx = top_k_indices + q * top_k;

        // Track which keys have been selected
        // Using a small visited bitmask stored in registers (up to 32 keys)
        for (int t = 0; t < actual_k; ++t) {
            float best_val = -2.0f;
            int   best_idx = -1;

            for (int k = 0; k < n_keys; ++k) {
                // Skip already-selected (mark as -2.0 after selection)
                if (s_sim[k] > best_val) {
                    best_val = s_sim[k];
                    best_idx = k;
                }
            }

            out_idx[t] = best_idx;
            if (best_idx >= 0) s_sim[best_idx] = -2.0f; // mark selected
        }

        // Fill remaining slots with -1 if top_k > actual_k
        for (int t = actual_k; t < top_k; ++t)
            out_idx[t] = -1;
    }
}
