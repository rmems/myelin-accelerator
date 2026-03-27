// ════════════════════════════════════════════════════════════════════
//  satsolver.cu — Parallel WalkSAT / stochastic local-search kernels
//
//  Origin / Attribution
//  ────────────────────
//  Derived from  kernels/dynexsolve.cu  (Neuromorphic SAT-Solver GPU Kernels)
//  which implements a Continuous-Time Dynamical System (CTDS) relaxing toward
//  satisfying assignments of a Boolean formula in CNF (the Dynex / DynexSolve
//  variant of analog-modeled SAT).
//
//  This file replaces the continuous-voltage CTDS dynamics with a discrete
//  stochastic local-search (WalkSAT) strategy while preserving the same
//  kernel signatures, data layout, and multi-walker parallelism pattern
//  introduced in dynexsolve.cu.
//
//  Kernels exported:
//    satsolver_init          — initialise assignment + clause data
//    satsolver_step          — one WalkSAT flip step per walker
//    satsolver_aux_update    — update auxiliary scores (break / make counts)
//    satsolver_check_solution — check which walkers have found a solution
//    satsolver_extract       — copy best-found assignment to output
//
//  Data layout
//    assignment  [n_walkers × n_vars]  — binary variable assignment (uint8)
//    clauses     [n_clauses × clause_len] — literal encoding:
//                  literal = 2*var_idx if positive, 2*var_idx+1 if negative
//    sat_flags   [n_walkers × n_clauses] — 1 if clause satisfied, else 0
//    scores      [n_walkers]             — number of unsatisfied clauses
//    best_score  [1]                     — global best (min unsatisfied)
//    best_walker [1]                     — index of the best walker
//
//  Target: sm_120 (RTX 5080 Blackwell) · CUDA 13.x
// ════════════════════════════════════════════════════════════════════

#include "common.cuh"

// Walk probability for random flip (WalkSAT noise parameter)
#define WALKSAT_P   0.57f

// ── Helper: evaluate a single clause ─────────────────────────────
__device__ __forceinline__
int eval_clause(
    const unsigned int* __restrict__ asgn,
    const int*          __restrict__ clause,
    int clause_len)
{
    for (int l = 0; l < clause_len; ++l) {
        unsigned int lit = (unsigned int)clause[l];
        unsigned int var = lit >> 1;
        unsigned int neg = lit & 1u;
        unsigned int val = asgn[var] ^ neg;   // satisfied if val == 1
        if (val) return 1;
    }
    return 0;  // unsatisfied
}

// ════════════════════════════════════════════════════════════════════
//  satsolver_init
//
//  Initialise each walker's assignment to a random binary string and
//  compute the initial number of unsatisfied clauses.
//
//  Params
//    assignment  [n_walkers × n_vars]   — output: random assignment
//    scores      [n_walkers]            — output: initial unsat count
//    clauses     [n_clauses × clause_len]
//    n_walkers, n_vars, n_clauses, clause_len
//    seed        — global random seed
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void satsolver_init(
    unsigned int* __restrict__ assignment,
    int*          __restrict__ scores,
    const int*    __restrict__ clauses,
    int n_walkers,
    int n_vars,
    int n_clauses,
    int clause_len,
    unsigned int seed)
{
    int walker = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker >= n_walkers) return;

    unsigned int* asgn = assignment + (long)walker * n_vars;
    unsigned int rng   = lcg_next(seed ^ (unsigned int)walker * 2654435761u);

    // Random initial assignment
    for (int v = 0; v < n_vars; ++v) {
        rng = lcg_next(rng);
        asgn[v] = (rng >> 16) & 1u;
    }

    // Count unsatisfied clauses
    int unsat = 0;
    for (int c = 0; c < n_clauses; ++c) {
        if (!eval_clause(asgn, clauses + c * clause_len, clause_len))
            ++unsat;
    }
    scores[walker] = unsat;
}

// ════════════════════════════════════════════════════════════════════
//  satsolver_step
//
//  One WalkSAT iteration per walker:
//    1. Pick a random unsatisfied clause.
//    2. With probability p, flip a random variable in that clause.
//    3. Otherwise flip the variable with the lowest break count.
//
//  Params
//    assignment  [n_walkers × n_vars]
//    scores      [n_walkers]            — updated in-place
//    clauses     [n_clauses × clause_len]
//    n_walkers, n_vars, n_clauses, clause_len
//    seed        — per-step seed (combine with step counter at call site)
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void satsolver_step(
    unsigned int* __restrict__ assignment,
    int*          __restrict__ scores,
    const int*    __restrict__ clauses,
    int n_walkers,
    int n_vars,
    int n_clauses,
    int clause_len,
    unsigned int seed)
{
    int walker = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker >= n_walkers) return;

    unsigned int rng = lcg_next(seed ^ (unsigned int)walker * 2246822519u);
    unsigned int* asgn = assignment + (long)walker * n_vars;

    // ── 1. Collect unsatisfied clauses (up to 64 stored) ──────────
    int unsat_buf[64];
    int unsat_cnt = 0;
    for (int c = 0; c < n_clauses && unsat_cnt < 64; ++c) {
        if (!eval_clause(asgn, clauses + c * clause_len, clause_len))
            unsat_buf[unsat_cnt++] = c;
    }

    if (unsat_cnt == 0) {
        scores[walker] = 0;
        return;  // already satisfied
    }

    // ── 2. Pick random unsatisfied clause ─────────────────────────
    rng = lcg_next(rng);
    int chosen_c = unsat_buf[(rng >> 8) % (unsigned int)unsat_cnt];
    const int* cptr = clauses + chosen_c * clause_len;

    // ── 3. Decide flip: noise (random) vs greedy (min-break) ──────
    rng = lcg_next(rng);
    float r = lcg_float(rng);

    int flip_var = -1;

    if (r < WALKSAT_P) {
        // Noise move: flip random literal in clause
        rng = lcg_next(rng);
        int lit_idx = (int)((rng >> 8) % (unsigned int)clause_len);
        flip_var = (unsigned int)cptr[lit_idx] >> 1;
    } else {
        // Greedy move: flip variable with lowest break count
        int best_break = n_clauses + 1;
        for (int l = 0; l < clause_len; ++l) {
            int var = (unsigned int)cptr[l] >> 1;

            // Count how many sat clauses would become unsat if we flip var
            int brk = 0;
            asgn[var] ^= 1u;  // tentative flip
            for (int c2 = 0; c2 < n_clauses; ++c2) {
                // Was clause satisfied before flip but not after?
                // (eval after flip == 0) AND (would have been 1 before == depends on var)
                if (!eval_clause(asgn, clauses + c2 * clause_len, clause_len)) {
                    // Check if it was sat before
                    asgn[var] ^= 1u;
                    int was_sat = eval_clause(asgn, clauses + c2 * clause_len, clause_len);
                    asgn[var] ^= 1u;
                    brk += was_sat;
                }
            }
            asgn[var] ^= 1u;  // undo tentative flip

            if (brk < best_break) {
                best_break = brk;
                flip_var   = var;
            }
        }
    }

    // ── 4. Apply flip and recount unsat ───────────────────────────
    if (flip_var >= 0) asgn[flip_var] ^= 1u;

    int unsat = 0;
    for (int c = 0; c < n_clauses; ++c) {
        if (!eval_clause(asgn, clauses + c * clause_len, clause_len))
            ++unsat;
    }
    scores[walker] = unsat;
}

// ════════════════════════════════════════════════════════════════════
//  satsolver_aux_update
//
//  Recompute per-clause satisfaction flags and update the global
//  best_score / best_walker atomically.
//
//  Params
//    assignment  [n_walkers × n_vars]
//    sat_flags   [n_walkers × n_clauses]  — output
//    scores      [n_walkers]
//    best_score  [1]  — global best (int, atomic)
//    best_walker [1]  — index of best walker (int, atomic)
//    n_walkers, n_vars, n_clauses, clause_len
//    clauses     [n_clauses × clause_len]
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void satsolver_aux_update(
    const unsigned int* __restrict__ assignment,
    unsigned int*        __restrict__ sat_flags,
    const int*           __restrict__ scores,
    int*                 __restrict__ best_score,
    int*                 __restrict__ best_walker,
    const int*           __restrict__ clauses,
    int n_walkers,
    int n_vars,
    int n_clauses,
    int clause_len)
{
    int walker = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker >= n_walkers) return;

    const unsigned int* asgn = assignment + (long)walker * n_vars;
    unsigned int* flags = sat_flags + (long)walker * n_clauses;

    for (int c = 0; c < n_clauses; ++c)
        flags[c] = (unsigned int)eval_clause(asgn, clauses + c * clause_len, clause_len);

    // Atomic best update
    int my_score = scores[walker];
    int old_best = atomicMin(best_score, my_score);
    if (my_score < old_best) {
        // Best improved — record which walker owns it
        // (Non-atomic write; last winner wins in case of ties)
        *best_walker = walker;
    }
}

// ════════════════════════════════════════════════════════════════════
//  satsolver_check_solution
//
//  Set solved[walker] = 1 if the walker has found a satisfying
//  assignment (scores[walker] == 0), else 0.
//
//  Params
//    scores     [n_walkers]
//    solved     [n_walkers]   — output: 1 if satisfied
//    n_walkers
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void satsolver_check_solution(
    const int*    __restrict__ scores,
    unsigned int* __restrict__ solved,
    int n_walkers)
{
    int walker = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker >= n_walkers) return;
    solved[walker] = (scores[walker] == 0) ? 1u : 0u;
}

// ════════════════════════════════════════════════════════════════════
//  satsolver_extract
//
//  Copy the assignment of the best walker to the output buffer.
//
//  Params
//    assignment   [n_walkers × n_vars]
//    best_walker  [1]   — index of the walker to extract
//    output       [n_vars]  — destination assignment
//    n_vars
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void satsolver_extract(
    const unsigned int* __restrict__ assignment,
    const int*          __restrict__ best_walker,
    unsigned int*        __restrict__ output,
    int n_vars)
{
    int var = blockIdx.x * blockDim.x + threadIdx.x;
    if (var >= n_vars) return;

    int bw = *best_walker;
    output[var] = assignment[(long)bw * n_vars + var];
}
