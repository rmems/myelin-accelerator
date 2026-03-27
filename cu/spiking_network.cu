// ════════════════════════════════════════════════════════════════════
//  spiking_network.cu — LIF / STDP spiking-network CUDA kernels
//
//  Kernels exported (name-exact for PTX symbol lookup in kernel.rs):
//    poisson_encode       — rate-coded Poisson spike train
//    lif_step             — LIF neuron step (unweighted)
//    lif_step_weighted    — LIF step with synaptic weight matrix
//    spike_rate           — windowed firing-rate estimator
//    reset_membrane       — reset membrane to resting potential
//    stdp_update          — spike-timing-dependent plasticity
//    neuro_bias_logits    — add neuromodulator bias to logit array
//
//  Parameters follow the 16-neuron / 16-channel architecture in
//  neuro-spike-core/src/snn/engine.rs.
//
//  Target: sm_120 (RTX 5080 Blackwell) · CUDA 13.x
// ════════════════════════════════════════════════════════════════════

#include "common.cuh"

// ── LIF model constants ───────────────────────────────────────────
#define LIF_DECAY        0.85f   // membrane decay per tick (1 - dt/tau_m)
#define LIF_THRESHOLD    1.0f    // normalised firing threshold
#define LIF_RESET        0.0f    // reset potential after spike
#define LIF_REFRACT_TICK 2       // integer ticks of absolute refractory period

// ── STDP constants (match stdp.rs) ───────────────────────────────
#define STDP_A_PLUS   0.01f
#define STDP_A_MINUS  0.012f
#define STDP_TAU      20.0f
#define STDP_W_MIN    0.0f
#define STDP_W_MAX    2.0f

// ════════════════════════════════════════════════════════════════════
//  poisson_encode
//
//  Convert a normalised float stimulus [0,1] into a Poisson spike (0/1)
//  for each input channel.
//
//  Params
//    stimuli   [n_channels] — normalised input rates in [0,1]
//    spikes    [n_channels] — output: 1 if spike fired, 0 otherwise
//    n         — number of channels to process
//    seed      — per-tick random seed (XOR with thread idx for variety)
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void poisson_encode(
    const float* __restrict__ stimuli,
    unsigned int* __restrict__ spikes,
    int n,
    unsigned int seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    unsigned int rng = lcg_next(seed ^ (unsigned int)tid);
    float threshold = stimuli[tid];
    threshold = fmaxf(0.0f, fminf(1.0f, threshold));

    float r = lcg_float(rng);
    spikes[tid] = (r < threshold) ? 1u : 0u;
}

// ════════════════════════════════════════════════════════════════════
//  lif_step
//
//  LIF neuron update without synaptic weights — each neuron receives
//  a pre-summed current `I_ext`.
//
//  Params
//    membrane    [n_neurons]  — membrane potential (read-write)
//    I_ext       [n_neurons]  — external current injection
//    refract     [n_neurons]  — refractory counter (uint, read-write)
//    spikes_out  [n_neurons]  — output spike flags (0/1)
//    n_neurons   — number of LIF neurons
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void lif_step(
    float* __restrict__        membrane,
    const float* __restrict__  I_ext,
    unsigned int* __restrict__ refract,
    unsigned int* __restrict__ spikes_out,
    int n_neurons)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons) return;

    unsigned int ref = refract[tid];
    float v = membrane[tid];
    unsigned int spike = 0u;

    if (ref > 0u) {
        // Absolute refractory period: clamp membrane at reset
        v = LIF_RESET;
        refract[tid] = ref - 1u;
    } else {
        // Leaky integration
        v = fmaf(LIF_DECAY, v, I_ext[tid]);

        if (v >= LIF_THRESHOLD) {
            spike = 1u;
            v = LIF_RESET;
            refract[tid] = (unsigned int)LIF_REFRACT_TICK;
        }
    }

    membrane[tid]   = v;
    spikes_out[tid] = spike;
}

// ════════════════════════════════════════════════════════════════════
//  lif_step_weighted
//
//  LIF neuron update with a full weight matrix.  Each neuron's input
//  current is the dot product of input_spikes and its weight row.
//
//  Uses shared memory to cache a tile of the weight row.
//
//  Params
//    membrane      [n_neurons]              — read-write
//    weights       [n_neurons × n_inputs]  — row-major weight matrix
//    input_spikes  [n_inputs]               — binary spike inputs (float 0/1)
//    refract       [n_neurons]              — refractory counter
//    spikes_out    [n_neurons]              — output spikes
//    n_neurons, n_inputs
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void lif_step_weighted(
    float* __restrict__        membrane,
    const float* __restrict__  weights,
    const float* __restrict__  input_spikes,
    unsigned int* __restrict__ refract,
    unsigned int* __restrict__ spikes_out,
    int n_neurons,
    int n_inputs)
{
    extern __shared__ float s_inputs[];   // n_inputs floats cached in SMEM

    // Cooperatively load input spikes into shared memory
    for (int i = threadIdx.x; i < n_inputs; i += blockDim.x)
        s_inputs[i] = input_spikes[i];
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons) return;

    unsigned int ref = refract[tid];
    float v = membrane[tid];
    unsigned int spike = 0u;

    if (ref > 0u) {
        v = LIF_RESET;
        refract[tid] = ref - 1u;
    } else {
        // Dot product: I = W[tid, :] · spikes
        const float* w_row = weights + (long)tid * n_inputs;
        float I = 0.0f;
        for (int j = 0; j < n_inputs; ++j)
            I = fmaf(w_row[j], s_inputs[j], I);

        v = fmaf(LIF_DECAY, v, I);

        if (v >= LIF_THRESHOLD) {
            spike = 1u;
            v = LIF_RESET;
            refract[tid] = (unsigned int)LIF_REFRACT_TICK;
        }
    }

    membrane[tid]   = v;
    spikes_out[tid] = spike;
}

// ════════════════════════════════════════════════════════════════════
//  spike_rate
//
//  Estimate windowed firing rate for each neuron by accumulating a
//  decaying trace.
//
//  rate[i] = ALPHA * rate[i] + (1 - ALPHA) * spike[i]
//
//  Params
//    rates       [n_neurons] — exponential rate estimate (read-write)
//    spikes      [n_neurons] — current spike flags (float 0/1)
//    n_neurons
//    alpha       — trace decay coefficient (e.g. 0.95)
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void spike_rate(
    float* __restrict__       rates,
    const float* __restrict__ spikes,
    int n_neurons,
    float alpha)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons) return;

    float r = rates[tid];
    float s = spikes[tid];
    rates[tid] = fmaf(alpha, r, fmaf(1.0f - alpha, s, 0.0f));
}

// ════════════════════════════════════════════════════════════════════
//  reset_membrane
//
//  Unconditionally reset all membrane potentials to `v_reset`.
//  Used at episode boundaries or full network resets.
//
//  Params
//    membrane   [n_neurons] — membrane array
//    n_neurons
//    v_reset    — target reset value (typically 0.0)
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void reset_membrane(
    float* __restrict__ membrane,
    int n_neurons,
    float v_reset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons) return;
    membrane[tid] = v_reset;
}

// ════════════════════════════════════════════════════════════════════
//  stdp_update
//
//  Pairwise STDP weight update:
//    If post spiked:  dw += A+ * exp(-|dt| / tau)  (LTP)
//    If pre  spiked:  dw -= A- * exp(-|dt| / tau)  (LTD)
//
//  Weights are clamped to [W_MIN, W_MAX] after each update.
//
//  Params
//    weights       [n_post × n_pre]  — synaptic weight matrix (row-major)
//    pre_spikes    [n_pre]           — binary pre-synaptic spike flags
//    post_spikes   [n_post]          — binary post-synaptic spike flags
//    pre_traces    [n_pre]           — pre-synaptic eligibility traces
//    post_traces   [n_post]          — post-synaptic eligibility traces
//    n_post, n_pre
//    dt_ms         — time elapsed since last pair event (ms)
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void stdp_update(
    float* __restrict__       weights,
    const float* __restrict__ pre_spikes,
    const float* __restrict__ post_spikes,
    float* __restrict__       pre_traces,
    float* __restrict__       post_traces,
    int n_post,
    int n_pre,
    float dt_ms)
{
    // One thread per (post, pre) synapse — 2-D grid if needed
    int post = blockIdx.y * blockDim.y + threadIdx.y;
    int pre  = blockIdx.x * blockDim.x + threadIdx.x;
    if (post >= n_post || pre >= n_pre) return;

    // Decay factor shared across the warp row
    float decay = __expf(-dt_ms / STDP_TAU);

    // Update traces (only threads on diagonal do the scalar work;
    // all threads read from pre/post arrays already in L2)
    float pre_tr, post_tr;
    if (pre < n_pre)  pre_tr  = fmaf(decay, pre_traces[pre],   pre_spikes[pre]);
    if (post < n_post) post_tr = fmaf(decay, post_traces[post], post_spikes[post]);

    // Write traces (avoid races: each row-thread writes its own pre)
    if (threadIdx.y == 0 && pre < n_pre)   pre_traces[pre]   = pre_tr;
    if (threadIdx.x == 0 && post < n_post) post_traces[post] = post_tr;

    // Weight update
    long idx = (long)post * n_pre + pre;
    float w = weights[idx];

    float ps = post_spikes[post];
    float prs = pre_spikes[pre];

    // LTP: post fired — potentiate weights from active pre traces
    w += STDP_A_PLUS  * ps  * pre_tr;
    // LTD: pre fired  — depress weights to inactive post traces
    w -= STDP_A_MINUS * prs * post_tr;

    w = fmaxf(STDP_W_MIN, fminf(STDP_W_MAX, w));
    weights[idx] = w;
}

// ════════════════════════════════════════════════════════════════════
//  neuro_bias_logits
//
//  Add a neuromodulator-scaled bias vector to a logit array.
//  logits[i] += scale * bias[i]
//
//  Used to inject dopamine / cortisol / ACh bias into the readout
//  layer before softmax.
//
//  Params
//    logits   [n]  — float logit array (read-write)
//    bias     [n]  — per-element bias template
//    n
//    scale    — scalar modulator gain
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void neuro_bias_logits(
    float* __restrict__       logits,
    const float* __restrict__ bias,
    int n,
    float scale)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    logits[tid] = fmaf(scale, bias[tid], logits[tid]);
}
