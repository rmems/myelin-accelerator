# Myelin-Accelerator

Blackwell-first CUDA kernels for neuromorphic inference, SAT search, and routing-heavy GPU workloads on RTX 5080-class hardware.

This repo is the low-level compute layer behind the stack: CUDA PTX modules, Rust bindings, and the launch paths that keep the GPU busy instead of serializing work through one thread at a time.

## Current State

- `sm_120` target: the kernels are built and tuned for RTX 5080 / Blackwell.
- `16 GB VRAM` discipline: kernel footprints are kept static and bounded; the big memory costs are in user-managed tensors, not temporary scratch.
- `SAT path`: `atomicMin` is gone from the hot reduction path.
- `Routing path`: `cosine_similarity_top_k` now uses warp-participating top-k reduction instead of a single-thread selection tail.
- `Rust FFI`: kernel symbols are loaded through `src/gpu/kernel.rs`; the codebase stays ABI-consistent with the CUDA side.

## What Changed

- `satsolver.cu` now reduces `(score, walker)` with warp shuffles and shared memory, then writes one result per block before a final reduction pass.
- `vector_similarity.cu` now keeps top-k selection parallel across the warp, with register-resident candidates and block-local merge stages.
- `.gitignore` now excludes generated PTX, build outputs, CMake artifacts, IDE files, and other local-only clutter.

## Why It Matters

- Better SM occupancy on Blackwell.
- No global atomic serialization in the SAT best-score path.
- No single-thread MoE routing bottleneck.
- Lower noise in the repo from generated files that do not belong in source control.

## Provenance

Recent kernel work in this branch was authored with GPT-5.4 (`xhigh`) assistance.

## Usage

```toml
[dependencies]
myelin-accelerator = "0.9.2"
```

## License

GPL-3.0. See [LICENSE](LICENSE) for details.
