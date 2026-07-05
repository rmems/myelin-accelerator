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

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a high-level architecture overview, crate boundaries, and the CUDA build/loading pipeline.

## Module Map

- `src/lib.rs`: crate root; feature-gated exports and public API surface.
- `src/bitpacking.rs`: CPU-side bitpacking helpers used by higher-level code.
- `src/gpu/`: CUDA-backed implementation behind the `cuda` feature.
  - `context.rs`: CUDA driver/runtime initialization and device context management.
  - `kernel.rs`: PTX module loading and kernel symbol resolution.
  - `memory.rs`: device buffer allocation and transfers.
  - `accelerator.rs`: high-level GPU entry points that orchestrate launches.
  - `error.rs`: GPU/driver error types.
- `src/gpu_stub.rs`: stub implementation used when the `cuda` feature is disabled.
- `cu/`: CUDA kernel sources compiled to PTX by `build.rs`.

## Usage

```toml
[dependencies]
myelin-accelerator = "0.1.0"
```

## License

Licensed under either of:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.
