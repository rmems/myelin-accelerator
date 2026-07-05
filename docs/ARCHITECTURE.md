# Architecture

This crate is the low-level GPU compute layer for the Myelin stack: CUDA kernels compiled to PTX plus a thin Rust API that loads modules and launches kernels.

## High-level layers

1. **CUDA kernels (`cu/`)**
   - CUDA `.cu` files implement the performance-critical primitives (SAT search reductions, routing/top-k, spiking network kernels).
   - `cu/common.cuh` contains shared device-side helpers.

2. **Build pipeline (`build.rs`)**
   - When the `cuda` feature is enabled, `build.rs` uses `nvcc` to compile the CUDA sources into PTX in `OUT_DIR`.
   - When the `cuda` feature is disabled, `build.rs` emits *stub PTX* so the crate can still build without a CUDA toolchain.
   - Environment knobs:
     - `MYELIN_CUDA_ARCH` (default `sm_120`)
     - `MYELIN_PTX_VERSION` (defaults to the stub PTX version in `build.rs`)
     - `CUDA_NVCC` / `CUDA_HOME` / `CUDA_PATH` to locate `nvcc`

3. **Rust API (`src/`)**
   - `src/gpu/` is compiled when `--features cuda` is enabled.
   - `src/gpu_stub.rs` provides a buildable (non-CUDA) fallback API when `cuda` is not enabled.

## Boundary and responsibilities

- **CUDA code owns**: device-side algorithms, shared-memory layouts, warp/block reductions, and anything that must be tuned to a specific GPU architecture.
- **Rust code owns**: module loading, symbol resolution, buffer management, launch orchestration, and presenting a safe(er) API to callers.
- **FFI boundary**: Rust launches kernels by loading PTX and calling exported kernel symbols; ABI/launch signatures must remain consistent across `cu/` and `src/gpu/kernel.rs`.

## Where to look

- Kernel compilation: `build.rs`
- PTX loading + symbol resolution: `src/gpu/kernel.rs`
- Public entry points: `src/lib.rs` and `src/gpu/accelerator.rs`
