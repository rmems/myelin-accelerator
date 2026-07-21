# CLAUDE.md

Guidance for Claude (and other AI coding agents) working in this repository.

## Project

`myelin-accelerator` is a Rust crate exposing safe FFI wrappers around a set
of hand-written CUDA kernels (spiking-network simulation, vector similarity
search, and a SAT solver), targeting Blackwell / RTX 5080-class GPUs
(`sm_120`). It builds cleanly with or without a CUDA toolkit present via the
`cuda` Cargo feature:

- **Default (`cuda` feature off):** `src/gpu_stub.rs` is used, `build.rs`
  writes stub PTX files, no GPU/nvcc required. This is the CPU-safe path and
  should always work in CI and sandboxed environments.
- **`--features cuda`:** `src/gpu/` is compiled, `build.rs` invokes `nvcc` to
  compile `cu/*.cu` → PTX, embedded at compile time via `include_str!`.
  Prefer **CUDA toolkit 13.2+** (local baseline **13.3** on ShipOfTheseus
  when installed); keep `CUDA_NVCC` / `CUDA_HOME` pointed at the active
  tree if `/usr/local/cuda` lags. Runtime needs an `sm_120`-capable driver
  (≥ 570; UMD **13.x** on current Blackwell hosts).

## Build and test

Run CPU-safe checks first — these must always pass and require no GPU:

```bash
cargo test --locked
cargo build --locked --no-default-features
```

If a CUDA toolkit is available, also verify the GPU path (set `CUDA_NVCC` if
`nvcc` isn't on `PATH`):

```bash
CUDA_NVCC=/usr/local/cuda/bin/nvcc cargo build --locked --features cuda
```

### CMake / CLion integration

`CMakeLists.txt` wires `nvcc` through custom commands (not CMake's native
`CUDA` language — see `REVIEW.md` for why) so CLion can drive both the CUDA
PTX compilation and the Cargo test suite via CTest:

```bash
cmake --build cmake-build-debug --target cuda_kernels
ctest --test-dir cmake-build-debug --output-on-failure
```

CTest covers: `cargo_tests`, `cargo_build_no_default_features`,
`cargo_fmt_check`, `cargo_clippy_no_default`, `cargo_build_bench_example`,
`cuda_kernel_build`.

GPU kernel benchmarks need **both** features (not just `bench`):

```bash
CUDA_NVCC=/usr/local/cuda/bin/nvcc \
  cargo run --example benchmark --profile bench --features bench,cuda
```

### Local CUDA GPU quality gate (preferred over cloud)

Full GPU proof is **local / self-hosted** (Blackwell + driver). Cloud runners
rarely have `sm_120`. See `REVIEW.md` §6 for the full gate. Quick local set:

```bash
export CUDA_NVCC=/usr/local/cuda/bin/nvcc
cargo build --locked --features cuda
cargo test --locked --features cuda
cargo test --locked --features cuda -- --ignored --nocapture
cargo run --locked --example benchmark --profile bench --features bench,cuda
```

Do not create a new build directory — reuse the existing
`cmake-build-debug` directory tracked by the active CLion CMake profile.

## Conventions

- `CMakeLists.txt` is tracked in git (do not re-add it to `.gitignore`);
  `CMakeCache.txt`, `CMakeFiles/`, `cmake-build-*/`, `Makefile`, and
  `cmake_install.cmake` are generated and stay ignored.
- Keep `nvcc` flags in `CMakeLists.txt` and `build.rs` in sync
  (`-std=c++17`, `-D__STRICT_ANSI__`, `--allow-unsupported-compiler`,
  `--expt-relaxed-constexpr`, `-Xcompiler -fno-builtin`) — they exist to
  work around GCC/Clang host-compiler incompatibilities with `nvcc`, not out
  of preference.
- `nvtx` (`range_push!`/`range_pop!`) is a mandatory dependency (used only
  under the `cuda` feature) for profiling with Nsight Systems/Compute. It
  has no Cargo features to enable — its macros are always available from
  the crate root.
- Never weaken or `#[ignore]` a failing test to make CI green; GPU-only
  tests are already marked `#[ignore] // requires GPU + driver ≥ 570` and
  are skipped by default — that pattern should be followed for any new
  GPU-only test.

## See also

- `REVIEW.md` — detailed record of the CUDA/CMake tooling and `nvtx` fixes.
- `AGENTS.md` — brief agent build/test primer.
