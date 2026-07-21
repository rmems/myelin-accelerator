# AGENTS.md

## Project
This is a Rust + CUDA kernel crate for Blackwell / RTX 5080-class GPU work.

## Build and test
Run CPU-safe checks first:

```bash
cargo test --locked
cargo build --locked --no-default-features
```

Local GPU path (after CPU checks): toolkit **13.2+** recommended, **13.3**
preferred when present. Point `CUDA_NVCC` at the active `nvcc` if the
`/usr/local/cuda` symlink is still on an older tree. Full local quality gate
is documented in `REVIEW.md` §6 and `CLAUDE.md` (not a cloud-only check).
