# AGENTS.md

## Project
This is a Rust + CUDA kernel crate for Blackwell / RTX 5080-class GPU work.

## Build and test
Run CPU-safe checks first:

```bash
cargo test --locked
cargo build --locked --no-default-features