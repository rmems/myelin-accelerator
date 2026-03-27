// myelin-accelerator: safe Rust FFI wrappers around CUDA spiking-network kernels.
pub mod gpu;

// Re-export the main public API at the crate root for ergonomic use.
pub use gpu::{GpuAccelerator, GpuContext, GpuBuffer, KernelModule, GpuError};
