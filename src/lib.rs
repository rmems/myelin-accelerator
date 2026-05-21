// myelin-accelerator: safe Rust FFI wrappers around CUDA spiking-network kernels.
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(not(feature = "cuda"))]
pub mod gpu_stub;

// Re-export the main public API at the crate root for ergonomic use.
#[cfg(feature = "cuda")]
pub use gpu::{GpuAccelerator, GpuBuffer, GpuContext, GpuError, KernelModule};
#[cfg(not(feature = "cuda"))]
pub use gpu_stub::{GpuAccelerator, GpuBuffer, GpuContext, GpuError, KernelModule};
