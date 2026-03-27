// ════════════════════════════════════════════════════════════════════
//  gpu/accelerator.rs — High-level GPU accelerator facade
//
//  GpuAccelerator bundles the CUDA context, loaded PTX modules, and
//  convenience launchers for the neuro-spike kernel set.
//
//  When no GPU is present the struct can still be constructed in
//  "stub" mode — all kernel calls return GpuError::NoGpu so the
//  caller can fall back to the CPU engine gracefully.
// ════════════════════════════════════════════════════════════════════

use crate::gpu::context::GpuContext;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernel::KernelModule;
use tracing::warn;

/// Facade that owns a CUDA context and the compiled PTX modules.
pub struct GpuAccelerator {
    /// `None` when running in CPU-only / stub mode.
    _ctx: Option<GpuContext>,
    /// `None` when PTX modules failed to load.
    modules: Option<KernelModule>,
}

impl GpuAccelerator {
    /// Attempt to initialise GPU.  Returns a stub if no device is available.
    pub fn new() -> Self {
        match GpuContext::init() {
            Ok(ctx) => {
                match KernelModule::load() {
                    Ok(modules) => Self { _ctx: Some(ctx), modules: Some(modules) },
                    Err(e) => {
                        warn!("[GPU] PTX load failed (CPU fallback): {e}");
                        Self { _ctx: Some(ctx), modules: None }
                    }
                }
            }
            Err(e) => {
                warn!("[GPU] No CUDA device (CPU fallback): {e}");
                Self { _ctx: None, modules: None }
            }
        }
    }

    /// `true` if a GPU context and all PTX modules are ready.
    pub fn is_ready(&self) -> bool {
        self._ctx.is_some() && self.modules.is_some()
    }

    /// Borrow the loaded kernel module, or return an error.
    pub fn kernels(&self) -> GpuResult<&KernelModule> {
        self.modules.as_ref().ok_or(GpuError::NoGpu)
    }
}

impl Default for GpuAccelerator {
    fn default() -> Self { Self::new() }
}
