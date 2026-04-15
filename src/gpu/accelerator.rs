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
use crate::gpu::memory::GpuBuffer;
use cust::launch;
use cust::stream::{Stream, StreamFlags};
use tracing::warn;

const SATSOLVER_BLOCK_SIZE: u32 = 256;
const SATSOLVER_SHARED_MEM_BYTES: u32 = 0;

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
            Ok(ctx) => match KernelModule::load() {
                Ok(modules) => Self {
                    _ctx: Some(ctx),
                    modules: Some(modules),
                },
                Err(e) => {
                    warn!("[GPU] PTX load failed (CPU fallback): {e}");
                    Self {
                        _ctx: Some(ctx),
                        modules: None,
                    }
                }
            },
            Err(e) => {
                warn!("[GPU] No CUDA device (CPU fallback): {e}");
                Self {
                    _ctx: None,
                    modules: None,
                }
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

    /// Copy the best SAT walker assignment into `output`.
    ///
    /// This wrapper matches the updated CUDA signature and blocks until the
    /// extract kernel has completed.
    pub fn satsolver_extract(
        &self,
        assignment: &GpuBuffer<u8>,
        best_walker: &GpuBuffer<i32>,
        output: &mut GpuBuffer<u8>,
        n_vars: i32,
        n_walkers: i32,
    ) -> GpuResult<()> {
        if n_vars < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_extract: n_vars must be >= 0, got {n_vars}"
            )));
        }
        if n_walkers <= 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_extract: n_walkers must be > 0, got {n_walkers}"
            )));
        }
        if n_vars == 0 {
            return Ok(());
        }

        let n_vars = n_vars as usize;
        let n_walkers_usize = n_walkers as usize;
        Self::expect_len(
            "assignment",
            assignment.len(),
            n_walkers_usize.saturating_mul(n_vars),
        )?;
        Self::expect_len("best_walker", best_walker.len(), 1)?;
        Self::expect_len("output", output.len(), n_vars)?;

        let kernels = self.kernels()?;
        let satsolver_extract = kernels.get_function("satsolver_extract")?;
        let stream = Self::new_stream()?;
        let grid = Self::ceil_div_u32(n_vars as u32, SATSOLVER_BLOCK_SIZE);
        let block = SATSOLVER_BLOCK_SIZE;

        unsafe {
            launch!(satsolver_extract<<<grid, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                assignment.as_device_ptr(),
                best_walker.as_device_ptr(),
                output.as_device_ptr(),
                n_vars as i32,
                n_walkers,
            ))
            .map_err(|e| GpuError::LaunchFailed(format!("satsolver_extract launch: {e:?}")))?;
        }

        stream
            .synchronize()
            .map_err(|e| GpuError::LaunchFailed(format!("satsolver_extract sync: {e:?}")))?;
        Ok(())
    }

    /// Recompute SAT clause flags and reduce scores to one `(best_score, best_walker)`
    /// pair using the two-pass atomics-free CUDA path.
    ///
    /// `best_score` and `best_walker` must each be length 1. Intermediate per-block
    /// partial buffers are allocated internally using `grid.x = ceil_div(n_walkers, 256)`.
    pub fn satsolver_aux_reduce_best(
        &self,
        assignment: &GpuBuffer<u8>,
        sat_flags: &mut GpuBuffer<u8>,
        scores: &GpuBuffer<i32>,
        best_score: &mut GpuBuffer<i32>,
        best_walker: &mut GpuBuffer<i32>,
        clauses: &GpuBuffer<i32>,
        n_walkers: i32,
        n_vars: i32,
        n_clauses: i32,
        clause_len: i32,
    ) -> GpuResult<()> {
        if n_walkers <= 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: n_walkers must be > 0, got {n_walkers}"
            )));
        }
        if n_vars < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: n_vars must be >= 0, got {n_vars}"
            )));
        }
        if n_clauses < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: n_clauses must be >= 0, got {n_clauses}"
            )));
        }
        if clause_len < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: clause_len must be >= 0, got {clause_len}"
            )));
        }

        let n_walkers_usize = n_walkers as usize;
        let n_vars_usize = n_vars as usize;
        let n_clauses_usize = n_clauses as usize;
        let clause_len_usize = clause_len as usize;

        Self::expect_len(
            "assignment",
            assignment.len(),
            n_walkers_usize.saturating_mul(n_vars_usize),
        )?;
        Self::expect_len(
            "sat_flags",
            sat_flags.len(),
            n_walkers_usize.saturating_mul(n_clauses_usize),
        )?;
        Self::expect_len("scores", scores.len(), n_walkers_usize)?;
        Self::expect_len("best_score", best_score.len(), 1)?;
        Self::expect_len("best_walker", best_walker.len(), 1)?;
        Self::expect_len(
            "clauses",
            clauses.len(),
            n_clauses_usize.saturating_mul(clause_len_usize),
        )?;

        let kernels = self.kernels()?;
        let satsolver_aux_update = kernels.get_function("satsolver_aux_update")?;
        let satsolver_best_reduce_pass2 = kernels.get_function("satsolver_best_reduce_pass2")?;
        let stream = Self::new_stream()?;
        let grid_x = Self::ceil_div_u32(n_walkers as u32, SATSOLVER_BLOCK_SIZE);
        let block = SATSOLVER_BLOCK_SIZE;
        let partial_len = grid_x as usize;
        let partial_scores = GpuBuffer::<i32>::alloc(partial_len)?;
        let partial_walkers = GpuBuffer::<i32>::alloc(partial_len)?;

        unsafe {
            launch!(satsolver_aux_update<<<grid_x, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                assignment.as_device_ptr(),
                sat_flags.as_device_ptr(),
                scores.as_device_ptr(),
                partial_scores.as_device_ptr(),
                partial_walkers.as_device_ptr(),
                clauses.as_device_ptr(),
                n_walkers,
                n_vars,
                n_clauses,
                clause_len,
            ))
            .map_err(|e| GpuError::LaunchFailed(format!("satsolver_aux_update launch: {e:?}")))?;

            launch!(satsolver_best_reduce_pass2<<<1u32, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                partial_scores.as_device_ptr(),
                partial_walkers.as_device_ptr(),
                best_score.as_device_ptr(),
                best_walker.as_device_ptr(),
                partial_len as i32,
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(format!("satsolver_best_reduce_pass2 launch: {e:?}"))
            })?;
        }

        stream.synchronize().map_err(|e| {
            GpuError::LaunchFailed(format!("satsolver_aux_reduce_best sync: {e:?}"))
        })?;
        Ok(())
    }

    fn new_stream() -> GpuResult<Stream> {
        Stream::new(StreamFlags::DEFAULT, None)
            .map_err(|e| GpuError::LaunchFailed(format!("stream creation failed: {e:?}")))
    }

    fn expect_len(name: &str, actual: usize, minimum: usize) -> GpuResult<()> {
        if actual < minimum {
            return Err(GpuError::MemoryError(format!(
                "{name} too small: need at least {minimum} elements, got {actual}"
            )));
        }
        Ok(())
    }

    fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
        value.div_ceil(divisor)
    }
}

impl Default for GpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}
