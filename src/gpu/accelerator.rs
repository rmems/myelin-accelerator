use crate::gpu::context::GpuContext;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernel::KernelModule;
use crate::gpu::memory::GpuBuffer;
use cust::launch;
use cust::stream::{Stream, StreamFlags};
use std::cell::RefCell;
use tracing::warn;

const SATSOLVER_BLOCK_SIZE: u32 = 256;
const SATSOLVER_SHARED_MEM_BYTES: u32 = 0;

pub struct GpuAccelerator {
    _ctx: Option<GpuContext>,
    modules: Option<KernelModule>,
    stream: Option<Stream>,
    aux_partial_scores: RefCell<Option<GpuBuffer<i32>>>,
    aux_partial_walkers: RefCell<Option<GpuBuffer<i32>>>,
}

impl GpuAccelerator {
    pub fn new() -> Self {
        match GpuContext::init() {
            Ok(ctx) => match (
                KernelModule::load(),
                Stream::new(StreamFlags::DEFAULT, None),
            ) {
                (Ok(modules), Ok(stream)) => Self {
                    _ctx: Some(ctx),
                    modules: Some(modules),
                    stream: Some(stream),
                    aux_partial_scores: RefCell::new(None),
                    aux_partial_walkers: RefCell::new(None),
                },
                (Err(e), _) => {
                    warn!("[GPU] PTX load failed (CPU fallback): {e}");
                    Self {
                        _ctx: Some(ctx),
                        modules: None,
                        stream: None,
                        aux_partial_scores: RefCell::new(None),
                        aux_partial_walkers: RefCell::new(None),
                    }
                }
                (_, Err(e)) => {
                    warn!("[GPU] stream creation failed (CPU fallback): {e:?}");
                    Self {
                        _ctx: Some(ctx),
                        modules: None,
                        stream: None,
                        aux_partial_scores: RefCell::new(None),
                        aux_partial_walkers: RefCell::new(None),
                    }
                }
            },
            Err(e) => {
                warn!("[GPU] No CUDA device (CPU fallback): {e}");
                Self {
                    _ctx: None,
                    modules: None,
                    stream: None,
                    aux_partial_scores: RefCell::new(None),
                    aux_partial_walkers: RefCell::new(None),
                }
            }
        }
    }

    pub fn is_ready(&self) -> bool {
        self._ctx.is_some() && self.modules.is_some() && self.stream.is_some()
    }

    pub fn kernels(&self) -> GpuResult<&KernelModule> {
        self.modules.as_ref().ok_or(GpuError::NoGpu)
    }

    pub fn synchronize(&self) -> GpuResult<()> {
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        stream
            .synchronize()
            .map_err(|e| GpuError::LaunchFailed(format!("stream sync: {e:?}")))
    }

    pub fn satsolver_extract(
        &self,
        assignment: &GpuBuffer<u8>,
        best_walker: &GpuBuffer<i32>,
        output: &mut GpuBuffer<u8>,
        n_vars: i32,
        n_walkers: i32,
    ) -> GpuResult<()> {
        self.satsolver_extract_async(assignment, best_walker, output, n_vars, n_walkers)?;
        self.synchronize()
    }

    pub fn satsolver_extract_async(
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
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
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

        Ok(())
    }

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
        self.satsolver_aux_reduce_best_async(
            assignment,
            sat_flags,
            scores,
            best_score,
            best_walker,
            clauses,
            n_walkers,
            n_vars,
            n_clauses,
            clause_len,
        )?;
        self.synchronize()
    }

    pub fn satsolver_aux_reduce_best_async(
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
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        let grid_x = Self::ceil_div_u32(n_walkers as u32, SATSOLVER_BLOCK_SIZE);
        let block = SATSOLVER_BLOCK_SIZE;
        let partial_len = grid_x as usize;
        let mut partial_scores = self.aux_partial_scores.borrow_mut();
        let mut partial_walkers = self.aux_partial_walkers.borrow_mut();
        let need_partial_realloc = partial_scores
            .as_ref()
            .is_none_or(|b| b.len() < partial_len)
            || partial_walkers
                .as_ref()
                .is_none_or(|b| b.len() < partial_len);
        if need_partial_realloc {
            stream.synchronize().map_err(|e| {
                GpuError::LaunchFailed(format!("stream sync before partial realloc: {e:?}"))
            })?;
            let scores = GpuBuffer::<i32>::alloc(partial_len)?;
            let walkers = GpuBuffer::<i32>::alloc(partial_len)?;
            *partial_scores = Some(scores);
            *partial_walkers = Some(walkers);
        }
        let partial_scores = partial_scores.as_ref().expect("partial_scores buffer");
        let partial_walkers = partial_walkers.as_ref().expect("partial_walkers buffer");

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

        Ok(())
    }

    pub fn poisson_encode(
        &self,
        stimuli: &GpuBuffer<f32>,
        spikes: &mut GpuBuffer<u32>,
        seed: u32,
    ) -> GpuResult<()> {
        self.poisson_encode_async(stimuli, spikes, seed)?;
        self.synchronize()
    }

    pub fn poisson_encode_async(
        &self,
        stimuli: &GpuBuffer<f32>,
        spikes: &mut GpuBuffer<u32>,
        seed: u32,
    ) -> GpuResult<()> {
        let n = stimuli.len();
        Self::expect_len("spikes", spikes.len(), n)?;

        let kernels = self.kernels()?;
        let func = kernels.get_function("poisson_encode")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;

        let block = 256;
        let grid = Self::ceil_div_u32(n as u32, block);

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                stimuli.as_device_ptr(),
                spikes.as_device_ptr(),
                n as i32,
                seed,
            ))
            .map_err(|e| GpuError::LaunchFailed(format!("poisson_encode launch: {e:?}")))?;
        }

        Ok(())
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

impl Drop for GpuAccelerator {
    fn drop(&mut self) {
        let _ = self.synchronize();
    }
}

impl Default for GpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}
