use std::fmt;

pub type GpuResult<T> = Result<T, GpuError>;

#[derive(Debug)]
pub enum GpuError {
    NoGpu,
    InitFailed(String),
    ModuleLoadFailed(String),
    KernelNotFound(String),
    MemoryError(String),
    LaunchFailed(String),
    CudaError(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoGpu => write!(f, "No GPU available (built without `cuda` feature)"),
            GpuError::InitFailed(s) => write!(f, "GPU init failed: {s}"),
            GpuError::ModuleLoadFailed(s) => write!(f, "PTX module load failed: {s}"),
            GpuError::KernelNotFound(s) => write!(f, "Kernel not found: {s}"),
            GpuError::MemoryError(s) => write!(f, "GPU memory error: {s}"),
            GpuError::LaunchFailed(s) => write!(f, "Kernel launch failed: {s}"),
            GpuError::CudaError(s) => write!(f, "CUDA error: {s}"),
        }
    }
}

impl std::error::Error for GpuError {}

pub struct GpuContext;
impl GpuContext {
    pub fn init() -> GpuResult<Self> {
        Err(GpuError::NoGpu)
    }
    pub fn is_available() -> bool {
        false
    }
}

pub struct KernelModule;
impl KernelModule {
    pub fn load() -> GpuResult<Self> {
        Err(GpuError::NoGpu)
    }
    pub fn load_satsolver() -> GpuResult<Self> {
        Err(GpuError::NoGpu)
    }
}

pub struct GpuBuffer<T> {
    data: Vec<T>,
}
impl<T: Default + Clone> GpuBuffer<T> {
    pub fn alloc(len: usize) -> GpuResult<Self> {
        Ok(Self {
            data: vec![T::default(); len],
        })
    }
    pub fn from_slice(data: &[T]) -> GpuResult<Self> {
        Ok(Self {
            data: data.to_vec(),
        })
    }
    pub fn to_vec(&self) -> GpuResult<Vec<T>> {
        Ok(self.data.clone())
    }
    pub fn upload(&mut self, data: &[T]) -> GpuResult<()> {
        if data.len() != self.data.len() {
            return Err(GpuError::MemoryError(format!(
                "upload: length mismatch, buffer has {} elements but input has {}",
                self.data.len(),
                data.len()
            )));
        }
        self.data.clone_from_slice(data);
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

pub struct GpuAccelerator;
impl GpuAccelerator {
    pub fn new() -> Self {
        Self
    }
    pub fn is_ready(&self) -> bool {
        false
    }
    pub fn kernels(&self) -> GpuResult<&KernelModule> {
        Err(GpuError::NoGpu)
    }
    pub fn satsolver_extract(
        &self,
        _: &GpuBuffer<u8>,
        _: &GpuBuffer<i32>,
        _: &mut GpuBuffer<u8>,
        _: i32,
        _: i32,
    ) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }
    pub fn satsolver_aux_reduce_best(
        &self,
        _: &GpuBuffer<u8>,
        _: &mut GpuBuffer<u8>,
        _: &GpuBuffer<i32>,
        _: &mut GpuBuffer<i32>,
        _: &mut GpuBuffer<i32>,
        _: &GpuBuffer<i32>,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
    ) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }
    pub fn poisson_encode(
        &self,
        _: &GpuBuffer<f32>,
        _: &mut GpuBuffer<u32>,
        _: u32,
    ) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }

    pub fn satsolver_extract_async(
        &self,
        _: &GpuBuffer<u8>,
        _: &GpuBuffer<i32>,
        _: &mut GpuBuffer<u8>,
        _: i32,
        _: i32,
    ) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }

    pub fn satsolver_aux_reduce_best_async(
        &self,
        _: &GpuBuffer<u8>,
        _: &mut GpuBuffer<u8>,
        _: &GpuBuffer<i32>,
        _: &mut GpuBuffer<i32>,
        _: &mut GpuBuffer<i32>,
        _: &GpuBuffer<i32>,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
    ) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }

    pub fn poisson_encode_async(
        &self,
        _: &GpuBuffer<f32>,
        _: &mut GpuBuffer<u32>,
        _: u32,
    ) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }

    pub fn synchronize(&self) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }
}
impl Default for GpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}
