//! The hipBLASLt GEMMs of a HIP engine, with the plan table (spec 12.1, "The plan table of a HIP engine").
//!
//! The first GEMM of a new shape picks its algorithm. The engine takes the heuristic candidates of hipBLASLt in order.
//! It keeps the first one that gives equal bits in two runs on the same inputs, and the plan table keeps the pick.

use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_void};
use std::fmt;

use crate::hip::{DeviceBuffer, HipError, Stream};
use crate::plan::{DataType, GemmShape, PlanError, PlanTable, refused};

#[repr(C)]
struct ShimShape {
    m: i32,
    n: i32,
    k: i32,
    trans_a: i32,
    trans_b: i32,
    lda: i64,
    ldb: i64,
    ldc: i64,
    batch: i32,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
    type_ab: i32,
    type_cd: i32,
    epilogue: i32,
}

unsafe extern "C" {
    fn nmk_blaslt_create(out: *mut *mut c_void) -> c_int;
    fn nmk_blaslt_destroy(handle: *mut c_void) -> c_int;
    fn nmk_blaslt_version(out: *mut c_int) -> c_int;
    fn nmk_gemm_candidates(
        handle: *mut c_void,
        shape: *const ShimShape,
        max_workspace: u64,
        max: i32,
        indices: *mut i32,
        workspaces: *mut u64,
        count: *mut i32,
    ) -> c_int;
    fn nmk_gemm_kernel_name(handle: *mut c_void, index: i32, buf: *mut c_char, len: u64) -> c_int;
    fn nmk_gemm_prepare(
        handle: *mut c_void,
        shape: *const ShimShape,
        index: i32,
        max_workspace: u64,
        out: *mut *mut c_void,
        workspace: *mut u64,
    ) -> c_int;
    fn nmk_gemm_run(
        handle: *mut c_void,
        gemm: *mut c_void,
        alpha: f32,
        a: *const c_void,
        b: *const c_void,
        beta: f32,
        c: *const c_void,
        d: *mut c_void,
        bias: *const c_void,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> c_int;
    fn nmk_gemm_free(gemm: *mut c_void);
}

fn shim(shape: &GemmShape) -> ShimShape {
    let int = |value: u32| i32::try_from(value).expect("a GEMM size fits i32");
    let long = |value: u64| i64::try_from(value).expect("a GEMM stride fits i64");
    ShimShape {
        m: int(shape.m),
        n: int(shape.n),
        k: int(shape.k),
        trans_a: i32::from(shape.trans_a),
        trans_b: i32::from(shape.trans_b),
        lda: long(shape.lda),
        ldb: long(shape.ldb),
        ldc: long(shape.ldc),
        batch: int(shape.batch),
        stride_a: long(shape.stride_a),
        stride_b: long(shape.stride_b),
        stride_c: long(shape.stride_c),
        type_ab: shape.type_ab.code(),
        type_cd: shape.type_cd.code(),
        epilogue: i32::from(shape.bias),
    }
}

/// An error of a GEMM. Each one stops the job or the open (spec 8.5).
#[derive(Debug)]
pub enum GemmError {
    Hip(HipError),
    /// A hipBLASLt call of the shim failed with this status. 1000 is a bad argument of the shim.
    Blaslt {
        call: &'static str,
        status: i32,
    },
    Plan(PlanError),
    /// No heuristic candidate gave equal bits in two runs, or each one was refused.
    NoCandidate {
        shape: GemmShape,
        tried: Vec<String>,
    },
}

impl fmt::Display for GemmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GemmError::Hip(error) => write!(f, "{error}"),
            GemmError::Blaslt { call, status } => write!(f, "{call} failed with hipBLASLt status {status}"),
            GemmError::Plan(error) => write!(f, "{error}"),
            GemmError::NoCandidate { shape, tried } => write!(
                f,
                "no hipBLASLt candidate for {shape:?} gave the same bits twice; tried: {}",
                tried.join("; ")
            ),
        }
    }
}

impl std::error::Error for GemmError {}

impl From<HipError> for GemmError {
    fn from(error: HipError) -> Self {
        GemmError::Hip(error)
    }
}

impl From<PlanError> for GemmError {
    fn from(error: PlanError) -> Self {
        GemmError::Plan(error)
    }
}

fn blaslt(call: &'static str, status: c_int) -> Result<(), GemmError> {
    if status == 0 {
        Ok(())
    } else {
        Err(GemmError::Blaslt { call, status })
    }
}

/// The version of hipBLASLt at run time, as `hipblasLtGetVersion` gives it.
pub fn runtime_version() -> Result<i32, GemmError> {
    let mut version = 0;
    // SAFETY: the pointer is valid for the call.
    blaslt("hipblasLtGetVersion", unsafe { nmk_blaslt_version(&mut version) })?;
    Ok(version)
}

/// The lineage of the plan table on the current device: the hipBLASLt version, the HIP runtime version, the gfx
/// target and the PCI ids of the device (spec 12.1).
pub fn lineage() -> Result<String, GemmError> {
    Ok(format!(
        "hipblaslt {} ({}); hip {}; gfx1151; pci {}",
        runtime_version()?,
        env!("NAGAYA_MK_HIP_BUILD_HIPBLASLT"),
        crate::hip::runtime_version()?,
        crate::hip::pci_ids()?
    ))
}

/// A handle of hipBLASLt. Its calls go to the device of the thread that made it.
pub struct Blaslt(*mut c_void);

// SAFETY: the handle can move between threads; each engine still calls from one thread.
unsafe impl Send for Blaslt {}

impl Blaslt {
    pub fn new() -> Result<Blaslt, GemmError> {
        let mut handle = std::ptr::null_mut();
        // SAFETY: the pointer is valid for the call.
        blaslt("hipblasLtCreate", unsafe { nmk_blaslt_create(&mut handle) })?;
        Ok(Blaslt(handle))
    }

    /// The heuristic candidates of `shape`, best first: each solution index with its workspace bytes.
    pub fn candidates(&self, shape: &GemmShape, max_workspace: u64, max: usize) -> Result<Vec<(i32, u64)>, GemmError> {
        let mut indices = vec![0i32; max];
        let mut workspaces = vec![0u64; max];
        let mut count = 0;
        let shape = shim(shape);
        // SAFETY: the arrays hold `max` values, and the shim writes at most that many.
        blaslt("hipblasLtMatmulAlgoGetHeuristic", unsafe {
            nmk_gemm_candidates(
                self.0,
                &shape,
                max_workspace,
                max as i32,
                indices.as_mut_ptr(),
                workspaces.as_mut_ptr(),
                &mut count,
            )
        })?;
        let count = count as usize;
        Ok(indices.into_iter().zip(workspaces).take(count).collect())
    }

    /// The kernel name of a solution index.
    pub fn kernel_name(&self, index: i32) -> Result<String, GemmError> {
        let mut buf = vec![0 as c_char; 1024];
        // SAFETY: the buffer holds `len` bytes, and the shim ends the name with a NUL.
        blaslt("getKernelNameFromAlgo", unsafe {
            nmk_gemm_kernel_name(self.0, index, buf.as_mut_ptr(), buf.len() as u64)
        })?;
        // SAFETY: the shim wrote a NUL-terminated string.
        Ok(unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// Prepare `shape` with the algorithm of `index`. It fails when the algorithm does not support the shape.
    pub fn prepare(&self, shape: &GemmShape, index: i32, max_workspace: u64) -> Result<Gemm, GemmError> {
        let mut ptr = std::ptr::null_mut();
        let mut workspace = 0;
        let s = shim(shape);
        // SAFETY: the pointers are valid for the call.
        blaslt("matmulIsAlgoSupported", unsafe {
            nmk_gemm_prepare(self.0, &s, index, max_workspace, &mut ptr, &mut workspace)
        })?;
        Ok(Gemm {
            ptr,
            index,
            workspace,
            shape: *shape,
        })
    }

    /// Run a prepared GEMM on `stream`: `D = alpha op(A) op(B) + beta C`, plus the bias of each row of D when the
    /// shape has a bias. `bias` is null for a shape without one.
    ///
    /// # Safety
    ///
    /// Each pointer must be device memory that holds the elements of the shape (`GemmShape::elements`). `c` and `d`
    /// can be the same. The bias holds `m` bf16 values. The workspace must hold `gemm.workspace()` bytes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run(
        &self,
        gemm: &Gemm,
        alpha: f32,
        a: *const c_void,
        b: *const c_void,
        beta: f32,
        c: *const c_void,
        d: *mut c_void,
        bias: *const c_void,
        workspace: &DeviceBuffer,
        stream: &Stream,
    ) -> Result<(), GemmError> {
        // SAFETY: the caller keeps the rules above.
        blaslt("hipblasLtMatmul", unsafe {
            nmk_gemm_run(
                self.0,
                gemm.ptr,
                alpha,
                a,
                b,
                beta,
                c,
                d,
                bias,
                workspace.ptr(),
                workspace.bytes() as u64,
                stream.raw(),
            )
        })
    }
}

impl Drop for Blaslt {
    fn drop(&mut self) {
        // SAFETY: the handle came from hipblasLtCreate, and nothing uses it after this. `Gemms` drops its prepared
        // GEMMs before its handle.
        let status = unsafe { nmk_blaslt_destroy(self.0) };
        if status != 0 && !std::thread::panicking() {
            panic!("hipblasLtDestroy failed with status {status}");
        }
    }
}

/// A prepared GEMM: one shape with the algorithm of one solution index.
pub struct Gemm {
    ptr: *mut c_void,
    index: i32,
    workspace: u64,
    shape: GemmShape,
}

// SAFETY: the prepared descriptors can move between threads; each engine still calls from one thread.
unsafe impl Send for Gemm {}

impl Gemm {
    pub fn index(&self) -> i32 {
        self.index
    }

    pub fn workspace(&self) -> u64 {
        self.workspace
    }

    pub fn shape(&self) -> &GemmShape {
        &self.shape
    }
}

impl Drop for Gemm {
    fn drop(&mut self) {
        // SAFETY: the pointer came from nmk_gemm_prepare, and nothing frees it again.
        unsafe { nmk_gemm_free(self.ptr) }
    }
}

/// The count of heuristic candidates that a pick reads.
const CANDIDATES: usize = 16;

/// Random bytes for the inputs of a pick: bf16 or f32 values in [-1, 1).
fn random_values(count: usize, dtype: DataType, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let x = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    };
    let mut bytes = Vec::with_capacity(count * dtype.bytes());
    for _ in 0..count {
        match dtype {
            DataType::Bf16 => bytes.extend_from_slice(&crate::bf16::from_f32(next()).to_le_bytes()),
            DataType::F32 => bytes.extend_from_slice(&next().to_le_bytes()),
        }
    }
    bytes
}

/// The GEMMs of one engine: the hipBLASLt handle, the plan table, the prepared shapes and one workspace.
pub struct Gemms {
    table: PlanTable,
    prepared: BTreeMap<GemmShape, Gemm>,
    workspace: DeviceBuffer,
    /// The handle drops last: Rust drops the fields in this order, so each prepared GEMM goes first.
    blaslt: Blaslt,
}

impl Gemms {
    /// The GEMMs of `table`, with a workspace of `max_workspace` bytes for each GEMM.
    pub fn new(table: PlanTable, max_workspace: u64) -> Result<Gemms, GemmError> {
        Ok(Gemms {
            blaslt: Blaslt::new()?,
            table,
            prepared: BTreeMap::new(),
            workspace: DeviceBuffer::new(max_workspace as usize)?,
        })
    }

    pub fn table(&self) -> &PlanTable {
        &self.table
    }

    pub fn blaslt(&self) -> &Blaslt {
        &self.blaslt
    }

    /// Prepare `shape`. A shape with no pick gets one first: the first heuristic candidate that the name rule allows
    /// and that gives equal bits in two runs. The table keeps the pick before this returns. Give the solution index.
    pub fn prepare(&mut self, shape: &GemmShape, stream: &Stream) -> Result<i32, GemmError> {
        if let Some(gemm) = self.prepared.get(shape) {
            return Ok(gemm.index);
        }
        let max = self.workspace.bytes() as u64;
        let gemm = match self.table.pick(shape) {
            Some(index) => self.blaslt.prepare(shape, index, max)?,
            None => {
                let gemm = self.choose(shape, stream)?;
                self.table.record(*shape, gemm.index)?;
                gemm
            }
        };
        let index = gemm.index;
        self.prepared.insert(*shape, gemm);
        Ok(index)
    }

    fn choose(&self, shape: &GemmShape, stream: &Stream) -> Result<Gemm, GemmError> {
        let max = self.workspace.bytes() as u64;
        let (a_count, b_count, c_count) = shape.elements();
        let a = DeviceBuffer::from_slice(&random_values(a_count, shape.type_ab, 1))?;
        let b = DeviceBuffer::from_slice(&random_values(b_count, shape.type_ab, 2))?;
        let c = DeviceBuffer::from_slice(&random_values(c_count, shape.type_cd, 3))?;
        let bias = DeviceBuffer::from_slice(&random_values(shape.m as usize, DataType::Bf16, 4))?;
        let bias = if shape.bias {
            bias.ptr().cast_const()
        } else {
            std::ptr::null()
        };
        let d_bytes = c_count * shape.type_cd.bytes();
        let first = DeviceBuffer::new(d_bytes)?;
        let second = DeviceBuffer::new(d_bytes)?;
        let mut tried = Vec::new();
        for (index, _) in self.blaslt.candidates(shape, max, CANDIDATES)? {
            let name = self.blaslt.kernel_name(index)?;
            if let Some(why) = refused(&name) {
                tried.push(format!("{index} {name}: {why}"));
                continue;
            }
            let gemm = match self.blaslt.prepare(shape, index, max) {
                Ok(gemm) => gemm,
                Err(error) => {
                    tried.push(format!("{index} {name}: {error}"));
                    continue;
                }
            };
            // The two outputs start from different bytes, so an element that a run does not write differs too.
            first.write(&vec![0u8; d_bytes])?;
            second.write(&vec![0xffu8; d_bytes])?;
            for out in [&first, &second] {
                // SAFETY: each buffer holds the elements of the shape, and the workspace holds `max` bytes.
                unsafe {
                    self.blaslt.run(
                        &gemm,
                        1.0,
                        a.ptr(),
                        b.ptr(),
                        0.5,
                        c.ptr(),
                        out.ptr(),
                        bias,
                        &self.workspace,
                        stream,
                    )?;
                }
            }
            stream.synchronize()?;
            // Only the elements of D count: a leading dimension past `m` leaves the rest of each column as it was.
            let (one, two) = (first.to_vec::<u8>()?, second.to_vec::<u8>()?);
            let size = shape.type_cd.bytes();
            if shape
                .output_elements()
                .all(|e| one[e * size..(e + 1) * size] == two[e * size..(e + 1) * size])
            {
                return Ok(gemm);
            }
            tried.push(format!("{index} {name}: two runs gave other bits"));
        }
        Err(GemmError::NoCandidate { shape: *shape, tried })
    }

    /// Run `shape` on `stream`: `D = alpha op(A) op(B) + beta C`. It prepares the shape first when it must. The shape
    /// must have no bias.
    ///
    /// # Safety
    ///
    /// Each pointer must be device memory that holds the elements of the shape. `c` and `d` can be the same.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run(
        &mut self,
        shape: &GemmShape,
        alpha: f32,
        a: *const c_void,
        b: *const c_void,
        beta: f32,
        c: *const c_void,
        d: *mut c_void,
        stream: &Stream,
    ) -> Result<(), GemmError> {
        assert!(!shape.bias, "{shape:?}: a shape with a bias runs with run_bias");
        // SAFETY: the caller keeps the rules of the pointers.
        unsafe { self.run_any(shape, alpha, a, b, beta, c, d, std::ptr::null(), stream) }
    }

    /// Run `shape`, a shape with a bias, on `stream`: `D = alpha op(A) op(B) + beta C + bias`.
    ///
    /// # Safety
    ///
    /// As for [`Gemms::run`]. The bias holds `m` bf16 values.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_bias(
        &mut self,
        shape: &GemmShape,
        alpha: f32,
        a: *const c_void,
        b: *const c_void,
        beta: f32,
        c: *const c_void,
        d: *mut c_void,
        bias: *const c_void,
        stream: &Stream,
    ) -> Result<(), GemmError> {
        assert!(
            shape.bias && !bias.is_null(),
            "{shape:?}: run_bias needs a shape with a bias, and the bias"
        );
        // SAFETY: the caller keeps the rules of the pointers.
        unsafe { self.run_any(shape, alpha, a, b, beta, c, d, bias, stream) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn run_any(
        &mut self,
        shape: &GemmShape,
        alpha: f32,
        a: *const c_void,
        b: *const c_void,
        beta: f32,
        c: *const c_void,
        d: *mut c_void,
        bias: *const c_void,
        stream: &Stream,
    ) -> Result<(), GemmError> {
        self.prepare(shape, stream)?;
        let gemm = &self.prepared[shape];
        // SAFETY: the caller keeps the rules of the pointers, and the workspace holds the bytes of the pick.
        unsafe {
            self.blaslt
                .run(gemm, alpha, a, b, beta, c, d, bias, &self.workspace, stream)
        }
    }
}
