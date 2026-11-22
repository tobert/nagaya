//! Minimal CUDA Driver API (`libcuda`) wrapper for loading `.cubin` modules and launching persistent megakernels.
//!
//! Links `libcuda` directly with no `libcudart` dependency. Every call must run on the device's owner thread
//! (spec 3.2).

#![allow(non_camel_case_types)]

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::ptr;

use crate::{KERNEL_ENTRY, match_addresses};

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
pub type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;

unsafe extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDriverGetVersion(driver_version: *mut c_int) -> CUresult;
    fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuDeviceGetPCIBusId(pci_bus_id: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    fn cuDeviceGetAttribute(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    #[link_name = "cuDevicePrimaryCtxRelease_v2"]
    fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleUnload(hmod: CUmodule) -> CUresult;
    fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char) -> CUresult;
    #[link_name = "cuMemAlloc_v2"]
    fn cuMemAlloc(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    #[link_name = "cuMemFree_v2"]
    fn cuMemFree(dptr: CUdeviceptr) -> CUresult;
    #[link_name = "cuMemAllocHost_v2"]
    fn cuMemAllocHost(pp: *mut *mut c_void, bytesize: usize) -> CUresult;
    fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    #[link_name = "cuMemcpyHtoD_v2"]
    fn cuMemcpyHtoD(dst_device: CUdeviceptr, src_host: *const c_void, byte_count: usize) -> CUresult;
    #[link_name = "cuMemcpyDtoH_v2"]
    fn cuMemcpyDtoH(dst_host: *mut c_void, src_device: CUdeviceptr, byte_count: usize) -> CUresult;
    #[link_name = "cuMemsetD8_v2"]
    fn cuMemsetD8(dst_device: CUdeviceptr, uc: u8, n: usize) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        hstream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
}

fn check(code: CUresult, what: &str) -> Result<(), String> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what} failed with CUDA driver error {code}"))
    }
}

/// Normalize a CUDA PCI bus id (`0000000f:01:00.0` or `000f:01:00.0`) to `dddd:bb:dd.f` lowercase.
fn normalize_pci(raw: &str) -> String {
    let lower = raw.trim().to_ascii_lowercase();
    if let Some((domain, rest)) = lower.split_once(':')
        && domain.len() > 4
    {
        return format!("{}:{rest}", &domain[domain.len() - 4..]);
    }
    lower
}

/// One opened CUDA GPU with its primary context and loaded persistent megakernel module.
pub struct CudaDevice {
    device: CUdevice,
    ctx: CUcontext,
    module: CUmodule,
    kernel: CUfunction,
    sm_count: u32,
    driver_version: i32,
    pci_address: String,
}

// One owner thread drives the device at a time (spec 3.2).
unsafe impl Send for CudaDevice {}

impl CudaDevice {
    /// Open the CUDA GPU at `pci_address` and load `cubin` (`mk_persistent_interpreter`).
    pub fn open_by_pci(pci_address: &str, cubin: &[u8]) -> Result<Self, String> {
        unsafe {
            check(cuInit(0), "cuInit")?;
            let mut driver_version: c_int = 0;
            check(cuDriverGetVersion(&mut driver_version), "cuDriverGetVersion")?;
            let mut count: c_int = 0;
            check(cuDeviceGetCount(&mut count), "cuDeviceGetCount")?;
            if count <= 0 {
                return Err("no CUDA GPU is visible to the driver".to_owned());
            }
            let mut devices = Vec::with_capacity(count as usize);
            let mut addrs = Vec::with_capacity(count as usize);
            for ordinal in 0..count {
                let mut dev: CUdevice = 0;
                check(cuDeviceGet(&mut dev, ordinal), "cuDeviceGet")?;
                let mut buf = [0u8; 32];
                check(
                    cuDeviceGetPCIBusId(buf.as_mut_ptr().cast::<c_char>(), buf.len() as c_int, dev),
                    "cuDeviceGetPCIBusId",
                )?;
                let raw = CStr::from_ptr(buf.as_ptr().cast::<c_char>())
                    .to_str()
                    .map_err(|e| format!("invalid PCI bus id UTF-8: {e}"))?;
                devices.push(dev);
                addrs.push(normalize_pci(raw));
            }
            let wanted = [pci_address.to_owned()];
            let idx = match_addresses(&addrs, &wanted)?[0];
            let device = devices[idx];
            let matched_pci = addrs[idx].clone();

            let mut sm_count: c_int = 0;
            check(
                cuDeviceGetAttribute(&mut sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
                "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)",
            )?;
            let mut ctx: CUcontext = ptr::null_mut();
            check(cuDevicePrimaryCtxRetain(&mut ctx, device), "cuDevicePrimaryCtxRetain")?;
            if let Err(err) = check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent") {
                let _ = cuDevicePrimaryCtxRelease(device);
                return Err(err);
            }
            let mut module: CUmodule = ptr::null_mut();
            if let Err(err) = check(cuModuleLoadData(&mut module, cubin.as_ptr().cast()), "cuModuleLoadData") {
                let _ = cuDevicePrimaryCtxRelease(device);
                return Err(err);
            }
            let entry = CString::new(KERNEL_ENTRY).expect("kernel entry has no NUL byte");
            let mut kernel: CUfunction = ptr::null_mut();
            if let Err(err) = check(
                cuModuleGetFunction(&mut kernel, module, entry.as_ptr()),
                "cuModuleGetFunction",
            ) {
                let _ = cuModuleUnload(module);
                let _ = cuDevicePrimaryCtxRelease(device);
                return Err(err);
            }
            Ok(Self {
                device,
                ctx,
                module,
                kernel,
                sm_count: u32::try_from(sm_count).unwrap_or(1).max(1),
                driver_version,
                pci_address: matched_pci,
            })
        }
    }

    /// Number of streaming multiprocessors on this GPU.
    pub fn sm_count(&self) -> u32 {
        self.sm_count
    }

    /// CUDA driver version integer (for example `13020` for CUDA 13.2).
    pub fn driver_version(&self) -> i32 {
        self.driver_version
    }

    /// Lowercase PCI address of this GPU (`dddd:bb:dd.f`).
    pub fn pci_address(&self) -> &str {
        &self.pci_address
    }

    /// Bind this device's primary context to the current thread.
    pub fn set_current(&self) -> Result<(), String> {
        unsafe { check(cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent") }
    }

    /// Allocate `bytes` of device memory.
    pub fn alloc_device(&self, bytes: usize) -> Result<CUdeviceptr, String> {
        let mut ptr: CUdeviceptr = 0;
        unsafe { check(cuMemAlloc(&mut ptr, bytes.max(1)), "cuMemAlloc")? };
        Ok(ptr)
    }

    /// Free device memory allocated by [`Self::alloc_device`].
    pub fn free_device(&self, ptr: CUdeviceptr) -> Result<(), String> {
        if ptr == 0 {
            return Ok(());
        }
        unsafe { check(cuMemFree(ptr), "cuMemFree") }
    }

    /// Allocate `bytes` of page-locked host memory.
    pub fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut c_void, String> {
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe { check(cuMemAllocHost(&mut ptr, bytes.max(1)), "cuMemAllocHost")? };
        Ok(ptr)
    }

    /// Free page-locked host memory allocated by [`Self::alloc_host_pinned`].
    ///
    /// # Safety
    /// `ptr` must be null or a pointer previously returned by [`Self::alloc_host_pinned`] and not yet freed.
    pub unsafe fn free_host_pinned(&self, ptr: *mut c_void) -> Result<(), String> {
        if ptr.is_null() {
            return Ok(());
        }
        unsafe { check(cuMemFreeHost(ptr), "cuMemFreeHost") }
    }

    /// Copy `bytes` from host pointer `src` to device pointer `dst`.
    ///
    /// # Safety
    /// `src` must be valid for `bytes` readable bytes and `dst` must hold at least `bytes` device bytes.
    pub unsafe fn copy_htod(&self, dst: CUdeviceptr, src: *const c_void, bytes: usize) -> Result<(), String> {
        unsafe { check(cuMemcpyHtoD(dst, src, bytes), "cuMemcpyHtoD") }
    }

    /// Copy `bytes` from device pointer `src` to host pointer `dst`.
    ///
    /// # Safety
    /// `dst` must be valid for `bytes` writable bytes and `src` must hold at least `bytes` device bytes.
    pub unsafe fn copy_dtoh(&self, dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> Result<(), String> {
        unsafe { check(cuMemcpyDtoH(dst, src, bytes), "cuMemcpyDtoH") }
    }

    /// Fill `bytes` of device memory at `dst` with zeros.
    pub fn memset_zero(&self, dst: CUdeviceptr, bytes: usize) -> Result<(), String> {
        unsafe { check(cuMemsetD8(dst, 0, bytes), "cuMemsetD8") }
    }

    /// Synchronize the current context.
    pub fn sync(&self) -> Result<(), String> {
        unsafe { check(cuCtxSynchronize(), "cuCtxSynchronize") }
    }

    /// Launch `mk_persistent_interpreter` with `grid_blocks` blocks of `block_threads` threads and synchronize.
    ///
    /// # Safety
    /// All device pointers must be valid for the duration of the kernel and large enough for the ring.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_megakernel(
        &self,
        grid_blocks: u32,
        block_threads: u32,
        tasks_dev: CUdeviceptr,
        num_tasks: u32,
        scoreboard_dev: CUdeviceptr,
        base_event: u32,
        stats_dev: CUdeviceptr,
        scratch_dev: CUdeviceptr,
        profile_dev: CUdeviceptr,
    ) -> Result<(), String> {
        let mut p_tasks = tasks_dev;
        let mut p_num = num_tasks;
        let mut p_sb = scoreboard_dev;
        let mut p_base = base_event;
        let mut p_stats = stats_dev;
        let mut p_scratch = scratch_dev;
        let mut p_profile = profile_dev;
        let mut params: [*mut c_void; 7] = [
            (&mut p_tasks as *mut CUdeviceptr).cast(),
            (&mut p_num as *mut u32).cast(),
            (&mut p_sb as *mut CUdeviceptr).cast(),
            (&mut p_base as *mut u32).cast(),
            (&mut p_stats as *mut CUdeviceptr).cast(),
            (&mut p_scratch as *mut CUdeviceptr).cast(),
            (&mut p_profile as *mut CUdeviceptr).cast(),
        ];
        unsafe {
            check(
                cuLaunchKernel(
                    self.kernel,
                    grid_blocks,
                    1,
                    1,
                    block_threads,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel(mk_persistent_interpreter)",
            )?;
        }
        self.sync()
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = cuCtxSetCurrent(self.ctx);
            let _ = cuModuleUnload(self.module);
            let _ = cuDevicePrimaryCtxRelease(self.device);
        }
    }
}
