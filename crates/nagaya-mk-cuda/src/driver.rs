//! Minimal CUDA Driver API (`libcuda`) wrapper for loading `.cubin` modules and launching persistent megakernels.
//!
//! Links `libcuda` directly with no `libcudart` dependency. One thread drives a device at a time (spec 3.2).
//! Each call makes the primary context current on the calling thread first, so a device can move to another thread.

#![allow(non_camel_case_types)]

use std::cell::Cell;
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
const CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH: c_int = 95;

unsafe extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuGetErrorName(error: CUresult, p_str: *mut *const c_char) -> CUresult;
    fn cuGetErrorString(error: CUresult, p_str: *mut *const c_char) -> CUresult;
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
    fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
        num_blocks: *mut c_int,
        func: CUfunction,
        block_size: c_int,
        dynamic_smem_size: usize,
    ) -> CUresult;
    fn cuLaunchCooperativeKernel(
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
    ) -> CUresult;
}

/// Read one driver text through `cuGetErrorName` or `cuGetErrorString`. These calls need no context.
fn driver_text(
    code: CUresult,
    get: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
    fallback: &str,
) -> String {
    let mut text: *const c_char = ptr::null();
    // SAFETY: the driver writes a pointer to a static NUL-terminated string, or leaves null on an error.
    let status = unsafe { get(code, &mut text) };
    if status != CUDA_SUCCESS || text.is_null() {
        return fallback.to_owned();
    }
    // SAFETY: `text` is not null and the driver owns a NUL-terminated string.
    unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
}

/// What the reader can do about one driver error code.
fn advice(code: CUresult) -> &'static str {
    match code {
        2 => "Free GPU memory, or ask for fewer bytes.",
        720 => "The grid is larger than the GPU can hold at once. Use fewer blocks.",
        3 | 100 | 101 => "Check that the driver is loaded: run `nvidia-smi`.",
        201 | 202 | 219 | 400 => "The context of the device is gone or destroyed. Open the device again.",
        209 | 222 | 300 | 301 => "The cubin does not fit this GPU. Build it for `sm_121a` with the same CUDA toolkit.",
        216 | 217 | 704 | 708 => {
            "The call is not allowed now. Check the order of the calls and the context of the thread."
        }
        700 | 701 | 702 | 715 | 719 | 999 => {
            "A kernel failed, and the context cannot run again. Restart the process and check the task pointers."
        }
        _ => "Check that no other process holds the GPU, and that the driver matches the CUDA toolkit.",
    }
}

/// Refuse a grid that the GPU cannot hold all at once. Every block waits on the others, so a block that is not
/// resident would wait for ever.
fn residency_problem(grid_blocks: u32, block_threads: u32, blocks_per_sm: u32, sm_count: u32) -> Result<(), String> {
    if grid_blocks == 0 || block_threads == 0 {
        return Err(format!(
            "a launch of {grid_blocks} blocks with {block_threads} threads each has no work. \
             Ask for at least one block and one thread"
        ));
    }
    if blocks_per_sm == 0 {
        return Err(format!(
            "no block of {block_threads} threads fits on one multiprocessor, so the kernel cannot run. \
             Use fewer threads in each block"
        ));
    }
    let resident = blocks_per_sm * sm_count;
    if grid_blocks > resident {
        return Err(format!(
            "the grid of {grid_blocks} blocks is larger than the {resident} blocks that the GPU holds at once \
             ({blocks_per_sm} blocks on each of {sm_count} multiprocessors). Each block waits on the others, so the \
             launch would hang. Use a grid of {resident} blocks or fewer"
        ));
    }
    Ok(())
}

/// Build the message of one driver error: what failed, the driver's name and text, and what to do.
fn describe(code: CUresult, what: &str) -> String {
    let name = driver_text(code, cuGetErrorName, "an unknown error");
    let text = driver_text(code, cuGetErrorString, "the driver gives no text");
    format!(
        "{what} failed with CUDA driver error {code} ({name}): {text}. {}",
        advice(code)
    )
}

fn check(code: CUresult, what: &str) -> Result<(), String> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(describe(code, what))
    }
}

/// Convert the multiprocessor count that the driver reports. A count below 1 is an error, not a default.
fn sm_count_from(raw: c_int) -> Result<u32, String> {
    match u32::try_from(raw) {
        Ok(count) if count >= 1 => Ok(count),
        _ => Err(format!(
            "the driver reports {raw} multiprocessors for the GPU, so the grid width is unknown. \
             Check the driver with `nvidia-smi`, and do not run on this GPU"
        )),
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
    live_buffers: Cell<usize>,
}

// One thread drives the device at a time (spec 3.2). Each call binds the context to the calling thread, so a move
// to another thread is safe. The type is not `Sync`: the raw handles stop that.
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
            let sm_count = sm_count_from(sm_count)?;
            let mut cooperative: c_int = 0;
            check(
                cuDeviceGetAttribute(&mut cooperative, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device),
                "cuDeviceGetAttribute(COOPERATIVE_LAUNCH)",
            )?;
            if cooperative == 0 {
                return Err(format!(
                    "the GPU at {matched_pci} cannot run a cooperative launch, so the driver cannot guarantee that \
                     every block is resident. Use a GPU of compute capability 6.0 or newer with a current driver"
                ));
            }
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
                sm_count,
                driver_version,
                pci_address: matched_pci,
                live_buffers: Cell::new(0),
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

    /// Count of buffers that this device allocated and did not free. A test uses it to find a leak.
    pub fn live_buffers(&self) -> usize {
        self.live_buffers.get()
    }

    /// Bind this device's primary context to the current thread. Each call of the device does this itself.
    pub fn set_current(&self) -> Result<(), String> {
        unsafe { check(cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent") }
    }

    /// Allocate `bytes` of device memory.
    pub fn alloc_device(&self, bytes: usize) -> Result<CUdeviceptr, String> {
        self.set_current()?;
        let mut ptr: CUdeviceptr = 0;
        unsafe { check(cuMemAlloc(&mut ptr, bytes.max(1)), "cuMemAlloc")? };
        self.live_buffers.set(self.live_buffers.get() + 1);
        Ok(ptr)
    }

    /// Free device memory allocated by [`Self::alloc_device`].
    pub fn free_device(&self, ptr: CUdeviceptr) -> Result<(), String> {
        if ptr == 0 {
            return Ok(());
        }
        self.set_current()?;
        unsafe { check(cuMemFree(ptr), "cuMemFree")? };
        self.live_buffers.set(self.live_buffers.get() - 1);
        Ok(())
    }

    /// Allocate `bytes` of page-locked host memory.
    pub fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut c_void, String> {
        self.set_current()?;
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe { check(cuMemAllocHost(&mut ptr, bytes.max(1)), "cuMemAllocHost")? };
        self.live_buffers.set(self.live_buffers.get() + 1);
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
        self.set_current()?;
        unsafe { check(cuMemFreeHost(ptr), "cuMemFreeHost")? };
        self.live_buffers.set(self.live_buffers.get() - 1);
        Ok(())
    }

    /// Copy `bytes` from host pointer `src` to device pointer `dst`.
    ///
    /// # Safety
    /// `src` must be valid for `bytes` readable bytes and `dst` must hold at least `bytes` device bytes.
    pub unsafe fn copy_htod(&self, dst: CUdeviceptr, src: *const c_void, bytes: usize) -> Result<(), String> {
        self.set_current()?;
        unsafe { check(cuMemcpyHtoD(dst, src, bytes), "cuMemcpyHtoD") }
    }

    /// Copy `bytes` from device pointer `src` to host pointer `dst`.
    ///
    /// # Safety
    /// `dst` must be valid for `bytes` writable bytes and `src` must hold at least `bytes` device bytes.
    pub unsafe fn copy_dtoh(&self, dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> Result<(), String> {
        self.set_current()?;
        unsafe { check(cuMemcpyDtoH(dst, src, bytes), "cuMemcpyDtoH") }
    }

    /// Fill `bytes` of device memory at `dst` with zeros.
    pub fn memset_zero(&self, dst: CUdeviceptr, bytes: usize) -> Result<(), String> {
        self.set_current()?;
        unsafe { check(cuMemsetD8(dst, 0, bytes), "cuMemsetD8") }
    }

    /// Synchronize the current context.
    pub fn sync(&self) -> Result<(), String> {
        self.set_current()?;
        unsafe { check(cuCtxSynchronize(), "cuCtxSynchronize") }
    }

    /// Blocks of `block_threads` threads that the GPU holds at once, from the occupancy of the loaded kernel.
    pub fn max_resident_blocks(&self, block_threads: u32) -> Result<u32, String> {
        self.set_current()?;
        let threads = c_int::try_from(block_threads)
            .map_err(|_| format!("{block_threads} threads in a block is more than the driver takes"))?;
        let mut per_sm: c_int = 0;
        unsafe {
            check(
                cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut per_sm, self.kernel, threads, 0),
                "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            )?;
        }
        let per_sm = u32::try_from(per_sm).unwrap_or(0);
        Ok(per_sm * self.sm_count)
    }

    /// Launch `mk_persistent_interpreter` with `grid_blocks` blocks of `block_threads` threads and synchronize.
    ///
    /// The launch is cooperative, so the driver guarantees that all blocks are resident. The call refuses a grid that
    /// the GPU cannot hold, because a block that is not resident never gets its turn and the others wait for ever.
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
        self.set_current()?;
        let resident = self.max_resident_blocks(block_threads)?;
        residency_problem(grid_blocks, block_threads, resident / self.sm_count, self.sm_count)?;
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
                cuLaunchCooperativeKernel(
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
                ),
                "cuLaunchCooperativeKernel(mk_persistent_interpreter)",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_message_names_the_code_the_cause_and_the_next_step() {
        let message = check(700, "cuLaunchKernel").unwrap_err();
        assert!(message.starts_with("cuLaunchKernel failed"), "{message}");
        assert!(message.contains("CUDA_ERROR_ILLEGAL_ADDRESS"), "{message}");
        assert!(message.contains("illegal memory access"), "{message}");
        assert!(message.contains("Restart the process"), "{message}");
        let unknown = check(123_456, "cuInit").unwrap_err();
        assert!(unknown.contains("123456"), "{unknown}");
        assert!(unknown.contains("Check that no other process"), "{unknown}");
    }

    #[test]
    fn a_grid_that_is_not_resident_is_refused_with_the_cause() {
        assert!(residency_problem(48, 256, 6, 48).is_ok());
        assert!(residency_problem(288, 256, 6, 48).is_ok());
        let big = residency_problem(289, 256, 6, 48).unwrap_err();
        assert!(big.contains("289") && big.contains("288"), "{big}");
        assert!(big.contains("hang") && big.contains("Use a grid of 288"), "{big}");
        assert!(
            residency_problem(1, 256, 0, 48)
                .unwrap_err()
                .contains("Use fewer threads")
        );
        assert!(residency_problem(0, 256, 6, 48).is_err());
        assert!(residency_problem(1, 0, 6, 48).is_err());
    }

    #[test]
    fn a_bad_multiprocessor_count_is_an_error() {
        assert_eq!(sm_count_from(48).unwrap(), 48);
        for bad in [0, -1] {
            let message = sm_count_from(bad).unwrap_err();
            assert!(message.contains(&bad.to_string()), "{message}");
            assert!(message.contains("nvidia-smi"), "{message}");
        }
    }
}
