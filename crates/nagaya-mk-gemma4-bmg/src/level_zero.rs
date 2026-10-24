//! Thin, zero-dependency Rust FFI wrapper over Intel Level Zero (`libze_loader.so`).
//! Supports isolated per-card contexts (`B70Card`) and Linux `DMA-BUF` file descriptor
//! export/import for zero-copy GPU GDDR6 VRAM handoff across `systemd` process restarts.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr;

pub type ze_result_t = u32;
pub type ze_bool_t = u8;
pub type ze_driver_handle_t = *mut c_void;
pub type ze_device_handle_t = *mut c_void;
pub type ze_context_handle_t = *mut c_void;
pub type ze_command_queue_handle_t = *mut c_void;
pub type ze_command_list_handle_t = *mut c_void;
pub type ze_module_handle_t = *mut c_void;
pub type ze_kernel_handle_t = *mut c_void;

pub const ZE_RESULT_SUCCESS: ze_result_t = 0;

pub const ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 3;
pub const ZE_STRUCTURE_TYPE_CONTEXT_DESC: u32 = 13;
pub const ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC: u32 = 14;
pub const ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC: u32 = 15;
pub const ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC: u32 = 21;
pub const ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC: u32 = 22;
pub const ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES: u32 = 23;
pub const ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC: u32 = 24;
pub const ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD: u32 = 25;
pub const ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD: u32 = 26;
pub const ZE_STRUCTURE_TYPE_MODULE_DESC: u32 = 27;
pub const ZE_STRUCTURE_TYPE_KERNEL_DESC: u32 = 29;

pub const ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES: u32 = 1;
pub const ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES: u32 = 0x10008;

pub const ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF: u32 = 2;

pub const ZE_MODULE_FORMAT_IL_SPIRV: u32 = 0;
pub const ZE_MODULE_FORMAT_NATIVE: u32 = 1;

pub const ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS: u32 = 1;
pub const ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS: u32 = 2;

#[repr(C)]
pub struct ze_device_properties_t {
    pub stype: u32,
    pub pNext: *mut c_void,
    pub r#type: u32,
    pub vendorId: u32,
    pub deviceId: u32,
    pub flags: u32,
    pub subdeviceId: u32,
    pub coreClockRate: u32,
    pub maxMemAllocSize: u64,
    pub maxHardwareContexts: u32,
    pub maxCommandQueuePriority: u32,
    pub numThreadsPerEU: u32,
    pub physicalEUSimdWidth: u32,
    pub numEUsPerSubslice: u32,
    pub numSubslicesPerSlice: u32,
    pub numSlices: u32,
    pub timerResolution: u64,
    pub timestampValidBits: u32,
    pub kernelTimestampValidBits: u32,
    pub uuid: [u8; 16],
    pub name: [c_char; 256],
}

#[repr(C)]
pub struct ze_driver_properties_t {
    pub stype: u32,
    pub pNext: *mut c_void,
    pub uuid: [u8; 16],
    pub driverVersion: u32,
}

#[repr(C)]
pub struct ze_pci_address_ext_t {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub function: u32,
}

#[repr(C)]
pub struct ze_pci_speed_ext_t {
    pub genVersion: i32,
    pub width: i32,
    pub maxBandwidth: i64,
}

#[repr(C)]
pub struct ze_pci_ext_properties_t {
    pub stype: u32,
    pub pNext: *mut c_void,
    pub address: ze_pci_address_ext_t,
    pub maxSpeed: ze_pci_speed_ext_t,
}

#[repr(C)]
pub struct ze_context_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct ze_command_queue_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub ordinal: u32,
    pub index: u32,
    pub flags: u32,
    pub mode: u32,
    pub priority: u32,
}

#[repr(C)]
pub struct ze_command_list_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub commandQueueGroupOrdinal: u32,
    pub flags: u32,
}

#[repr(C)]
pub struct ze_host_mem_alloc_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct ze_device_mem_alloc_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub ordinal: u32,
}

#[repr(C)]
pub struct ze_external_memory_export_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct ze_external_memory_export_fd_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub fd: c_int,
}

#[repr(C)]
pub struct ze_external_memory_import_fd_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub fd: c_int,
}

#[repr(C)]
pub struct ze_memory_allocation_properties_t {
    pub stype: u32,
    pub pNext: *mut c_void,
    pub r#type: u32,
    pub id: u64,
    pub pageSize: u64,
}

#[repr(C)]
pub struct ze_module_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub format: u32,
    pub inputSize: usize,
    pub pInputModule: *const u8,
    pub pBuildFlags: *const c_char,
    pub pConstants: *const c_void,
}

#[repr(C)]
pub struct ze_kernel_desc_t {
    pub stype: u32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub pKernelName: *const c_char,
}

#[repr(C)]
pub struct ze_group_count_t {
    pub groupCountX: u32,
    pub groupCountY: u32,
    pub groupCountZ: u32,
}

unsafe extern "C" {
    pub fn zeInit(flags: u32) -> ze_result_t;
    pub fn zeDriverGet(pCount: *mut u32, phDrivers: *mut ze_driver_handle_t) -> ze_result_t;
    pub fn zeDeviceGet(
        hDriver: ze_driver_handle_t,
        pCount: *mut u32,
        phDevices: *mut ze_device_handle_t,
    ) -> ze_result_t;
    pub fn zeDeviceGetProperties(
        hDevice: ze_device_handle_t,
        pDeviceProperties: *mut ze_device_properties_t,
    ) -> ze_result_t;
    pub fn zeDriverGetProperties(
        hDriver: ze_driver_handle_t,
        pDriverProperties: *mut ze_driver_properties_t,
    ) -> ze_result_t;
    pub fn zeDevicePciGetPropertiesExt(
        hDevice: ze_device_handle_t,
        pPciProperties: *mut ze_pci_ext_properties_t,
    ) -> ze_result_t;
    pub fn zeDeviceCanAccessPeer(
        hDevice: ze_device_handle_t,
        hPeerDevice: ze_device_handle_t,
        value: *mut ze_bool_t,
    ) -> ze_result_t;
    pub fn zeContextCreate(
        hDriver: ze_driver_handle_t,
        desc: *const ze_context_desc_t,
        phContext: *mut ze_context_handle_t,
    ) -> ze_result_t;
    pub fn zeContextDestroy(hContext: ze_context_handle_t) -> ze_result_t;
    pub fn zeCommandQueueDestroy(hCommandQueue: ze_command_queue_handle_t) -> ze_result_t;
    pub fn zeCommandListDestroy(hCommandList: ze_command_list_handle_t) -> ze_result_t;
    pub fn zeModuleDestroy(hModule: ze_module_handle_t) -> ze_result_t;
    pub fn zeKernelDestroy(hKernel: ze_kernel_handle_t) -> ze_result_t;
    pub fn zeCommandQueueCreate(
        hContext: ze_context_handle_t,
        hDevice: ze_device_handle_t,
        desc: *const ze_command_queue_desc_t,
        phCommandQueue: *mut ze_command_queue_handle_t,
    ) -> ze_result_t;
    pub fn zeCommandListCreate(
        hContext: ze_context_handle_t,
        hDevice: ze_device_handle_t,
        desc: *const ze_command_list_desc_t,
        phCommandList: *mut ze_command_list_handle_t,
    ) -> ze_result_t;
    pub fn zeMemAllocHost(
        hContext: ze_context_handle_t,
        host_desc: *const ze_host_mem_alloc_desc_t,
        size: usize,
        alignment: usize,
        pptr: *mut *mut c_void,
    ) -> ze_result_t;
    pub fn zeMemAllocDevice(
        hContext: ze_context_handle_t,
        device_desc: *const ze_device_mem_alloc_desc_t,
        size: usize,
        alignment: usize,
        hDevice: ze_device_handle_t,
        pptr: *mut *mut c_void,
    ) -> ze_result_t;
    pub fn zeMemGetAllocProperties(
        hContext: ze_context_handle_t,
        ptr: *const c_void,
        pMemAllocProperties: *mut ze_memory_allocation_properties_t,
        phDevice: *mut ze_device_handle_t,
    ) -> ze_result_t;
    pub fn zeMemFree(hContext: ze_context_handle_t, ptr: *mut c_void) -> ze_result_t;
}

unsafe extern "C" {
    fn dup(fd: c_int) -> c_int;
    fn close(fd: c_int) -> c_int;
    pub fn zeModuleCreate(
        hContext: ze_context_handle_t,
        hDevice: ze_device_handle_t,
        desc: *const ze_module_desc_t,
        phModule: *mut ze_module_handle_t,
        phBuildLog: *mut *mut c_void,
    ) -> ze_result_t;
    pub fn zeKernelCreate(
        hModule: ze_module_handle_t,
        desc: *const ze_kernel_desc_t,
        phKernel: *mut ze_kernel_handle_t,
    ) -> ze_result_t;
    pub fn zeKernelSetIndirectAccess(hKernel: ze_kernel_handle_t, flags: u32) -> ze_result_t;
    pub fn zeKernelSetGroupSize(
        hKernel: ze_kernel_handle_t,
        groupSizeX: u32,
        groupSizeY: u32,
        groupSizeZ: u32,
    ) -> ze_result_t;
    pub fn zeKernelSetArgumentValue(
        hKernel: ze_kernel_handle_t,
        argIndex: u32,
        argSize: usize,
        pArgValue: *const c_void,
    ) -> ze_result_t;
    pub fn zeCommandListAppendLaunchKernel(
        hCommandList: ze_command_list_handle_t,
        hKernel: ze_kernel_handle_t,
        pLaunchFuncArgs: *const ze_group_count_t,
        hSignalEvent: *mut c_void,
        numWaitEvents: u32,
        phWaitEvents: *mut *mut c_void,
    ) -> ze_result_t;
    pub fn zeCommandListAppendMemoryCopy(
        hCommandList: ze_command_list_handle_t,
        dstptr: *mut c_void,
        srcptr: *const c_void,
        size: usize,
        hSignalEvent: *mut c_void,
        numWaitEvents: u32,
        phWaitEvents: *mut *mut c_void,
    ) -> ze_result_t;
    pub fn zeCommandListAppendMemoryFill(
        hCommandList: ze_command_list_handle_t,
        ptr: *mut c_void,
        pattern: *const c_void,
        pattern_size: usize,
        size: usize,
        hSignalEvent: *mut c_void,
        numWaitEvents: u32,
        phWaitEvents: *mut *mut c_void,
    ) -> ze_result_t;
    pub fn zeCommandListAppendBarrier(
        hCommandList: ze_command_list_handle_t,
        hSignalEvent: *mut c_void,
        numWaitEvents: u32,
        phWaitEvents: *mut *mut c_void,
    ) -> ze_result_t;
    pub fn zeCommandListClose(hCommandList: ze_command_list_handle_t) -> ze_result_t;
    pub fn zeCommandListReset(hCommandList: ze_command_list_handle_t) -> ze_result_t;
    pub fn zeCommandQueueExecuteCommandLists(
        hCommandQueue: ze_command_queue_handle_t,
        numCommandLists: u32,
        phCommandLists: *const ze_command_list_handle_t,
        hFence: *mut c_void,
    ) -> ze_result_t;
    pub fn zeCommandQueueSynchronize(hCommandQueue: ze_command_queue_handle_t, timeout: u64) -> ze_result_t;
}

/// Work-groups per megakernel launch, one per Xe2 core.
pub const MEGAKERNEL_GROUPS: u32 = 32;

#[inline]
pub fn check_ze(res: ze_result_t, msg: &str) {
    if res != ZE_RESULT_SUCCESS {
        panic!("Level Zero error 0x{:08x} in {}", res, msg);
    }
}

pub struct B70Card {
    pub index: usize,
    pub driver: ze_driver_handle_t,
    pub context: ze_context_handle_t,
    pub handle: ze_device_handle_t,
    pub name: String,
    /// The PCI address of the card, as `dddd:bb:dd.f` in lower-case hex (spec 2.1).
    pub pci_address: String,
    /// The PCI vendor id and device id of the card.
    pub vendor_id: u32,
    pub device_id: u32,
    pub slices: u32,
    pub subslices_per_slice: u32,
    pub eus_per_subslice: u32,
    pub threads_per_eu: u32,
    pub clock_mhz: u32,
    pub compute_queue: ze_command_queue_handle_t,
    pub copy_queue: ze_command_queue_handle_t,
    pub cmd_list: ze_command_list_handle_t,
    /// A command list on the copy queue for `wake`, and the 64 device bytes it copies.
    wake_list: ze_command_list_handle_t,
    wake_buf: *mut u8,
    wake_lock: std::sync::Mutex<()>,
    /// Device copy of the task ring the kernel reads: each work-group reads every descriptor,
    /// and reading them from host USM costs a PCIe round trip per task.
    ring_dev: *mut u8,
    pub module: ze_module_handle_t,
    pub kernel: ze_kernel_handle_t,
}

unsafe impl Send for B70Card {}
unsafe impl Sync for B70Card {}

impl Drop for B70Card {
    /// Destroy the card's objects and its context (engine-abi "`close` in Rust"). The owner of each other buffer of the
    /// context frees it first, because a destroyed context does not free the memory that it holds.
    fn drop(&mut self) {
        // No kernel runs: each launch waits for its end before it returns.
        self.free(self.ring_dev as *mut c_void);
        self.free(self.wake_buf as *mut c_void);
        unsafe {
            check_ze(zeKernelDestroy(self.kernel), "zeKernelDestroy");
            check_ze(zeModuleDestroy(self.module), "zeModuleDestroy");
            check_ze(zeCommandListDestroy(self.wake_list), "zeCommandListDestroy(wake)");
            check_ze(zeCommandListDestroy(self.cmd_list), "zeCommandListDestroy");
            check_ze(zeCommandQueueDestroy(self.copy_queue), "zeCommandQueueDestroy(copy)");
            check_ze(
                zeCommandQueueDestroy(self.compute_queue),
                "zeCommandQueueDestroy(compute)",
            );
            check_ze(zeContextDestroy(self.context), "zeContextDestroy");
        }
    }
}

impl B70Card {
    pub fn xe2_cores(&self) -> u32 {
        self.slices * self.subslices_per_slice
    }
    pub fn xmx_engines(&self) -> u32 {
        self.xe2_cores() * self.eus_per_subslice
    }

    pub fn alloc_host_usm<T>(&self, count: usize) -> *mut T {
        self.alloc_host_usm_flags(count, 2) // ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED
    }

    /// Host USM the CPU caches, for outputs the host reads only after the kernel has finished
    /// (logits, routing decisions): reading uncached memory runs at a fraction of DDR5 speed.
    pub fn alloc_host_usm_cached<T>(&self, count: usize) -> *mut T {
        self.alloc_host_usm_flags(count, 0)
    }

    fn alloc_host_usm_flags<T>(&self, count: usize, flags: u32) -> *mut T {
        unsafe {
            let desc = ze_host_mem_alloc_desc_t {
                stype: ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                pNext: ptr::null(),
                flags,
            };
            let mut ptr: *mut c_void = ptr::null_mut();
            let bytes = count * std::mem::size_of::<T>();
            check_ze(
                zeMemAllocHost(self.context, &desc, bytes, 64, &mut ptr),
                "zeMemAllocHost",
            );
            ptr::write_bytes(ptr as *mut u8, 0, bytes);
            ptr as *mut T
        }
    }

    pub fn alloc_device_usm<T>(&self, count: usize) -> *mut T {
        unsafe {
            let desc = ze_device_mem_alloc_desc_t {
                stype: ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                pNext: ptr::null(),
                flags: 0,
                ordinal: 0,
            };
            let mut ptr: *mut c_void = ptr::null_mut();
            let bytes = count * std::mem::size_of::<T>();
            check_ze(
                zeMemAllocDevice(self.context, &desc, bytes, 64, self.handle, &mut ptr),
                "zeMemAllocDevice",
            );
            ptr as *mut T
        }
    }

    /// Allocate high-bandwidth B70 GDDR6 VRAM with `ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF`
    /// and export it as a Linux `DMA-BUF` file descriptor for `systemd` `FDSTORE=1` hot handoff!
    pub fn alloc_exportable_vram_dmabuf<T>(&self, count: usize) -> (*mut T, c_int) {
        unsafe {
            let exp_desc = ze_external_memory_export_desc_t {
                stype: ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC,
                pNext: ptr::null(),
                flags: ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
            };
            let dev_desc = ze_device_mem_alloc_desc_t {
                stype: ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                pNext: &exp_desc as *const _ as *const c_void,
                flags: 0,
                ordinal: 0,
            };
            let mut ptr: *mut c_void = ptr::null_mut();
            let bytes = count * std::mem::size_of::<T>();
            check_ze(
                zeMemAllocDevice(self.context, &dev_desc, bytes, 64, self.handle, &mut ptr),
                "zeMemAllocDevice(DMA-BUF exportable)",
            );

            let mut exp_fd = ze_external_memory_export_fd_t {
                stype: ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD,
                pNext: ptr::null(),
                flags: ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
                fd: -1,
            };
            let mut mprops = ze_memory_allocation_properties_t {
                stype: ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: &mut exp_fd as *mut _ as *mut c_void,
                r#type: 0,
                id: 0,
                pageSize: 0,
            };
            check_ze(
                zeMemGetAllocProperties(self.context, ptr, &mut mprops, ptr::null_mut()),
                "zeMemGetAllocProperties(EXPORT_FD)",
            );
            assert!(exp_fd.fd >= 0, "Failed to export valid DMA-BUF fd");
            (ptr as *mut T, exp_fd.fd)
        }
    }

    /// Frees device memory this card's context allocated or imported.
    pub fn free(&self, ptr: *mut c_void) {
        unsafe { check_ze(zeMemFree(self.context, ptr), "zeMemFree") };
    }

    /// Re-import an existing Linux `DMA-BUF` file descriptor (e.g., recovered from `systemd`
    /// `$LISTEN_FDS` across a process restart) into a Level Zero context without copying!
    /// The driver closes the fd it imports, so this imports a duplicate and `dmabuf_fd` stays
    /// open for the caller.
    pub fn import_vram_from_dmabuf_fd<T>(
        &self,
        target_ctx: ze_context_handle_t,
        dmabuf_fd: c_int,
        count: usize,
    ) -> *mut T {
        self.try_import_vram_from_dmabuf_fd(target_ctx, dmabuf_fd, count)
            .unwrap_or_else(|e| panic!("{}", e))
    }

    /// `import_vram_from_dmabuf_fd`, returning the driver's error instead of panicking.
    pub fn try_import_vram_from_dmabuf_fd<T>(
        &self,
        target_ctx: ze_context_handle_t,
        dmabuf_fd: c_int,
        count: usize,
    ) -> Result<*mut T, String> {
        unsafe {
            let dup_fd = dup(dmabuf_fd);
            if dup_fd < 0 {
                return Err(format!("dup of DMA-BUF fd {} failed", dmabuf_fd));
            }
            let imp_fd = ze_external_memory_import_fd_t {
                stype: ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
                pNext: ptr::null(),
                flags: ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
                fd: dup_fd,
            };
            let dev_desc = ze_device_mem_alloc_desc_t {
                stype: ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                pNext: &imp_fd as *const _ as *const c_void,
                flags: 0,
                ordinal: 0,
            };
            let mut ptr: *mut c_void = ptr::null_mut();
            let bytes = count * std::mem::size_of::<T>();
            let r = zeMemAllocDevice(target_ctx, &dev_desc, bytes, 64, self.handle, &mut ptr);
            if r != ZE_RESULT_SUCCESS {
                close(dup_fd);
                return Err(format!(
                    "Level Zero error 0x{:08x} importing DMA-BUF fd {}",
                    r, dmabuf_fd
                ));
            }
            Ok(ptr as *mut T)
        }
    }

    /// Execute a hardware PCIe 5.0 DMA transfer (`Host <-> Device GDDR6 VRAM` or `Device <-> Device`)
    pub fn memcpy_sync(&self, dst: *mut c_void, src: *const c_void, bytes: usize) {
        unsafe {
            check_ze(zeCommandListReset(self.cmd_list), "zeCommandListReset(memcpy)");
            check_ze(
                zeCommandListAppendMemoryCopy(self.cmd_list, dst, src, bytes, ptr::null_mut(), 0, ptr::null_mut()),
                "zeCommandListAppendMemoryCopy",
            );
            check_ze(zeCommandListClose(self.cmd_list), "zeCommandListClose(memcpy)");
            let lists = [self.cmd_list];
            check_ze(
                zeCommandQueueExecuteCommandLists(self.compute_queue, 1, lists.as_ptr(), ptr::null_mut()),
                "zeCommandQueueExecuteCommandLists(memcpy)",
            );
            check_ze(
                zeCommandQueueSynchronize(self.compute_queue, u64::MAX),
                "zeCommandQueueSynchronize(memcpy)",
            );
        }
    }

    /// Fills `bytes` of device memory at `dst` with zeros and waits.
    pub fn fill_zero_sync(&self, dst: *mut c_void, bytes: usize) {
        let zero = 0u32;
        unsafe {
            check_ze(zeCommandListReset(self.cmd_list), "zeCommandListReset(fill)");
            check_ze(
                zeCommandListAppendMemoryFill(
                    self.cmd_list,
                    dst,
                    &zero as *const u32 as *const c_void,
                    4,
                    bytes,
                    ptr::null_mut(),
                    0,
                    ptr::null_mut(),
                ),
                "zeCommandListAppendMemoryFill",
            );
            check_ze(zeCommandListClose(self.cmd_list), "zeCommandListClose(fill)");
            let lists = [self.cmd_list];
            check_ze(
                zeCommandQueueExecuteCommandLists(self.compute_queue, 1, lists.as_ptr(), ptr::null_mut()),
                "zeCommandQueueExecuteCommandLists(fill)",
            );
            check_ze(
                zeCommandQueueSynchronize(self.compute_queue, u64::MAX),
                "zeCommandQueueSynchronize(fill)",
            );
        }
    }

    /// Keeps the card out of its idle power state: queues a 64-byte copy on the copy engine
    /// and returns without waiting for it. After 10 ms or more idle, the card's next command
    /// waits 100-550 us to wake.
    pub fn wake(&self) {
        let _guard = self.wake_lock.lock().unwrap();
        unsafe {
            check_ze(
                zeCommandQueueSynchronize(self.copy_queue, u64::MAX),
                "zeCommandQueueSynchronize(wake)",
            );
            check_ze(zeCommandListReset(self.wake_list), "zeCommandListReset(wake)");
            check_ze(
                zeCommandListAppendMemoryCopy(
                    self.wake_list,
                    self.wake_buf.add(64) as *mut c_void,
                    self.wake_buf as *const c_void,
                    64,
                    ptr::null_mut(),
                    0,
                    ptr::null_mut(),
                ),
                "zeCommandListAppendMemoryCopy(wake)",
            );
            check_ze(zeCommandListClose(self.wake_list), "zeCommandListClose(wake)");
            check_ze(
                zeCommandQueueExecuteCommandLists(self.copy_queue, 1, &self.wake_list, ptr::null_mut()),
                "zeCommandQueueExecuteCommandLists(wake)",
            );
        }
    }

    pub unsafe fn launch_megakernel(
        &self,
        task_ring: *const c_void,
        num_tasks: u32,
        scoreboard: *mut c_void,
        done_events: usize,
        scoreboard_events: usize,
        telemetry_ring: *const c_void,
        core_stats: *const c_void,
        scratch: *mut c_void,
    ) {
        unsafe {
            let gpu_id_u32 = self.index as u32;
            let group_count = ze_group_count_t {
                groupCountX: MEGAKERNEL_GROUPS,
                groupCountY: 1,
                groupCountZ: 1,
            };

            check_ze(zeCommandListReset(self.cmd_list), "zeCommandListReset");
            // The scoreboard is device memory, filled before the kernel reads it: events below
            // `done_events` read as finished by every work-group, the rest as not started.
            let (done, zero) = (MEGAKERNEL_GROUPS, 0u32);
            for (value, from, to) in [(&done, 0, done_events), (&zero, done_events, scoreboard_events)] {
                if to > from {
                    check_ze(
                        zeCommandListAppendMemoryFill(
                            self.cmd_list,
                            (scoreboard as *mut u32).add(from) as *mut c_void,
                            value as *const u32 as *const c_void,
                            4,
                            (to - from) * 4,
                            ptr::null_mut(),
                            0,
                            ptr::null_mut(),
                        ),
                        "zeCommandListAppendMemoryFill(scoreboard)",
                    );
                }
            }
            let task_ring = if task_ring.is_null() {
                task_ring
            } else {
                assert!(num_tasks as usize <= crate::protocol::MAX_TASKS_PER_CARD);
                check_ze(
                    zeCommandListAppendMemoryCopy(
                        self.cmd_list,
                        self.ring_dev as *mut c_void,
                        task_ring,
                        num_tasks as usize * 64,
                        ptr::null_mut(),
                        0,
                        ptr::null_mut(),
                    ),
                    "zeCommandListAppendMemoryCopy(ring)",
                );
                self.ring_dev as *const c_void
            };
            check_ze(
                zeCommandListAppendBarrier(self.cmd_list, ptr::null_mut(), 0, ptr::null_mut()),
                "zeCommandListAppendBarrier",
            );
            check_ze(
                zeKernelSetGroupSize(self.kernel, 1024, 1, 1),
                "zeKernelSetGroupSize(1024)",
            );

            check_ze(
                zeKernelSetArgumentValue(self.kernel, 0, 8, &task_ring as *const _ as *const c_void),
                "arg 0",
            );
            check_ze(
                zeKernelSetArgumentValue(self.kernel, 1, 4, &num_tasks as *const _ as *const c_void),
                "arg 1",
            );
            check_ze(
                zeKernelSetArgumentValue(self.kernel, 2, 8, &scoreboard as *const _ as *const c_void),
                "arg 2",
            );
            check_ze(
                zeKernelSetArgumentValue(self.kernel, 3, 8, &telemetry_ring as *const _ as *const c_void),
                "arg 3",
            );
            check_ze(
                zeKernelSetArgumentValue(self.kernel, 4, 8, &core_stats as *const _ as *const c_void),
                "arg 4",
            );
            check_ze(
                zeKernelSetArgumentValue(self.kernel, 5, 4, &gpu_id_u32 as *const _ as *const c_void),
                "arg 5",
            );
            check_ze(
                zeKernelSetArgumentValue(self.kernel, 6, 8, &scratch as *const _ as *const c_void),
                "arg 6",
            );

            check_ze(
                zeCommandListAppendLaunchKernel(
                    self.cmd_list,
                    self.kernel,
                    &group_count,
                    ptr::null_mut(),
                    0,
                    ptr::null_mut(),
                ),
                "zeCommandListAppendLaunchKernel",
            );
            check_ze(zeCommandListClose(self.cmd_list), "zeCommandListClose");
            check_ze(
                zeCommandQueueExecuteCommandLists(self.compute_queue, 1, &self.cmd_list, ptr::null_mut()),
                "zeCommandQueueExecuteCommandLists",
            );
            check_ze(
                zeCommandQueueSynchronize(self.compute_queue, u64::MAX),
                "zeCommandQueueSynchronize",
            );
        }
    }
}

pub struct L0Cluster {
    pub driver: ze_driver_handle_t,
    pub cards: Vec<B70Card>,
}

impl L0Cluster {
    /// The version of the Level Zero driver (`zeDriverGetProperties`), for the plan table lineage (spec 12.1).
    pub fn driver_version(&self) -> u32 {
        unsafe {
            let mut props: ze_driver_properties_t = std::mem::zeroed();
            props.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
            check_ze(zeDriverGetProperties(self.driver, &mut props), "zeDriverGetProperties");
            props.driverVersion
        }
    }

    /// The index in `cards` of the card at each PCI address, in the order of `addresses`.
    ///
    /// The order of `zeDeviceGet` can differ from the PCI order, so an engine finds its cards by address. An address
    /// that no card has, or that two cards have, is an error. Two equal addresses in `addresses` are an error.
    pub fn cards_at(&self, addresses: &[String]) -> Result<Vec<usize>, String> {
        crate::spec::match_addresses(
            &self
                .cards
                .iter()
                .map(|card| card.pci_address.clone())
                .collect::<Vec<_>>(),
            addresses,
        )
    }
}

unsafe impl Send for L0Cluster {}
unsafe impl Sync for L0Cluster {}

impl L0Cluster {
    pub fn init(kernel_bin: &[u8], kernel_name: &str) -> Self {
        unsafe {
            check_ze(zeInit(0), "zeInit");
            let mut drv_count = 0;
            check_ze(zeDriverGet(&mut drv_count, ptr::null_mut()), "zeDriverGet count");
            assert!(drv_count > 0, "No Level Zero driver found");
            let mut driver = ptr::null_mut();
            let mut one = 1;
            check_ze(zeDriverGet(&mut one, &mut driver), "zeDriverGet");

            let mut dev_count = 0;
            check_ze(
                zeDeviceGet(driver, &mut dev_count, ptr::null_mut()),
                "zeDeviceGet count",
            );
            let mut raw_devs = vec![ptr::null_mut(); dev_count as usize];
            check_ze(
                zeDeviceGet(driver, &mut dev_count, raw_devs.as_mut_ptr()),
                "zeDeviceGet",
            );

            let c_kname = CString::new(kernel_name).unwrap();
            let mut cards = Vec::with_capacity(dev_count as usize);

            for (idx, &dev) in raw_devs.iter().enumerate() {
                let ctx_desc = ze_context_desc_t {
                    stype: ZE_STRUCTURE_TYPE_CONTEXT_DESC,
                    pNext: ptr::null(),
                    flags: 0,
                };
                let mut context = ptr::null_mut();
                check_ze(
                    zeContextCreate(driver, &ctx_desc, &mut context),
                    "zeContextCreate(per-card)",
                );

                let mut props: ze_device_properties_t = std::mem::zeroed();
                props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                check_ze(zeDeviceGetProperties(dev, &mut props), "zeDeviceGetProperties");

                let name = CStr::from_ptr(props.name.as_ptr()).to_string_lossy().into_owned();
                let mut pci: ze_pci_ext_properties_t = std::mem::zeroed();
                pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
                check_ze(
                    zeDevicePciGetPropertiesExt(dev, &mut pci),
                    "zeDevicePciGetPropertiesExt",
                );
                let pci_address = format!(
                    "{:04x}:{:02x}:{:02x}.{:x}",
                    pci.address.domain, pci.address.bus, pci.address.device, pci.address.function
                );

                let q_desc = ze_command_queue_desc_t {
                    stype: ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                    pNext: ptr::null(),
                    ordinal: 0,
                    index: 0,
                    flags: 0,
                    mode: ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                    priority: 0,
                };
                let mut compute_queue = ptr::null_mut();
                check_ze(
                    zeCommandQueueCreate(context, dev, &q_desc, &mut compute_queue),
                    "zeCommandQueueCreate(compute)",
                );

                let copy_q_desc = ze_command_queue_desc_t {
                    stype: ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                    pNext: ptr::null(),
                    ordinal: 1,
                    index: 0,
                    flags: 0,
                    mode: ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                    priority: 0,
                };
                let mut copy_queue = ptr::null_mut();
                check_ze(
                    zeCommandQueueCreate(context, dev, &copy_q_desc, &mut copy_queue),
                    "zeCommandQueueCreate(copy)",
                );

                let cl_desc = ze_command_list_desc_t {
                    stype: ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                    pNext: ptr::null(),
                    commandQueueGroupOrdinal: 0,
                    flags: 0,
                };
                let mut cmd_list = ptr::null_mut();
                check_ze(
                    zeCommandListCreate(context, dev, &cl_desc, &mut cmd_list),
                    "zeCommandListCreate",
                );
                let wake_desc = ze_command_list_desc_t {
                    commandQueueGroupOrdinal: 1,
                    ..cl_desc
                };
                let mut wake_list = ptr::null_mut();
                check_ze(
                    zeCommandListCreate(context, dev, &wake_desc, &mut wake_list),
                    "zeCommandListCreate(wake)",
                );

                let mod_desc = ze_module_desc_t {
                    stype: ZE_STRUCTURE_TYPE_MODULE_DESC,
                    pNext: ptr::null(),
                    format: ZE_MODULE_FORMAT_NATIVE,
                    inputSize: kernel_bin.len(),
                    pInputModule: kernel_bin.as_ptr(),
                    pBuildFlags: ptr::null(),
                    pConstants: ptr::null(),
                };
                let mut module = ptr::null_mut();
                check_ze(
                    zeModuleCreate(context, dev, &mod_desc, &mut module, ptr::null_mut()),
                    "zeModuleCreate(NATIVE bmg-g31)",
                );

                let k_desc = ze_kernel_desc_t {
                    stype: ZE_STRUCTURE_TYPE_KERNEL_DESC,
                    pNext: ptr::null(),
                    flags: 0,
                    pKernelName: c_kname.as_ptr(),
                };
                let mut kernel = ptr::null_mut();
                check_ze(zeKernelCreate(module, &k_desc, &mut kernel), "zeKernelCreate");
                check_ze(zeKernelSetIndirectAccess(kernel, 7), "zeKernelSetIndirectAccess");

                cards.push(B70Card {
                    index: idx,
                    driver,
                    context,
                    handle: dev,
                    name,
                    pci_address,
                    vendor_id: props.vendorId,
                    device_id: props.deviceId,
                    slices: props.numSlices,
                    subslices_per_slice: props.numSubslicesPerSlice,
                    eus_per_subslice: props.numEUsPerSubslice,
                    threads_per_eu: props.numThreadsPerEU,
                    clock_mhz: props.coreClockRate,
                    compute_queue,
                    copy_queue,
                    cmd_list,
                    wake_list,
                    wake_buf: ptr::null_mut(),
                    wake_lock: std::sync::Mutex::new(()),
                    ring_dev: ptr::null_mut(),
                    module,
                    kernel,
                });
                let card = cards.last_mut().unwrap();
                card.wake_buf = card.alloc_device_usm::<u8>(128);
                card.ring_dev = card.alloc_device_usm::<u8>(crate::protocol::MAX_TASKS_PER_CARD * 64);
            }

            Self { driver, cards }
        }
    }
}
