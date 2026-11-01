//! The binding of the HIP runtime that the ops use: device memory, copies, streams and events.
//!
//! The HIP device is the device that is current on the calling thread. Make each call of one engine from one thread.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt;

unsafe extern "C" {
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn hipFree(ptr: *mut c_void) -> c_int;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> c_int;
    fn hipMemsetAsync(dst: *mut c_void, value: c_int, size: usize, stream: *mut c_void) -> c_int;
    fn hipMemcpy2DAsync(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn hipStreamCreate(stream: *mut *mut c_void) -> c_int;
    fn hipStreamDestroy(stream: *mut c_void) -> c_int;
    fn hipStreamSynchronize(stream: *mut c_void) -> c_int;
    fn hipEventCreate(event: *mut *mut c_void) -> c_int;
    fn hipEventDestroy(event: *mut c_void) -> c_int;
    fn hipEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
    fn hipEventSynchronize(event: *mut c_void) -> c_int;
    fn hipEventElapsedTime(ms: *mut f32, start: *mut c_void, stop: *mut c_void) -> c_int;
    fn hipGetErrorString(error: c_int) -> *const c_char;
    fn hipGetLastError() -> c_int;
    fn hipRuntimeGetVersion(version: *mut c_int) -> c_int;
    fn hipDeviceGetPCIBusId(bus: *mut c_char, len: c_int, device: c_int) -> c_int;
    fn hipGetDevice(device: *mut c_int) -> c_int;
}

const HOST_TO_DEVICE: c_int = 1;
const DEVICE_TO_HOST: c_int = 2;
const DEVICE_TO_DEVICE: c_int = 3;

/// A failed HIP call: its name, its code and the text of the runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HipError {
    pub call: &'static str,
    pub code: i32,
    pub text: String,
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed with HIP error {}: {}", self.call, self.code, self.text)
    }
}

impl std::error::Error for HipError {}

/// Give `Ok` for a code of 0, or the error of `call`.
pub fn check(call: &'static str, code: c_int) -> Result<(), HipError> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: hipGetErrorString gives a static string for each code.
    let text = unsafe { CStr::from_ptr(hipGetErrorString(code)) }
        .to_string_lossy()
        .into_owned();
    Err(HipError { call, code, text })
}

/// The error of the last kernel launch on this thread, if any. A launch has no return code, so check after it.
pub fn last_error(call: &'static str) -> Result<(), HipError> {
    // SAFETY: no argument.
    check(call, unsafe { hipGetLastError() })
}

/// The version of the HIP runtime, as `hipRuntimeGetVersion` gives it.
pub fn runtime_version() -> Result<i32, HipError> {
    let mut version = 0;
    // SAFETY: the pointer is valid for the call.
    check("hipRuntimeGetVersion", unsafe { hipRuntimeGetVersion(&mut version) })?;
    Ok(version)
}

/// The PCI vendor and device ids of the current device, as `vvvv:dddd` from sysfs.
pub fn pci_ids() -> Result<String, HipError> {
    let mut device = 0;
    // SAFETY: the pointer is valid for the call.
    check("hipGetDevice", unsafe { hipGetDevice(&mut device) })?;
    let mut bus = [0 as c_char; 64];
    // SAFETY: the buffer holds `len` bytes.
    check("hipDeviceGetPCIBusId", unsafe {
        hipDeviceGetPCIBusId(bus.as_mut_ptr(), bus.len() as c_int, device)
    })?;
    // SAFETY: the runtime wrote a NUL-terminated string into the buffer.
    let bus = unsafe { CStr::from_ptr(bus.as_ptr()) }.to_string_lossy().to_lowercase();
    let read = |name: &str| {
        let path = format!("/sys/bus/pci/devices/{bus}/{name}");
        std::fs::read_to_string(&path)
            .map(|text| text.trim().trim_start_matches("0x").to_owned())
            .map_err(|error| HipError {
                call: "the PCI ids from sysfs",
                code: -1,
                text: format!("read {path}: {error}"),
            })
    };
    Ok(format!("{}:{}", read("vendor")?, read("device")?))
}

/// Copy `height` rows of `width` bytes on `stream`, from rows `spitch` bytes apart to rows `dpitch` bytes apart, in
/// device memory.
///
/// # Safety
///
/// Both pointers must be device memory that holds the rows at their pitches, and the regions must not overlap.
pub unsafe fn copy_2d(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    stream: &Stream,
) -> Result<(), HipError> {
    assert!(
        width <= dpitch && width <= spitch,
        "a row of {width} bytes is wider than its pitch"
    );
    if height == 0 {
        return Ok(());
    }
    // SAFETY: the caller keeps the rules above.
    check("hipMemcpy2DAsync", unsafe {
        hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, DEVICE_TO_DEVICE, stream.0)
    })
}

/// Device memory. It is freed on drop.
#[derive(Debug)]
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

// SAFETY: the buffer is a device address. The HIP runtime allows its use from any thread.
unsafe impl Send for DeviceBuffer {}

impl DeviceBuffer {
    /// Allocate `bytes` of device memory. The contents are not set.
    pub fn new(bytes: usize) -> Result<DeviceBuffer, HipError> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: the pointer is valid for the call. A size of 0 gives a null pointer.
        check("hipMalloc", unsafe { hipMalloc(&mut ptr, bytes.max(1)) })?;
        Ok(DeviceBuffer { ptr, bytes })
    }

    /// A buffer with a copy of `data`.
    pub fn from_slice<T: Copy>(data: &[T]) -> Result<DeviceBuffer, HipError> {
        let buffer = DeviceBuffer::new(std::mem::size_of_val(data))?;
        buffer.write(data)?;
        Ok(buffer)
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Copy `data` to the start of the buffer. It must fit.
    pub fn write<T: Copy>(&self, data: &[T]) -> Result<(), HipError> {
        let bytes = std::mem::size_of_val(data);
        assert!(
            bytes <= self.bytes,
            "a write of {bytes} bytes into a buffer of {}",
            self.bytes
        );
        // SAFETY: the source holds `bytes` bytes, and the buffer holds at least that.
        check("hipMemcpy", unsafe {
            hipMemcpy(self.ptr, data.as_ptr().cast(), bytes, HOST_TO_DEVICE)
        })
    }

    /// Copy the start of the buffer into `data`. It waits for the work before it on the null stream.
    pub fn read<T: Copy>(&self, data: &mut [T]) -> Result<(), HipError> {
        let bytes = std::mem::size_of_val(data);
        assert!(
            bytes <= self.bytes,
            "a read of {bytes} bytes from a buffer of {}",
            self.bytes
        );
        // SAFETY: the buffer holds at least `bytes` bytes, and so does the target.
        check("hipMemcpy", unsafe {
            hipMemcpy(data.as_mut_ptr().cast(), self.ptr, bytes, DEVICE_TO_HOST)
        })
    }

    /// The whole buffer as values of `T`.
    pub fn to_vec<T: Copy + Default>(&self) -> Result<Vec<T>, HipError> {
        let mut data = vec![T::default(); self.bytes / std::mem::size_of::<T>()];
        self.read(&mut data)?;
        Ok(data)
    }

    /// Copy `bytes` from the start of `source` into the start of this buffer.
    pub fn copy_from(&self, source: &DeviceBuffer, bytes: usize) -> Result<(), HipError> {
        assert!(
            bytes <= self.bytes && bytes <= source.bytes,
            "a copy of {bytes} bytes does not fit"
        );
        // SAFETY: both buffers hold at least `bytes` bytes.
        check("hipMemcpy", unsafe {
            hipMemcpy(self.ptr, source.ptr, bytes, DEVICE_TO_DEVICE)
        })
    }

    /// Set each byte to 0 on `stream`.
    pub fn zero(&self, stream: &Stream) -> Result<(), HipError> {
        // SAFETY: the buffer holds `bytes` bytes.
        check("hipMemsetAsync", unsafe {
            hipMemsetAsync(self.ptr, 0, self.bytes, stream.0)
        })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: the pointer came from hipMalloc, and nothing frees it again.
        let result = check("hipFree", unsafe { hipFree(self.ptr) });
        if let Err(error) = result
            && !std::thread::panicking()
        {
            panic!("{error}");
        }
    }
}

/// A HIP stream. It is destroyed on drop.
#[derive(Debug)]
pub struct Stream(*mut c_void);

// SAFETY: a stream handle can move between threads; each engine still calls from one thread.
unsafe impl Send for Stream {}

impl Stream {
    pub fn new() -> Result<Stream, HipError> {
        let mut stream = std::ptr::null_mut();
        // SAFETY: the pointer is valid for the call.
        check("hipStreamCreate", unsafe { hipStreamCreate(&mut stream) })?;
        Ok(Stream(stream))
    }

    pub fn raw(&self) -> *mut c_void {
        self.0
    }

    /// Wait until the work on the stream ends.
    pub fn synchronize(&self) -> Result<(), HipError> {
        // SAFETY: the stream is live.
        check("hipStreamSynchronize", unsafe { hipStreamSynchronize(self.0) })
    }

    /// The milliseconds of the GPU work that `work` puts on this stream. It waits for the work.
    pub fn time<E>(&self, work: impl FnOnce() -> Result<(), E>) -> Result<f32, E>
    where
        E: From<HipError>,
    {
        let mut start = std::ptr::null_mut();
        let mut stop = std::ptr::null_mut();
        // SAFETY: the pointers are valid for the calls, and the events are destroyed below.
        unsafe {
            check("hipEventCreate", hipEventCreate(&mut start))?;
            check("hipEventCreate", hipEventCreate(&mut stop))?;
            check("hipEventRecord", hipEventRecord(start, self.0))?;
        }
        let result = work();
        let mut ms = 0.0;
        // SAFETY: the events are live.
        let timed = unsafe {
            check("hipEventRecord", hipEventRecord(stop, self.0))
                .and_then(|()| check("hipEventSynchronize", hipEventSynchronize(stop)))
                .and_then(|()| check("hipEventElapsedTime", hipEventElapsedTime(&mut ms, start, stop)))
        };
        // SAFETY: the events are live, and nothing uses them after this.
        unsafe {
            hipEventDestroy(start);
            hipEventDestroy(stop);
        }
        result?;
        timed?;
        Ok(ms)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: the stream came from hipStreamCreate.
        unsafe {
            hipStreamDestroy(self.0);
        }
    }
}
