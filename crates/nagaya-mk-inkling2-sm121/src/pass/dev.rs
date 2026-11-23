//! A device buffer that frees itself, and the reader of the checkpoint (`cuda` feature).

use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::safetensors::SafeTensorsHeader;

use crate::spec::WEIGHTS_FILE;

/// A buffer of device memory. `Drop` frees it and prints a failure to stderr.
pub struct Dev<'d> {
    dev: &'d CudaDevice,
    ptr: CUdeviceptr,
    bytes: usize,
}

impl<'d> Dev<'d> {
    /// Allocate `bytes` bytes (at least 16).
    pub fn alloc(dev: &'d CudaDevice, bytes: usize) -> Result<Self, String> {
        let bytes = bytes.max(16);
        let ptr = dev.alloc_device(bytes)?;
        Ok(Self { dev, ptr, bytes })
    }

    /// Allocate a buffer and copy `data` into it.
    pub fn upload(dev: &'d CudaDevice, data: &[u8]) -> Result<Self, String> {
        let buffer = Self::alloc(dev, data.len())?;
        buffer.write(0, data)?;
        Ok(buffer)
    }

    /// Allocate a buffer and copy the words of `data` into it.
    pub fn upload_f32(dev: &'d CudaDevice, data: &[f32]) -> Result<Self, String> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::upload(dev, &bytes)
    }

    pub fn ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Copy `data` to the byte `offset` of the buffer.
    pub fn write(&self, offset: usize, data: &[u8]) -> Result<(), String> {
        if offset + data.len() > self.bytes {
            return Err(format!(
                "a write of {} bytes at {offset} passes the end of a buffer of {} bytes",
                data.len(),
                self.bytes
            ));
        }
        // SAFETY: the range is inside the allocation, and `data` is valid for its length.
        unsafe {
            self.dev
                .copy_htod(self.ptr + offset as u64, data.as_ptr().cast::<c_void>(), data.len())
        }
    }

    /// Read `len` bytes from the byte `offset`.
    pub fn read(&self, offset: usize, len: usize) -> Result<Vec<u8>, String> {
        if offset + len > self.bytes {
            return Err(format!(
                "a read of {len} bytes at {offset} passes the end of a buffer of {} bytes",
                self.bytes
            ));
        }
        let mut out = vec![0u8; len];
        // SAFETY: the range is inside the allocation, and `out` has room for `len` bytes.
        unsafe {
            self.dev
                .copy_dtoh(out.as_mut_ptr().cast::<c_void>(), self.ptr + offset as u64, len)?;
        }
        Ok(out)
    }

    /// Set every byte to 0.
    pub fn zero(&self) -> Result<(), String> {
        self.dev.memset_zero(self.ptr, self.bytes)
    }
}

impl Drop for Dev<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.dev.free_device(self.ptr) {
            eprintln!(
                "nagaya-mk-inkling2-sm121: a device buffer of {} bytes did not free: {error}",
                self.bytes
            );
        }
    }
}

impl std::fmt::Debug for Dev<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dev({:#x}, {} bytes)", self.ptr, self.bytes)
    }
}

/// The checkpoint file: its header and a handle that reads ranges of tensors.
pub struct Checkpoint {
    header: SafeTensorsHeader,
    file: File,
}

impl Checkpoint {
    /// Open the weights file in the model folder `dir`.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let path = dir.join(WEIGHTS_FILE);
        let header = SafeTensorsHeader::read(&path)?;
        let file = File::open(&path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        Ok(Self { header, file })
    }

    /// The shape of the tensor `name`, or an error that names it.
    fn meta(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<(u64, u64), String> {
        let meta = self.header.tensors.get(name).ok_or_else(|| {
            format!("the checkpoint holds no tensor {name}. The folder is not Inkling 2 Mini, or the file is damaged")
        })?;
        if meta.dtype != dtype || meta.shape != shape {
            return Err(format!(
                "the tensor {name} is {} {:?}, and the engine needs {dtype} {shape:?}. The folder is not Inkling 2 Mini",
                meta.dtype, meta.shape
            ));
        }
        Ok((self.header.data_start + meta.data_offset, meta.nbytes))
    }

    /// The bytes of the tensor `name`, with the dtype and the shape checked.
    pub fn bytes(&mut self, name: &str, dtype: &str, shape: &[usize]) -> Result<Vec<u8>, String> {
        let (_, nbytes) = self.meta(name, dtype, shape)?;
        self.range(name, dtype, shape, 0, nbytes as usize)
    }

    /// `len` bytes of the tensor `name` from the byte `offset` of its payload.
    pub fn range(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[usize],
        offset: usize,
        len: usize,
    ) -> Result<Vec<u8>, String> {
        let (start, nbytes) = self.meta(name, dtype, shape)?;
        if (offset + len) as u64 > nbytes {
            return Err(format!(
                "a read of {len} bytes at {offset} passes the end of the tensor {name} ({nbytes} bytes)"
            ));
        }
        self.file
            .seek(SeekFrom::Start(start + offset as u64))
            .map_err(|e| format!("cannot seek in the tensor {name}: {e}"))?;
        let mut out = vec![0u8; len];
        self.file
            .read_exact(&mut out)
            .map_err(|e| format!("cannot read {len} bytes of the tensor {name}: {e}"))?;
        Ok(out)
    }

    /// A `bf16` or `f32` tensor widened to `f32`. Both widen exactly.
    pub fn f32s(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let dtype = self
            .header
            .tensors
            .get(name)
            .map(|m| m.dtype.clone())
            .ok_or_else(|| format!("the checkpoint holds no tensor {name}. The folder is not Inkling 2 Mini"))?;
        let bytes = self.bytes(name, &dtype, shape)?;
        match dtype.as_str() {
            "BF16" => Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                .collect()),
            "F32" => Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()),
            other => Err(format!(
                "the tensor {name} is {other}, and the engine needs BF16 or F32"
            )),
        }
    }
}
