//! Helpers that several test files share: the reader of the fixture of the vLLM kernels, and (with the feature `cuda`)
//! the helpers of a GPU test.

#![allow(dead_code)]

use std::collections::BTreeMap;

use serde_json::Value;

pub struct Fixture {
    pub data: Vec<u8>,
    pub tensors: BTreeMap<String, (Vec<usize>, usize, usize)>,
    pub classes: Value,
}

impl Fixture {
    pub fn load() -> Self {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/quant_kernels.safetensors");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read the fixture {path}: {e}"));
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(&bytes[8..8 + header_len]).expect("the header is JSON");
        let classes: Value = serde_json::from_str(header["__metadata__"]["classes"].as_str().expect("classes"))
            .expect("the classes are JSON");
        let mut tensors = BTreeMap::new();
        for (name, entry) in header.as_object().unwrap() {
            if name == "__metadata__" {
                continue;
            }
            let shape = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let offsets = entry["data_offsets"].as_array().unwrap();
            let (a, b) = (
                offsets[0].as_u64().unwrap() as usize,
                offsets[1].as_u64().unwrap() as usize,
            );
            tensors.insert(name.clone(), (shape, a, b));
        }
        Self {
            data: bytes[8 + header_len..].to_vec(),
            tensors,
            classes,
        }
    }

    pub fn raw(&self, name: &str) -> (&[u8], &[usize]) {
        let (shape, a, b) = self
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("the fixture has no tensor {name}"));
        (&self.data[*a..*b], shape)
    }

    /// The rows of a bf16 tensor, widened to f32 (exact).
    pub fn rows_f32(&self, name: &str) -> Vec<Vec<f32>> {
        let (raw, shape) = self.raw(name);
        raw.chunks_exact(2 * shape[1])
            .map(|row| {
                row.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                    .collect()
            })
            .collect()
    }

    pub fn class(&self, set: &str, row: usize) -> &str {
        self.classes[set][row].as_str().expect("a class name")
    }
}

/// The helpers of a GPU test. They need the feature `cuda`.
#[cfg(feature = "cuda")]
pub mod gpu {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};

    use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
    use nagaya_mk_cuda::ring::Ring;
    use nagaya_mk_cuda::{
        INKLING2_CUBIN, INKLING2_PROBE_CUBIN, INKLING2_PROBE_PROFILE_CUBIN, INKLING2_PROFILE_CUBIN, test_cards,
    };

    /// The grid widths of a gate: the width of the GPU, and widths that divide no count of items.
    pub const GRIDS: [u32; 4] = [48, 24, 7, 1];

    /// Open the GPU that `NAGAYA_TEST_CARDS` names, with the interpreter.
    pub fn open() -> CudaDevice {
        CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU")
    }

    /// Open the GPU with the interpreter that holds only the cache format `format`.
    pub fn open_cache(format: nagaya_mk_cuda::attn::CacheFormat) -> CudaDevice {
        CudaDevice::open_by_pci(&test_cards::card(), format.cubin()).expect("open the GPU with the cubin of the format")
    }

    /// Open the GPU with the profile interpreter that holds the cache format `format`.
    pub fn open_profile_cache(format: nagaya_mk_cuda::attn::CacheFormat) -> CudaDevice {
        CudaDevice::open_by_pci(&test_cards::card(), format.profile_cubin())
            .expect("open the GPU with the profile cubin of the format")
    }

    /// Open the GPU with the interpreter that writes a profile of each task.
    pub fn open_profile() -> CudaDevice {
        CudaDevice::open_by_pci(&test_cards::card(), INKLING2_PROFILE_CUBIN)
            .expect("open the GPU with the profile cubin")
    }

    /// Open the GPU with the probe interpreter. It holds the probe ops, and no engine cubin does.
    pub fn open_probe() -> CudaDevice {
        CudaDevice::open_by_pci(&test_cards::card(), INKLING2_PROBE_CUBIN).expect("open the GPU with the probe cubin")
    }

    /// Open the GPU with the probe interpreter that also writes a profile of each task.
    pub fn open_probe_profile() -> CudaDevice {
        CudaDevice::open_by_pci(&test_cards::card(), INKLING2_PROBE_PROFILE_CUBIN)
            .expect("open the GPU with the probe profile cubin")
    }

    pub fn upload<T: Copy>(dev: &CudaDevice, data: &[T]) -> CUdeviceptr {
        let bytes = std::mem::size_of_val(data);
        let ptr = dev.alloc_device(bytes.max(16)).expect("allocate");
        unsafe { dev.copy_htod(ptr, data.as_ptr().cast::<c_void>(), bytes) }.expect("upload");
        ptr
    }

    pub fn download<T: Copy + Default>(dev: &CudaDevice, ptr: CUdeviceptr, count: usize) -> Vec<T> {
        let mut out = vec![T::default(); count];
        let bytes = std::mem::size_of_val(out.as_slice());
        unsafe { dev.copy_dtoh(out.as_mut_ptr().cast::<c_void>(), ptr, bytes) }.expect("download");
        out
    }

    /// Run the ring in a thread with a watchdog, so that a hang fails the test and does not block the GPU.
    pub fn run_with_watchdog(ring: &Ring<'_>, grid: u32, threads: u32, shared: u32) {
        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for _ in 0..3000 {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    if done.load(Ordering::SeqCst) {
                        return;
                    }
                }
                eprintln!("the ring did not finish within 30 s at grid {grid}; it hangs");
                std::process::exit(2);
            });
            unsafe { ring.run_grid(grid, threads, shared).unwrap() };
            done.store(true, Ordering::SeqCst);
        });
    }
}
