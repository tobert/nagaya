//! Helpers of the MoE gates (`cuda_route.rs`, `cuda_up.rs`, `cuda_down.rs`): the reader of the checkpoint of Inkling 2
//! Mini, the generator of rows, and the pair list that the device writes.

#![allow(dead_code)]
#![cfg(feature = "cuda")]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::protocol::{EXPERT_MAX_ROWS, EXPERT_PAIR_HEADER_BYTES, ExpertPair};
use nagaya_mk_cuda::safetensors::SafeTensorsHeader;

/// The folder of Mini, or `None` with one line on stderr. `NAGAYA_REQUIRE_MODELS=1` makes an absence a failure.
pub fn model_dir(test: &str) -> Option<PathBuf> {
    match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(dir) if PathBuf::from(&dir).join("model-00001-of-00001.safetensors").is_file() => Some(dir.into()),
        _ => {
            let require = std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1");
            assert!(
                !require,
                "NAGAYA_REQUIRE_MODELS=1 and INKLING2_MODEL_DIR holds no model-00001-of-00001.safetensors; set the variable to the folder of Inkling 2 Mini"
            );
            eprintln!("{test}: skipped: INKLING2_MODEL_DIR holds no model of Inkling 2 Mini");
            None
        }
    }
}

pub struct Checkpoint {
    header: SafeTensorsHeader,
    file: File,
}

impl Checkpoint {
    pub fn open(dir: &std::path::Path) -> Self {
        let path = dir.join("model-00001-of-00001.safetensors");
        let header = SafeTensorsHeader::read(&path).unwrap_or_else(|e| panic!("{e}"));
        let file = File::open(&path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
        Self { header, file }
    }

    /// `len` bytes of the tensor `name`, from `offset` bytes into it.
    pub fn range(&mut self, name: &str, offset: usize, len: usize) -> Vec<u8> {
        let meta = self
            .header
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("the checkpoint has no tensor {name}"));
        assert!(
            offset + len <= meta.nbytes as usize,
            "{name}: the range passes the end of the tensor"
        );
        self.file
            .seek(SeekFrom::Start(
                self.header.data_start + meta.data_offset + offset as u64,
            ))
            .unwrap();
        let mut bytes = vec![0u8; len];
        self.file
            .read_exact(&mut bytes)
            .unwrap_or_else(|e| panic!("cannot read {name}: {e}"));
        bytes
    }

    /// The whole tensor `name`.
    pub fn all(&mut self, name: &str) -> Vec<u8> {
        let nbytes = self
            .header
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("the checkpoint has no tensor {name}"))
            .nbytes as usize;
        self.range(name, 0, nbytes)
    }

    pub fn f32s(&mut self, name: &str, offset: usize, count: usize) -> Vec<f32> {
        self.range(name, offset * 4, count * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    /// A whole `bf16` tensor, widened to `f32` (exact).
    pub fn bf16_as_f32(&mut self, name: &str) -> Vec<f32> {
        widen_bf16(&self.all(name))
    }
}

pub fn widen_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
        .collect()
}

/// A generator of rows that is the same in each run: xorshift.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// An activation row: a sum of 3 uniform values, with a few large outliers, like a normalized hidden row.
pub fn activation_row(rng: &mut Rng, k: usize, magnitude: f32) -> Vec<f32> {
    let mut row: Vec<f32> = (0..k)
        .map(|_| (rng.unit() + rng.unit() + rng.unit()) * magnitude)
        .collect();
    for _ in 0..3 {
        let i = rng.next() as usize % k;
        row[i] *= 25.0;
    }
    row
}

/// Read the pair list that the device wrote at `list`, for room for `capacity` pairs: the live count, and the live pairs.
pub fn download_pairs(dev: &CudaDevice, list: CUdeviceptr, capacity: usize) -> Vec<ExpertPair> {
    let bytes = EXPERT_PAIR_HEADER_BYTES + capacity * std::mem::size_of::<ExpertPair>();
    let raw: Vec<u8> = super::common::gpu::download(dev, list, bytes);
    let live = u32::from_le_bytes(raw[..4].try_into().unwrap()) as usize;
    assert!(
        live <= capacity,
        "the live count {live} is above the room of {capacity} pairs"
    );
    let word = |at: usize| u32::from_le_bytes(raw[at..at + 4].try_into().unwrap());
    (0..live)
        .map(|i| {
            let base = EXPERT_PAIR_HEADER_BYTES + i * std::mem::size_of::<ExpertPair>();
            let mut pair = ExpertPair {
                expert: word(base),
                count: word(base + 4),
                ..Default::default()
            };
            for j in 0..EXPERT_MAX_ROWS {
                pair.x_row[j] = u16::from_le_bytes(raw[base + 8 + 2 * j..][..2].try_into().unwrap());
                pair.y_row[j] = u16::from_le_bytes(raw[base + 40 + 2 * j..][..2].try_into().unwrap());
                pair.weight[j] = f32::from_bits(word(base + 72 + 4 * j));
            }
            pair
        })
        .collect()
}
