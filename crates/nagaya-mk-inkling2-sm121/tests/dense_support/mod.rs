//! Helpers of the GPU tests of the dense ops: the checkpoint of Inkling 2 Mini, random rows and the row factor.

#![allow(dead_code)]

use std::fs::File;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use nagaya_mk_cuda::safetensors::SafeTensorsHeader;
use nagaya_mk_inkling2_bmg::host::ops;

pub const MODEL_VAR: &str = "INKLING2_MODEL_DIR";
pub const CHECKPOINT: &str = "model-00001-of-00001.safetensors";
pub const HIDDEN: usize = 2304;

/// The read peak of the GB10 that the probe measured, in GB/s.
pub const PEAK_GBPS: f64 = 252.0;

pub struct Checkpoint {
    header: SafeTensorsHeader,
    file: Mutex<File>,
}

/// The checkpoint, or `None` when the model is absent and the test skips. With `NAGAYA_REQUIRE_MODELS` an absence fails.
pub fn checkpoint(who: &str) -> Option<&'static Checkpoint> {
    static CELL: OnceLock<Option<Checkpoint>> = OnceLock::new();
    CELL.get_or_init(|| {
        let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
        let Some(dir) = std::env::var_os(MODEL_VAR) else {
            assert!(!require, "{MODEL_VAR} is not set and NAGAYA_REQUIRE_MODELS is set");
            eprintln!("{who}: skipped, {MODEL_VAR} is not set");
            return None;
        };
        let path = PathBuf::from(dir).join(CHECKPOINT);
        if !path.exists() {
            assert!(
                !require,
                "{} is absent and NAGAYA_REQUIRE_MODELS is set",
                path.display()
            );
            eprintln!("{who}: skipped, {} is absent", path.display());
            return None;
        }
        let header = SafeTensorsHeader::read(&path).unwrap_or_else(|e| panic!("{e}"));
        let file = File::open(&path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
        Some(Checkpoint {
            header,
            file: Mutex::new(file),
        })
    })
    .as_ref()
}

impl Checkpoint {
    /// The bytes of a tensor, with its dtype and shape checked.
    pub fn bytes(&self, name: &str, dtype: &str, shape: &[usize]) -> Vec<u8> {
        let meta = self
            .header
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("the checkpoint has no tensor {name}"));
        assert_eq!((meta.dtype.as_str(), &meta.shape[..]), (dtype, shape), "{name}");
        self.header
            .read_tensor(&mut self.file.lock().unwrap(), name)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// A `bf16` tensor widened to `f32` (exact).
    pub fn bf16(&self, name: &str, shape: &[usize]) -> Vec<f32> {
        widen(&self.bytes(name, "BF16", shape))
    }
}

/// `bf16` bytes widened to `f32`.
pub fn widen(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| ops::bf16_to_f32(u16::from_le_bytes(*b)))
        .collect()
}

pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in `[-scale, scale]` with all 24 bits of mantissa random.
    pub fn value(&mut self, scale: f32) -> f32 {
        let r = self.next();
        ((r >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * scale
    }
}

/// `rows` rows of `k` values of a residual: a scale of 1 to 8 for each row, so that the row factors differ.
pub fn make_rows(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let scale = 1.0 + (r % 8) as f32;
        out.extend((0..k).map(|_| rng.value(scale)));
    }
    out
}

/// The row factor, the way `ops::rms_norm` computes it. The producer task of a row writes it.
pub fn row_factor(x: &[f32]) -> f32 {
    let mean = ops::dot(x, x) / x.len() as f32;
    1.0 / (mean + ops::EPS).sqrt()
}

pub fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
