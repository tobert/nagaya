//! The input generator of the op tests: a stream of u64 values that is a pure function of a seed and an index.
//!
//! `tools/op_fixtures.py` has the same generator, and the test `the_generator_matches_the_python_generator` compares the
//! two. Thus Python writes the expected outputs for inputs that Rust remakes with no fixture file. Every operation is an
//! integer operation or an IEEE f64 operation that is exact in both languages, so the values are equal bit for bit.
//! The reference's own op tests use numpy's `default_rng`; this generator has the same shapes of input but not the same values.

#![allow(dead_code)]

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// The finalizer of splitmix64.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The value at `index` of the stream of `seed`.
pub fn u64_at(seed: u64, index: u64) -> u64 {
    let key = mix(seed.wrapping_mul(GOLDEN) ^ 0x6F70_735F_7465_7374);
    mix(key.wrapping_add((index.wrapping_add(1)).wrapping_mul(GOLDEN)))
}

/// A value in `0..n` at `index`. The remainder bias is no matter at these sizes.
pub fn below(seed: u64, index: u64, n: u64) -> u64 {
    u64_at(seed, index) % n
}

/// A value of the Irwin-Hall distribution with four terms, scaled to unit variance, at `index`. It is a sum of four
/// 16-bit integers, so it is exact, and it has the bell shape with no tail beyond 3.5 sigma.
pub fn normal_at(seed: u64, index: u64) -> f32 {
    let u = u64_at(seed, index);
    let sum = (u & 0xFFFF) + ((u >> 16) & 0xFFFF) + ((u >> 32) & 0xFFFF) + ((u >> 48) & 0xFFFF);
    let sd = (4.0 * (65536.0f64 * 65536.0 - 1.0) / 12.0).sqrt();
    ((sum as f64 - 4.0 * 32767.5) / sd) as f32
}

/// `n` values of `normal_at`, from index 0.
pub fn normals(seed: u64, n: usize) -> Vec<f32> {
    (0..n as u64).map(|index| normal_at(seed, index)).collect()
}

/// `n` normal values, each times `scale`, with the product taken in f32.
pub fn normals_scaled(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    normals(seed, n).into_iter().map(|value| value * scale).collect()
}

/// `n` uniform values in `[0, 1)`, at 24 bits.
pub fn uniforms(seed: u64, n: usize) -> Vec<f32> {
    (0..n as u64)
        .map(|index| (u64_at(seed, index) >> 40) as f32 / 16_777_216.0)
        .collect()
}

/// `n` bytes of the stream: byte `i` is byte `i % 8` of the little-endian value at index `i / 8`.
pub fn bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n + 8);
    let mut index = 0;
    while out.len() < n {
        out.extend_from_slice(&u64_at(seed, index).to_le_bytes());
        index += 1;
    }
    out.truncate(n);
    out
}

/// The quantization types of the reference's GGUF files, with the numbers of `quant.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qtype {
    F32 = 0,
    Q8_0 = 8,
    Q6K = 14,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq4Xs = 23,
    Bf16 = 30,
}

impl Qtype {
    pub fn name(self) -> &'static str {
        match self {
            Qtype::F32 => "F32",
            Qtype::Q8_0 => "Q8_0",
            Qtype::Q6K => "Q6_K",
            Qtype::Iq4Nl => "IQ4_NL",
            Qtype::Iq3S => "IQ3_S",
            Qtype::Iq4Xs => "IQ4_XS",
            Qtype::Bf16 => "BF16",
        }
    }

    /// The values in one block, and the bytes of one block.
    pub fn block(self) -> (usize, usize) {
        match self {
            Qtype::F32 => (1, 4),
            Qtype::Bf16 => (1, 2),
            Qtype::Q8_0 => (32, 34),
            Qtype::Iq4Nl => (32, 18),
            Qtype::Q6K => (256, 210),
            Qtype::Iq3S => (256, 110),
            Qtype::Iq4Xs => (256, 136),
        }
    }

    /// The bytes of a row of `k` values.
    pub fn row_bytes(self, k: usize) -> usize {
        let (values, bytes) = self.block();
        assert_eq!(
            k % values,
            0,
            "{} needs a multiple of {values} values, not {k}",
            self.name()
        );
        k / values * bytes
    }

    /// The byte offset of the f16 `d` in a block, if the type has one.
    fn d_offset(self) -> Option<usize> {
        match self {
            Qtype::Q8_0 | Qtype::Iq4Nl | Qtype::Iq3S | Qtype::Iq4Xs => Some(0),
            Qtype::Q6K => Some(208),
            Qtype::F32 | Qtype::Bf16 => None,
        }
    }
}

/// `rows` rows of `k` values in the GGUF layout of `qtype`, as random bytes with a finite f16 `d` in each block.
///
/// Random bytes could make an infinite or NaN `d`. The `d` of block `j` is the f16 with the bits `0x1400 + u % 0x1000`,
/// where `u` is the value at index `j` of the stream of `seed + 0x1000_0000`: a positive value from 2^-10 to 2^-6.
/// F32 rows hold normal values. BF16 rows hold the upper half of an f32 normal value.
pub fn raw_rows(qtype: Qtype, seed: u64, rows: usize, k: usize) -> Vec<u8> {
    let row = qtype.row_bytes(k);
    match qtype {
        Qtype::F32 => normals(seed, rows * k).iter().flat_map(|v| v.to_le_bytes()).collect(),
        Qtype::Bf16 => normals(seed, rows * k)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        _ => {
            let mut out = bytes(seed, rows * row);
            let (_, block) = qtype.block();
            let offset = qtype.d_offset().expect("a block type has a d");
            for j in 0..(rows * row / block) {
                let d = 0x1400 + (u64_at(seed + 0x1000_0000, j as u64) % 0x1000) as u16;
                out[j * block + offset..j * block + offset + 2].copy_from_slice(&d.to_le_bytes());
            }
            out
        }
    }
}

/// A permutation of `0..n`: the indices in the order of their stream values.
pub fn permutation(seed: u64, n: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&i| (u64_at(seed, u64::from(i)), i));
    order
}

/// `n` values in `0..bound`, from index 0.
pub fn integers(seed: u64, n: usize, bound: u64) -> Vec<u32> {
    (0..n as u64).map(|i| below(seed, i, bound) as u32).collect()
}
