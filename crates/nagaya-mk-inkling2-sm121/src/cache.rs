//! The host image of the cache of one run, and the pack of its rows.
//!
//! The layout is in `nagaya_mk_cuda::attn`: a plane for the keys and a plane for the values, a block of rows for each KV
//! head, and one row of 128 values for each position. A row is `f32` (the oracle), `bf16` or `e4m3`, the last two with a unit scale. A scale
//! never covers more than one position, so the row of a new token is complete when its projection writes it.
//!
//! The device writes its rows in the epilogue of the projection. This image is the host form of that write. The tests of
//! the attention use it to fill a cache, and a later pass uses it to check the cache of a run.

use crate::quant::{FormatError, decode_e4m3};
use nagaya_mk_cuda::attn::{CacheFormat, HEAD_DIM, KV_HEADS, plane_bytes, row_of, row_offset};

/// The positive finite e4m3 values, by code 0 to 126. The code 127 is NaN.
fn positive_values() -> &'static [f32; 127] {
    static TABLE: std::sync::OnceLock<[f32; 127]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|code| decode_e4m3(code as u8).expect("codes 0 to 126 are finite")))
}

/// The e4m3 code of a finite `f32`: the nearest value, ties to the even code, clamped to 448. It gives the code of
/// `quant::encode_e4m3`, with a binary search in place of a scan, so that a cache of millions of values packs fast.
pub fn pack_e4m3(value: f32) -> Result<u8, FormatError> {
    if !value.is_finite() {
        return Err(FormatError::NonFinite);
    }
    let sign: u8 = if value.is_sign_negative() { 0x80 } else { 0 };
    let table = positive_values();
    let mag = value.abs();
    if mag >= table[126] {
        return Ok(sign | 126);
    }
    // table[i - 1] < mag <= table[i]
    let i = table.partition_point(|t| *t < mag);
    if i == 0 {
        return Ok(sign);
    }
    let (below, above) = (mag - table[i - 1], table[i] - mag);
    let code = if below < above || (below == above && (i - 1) % 2 == 0) {
        i - 1
    } else {
        i
    };
    Ok(sign | code as u8)
}

/// The bf16 value of a finite `f32`: the nearest value, ties to the even code. It is the rounding of the device
/// (`cvt.rn.bf16.f32`). A value that is too large for bf16 is an error, as the device would write an infinity.
pub fn pack_bf16(value: f32) -> Result<u16, FormatError> {
    if !value.is_finite() {
        return Err(FormatError::NonFinite);
    }
    let bits = value.to_bits();
    // Add the half of the dropped bits, and one more when the kept lowest bit is 1: ties go to the even code.
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    let code = (rounded >> 16) as u16;
    if !f32::from_bits(u32::from(code) << 16).is_finite() {
        return Err(FormatError::NonFinite);
    }
    Ok(code)
}

/// Pack one row of 128 values in `format`, and write it to `out` (`format.row_bytes()` bytes).
pub fn pack_row(format: CacheFormat, row: &[f32], out: &mut [u8]) -> Result<(), FormatError> {
    assert_eq!(row.len(), HEAD_DIM, "a cache row has {HEAD_DIM} values");
    assert_eq!(out.len(), format.row_bytes(), "the output is not one row of the format");
    match format {
        CacheFormat::F32 => {
            for (v, o) in row.iter().zip(out.as_chunks_mut::<4>().0) {
                if !v.is_finite() {
                    return Err(FormatError::NonFinite);
                }
                *o = v.to_le_bytes();
            }
        }
        CacheFormat::E4m3 => {
            for (v, o) in row.iter().zip(out.iter_mut()) {
                *o = pack_e4m3(*v)?;
            }
        }
        CacheFormat::Bf16 => {
            for (v, o) in row.iter().zip(out.as_chunks_mut::<2>().0) {
                *o = pack_bf16(*v)?.to_le_bytes();
            }
        }
    }
    Ok(())
}

/// Unpack one row to `f32`. The result is exact for each format.
pub fn unpack_row(format: CacheFormat, bytes: &[u8]) -> Result<Vec<f32>, FormatError> {
    assert_eq!(
        bytes.len(),
        format.row_bytes(),
        "the input is not one row of the format"
    );
    match format {
        CacheFormat::F32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()),
        CacheFormat::E4m3 => bytes.iter().map(|b| decode_e4m3(*b)).collect(),
        CacheFormat::Bf16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect()),
    }
}

/// The host image of the two planes of one run.
pub struct RunCache {
    pub format: CacheFormat,
    /// 0 for a full run, or the window of a sliding run.
    pub window: usize,
    pub plane_rows: usize,
    pub keys: Vec<u8>,
    pub values: Vec<u8>,
}

impl RunCache {
    /// An image with every byte zero, so that a row that nothing wrote reads as the value 0.
    pub fn new(format: CacheFormat, window: usize, plane_rows: usize) -> Self {
        let bytes = plane_bytes(plane_rows, format);
        Self {
            format,
            window,
            plane_rows,
            keys: vec![0; bytes],
            values: vec![0; bytes],
        }
    }

    /// Write the key row and the value row of KV head `kvh` at `position`. A sliding run writes at the position modulo
    /// the ring, and it overwrites the position that is one ring older.
    pub fn write(&mut self, position: usize, kvh: usize, key: &[f32], value: &[f32]) -> Result<(), FormatError> {
        assert!(kvh < KV_HEADS, "KV head {kvh} of {KV_HEADS}");
        let at = row_offset(
            kvh,
            row_of(position, self.window, self.plane_rows),
            self.plane_rows,
            self.format,
        );
        let n = self.format.row_bytes();
        pack_row(self.format, key, &mut self.keys[at..at + n])?;
        pack_row(self.format, value, &mut self.values[at..at + n])
    }

    /// The key row and the value row of KV head `kvh` that the cache holds at `position`, as `f32`.
    pub fn read(&self, position: usize, kvh: usize) -> Result<(Vec<f32>, Vec<f32>), FormatError> {
        let at = row_offset(
            kvh,
            row_of(position, self.window, self.plane_rows),
            self.plane_rows,
            self.format,
        );
        let n = self.format.row_bytes();
        Ok((
            unpack_row(self.format, &self.keys[at..at + n])?,
            unpack_row(self.format, &self.values[at..at + n])?,
        ))
    }

    /// The bytes that a decode step reads at `position`, for each plane pair: the rows of the keys and the values of
    /// the 8 KV heads in the visible range.
    pub fn bytes_read(&self, position: usize) -> usize {
        let lowest = nagaya_mk_cuda::attn::lowest_key(position, self.window);
        2 * (position + 1 - lowest) * KV_HEADS * self.format.row_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::encode_e4m3;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    #[test]
    fn a_bf16_value_is_the_f32_value_rounded_to_nearest_even() {
        // The bits of f32 1.0 are 0x3F80_0000. Half a step above is a tie: the even code 0x3F80 wins.
        assert_eq!(pack_bf16(1.0), Ok(0x3F80));
        assert_eq!(pack_bf16(f32::from_bits(0x3F80_8000)), Ok(0x3F80));
        // A tie above an odd code rounds up to the even code.
        assert_eq!(pack_bf16(f32::from_bits(0x3F81_8000)), Ok(0x3F82));
        // Just under and just over the tie.
        assert_eq!(pack_bf16(f32::from_bits(0x3F80_7FFF)), Ok(0x3F80));
        assert_eq!(pack_bf16(f32::from_bits(0x3F80_8001)), Ok(0x3F81));
        assert_eq!(pack_bf16(-0.0), Ok(0x8000));
        assert_eq!(pack_bf16(f32::MAX), Err(FormatError::NonFinite));
        assert_eq!(pack_bf16(f32::NAN), Err(FormatError::NonFinite));
        // A bf16 row round-trips exactly and takes 2 bytes for each value.
        let row: Vec<f32> = (0..HEAD_DIM).map(|i| (i as f32 - 60.0) * 0.37).collect();
        let mut bytes = vec![0u8; CacheFormat::Bf16.row_bytes()];
        pack_row(CacheFormat::Bf16, &row, &mut bytes).unwrap();
        assert_eq!(bytes.len(), 256);
        let back = unpack_row(CacheFormat::Bf16, &bytes).unwrap();
        for (v, b) in row.iter().zip(&back) {
            assert_eq!(*b, f32::from_bits(u32::from(pack_bf16(*v).unwrap()) << 16));
            assert!((b - v).abs() <= v.abs() * 2f32.powi(-8), "{v} -> {b}");
        }
        let again = pack_row_bytes(&back);
        assert_eq!(again, bytes, "an unpacked row packs to the same bytes");
    }

    fn pack_row_bytes(row: &[f32]) -> Vec<u8> {
        let mut bytes = vec![0u8; CacheFormat::Bf16.row_bytes()];
        pack_row(CacheFormat::Bf16, row, &mut bytes).unwrap();
        bytes
    }

    #[test]
    fn the_fast_pack_gives_the_code_of_the_reference_encoder() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut values: Vec<f32> = (0..20_000).map(|_| rng.unit() * 600.0).collect();
        values.extend((0..20_000).map(|_| rng.unit() * 0.01));
        // The midpoints between neighbors decide the ties, and the codes at the ends decide the clamp.
        let t = positive_values();
        values.extend(t.windows(2).map(|w| (w[0] + w[1]) / 2.0));
        values.extend(t.iter().flat_map(|v| [*v, -*v]));
        values.extend([0.0, -0.0, 448.0, 1.0e9, -1.0e9, 2f32.powi(-10), 2f32.powi(-9) * 1.5]);
        for v in values {
            assert_eq!(pack_e4m3(v), encode_e4m3(v), "value {v:e}");
        }
        assert_eq!(pack_e4m3(f32::NAN), Err(FormatError::NonFinite));
    }

    #[test]
    fn an_e4m3_row_round_trips_through_its_own_values() {
        let mut rng = Rng(7);
        let row: Vec<f32> = (0..HEAD_DIM).map(|_| rng.unit() * 4.0).collect();
        let mut bytes = vec![0u8; HEAD_DIM];
        pack_row(CacheFormat::E4m3, &row, &mut bytes).unwrap();
        let once = unpack_row(CacheFormat::E4m3, &bytes).unwrap();
        let mut again = vec![0u8; HEAD_DIM];
        pack_row(CacheFormat::E4m3, &once, &mut again).unwrap();
        assert_eq!(bytes, again);
        // A value in the normal range is within 1/16 of itself: 3 bits of mantissa.
        for (v, u) in row.iter().zip(&once) {
            assert!((v - u).abs() <= v.abs() / 16.0 + 2f32.powi(-10), "{v} {u}");
        }
    }

    #[test]
    fn an_f32_row_is_exact_and_a_non_finite_value_is_an_error() {
        let row: Vec<f32> = (0..HEAD_DIM).map(|i| i as f32 * 0.37 - 20.0).collect();
        let mut bytes = vec![0u8; 512];
        pack_row(CacheFormat::F32, &row, &mut bytes).unwrap();
        assert_eq!(unpack_row(CacheFormat::F32, &bytes).unwrap(), row);
        let mut bad = row.clone();
        bad[5] = f32::INFINITY;
        assert_eq!(
            pack_row(CacheFormat::F32, &bad, &mut bytes),
            Err(FormatError::NonFinite)
        );
    }

    #[test]
    fn a_sliding_run_writes_a_position_over_the_position_one_ring_older() {
        let mut cache = RunCache::new(CacheFormat::F32, 1024, 1536);
        let (k0, v0) = (vec![1.0; HEAD_DIM], vec![2.0; HEAD_DIM]);
        let (k1, v1) = (vec![3.0; HEAD_DIM], vec![4.0; HEAD_DIM]);
        cache.write(10, 3, &k0, &v0).unwrap();
        assert_eq!(cache.read(10, 3).unwrap(), (k0.clone(), v0.clone()));
        cache.write(10 + 1536, 3, &k1, &v1).unwrap();
        assert_eq!(cache.read(10, 3).unwrap(), (k1.clone(), v1.clone()));
        // The row of position 10 is not the row of position 11, and another head is not touched.
        assert_eq!(cache.read(11, 3).unwrap().0, vec![0.0; HEAD_DIM]);
        assert_eq!(cache.read(10, 2).unwrap().0, vec![0.0; HEAD_DIM]);
    }

    #[test]
    fn a_full_run_keeps_each_position_in_its_own_row() {
        let mut cache = RunCache::new(CacheFormat::E4m3, 0, 128);
        for p in [0usize, 1, 127] {
            cache
                .write(p, 0, &vec![p as f32; HEAD_DIM], &vec![-(p as f32); HEAD_DIM])
                .unwrap();
        }
        assert_eq!(cache.read(127, 0).unwrap().0[0], 128.0);
        assert_eq!(cache.read(1, 0).unwrap().1[0], -1.0);
        assert_eq!(cache.keys.len(), 8 * 128 * 128);
        assert_eq!(cache.bytes_read(100), 2 * 101 * 8 * 128);
    }
}
