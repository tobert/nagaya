//! Attention on gfx1151: the scores and the output are hipBLASLt GEMMs, and `hip/attention.hip` computes the softmax
//! between them.
//!
//! The scores stay in f32 until the softmax. The probabilities go to bf16 for the GEMM with the values, which sums
//! in f32. A pass takes the heads of one key group in chunks, so the scratch of the scores has a fixed size.

use std::ffi::{c_int, c_void};

use crate::gemm::{GemmError, Gemms};
use crate::hip::{DeviceBuffer, Stream, check};
use crate::plan::{DataType, GemmShape};

unsafe extern "C" {
    fn nmk_flash(
        q: *const c_void,
        q_stride: i64,
        k: *const c_void,
        k_stride: i64,
        v: *const c_void,
        v_stride: i64,
        out: *mut c_void,
        out_stride: i64,
        lq: c_int,
        lk: c_int,
        heads: c_int,
        group: c_int,
        scale: f32,
        causal: c_int,
        causal_offset: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn nmk_softmax(
        scores: *const f32,
        probabilities: *mut c_void,
        matrices: c_int,
        lq: c_int,
        lk: c_int,
        causal: c_int,
        causal_offset: c_int,
        stream: *mut c_void,
    ) -> c_int;
}

/// The keys that a query sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mask {
    /// Each query sees each key.
    None,
    /// Query `q` sees the keys `0..=q + offset`.
    Causal { offset: usize },
}

/// One attention call: `lq` queries over `lk` keys, for `heads` query heads that share `kv_heads` key heads. Each
/// tensor is bf16 and row-major, with a row stride in elements. Head `h` of a row starts at `h * head_dim`.
#[derive(Clone, Copy, Debug)]
pub struct AttentionShape {
    pub lq: usize,
    pub lk: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub q_stride: usize,
    pub k_stride: usize,
    pub v_stride: usize,
    pub out_stride: usize,
    pub mask: Mask,
}

/// The scratch of the scores and the probabilities.
pub struct Attention {
    scores: DeviceBuffer,
    probabilities: DeviceBuffer,
    capacity: usize,
}

fn u32_of(value: usize) -> u32 {
    u32::try_from(value).expect("an attention size fits u32")
}

impl Attention {
    /// Scratch for `capacity` scores: at least `lq * lk` of the largest call. A pass holds as many heads as fit.
    pub fn new(capacity: usize) -> Result<Attention, GemmError> {
        Ok(Attention {
            scores: DeviceBuffer::new(capacity * 4)?,
            probabilities: DeviceBuffer::new(capacity * 2)?,
            capacity,
        })
    }

    /// The bytes of the scratch.
    pub fn bytes(&self) -> usize {
        self.scores.bytes() + self.probabilities.bytes()
    }

    /// `out = softmax(q k^T / sqrt(head_dim)) v` for each head, on `stream`.
    ///
    /// # Safety
    ///
    /// `q`, `k`, `v` and `out` must be device memory with the rows of the shape at its strides. `out` must not
    /// overlap the inputs.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run(
        &self,
        gemms: &mut Gemms,
        s: &AttentionShape,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        stream: &Stream,
    ) -> Result<(), GemmError> {
        assert!(
            s.kv_heads > 0 && s.heads.is_multiple_of(s.kv_heads),
            "{s:?}: the heads must split into the key groups"
        );
        assert!(s.lq > 0 && s.lk > 0, "{s:?}: an attention call needs a query and a key");
        let matrix = s.lq * s.lk;
        assert!(
            matrix <= self.capacity,
            "{s:?}: one head needs {matrix} scores, the scratch holds {}",
            self.capacity
        );
        let group = s.heads / s.kv_heads;
        let chunk = group.min(self.capacity / matrix);
        let d = s.head_dim;
        let scale = 1.0 / (d as f32).sqrt();
        let (causal, offset) = match s.mask {
            Mask::None => (0, 0),
            Mask::Causal { offset } => (1, offset),
        };
        let at = |ptr: *const c_void, elements: usize| ptr.cast::<u16>().wrapping_add(elements).cast::<c_void>();
        for g in 0..s.kv_heads {
            let mut first = 0;
            while first < group {
                let n = chunk.min(group - first);
                let h0 = g * group + first;
                let scores = GemmShape {
                    m: u32_of(s.lk),
                    n: u32_of(s.lq),
                    k: u32_of(d),
                    trans_a: true,
                    trans_b: false,
                    lda: s.k_stride as u64,
                    ldb: s.q_stride as u64,
                    ldc: s.lk as u64,
                    batch: u32_of(n),
                    stride_a: 0,
                    stride_b: if n > 1 { d as u64 } else { 0 },
                    stride_c: if n > 1 { matrix as u64 } else { 0 },
                    type_ab: DataType::Bf16,
                    type_cd: DataType::F32,
                    bias: false,
                };
                // SAFETY: the caller gives the rows of the shape; the scratch holds `n` score matrices.
                unsafe {
                    gemms.run(
                        &scores,
                        scale,
                        at(k, g * d),
                        at(q, h0 * d),
                        0.0,
                        self.scores.ptr(),
                        self.scores.ptr(),
                        stream,
                    )?;
                    check(
                        "nmk_softmax",
                        nmk_softmax(
                            self.scores.ptr().cast(),
                            self.probabilities.ptr(),
                            n as c_int,
                            s.lq as c_int,
                            s.lk as c_int,
                            causal,
                            offset as c_int,
                            stream.raw(),
                        ),
                    )?;
                }
                let values = GemmShape {
                    m: u32_of(d),
                    n: u32_of(s.lq),
                    k: u32_of(s.lk),
                    trans_a: false,
                    trans_b: false,
                    lda: s.v_stride as u64,
                    ldb: s.lk as u64,
                    ldc: s.out_stride as u64,
                    batch: u32_of(n),
                    stride_a: 0,
                    stride_b: if n > 1 { matrix as u64 } else { 0 },
                    stride_c: if n > 1 { d as u64 } else { 0 },
                    type_ab: DataType::Bf16,
                    type_cd: DataType::Bf16,
                    bias: false,
                };
                let o = at(out, h0 * d).cast_mut();
                // SAFETY: the caller gives the rows of the shape; the probabilities hold `n` matrices.
                unsafe {
                    gemms.run(&values, 1.0, at(v, g * d), self.probabilities.ptr(), 0.0, o, o, stream)?;
                }
                first += n;
            }
        }
        Ok(())
    }
}

/// The head size of `flash`.
pub const FLASH_HEAD_DIM: usize = 128;

/// `out = softmax(q k^T / sqrt(d)) v` for each head in one fused kernel (`hip/flash.hip`): the scores stay in
/// registers, so no scratch holds them. The head size must be 128. The probabilities round to bf16 before the values,
/// as `Attention::run` rounds them.
///
/// # Safety
///
/// As for [`Attention::run`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash(
    s: &AttentionShape,
    q: *const c_void,
    k: *const c_void,
    v: *const c_void,
    out: *mut c_void,
    stream: &Stream,
) -> Result<(), crate::hip::HipError> {
    assert_eq!(
        s.head_dim, FLASH_HEAD_DIM,
        "{s:?}: the flash kernel takes heads of {FLASH_HEAD_DIM}"
    );
    assert!(
        s.kv_heads > 0 && s.heads.is_multiple_of(s.kv_heads),
        "{s:?}: the heads must split into the key groups"
    );
    let (causal, offset) = match s.mask {
        Mask::None => (0, 0),
        Mask::Causal { offset } => (1, offset),
    };
    let int = |x: usize| c_int::try_from(x).expect("an attention size fits i32");
    // SAFETY: the caller keeps the rules of `Attention::run`.
    check("nmk_flash", unsafe {
        nmk_flash(
            q,
            s.q_stride as i64,
            k,
            s.k_stride as i64,
            v,
            s.v_stride as i64,
            out,
            s.out_stride as i64,
            int(s.lq),
            int(s.lk),
            int(s.heads),
            int(s.heads / s.kv_heads),
            1.0 / (s.head_dim as f32).sqrt(),
            causal,
            int(offset),
            stream.raw(),
        )
    })
}
