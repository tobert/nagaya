//! The row and element ops of `hip/ops.hip`. Each tensor is bf16 in device memory, row-major, with a row stride in
//! elements. Each op runs on a stream and returns when it is queued.
//!
//! # Safety
//!
//! Each function takes device pointers. The caller must give memory that holds each row that the op reads or writes,
//! at the given strides, and must keep it alive until the stream reaches the op.

use std::ffi::{c_int, c_void};

use crate::hip::{HipError, Stream, check};

unsafe extern "C" {
    fn nmk_rmsnorm(
        x: *const c_void,
        x_stride: i64,
        w: *const c_void,
        offset: f32,
        out: *mut c_void,
        out_stride: i64,
        rows: c_int,
        dim: c_int,
        eps: f32,
        stream: *mut c_void,
    ) -> c_int;
    fn nmk_layernorm_modulate(
        x: *const c_void,
        x_stride: i64,
        scale: *const c_void,
        out: *mut c_void,
        out_stride: i64,
        rows: c_int,
        dim: c_int,
        eps: f32,
        stream: *mut c_void,
    ) -> c_int;
    fn nmk_gate_add(
        h: *mut c_void,
        h_stride: i64,
        y: *const c_void,
        y_stride: i64,
        gate: *const c_void,
        rows: c_int,
        dim: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn nmk_silu_mul(
        x: *const c_void,
        x_stride: i64,
        half: c_int,
        out: *mut c_void,
        out_stride: i64,
        rows: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn nmk_gelu_tanh(x: *mut c_void, stride: i64, rows: c_int, dim: c_int, stream: *mut c_void) -> c_int;
    fn nmk_silu(x: *mut c_void, n: i64, stream: *mut c_void) -> c_int;
    fn nmk_head_norm_rope(
        x: *mut c_void,
        stride: i64,
        rows: c_int,
        heads: c_int,
        head_dim: c_int,
        w: *const c_void,
        eps: f32,
        cs: *const f32,
        pairs: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn nmk_gather_rows(
        table: *const c_void,
        ids: *const i32,
        out: *mut c_void,
        rows: c_int,
        dim: c_int,
        stream: *mut c_void,
    ) -> c_int;
}

fn int(value: usize) -> c_int {
    c_int::try_from(value).expect("an op size fits i32")
}

fn stride(value: usize) -> i64 {
    i64::try_from(value).expect("a stride fits i64")
}

/// `out = x / sqrt(mean(x^2) + eps) * (w + offset)` for each row of `dim` values. `offset` is 0 for the RMSNorm of
/// Qwen3, and 1 for the zero-centred RMSNorm of the transformer's text input.
///
/// # Safety
///
/// See the module.
#[allow(clippy::too_many_arguments)]
pub unsafe fn rmsnorm(
    x: *const c_void,
    x_stride: usize,
    w: *const c_void,
    offset: f32,
    out: *mut c_void,
    out_stride: usize,
    rows: usize,
    dim: usize,
    eps: f32,
    stream: &Stream,
) -> Result<(), HipError> {
    // SAFETY: the caller keeps the rules of the module.
    check("nmk_rmsnorm", unsafe {
        nmk_rmsnorm(
            x,
            stride(x_stride),
            w,
            offset,
            out,
            stride(out_stride),
            int(rows),
            int(dim),
            eps,
            stream.raw(),
        )
    })
}

/// `out = layer_norm(x) * (1 + scale)` for each row, with no affine weight. `scale` holds `dim` values for every row.
///
/// # Safety
///
/// See the module.
#[allow(clippy::too_many_arguments)]
pub unsafe fn layernorm_modulate(
    x: *const c_void,
    x_stride: usize,
    scale: *const c_void,
    out: *mut c_void,
    out_stride: usize,
    rows: usize,
    dim: usize,
    eps: f32,
    stream: &Stream,
) -> Result<(), HipError> {
    // SAFETY: the caller keeps the rules of the module.
    check("nmk_layernorm_modulate", unsafe {
        nmk_layernorm_modulate(
            x,
            stride(x_stride),
            scale,
            out,
            stride(out_stride),
            int(rows),
            int(dim),
            eps,
            stream.raw(),
        )
    })
}

/// `h = h + tanh(gate) * y` for each row: the gated residual of a transformer block. `gate` holds `dim` values.
///
/// # Safety
///
/// See the module.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gate_add(
    h: *mut c_void,
    h_stride: usize,
    y: *const c_void,
    y_stride: usize,
    gate: *const c_void,
    rows: usize,
    dim: usize,
    stream: &Stream,
) -> Result<(), HipError> {
    // SAFETY: the caller keeps the rules of the module.
    check("nmk_gate_add", unsafe {
        nmk_gate_add(
            h,
            stride(h_stride),
            y,
            stride(y_stride),
            gate,
            int(rows),
            int(dim),
            stream.raw(),
        )
    })
}

/// `out[r, j] = silu(x[r, j]) * x[r, half + j]`: the SwiGLU of a gate and an up projection in one row.
///
/// # Safety
///
/// See the module.
pub unsafe fn silu_mul(
    x: *const c_void,
    x_stride: usize,
    half: usize,
    out: *mut c_void,
    out_stride: usize,
    rows: usize,
    stream: &Stream,
) -> Result<(), HipError> {
    // SAFETY: the caller keeps the rules of the module.
    check("nmk_silu_mul", unsafe {
        nmk_silu_mul(
            x,
            stride(x_stride),
            int(half),
            out,
            stride(out_stride),
            int(rows),
            stream.raw(),
        )
    })
}

/// `x = gelu(x)` with the tanh form, in place.
///
/// # Safety
///
/// See the module.
pub unsafe fn gelu_tanh(
    x: *mut c_void,
    x_stride: usize,
    rows: usize,
    dim: usize,
    stream: &Stream,
) -> Result<(), HipError> {
    // SAFETY: the caller keeps the rules of the module.
    check("nmk_gelu_tanh", unsafe {
        nmk_gelu_tanh(x, stride(x_stride), int(rows), int(dim), stream.raw())
    })
}

/// `x = silu(x)` for `n` packed values, in place.
///
/// # Safety
///
/// See the module.
pub unsafe fn silu(x: *mut c_void, n: usize, stream: &Stream) -> Result<(), HipError> {
    // SAFETY: the caller keeps the rules of the module.
    check("nmk_silu", unsafe { nmk_silu(x, stride(n), stream.raw()) })
}

/// The pairs of a rotation: `(2i, 2i + 1)` for the transformer, or `(i, i + head_dim / 2)` for the encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopePairs {
    Adjacent,
    Halves,
}

/// For each row and each of `heads` heads, in place: RMSNorm over the head with weight `w`, then the rotation of each
/// pair by the table `cs` of `rope.rs`. `w` can be null for a norm with no weight.
///
/// # Safety
///
/// See the module. `cs` must be device memory with `rows * head_dim` f32 values.
#[allow(clippy::too_many_arguments)]
pub unsafe fn head_norm_rope(
    x: *mut c_void,
    x_stride: usize,
    rows: usize,
    heads: usize,
    head_dim: usize,
    w: *const c_void,
    eps: f32,
    cs: *const c_void,
    pairs: RopePairs,
    stream: &Stream,
) -> Result<(), HipError> {
    // SAFETY: the caller keeps the rules of the module.
    check("nmk_head_norm_rope", unsafe {
        nmk_head_norm_rope(
            x,
            stride(x_stride),
            int(rows),
            int(heads),
            int(head_dim),
            w,
            eps,
            cs.cast(),
            c_int::from(pairs == RopePairs::Adjacent),
            stream.raw(),
        )
    })
}

/// `out[r] = table[ids[r]]` for `rows` ids: the rows of an embedding table of `dim` values.
///
/// # Safety
///
/// See the module. `ids` must be device memory with `rows` i32 ids, each a row of the table.
pub unsafe fn gather_rows(
    table: *const c_void,
    ids: *const c_void,
    out: *mut c_void,
    rows: usize,
    dim: usize,
    stream: &Stream,
) -> Result<(), HipError> {
    // SAFETY: the caller keeps the rules of the module.
    check("nmk_gather_rows", unsafe {
        nmk_gather_rows(table, ids.cast(), out, int(rows), int(dim), stream.raw())
    })
}
