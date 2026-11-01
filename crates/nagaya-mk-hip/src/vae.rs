//! The ops of the VAE decoder of Qwen-Image 2.1 (`hip/vae.hip`), and its convs as an im2col copy and one GEMM.
//!
//! Each image is channels-last in device memory: one row of channels for each pixel, in row order. The weight of a
//! 3 by 3 conv is `[out, 3, 3, in]`: the torch weight `[out, in, 3, 3]` with its axes moved at the load.
//!
//! # Safety
//!
//! Each function takes device pointers. The caller must give memory that holds each value that the op reads or
//! writes, and must keep it alive until the stream reaches the op.

use std::ffi::{c_int, c_void};

use crate::gemm::{GemmError, Gemms};
use crate::hip::{DeviceBuffer, HipError, Stream, check};
use crate::plan::{DataType, GemmShape};

unsafe extern "C" {
    fn nmk_im2col3x3(
        x: *const c_void,
        channels: c_int,
        height: c_int,
        width: c_int,
        up: c_int,
        row0: c_int,
        rows: c_int,
        out: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn nmk_vae_norm(
        x: *const c_void,
        gamma: *const c_void,
        out: *mut c_void,
        pixels: c_int,
        channels: c_int,
        silu: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn nmk_dup_up_add(
        src: *const c_void,
        src_channels: c_int,
        src_height: c_int,
        src_width: c_int,
        factor_t: c_int,
        dst: *mut c_void,
        dst_channels: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn nmk_channel_affine(
        x: *const c_void,
        scale: *const c_void,
        shift: *const c_void,
        out: *mut c_void,
        n: i64,
        channels: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn nmk_to_bytes(x: *const c_void, out: *mut c_void, n: i64, stream: *mut c_void) -> c_int;
}

fn int(value: usize) -> c_int {
    c_int::try_from(value).expect("a VAE size fits i32")
}

/// The RMS norm of the VAE over the channels of each pixel, then SiLU when `silu` is true.
///
/// # Safety
///
/// See the module.
pub unsafe fn norm(
    x: *const c_void,
    gamma: *const c_void,
    out: *mut c_void,
    pixels: usize,
    channels: usize,
    silu: bool,
    stream: &Stream,
) -> Result<(), HipError> {
    // SAFETY: the caller keeps the rules of the module.
    check("nmk_vae_norm", unsafe {
        nmk_vae_norm(
            x,
            gamma,
            out,
            int(pixels),
            int(channels),
            c_int::from(silu),
            stream.raw(),
        )
    })
}

/// `dst += ` the DupUp3D shortcut of `src`: `src` is `height` by `width` with `src_channels`, and `dst` is twice the
/// size with `dst_channels`. `factor_t` is 2 for a block with a time upsample, else 1.
///
/// # Safety
///
/// See the module.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dup_up_add(
    src: *const c_void,
    src_channels: usize,
    height: usize,
    width: usize,
    factor_t: usize,
    dst: *mut c_void,
    dst_channels: usize,
    stream: &Stream,
) -> Result<(), HipError> {
    // SAFETY: the caller keeps the rules of the module.
    check("nmk_dup_up_add", unsafe {
        nmk_dup_up_add(
            src,
            int(src_channels),
            int(height),
            int(width),
            int(factor_t),
            dst,
            int(dst_channels),
            stream.raw(),
        )
    })
}

/// `out = x * scale + shift` for each channel of `n` values: the latent before the decode.
///
/// # Safety
///
/// See the module.
pub unsafe fn channel_affine(
    x: *const c_void,
    scale: *const c_void,
    shift: *const c_void,
    out: *mut c_void,
    n: usize,
    channels: usize,
    stream: &Stream,
) -> Result<(), HipError> {
    // SAFETY: the caller keeps the rules of the module.
    check("nmk_channel_affine", unsafe {
        nmk_channel_affine(x, scale, shift, out, n as i64, int(channels), stream.raw())
    })
}

/// The 8-bit values of `n` decoded values.
///
/// # Safety
///
/// See the module. `out` holds `n` bytes.
pub unsafe fn to_bytes(x: *const c_void, out: *mut c_void, n: usize, stream: &Stream) -> Result<(), HipError> {
    // SAFETY: the caller keeps the rules of the module.
    check("nmk_to_bytes", unsafe { nmk_to_bytes(x, out, n as i64, stream.raw()) })
}

/// One 3 by 3 conv with a padding of 1, and the bias of each output channel.
#[derive(Clone, Copy, Debug)]
pub struct Conv3x3 {
    /// `[out, 3, 3, in]` in bf16.
    pub weight: *const c_void,
    /// `[out]` in bf16.
    pub bias: *const c_void,
    pub inputs: usize,
    pub outputs: usize,
}

impl Conv3x3 {
    /// `out = conv(x) + bias (+ residual)` for an output of `height` by `width`. The source is `height / up` by
    /// `width / up`, so `up` 2 runs a nearest 2x upsample first. The rows go in bands that fit `scratch`.
    ///
    /// # Safety
    ///
    /// See the module. `out` must not overlap `x`. `residual` can be `out` itself.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run(
        &self,
        gemms: &mut Gemms,
        x: *const c_void,
        height: usize,
        width: usize,
        up: usize,
        residual: Option<*const c_void>,
        out: *mut c_void,
        scratch: &DeviceBuffer,
        stream: &Stream,
    ) -> Result<(), GemmError> {
        let row_bytes = width * 9 * self.inputs * 2;
        let band = (scratch.bytes() / row_bytes).min(height);
        assert!(
            band > 0,
            "the im2col scratch of {} bytes holds no row of {row_bytes}",
            scratch.bytes()
        );
        let mut row0 = 0;
        while row0 < height {
            let rows = band.min(height - row0);
            // SAFETY: the scratch holds `rows` rows of the copy.
            check("nmk_im2col3x3", unsafe {
                nmk_im2col3x3(
                    x,
                    int(self.inputs),
                    int(height),
                    int(width),
                    int(up),
                    int(row0),
                    int(rows),
                    scratch.ptr(),
                    stream.raw(),
                )
            })?;
            let shape = GemmShape::linear(
                (rows * width) as u32,
                (9 * self.inputs) as u32,
                self.outputs as u32,
                DataType::Bf16,
            )
            .with_bias();
            let offset = row0 * width * self.outputs;
            let d = out.cast::<u16>().wrapping_add(offset).cast::<c_void>();
            let (beta, c) = match residual {
                Some(r) => (1.0, r.cast::<u16>().wrapping_add(offset).cast::<c_void>()),
                None => (0.0, d.cast_const()),
            };
            // SAFETY: the scratch holds the copy, and `out` and the residual hold the band.
            unsafe {
                gemms.run_bias(&shape, 1.0, self.weight, scratch.ptr(), beta, c, d, self.bias, stream)?;
            }
            row0 += rows;
        }
        Ok(())
    }
}
