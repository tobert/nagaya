//! The VAE decoder of Qwen-Image 2.1 on gfx1151 (spec 11.9c, "The decode"; milestone 22 step 7).
//!
//! The decoder runs one image of one frame: the image specialization of the Wan decoder, with its time axis folded
//! away. Each activation is channels-last, one row of channels for each pixel. The file holds F32, and the load
//! rounds each weight to bf16 (spec 11.9c, "The load"), because diffusers runs the VAE in bf16.

use std::ffi::c_void;
use std::fmt;
use std::path::Path;

use nagaya_mk_hip::attention::{Attention, AttentionShape, Mask};
use nagaya_mk_hip::bf16;
use nagaya_mk_hip::gemm::{GemmError, Gemms};
use nagaya_mk_hip::hip::{DeviceBuffer, HipError, Stream};
use nagaya_mk_hip::plan::{DataType, GemmShape};
use nagaya_mk_hip::vae::{self, Conv3x3};

use crate::safetensors::{Dtype, Safetensors, SafetensorsError};

/// The channels of the latent.
pub const Z: usize = 64;
/// The scale from a latent pixel to an image pixel.
pub const SCALE: usize = 16;
/// The channels of the image: RGBA.
pub const CHANNELS: usize = 4;
/// The up blocks: (in, out, the upsample, the factor of time of the shortcut).
const UPS: [(usize, usize, bool, usize); 5] = [
    (1152, 1152, true, 2),
    (1152, 1152, true, 2),
    (1152, 576, true, 2),
    (576, 288, true, 1),
    (288, 144, false, 0),
];
const MID: usize = 1152;
const OUT: usize = 144;

/// An error of the decoder.
#[derive(Debug)]
pub enum VaeError {
    File(SafetensorsError),
    Load(String),
    Hip(HipError),
    Gemm(GemmError),
}

impl fmt::Display for VaeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VaeError::File(e) => write!(f, "{e}"),
            VaeError::Load(e) => write!(f, "the VAE: {e}"),
            VaeError::Hip(e) => write!(f, "{e}"),
            VaeError::Gemm(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for VaeError {}

impl From<SafetensorsError> for VaeError {
    fn from(e: SafetensorsError) -> Self {
        VaeError::File(e)
    }
}

impl From<HipError> for VaeError {
    fn from(e: HipError) -> Self {
        VaeError::Hip(e)
    }
}

impl From<GemmError> for VaeError {
    fn from(e: GemmError) -> Self {
        VaeError::Gemm(e)
    }
}

/// A conv in device memory: a 3 by 3 weight `[out, 3, 3, in]`, or a 1 by 1 weight `[out, in]`, and its bias.
struct Conv {
    weight: DeviceBuffer,
    bias: DeviceBuffer,
    inputs: usize,
    outputs: usize,
}

impl Conv {
    fn conv3x3(&self) -> Conv3x3 {
        Conv3x3 {
            weight: self.weight.ptr(),
            bias: self.bias.ptr(),
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }
}

struct Resnet {
    norm1: DeviceBuffer,
    conv1: Conv,
    norm2: DeviceBuffer,
    conv2: Conv,
    shortcut: Option<Conv>,
}

struct Up {
    resnets: Vec<Resnet>,
    upsample: Option<Conv>,
    factor_t: usize,
    inputs: usize,
    outputs: usize,
}

/// The reader of the VAE file, which checks that every tensor is read or skipped by name.
struct Reader<'a> {
    file: &'a Safetensors,
    read: std::cell::RefCell<std::collections::BTreeSet<String>>,
}

impl Reader<'_> {
    fn f32s(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, VaeError> {
        self.read.borrow_mut().insert(name.to_owned());
        Ok(self
            .file
            .read(name, Dtype::F32, shape)?
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
            .collect())
    }

    fn bf16(&self, values: &[f32]) -> Result<DeviceBuffer, VaeError> {
        Ok(DeviceBuffer::from_slice(
            &values.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<u16>>(),
        )?)
    }

    fn vector(&self, name: &str, shape: &[usize]) -> Result<DeviceBuffer, VaeError> {
        let values = self.f32s(name, shape)?;
        self.bf16(&values)
    }

    /// A 3 by 3 conv: the torch weight `[out, in, 3, 3]` moves to `[out, 3, 3, in]`.
    fn conv3x3(&self, base: &str, inputs: usize, outputs: usize) -> Result<Conv, VaeError> {
        let w = self.f32s(&format!("{base}.weight"), &[outputs, inputs, 3, 3])?;
        let mut moved = vec![0.0f32; w.len()];
        for o in 0..outputs {
            for c in 0..inputs {
                for t in 0..9 {
                    moved[(o * 9 + t) * inputs + c] = w[(o * inputs + c) * 9 + t];
                }
            }
        }
        Ok(Conv {
            weight: self.bf16(&moved)?,
            bias: self.vector(&format!("{base}.bias"), &[outputs])?,
            inputs,
            outputs,
        })
    }

    /// A 1 by 1 conv: `[out, in, 1, 1]` is the weight of a linear layer.
    fn conv1x1(&self, base: &str, inputs: usize, outputs: usize) -> Result<Conv, VaeError> {
        Ok(Conv {
            weight: self.vector(&format!("{base}.weight"), &[outputs, inputs, 1, 1])?,
            bias: self.vector(&format!("{base}.bias"), &[outputs])?,
            inputs,
            outputs,
        })
    }

    fn resnet(&self, base: &str, inputs: usize, outputs: usize) -> Result<Resnet, VaeError> {
        Ok(Resnet {
            norm1: self.vector(&format!("{base}.norm1.gamma"), &[inputs, 1, 1, 1])?,
            conv1: self.conv3x3(&format!("{base}.conv1"), inputs, outputs)?,
            norm2: self.vector(&format!("{base}.norm2.gamma"), &[outputs, 1, 1, 1])?,
            conv2: self.conv3x3(&format!("{base}.conv2"), outputs, outputs)?,
            shortcut: if inputs != outputs {
                Some(self.conv1x1(&format!("{base}.conv_shortcut"), inputs, outputs)?)
            } else {
                None
            },
        })
    }
}

/// The VAE decoder in device memory.
pub struct VaeDecoder {
    mean: DeviceBuffer,
    std: DeviceBuffer,
    post_quant: Conv,
    conv_in: Conv,
    mid: [Resnet; 2],
    attn_norm: DeviceBuffer,
    attn_qkv: Conv,
    attn_proj: Conv,
    ups: Vec<Up>,
    norm_out: DeviceBuffer,
    conv_out: Conv,
}

fn bf16_vector(values: &[f64]) -> Result<DeviceBuffer, HipError> {
    // The pipeline makes a tensor of the config's list in f32, then moves it to bf16 (pipeline.py:842-852).
    DeviceBuffer::from_slice(&values.iter().map(|&x| bf16::from_f32(x as f32)).collect::<Vec<u16>>())
}

/// The buffers of a decode: pixels by channels, in bf16.
fn buffer(pixels: usize, channels: usize) -> Result<DeviceBuffer, HipError> {
    DeviceBuffer::new(pixels * channels * 2)
}

impl VaeDecoder {
    /// Load the decoder from the VAE folder: `diffusion_pytorch_model.safetensors` and `config.json`. It loads the
    /// decoder and `post_quant_conv`, and skips the encoder, `quant_conv` and the time convs by name. Any other
    /// tensor refuses the load.
    pub fn load(dir: &Path) -> Result<VaeDecoder, VaeError> {
        let config_path = dir.join("config.json");
        let text = std::fs::read_to_string(&config_path)
            .map_err(|e| VaeError::Load(format!("cannot read {}: {e}", config_path.display())))?;
        let config: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| VaeError::Load(format!("{} is not JSON: {e}", config_path.display())))?;
        let list = |key: &str| -> Result<Vec<f64>, VaeError> {
            let values: Option<Vec<f64>> = config[key]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_f64()).collect());
            match values {
                Some(v) if v.len() == Z => Ok(v),
                _ => Err(VaeError::Load(format!(
                    "{} has no {key} of {Z} numbers",
                    config_path.display()
                ))),
            }
        };
        let file = Safetensors::open(&dir.join("diffusion_pytorch_model.safetensors"))?;
        let r = Reader {
            file: &file,
            read: Default::default(),
        };
        let mut ups = Vec::new();
        for (i, &(inputs, outputs, upsample, factor_t)) in UPS.iter().enumerate() {
            let base = format!("decoder.up_blocks.{i}");
            let resnets = (0..3)
                .map(|j| {
                    r.resnet(
                        &format!("{base}.resnets.{j}"),
                        if j == 0 { inputs } else { outputs },
                        outputs,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            ups.push(Up {
                resnets,
                upsample: if upsample {
                    Some(r.conv3x3(&format!("{base}.upsampler.resample.1"), outputs, outputs)?)
                } else {
                    None
                },
                factor_t,
                inputs,
                outputs,
            });
        }
        let decoder = VaeDecoder {
            mean: bf16_vector(&list("latents_mean")?)?,
            std: bf16_vector(&list("latents_std")?)?,
            post_quant: r.conv1x1("post_quant_conv", Z, Z)?,
            conv_in: r.conv3x3("decoder.conv_in", Z, MID)?,
            mid: [
                r.resnet("decoder.mid_block.resnets.0", MID, MID)?,
                r.resnet("decoder.mid_block.resnets.1", MID, MID)?,
            ],
            attn_norm: r.vector("decoder.mid_block.attentions.0.norm.gamma", &[MID, 1, 1])?,
            attn_qkv: r.conv1x1("decoder.mid_block.attentions.0.to_qkv", MID, 3 * MID)?,
            attn_proj: r.conv1x1("decoder.mid_block.attentions.0.proj", MID, MID)?,
            ups,
            norm_out: r.vector("decoder.norm_out.gamma", &[OUT, 1, 1, 1])?,
            conv_out: r.conv3x3("decoder.conv_out", OUT, CHANNELS)?,
        };
        let read = r.read.borrow();
        for name in file.tensors().keys() {
            let skipped =
                name.starts_with("encoder.") || name.starts_with("quant_conv.") || name.contains(".time_conv.");
            if !skipped && !read.contains(name) {
                return Err(VaeError::Load(format!(
                    "the tensor `{name}` is in no list of the decoder's load"
                )));
            }
        }
        Ok(decoder)
    }

    /// The scratch of the im2col copy of one conv.
    fn scratch() -> Result<DeviceBuffer, HipError> {
        DeviceBuffer::new(256 << 20)
    }

    /// `out = x W^T + bias (+ residual)` for `pixels` rows: a 1 by 1 conv.
    #[allow(clippy::too_many_arguments)]
    unsafe fn conv1x1(
        gemms: &mut Gemms,
        conv: &Conv,
        x: *const c_void,
        pixels: usize,
        residual: Option<*const c_void>,
        out: *mut c_void,
        stream: &Stream,
    ) -> Result<(), VaeError> {
        let shape =
            GemmShape::linear(pixels as u32, conv.inputs as u32, conv.outputs as u32, DataType::Bf16).with_bias();
        let (beta, c) = match residual {
            Some(r) => (1.0, r),
            None => (0.0, out.cast_const()),
        };
        // SAFETY: the caller gives `pixels` rows of the conv's channels.
        unsafe { gemms.run_bias(&shape, 1.0, conv.weight.ptr(), x, beta, c, out, conv.bias.ptr(), stream)? };
        Ok(())
    }

    /// One residual block on an `h` by `w` image: `conv2(silu(norm2(conv1(silu(norm1(x)))))) + shortcut(x)`.
    #[allow(clippy::too_many_arguments)]
    fn resnet(
        &self,
        gemms: &mut Gemms,
        block: &Resnet,
        x: &DeviceBuffer,
        inputs: usize,
        outputs: usize,
        h: usize,
        w: usize,
        scratch: &DeviceBuffer,
        stream: &Stream,
    ) -> Result<DeviceBuffer, VaeError> {
        let pixels = h * w;
        let out = buffer(pixels, outputs)?;
        let normed = buffer(pixels, inputs)?;
        let mid = buffer(pixels, outputs)?;
        // SAFETY: each buffer holds `pixels` rows of its channels.
        unsafe {
            match &block.shortcut {
                Some(conv) => VaeDecoder::conv1x1(gemms, conv, x.ptr(), pixels, None, out.ptr(), stream)?,
                None => out.copy_from(x, pixels * outputs * 2)?,
            }
            vae::norm(x.ptr(), block.norm1.ptr(), normed.ptr(), pixels, inputs, true, stream)?;
            block
                .conv1
                .conv3x3()
                .run(gemms, normed.ptr(), h, w, 1, None, mid.ptr(), scratch, stream)?;
            vae::norm(
                mid.ptr(),
                block.norm2.ptr(),
                normed.ptr(),
                pixels,
                outputs,
                true,
                stream,
            )?;
            let residual = Some(out.ptr().cast_const());
            block
                .conv2
                .conv3x3()
                .run(gemms, normed.ptr(), h, w, 1, residual, out.ptr(), scratch, stream)?;
        }
        stream.synchronize()?;
        Ok(out)
    }

    /// The attention block of the mid block: one head over the pixels, with the channels as its width.
    fn attention(
        &self,
        gemms: &mut Gemms,
        x: &DeviceBuffer,
        pixels: usize,
        stream: &Stream,
    ) -> Result<DeviceBuffer, VaeError> {
        let normed = buffer(pixels, MID)?;
        let qkv = buffer(pixels, 3 * MID)?;
        let mixed = buffer(pixels, MID)?;
        let out = buffer(pixels, MID)?;
        let attention = Attention::new(pixels * pixels)?;
        let shape = AttentionShape {
            lq: pixels,
            lk: pixels,
            heads: 1,
            kv_heads: 1,
            head_dim: MID,
            q_stride: 3 * MID,
            k_stride: 3 * MID,
            v_stride: 3 * MID,
            out_stride: MID,
            mask: Mask::None,
        };
        let at = |e: usize| qkv.ptr().cast::<u16>().wrapping_add(e).cast::<c_void>();
        // SAFETY: each buffer holds `pixels` rows of its channels.
        unsafe {
            vae::norm(x.ptr(), self.attn_norm.ptr(), normed.ptr(), pixels, MID, false, stream)?;
            VaeDecoder::conv1x1(gemms, &self.attn_qkv, normed.ptr(), pixels, None, qkv.ptr(), stream)?;
            attention.run(gemms, &shape, at(0), at(MID), at(2 * MID), mixed.ptr(), stream)?;
            let residual = Some(x.ptr().cast_const());
            VaeDecoder::conv1x1(gemms, &self.attn_proj, mixed.ptr(), pixels, residual, out.ptr(), stream)?;
        }
        stream.synchronize()?;
        Ok(out)
    }

    /// Decode one latent `[h * w, 64]` (token rows, as the transformer gives them) into RGBA bytes of a `16 h` by
    /// `16 w` image, in row order.
    pub fn decode(
        &self,
        gemms: &mut Gemms,
        latent: &DeviceBuffer,
        h: usize,
        w: usize,
        stream: &Stream,
    ) -> Result<Vec<u8>, VaeError> {
        let scratch = VaeDecoder::scratch()?;
        let pixels = h * w;
        assert!(latent.bytes() >= pixels * Z * 2, "the latent holds {pixels} tokens");
        let latent = latent.ptr().cast_const();
        let scaled = buffer(pixels, Z)?;
        let quant = buffer(pixels, Z)?;
        let mut x = buffer(pixels, MID)?;
        // SAFETY: the latent holds `pixels` rows of 64; each buffer holds its rows.
        unsafe {
            vae::channel_affine(
                latent,
                self.std.ptr(),
                self.mean.ptr(),
                scaled.ptr(),
                pixels * Z,
                Z,
                stream,
            )?;
            VaeDecoder::conv1x1(gemms, &self.post_quant, scaled.ptr(), pixels, None, quant.ptr(), stream)?;
            self.conv_in
                .conv3x3()
                .run(gemms, quant.ptr(), h, w, 1, None, x.ptr(), &scratch, stream)?;
        }
        x = self.resnet(gemms, &self.mid[0], &x, MID, MID, h, w, &scratch, stream)?;
        x = self.attention(gemms, &x, pixels, stream)?;
        x = self.resnet(gemms, &self.mid[1], &x, MID, MID, h, w, &scratch, stream)?;
        let (mut h, mut w) = (h, w);
        for up in &self.ups {
            let copy = x;
            x = self.resnet(
                gemms,
                &up.resnets[0],
                &copy,
                up.inputs,
                up.outputs,
                h,
                w,
                &scratch,
                stream,
            )?;
            for block in &up.resnets[1..] {
                x = self.resnet(gemms, block, &x, up.outputs, up.outputs, h, w, &scratch, stream)?;
            }
            if let Some(conv) = &up.upsample {
                let big = buffer(4 * h * w, up.outputs)?;
                // SAFETY: `x` holds the small image, and `big` the image of twice the size.
                unsafe {
                    conv.conv3x3()
                        .run(gemms, x.ptr(), 2 * h, 2 * w, 2, None, big.ptr(), &scratch, stream)?;
                    vae::dup_up_add(copy.ptr(), up.inputs, h, w, up.factor_t, big.ptr(), up.outputs, stream)?;
                }
                stream.synchronize()?;
                x = big;
                (h, w) = (2 * h, 2 * w);
            }
        }
        let pixels = h * w;
        let normed = buffer(pixels, OUT)?;
        let rgba = buffer(pixels, CHANNELS)?;
        let bytes = DeviceBuffer::new(pixels * CHANNELS)?;
        // SAFETY: each buffer holds `pixels` rows of its channels.
        unsafe {
            vae::norm(x.ptr(), self.norm_out.ptr(), normed.ptr(), pixels, OUT, true, stream)?;
            self.conv_out
                .conv3x3()
                .run(gemms, normed.ptr(), h, w, 1, None, rgba.ptr(), &scratch, stream)?;
            vae::to_bytes(rgba.ptr(), bytes.ptr(), pixels * CHANNELS, stream)?;
        }
        stream.synchronize()?;
        Ok(bytes.to_vec::<u8>()?)
    }
}
