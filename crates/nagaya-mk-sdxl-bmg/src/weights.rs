//! The weights of the engine: the four files of a checkpoint in the folder layout of diffusers, and their upload.
//!
//! Each tensor keeps its dtype: F16 for the UNet and the text encoders, F32 for the stock VAE (spec 11.9). An op that
//! needs F32 weights (LayerNorm, an embedding table) gets an F32 copy at load; F16 to F32 is exact.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nagaya_mk_bmg::safetensors::{Dtype, MappedSafeTensors};

/// Where a tensor is on the card, and its dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub at: u64,
    /// True for F32 values, false for F16.
    pub f32: bool,
}

/// A sink for the upload: a card, or a count of bytes for the plan.
pub trait Upload {
    /// Device memory that holds `data`. A count gives an address of 0 and copies nothing.
    fn upload(&mut self, data: &[u8]) -> u64;
    /// True when `upload` copies. A count copies nothing, so the caller need not compute the data.
    fn writes(&self) -> bool;
}

/// An upload that only counts bytes.
#[derive(Default)]
pub struct Count {
    pub bytes: u64,
}

impl Upload for Count {
    fn upload(&mut self, data: &[u8]) -> u64 {
        self.bytes += data.len().div_ceil(256) as u64 * 256;
        0
    }

    fn writes(&self) -> bool {
        false
    }
}

/// The files of a checkpoint in the folder layout of diffusers, with the prefix of their tensor names here.
pub fn files(dir: &Path) -> Result<Vec<(&'static str, PathBuf)>, String> {
    let pick = |folder: &str, stem: &str, prefer_f16: bool| -> Result<PathBuf, String> {
        let f16 = dir.join(folder).join(format!("{stem}.fp16.safetensors"));
        let plain = dir.join(folder).join(format!("{stem}.safetensors"));
        let order = if prefer_f16 { [&f16, &plain] } else { [&plain, &f16] };
        order.into_iter().find(|p| p.is_file()).cloned().ok_or_else(|| {
            format!(
                "{} has no {stem}.safetensors or {stem}.fp16.safetensors in {folder}/",
                dir.display()
            )
        })
    };
    Ok(vec![
        ("te1.", pick("text_encoder", "model", true)?),
        ("te2.", pick("text_encoder_2", "model", true)?),
        ("unet.", pick("unet", "diffusion_pytorch_model", true)?),
        // The stock VAE is F32 (spec 11.9), so the plain file comes first.
        ("vae.", pick("vae", "diffusion_pytorch_model", false)?),
    ])
}

/// True for a tensor whose op needs F32 values: a LayerNorm, or an embedding table.
pub fn needs_f32(name: &str) -> bool {
    let layer_norm =
        name.contains("layer_norm") || name.contains(".norm1.") || name.contains(".norm2.") || name.contains(".norm3.");
    let transformer = name.starts_with("te") || name.contains("transformer_blocks");
    (layer_norm && transformer)
        || name.ends_with("token_embedding.weight")
        || name.ends_with("position_embedding.weight")
}

/// The F32 values of F16 bits.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as f32;
    match exp {
        0 => sign * mant * 2f32.powi(-24),
        31 => {
            if mant == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// The bytes of `f32` values.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The `f32` values of little-endian bytes.
pub fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The planes of a weight for `OP_GEMM` on the card: their address and their count (2 for F16, 3 for F32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Planes {
    pub at: u64,
    pub planes: u32,
}

/// True for a weight that runs on the matrix units: a linear or a convolution weight with inputs in 64s and outputs
/// in 16s, which is not an embedding table.
pub fn gemm_eligible(name: &str, shape: &[usize]) -> bool {
    let kernel_ok = match shape {
        [_, _] => true,
        [_, _, k, k2] => k == k2 && (*k == 1 || *k == 3),
        _ => false,
    };
    kernel_ok
        && name.ends_with(".weight")
        && !name.contains("embedding.weight")
        && shape[1].is_multiple_of(64)
        && shape[0].is_multiple_of(16)
}

/// The tensors of the engine on its card, by their prefixed names. A weight with planes has no native copy on the
/// card: its address in `tensors` is 0.
pub struct Weights {
    tensors: HashMap<String, (Tensor, Vec<usize>)>,
    f32_copies: HashMap<String, u64>,
    planes: HashMap<String, Planes>,
}

impl Weights {
    /// Upload each tensor of the checkpoint in `dir` into `sink`, and an F32 copy of each tensor that needs one.
    pub fn load(dir: &Path, sink: &mut dyn Upload) -> Result<Weights, String> {
        let mut tensors = HashMap::new();
        let mut f32_copies = HashMap::new();
        let mut planes = HashMap::new();
        for (prefix, path) in files(dir)? {
            let mapped = MappedSafeTensors::open(&path)?;
            let view = mapped.view()?;
            let mut names: Vec<&String> = view.tensors.keys().collect();
            names.sort();
            // The planes of the weights of the matrix units, made on host threads in batches, then uploaded in order.
            let eligible: Vec<&String> = names
                .iter()
                .copied()
                .filter(|n| gemm_eligible(&format!("{prefix}{n}"), &view.tensors[*n].shape))
                .collect();
            for batch in eligible.chunks(32) {
                let made: Vec<(String, u32, Vec<u8>)> = std::thread::scope(|scope| {
                    let threads: Vec<_> = batch
                        .iter()
                        .map(|name| {
                            let meta = &view.tensors[*name];
                            let bytes = view.tensor_bytes(name).expect("a tensor of the header");
                            let writes = sink.writes();
                            scope.spawn(move || plane_bytes(name, meta, bytes, writes))
                        })
                        .collect();
                    threads
                        .into_iter()
                        .map(|t| t.join().expect("a plane thread panicked"))
                        .collect::<Result<_, _>>()
                })?;
                for (name, count, bytes) in made {
                    let at = sink.upload(&bytes);
                    planes.insert(format!("{prefix}{name}"), Planes { at, planes: count });
                }
            }
            for name in names {
                let meta = &view.tensors[name];
                let full = format!("{prefix}{name}");
                let f32 = match meta.dtype {
                    Dtype::F16 => false,
                    Dtype::F32 => true,
                    other => {
                        return Err(format!(
                            "tensor {full} is {other:?}; the image engine reads F16 and F32"
                        ));
                    }
                };
                let bytes = view.tensor_bytes(name).expect("a tensor of the header");
                let at = if planes.contains_key(&full) {
                    0
                } else {
                    sink.upload(bytes)
                };
                if needs_f32(&full) {
                    let copy = if !sink.writes() {
                        sink.upload(&vec![0u8; meta.shape.iter().product::<usize>() * 4])
                    } else if f32 {
                        at
                    } else {
                        let values: Vec<f32> = bytes
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
                            .collect();
                        sink.upload(&f32_bytes(&values))
                    };
                    f32_copies.insert(full.clone(), copy);
                }
                tensors.insert(full, (Tensor { at, f32 }, meta.shape.clone()));
            }
        }
        Ok(Weights {
            tensors,
            f32_copies,
            planes,
        })
    }

    /// The planes of the weight `name` for the matrix units, when it has them. The shape must be `shape`.
    pub fn planes(&self, name: &str, shape: &[usize]) -> Result<Option<Planes>, String> {
        self.shaped(name, shape)?;
        Ok(self.planes.get(name).copied())
    }

    fn shaped(&self, name: &str, shape: &[usize]) -> Result<Tensor, String> {
        let (tensor, got) = self.tensors.get(name).ok_or_else(|| {
            format!("the checkpoint has no tensor {name}; use an SDXL checkpoint in the layout of diffusers")
        })?;
        if got != shape {
            return Err(format!("tensor {name} has the shape {got:?}, and SDXL needs {shape:?}"));
        }
        Ok(*tensor)
    }

    /// The native tensor `name`, which must have `shape`. A weight with planes has no native copy: that is an error.
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Tensor, String> {
        let tensor = self.shaped(name, shape)?;
        if self.planes.contains_key(name) {
            return Err(format!(
                "tensor {name} has planes for the matrix units and no native copy; use the planes"
            ));
        }
        Ok(tensor)
    }

    /// The F32 copy of the tensor `name`, which must have `shape`.
    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<u64, String> {
        self.shaped(name, shape)?;
        self.f32_copies
            .get(name)
            .copied()
            .ok_or_else(|| format!("tensor {name} has no F32 copy; this is a defect of needs_f32"))
    }
}

/// The planes of one weight for the matrix units: its name, the count of planes, and their bytes. A count makes zeros
/// of the right length and computes nothing.
fn plane_bytes(
    name: &str,
    meta: &nagaya_mk_bmg::safetensors::TensorMeta,
    bytes: &[u8],
    writes: bool,
) -> Result<(String, u32, Vec<u8>), String> {
    let shape = &meta.shape;
    let (cout, cin) = (shape[0], shape[1]);
    let kernel = if shape.len() == 4 { shape[2] } else { 1 };
    let count: u32 = match meta.dtype {
        Dtype::F16 => 2,
        Dtype::F32 => 3,
        other => {
            return Err(format!(
                "tensor {name} is {other:?}; the image engine reads F16 and F32"
            ));
        }
    };
    if !writes {
        return Ok((
            name.to_owned(),
            count,
            vec![0u8; count as usize * cout * cin * kernel * kernel * 2],
        ));
    }
    let values: Vec<f32> = match meta.dtype {
        Dtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
            .collect(),
        _ => f32_values(bytes),
    };
    let planes = nagaya_mk_bmg::tasks::gemm_planes(&values, cout, cin, kernel, count as usize);
    Ok((
        name.to_owned(),
        count,
        planes.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ))
}
