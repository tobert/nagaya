//! The weights of the engine: the four files of a checkpoint in the folder layout of diffusers, or one single file with
//! Stability's tensor names (`single.rs`), and their upload.
//!
//! Each tensor keeps its dtype: F16 for the UNet and the text encoders, F32 for the stock VAE (spec 11.9). An op that
//! needs F32 weights (LayerNorm, an embedding table) gets an F32 copy at load; F16 to F32 is exact.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nagaya_mk_bmg::safetensors::{Dtype, MappedSafeTensors};

use crate::single::Take;
use crate::spec::Config;

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

/// The F32 values of F16 bits. The B70 engines share the decoder.
pub use nagaya_mk_bmg::safetensors::f16_to_f32;

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

/// One tensor of a checkpoint, by its prefixed name in the folder layout of diffusers.
pub struct Item<'a> {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: Cow<'a, [u8]>,
}

/// The files of the checkpoint of a config, with the prefix of their tensor names: the folder of diffusers, or the
/// single file with Stability's tensor names, whose names carry their own prefix.
pub fn sources(config: &Config) -> Result<Vec<(&'static str, PathBuf)>, String> {
    match &config.checkpoint {
        Some(file) => Ok(vec![("", file.clone())]),
        None => files(&config.model_dir),
    }
}

/// The positions of a CLIP encoder, which some exports of transformers keep in the file (spec 11.9b, the inpainting
/// checkpoint). No op reads them.
const POSITION_IDS: &str = "text_model.embeddings.position_ids";

/// Check that the tensor `name` holds the positions 0 to 76 in I64, as the engine gives them. Other values would make
/// another encoder, so they are an error.
fn position_ids(name: &str, dtype: Dtype, shape: &[usize], bytes: &[u8]) -> Result<(), String> {
    let want: Vec<u8> = (0..crate::spec::CHUNK_IDS as i64)
        .flat_map(|v| v.to_le_bytes())
        .collect();
    if dtype != Dtype::I64 || shape != [1, crate::spec::CHUNK_IDS] || bytes != want.as_slice() {
        return Err(format!(
            "tensor {name} is {dtype:?} of shape {shape:?} and does not hold the positions 0 to 76; the engine gives \
             each position its index, so use a checkpoint with those positions or none"
        ));
    }
    Ok(())
}

/// The tensors of mapped files, from `sources`. `single` maps the names of the single file (`single.rs`).
pub fn items<'a>(maps: &'a [(&'static str, MappedSafeTensors)], single: bool) -> Result<Vec<Item<'a>>, String> {
    let mut items = Vec::new();
    for (prefix, map) in maps {
        let view = map.view()?;
        let mut names: Vec<&String> = view.tensors.keys().collect();
        names.sort();
        for name in names {
            let meta = &view.tensors[name];
            let bytes = view.tensor_bytes(name).expect("a tensor of the header");
            if !single && name == POSITION_IDS {
                // The engine gives each position its index, so these ids carry nothing when they are 0 to 76.
                position_ids(&format!("{prefix}{name}"), meta.dtype, &meta.shape, bytes)?;
                continue;
            }
            if !single {
                items.push(Item {
                    name: format!("{prefix}{name}"),
                    dtype: meta.dtype,
                    shape: meta.shape.clone(),
                    bytes: Cow::Borrowed(bytes),
                });
                continue;
            }
            for (to, take) in crate::single::map_name(name)? {
                items.push(take_item(name, to, take, meta.dtype, &meta.shape, bytes)?);
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    if let Some(twice) = items.iter().find(|item| !seen.insert(item.name.as_str())) {
        return Err(format!("the checkpoint gives the tensor {} twice", twice.name));
    }
    Ok(items)
}

/// The tensor `to` of the folder layout, from the tensor `from` of a single file. An F16 tensor of the VAE becomes
/// F32, because the VAE runs in F32 (spec 11.9); F16 to F32 is exact.
fn take_item<'a>(
    from: &str,
    to: String,
    take: Take,
    dtype: Dtype,
    shape: &[usize],
    bytes: &'a [u8],
) -> Result<Item<'a>, String> {
    let size = dtype.size_bytes();
    let (shape, bytes): (Vec<usize>, Cow<'a, [u8]>) = match take {
        Take::Whole => (shape.to_vec(), Cow::Borrowed(bytes)),
        Take::Rows { from: row, count } => {
            if shape.is_empty() || shape[0] < row + count {
                return Err(format!(
                    "tensor {from} has the shape {shape:?}, and {to} needs its rows {row}..{}",
                    row + count
                ));
            }
            let row_bytes = shape[1..].iter().product::<usize>() * size;
            let mut rows = shape.to_vec();
            rows[0] = count;
            (rows, Cow::Borrowed(&bytes[row * row_bytes..(row + count) * row_bytes]))
        }
        Take::Transpose => {
            let [r, c] = shape[..] else {
                return Err(format!(
                    "tensor {from} has the shape {shape:?}, and {to} needs a matrix"
                ));
            };
            let mut out = vec![0u8; bytes.len()];
            for i in 0..r {
                for j in 0..c {
                    out[(j * r + i) * size..(j * r + i + 1) * size]
                        .copy_from_slice(&bytes[(i * c + j) * size..(i * c + j + 1) * size]);
                }
            }
            (vec![c, r], Cow::Owned(out))
        }
        Take::Squeeze => {
            let mut squeezed = shape.to_vec();
            while squeezed.len() > 2 && squeezed.last() == Some(&1) {
                squeezed.pop();
            }
            (squeezed, Cow::Borrowed(bytes))
        }
    };
    if to.starts_with("vae.") && dtype == Dtype::F16 {
        let values: Vec<f32> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
            .collect();
        return Ok(Item {
            name: to,
            dtype: Dtype::F32,
            shape,
            bytes: Cow::Owned(f32_bytes(&values)),
        });
    }
    Ok(Item {
        name: to,
        dtype,
        shape,
        bytes,
    })
}

/// A LoRA delta of one module on the card (spec 11.9a): `up` holds the scale, and `scratch` holds the hidden values
/// between the two matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adapter {
    /// The down matrix: `rank` filters of `cin * kernel * kernel` values, in F32.
    pub down: u64,
    /// The up matrix, scaled: `cout` rows of `rank` values, in F32.
    pub up: u64,
    pub rank: usize,
    /// The kernel of the module: 1 for a linear layer, 3 for a 3 x 3 convolution.
    pub kernel: usize,
    /// Rows of `rank` values, for the hidden values of the largest batch of the job.
    pub scratch: u64,
}

/// The tensors of the engine on its card, by their prefixed names. A weight with planes has no native copy on the
/// card: its address in `tensors` is 0.
pub struct Weights {
    tensors: HashMap<String, (Tensor, Vec<usize>)>,
    f32_copies: HashMap<String, u64>,
    planes: HashMap<String, Planes>,
    /// The LoRA deltas of the job, by the prefixed name of their module, in the order of the request.
    adapters: RefCell<HashMap<String, Vec<Adapter>>>,
    /// True while [`Weights::probe`] runs: a lookup that fails records its error and gives address 0.
    probing: Cell<bool>,
    missing: RefCell<Vec<String>>,
}

impl Weights {
    /// Upload each tensor of the checkpoint of `config` into `sink`, and an F32 copy of each tensor that needs one.
    pub fn load(config: &Config, sink: &mut dyn Upload) -> Result<Weights, String> {
        let maps = sources(config)?
            .into_iter()
            .map(|(prefix, path)| Ok((prefix, MappedSafeTensors::open(&path)?)))
            .collect::<Result<Vec<_>, String>>()?;
        Weights::from_items(&items(&maps, config.checkpoint.is_some())?, sink)
    }

    /// Upload these tensors into `sink`, and an F32 copy of each tensor that needs one.
    pub fn from_items(items: &[Item], sink: &mut dyn Upload) -> Result<Weights, String> {
        let mut tensors = HashMap::new();
        let mut f32_copies = HashMap::new();
        let mut planes = HashMap::new();
        for item in items {
            if !matches!(item.dtype, Dtype::F16 | Dtype::F32) {
                return Err(format!(
                    "tensor {} is {:?}; the image engine reads F16 and F32",
                    item.name, item.dtype
                ));
            }
        }
        // The planes of the weights of the matrix units, made on host threads in batches, then uploaded in order.
        let eligible: Vec<&Item> = items
            .iter()
            .filter(|item| gemm_eligible(&item.name, &item.shape))
            .collect();
        for batch in eligible.chunks(32) {
            let writes = sink.writes();
            let made: Vec<(u32, Vec<u8>)> = std::thread::scope(|scope| {
                let threads: Vec<_> = batch
                    .iter()
                    .map(|item| scope.spawn(move || plane_bytes(item, writes)))
                    .collect();
                threads
                    .into_iter()
                    .map(|t| t.join().expect("a plane thread panicked"))
                    .collect()
            });
            for (item, (count, bytes)) in batch.iter().zip(made) {
                let at = sink.upload(&bytes);
                planes.insert(item.name.clone(), Planes { at, planes: count });
            }
        }
        for item in items {
            let f32 = item.dtype == Dtype::F32;
            let at = if planes.contains_key(&item.name) {
                0
            } else {
                sink.upload(&item.bytes)
            };
            if needs_f32(&item.name) {
                // An F32 tensor is its own copy. A count must count the same bytes as the upload.
                let copy = if f32 {
                    at
                } else if !sink.writes() {
                    sink.upload(&vec![0u8; item.shape.iter().product::<usize>() * 4])
                } else {
                    let values: Vec<f32> = item
                        .bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
                        .collect();
                    sink.upload(&f32_bytes(&values))
                };
                f32_copies.insert(item.name.clone(), copy);
            }
            tensors.insert(item.name.clone(), (Tensor { at, f32 }, item.shape.clone()));
        }
        Ok(Weights {
            tensors,
            f32_copies,
            planes,
            adapters: RefCell::new(HashMap::new()),
            probing: Cell::new(false),
            missing: RefCell::new(Vec::new()),
        })
    }

    /// The shape of the tensor `name`, or `None` when the checkpoint has no such tensor.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|(_, shape)| shape.as_slice())
    }

    /// Set the LoRA deltas of the job. An empty map is no LoRA.
    pub fn set_adapters(&self, adapters: HashMap<String, Vec<Adapter>>) {
        *self.adapters.borrow_mut() = adapters;
    }

    /// The LoRA deltas of the module `name`, with its prefix and no `.weight`.
    pub fn adapters(&self, name: &str) -> Vec<Adapter> {
        self.adapters.borrow().get(name).cloned().unwrap_or_default()
    }

    /// The modules that a LoRA can change: each name with no `.weight`, and its shape. These are the linear layers and
    /// the convolutions of the UNet and of both text encoders, not the embedding tables.
    pub fn modules(&self) -> Vec<(String, Vec<usize>)> {
        let mut out: Vec<(String, Vec<usize>)> = self
            .tensors
            .iter()
            .filter_map(|(name, (_, shape))| {
                let module = name.strip_suffix(".weight")?;
                let prefixed = ["unet.", "te1.", "te2."].iter().any(|p| module.starts_with(p));
                let kind_ok = match shape[..] {
                    [_, _] => true,
                    [_, _, k, k2] => k == k2 && (k == 1 || k == 3),
                    _ => false,
                };
                (prefixed && kind_ok && !module.contains("embeddings.")).then(|| (module.to_owned(), shape.clone()))
            })
            .collect();
        out.sort();
        out
    }

    /// Run `build`, which looks up tensors, and give each lookup that failed. During the probe a failed lookup gives
    /// address 0, so `build` must not run what it builds.
    pub fn probe(&self, build: impl FnOnce()) -> Result<(), String> {
        self.probing.set(true);
        build();
        self.probing.set(false);
        let mut missing = std::mem::take(&mut *self.missing.borrow_mut());
        // A graph can look up one tensor more than once.
        missing.sort();
        missing.dedup();
        match missing.first() {
            None => Ok(()),
            Some(first) => Err(format!(
                "the checkpoint lacks {} tensors that SDXL needs, such as: {first}",
                missing.len()
            )),
        }
    }

    /// The error of a lookup: an error, or a record and address 0 during a probe.
    fn failed<T>(&self, message: String, zero: T) -> Result<T, String> {
        if self.probing.get() {
            self.missing.borrow_mut().push(message);
            return Ok(zero);
        }
        Err(message)
    }

    /// The planes of the weight `name` for the matrix units, when it has them. The shape must be `shape`.
    pub fn planes(&self, name: &str, shape: &[usize]) -> Result<Option<Planes>, String> {
        self.shaped(name, shape)?;
        Ok(self.planes.get(name).copied())
    }

    fn shaped(&self, name: &str, shape: &[usize]) -> Result<Tensor, String> {
        let zero = Tensor { at: 0, f32: false };
        let Some((tensor, got)) = self.tensors.get(name) else {
            return self.failed(
                format!("the checkpoint has no tensor {name}; use an SDXL checkpoint in the layout of diffusers"),
                zero,
            );
        };
        if got != shape {
            return self.failed(
                format!("tensor {name} has the shape {got:?}, and SDXL needs {shape:?}"),
                zero,
            );
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
        match self.f32_copies.get(name) {
            Some(at) => Ok(*at),
            None if self.probing.get() && !self.tensors.contains_key(name) => Ok(0),
            None => Err(format!("tensor {name} has no F32 copy; this is a defect of needs_f32")),
        }
    }
}

/// The planes of one weight for the matrix units: the count of planes, and their bytes. A count makes zeros of the
/// right length and computes nothing.
fn plane_bytes(item: &Item, writes: bool) -> (u32, Vec<u8>) {
    let shape = &item.shape;
    let (cout, cin) = (shape[0], shape[1]);
    let kernel = if shape.len() == 4 { shape[2] } else { 1 };
    let count: u32 = if item.dtype == Dtype::F16 { 2 } else { 3 };
    if !writes {
        return (count, vec![0u8; count as usize * cout * cin * kernel * kernel * 2]);
    }
    let values: Vec<f32> = match item.dtype {
        Dtype::F16 => item
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
            .collect(),
        _ => f32_values(&item.bytes),
    };
    let planes = nagaya_mk_bmg::tasks::gemm_planes(&values, cout, cin, kernel, count as usize);
    (count, planes.iter().flat_map(|v| v.to_le_bytes()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sink that uploads, and counts its bytes as [`Count`] does.
    #[derive(Default)]
    struct Uploads {
        bytes: u64,
        next: u64,
    }

    impl Upload for Uploads {
        fn upload(&mut self, data: &[u8]) -> u64 {
            self.bytes += data.len().div_ceil(256) as u64 * 256;
            self.next += 256;
            self.next
        }

        fn writes(&self) -> bool {
            true
        }
    }

    fn item(name: &str, dtype: Dtype, values: usize) -> Item<'static> {
        Item {
            name: name.to_owned(),
            dtype,
            shape: vec![values],
            bytes: Cow::Owned(vec![0u8; values * dtype.size_bytes()]),
        }
    }

    /// The plan counts the bytes that the open uploads, so the limits of both agree (`in_place::open`). An F32 tensor
    /// that needs an F32 copy is its own copy.
    #[test]
    fn the_count_equals_the_upload_for_f16_and_f32_tensors_that_need_f32() {
        let items = [
            item("te1.text_model.final_layer_norm.weight", Dtype::F32, 768),
            item("te1.text_model.final_layer_norm.bias", Dtype::F16, 768),
            item("unet.conv_in.bias", Dtype::F16, 320),
        ];
        let (mut count, mut uploads) = (Count::default(), Uploads::default());
        Weights::from_items(&items, &mut count).unwrap();
        Weights::from_items(&items, &mut uploads).unwrap();
        assert_eq!(count.bytes, uploads.bytes);
    }
}
