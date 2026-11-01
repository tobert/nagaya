//! The plan of the Qwen-Image engine (spec 2.4, 11.9c "The load"): its limits from the tensor headers of its files,
//! with no weight bytes read and no GPU. The engine's open loads the same tensors that the plan counts.

use std::path::{Path, PathBuf};

use nagaya_abi::{
    Capability, ChainLimits, Determinism, Inputs, Kind, Lifetime, Limits, MandatoryTerm, MemoryLimits, SlotLimit, Tier,
};

use crate::names::{EncoderNames, LAYER_PARTS, LAYERS};
use crate::safetensors::{Dtype, Safetensors};

/// The most template ids (spec 11.9c), and so the most conditioning rows after the cut of 14.
pub const MAX_IDS: usize = 1024;
pub const CUT: usize = 14;
pub const MAX_ROWS: usize = MAX_IDS - CUT;
/// The bytes of one conditioning row: 4096 values in bf16.
pub const ROW_BYTES: usize = 4096 * 2;
/// The most images of a job, and the most pixels of each (spec 11.9c).
pub const MAX_IMAGES: usize = 4;
pub const MAX_PIXELS: usize = 2_097_152;
/// The longest quantum in milliseconds (D8). Spec 11.9c gives 60000 until a measure.
pub const MAX_QUANTUM_MS: u32 = 60_000;
/// The scale from a latent pixel to an image pixel.
const SCALE: usize = 16;

/// The files of a variant: the profile names each one (spec 11.9c, "The engine and its weights").
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// The base folder: `transformer/`, `vae/` and `processor/`.
    pub model: PathBuf,
    /// The safetensors file of the text encoder.
    pub encoder: PathBuf,
    /// The LoRA of the variant and its strength.
    pub lora: Option<(PathBuf, f32)>,
    /// The directory of the plan table of the GEMMs (spec 12.1), or none for a table in memory.
    pub plan_dir: Option<PathBuf>,
}

impl Config {
    /// The config of a profile entry: `model`, `encoder`, and the optional `lora`, `lora_strength` and `plan_dir`. A
    /// key that the engine does not know is an error.
    pub fn from_toml(table: &toml::Table) -> Result<Config, String> {
        for key in table.keys() {
            if !["model", "encoder", "lora", "lora_strength", "plan_dir"].contains(&key.as_str()) {
                return Err(format!(
                    "the qwenimage21 engine has no config `{key}`; the keys are `model`, `encoder`, `lora`, \
                     `lora_strength` and `plan_dir`"
                ));
            }
        }
        let path = |key: &str| -> Result<Option<PathBuf>, String> {
            match table.get(key) {
                None => Ok(None),
                Some(value) => value
                    .as_str()
                    .map(|text| Some(PathBuf::from(text)))
                    .ok_or_else(|| format!("the config `{key}` of the qwenimage21 engine is a path")),
            }
        };
        let need = |key: &str| -> Result<PathBuf, String> {
            path(key)?.ok_or_else(|| format!("the qwenimage21 engine needs the path `{key}` in its config"))
        };
        let strength = match table.get("lora_strength") {
            None => 1.0,
            Some(value) => value
                .as_float()
                .filter(|s| s.is_finite())
                .ok_or_else(|| "the config `lora_strength` is a finite number".to_owned())?
                as f32,
        };
        let lora = path("lora")?.map(|file| (file, strength));
        if lora.is_none() && table.contains_key("lora_strength") {
            return Err("the config has `lora_strength` and no `lora`".to_owned());
        }
        Ok(Config {
            model: need("model")?,
            encoder: need("encoder")?,
            lora,
            plan_dir: path("plan_dir")?,
        })
    }
}

/// The bytes of the bf16 weights that the open loads, from the headers: the text tower, the transformer and the VAE
/// decoder. A file that lacks a tensor of the load is an error here, before any open.
pub fn weights_bytes(config: &Config) -> Result<u64, String> {
    let encoder = Safetensors::open(&config.encoder).map_err(|e| e.to_string())?;
    let names = EncoderNames::check(encoder.tensors().keys().map(String::as_str))?;
    let bf16 = |elements: usize| (elements * 2) as u64;
    let mut loaded = vec![names.embed()];
    for i in 0..LAYERS {
        loaded.extend(LAYER_PARTS.iter().map(|(part, _)| names.layer(i, part)));
    }
    let mut bytes = 0u64;
    for name in &loaded {
        bytes += bf16(encoder.info(name).map_err(|e| e.to_string())?.elements());
    }
    let transformer = config.model.join("transformer");
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&transformer)
        .map_err(|e| format!("cannot list {}: {e}", transformer.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(format!("{} has no safetensors file", transformer.display()));
    }
    for shard in &shards {
        let file = Safetensors::open(shard).map_err(|e| e.to_string())?;
        for info in file.tensors().values() {
            if info.dtype != Dtype::Bf16 {
                return Err(format!("{} holds a tensor that is not bf16", shard.display()));
            }
            bytes += bf16(info.elements());
        }
    }
    let vae =
        Safetensors::open(&config.model.join("vae/diffusion_pytorch_model.safetensors")).map_err(|e| e.to_string())?;
    for (name, info) in vae.tensors() {
        if (name.starts_with("decoder.") || name.starts_with("post_quant_conv.")) && !name.contains(".time_conv.") {
            bytes += bf16(info.elements());
        }
    }
    Ok(bytes)
}

/// The work buffers of the largest job, for the plan: one image of 2 MP at a time over 1010 condition rows, with the
/// keys and values of the condition in 32 blocks. Step 8 of milestone 22 measures it.
pub fn scratch_bytes() -> u64 {
    let tokens = MAX_PIXELS / (SCALE * SCALE);
    let qkv = (MAX_ROWS + tokens) * 3 * 4096 * 2;
    let work = tokens * (4 * 4096 + 2 * 12288 + 12288) * 2;
    // The scores in f32 and the probabilities in bf16, for one head at least.
    let scores = tokens * (MAX_ROWS + tokens) * 6;
    let kv = 32 * MAX_ROWS * 2 * 4096 * 2;
    // The latents of 4 images, and the im2col scratch and the largest activations of the decode.
    let latents = MAX_IMAGES * tokens * 64 * 2;
    let decode = (256 << 20) + 6 * MAX_PIXELS * 288 * 2;
    (qkv + work + scores + kv + latents + decode) as u64
}

fn term(name: &str, bytes: u64) -> MandatoryTerm {
    MandatoryTerm {
        name: name.to_owned(),
        tier: Tier::Device(0),
        bytes,
        lifetime: Lifetime::Engine,
        kinds: vec![Kind::Embed, Kind::Step],
    }
}

/// The limits of the engine with `weights` bytes of weights.
pub fn limits(weights: u64) -> Limits {
    Limits {
        // A drop frees the weights and the work buffers.
        capabilities: vec![Capability::Image, Capability::Close],
        parts: Vec::new(),
        inputs: Inputs::Tokens,
        devices: vec![0],
        stages: Vec::new(),
        max_segments: 0,
        max_pass_inputs: 0,
        max_logit_rows: 0,
        // One job at a time. The conditioning is in the store, and the engine declares no part (spec 6.16).
        slots: vec![SlotLimit {
            bytes: 0,
            count: 1,
            tokens: MAX_IDS as u64,
        }],
        max_quantum_ms: MAX_QUANTUM_MS,
        memory: MemoryLimits {
            mandatory: vec![term("weights", weights), term("scratch", scratch_bytes())],
            ..Default::default()
        },
        // A conditioning is one snapshot with no chain (spec 6.16).
        chain: ChainLimits {
            max_depth: 0,
            delta_fraction: 0.0,
        },
        segments_exact: false,
        determinism: Determinism::Pinned,
        bit_dependence: Vec::new(),
        taps: Vec::new(),
        checkpoint_grain: None,
        residual_row_bytes: 0,
        feed: None,
    }
}

/// The limits of the engine for a config, from the headers of its files (spec 2.4).
pub fn plan(config: &Config) -> Result<Limits, String> {
    Ok(limits(weights_bytes(config)?))
}

/// True when `path` names a file that exists. The factory checks the files of a config before a load.
pub fn exists(path: &Path) -> bool {
    path.is_file()
}
