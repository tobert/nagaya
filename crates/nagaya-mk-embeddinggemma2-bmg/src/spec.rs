//! The numbers of the text tower of EmbeddingGemma 2, which the engine builds for, and the config of the engine.
//!
//! The numbers are those of `google/embeddinggemma-2` (`config.json`, `text_config`). `check_config` compares the
//! config of a checkpoint with them, so another checkpoint is an error at the plan, not a wrong vector.

use std::path::{Path, PathBuf};

/// The model name of the identity.
pub const MODEL: &str = "embeddinggemma-2";
/// The prefix of the tensor names of the text tower.
pub const PREFIX: &str = "language_model.";
/// The file of the weights in the checkpoint directory.
pub const WEIGHTS_FILE: &str = "model.safetensors";

pub const HIDDEN: usize = 512;
pub const LAYERS: usize = 24;
pub const HEADS: usize = 4;
pub const INTERMEDIATE: usize = 2048;
/// The width of the per-layer input of each layer (`hidden_size_per_layer_input`).
pub const PER_LAYER: usize = 512;
/// The length of the vector: the output of `embedding_projection`.
pub const DIMENSION: usize = 768;
pub const VOCAB: usize = 262_144;
/// A sliding layer sees the keys within this many positions of the query, on both sides.
pub const WINDOW: usize = 512;
pub const EPS: f32 = 1e-6;
/// The most ids of one input (spec 11.10, `context_tokens`).
pub const MAX_TOKENS: usize = 2048;
/// The full-attention layers; the others are sliding layers.
pub const FULL_LAYERS: [usize; 4] = [5, 11, 17, 23];

/// The attention of one kind of layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attn {
    pub kv_heads: usize,
    pub head_dim: usize,
    /// The rotary theta. Each layer rotates the whole head.
    pub theta: f32,
    /// The window of `OP_BIATTN`: 0 for every key.
    pub window: usize,
}

pub const SLIDING: Attn = Attn {
    kv_heads: 2,
    head_dim: 256,
    theta: 1e4,
    window: WINDOW,
};
pub const FULL: Attn = Attn {
    kv_heads: 1,
    head_dim: 512,
    theta: 1e6,
    window: 0,
};

/// The attention of layer `layer`.
pub fn attn(layer: usize) -> Attn {
    if FULL_LAYERS.contains(&layer) { FULL } else { SLIDING }
}

/// The scale of the embeddings: sqrt(512) in f32. The reference in f32 keeps it in f32; a bf16 run rounds it to 22.625.
pub fn embed_scale() -> f32 {
    (HIDDEN as f32).sqrt()
}

/// The scale of the per-layer projection: 512^-0.5 in f32.
pub fn per_layer_scale() -> f32 {
    (HIDDEN as f32).powf(-0.5)
}

/// The config of the engine: the checkpoint directory. It holds `config.json` and `model.safetensors`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub model_dir: PathBuf,
}

/// Check the `text_config` of the checkpoint in `dir` against the numbers of this engine.
pub fn check_config(dir: &Path) -> Result<(), String> {
    let path = dir.join("config.json");
    let bytes = std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let config: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{} is not JSON: {e}", path.display()))?;
    let text = &config["text_config"];
    let want = |key: &str, value: serde_json::Value| -> Result<(), String> {
        if text[key] == value {
            Ok(())
        } else {
            Err(format!(
                "{} has text_config.{key} {}, and this engine needs {value}; use the checkpoint google/embeddinggemma-2",
                path.display(),
                text[key]
            ))
        }
    };
    want("vocab_size", VOCAB.into())?;
    want("hidden_size", HIDDEN.into())?;
    want("num_hidden_layers", LAYERS.into())?;
    want("num_attention_heads", HEADS.into())?;
    want("num_key_value_heads", SLIDING.kv_heads.into())?;
    want("head_dim", SLIDING.head_dim.into())?;
    want("intermediate_size", INTERMEDIATE.into())?;
    want("hidden_size_per_layer_input", PER_LAYER.into())?;
    want("embedding_dim", DIMENSION.into())?;
    want("sliding_window", WINDOW.into())?;
    want("rms_norm_eps", serde_json::json!(1e-6))?;
    want("hidden_activation", "gelu_pytorch_tanh".into())?;
    want("attention_bias", false.into())?;
    let rope = &text["rope_parameters"];
    for (kind, attn) in [("full_attention", FULL), ("sliding_attention", SLIDING)] {
        let theta = rope[kind]["rope_theta"].as_f64();
        if theta != Some(attn.theta as f64) || rope[kind]["rope_type"] != "default" {
            return Err(format!(
                "{} has the rope of {kind} {}, and this engine needs the default rope with theta {}",
                path.display(),
                rope[kind],
                attn.theta
            ));
        }
    }
    let types = text["layer_types"].as_array().map(|types| {
        types
            .iter()
            .enumerate()
            .filter(|(_, t)| *t == "full_attention")
            .map(|(i, _)| i)
            .collect::<Vec<usize>>()
    });
    if types.as_deref() != Some(&FULL_LAYERS[..]) || text["layer_types"].as_array().map(Vec::len) != Some(LAYERS) {
        return Err(format!(
            "{} has the layer types {}, and this engine needs full attention at layers {FULL_LAYERS:?} of {LAYERS}",
            path.display(),
            text["layer_types"]
        ));
    }
    for layer in FULL_LAYERS {
        let entry = &text["per_layer_config"][format!("{layer:02}")];
        if entry["head_dim"] != FULL.head_dim || entry["num_key_value_heads"] != FULL.kv_heads {
            return Err(format!(
                "{} has the per-layer config {entry} for layer {layer}, and this engine needs head_dim {} and {} KV head",
                path.display(),
                FULL.head_dim,
                FULL.kv_heads
            ));
        }
    }
    Ok(())
}
