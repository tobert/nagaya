//! The tensor names of the text tower of Qwen3-VL-8B, the encoder of Qwen-Image 2.1 (spec 11.9c, "The load").
//!
//! The Heretic file of FUSAL has the names of ComfyUI (`model.layers.*`). The base folder has the names of
//! transformers (`model.language_model.layers.*`). The engine reads both forms. It loads the embedding and the 36
//! layers, and it skips the vision tower, `lm_head` and the final norm by name. Any other name refuses the load.

use std::collections::BTreeSet;

/// The layers of the text tower.
pub const LAYERS: usize = 36;
/// The width of the hidden state.
pub const HIDDEN: usize = 4096;
/// The query heads and the key heads.
pub const HEADS: usize = 32;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
/// The width of the MLP.
pub const FFN: usize = 12288;
/// The rows of the embedding table.
pub const VOCAB: usize = 151_936;
/// The epsilon of each RMSNorm.
pub const EPS: f32 = 1e-6;
/// The theta of the rotary positions.
pub const THETA: f32 = 5_000_000.0;

/// The part of each layer, with its shape.
pub const LAYER_PARTS: [(&str, &[usize]); 11] = [
    ("input_layernorm.weight", &[HIDDEN]),
    ("self_attn.q_proj.weight", &[HEADS * HEAD_DIM, HIDDEN]),
    ("self_attn.k_proj.weight", &[KV_HEADS * HEAD_DIM, HIDDEN]),
    ("self_attn.v_proj.weight", &[KV_HEADS * HEAD_DIM, HIDDEN]),
    ("self_attn.q_norm.weight", &[HEAD_DIM]),
    ("self_attn.k_norm.weight", &[HEAD_DIM]),
    ("self_attn.o_proj.weight", &[HIDDEN, HEADS * HEAD_DIM]),
    ("post_attention_layernorm.weight", &[HIDDEN]),
    ("mlp.gate_proj.weight", &[FFN, HIDDEN]),
    ("mlp.up_proj.weight", &[FFN, HIDDEN]),
    ("mlp.down_proj.weight", &[HIDDEN, FFN]),
];

/// The names of one file of the encoder: the prefix of its text tower.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderNames {
    prefix: &'static str,
}

impl EncoderNames {
    /// Find the form of `names`, and check that the file holds each tensor of the text tower and no tensor that is
    /// in neither list.
    pub fn check<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<EncoderNames, String> {
        let names: BTreeSet<&str> = names.into_iter().collect();
        let prefix = if names.contains("model.embed_tokens.weight") {
            "model."
        } else if names.contains("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else {
            return Err("the file has no embed_tokens of a Qwen3-VL text tower in either name form".to_owned());
        };
        let form = EncoderNames { prefix };
        let mut wanted: BTreeSet<String> = (0..LAYERS)
            .flat_map(|i| LAYER_PARTS.iter().map(move |(part, _)| (i, *part)))
            .map(|(i, part)| form.layer(i, part))
            .collect();
        wanted.insert(form.embed());
        for name in &wanted {
            if !names.contains(name.as_str()) {
                return Err(format!("the file lacks the tensor `{name}` of the text tower"));
            }
        }
        let skipped = |name: &str| {
            name == "lm_head.weight" || name == format!("{prefix}norm.weight") || name.starts_with("model.visual.")
        };
        for name in names {
            if !wanted.contains(name) && !skipped(name) {
                return Err(format!("the tensor `{name}` is in no list of the encoder's load"));
            }
        }
        Ok(form)
    }

    pub fn embed(&self) -> String {
        format!("{}embed_tokens.weight", self.prefix)
    }

    pub fn layer(&self, index: usize, part: &str) -> String {
        format!("{}layers.{index}.{part}", self.prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tower(prefix: &str) -> Vec<String> {
        let mut names: Vec<String> = (0..LAYERS)
            .flat_map(|i| {
                LAYER_PARTS
                    .iter()
                    .map(move |(part, _)| format!("{prefix}layers.{i}.{part}"))
            })
            .collect();
        names.push(format!("{prefix}embed_tokens.weight"));
        names
    }

    fn check(names: &[String]) -> Result<EncoderNames, String> {
        EncoderNames::check(names.iter().map(String::as_str))
    }

    #[test]
    fn both_name_forms_are_read() {
        let mut comfy = tower("model.");
        comfy.extend(
            [
                "lm_head.weight",
                "model.norm.weight",
                "model.visual.blocks.0.attn.qkv.weight",
            ]
            .map(String::from),
        );
        let form = check(&comfy).unwrap();
        assert_eq!(form.layer(3, "mlp.up_proj.weight"), "model.layers.3.mlp.up_proj.weight");
        let mut hf = tower("model.language_model.");
        hf.extend(
            [
                "lm_head.weight",
                "model.language_model.norm.weight",
                "model.visual.merger.norm.weight",
            ]
            .map(String::from),
        );
        let form = check(&hf).unwrap();
        assert_eq!(form.embed(), "model.language_model.embed_tokens.weight");
    }

    #[test]
    fn a_missing_or_an_unknown_tensor_refuses_the_load() {
        let mut names = tower("model.");
        names.retain(|n| n != "model.layers.35.mlp.down_proj.weight");
        assert!(check(&names).unwrap_err().contains("layers.35.mlp.down_proj"));
        let mut names = tower("model.");
        names.push("model.layers.36.mlp.down_proj.weight".to_owned());
        assert!(check(&names).unwrap_err().contains("in no list"));
        let mut names = tower("model.");
        names.push("model.language_model.norm.weight".to_owned());
        assert!(
            check(&names).unwrap_err().contains("in no list"),
            "the norm of the other form is not skipped"
        );
        assert!(check(&["x.weight".to_owned()]).is_err());
    }
}
