//! The constants and the declared shapes of the mk engine.
//!
//! The numbers come from mk's `src/ops/model.h` and `src/engine/engine.h`. The engine has one device group and one
//! context pool in this binding (docs/engine-abi.md, "Proposed: v0 mk binding choices").

use std::collections::BTreeMap;
use std::path::Path;

use nagaya_abi::{
    BitClass, BitSetting, Capability, ChainLimits, Determinism, Dtype, ElasticTerm, Identity, Inputs, Kind, Lifetime,
    Limits, MandatoryTerm, MemoryLimits, PartKind, PartLimits, SlotLimit, Stage, Tier,
};

/// The vocabulary size (mk `kVocab`).
pub const VOCAB: u32 = 248_320;

/// The most tokens in one context (mk `kMaxContext`).
pub const MAX_CONTEXT: u32 = 32_768;

/// The most tokens in one pass (mk `kMaxPassTokens`).
pub const MAX_PASS_TOKENS: u32 = 256;

/// A context capacity is a multiple of this many tokens (mk `check_capacity`).
pub const CAPACITY_STEP: u32 = 256;

/// The context capacity that this binding creates, in tokens.
pub const CONTEXT_CAPACITY: u32 = 256;

/// The number of contexts in the declared pool.
pub const CONTEXT_COUNT: u32 = 2;

/// The longest quantum that mk declares, in milliseconds.
pub const MAX_QUANTUM_MS: u32 = 30_000;

/// The most segments in one pass (mk `kMaxSegments`).
pub const MAX_SEGMENTS: u32 = 16;

/// The number of layers (mk `kLayers`).
pub const LAYERS: u32 = 48;

/// The bytes of the recurrent state, part 0.
pub const GDN_STATE_BYTES: u64 = 4 * 36 * 48 * 128 * 128;

/// The bytes of the GDN convolution history, part 1.
pub const GDN_CONV_BYTES: u64 = 4 * 36 * 3 * 10_240;

/// The bytes of the PLE convolution history, part 2.
pub const PLE_CONV_BYTES: u64 = 4 * 9 * 10_240;

/// The bytes of one K/V row, part 3.
pub const KV_ROW_BYTES: u64 = 4 * 12 * 2 * 2 * 256;

/// The bytes of one index-key row, part 4.
pub const IDX_ROW_BYTES: u64 = 4 * 12 * 128;

/// The bytes of the PLE history, part 5.
pub const PLE_HIST_BYTES: u64 = 8;

/// The context bytes that mk allocates for one capacity (`harness/engine.py` `context_bytes`).
pub fn context_bytes(capacity: u32) -> u64 {
    // The fixed state and history, then K/V, index keys and the derived pooled keys, for each token of capacity.
    let fixed = 36 * 48 * 128 * 128 + 36 * 3 * 10_240 + 9 * 10_240;
    let per_token = 12 * 2 * 2 * 256 + 12 * 128 + 12 * 128 / 4;
    4 * (fixed + per_token * capacity as u64)
}

/// The parts of one mk state, in mk's part order.
pub fn parts() -> Vec<PartLimits> {
    vec![
        PartLimits {
            name: "gdn_state".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::F32,
            shape: vec![36, 48, 128, 128],
        },
        PartLimits {
            name: "gdn_conv".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::F32,
            shape: vec![36, 3, 10_240],
        },
        PartLimits {
            name: "ple_conv".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::F32,
            shape: vec![9, 10_240],
        },
        PartLimits {
            name: "kv".to_owned(),
            kind: PartKind::Cells,
            dtype: Dtype::F32,
            shape: vec![12, 2, 2, 256],
        },
        PartLimits {
            name: "idx_keys".to_owned(),
            kind: PartKind::Cells,
            dtype: Dtype::F32,
            shape: vec![12, 128],
        },
        PartLimits {
            name: "ple_hist".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::U32,
            shape: vec![2],
        },
        PartLimits {
            name: "idx_pooled".to_owned(),
            kind: PartKind::Derived,
            dtype: Dtype::F32,
            shape: vec![12, 128],
        },
    ]
}

/// The limits of the engine for the fixed context pool.
pub fn limits(arena_bytes: u64) -> Limits {
    Limits {
        // The ABI takes each capability from its accessor. This binding has `parts()` and `seal()` only.
        capabilities: vec![Capability::Parts, Capability::Seal],
        parts: parts(),
        inputs: Inputs::Tokens,
        devices: vec![0],
        stages: vec![Stage {
            device: 0,
            layers: 0..LAYERS,
            head: true,
        }],
        max_segments: MAX_SEGMENTS,
        max_pass_inputs: MAX_PASS_TOKENS,
        max_logit_rows: MAX_SEGMENTS,
        slots: vec![SlotLimit {
            bytes: context_bytes(CONTEXT_CAPACITY),
            count: CONTEXT_COUNT,
        }],
        max_quantum_ms: MAX_QUANTUM_MS,
        memory: MemoryLimits {
            mandatory: vec![
                MandatoryTerm {
                    name: "weights".to_owned(),
                    tier: Tier::Device(0),
                    bytes: arena_bytes,
                    lifetime: Lifetime::Engine,
                    kinds: vec![Kind::Read, Kind::Prefill, Kind::Decode],
                },
                MandatoryTerm {
                    name: "context".to_owned(),
                    tier: Tier::Device(0),
                    bytes: context_bytes(CONTEXT_CAPACITY),
                    lifetime: Lifetime::Context,
                    kinds: vec![Kind::Prefill, Kind::Decode],
                },
            ],
            elastic: Vec::<ElasticTerm>::new(),
        },
        chain: ChainLimits {
            max_depth: 0,
            delta_fraction: 0.0,
        },
        segments_exact: true,
        determinism: Determinism::Always,
        bit_dependence: Vec::new(),
        taps: Vec::new(),
    }
}

/// The identity of the engine and its weights.
pub fn identity(
    shards: &[std::path::PathBuf],
    library: &Path,
    library_sha256: &str,
    fast_qmm: bool,
    checkpoint_digest: &str,
) -> Identity {
    let mut numbers = BTreeMap::new();
    numbers.insert("engine_lib".to_owned(), library.display().to_string());
    numbers.insert("engine_lib_sha256".to_owned(), library_sha256.to_owned());
    numbers.insert("fast_qmm".to_owned(), u8::from(fast_qmm).to_string());
    numbers.insert("fused".to_owned(), "1".to_owned());
    numbers.insert("device".to_owned(), "0".to_owned());
    numbers.insert("shards".to_owned(), shards.len().to_string());
    numbers.insert("max_context".to_owned(), MAX_CONTEXT.to_string());
    numbers.insert("max_pass_tokens".to_owned(), MAX_PASS_TOKENS.to_string());
    numbers.insert("canary".to_owned(), "not run".to_owned());
    Identity {
        model: model_name(shards.first().map(std::path::PathBuf::as_path)),
        checkpoint_digest: format!("spot-sha256:{checkpoint_digest}"),
        tokenizer_hash: "none".to_owned(),
        template: String::new(),
        numbers,
        plan_table: "none".to_owned(),
        bit_settings: vec![
            BitSetting {
                name: "MK_FAST_QMM".to_owned(),
                value: u8::from(fast_qmm).to_string(),
                class: BitClass::Approx,
            },
            BitSetting {
                name: "fused".to_owned(),
                value: "1".to_owned(),
                class: BitClass::Approx,
            },
        ],
        canary_hash: String::new(),
    }
}

/// The model name from a shard file name, without the split suffix.
pub fn model_name(path: Option<&Path>) -> String {
    let Some(path) = path else {
        return String::new();
    };
    let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
        return String::new();
    };
    let Some((left, right)) = stem.rsplit_once("-of-") else {
        return stem.to_owned();
    };
    if right.chars().all(|character| character.is_ascii_digit())
        && let Some((model, shard)) = left.rsplit_once('-')
        && shard.chars().all(|character| character.is_ascii_digit())
    {
        return model.to_owned();
    }
    stem.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_declared_part_shapes_use_the_mk_sizes() {
        // The GPU test pins these constants to mk's own mk_ctx_part_bytes.
        let parts = parts();
        let size = |index: usize| parts[index].shape.iter().product::<u64>() * 4;
        assert_eq!(size(0), GDN_STATE_BYTES);
        assert_eq!(size(1), GDN_CONV_BYTES);
        assert_eq!(size(2), PLE_CONV_BYTES);
        assert_eq!(size(3), KV_ROW_BYTES);
        assert_eq!(size(4), IDX_ROW_BYTES);
        assert_eq!(size(5), PLE_HIST_BYTES);
        assert_eq!(parts.len(), 7);
        assert_eq!(parts[6].kind, PartKind::Derived);
    }

    #[test]
    fn a_context_is_the_fixed_state_and_its_rows() {
        assert_eq!(
            context_bytes(0),
            4 * (36 * 48 * 128 * 128 + 36 * 3 * 10_240 + 9 * 10_240)
        );
        assert_eq!(
            context_bytes(256) - context_bytes(0),
            256 * 4 * (12 * 2 * 2 * 256 + 12 * 128 + 12 * 128 / 4)
        );
    }

    #[test]
    fn the_limits_declare_two_contexts_and_mk_parts() {
        let limits = limits(94);
        let names: Vec<&str> = limits.parts.iter().map(|part| part.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "gdn_state",
                "gdn_conv",
                "ple_conv",
                "kv",
                "idx_keys",
                "ple_hist",
                "idx_pooled"
            ]
        );
        assert_eq!(limits.parts[3].kind, PartKind::Cells);
        assert_eq!(limits.parts[4].kind, PartKind::Cells);
        assert_eq!(limits.parts[6].kind, PartKind::Derived);
        assert_eq!(limits.slots.len(), 1);
        assert_eq!(limits.slots[0].count, 2);
        assert_eq!(limits.slots[0].bytes, context_bytes(256));
        assert_eq!(limits.max_segments, 16);
        assert_eq!(limits.max_quantum_ms, 30_000);
        assert!(
            limits
                .memory
                .mandatory
                .iter()
                .any(|term| term.name == "weights" && term.bytes == 94)
        );
    }

    #[test]
    fn the_limits_declare_only_the_reachable_capabilities() {
        // The ABI takes a capability from its accessor. mk has parts() and seal(), and no accessor for contexts,
        // poison or the pass check, so the limits must not declare those.
        let limits = limits(0);
        assert_eq!(limits.capabilities, vec![Capability::Parts, Capability::Seal]);
    }

    #[test]
    fn the_model_name_drops_the_split_suffix() {
        assert_eq!(
            model_name(Some(Path::new(
                "/tank/models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
            ))),
            "Qwen3.8-Flash-Next-UD-IQ4_XS"
        );
        assert_eq!(model_name(Some(Path::new("plain.gguf"))), "plain");
        assert_eq!(model_name(None), "");
    }
}
