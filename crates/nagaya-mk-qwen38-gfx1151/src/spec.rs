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

/// The longest quantum that mk declares, in milliseconds. mk's watchdog aborts a pass at 10 s (`engine.h:302-306`).
pub const MAX_QUANTUM_MS: u32 = 10_000;

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
            block_format: None,
        },
        PartLimits {
            name: "gdn_conv".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::F32,
            shape: vec![36, 3, 10_240],
            block_format: None,
        },
        PartLimits {
            name: "ple_conv".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::F32,
            shape: vec![9, 10_240],
            block_format: None,
        },
        PartLimits {
            name: "kv".to_owned(),
            kind: PartKind::Cells,
            dtype: Dtype::F32,
            shape: vec![12, 2, 2, 256],
            block_format: None,
        },
        PartLimits {
            name: "idx_keys".to_owned(),
            kind: PartKind::Cells,
            dtype: Dtype::F32,
            shape: vec![12, 128],
            block_format: None,
        },
        PartLimits {
            name: "ple_hist".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::U32,
            shape: vec![2],
            block_format: None,
        },
        PartLimits {
            name: "idx_pooled".to_owned(),
            kind: PartKind::Derived,
            dtype: Dtype::F32,
            shape: vec![12, 128],
            block_format: None,
        },
    ]
}

/// The part index, when mk stores that part. mk's `state_bytes` dies on part 6 (derived) and on any part past 5.
pub fn stored_part(part: u32) -> Result<u32, String> {
    if part < 6 {
        Ok(part)
    } else {
        Err(format!(
            "mk has no stored part {part}; parts 0-5 are stored and part 6 is derived"
        ))
    }
}

/// The floats of one mk scratch buffer for each token of a pass. The names are mk's `scratch()` names.
///
/// mk sizes each buffer at `kMaxPassTokens`, whatever the pass's token count, so this table is the pool at its
/// largest. The sizes come from mk's `src/engine/blocks.hip` and `src/engine/engine.hip` at 555a2a3.
#[cfg(test)]
const SCRATCH_PER_TOKEN: [(&str, u64); 48] = [
    // The gated DeltaNet block (blocks.hip:34-39).
    ("gdn_qkv", 10_240), // kGdnConvDim
    ("gdn_qkvc", 10_240),
    ("gdn_z", 6_144), // kGdnValDim
    ("gdn_a", 48),    // kGdnVHeads
    ("gdn_b", 48),
    ("gdn_core", 6_144),
    // The mixture of experts block (blocks.hip:68-84).
    ("moe_logits", 512), // kExperts
    ("moe_tidx", 10),    // kTopK
    ("moe_tw", 10),
    ("moe_asg", 10),
    ("moe_tok", 10),
    ("moe_gu", 12_800), // kTopK * 2 * kFf
    ("moe_h", 6_400),   // kTopK * kFf
    ("moe_y", 25_600),  // kTopK * kEmbd
    ("moe_sg", 640),    // kFfShared
    ("moe_su", 640),
    ("moe_sh", 640),
    ("moe_sd", 2_560), // kEmbd
    // The attention block (blocks.hip:143-156).
    ("att_q", 12_288), // kHeads * 2 * kHeadDim
    ("att_k", 512),    // kKvHeads * kHeadDim
    ("att_v", 512),
    ("att_iq", 512),   // kIdxHeads * kIdxDim
    ("att_ik", 128),   // kIdxDim
    ("att_qo", 6_144), // kHeads * kHeadDim
    ("att_go", 6_144),
    ("att_iqo", 512),
    ("att_scores", 8_192), // kMaxContext / kIdxRatio
    ("att_picked", 512),   // kIdxBudget / kIdxRatio
    ("att_sel", 2_075),    // kAttnMaxSelMasked
    ("att_seln", 1),
    ("att_out", 6_144),
    // The pass's own rows and the hyper-connection mixer (engine.hip:1203-1212).
    ("e", 2_560),  // kEmbd
    ("h", 10_240), // kHcDim
    ("xn", 10_240),
    ("down", 320), // kHcRank
    ("act", 320),
    ("up", 10_240),
    ("z4", 4),
    ("inj", 4),
    ("mixed", 2_560),
    ("out", 2_560),
    // The PLE block of layer 1 (engine.hip:1254-1260).
    ("ple_emb", 2_560),
    ("ple_k", 10_240),
    ("ple_kn", 10_240),
    ("ple_v", 2_560),
    ("ple_gv", 10_240),
    ("ple_gvn", 10_240),
    ("ple_y", 10_240),
];

/// The mk scratch buffers that do not grow with the token count, in floats.
#[cfg(test)]
const SCRATCH_FIXED: [(&str, u64); 8] = [
    ("moe_cnt", 512), // kExperts
    ("moe_off", 513), // kExperts + 1
    ("moe_igu", 513),
    ("moe_idn", 513),
    ("moe_sgt", 256), // kMaxPassTokens
    // The head's logits row for each segment (engine.hip:1293).
    ("logits", 16 * 248_320),
    // The largest split-K slice (engine.hip:1184): a small pass asks for the second one (engine.hip:1188).
    ("qmm_part", 16 * 256 * 640),
    ("qmm_part2", 16 * 256 * 640),
];

/// The floats that mk's pass scratch asks for, one entry for each named buffer at its largest ask.
#[cfg(test)]
pub fn scratch_buffers() -> Vec<(&'static str, u64)> {
    let tokens = u64::from(MAX_PASS_TOKENS);
    SCRATCH_PER_TOKEN
        .iter()
        .map(|(name, per_token)| (*name, per_token * tokens))
        .chain(SCRATCH_FIXED.iter().copied())
        .collect()
}

/// The bytes of mk's pass scratch pool at its largest: every named buffer, sized for `kMaxPassTokens`.
#[cfg(test)]
pub fn pass_scratch_bytes() -> u64 {
    scratch_buffers().iter().map(|(_, floats)| floats).sum::<u64>() * 4
}

/// The pass scratch that this binding declares for mk on gfx1151, in bytes.
///
/// The source is the measured device cost, not the pool's ask. The pool asks for 263 934 988 bytes: the sum of
/// `scratch_buffers` at `MAX_PASS_TOKENS`. The rest is the HIP allocator's granularity and the runtime's one-time
/// setup for the first pass, which the engine also keeps for its life. The ledger holds the larger number, so it
/// never under-declares the pool.
///
/// Measured on 2026-10-18 on gfx1151, by
/// `cargo run -p nagaya-mk-qwen38-gfx1151 --example pass_scratch --features hip`, in fresh engines. The reading is
/// 323 731 456 or 323 735 552 bytes: one HIP page apart. The declared value is the larger reading rounded up to
/// 309 MiB. The one-page movement is why the value is rounded: an exact maximum would refuse a library whose pool
/// did not change.
pub const PASS_SCRATCH_BYTES: u64 = 324_009_984;

/// The reason that a measured pass scratch is past the declared constant, or `Ok` when it fits.
pub fn scratch_problem(measured: u64) -> Result<(), String> {
    if measured > PASS_SCRATCH_BYTES {
        return Err(format!(
            "the mk pass scratch measures {measured} bytes of device memory; this binding declares \
             {PASS_SCRATCH_BYTES}. Use the mk commit that milestone 9c measured, or measure again and raise the constant"
        ));
    }
    Ok(())
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
                    name: "pass_scratch".to_owned(),
                    tier: Tier::Device(0),
                    bytes: PASS_SCRATCH_BYTES,
                    lifetime: Lifetime::Engine,
                    kinds: vec![Kind::Prefill, Kind::Decode],
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
        checkpoint_grain: None,
    }
}

/// The bytes of one declared part: the whole shape for a fixed part, one row for a cells or ring part.
fn declared_bytes(part: &PartLimits) -> u64 {
    part.shape.iter().product::<u64>() * part.dtype.element_bytes()
}

/// The reason that mk's part byte counts do not match the declared shapes, or `None` when they agree.
///
/// A library with a larger part would write past the Rust buffer, so open must refuse it.
pub fn parts_problem(mk_state_bytes: impl Fn(u32) -> u64, mk_part_bytes: impl Fn(u32, u32) -> u64) -> Option<String> {
    for (index, part) in parts().iter().enumerate() {
        if !part.is_stored() {
            continue;
        }
        let index = index as u32;
        let declared = declared_bytes(part);
        let actual = if part.is_row() {
            mk_part_bytes(index, 1)
        } else {
            mk_state_bytes(index)
        };
        if actual != declared {
            return Some(format!(
                "mk says part {index} ({}) holds {actual} bytes; this binding declares {declared}",
                part.name
            ));
        }
    }
    None
}

/// The identity of the engine and its weights.
pub fn identity(
    shards: &[std::path::PathBuf],
    library: &Path,
    library_sha256: &str,
    fast_qmm: bool,
    tiles: bool,
    checkpoint_digest: &str,
) -> Identity {
    let mut numbers = BTreeMap::new();
    numbers.insert("engine_lib".to_owned(), library.display().to_string());
    numbers.insert("engine_lib_sha256".to_owned(), library_sha256.to_owned());
    numbers.insert("fast_qmm".to_owned(), u8::from(fast_qmm).to_string());
    numbers.insert("tiles".to_owned(), u8::from(tiles).to_string());
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
                name: "MK_TILES".to_owned(),
                value: u8::from(tiles).to_string(),
                // The tiled and the GGUF layouts use different kernels. The binding has not measured whether they
                // give equal bits, so each layout defines its own reference.
                class: BitClass::ReferenceDefining,
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
    use std::path::{Path, PathBuf};

    use super::*;

    #[test]
    fn a_part_bytes_ask_refuses_a_part_that_mk_does_not_store() {
        for part in 0..=5 {
            assert_eq!(stored_part(part), Ok(part), "part {part}");
        }
        // mk's state_bytes dies on part 6, which is derived, and on a part past 5. The binding must not call it.
        assert!(stored_part(6).is_err());
        assert!(stored_part(99).is_err());
    }

    #[test]
    fn a_library_with_another_part_size_is_refused() {
        let fixed = |part: u32| match part {
            0 => GDN_STATE_BYTES,
            1 => GDN_CONV_BYTES,
            2 => PLE_CONV_BYTES,
            5 => PLE_HIST_BYTES,
            _ => unreachable!("part {part} is not fixed"),
        };
        let rows = |part: u32, n: u32| match part {
            3 => u64::from(n) * KV_ROW_BYTES,
            4 => u64::from(n) * IDX_ROW_BYTES,
            _ => unreachable!("part {part} is not a row part"),
        };
        assert_eq!(parts_problem(fixed, rows), None);
        // A library with a larger recurrent state would write past the Rust buffer.
        let larger = |part: u32| {
            if part == 0 { GDN_STATE_BYTES + 4 } else { fixed(part) }
        };
        let problem = parts_problem(larger, rows).expect("a larger part must be a mismatch");
        assert!(problem.contains("gdn_state"), "{problem}");
    }

    #[test]
    fn mk_declares_no_checkpoint_grain() {
        assert_eq!(limits(0).checkpoint_grain, None);
    }

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
        assert_eq!(limits.max_quantum_ms, 10_000);
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
    fn the_identity_records_the_tile_setting() {
        let shards = vec![PathBuf::from("/models/Qwen-00001-of-00003.gguf")];
        let library = Path::new("/lib/libmkengine.so");
        let on = identity(&shards, library, "sha", true, true, "spot");
        let off = identity(&shards, library, "sha", true, false, "spot");
        assert_eq!(on.numbers.get("tiles").map(String::as_str), Some("1"));
        assert_eq!(off.numbers.get("tiles").map(String::as_str), Some("0"));
        assert_ne!(on, off);
        assert!(on.bit_settings.iter().any(|setting| setting.name == "MK_TILES"));
        assert!(off.bit_settings.iter().any(|setting| setting.name == "MK_TILES"));
    }

    #[test]
    fn the_limits_declare_the_pass_scratch_as_a_mandatory_term() {
        let limits = limits(94);
        let term = limits
            .memory
            .mandatory
            .iter()
            .find(|term| term.name == "pass_scratch")
            .expect("the limits must declare the pass scratch (docs/engine-abi.md)");
        assert_eq!(term.tier, Tier::Device(0));
        assert_eq!(term.bytes, PASS_SCRATCH_BYTES);
        assert_eq!(term.lifetime, Lifetime::Engine);
        assert_eq!(term.kinds, vec![Kind::Prefill, Kind::Decode]);
    }

    #[test]
    fn a_measured_scratch_above_the_declared_constant_is_refused() {
        assert_eq!(scratch_problem(PASS_SCRATCH_BYTES - 1), Ok(()));
        assert_eq!(scratch_problem(PASS_SCRATCH_BYTES), Ok(()));
        let above = PASS_SCRATCH_BYTES + 1;
        let error = scratch_problem(above).expect_err("one byte above the constant must refuse");
        assert!(error.contains("pass scratch"), "{error}");
        assert!(error.contains(&PASS_SCRATCH_BYTES.to_string()), "{error}");
        assert!(error.contains(&above.to_string()), "{error}");
    }

    #[test]
    fn the_declared_scratch_holds_the_sum_of_the_mk_scratch_buffers() {
        // The names and the sizes come from mk's src/engine/blocks.hip and src/engine/engine.hip at 555a2a3.
        let buffers = scratch_buffers();
        assert_eq!(buffers.len(), 56, "one entry for each named mk scratch buffer");
        let asked = pass_scratch_bytes();
        assert_eq!(asked, 263_934_988, "the pool's ask at kMaxPassTokens");
        // The declared constant is the measured device cost, which is the ask plus the allocator's granularity.
        assert!(
            PASS_SCRATCH_BYTES >= asked,
            "the declared {} bytes must hold the pool's {asked} bytes",
            PASS_SCRATCH_BYTES
        );
        // The largest single buffer is the MoE block's expert rows: kMaxPassTokens x kTopK x kEmbd floats.
        assert_eq!(buffers.iter().map(|(_, floats)| *floats).max(), Some(6_553_600));
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
