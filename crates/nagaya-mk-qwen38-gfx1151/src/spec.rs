//! The constants and the declared shapes of the Qwen3.8 engine.
//!
//! The numbers come from the reference's `src/ops/model.h` and `src/engine/engine.h`. The engine has one device group and one
//! context pool in this binding (docs/engine-abi.md, "v0 binding choices").

use std::collections::BTreeMap;
use std::path::Path;

use nagaya_abi::{
    BitClass, BitSetting, Capability, ChainLimits, CoverageSpan, Determinism, Dtype, ElasticTerm, Identity, Inputs,
    Kind, Lifetime, Limits, MandatoryTerm, MemoryLimits, PartKind, PartLimits, SlotLimit, Stage, Tier,
};

/// The vocabulary size (the reference `kVocab`).
pub const VOCAB: u32 = 248_320;

/// The most tokens in one context (the reference `kMaxContext`).
pub const MAX_CONTEXT: u32 = 32_768;

/// The most tokens in one pass (the reference `kMaxPassTokens`).
pub const MAX_PASS_TOKENS: u32 = 256;
/// The checkpoint grain in tokens (spec 6.13). On gfx1151, a copy at 512 tokens restored and replayed to the bits of
/// the live context, and splits across 256 and 512 gave equal bits (`tests/hip.rs`).
pub const CHECKPOINT_GRAIN: u64 = 256;

/// A context capacity is a multiple of this many tokens (the reference `check_capacity`).
pub const CAPACITY_STEP: u32 = 256;

/// One class of contexts in the pool: the number of contexts and the capacity of each, in tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextClass {
    /// The number of contexts of this class.
    pub count: u32,
    /// The capacity of each context, in tokens.
    pub capacity: u32,
}

/// The pool that a handle creates at open. The small class comes first, so slot 0 is the cheapest to fill.
///
/// Decided by senpai 2026-10-20 night (vetoable): 12 contexts of 4096 tokens and 4 of 32 768 tokens. A real
/// kaijutsu context needs thousands of tokens. The count of 16 equals `MAX_SEGMENTS`, so a pass can hold one segment
/// for each context (Amy, 2026-10-20: "grow the pool"). The bytes are in `context_bytes`.
pub const CONTEXT_CLASSES: [ContextClass; 2] = [
    ContextClass {
        count: 12,
        capacity: 4096,
    },
    ContextClass {
        count: 4,
        capacity: MAX_CONTEXT,
    },
];

/// The capacity of the context that the pass scratch probe uses, in tokens. The probe runs 256 tokens in two passes.
/// The pool does not grow with a context's capacity (the engine sizes each buffer at `kMaxPassTokens`).
pub const PROBE_CAPACITY: u32 = CAPACITY_STEP;

/// The device bytes that the allocator adds to each context, over the state bytes of `context_bytes`.
///
/// `examples/context_pool.rs` measured it on gfx1151 on 2026-10-20 night. 12 contexts of 4096 tokens took 20 086 784
/// bytes more than their state bytes, which is 1 673 899 for each. 4 contexts of 32 768 tokens took 8 093 696 bytes
/// more, which is 2 023 424 for each. The declared value is 4 MiB, twice the larger reading. Thus a page of drift
/// does not refuse a sound library.
pub const CONTEXT_ALLOCATION_BYTES: u64 = 4 << 20;

/// The bytes that the ledger holds for one context of this capacity: its state bytes and the allocator's bytes.
pub fn context_declared_bytes(capacity: u32) -> u64 {
    context_bytes(capacity) + CONTEXT_ALLOCATION_BYTES
}

/// The bytes that the ledger holds for the whole declared pool.
pub fn pool_bytes() -> u64 {
    CONTEXT_CLASSES
        .iter()
        .map(|class| u64::from(class.count) * context_declared_bytes(class.capacity))
        .sum()
}

/// The reason that a class took more device memory than this binding declares for it, or `Ok` when it fits.
pub fn class_problem(class: ContextClass, measured: u64) -> Result<(), String> {
    let declared = u64::from(class.count) * context_declared_bytes(class.capacity);
    if measured > declared {
        return Err(format!(
            "{} contexts of {} tokens took {measured} bytes of device memory; this binding declares {declared}. \
             Use the build of the engine that the pool was measured on, or measure again and raise the constant",
            class.count, class.capacity
        ));
    }
    Ok(())
}

/// The longest quantum that the engine declares, in milliseconds. The watchdog of the engine aborts a pass at 10 s (`engine.h:302-306`).
pub const MAX_QUANTUM_MS: u32 = 10_000;

/// The most segments in one pass (the reference `kMaxSegments`).
pub const MAX_SEGMENTS: u32 = 16;

/// The number of layers (the reference `kLayers`).
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

/// The context bytes that the reference allocates for one capacity (`harness/engine.py` `context_bytes`).
pub fn context_bytes(capacity: u32) -> u64 {
    // The fixed state and history, then K/V, index keys and the derived pooled keys, for each token of capacity.
    let fixed = 36 * 48 * 128 * 128 + 36 * 3 * 10_240 + 9 * 10_240;
    let per_token = 12 * 2 * 2 * 256 + 12 * 128 + 12 * 128 / 4;
    4 * (fixed + per_token * capacity as u64)
}

/// The parts of one engine state, in the order of the engine.
pub fn parts() -> Vec<PartLimits> {
    vec![
        PartLimits {
            name: "gdn_state".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::F32,
            shape: vec![36, 48, 128, 128],
            block_format: None,
            stage: 0,
        },
        PartLimits {
            name: "gdn_conv".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::F32,
            shape: vec![36, 3, 10_240],
            block_format: None,
            stage: 0,
        },
        PartLimits {
            name: "ple_conv".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::F32,
            shape: vec![9, 10_240],
            block_format: None,
            stage: 0,
        },
        PartLimits {
            name: "kv".to_owned(),
            kind: PartKind::Cells,
            dtype: Dtype::F32,
            shape: vec![12, 2, 2, 256],
            block_format: None,
            stage: 0,
        },
        PartLimits {
            name: "idx_keys".to_owned(),
            kind: PartKind::Cells,
            dtype: Dtype::F32,
            shape: vec![12, 128],
            block_format: None,
            stage: 0,
        },
        PartLimits {
            name: "ple_hist".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::U32,
            shape: vec![2],
            block_format: None,
            stage: 0,
        },
        PartLimits {
            name: "idx_pooled".to_owned(),
            kind: PartKind::Derived,
            dtype: Dtype::F32,
            shape: vec![12, 128],
            block_format: None,
            stage: 0,
        },
    ]
}

/// The part index, when the engine stores that part. The `state_bytes` call of the engine dies on part 6 (derived) and on any part past 5.
pub fn stored_part(part: u32) -> Result<u32, String> {
    if part < 6 {
        Ok(part)
    } else {
        Err(format!(
            "the Qwen3.8 engine has no stored part {part}; parts 0-5 are stored and part 6 is derived; use a part from 0 to 5"
        ))
    }
}

/// The capacity of one slot in tokens. The slots follow `CONTEXT_CLASSES` in order, so the small class comes first.
pub fn slot_capacity(slot: u32) -> Option<u32> {
    let mut first = 0u32;
    for class in CONTEXT_CLASSES {
        if slot < first + class.count {
            return Some(class.capacity);
        }
        first += class.count;
    }
    None
}

/// The bytes of one row of a cells part: part 3 (`kv`) or part 4 (`idx_keys`).
pub fn row_bytes(part: usize) -> Option<u64> {
    match part {
        3 => Some(KV_ROW_BYTES),
        4 => Some(IDX_ROW_BYTES),
        _ => None,
    }
}

/// The positions that `len` bytes of one part hold. The engine holds the rows of a cells part from position 0.
///
/// A fixed or derived part holds no position. Bytes that are not whole rows are a defect of the caller, so this
/// panics (spec 8.5).
pub fn coverage_of(part: usize, len: usize) -> CoverageSpan {
    let Some(row) = row_bytes(part) else {
        return CoverageSpan { part, start: 0, end: 0 };
    };
    let len = len as u64;
    assert!(
        len.is_multiple_of(row),
        "{len} bytes of part {part} are not a whole number of rows of {row} bytes"
    );
    CoverageSpan {
        part,
        start: 0,
        end: len / row,
    }
}

/// Ok when a slot can hold the span: the engine writes the rows of a cells part from position 0, up to the capacity.
pub fn fits(span: &CoverageSpan, slot: u32) -> Result<(), String> {
    if row_bytes(span.part).is_none() {
        return Err(format!(
            "part {} is not a cells part; only parts 3 and 4 hold rows",
            span.part
        ));
    }
    let capacity = slot_capacity(slot).ok_or_else(|| format!("slot {slot} is not in the context pool"))?;
    if span.start != 0 {
        return Err(format!(
            "the span of part {} starts at position {}, but the Qwen3.8 engine writes rows from position 0; start the span at 0",
            span.part, span.start
        ));
    }
    if span.end > u64::from(capacity) {
        return Err(format!(
            "the span of part {} holds {} rows; slot {slot} holds {capacity} tokens",
            span.part, span.end
        ));
    }
    Ok(())
}

/// Ok when a cells read asks for a whole cells part. The engine reads a part whole, so a read from 0 of all rows is the only
/// range that it gives. The worker checks that `n` is the context's token count.
pub fn whole_read_problem(part: usize, from: u64, n: u64, len: usize) -> Result<(), String> {
    let row = row_bytes(part).ok_or_else(|| format!("part {part} is not a cells part; read it with part_read"))?;
    if from != 0 {
        return Err(format!(
            "the Qwen3.8 engine has no range read: a cells read starts from 0, and this one starts at {from}; read from 0"
        ));
    }
    if n.checked_mul(row) != Some(len as u64) {
        return Err(format!(
            "{n} rows of part {part} are {} bytes; the buffer holds {len} bytes",
            n.saturating_mul(row)
        ));
    }
    Ok(())
}

/// The floats of one scratch buffer for each token of a pass. The names are the `scratch()` names of the reference.
///
/// The engine sizes each buffer at `kMaxPassTokens`, whatever the pass's token count, so this table is the pool at its
/// largest. The sizes come from the reference's `src/engine/blocks.hip` and `src/engine/engine.hip` at 555a2a3.
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

/// The scratch buffers that do not grow with the token count, in floats.
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

/// The floats that the pass scratch asks for, one entry for each named buffer at its largest ask.
#[cfg(test)]
pub fn scratch_buffers() -> Vec<(&'static str, u64)> {
    let tokens = u64::from(MAX_PASS_TOKENS);
    SCRATCH_PER_TOKEN
        .iter()
        .map(|(name, per_token)| (*name, per_token * tokens))
        .chain(SCRATCH_FIXED.iter().copied())
        .collect()
}

/// The bytes of the pass scratch pool at its largest: every named buffer, sized for `kMaxPassTokens`.
#[cfg(test)]
pub fn pass_scratch_bytes() -> u64 {
    scratch_buffers().iter().map(|(_, floats)| floats).sum::<u64>() * 4
}

/// The pass scratch that this binding declares for the engine on gfx1151, in bytes.
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
            "the pass scratch of the Qwen3.8 engine measures {measured} bytes of device memory; this binding declares \
             {PASS_SCRATCH_BYTES}. Use the build of the engine that milestone 9c measured, or measure again and raise the constant"
        ));
    }
    Ok(())
}

/// The limits of the engine for the fixed context pool.
pub fn limits(arena_bytes: u64) -> Limits {
    Limits {
        // The ABI takes each capability from its accessor. This binding has `parts()`, `coverage()`, `seal()` and
        // `slots()`. `close` is the drop of the last handle, which frees the weights (engine-abi "close").
        capabilities: vec![
            Capability::Parts,
            Capability::Coverage,
            Capability::Seal,
            Capability::Slots,
            Capability::Close,
        ],
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
        slots: CONTEXT_CLASSES
            .iter()
            .map(|class| SlotLimit {
                bytes: context_bytes(class.capacity),
                count: class.count,
                tokens: u64::from(class.capacity),
            })
            .collect(),
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
                // The engine makes the whole pool at open, so the term holds every context and lives as long as the engine.
                MandatoryTerm {
                    name: "context".to_owned(),
                    tier: Tier::Device(0),
                    bytes: pool_bytes(),
                    lifetime: Lifetime::Engine,
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
        checkpoint_grain: Some(CHECKPOINT_GRAIN),
        feed: None,
        residual_row_bytes: 0,
        control: None,
    }
}

/// The bytes of one declared part: the whole shape for a fixed part, one row for a cells or ring part.
fn declared_bytes(part: &PartLimits) -> u64 {
    part.shape.iter().product::<u64>() * part.dtype.element_bytes()
}

/// The reason that the part byte counts of the engine do not match the declared shapes, or `None` when they agree.
///
/// A library with a larger part would write past the Rust buffer, so open must refuse it.
pub fn parts_problem(
    engine_state_bytes: impl Fn(u32) -> u64,
    engine_part_bytes: impl Fn(u32, u32) -> u64,
) -> Option<String> {
    for (index, part) in parts().iter().enumerate() {
        if !part.is_stored() {
            continue;
        }
        let index = index as u32;
        let declared = declared_bytes(part);
        let actual = if part.is_row() {
            engine_part_bytes(index, 1)
        } else {
            engine_state_bytes(index)
        };
        if actual != declared {
            return Some(format!(
                "the Qwen3.8 engine says part {index} ({}) holds {actual} bytes, but this binding declares {declared}; build the engine and the binding from one source",
                part.name
            ));
        }
    }
    None
}

/// The identity of the engine and its weights.
pub(crate) fn identity(
    shards: &[std::path::PathBuf],
    build: &crate::binding::Build,
    fast_qmm: bool,
    tiles: bool,
    checkpoint_digest: &str,
    text: &crate::gguf::TextKeys,
) -> Identity {
    let mut numbers = BTreeMap::new();
    // The build facts stay in the numbers, and thus in the identity digest, until the canary runs at each start
    // (spec 12.1). A file path is not a number, so no path is here.
    numbers.insert("engine_build_sha256".to_owned(), build.library_sha256.to_owned());
    numbers.insert("engine_compiler".to_owned(), build.compiler.to_owned());
    numbers.insert("engine_hip".to_owned(), build.hip.to_owned());
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
        family: "qwen38".to_owned(),
        checkpoint_digest: format!("spot-sha256:{checkpoint_digest}"),
        tokenizer_hash: text.tokenizer_hash.clone(),
        template: text.template.clone(),
        numbers,
        plan_table: "none".to_owned(),
        bit_settings: vec![
            BitSetting {
                name: "QWEN38_FAST_QMM".to_owned(),
                value: u8::from(fast_qmm).to_string(),
                class: BitClass::Approx,
            },
            BitSetting {
                name: "QWEN38_TILES".to_owned(),
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
    fn a_part_bytes_ask_refuses_a_part_that_the_engine_does_not_store() {
        for part in 0..=5 {
            assert_eq!(stored_part(part), Ok(part), "part {part}");
        }
        // The `state_bytes` call of the engine dies on part 6, which is derived, and on a part past 5. The binding must not call it.
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
    fn the_engine_declares_a_grain_of_256_and_segments_exact() {
        // Spec 6.13: a grain needs `segments_exact`. The GPU tests `an_interval_copy_restores_and_replays_to_the_bits_of_
        // the_live_context` and `a_split_across_256_does_not_change_the_bits_in_either_class` measured both.
        let limits = limits(0);
        assert_eq!(limits.checkpoint_grain, Some(256));
        assert!(limits.segments_exact);
        assert_eq!(
            nagaya_sched::IntervalPolicy::from_limits(&limits, 1).map(|policy| policy.map(|p| p.grain())),
            Ok(Some(256))
        );
    }

    #[test]
    fn the_declared_part_shapes_use_the_sizes_of_the_engine() {
        // The GPU test pins these constants to the part byte counts that the engine library reports.
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
    fn a_context_of_each_class_has_the_bytes_that_the_arithmetic_gives() {
        // fixed = 4 * (36*48*128*128 + 36*3*10240 + 9*10240) = 4 * 29 509 632 = 118 038 528 bytes.
        // per token = 4 * (12*2*2*256 + 12*128 + 12*128/4) = 4 * 14 208 = 56 832 bytes.
        assert_eq!(context_bytes(0), 118_038_528);
        // 4096 tokens: 118 038 528 + 4096 * 56 832 = 118 038 528 + 232 783 872.
        assert_eq!(context_bytes(4096), 350_822_400);
        // 32 768 tokens: 118 038 528 + 32 768 * 56 832 = 118 038 528 + 1 862 270 976.
        assert_eq!(context_bytes(32_768), 1_980_309_504);
    }

    #[test]
    fn the_limits_declare_two_context_sizes_and_the_engine_parts() {
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
        // The small class comes first, as in the fake engine. Slots 0-11 hold 4096 tokens; slots 12-15 hold 32 768.
        assert_eq!(
            limits.slots,
            vec![
                SlotLimit {
                    bytes: 350_822_400,
                    count: 12,
                    tokens: 4096
                },
                SlotLimit {
                    bytes: 1_980_309_504,
                    count: 4,
                    tokens: 32_768
                },
            ]
        );
        assert_eq!(
            limits.slots.iter().map(|class| class.count).sum::<u32>(),
            limits.max_segments,
            "a pass can hold one segment for each context"
        );
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
    fn the_context_term_holds_the_whole_pool() {
        // The engine makes every context of the pool at open, and the ledger adds the bytes of each term once.
        // The state bytes are 12 * 350 822 400 + 4 * 1 980 309 504 = 4 209 868 800 + 7 921 238 016. Each of the 16
        // contexts adds 4 MiB for the allocator: 16 * 4 194 304 = 67 108 864.
        let limits = limits(0);
        let term = limits
            .memory
            .mandatory
            .iter()
            .find(|term| term.name == "context")
            .expect("a context term");
        assert_eq!(term.bytes, 12_131_106_816 + 67_108_864);
        assert_eq!(term.lifetime, Lifetime::Engine);
        assert_eq!(
            term.bytes,
            limits
                .slots
                .iter()
                .map(|c| (c.bytes + CONTEXT_ALLOCATION_BYTES) * u64::from(c.count))
                .sum::<u64>()
        );
    }

    #[test]
    fn a_class_that_takes_more_than_it_declares_is_refused() {
        // The measured device cost on gfx1151, 2026-10-20 night: 4 229 955 584 bytes for 12 contexts of 4096 tokens,
        // and 7 929 331 712 bytes for 4 contexts of 32 768. Each is under its declared bytes.
        let small = CONTEXT_CLASSES[0];
        let large = CONTEXT_CLASSES[1];
        assert_eq!(class_problem(small, 4_229_955_584), Ok(()));
        assert_eq!(class_problem(large, 7_929_331_712), Ok(()));
        // 12 * (350 822 400 + 4 194 304) = 4 260 200 448.
        let declared = u64::from(small.count) * context_declared_bytes(small.capacity);
        assert_eq!(declared, 4_260_200_448);
        assert_eq!(class_problem(small, declared), Ok(()));
        let error = class_problem(small, declared + 1).expect_err("one byte over the declared bytes must refuse");
        assert!(error.contains("12 contexts of 4096 tokens"), "{error}");
        assert!(error.contains(&(declared + 1).to_string()), "{error}");
        assert!(error.contains(&declared.to_string()), "{error}");
    }

    #[test]
    fn the_limits_declare_only_the_reachable_capabilities() {
        // The ABI takes a capability from its accessor. The engine has parts(), coverage(), seal() and slots(), and no
        // accessor for contexts, poison or the pass check, so the limits must not declare those. The drop of the last
        // handle frees the weights, so the engine declares close.
        let limits = limits(0);
        assert_eq!(
            limits.capabilities,
            vec![
                Capability::Parts,
                Capability::Coverage,
                Capability::Seal,
                Capability::Slots,
                Capability::Close
            ]
        );
    }

    #[test]
    fn each_slot_has_the_capacity_of_its_class() {
        for slot in 0..12 {
            assert_eq!(slot_capacity(slot), Some(4096), "slot {slot}");
        }
        for slot in 12..16 {
            assert_eq!(slot_capacity(slot), Some(32_768), "slot {slot}");
        }
        assert_eq!(slot_capacity(16), None);
    }

    #[test]
    fn the_coverage_of_a_cells_part_is_its_rows_from_zero() {
        let kv = CoverageSpan {
            part: 3,
            start: 0,
            end: 2,
        };
        assert_eq!(coverage_of(3, 2 * KV_ROW_BYTES as usize), kv);
        let keys = CoverageSpan {
            part: 4,
            start: 0,
            end: 5,
        };
        assert_eq!(coverage_of(4, 5 * IDX_ROW_BYTES as usize), keys);
        assert_eq!(
            coverage_of(4, 0),
            CoverageSpan {
                part: 4,
                start: 0,
                end: 0
            }
        );
        // A fixed part holds no position.
        assert_eq!(
            coverage_of(0, 64),
            CoverageSpan {
                part: 0,
                start: 0,
                end: 0
            }
        );
    }

    #[test]
    #[should_panic(expected = "not a whole number of rows")]
    fn the_coverage_of_a_partial_row_is_a_defect() {
        coverage_of(3, KV_ROW_BYTES as usize + 1);
    }

    #[test]
    fn a_span_fits_a_slot_that_holds_its_rows() {
        let span = |part, start, end| CoverageSpan { part, start, end };
        assert_eq!(fits(&span(3, 0, 4096), 0), Ok(()));
        assert_eq!(fits(&span(4, 0, 4096), 11), Ok(()));
        let error = fits(&span(3, 0, 4097), 0).unwrap_err();
        assert!(error.contains("4096"), "{error}");
        assert_eq!(fits(&span(3, 0, 4097), 12), Ok(()));
        assert_eq!(fits(&span(3, 0, 32_768), 15), Ok(()));
        assert!(fits(&span(3, 0, 32_769), 15).is_err());
        let error = fits(&span(3, 1, 3), 0).unwrap_err();
        assert!(error.contains("position 0"), "{error}");
        assert!(fits(&span(0, 0, 1), 0).is_err(), "a fixed part has no rows");
        assert!(fits(&span(3, 0, 1), 16).is_err(), "slot 16 is not in the pool");
    }

    #[test]
    fn a_cells_read_takes_only_the_whole_part() {
        assert_eq!(whole_read_problem(3, 0, 2, 2 * KV_ROW_BYTES as usize), Ok(()));
        assert_eq!(whole_read_problem(4, 0, 2, 2 * IDX_ROW_BYTES as usize), Ok(()));
        assert_eq!(whole_read_problem(3, 0, 0, 0), Ok(()));
        let error = whole_read_problem(3, 1, 1, KV_ROW_BYTES as usize).unwrap_err();
        assert!(error.contains("from 0"), "{error}");
        let error = whole_read_problem(3, 0, 2, KV_ROW_BYTES as usize).unwrap_err();
        assert!(error.contains("bytes"), "{error}");
        let error = whole_read_problem(0, 0, 1, 64).unwrap_err();
        assert!(error.contains("not a cells part"), "{error}");
    }

    fn keys(tokenizer_hash: &str, template: &str) -> crate::gguf::TextKeys {
        crate::gguf::TextKeys {
            tokenizer_hash: tokenizer_hash.to_owned(),
            template: template.to_owned(),
        }
    }

    fn build(library_sha256: &'static str) -> crate::binding::Build {
        crate::binding::Build {
            library_sha256,
            compiler: "AMD clang version 22.0.0git",
            hip: "7.2.53211-9999",
        }
    }

    #[test]
    fn the_identity_holds_the_build_facts_and_no_file_path() {
        let shards = vec![PathBuf::from("/models/Qwen-00001-of-00003.gguf")];
        let one = identity(&shards, &build("aa"), true, true, "spot", &keys("sha256:t", "t"));
        let number = |key: &str| one.numbers.get(key).map(String::as_str);
        assert_eq!(number("engine_build_sha256"), Some("aa"));
        assert_eq!(number("engine_compiler"), Some("AMD clang version 22.0.0git"));
        assert_eq!(number("engine_hip"), Some("7.2.53211-9999"));
        // A file path is not a number (spec 12.1).
        assert_eq!(number("engine_lib"), None);
        assert!(
            one.numbers.values().all(|value| !value.starts_with('/')),
            "{:?}",
            one.numbers
        );
        // Until the canary runs at each start, the build facts stay in the identity digest (spec 12.1).
        let other = identity(&shards, &build("bb"), true, true, "spot", &keys("sha256:t", "t"));
        assert_ne!(one, other);
    }

    #[test]
    fn the_identity_carries_the_template() {
        let shards = vec![PathBuf::from("/models/Qwen-00001-of-00003.gguf")];
        let facts = &build("sha");
        let first = identity(&shards, facts, true, true, "spot", &keys("sha256:t", "one"));
        let second = identity(&shards, facts, true, true, "spot", &keys("sha256:t", "two"));
        assert_eq!(first.template, "one");
        assert_ne!(first, second);
    }

    #[test]
    fn the_identity_records_the_tile_setting() {
        let shards = vec![PathBuf::from("/models/Qwen-00001-of-00003.gguf")];
        let facts = &build("sha");
        let on = identity(&shards, facts, true, true, "spot", &keys("sha256:t", "t"));
        let off = identity(&shards, facts, true, false, "spot", &keys("sha256:t", "t"));
        assert_eq!(on.numbers.get("tiles").map(String::as_str), Some("1"));
        assert_eq!(off.numbers.get("tiles").map(String::as_str), Some("0"));
        assert_ne!(on, off);
        assert!(on.bit_settings.iter().any(|setting| setting.name == "QWEN38_TILES"));
        assert!(off.bit_settings.iter().any(|setting| setting.name == "QWEN38_TILES"));
    }

    #[test]
    fn the_identity_carries_the_tokenizer_hash() {
        let shards = vec![PathBuf::from("/models/Qwen-00001-of-00003.gguf")];
        let facts = &build("sha");
        let first = identity(&shards, facts, true, true, "spot", &keys("sha256:one", "t"));
        let second = identity(&shards, facts, true, true, "spot", &keys("sha256:two", "t"));
        assert_eq!(first.tokenizer_hash, "sha256:one");
        assert_ne!(first, second);
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
    fn the_declared_scratch_holds_the_sum_of_the_scratch_buffers() {
        // The names and the sizes come from the reference's src/engine/blocks.hip and src/engine/engine.hip at 555a2a3.
        let buffers = scratch_buffers();
        assert_eq!(buffers.len(), 56, "one entry for each named scratch buffer");
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
