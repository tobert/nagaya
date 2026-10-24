//! The limits, the parts and the rules of the Gemma 4 engine that need no card (engine-abi "How mkviolet maps").
//!
//! `plan` reads `config.json` and the tensor headers of the checkpoint. It reads no weight bytes. The checks of a
//! pass, of a part call and of a fit are plain functions here, so the host tests cover them.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::PathBuf;

use nagaya_abi::{
    ArenaPart, Capability, ChainLimits, CoverageSpan, Determinism, Dtype, FeedRule, Inputs, Kind, Lifetime, Limits,
    MandatoryTerm, MemoryLimits, PartId, PartKind, PartLimits, RESIDUAL_TERM, Segment, SegmentInputs, SlotLimit, Stage,
    State, Tier,
};

use crate::gemma4::{
    CARD_LAYERS, HIDDEN, KV_RING_ROWS, MAX_LOGIT_ROWS, NUM_LAYERS, SLIDING_WINDOW, VOCAB, check_config, kv_can_rewind,
    kv_ring, layer_geom, slot_device_bytes, weights_device_bytes, work_device_bytes, work_host_bytes,
};
use crate::protocol::MAX_PLAN_TOKENS;
use crate::sharding::Checkpoint;

/// The engine kind in a placement profile (AGENTS "Layout": kind, model, target).
pub const KIND: &str = "mk-gemma4-bmg";

/// The model name of the identity. Clients of mkviolet use it (`council_server.rs:39`).
pub const MODEL: &str = "gemma-4-26b-a4b";

/// The family library that the engine needs (spec 11.3).
pub const FAMILY: &str = "gemma4";

/// The feed block when the config names none: mkviolet's `MAX_BLOCK` (`main.rs:35`).
pub const DEFAULT_MAX_BLOCK: usize = 512;

/// The most positions in one slot. The measured `MAX_QUANTUM_MS` holds up to this size.
pub const MAX_SLOT_TOKENS: usize = 65_536;

/// The longest stage quantum, in milliseconds (D8). See `MAX_QUANTUM_MS` in engine-abi "mkviolet's limits".
pub const MAX_QUANTUM_MS: u32 = 2_000;

/// The bytes of one residual row between the stages: `HIDDEN` f32 values.
pub const RESIDUAL_ROW_BYTES: u64 = (HIDDEN * 4) as u64;

/// The most segments in a pass. A task ring runs one block of one slot.
pub const MAX_SEGMENTS: u32 = 1;

/// The bytes of the `meta` of one token: four little-endian u32 values (position, gap start, gap end, 0).
pub const META_BYTES: usize = 16;

/// The part name of the head slab in the arena: the final norm and the embedding table (spec 10.2).
pub const HEAD_PART: &str = "head";

/// The part name of the slab of layer `l` in the arena: `w` and two digits (spec 10.2).
pub fn layer_part(l: usize) -> String {
    format!("w{l:02}")
}

/// The parts of the arena: the slab of each layer on the card of its stage, and the head slab on the second card.
pub fn arena_parts() -> Vec<ArenaPart> {
    let mut parts = Vec::new();
    for (card, layers) in CARD_LAYERS.iter().enumerate() {
        parts.extend(layers.clone().map(|l| ArenaPart {
            device: card,
            part: layer_part(l),
        }));
    }
    parts.push(ArenaPart {
        device: 1,
        part: HEAD_PART.to_owned(),
    });
    parts
}

/// The config of one engine. The factory of the process reads it from the profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// The checkpoint directory: `config.json`, the safetensors shards and `tokenizer.json`.
    pub model_dir: PathBuf,
    /// The positions of each slot, in slot order.
    pub slots: Vec<usize>,
    /// The tokens of one block, and the feed block (engine-abi D11).
    pub max_block: usize,
}

impl Config {
    /// Check the sizes. The message names the value and the rule.
    pub fn check(&self) -> Result<(), String> {
        if self.max_block == 0 || self.max_block > MAX_PLAN_TOKENS {
            return Err(format!(
                "max_block is {}; it must be 1 to {MAX_PLAN_TOKENS} tokens",
                self.max_block
            ));
        }
        if self.slots.is_empty() {
            return Err("the config names no slot; give `slots`, the positions of each slot".to_owned());
        }
        if let Some(seq) = self.slots.iter().find(|seq| **seq == 0 || **seq > MAX_SLOT_TOKENS) {
            return Err(format!(
                "a slot of {seq} positions is outside 1 to {MAX_SLOT_TOKENS}; the measured bound of a quantum holds \
                 up to {MAX_SLOT_TOKENS}"
            ));
        }
        Ok(())
    }

    /// The largest slot, in positions.
    pub fn max_seq(&self) -> usize {
        self.slots.iter().copied().max().unwrap_or(0)
    }
}

/// The stage of layer `l`: 0 on card 0, 1 on card 1 (`CARD_LAYERS`).
pub fn stage_of_layer(l: usize) -> usize {
    if CARD_LAYERS[0].contains(&l) { 0 } else { 1 }
}

/// The part id of the K plane (`value` false) or the V plane of layer `l`. Parts go in layer order, K before V.
pub fn part_of(l: usize, value: bool) -> PartId {
    2 * l + usize::from(value)
}

/// The layer of a part, and true for a V plane. `None` for an id past the parts.
pub fn layer_of(part: PartId) -> Option<(usize, bool)> {
    (part < 2 * NUM_LAYERS).then_some((part / 2, part % 2 == 1))
}

/// The f32 values of one row of layer `l`'s K or V plane.
pub fn row_floats(l: usize) -> usize {
    layer_geom(l).kv_heads * layer_geom(l).head_dim
}

/// The parts of a state (engine-abi "mkviolet's parts"): `k.<ll>` and `v.<ll>` for each layer.
pub fn parts() -> Vec<PartLimits> {
    (0..NUM_LAYERS)
        .flat_map(|l| {
            let g = layer_geom(l);
            [false, true].map(|value| PartLimits {
                name: format!("{}.{l:02}", if value { "v" } else { "k" }),
                kind: if g.global { PartKind::Cells } else { PartKind::Ring },
                dtype: Dtype::F32,
                shape: vec![g.kv_heads as u64, g.head_dim as u64],
                block_format: None,
                stage: stage_of_layer(l) as u32,
            })
        })
        .collect()
}

/// The slot classes of the config: each run of equal sizes is one class.
pub fn slot_limits(slots: &[usize]) -> Vec<SlotLimit> {
    let mut classes: Vec<SlotLimit> = Vec::new();
    for &seq in slots {
        let bytes = slot_device_bytes(seq, 0..NUM_LAYERS);
        match classes.last_mut() {
            Some(class) if class.tokens == seq as u64 => class.count += 1,
            _ => classes.push(SlotLimit {
                bytes,
                count: 1,
                tokens: seq as u64,
            }),
        }
    }
    classes
}

fn term(name: &str, tier: Tier, bytes: u64) -> MandatoryTerm {
    MandatoryTerm {
        name: name.to_owned(),
        tier,
        bytes,
        lifetime: Lifetime::Engine,
        kinds: vec![Kind::Prefill, Kind::Decode, Kind::Read, Kind::Copy],
    }
}

/// The limits for a config, with the device bytes of the weights on each card.
pub fn limits(config: &Config, weights: [u64; 2]) -> Limits {
    let block = config.max_block;
    let max_seq = config.max_seq();
    let mut mandatory = Vec::new();
    for (card, layers) in CARD_LAYERS.iter().enumerate() {
        let tier = Tier::Device(card as u32);
        let head = card == 1;
        mandatory.push(term(&format!("weights.{card}"), tier, weights[card]));
        let kv: u64 = config
            .slots
            .iter()
            .map(|seq| slot_device_bytes(*seq, layers.clone()))
            .sum();
        mandatory.push(term(&format!("kv.{card}"), tier, kv));
        mandatory.push(term(
            &format!("work.{card}"),
            tier,
            work_device_bytes(block, max_seq, head) + cluster_device_bytes(),
        ));
        mandatory.push(term(
            &format!("pinned.{card}"),
            Tier::Host,
            work_host_bytes(layers.len(), block, head),
        ));
    }
    // The host looks up each embedding row in the mapped checkpoint (`write_embeddings`), so the table stays in memory.
    mandatory.push(term("embed", Tier::Host, (VOCAB * HIDDEN * 2) as u64));
    let stages = vec![
        Stage {
            device: 0,
            layers: CARD_LAYERS[0].start as u32..CARD_LAYERS[0].end as u32,
            head: false,
        },
        Stage {
            device: 1,
            layers: CARD_LAYERS[1].start as u32..CARD_LAYERS[1].end as u32,
            head: true,
        },
    ];
    let mut limits = Limits {
        capabilities: vec![
            Capability::Parts,
            Capability::Slots,
            Capability::Coverage,
            Capability::Stages,
            Capability::Arena,
            // A drop frees the weights, the shards and each card's context (`tests/xe_close.rs`).
            Capability::Close,
        ],
        parts: parts(),
        inputs: Inputs::Tokens,
        devices: vec![0, 1],
        stages,
        max_segments: MAX_SEGMENTS,
        max_pass_inputs: block as u32,
        max_logit_rows: MAX_LOGIT_ROWS as u32,
        slots: slot_limits(&config.slots),
        max_quantum_ms: MAX_QUANTUM_MS,
        memory: MemoryLimits {
            mandatory,
            elastic: Vec::new(),
        },
        chain: ChainLimits {
            max_depth: 0,
            delta_fraction: 0.0,
        },
        segments_exact: false,
        determinism: Determinism::Always,
        bit_dependence: Vec::new(),
        taps: Vec::new(),
        checkpoint_grain: None,
        residual_row_bytes: RESIDUAL_ROW_BYTES,
        feed: Some(FeedRule { block: block as u64 }),
    };
    let residual = limits.residual_bytes();
    limits.memory.mandatory.push(term(RESIDUAL_TERM, Tier::Host, residual));
    limits
}

/// The device bytes that `L0Cluster::init` takes on each card: the wake buffer and the device copy of the ring.
fn cluster_device_bytes() -> u64 {
    (128 + crate::protocol::MAX_TASKS_PER_CARD * 64) as u64
}

/// The limits of a config, from `config.json` and the tensor headers. It reads no weight bytes (spec 2.4).
pub fn plan(config: &Config) -> Result<Limits, String> {
    config.check()?;
    check_config(&config.model_dir)?;
    let ckpt = Checkpoint::open_dir(&config.model_dir)?;
    let weights = [
        weights_device_bytes(&ckpt, CARD_LAYERS[0].clone(), false)?,
        weights_device_bytes(&ckpt, CARD_LAYERS[1].clone(), true)?,
    ];
    Ok(limits(config, weights))
}

/// The index of the card at each wanted PCI address, in the order of `wanted`. `cards` holds the address of each card.
pub fn match_addresses(cards: &[String], wanted: &[String]) -> Result<Vec<usize>, String> {
    let mut found = Vec::with_capacity(wanted.len());
    for (index, address) in wanted.iter().enumerate() {
        let address = address.to_ascii_lowercase();
        if wanted[..index].iter().any(|other| other.eq_ignore_ascii_case(&address)) {
            return Err(format!("the profile names the PCI address {address} twice"));
        }
        let matches: Vec<usize> = (0..cards.len()).filter(|card| cards[*card] == address).collect();
        match matches[..] {
            [card] => found.push(card),
            [] => {
                return Err(format!(
                    "no Level Zero card has the PCI address {address}; the cards are at {}",
                    cards.join(", ")
                ));
            }
            _ => return Err(format!("two Level Zero cards give the PCI address {address}")),
        }
    }
    Ok(found)
}

/// One block that a pass runs, after its checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// The slot.
    pub slot: usize,
    /// The KV row of the first token. Token `t` writes row `pos0 + t`.
    pub pos0: usize,
    /// The tokens.
    pub tokens: Vec<u32>,
    /// The per-token meta of a branch block (position, gap start, gap end, 0), or `None` for a plain block.
    pub meta: Option<Vec<[u32; 4]>>,
    /// The block rows whose logits the last stage gives, in request order.
    pub logit_rows: Vec<usize>,
}

impl Block {
    /// The token count of the slot after the block: the end of a plain block, or `pos0` after a branch block, whose
    /// rows no one keeps (spec 6.14).
    pub fn n_after(&self) -> u64 {
        match self.meta {
            None => (self.pos0 + self.tokens.len()) as u64,
            Some(_) => self.pos0 as u64,
        }
    }
}

/// What a check of a pass knows about the slots of one stage.
pub struct SlotView<'a> {
    /// The positions of each slot.
    pub seqs: &'a [usize],
    /// The token count of each slot.
    pub n: &'a [u64],
    /// The position after the last row that each slot's rings hold (`CardShard::kv_high`).
    pub high: &'a [usize],
}

/// Check a pass before it changes anything, and give its block. The message says which rule breaks.
///
/// - A pass has one segment, of tokens below the vocabulary, at most `max_block` of them.
/// - A plain segment feeds the positions `pos0` to `pos0 + n`. `pos0` is at most the slot's token count. A lower
///   `pos0` goes back, and a ring must still hold the window before it (`kv_can_rewind`).
/// - A segment with `meta` is a branch block: 16 bytes for each token. Token `t` writes row `pos0 + t`, and `pos0`
///   is its first position. Its position and its gap are at most its row, and the fourth value is 0.
/// - `logit_rows` names rows of the block, `pos0` to `pos0 + n`. For a plain block they are positions.
pub fn check_pass(segments: &[Segment], view: &SlotView<'_>, max_block: usize) -> Result<Block, String> {
    let [segment] = segments else {
        return Err(format!(
            "the pass has {} segments; the engine takes {MAX_SEGMENTS}",
            segments.len()
        ));
    };
    let State::Slot(slot) = segment.state;
    let slot = slot as usize;
    if slot >= view.seqs.len() {
        return Err(format!("the engine has {} slots, not slot {slot}", view.seqs.len()));
    }
    let SegmentInputs::Tokens(tokens) = &segment.inputs else {
        return Err("the engine takes tokens only (D2)".to_owned());
    };
    let n = tokens.len();
    if n == 0 || n > max_block {
        return Err(format!("the segment has {n} tokens; a block holds 1 to {max_block}"));
    }
    if let Some(bad) = tokens.iter().find(|token| **token as usize >= VOCAB) {
        return Err(format!("token id {bad} is outside the vocabulary of {VOCAB}"));
    }
    if segment.positions.len() != n {
        return Err(format!(
            "the segment has {n} tokens and {} positions",
            segment.positions.len()
        ));
    }
    let pos0 = segment.positions[0] as usize;
    let seq = view.seqs[slot];
    if pos0 + n > seq {
        return Err(format!(
            "positions {pos0} to {} do not fit slot {slot} of {seq} positions",
            pos0 + n
        ));
    }
    let meta = if segment.meta.is_empty() {
        if segment
            .positions
            .iter()
            .enumerate()
            .any(|(t, p)| *p != (pos0 + t) as u64)
        {
            return Err("the positions of a plain segment are contiguous".to_owned());
        }
        None
    } else {
        if segment.meta.len() != n * META_BYTES {
            return Err(format!(
                "the meta has {} bytes; a branch block has {META_BYTES} for each of its {n} tokens",
                segment.meta.len()
            ));
        }
        let meta: Vec<[u32; 4]> = segment
            .meta
            .chunks_exact(META_BYTES)
            .map(|chunk| {
                std::array::from_fn(|i| u32::from_le_bytes(chunk[4 * i..4 * i + 4].try_into().expect("4 bytes")))
            })
            .collect();
        for (t, m) in meta.iter().enumerate() {
            let row = (pos0 + t) as u32;
            if u64::from(m[0]) != segment.positions[t] {
                return Err(format!(
                    "token {t} has the position {} in its meta and {} in positions",
                    m[0], segment.positions[t]
                ));
            }
            if m[0] > row || m[1] > m[2] || m[2] > row || m[3] != 0 {
                return Err(format!(
                    "the meta of token {t} is {m:?}; its position and its gap are at most its row {row}, and the last \
                     value is 0"
                ));
            }
        }
        Some(meta)
    };
    let held = view.n[slot];
    if pos0 as u64 > held {
        return Err(format!(
            "slot {slot} holds {held} tokens; a segment from position {pos0} would leave a gap"
        ));
    }
    if (pos0 as u64) < held && !kv_can_rewind(seq, view.high[slot], pos0) {
        return Err(format!(
            "slot {slot} cannot go back to position {pos0}: its rings no longer hold the window before it"
        ));
    }
    let wanted = &segment.outputs.logit_rows;
    if wanted.len() > MAX_LOGIT_ROWS {
        return Err(format!(
            "the segment wants {} logit rows; a block gives at most {MAX_LOGIT_ROWS}",
            wanted.len()
        ));
    }
    let mut logit_rows = Vec::with_capacity(wanted.len());
    for row in wanted {
        if *row < pos0 as u64 || *row >= (pos0 + n) as u64 {
            return Err(format!(
                "a logit row at {row} is not a row of the block, {pos0} to {}",
                pos0 + n
            ));
        }
        logit_rows.push(*row as usize - pos0);
    }
    Ok(Block {
        slot,
        pos0,
        tokens: tokens.clone(),
        meta,
        logit_rows,
    })
}

/// The first position that a part holds in a slot of `seq` positions with `n` tokens and rings up to `high`.
///
/// A global layer, or a slot with no ring, holds every position from 0. A ring holds the last `KV_RING_ROWS`
/// positions, less the positions that the slot wrote past `n` over the oldest of them (`save_kv_layer_from`).
pub fn base(seq: usize, l: usize, n: u64, high: usize) -> u64 {
    match kv_ring(seq, l) {
        0 => 0,
        ring => ((n as usize).saturating_sub(ring).max(high.saturating_sub(ring))).min(n as usize) as u64,
    }
}

/// Check that the positions `span` of layer `l` fit a slot of `seq` positions (`KvSnapshot::join`).
///
/// A global layer, or any layer of a slot with no ring, needs every position from 0. A ring needs at least the
/// sliding window before the end.
pub fn fits(seq: usize, l: usize, span: &CoverageSpan) -> Result<(), String> {
    if span.start > span.end || span.end > seq as u64 {
        return Err(format!(
            "positions {} to {} do not fit a slot of {seq} positions",
            span.start, span.end
        ));
    }
    if kv_ring(seq, l) == 0 {
        if span.start != 0 {
            return Err(format!(
                "layer {l} in a slot of {seq} positions needs every position from 0; the part starts at {}",
                span.start
            ));
        }
        return Ok(());
    }
    let need = span.end.saturating_sub(u64::from(SLIDING_WINDOW));
    if span.start > need {
        return Err(format!(
            "layer {l} needs the window of {SLIDING_WINDOW} positions before {}; the part starts at {}",
            span.end, span.start
        ));
    }
    Ok(())
}

/// The positions of the ring rows that a write of positions `rows` puts into layer `l` of a slot of `seq` positions:
/// all of them, or the last `KV_RING_ROWS` for a ring.
pub fn written(seq: usize, l: usize, rows: Range<u64>) -> Range<u64> {
    match kv_ring(seq, l) {
        0 => rows,
        ring => rows.end.saturating_sub(ring as u64).max(rows.start)..rows.end,
    }
}

/// The identity numbers and the plan table lineage of a build on its cards (spec 12.1).
pub struct Lineage {
    /// The SHA-256 of the kernel binary that the cards load, as `sha256:` and 64 hex digits.
    pub kernel_sha256: String,
    /// The `ocloc` version that compiled the kernels.
    pub ocloc: String,
    /// The IGC revision of that `ocloc`, or `unknown` when `ocloc` gives none.
    pub igc: String,
    /// The Level Zero driver version.
    pub driver: u32,
    /// For each stage, the PCI vendor id, device id and revision of its card.
    pub devices: Vec<(u32, u32, u32)>,
}

impl Lineage {
    /// The numbers of the identity.
    pub fn numbers(&self, max_block: usize) -> BTreeMap<String, String> {
        let mut numbers = BTreeMap::new();
        numbers.insert("kernel.megakernel_bmg.bin".to_owned(), self.kernel_sha256.clone());
        numbers.insert("ocloc".to_owned(), self.ocloc.clone());
        numbers.insert("igc".to_owned(), self.igc.clone());
        numbers.insert("driver".to_owned(), self.driver.to_string());
        numbers.insert("max_block".to_owned(), max_block.to_string());
        numbers.insert("ring_rows".to_owned(), KV_RING_ROWS.to_string());
        numbers.insert("canary".to_owned(), "not run".to_owned());
        numbers
    }

    /// The plan table lineage: the compiler, the driver and the device of each stage.
    pub fn plan_table(&self) -> String {
        let devices: Vec<String> = self
            .devices
            .iter()
            .map(|(vendor, device, revision)| format!("{vendor:04x}:{device:04x}:{revision:02x}"))
            .collect();
        format!(
            "ocloc {} igc {} bmg-g31; level-zero driver {}; devices {}",
            self.ocloc,
            self.igc,
            self.driver,
            devices.join(",")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nagaya_abi::OutputsWanted;

    fn config(slots: Vec<usize>) -> Config {
        Config {
            model_dir: PathBuf::from("/nonexistent"),
            slots,
            max_block: 512,
        }
    }

    fn segment(slot: u32, from: u64, tokens: &[u32], logits: &[u64]) -> Segment {
        Segment {
            state: State::Slot(slot),
            inputs: SegmentInputs::Tokens(tokens.to_vec()),
            positions: (from..from + tokens.len() as u64).collect(),
            meta: Vec::new(),
            outputs: OutputsWanted {
                logit_rows: logits.to_vec(),
            },
        }
    }

    /// The limits keep the rules of D1 and D11, and the residual term is 2 × 512 × 11,264 bytes (engine-abi D1).
    ///
    /// Falsified by a residual term of one buffer, and by a seal capability.
    #[test]
    fn the_limits_keep_the_rules_of_the_stages_and_the_feed_rule() {
        let limits = limits(&config(vec![4096, 65_536]), [1, 2]);
        assert_eq!(limits.check(), Ok(()));
        let residual = limits
            .memory
            .mandatory
            .iter()
            .find(|term| term.name == RESIDUAL_TERM)
            .expect("a residual term");
        assert_eq!(residual.bytes, 11_534_336);
        assert_eq!(limits.devices, vec![0, 1]);
        assert_eq!(limits.feed, Some(FeedRule { block: 512 }));
        assert_eq!(limits.max_pass_inputs, 512);
        assert_eq!(limits.parts.len(), 60);
        assert_eq!(limits.parts[10].name, "k.05");
        assert_eq!(limits.parts[10].kind, PartKind::Cells);
        assert_eq!(limits.parts[10].shape, vec![2, 512]);
        assert_eq!(limits.parts[3].name, "v.01");
        assert_eq!(limits.parts[3].kind, PartKind::Ring);
        assert_eq!(limits.parts[3].shape, vec![8, 256]);
        assert_eq!(limits.parts[29].stage, 0);
        assert_eq!(limits.parts[30].stage, 1);
        // A token costs 450,560 bytes in all parts (engine-abi "mkviolet's parts").
        let per_token: u64 = limits
            .parts
            .iter()
            .map(|part| part.shape.iter().product::<u64>() * 4)
            .sum();
        assert_eq!(per_token, 450_560);
        assert_eq!(limits.slots.len(), 2);
        assert_eq!(limits.slots[1].tokens, 65_536);
        for card in 0..2u32 {
            for name in ["weights", "kv", "work"] {
                let term = limits
                    .memory
                    .mandatory
                    .iter()
                    .find(|term| term.name == format!("{name}.{card}"))
                    .unwrap_or_else(|| panic!("a term {name}.{card}"));
                assert_eq!(term.tier, Tier::Device(card));
            }
        }
    }

    /// Equal sizes next to each other make one class, and a slot holds its size in tokens.
    #[test]
    fn equal_slot_sizes_make_one_class() {
        let classes = slot_limits(&[2048, 2048, 8192]);
        assert_eq!(classes.len(), 2);
        assert_eq!((classes[0].count, classes[0].tokens), (2, 2048));
        assert_eq!((classes[1].count, classes[1].tokens), (1, 8192));
        assert_eq!(classes[0].bytes, 2048 * 450_560);
    }

    #[test]
    fn a_config_outside_the_sizes_is_refused() {
        assert!(config(vec![]).check().is_err());
        assert!(config(vec![65_537]).check().is_err());
        assert!(config(vec![0]).check().is_err());
        let mut big = config(vec![4096]);
        big.max_block = 1025;
        assert!(big.check().unwrap_err().contains("max_block"));
        assert_eq!(config(vec![4096, 65_536]).check(), Ok(()));
    }

    fn view<'a>(seqs: &'a [usize], n: &'a [u64], high: &'a [usize]) -> SlotView<'a> {
        SlotView { seqs, n, high }
    }

    /// A plain segment starts at most at the slot's token count, and a logit row names a position of the block.
    ///
    /// Falsified by a check that takes a gap, or that maps a logit position to itself and not to its block row.
    #[test]
    fn a_plain_segment_starts_at_the_token_count_or_before() {
        let seqs = [4096];
        let block = check_pass(&[segment(0, 3, &[1, 2, 3], &[5])], &view(&seqs, &[3], &[3]), 512).unwrap();
        assert_eq!(block.pos0, 3);
        assert_eq!(block.logit_rows, vec![2]);
        assert_eq!(block.n_after(), 6);
        let gap = check_pass(&[segment(0, 4, &[1], &[])], &view(&seqs, &[3], &[3]), 512).unwrap_err();
        assert!(gap.contains("gap"), "{gap}");
        // A short slot goes back to any position.
        assert!(check_pass(&[segment(0, 1, &[1], &[])], &view(&seqs, &[3], &[3]), 512).is_ok());
        let outside = check_pass(&[segment(0, 0, &[1, 2], &[2])], &view(&seqs, &[0], &[0]), 512).unwrap_err();
        assert!(outside.contains("logit row"), "{outside}");
    }

    #[test]
    fn a_pass_outside_the_rules_is_refused() {
        let seqs = [16, 16];
        let v = view(&seqs, &[0, 0], &[0, 0]);
        let two = [segment(0, 0, &[1], &[]), segment(1, 0, &[1], &[])];
        assert!(check_pass(&two, &v, 512).unwrap_err().contains("segments"));
        assert!(
            check_pass(&[segment(2, 0, &[1], &[])], &v, 512)
                .unwrap_err()
                .contains("slot 2")
        );
        assert!(
            check_pass(&[segment(0, 0, &[VOCAB as u32], &[])], &v, 512)
                .unwrap_err()
                .contains("vocabulary")
        );
        assert!(
            check_pass(&[segment(0, 0, &[1; 17], &[])], &v, 512)
                .unwrap_err()
                .contains("do not fit")
        );
        assert!(
            check_pass(&[segment(0, 0, &[1; 3], &[])], &v, 2)
                .unwrap_err()
                .contains("block holds")
        );
        let mut rows = segment(0, 0, &[1], &[]);
        rows.inputs = SegmentInputs::Rows(vec![0; 4]);
        assert!(check_pass(&[rows], &v, 512).unwrap_err().contains("tokens only"));
        let mut skip = segment(0, 0, &[1, 2], &[]);
        skip.positions = vec![0, 2];
        assert!(check_pass(&[skip], &v, 512).unwrap_err().contains("contiguous"));
    }

    /// A slot with a ring goes back only while its rings hold the window before the new start.
    #[test]
    fn a_long_slot_goes_back_only_within_its_ring() {
        let seqs = [65_536];
        assert!(check_pass(&[segment(0, 9000, &[1], &[])], &view(&seqs, &[9100], &[9100]), 512).is_ok());
        let far = check_pass(&[segment(0, 100, &[1], &[])], &view(&seqs, &[9100], &[9100]), 512).unwrap_err();
        assert!(far.contains("go back"), "{far}");
    }

    fn meta_bytes(meta: &[[u32; 4]]) -> Vec<u8> {
        meta.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A branch block names its rows by `pos0 + t`, and it leaves the slot at `pos0`.
    ///
    /// Falsified by a check that takes a gap past the row, or that leaves the slot at the end of the block.
    #[test]
    fn a_branch_block_checks_its_meta_and_holds_no_row() {
        // The tail 2 and 3 at positions 2 and 3, then two branches of one token at position 4.
        let meta = [[2, 0, 0, 0], [3, 0, 0, 0], [4, 4, 4, 0], [4, 4, 5, 0]];
        let mut branch = segment(0, 2, &[1, 2, 3, 4], &[4, 5]);
        branch.positions = vec![2, 3, 4, 4];
        branch.meta = meta_bytes(&meta);
        let seqs = [4096];
        let block = check_pass(std::slice::from_ref(&branch), &view(&seqs, &[2], &[2]), 512).unwrap();
        assert_eq!(block.meta.as_deref(), Some(&meta[..]));
        assert_eq!(block.logit_rows, vec![2, 3]);
        assert_eq!(block.n_after(), 2);
        let mut bad = branch.clone();
        bad.meta = meta_bytes(&[[2, 0, 0, 0], [3, 0, 0, 0], [4, 4, 4, 0], [4, 4, 6, 0]]);
        assert!(
            check_pass(&[bad], &view(&seqs, &[2], &[2]), 512)
                .unwrap_err()
                .contains("gap")
        );
        let mut short = branch.clone();
        short.meta.pop();
        assert!(
            check_pass(&[short], &view(&seqs, &[2], &[2]), 512)
                .unwrap_err()
                .contains("bytes")
        );
        let mut other = branch;
        other.positions[3] = 3;
        assert!(
            check_pass(&[other], &view(&seqs, &[2], &[2]), 512)
                .unwrap_err()
                .contains("position")
        );
    }

    /// A ring holds the last 4,096 positions of a long slot, and a short slot holds every position (engine-abi
    /// "mkviolet's parts").
    #[test]
    fn a_ring_holds_the_last_positions_of_a_long_slot() {
        assert_eq!(base(4096, 0, 4000, 4000), 0);
        assert_eq!(base(65_536, 0, 9000, 9000), 9000 - 4096);
        assert_eq!(base(65_536, 5, 9000, 9000), 0, "a global layer holds every position");
        assert_eq!(
            base(65_536, 0, 9000, 9500),
            9500 - 4096,
            "rows past n wrote over the oldest"
        );
        assert_eq!(base(65_536, 0, 100, 100), 0);
    }

    /// A fit follows `KvSnapshot::join`: a ring needs the window, everything else needs every position.
    #[test]
    fn a_fit_follows_the_join_of_mkviolet() {
        let span = |start, end| CoverageSpan { part: 0, start, end };
        assert_eq!(fits(4096, 0, &span(0, 4096)), Ok(()));
        assert!(fits(4096, 0, &span(0, 4097)).is_err());
        assert!(fits(4096, 0, &span(10, 100)).is_err(), "a short slot has no ring");
        assert_eq!(fits(65_536, 0, &span(9000 - 4096, 9000)), Ok(()));
        assert_eq!(fits(65_536, 0, &span(9000 - 1024, 9000)), Ok(()));
        assert!(
            fits(65_536, 0, &span(9000 - 1023, 9000)).is_err(),
            "the window must be held"
        );
        assert!(
            fits(65_536, 5, &span(1, 9000)).is_err(),
            "a global layer needs every position"
        );
    }

    #[test]
    fn a_ring_write_keeps_its_last_rows() {
        assert_eq!(written(4096, 0, 0..4000), 0..4000);
        assert_eq!(written(65_536, 0, 0..9000), 9000 - 4096..9000);
        assert_eq!(written(65_536, 5, 0..9000), 0..9000);
    }

    /// Each wanted address maps to its card, in the order of the profile, also when `zeDeviceGet` gives another order.
    ///
    /// Falsified by a match in card order, and by a match that takes the first of two equal addresses.
    #[test]
    fn a_pci_address_maps_to_its_card() {
        let cards = vec!["0000:08:00.0".to_owned(), "0000:03:00.0".to_owned()];
        let wanted = vec!["0000:03:00.0".to_owned(), "0000:08:00.0".to_owned()];
        assert_eq!(match_addresses(&cards, &wanted), Ok(vec![1, 0]));
        let missing = match_addresses(&cards, &["0000:80:00.0".to_owned()]).unwrap_err();
        assert!(
            missing.contains("0000:80:00.0") && missing.contains("0000:08:00.0"),
            "{missing}"
        );
        let twice = vec!["0000:03:00.0".to_owned(), "0000:03:00.0".to_owned()];
        assert!(match_addresses(&cards, &twice).unwrap_err().contains("twice"));
        assert!(
            match_addresses(&twice, &["0000:03:00.0".to_owned()])
                .unwrap_err()
                .contains("two")
        );
    }
}
