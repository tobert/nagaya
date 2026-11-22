//! The fake engine with a feed rule, in one stage or in two (engine-abi D1, D11).
//!
//! Its bits depend on the token count of a pass, as the bits of the Gemma 4 engine do. Each layer keeps one cells part,
//! and the row at a position is a hash of the row before it, the input of the layer, the position, and the tokens of
//! the pass. The residual between two stages is the output of the last layer of the first stage. Thus the bits of a
//! state depend only on its tokens and on the split of the tokens into passes. One stage and two stages give the same
//! bits.

use std::collections::BTreeMap;

use super::{logit_row, mix};
use crate::{
    BitClass, BitSetting, Capability, ChainLimits, Coverage, CoverageSpan, Determinism, Dtype, Engine, FeedRule,
    Identity, Inputs, Kind, Lifetime, Limits, MandatoryTerm, MemoryLimits, Output, PartError, PartId, PartKind,
    PartLimits, Parts, PassError, PassErrorKind, PassOut, PassStatus, RESIDUAL_TERM, Residual, Segment, SegmentInputs,
    SlotId, SlotLimit, Slots, Stage, StageHandle, StageOut, Stages, State, Tap, Tier, WriteResult,
};

/// The bytes of one cells row.
const ROW: usize = 8;

/// The bytes of the output of a layer, and of one residual row.
const WIDTH: usize = 16;

/// The layers of the fake. Layer `l` has the cells part `k.<ll>`.
const LAYERS: usize = 2;

/// A defect of a fake with a feed rule, for the conformance gates (spec 8.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Defect {
    None,
    /// After `take_handles`, stage 1 reads the residual rows in reverse order. The stage path differs from a pass.
    ReverseResidual,
    /// A row also depends on the count of passes since the slot was clear. A restore then gives other bits.
    History,
}

/// The fake engine with a feed rule. It declares `always` with a feed rule, and no `segments_exact`.
pub struct FakeFeed {
    limits: Limits,
    identity: Identity,
    /// The stages, until `take_handles` moves them out.
    stages: Option<Vec<FakeStage>>,
    last_error: Option<String>,
}

impl FakeFeed {
    /// Make the engine in one stage, with a feed block of `block` tokens and the given slots.
    pub fn new(block: u64, slots: Vec<SlotLimit>) -> Self {
        Self::make(1, block, slots, Defect::None)
    }

    /// Make the engine in two stages on devices 0 and 1, with the `stages` capability. Its bits equal those of
    /// [`FakeFeed::new`].
    pub fn two_stages(block: u64, slots: Vec<SlotLimit>) -> Self {
        Self::make(2, block, slots, Defect::None)
    }

    /// Make a two-stage engine whose stage path reads the residual in reverse order. The stage gate must fail it.
    pub fn reverse_residual(block: u64, slots: Vec<SlotLimit>) -> Self {
        Self::make(2, block, slots, Defect::ReverseResidual)
    }

    /// Make a one-stage engine whose rows also depend on the passes since a clear. The feed gate must fail it.
    pub fn history(block: u64, slots: Vec<SlotLimit>) -> Self {
        Self::make(1, block, slots, Defect::History)
    }

    /// Give the engine logits over a vocabulary of `vocab` tokens (engine-abi, "Logit outputs").
    pub fn with_vocab(mut self, vocab: usize) -> Self {
        assert!(vocab > 0, "a vocabulary has at least one token");
        for stage in self.stages.as_mut().expect("set the vocabulary before take_handles") {
            stage.vocab = Some(vocab);
        }
        self
    }

    /// Add `bias` to each logit row. Call it after [`FakeFeed::with_vocab`].
    pub fn with_logit_bias(mut self, bias: Vec<f32>) -> Self {
        for stage in self.stages.as_mut().expect("set the bias before take_handles") {
            let vocab = stage
                .vocab
                .expect("a logit bias needs a vocabulary; call with_vocab first");
            assert_eq!(bias.len(), vocab, "a logit bias has one bias for each token");
            stage.bias = Some(bias.clone());
        }
        self
    }

    /// Let a pass hold up to `inputs` tokens, more than one block. A feed that breaks the rule then fits a pass, so a
    /// test can see it. A pass of an engine with stages still holds at most one residual buffer of rows.
    pub fn with_max_pass_inputs(mut self, inputs: u32) -> Self {
        self.limits.max_pass_inputs = inputs;
        if let Err(message) = self.limits.check() {
            panic!("the fake with a feed rule breaks a rule of its limits: {message}");
        }
        self
    }

    /// Name the family library that the engine needs (spec 11.3).
    pub fn with_family(mut self, family: &str) -> Self {
        self.identity.family = family.to_owned();
        self
    }

    fn make(stage_count: usize, block: u64, slot_limits: Vec<SlotLimit>, defect: Defect) -> Self {
        assert!(block > 0, "a feed block holds at least one token");
        let parts: Vec<PartLimits> = (0..LAYERS)
            .map(|layer| PartLimits {
                name: format!("k.{layer:02}"),
                kind: PartKind::Cells,
                dtype: Dtype::U8,
                shape: vec![ROW as u64],
                block_format: None,
                stage: if stage_count == 1 { 0 } else { layer as u32 },
            })
            .collect();
        let mut capabilities = vec![Capability::Parts, Capability::Slots, Capability::Coverage];
        let mut mandatory = vec![MandatoryTerm {
            name: "weights".to_owned(),
            tier: Tier::Device(0),
            bytes: 4096,
            lifetime: Lifetime::Engine,
            kinds: vec![Kind::Prefill, Kind::Decode, Kind::Read],
        }];
        let stages = if stage_count == 1 {
            vec![Stage {
                device: 0,
                layers: 0..LAYERS as u32,
                head: true,
            }]
        } else {
            capabilities.push(Capability::Stages);
            mandatory.push(MandatoryTerm {
                name: "weights.1".to_owned(),
                tier: Tier::Device(1),
                bytes: 4096,
                lifetime: Lifetime::Engine,
                kinds: vec![Kind::Prefill, Kind::Decode, Kind::Read],
            });
            mandatory.push(MandatoryTerm {
                name: RESIDUAL_TERM.to_owned(),
                tier: Tier::Host,
                bytes: 2 * block * WIDTH as u64,
                lifetime: Lifetime::Engine,
                kinds: vec![Kind::Prefill, Kind::Decode, Kind::Read],
            });
            (0..stage_count)
                .map(|stage| Stage {
                    device: stage as u32,
                    layers: stage as u32..stage as u32 + 1,
                    head: stage + 1 == stage_count,
                })
                .collect()
        };
        let limits = Limits {
            capabilities,
            parts: parts.clone(),
            inputs: Inputs::Tokens,
            devices: (0..stage_count as u32).collect(),
            stages,
            max_segments: 8,
            max_pass_inputs: block as u32,
            max_logit_rows: 8,
            slots: slot_limits.clone(),
            max_quantum_ms: 10,
            max_long_quantum_ms: None,
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
            residual_row_bytes: if stage_count == 1 { 0 } else { WIDTH as u64 },
            control: None,
            feed: Some(FeedRule { block }),
            layer_loops: None,
            cycles: None,
        };
        if let Err(message) = limits.check() {
            panic!("the fake with a feed rule breaks a rule of its limits: {message}");
        }
        let slot_tokens: Vec<u64> = slot_limits
            .iter()
            .flat_map(|class| std::iter::repeat_n(class.tokens, class.count as usize))
            .collect();
        let stages = (0..stage_count)
            .map(|stage| {
                let layers: Vec<PartId> = (0..LAYERS)
                    .filter(|layer| parts[*layer].stage as usize == stage)
                    .collect();
                FakeStage {
                    first: stage == 0,
                    last: stage + 1 == stage_count,
                    rows: vec![vec![Vec::new(); layers.len()]; slot_tokens.len()],
                    passes: vec![0; slot_tokens.len()],
                    layers,
                    slot_tokens: slot_tokens.clone(),
                    defect,
                    taken: false,
                    vocab: None,
                    bias: None,
                    last_error: None,
                }
            })
            .collect();
        let identity = Identity {
            model: format!("fake-feed-{stage_count}"),
            family: "fake".to_owned(),
            checkpoint_digest: "fake:feed".to_owned(),
            tokenizer_hash: "fake:0".to_owned(),
            template: "fake".to_owned(),
            numbers: BTreeMap::new(),
            plan_table: "fake".to_owned(),
            // D11 rule 3: the block is a bit setting that defines the reference bits.
            bit_settings: vec![BitSetting {
                name: "feed.block".to_owned(),
                value: block.to_string(),
                class: BitClass::ReferenceDefining,
            }],
            canary_hash: "fake:0".to_owned(),
            build: std::collections::BTreeMap::new(),
        };
        Self {
            limits,
            identity,
            stages: Some(stages),
            last_error: None,
        }
    }

    fn stages_mut(&mut self) -> Option<&mut Vec<FakeStage>> {
        self.stages.as_mut()
    }

    /// The stage that holds a part.
    fn stage_of(&self, part: PartId) -> Option<usize> {
        let stage = self.limits.parts.get(part)?.stage as usize;
        if self.limits.capabilities.contains(&Capability::Stages) {
            Some(stage)
        } else {
            Some(0)
        }
    }
}

/// One stage of the fake: its layers, and the rows of those layers in each slot.
struct FakeStage {
    first: bool,
    last: bool,
    /// The parts of this stage, one for each layer, in layer order.
    layers: Vec<PartId>,
    /// For each slot, for each layer of this stage, the rows.
    rows: Vec<Vec<Vec<[u8; ROW]>>>,
    /// For each slot, the passes since the last clear. Only the defect `History` reads it.
    passes: Vec<u64>,
    slot_tokens: Vec<u64>,
    defect: Defect,
    /// True after `take_handles`.
    taken: bool,
    vocab: Option<usize>,
    bias: Option<Vec<f32>>,
    last_error: Option<String>,
}

fn refused(message: String) -> PassError {
    PassError {
        kind: PassErrorKind::Refused,
        message,
    }
}

/// The row of a layer at a position: a hash of the row before it, the input, the position and the tokens of the pass.
fn layer_row(previous: &[u8; ROW], input: &[u8; WIDTH], position: u64, pass_tokens: u64, history: u64) -> [u8; ROW] {
    let mut hash = mix(0x6665_6564 ^ position);
    hash = mix(hash ^ pass_tokens.wrapping_mul(0x9E37_79B9));
    hash = mix(hash ^ history);
    hash = mix(hash ^ u64::from_le_bytes(*previous));
    hash = mix(hash ^ u64::from_le_bytes(input[..8].try_into().expect("8 bytes")));
    hash = mix(hash ^ u64::from_le_bytes(input[8..].try_into().expect("8 bytes")));
    hash.to_le_bytes()
}

/// The output of a layer: its row, and a hash of the row and the input.
fn layer_output(row: &[u8; ROW], input: &[u8; WIDTH]) -> [u8; WIDTH] {
    let tail = mix(u64::from_le_bytes(*row) ^ mix(u64::from_le_bytes(input[..8].try_into().expect("8 bytes"))));
    let mut output = [0u8; WIDTH];
    output[..ROW].copy_from_slice(row);
    output[ROW..].copy_from_slice(&tail.to_le_bytes());
    output
}

impl FakeStage {
    fn slot(&self, state: State) -> Option<usize> {
        let State::Slot(slot) = state;
        ((slot as usize) < self.rows.len()).then_some(slot as usize)
    }

    fn layer(&self, part: PartId) -> Option<usize> {
        self.layers.iter().position(|candidate| *candidate == part)
    }

    /// The rows that the slot holds in this stage. Every layer of a stage holds the same count.
    fn held(&self, slot: usize) -> u64 {
        self.rows[slot].first().map_or(0, |rows| rows.len() as u64)
    }

    /// Check a pass before it changes anything. It gives the token count of the pass.
    fn check(&self, segments: &[Segment], residual: &Residual) -> Result<u64, PassError> {
        if segments.len() > 8 {
            return Err(refused(format!(
                "the pass has {} segments; the fake takes 8",
                segments.len()
            )));
        }
        let mut total = 0u64;
        let mut seen = Vec::new();
        for segment in segments {
            let SegmentInputs::Tokens(tokens) = &segment.inputs else {
                return Err(refused("the fake takes tokens".to_owned()));
            };
            let slot = self
                .slot(segment.state)
                .ok_or_else(|| refused(format!("the fake has no slot {:?}", segment.state)))?;
            if seen.contains(&slot) {
                return Err(refused(format!("two segments of the pass name slot {slot}")));
            }
            seen.push(slot);
            let held = self.held(slot);
            let expected: Vec<u64> = (held..held + tokens.len() as u64).collect();
            if segment.positions != expected {
                return Err(refused(format!(
                    "a segment of slot {slot} feeds positions {:?}; the slot holds {held} rows, so they start there",
                    segment.positions.first()
                )));
            }
            if held + tokens.len() as u64 > self.slot_tokens[slot] {
                return Err(refused(format!(
                    "slot {slot} holds {} tokens; the pass would make {}",
                    self.slot_tokens[slot],
                    held + tokens.len() as u64
                )));
            }
            if let Some(row) = segment
                .outputs
                .logit_rows
                .iter()
                .find(|row| !segment.positions.contains(row))
            {
                return Err(refused(format!(
                    "a logit row at position {row} is not a position of the segment"
                )));
            }
            if !segment.outputs.logit_rows.is_empty() && self.vocab.is_none() {
                return Err(refused("the fake gives no logits without a vocabulary".to_owned()));
            }
            total += tokens.len() as u64;
        }
        if !self.first && residual.rows() as u64 != total {
            return Err(refused(format!(
                "the residual holds {} rows, but the pass has {total} tokens",
                residual.rows()
            )));
        }
        if !self.last && total as usize > residual.capacity() {
            return Err(refused(format!(
                "the pass has {total} tokens; the residual holds {} rows",
                residual.capacity()
            )));
        }
        Ok(total)
    }

    /// Run the layers of this stage for one pass. The checks ran before, so this changes state and does not fail.
    fn run(&mut self, segments: &[Segment], residual: &mut Residual, total: u64) -> Result<StageOut, PassError> {
        let inputs: Vec<[u8; WIDTH]> = if self.first {
            segments
                .iter()
                .flat_map(|segment| match &segment.inputs {
                    SegmentInputs::Tokens(tokens) => tokens.clone(),
                    _ => unreachable!("the check admits only tokens"),
                })
                .map(|token| {
                    let mut input = [0u8; WIDTH];
                    input[..4].copy_from_slice(&token.to_le_bytes());
                    input
                })
                .collect()
        } else {
            let mut rows: Vec<[u8; WIDTH]> = residual
                .bytes()
                .chunks(WIDTH)
                .map(|row| row.try_into().expect("a residual row has WIDTH bytes"))
                .collect();
            if self.taken && self.defect == Defect::ReverseResidual {
                rows.reverse();
            }
            rows
        };
        let mut outputs_of_pass = Vec::with_capacity(inputs.len());
        let mut outputs = Vec::new();
        let mut at = 0usize;
        for segment in segments {
            let slot = self.slot(segment.state).expect("the check found the slot");
            let history = if self.defect == Defect::History {
                self.passes[slot]
            } else {
                0
            };
            let mut rows_of_segment = BTreeMap::new();
            for &position in &segment.positions {
                let mut x = inputs[at];
                at += 1;
                for layer in 0..self.layers.len() {
                    let rows = &mut self.rows[slot][layer];
                    let previous = rows.last().copied().unwrap_or([0; ROW]);
                    let row = layer_row(&previous, &x, position, total, history);
                    rows.push(row);
                    x = layer_output(&row, &x);
                }
                outputs_of_pass.push(x);
                if self.last
                    && segment.outputs.logit_rows.contains(&position)
                    && let Some(vocab) = self.vocab
                {
                    rows_of_segment.insert(position, logit_row(&x, vocab, self.bias.as_deref()));
                }
            }
            self.passes[slot] += 1;
            // The output order is segment order, then request order.
            if self.last {
                for row in &segment.outputs.logit_rows {
                    outputs.push(Output {
                        name: "logits".to_owned(),
                        dtype: Dtype::F32,
                        bytes: rows_of_segment[row].clone(),
                    });
                }
            }
        }
        if !self.last {
            let bytes = residual
                .write(outputs_of_pass.len())
                .expect("the check fitted the rows to the residual");
            for (row, output) in bytes.chunks_mut(WIDTH).zip(&outputs_of_pass) {
                row.copy_from_slice(output);
            }
            return Ok(StageOut::Residual {
                status: PassStatus::Ok,
                ms: 0,
            });
        }
        Ok(StageOut::Last(PassOut {
            status: PassStatus::Ok,
            ms: 0,
            outputs,
        }))
    }

    fn refuse(&mut self, message: &str) -> WriteResult {
        self.last_error = Some(message.to_owned());
        WriteResult::Refused
    }
}

impl StageHandle for FakeStage {
    fn stage_pass(&mut self, segments: &[Segment], residual: &mut Residual) -> Result<StageOut, PassError> {
        let total = self.check(segments, residual)?;
        self.run(segments, residual, total)
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        Some(self)
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        Some(self)
    }
}

impl Parts for FakeStage {
    /// Read every row of a cells part, as `FakeAlways` does. The fake has no fixed part.
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        let (Some(slot), Some(layer)) = (self.slot(state), self.layer(part)) else {
            self.last_error = Some(format!("stage has no part {part} or no slot {state:?}"));
            return Err(PartError::Refused);
        };
        let rows = self.rows[slot][layer].len() as u64;
        self.cells_read(state, part, 0, rows, out)
    }

    fn part_write(&mut self, _state: State, _part: PartId, _bytes: &[u8]) -> WriteResult {
        self.refuse("the fake with a feed rule has no fixed part")
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        let (Some(slot), Some(layer)) = (self.slot(state), self.layer(part)) else {
            self.last_error = Some(format!("stage has no part {part} or no slot {state:?}"));
            return Err(PartError::Refused);
        };
        let rows = &self.rows[slot][layer];
        if from + n > rows.len() as u64 || out.len() != n as usize * ROW {
            self.last_error = Some(format!("the read of rows {from} to {} does not fit", from + n));
            return Err(PartError::Refused);
        }
        for (index, chunk) in out.chunks_mut(ROW).enumerate() {
            chunk.copy_from_slice(&rows[from as usize + index]);
        }
        Ok(())
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        let (Some(slot), Some(layer)) = (self.slot(state), self.layer(part)) else {
            return self.refuse("the stage has no such part or slot");
        };
        if from != self.rows[slot][layer].len() as u64 {
            return self.refuse("a cells part appends at its last row");
        }
        if bytes.len() != n as usize * ROW || from + n > self.slot_tokens[slot] {
            return self.refuse("the rows do not fit the part or the slot");
        }
        for chunk in bytes.chunks(ROW) {
            self.rows[slot][layer].push(chunk.try_into().expect("a chunk has ROW bytes"));
        }
        WriteResult::Ok
    }

    /// Keep the first `n` rows of each layer of this stage. A count above the rows is refused.
    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        let Some(slot) = self.slot(state) else {
            return self.refuse("unknown slot");
        };
        if self.rows[slot].iter().any(|rows| (rows.len() as u64) < n) {
            return self.refuse("n is larger than the rows of a layer");
        }
        for rows in &mut self.rows[slot] {
            rows.truncate(n as usize);
        }
        WriteResult::Ok
    }
}

impl Slots for FakeStage {
    fn slot_high(&self, slot: SlotId) -> u64 {
        self.rows.get(slot as usize).map_or(0, |layers| {
            layers.iter().map(|rows| rows.len() as u64).max().unwrap_or(0)
        })
    }

    fn slot_clear(&mut self, slot: SlotId) -> WriteResult {
        let Some(layers) = self.rows.get_mut(slot as usize) else {
            return self.refuse("unknown slot");
        };
        for rows in layers {
            rows.clear();
        }
        self.passes[slot as usize] = 0;
        WriteResult::Ok
    }
}

impl Coverage for FakeStage {
    fn coverage(&self, _state: State, part: PartId, bytes: &[u8]) -> CoverageSpan {
        CoverageSpan {
            part,
            start: 0,
            end: (bytes.len() / ROW) as u64,
        }
    }

    fn fits(&self, coverage: &CoverageSpan, slot: SlotId) -> Result<(), String> {
        let tokens = *self
            .slot_tokens
            .get(slot as usize)
            .ok_or_else(|| format!("no slot {slot}"))?;
        if coverage.start != 0 || coverage.end > tokens {
            return Err(format!(
                "rows {} to {} do not fit slot {slot} of {tokens} tokens",
                coverage.start, coverage.end
            ));
        }
        Ok(())
    }
}

impl Stages for FakeFeed {
    fn take_handles(&mut self) -> Vec<Box<dyn StageHandle + Send>> {
        let Some(stages) = self.stages.take() else {
            return Vec::new();
        };
        stages
            .into_iter()
            .map(|mut stage| {
                stage.taken = true;
                Box::new(stage) as Box<dyn StageHandle + Send>
            })
            .collect()
    }
}

impl Engine for FakeFeed {
    fn identity(&self) -> Identity {
        self.identity.clone()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    /// Run every stage in order on this thread. After `take_handles` the pass is refused.
    fn pass(&mut self, segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        let block = self.limits.residual_rows() as usize;
        let Some(stages) = self.stages.as_mut() else {
            return Err(refused(
                "the stages moved to their handles; run the pass through each stage handle".to_owned(),
            ));
        };
        let mut residual = Residual::new(WIDTH, block);
        // Before `take_handles`, every stage holds the same rows, so the check of the first stage covers them all. A
        // refusal changes nothing.
        let total = stages[0].check(segments, &residual)?;
        let mut out = None;
        for stage in stages.iter_mut() {
            out = Some(stage.run(segments, &mut residual, total)?);
        }
        match out {
            Some(StageOut::Last(out)) => Ok(out),
            _ => unreachable!("the last stage gives the outputs"),
        }
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.clone().or_else(|| {
            self.stages
                .as_ref()
                .and_then(|stages| stages.iter().find_map(|stage| stage.last_error.clone()))
        })
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.stages.as_ref()?;
        Some(self)
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.stages.as_ref()?;
        Some(self)
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.stages.as_ref()?;
        Some(self)
    }

    fn stages(&mut self) -> Option<&mut dyn Stages> {
        self.limits
            .capabilities
            .contains(&Capability::Stages)
            .then_some(self as &mut dyn Stages)
    }
}

/// Before `take_handles`, the engine sends each part call to the stage that holds the part.
impl Parts for FakeFeed {
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        let stage = self.stage_of(part).ok_or(PartError::Refused)?;
        self.stages_mut().ok_or(PartError::Refused)?[stage].part_read(state, part, out)
    }

    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        match (self.stage_of(part), self.stages_mut()) {
            (Some(stage), Some(stages)) => stages[stage].part_write(state, part, bytes),
            _ => WriteResult::Refused,
        }
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        let stage = self.stage_of(part).ok_or(PartError::Refused)?;
        self.stages_mut().ok_or(PartError::Refused)?[stage].cells_read(state, part, from, n, out)
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        match (self.stage_of(part), self.stages_mut()) {
            (Some(stage), Some(stages)) => stages[stage].cells_write(state, part, from, n, bytes),
            _ => WriteResult::Refused,
        }
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        let Some(stages) = self.stages_mut() else {
            return WriteResult::Refused;
        };
        if stages.iter().any(|stage| {
            stage
                .slot(state)
                .is_none_or(|slot| stage.rows[slot].iter().any(|r| (r.len() as u64) < n))
        }) {
            return WriteResult::Refused;
        }
        for stage in stages {
            assert_eq!(
                stage.set_n(state, n),
                WriteResult::Ok,
                "the check above admits the count"
            );
        }
        WriteResult::Ok
    }
}

impl Slots for FakeFeed {
    fn slot_high(&self, slot: SlotId) -> u64 {
        self.stages.as_ref().map_or(0, |stages| {
            stages.iter().map(|stage| stage.slot_high(slot)).max().unwrap_or(0)
        })
    }

    fn slot_clear(&mut self, slot: SlotId) -> WriteResult {
        let Some(stages) = self.stages_mut() else {
            return WriteResult::Refused;
        };
        if stages.iter().any(|stage| stage.rows.get(slot as usize).is_none()) {
            return WriteResult::Refused;
        }
        for stage in stages {
            assert_eq!(
                stage.slot_clear(slot),
                WriteResult::Ok,
                "the check above found the slot"
            );
        }
        WriteResult::Ok
    }
}

impl Coverage for FakeFeed {
    fn coverage(&self, state: State, part: PartId, bytes: &[u8]) -> CoverageSpan {
        let stages = self.stages.as_ref().expect("coverage before take_handles");
        stages[self.stage_of(part).unwrap_or(0)].coverage(state, part, bytes)
    }

    fn fits(&self, coverage: &CoverageSpan, slot: SlotId) -> Result<(), String> {
        let stages = self.stages.as_ref().ok_or("the stages moved to their handles")?;
        stages[self.stage_of(coverage.part).ok_or("unknown part")?].fits(coverage, slot)
    }
}
