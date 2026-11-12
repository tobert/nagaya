//! The `Engine` trait on top of the binding of the Qwen3.8 engine.

use std::sync::{Arc, OnceLock};

use nagaya_abi::{
    Control, ControlItem, Coverage, CoverageSpan, Dtype, Engine, Identity, Limits, Output, PartError, PartId, Parts,
    PassError, PassErrorKind, PassOut, PassStatus, Seal, SealError, Segment, SegmentInputs, SlotId, Slots, State, Tap,
    WriteResult,
};

use crate::canary::CanaryCheck;
use crate::control::{self, TapSet};
use crate::spec;
use crate::worker::{Call, ClassReading, PassProblem, PassRequest, Qwen38Error, Qwen38Paths, Worker, start};

/// The one worker in this process. Its HIP runtime and C++ engine stay until the process exits, because HIP teardown
/// stops the process (spec 3.3). The weights come and go with the handles (engine-abi "close").
static WORKER: OnceLock<Result<Arc<Worker>, Qwen38Error>> = OnceLock::new();

/// The worker, started on the first call. The shards must be the ones of the first call.
fn worker() -> Result<Arc<Worker>, Qwen38Error> {
    let paths = Qwen38Paths::from_env()?;
    let worker = match WORKER.get_or_init(|| start(paths.clone())) {
        Ok(worker) => Arc::clone(worker),
        Err(error) => return Err(error.clone()),
    };
    if worker.paths() != &paths {
        return Err(Qwen38Error::new(
            "QWEN38_SHARDS changed after the engine opened".to_owned(),
        ));
    }
    Ok(worker)
}

/// The identity and the limits of the engine on the shards of `QWEN38_SHARDS`, with no GPU and no load. An open gives
/// the same facts.
///
/// The canary hash of the identity comes from the table of records (spec 8.1). A checkpoint and settings with no line
/// are an error here, so the description is the checked form. `describe_with` gives the other form.
pub fn describe() -> Result<(Identity, Limits), Qwen38Error> {
    describe_with(CanaryCheck::Record)
}

/// Like `describe`, with the form of the canary check. With `Measure`, a key with no record gives an empty hash.
pub fn describe_with(check: CanaryCheck) -> Result<(Identity, Limits), Qwen38Error> {
    crate::worker::describe(&Qwen38Paths::from_env()?, check)
}

/// Open the Qwen3.8 engine, or make a new handle on the one that is open.
///
/// The first call loads the weights, and so does the first call after the last handle dropped. Every call creates a
/// new context pool for the handle. Every engine call goes to one worker thread, because the HIP device is the calling
/// thread's current device.
///
/// The canary of the load must match its record, or the open is an error (spec 8.1). `open_with` gives the other form.
pub fn open() -> Result<Qwen38Engine, Qwen38Error> {
    open_with(CanaryCheck::Record)
}

/// Like `open`, with the form of the canary check. With `Measure`, the open compares nothing: a person measures the
/// canary hash of a checkpoint that the table does not hold. The process uses `open`.
pub fn open_with(check: CanaryCheck) -> Result<Qwen38Engine, Qwen38Error> {
    let worker = worker()?;
    let load_canary = worker.acquire_checked(check)?;
    let pool = match worker.open_states(&spec::CONTEXT_CLASSES) {
        Ok(pool) => pool,
        Err(error) => {
            worker.release();
            return Err(error);
        }
    };
    Ok(Qwen38Engine {
        worker,
        load_canary,
        states: pool.ids.into_iter().map(Some).collect(),
        readings: pool.readings,
        last_error: None,
    })
}

/// The Qwen3.8 engine: a context pool, the engine's declared facts, and the last error.
pub struct Qwen38Engine {
    worker: Arc<Worker>,
    /// The context id of each slot. A slot whose clear failed has no context.
    states: Vec<Option<u64>>,
    readings: Vec<ClassReading>,
    last_error: Option<String>,
    /// The canary of the load that this handle's weights came from.
    load_canary: crate::canary::Report,
}

impl Qwen38Engine {
    /// The context id for one state.
    fn context(&self, state: State) -> Option<u64> {
        let State::Slot(slot) = state;
        self.states.get(slot as usize).copied().flatten()
    }

    /// The context's token count.
    pub fn n(&self, state: State) -> Result<u64, Qwen38Error> {
        let id = self
            .context(state)
            .ok_or_else(|| Qwen38Error::new(format!("{state:?} is not in this engine")))?;
        self.worker.n(id).map(u64::from)
    }

    /// The context's capacity in tokens.
    pub fn capacity(&self, state: State) -> Result<u64, Qwen38Error> {
        let id = self
            .context(state)
            .ok_or_else(|| Qwen38Error::new(format!("{state:?} is not in this engine")))?;
        self.worker.max(id).map(u64::from)
    }

    /// True when the context is poisoned.
    pub fn poisoned(&self, state: State) -> Result<bool, Qwen38Error> {
        let id = self
            .context(state)
            .ok_or_else(|| Qwen38Error::new(format!("{state:?} is not in this engine")))?;
        self.worker.poisoned(id)
    }

    /// True when the context owns its K/V. A fork does not.
    pub fn owns_kv(&self, state: State) -> Result<bool, Qwen38Error> {
        let id = self
            .context(state)
            .ok_or_else(|| Qwen38Error::new(format!("{state:?} is not in this engine")))?;
        self.worker.owns_kv(id)
    }

    /// The engine's byte count for one stored part and token count.
    pub fn part_bytes(&self, part: u32, n: u32) -> Result<u64, Qwen38Error> {
        spec::stored_part(part)?;
        self.worker
            .part_bytes(part, n)
            .ok_or_else(|| Qwen38Error::new("the engine worker stopped; restart the process"))
    }

    /// The pass scratch that the probe measured at open, in bytes (milestone 9c).
    pub fn pass_scratch(&self) -> u64 {
        self.worker.spec().pass_scratch
    }

    /// The canary of the load that this handle's weights came from. The open has checked it (spec 8.1).
    pub fn load_canary(&self) -> &crate::canary::Report {
        &self.load_canary
    }

    /// Run the canary of the engine (spec 8.1) and report its row hashes and its time.
    ///
    /// The worker thread runs it in two contexts of its own and frees them. It touches no context of this handle.
    /// It checks no record: nothing compares the hashes yet. A pass that fails is an error.
    pub fn canary(&self) -> Result<crate::canary::Report, Qwen38Error> {
        self.worker.canary()
    }

    /// What each class of contexts took when the engine made it, in the order of `spec::CONTEXT_CLASSES`.
    pub fn pool_readings(&self) -> &[ClassReading] {
        &self.readings
    }

    /// Record one refused pass and return its error. No context changed.
    fn failed(&mut self, message: String) -> PassError {
        self.last_error = Some(message.clone());
        refused(message)
    }
}

/// The first row of a cells write. The engine writes rows from 0 and cannot place them later (engine-abi, "The first
/// row of a cells write").
fn write_from(from: u64) -> Result<(), String> {
    if from == 0 {
        Ok(())
    } else {
        Err(format!(
            "the Qwen3.8 engine writes rows from 0 and cannot place rows at position {from}; write from 0"
        ))
    }
}

/// The close of the engine (engine-abi "close"): the handle frees its contexts, and the last handle or hold frees the
/// weights.
impl Drop for Qwen38Engine {
    fn drop(&mut self) {
        let ids = std::mem::take(&mut self.states).into_iter().flatten().collect();
        self.worker.free_states(ids);
        self.worker.release();
    }
}

/// A hold keeps the engine's weights loaded with no context pool. A test binary or a bench keeps one, so that each
/// handle it opens and drops does not load the weights again.
pub struct Hold {
    worker: Arc<Worker>,
}

/// Hold the engine's weights: load them when no handle or hold has them.
pub fn hold() -> Result<Hold, Qwen38Error> {
    let worker = worker()?;
    worker.acquire_checked(CanaryCheck::Record)?;
    Ok(Hold { worker })
}

impl Drop for Hold {
    fn drop(&mut self) {
        self.worker.release();
    }
}

/// Build the logit outputs from the engine's bytes.
///
/// The pass ran before this call, so a buffer that is too short is a fault. The error says `Poison`: the state is
/// unknown, and the nagaya must free it (D6, spec 8.2).
fn logit_outputs(bytes: &[u8], wants: &[bool], row: usize) -> Result<Vec<Output>, PassError> {
    let mut outputs = Vec::new();
    for (index, want) in wants.iter().enumerate() {
        if !*want {
            continue;
        }
        let start = index * row;
        let end = start + row;
        let Some(slice) = bytes.get(start..end) else {
            return Err(PassError {
                kind: PassErrorKind::Ran(PassStatus::Poison),
                message: format!(
                    "the Qwen3.8 engine returned {} logits bytes, but segment {index} needs bytes {start}..{end}",
                    bytes.len()
                ),
            });
        };
        outputs.push(Output {
            name: "logits".to_owned(),
            dtype: Dtype::F32,
            bytes: slice.to_vec(),
        });
    }
    Ok(outputs)
}

/// Build the tap outputs from the bytes of the worker, in the order of the layout (`control::tap_layout`). The pass ran
/// before this call, so a count or a size that differs from the layout is a fault. The error says `Poison`, as for the
/// logits.
fn tap_outputs(bytes: Vec<(&'static str, Vec<u8>)>, wants: &[bool], taps: TapSet) -> Result<Vec<Output>, PassError> {
    let fault = |message: String| PassError {
        kind: PassErrorKind::Ran(PassStatus::Poison),
        message,
    };
    let layout = control::tap_layout(wants, taps).map_err(fault)?;
    if bytes.len() != layout.len() {
        return Err(fault(format!(
            "the Qwen3.8 engine returned {} tap outputs, but the pass asks for {}",
            bytes.len(),
            layout.len()
        )));
    }
    let mut outputs = Vec::with_capacity(layout.len());
    for (want, (name, bytes)) in layout.iter().zip(bytes) {
        if name != want.name || bytes.len() != want.bytes {
            return Err(fault(format!(
                "the Qwen3.8 engine returned the tap `{name}` of {} bytes, but segment {} needs the tap `{}` of {} bytes",
                bytes.len(),
                want.segment,
                want.name,
                want.bytes
            )));
        }
        outputs.push(Output {
            name: name.to_owned(),
            dtype: Dtype::F32,
            bytes,
        });
    }
    Ok(outputs)
}

/// The context of one segment: its worker id and its token count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SegmentContext {
    /// The worker's context id.
    id: u64,
    /// The token count of the context.
    n: u64,
}

/// The engine call for one pass.
#[derive(Debug, Default, PartialEq, Eq)]
struct PassPlan {
    /// The context ids, one for each segment.
    contexts: Vec<u64>,
    /// The token count of each segment.
    lens: Vec<u32>,
    /// The concatenated token ids.
    tokens: Vec<i32>,
    /// True for each segment that wants its logit row.
    wants: Vec<bool>,
    /// The control list of each segment.
    control: Vec<Vec<ControlItem>>,
    /// The taps that the pass names.
    taps: TapSet,
}

impl PassPlan {
    /// True when the caller wants one logits row for each segment.
    fn want_logits(&self) -> bool {
        self.wants.iter().any(|want| *want)
    }
}

/// A pass that the binding refused before it ran. Nothing changed.
fn refused(message: String) -> PassError {
    PassError {
        kind: PassErrorKind::Refused,
        message,
    }
}

/// Build the engine call for one pass, or say why the binding refuses it.
///
/// Every check here runs before the binding calls the engine. Thus a refused pass changes no context.
///
/// The engine gives the taps `residual` and `fold` and a control table (engine-abi "The residual of the Qwen3.8
/// engine"). An unknown tap, a tap with no logit row and a control list that breaks a rule of the table are refused.
/// The worker refuses a list that names an entry that no write filled, because it holds the table.
fn plan_pass(segments: &[Segment], taps: &[Tap], contexts: &[SegmentContext]) -> Result<PassPlan, PassError> {
    assert_eq!(segments.len(), contexts.len(), "every segment needs its context");
    let tap_set = control::tap_set(taps).map_err(refused)?;
    let mut plan = PassPlan {
        taps: tap_set,
        ..PassPlan::default()
    };
    for (index, segment) in segments.iter().enumerate() {
        let context = contexts[index];
        if let Some(problem) = control::list_problem(index, &segment.control) {
            return Err(refused(problem));
        }
        let segment_tokens = match &segment.inputs {
            SegmentInputs::Tokens(tokens) => tokens,
            _ => {
                return Err(refused(format!(
                    "segment {index}: the Qwen3.8 engine takes token ids, not rows; send token ids"
                )));
            }
        };
        if segment_tokens.is_empty() {
            return Err(refused(format!(
                "segment {index} has no tokens; give each segment at least one token"
            )));
        }
        if segment.positions.len() != segment_tokens.len() {
            return Err(refused(format!(
                "segment {index} has {} tokens and {} positions",
                segment_tokens.len(),
                segment.positions.len()
            )));
        }
        for (offset, position) in segment.positions.iter().enumerate() {
            let want = context.n + offset as u64;
            if *position != want {
                return Err(refused(format!(
                    "segment {index} has position {position}, but the Qwen3.8 engine appends at {want}: the context holds {} tokens; start the segment at {want}",
                    context.n
                )));
            }
        }
        if !segment.meta.is_empty() {
            return Err(refused(format!(
                "segment {index} has {} metadata bytes, but the Qwen3.8 engine reads no per-token metadata; send none",
                segment.meta.len()
            )));
        }
        for token in segment_tokens {
            if *token >= spec::VOCAB {
                return Err(refused(format!(
                    "segment {index}: token {token} is past the vocabulary of {}; send an id below that count",
                    spec::VOCAB
                )));
            }
        }
        let mut want_logits = false;
        if !segment.outputs.logit_rows.is_empty() {
            let last = *segment.positions.last().expect("the token count is not zero");
            if segment.outputs.logit_rows != [last] {
                return Err(refused(format!(
                    "segment {index} asks for logit rows {:?}, but the Qwen3.8 engine gives only the row at its last token {last}; ask for that row",
                    segment.outputs.logit_rows
                )));
            }
            want_logits = true;
        }
        plan.contexts.push(context.id);
        plan.lens.push(segment_tokens.len() as u32);
        plan.tokens.extend(segment_tokens.iter().map(|token| *token as i32));
        plan.wants.push(want_logits);
        plan.control.push(segment.control.clone());
    }
    control::tap_layout(&plan.wants, plan.taps).map_err(refused)?;
    if plan.tokens.len() > spec::MAX_PASS_TOKENS as usize {
        return Err(refused(format!(
            "the pass holds {} tokens, but the Qwen3.8 engine takes at most {}; split the pass",
            plan.tokens.len(),
            spec::MAX_PASS_TOKENS
        )));
    }
    Ok(plan)
}

impl Engine for Qwen38Engine {
    fn identity(&self) -> Identity {
        self.worker.spec().identity.clone()
    }

    fn limits(&self) -> Limits {
        self.worker.spec().limits.clone()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        if segments.is_empty() {
            return Err(self.failed("a pass needs at least one segment".to_owned()));
        }
        if segments.len() > spec::MAX_SEGMENTS as usize {
            return Err(self.failed(format!(
                "a pass holds {} segments, but the Qwen3.8 engine takes at most {}; split the pass",
                segments.len(),
                spec::MAX_SEGMENTS
            )));
        }
        let mut contexts = Vec::with_capacity(segments.len());
        for (index, segment) in segments.iter().enumerate() {
            let Some(id) = self.context(segment.state) else {
                return Err(self.failed(format!("segment {index}: {:?} is not in this engine", segment.state)));
            };
            let n = match self.worker.n(id) {
                Ok(n) => u64::from(n),
                Err(error) => return Err(self.failed(format!("segment {index}: {error}"))),
            };
            contexts.push(SegmentContext { id, n });
        }
        let plan = match plan_pass(segments, taps, &contexts) {
            Ok(plan) => plan,
            Err(error) => return Err(self.failed(error.message)),
        };

        let want_logits = plan.want_logits();
        let reply = match self.worker.pass(PassRequest {
            contexts: plan.contexts,
            lens: plan.lens,
            tokens: plan.tokens,
            want_logits,
            wants: plan.wants.clone(),
            control: plan.control,
            taps: plan.taps,
        }) {
            Ok(reply) => reply,
            Err(PassProblem { message, poison }) => {
                self.last_error = Some(message.clone());
                return Err(PassError {
                    kind: if poison {
                        PassErrorKind::Ran(PassStatus::Poison)
                    } else {
                        PassErrorKind::Refused
                    },
                    message,
                });
            }
        };

        let mut outputs = Vec::new();
        if let Some(bytes) = reply.logits {
            let row = spec::VOCAB as usize * 4;
            match logit_outputs(&bytes, &plan.wants, row) {
                Ok(found) => outputs = found,
                Err(error) => {
                    self.last_error = Some(error.message.clone());
                    return Err(error);
                }
            }
        }
        match tap_outputs(reply.taps, &plan.wants, plan.taps) {
            Ok(found) => outputs.extend(found),
            Err(error) => {
                self.last_error = Some(error.message.clone());
                return Err(error);
            }
        }
        Ok(PassOut {
            status: PassStatus::Ok,
            ms: reply.ms.max(0.0).round() as u64,
            outputs,
        })
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.clone()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        Some(self)
    }

    fn seal(&mut self) -> Option<&mut dyn Seal> {
        Some(self)
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        Some(self)
    }

    fn control(&mut self) -> Option<&mut dyn Control> {
        Some(self)
    }
}

/// The table of control vectors (engine-abi "The residual of the Qwen3.8 engine"). The write goes to the worker thread,
/// which owns the device.
impl Control for Qwen38Engine {
    fn control_write(&mut self, index: u32, bytes: &[u8]) -> Result<(), String> {
        let result = self.worker.control_write(index, bytes.to_vec());
        if let Err(message) = &result {
            self.last_error = Some(message.clone());
        }
        result
    }
}

impl Qwen38Engine {
    /// Map one worker call onto a `WriteResult`.
    fn write_result(&mut self, call: Call) -> WriteResult {
        match call {
            Call::Ok => WriteResult::Ok,
            Call::Refused(message) => {
                self.last_error = Some(message);
                WriteResult::Refused
            }
            Call::Failed(message) => {
                self.last_error = Some(message);
                WriteResult::Failed
            }
        }
    }
}

impl Parts for Qwen38Engine {
    fn part_read(&mut self, state: State, part: usize, out: &mut [u8]) -> Result<(), PartError> {
        let Some(id) = self.context(state) else {
            self.last_error = Some(format!("{state:?} is not in this engine"));
            return Err(PartError::Refused);
        };
        match self.worker.read(id, part, out.len()) {
            Ok(bytes) => {
                if bytes.len() != out.len() {
                    self.last_error = Some(format!(
                        "part {part} gave {} bytes; the buffer holds {}",
                        bytes.len(),
                        out.len()
                    ));
                    return Err(PartError::Failed);
                }
                out.copy_from_slice(&bytes);
                Ok(())
            }
            Err(Call::Refused(message)) => {
                self.last_error = Some(message);
                Err(PartError::Refused)
            }
            Err(Call::Failed(message)) => {
                self.last_error = Some(message);
                Err(PartError::Failed)
            }
            Err(Call::Ok) => {
                self.last_error = Some("the read reply was empty".to_owned());
                Err(PartError::Failed)
            }
        }
    }

    fn part_write(&mut self, state: State, part: usize, bytes: &[u8]) -> WriteResult {
        let Some(id) = self.context(state) else {
            self.last_error = Some(format!("{state:?} is not in this engine"));
            return WriteResult::Refused;
        };
        let call = self.worker.write(id, part, bytes.to_vec());
        self.write_result(call)
    }

    /// The engine has no range read. A read from 0 of all the context's rows is a whole read, and the engine gives it.
    fn cells_read(&mut self, state: State, part: usize, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        if let Err(message) = spec::whole_read_problem(part, from, n, out.len()) {
            self.last_error = Some(message);
            return Err(PartError::Refused);
        }
        // The worker refuses a buffer that does not hold the context's n rows, so `n` must be the token count.
        self.part_read(state, part, out)
    }

    fn cells_write(&mut self, state: State, part: usize, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        if let Err(message) = write_from(from) {
            self.last_error = Some(message);
            return WriteResult::Refused;
        }
        let Some(id) = self.context(state) else {
            self.last_error = Some(format!("{state:?} is not in this engine"));
            return WriteResult::Refused;
        };
        if n > u64::from(u32::MAX) {
            self.last_error = Some(format!(
                "{n} rows do not fit the 32-bit count of the Qwen3.8 engine; read fewer rows"
            ));
            return WriteResult::Refused;
        }
        let call = self.worker.cells_write(id, part, n as u32, bytes.to_vec());
        self.write_result(call)
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        let Some(id) = self.context(state) else {
            self.last_error = Some(format!("{state:?} is not in this engine"));
            return WriteResult::Refused;
        };
        if n > u64::from(u32::MAX) {
            self.last_error = Some(format!(
                "{n} tokens do not fit the 32-bit count of the Qwen3.8 engine; read fewer tokens"
            ));
            return WriteResult::Refused;
        }
        let call = self.worker.set_n(id, n as u32);
        self.write_result(call)
    }
}

/// The engine's contexts are the slots of v0 (engine-abi, "v0 binding choices").
impl Slots for Qwen38Engine {
    /// The context's token count: the rows of each cells part. A slot with no context holds no row.
    ///
    /// A worker that stopped cannot answer, and the trait has no error, so this panics (spec 8.5).
    fn slot_high(&self, slot: SlotId) -> u64 {
        let Some(id) = self.context(State::Slot(slot)) else {
            return 0;
        };
        match self.worker.n(id) {
            Ok(n) => u64::from(n),
            Err(error) => panic!("cannot read the token count of the engine slot {slot}: {error}"),
        }
    }

    /// Free the context and make a new one of the same capacity. The engine has no call that empties a context.
    ///
    /// The context term holds the pool (spec 4.2), so the new context needs no new admission. When the engine cannot
    /// make it, the slot has no context, and the result is `Failed`.
    fn slot_clear(&mut self, slot: SlotId) -> WriteResult {
        let Some(id) = self.context(State::Slot(slot)) else {
            self.last_error = Some(format!("slot {slot} has no context in this engine"));
            return WriteResult::Refused;
        };
        let call = self.worker.clear(id);
        if matches!(call, Call::Failed(_)) {
            self.states[slot as usize] = None;
        }
        self.write_result(call)
    }
}

/// The engine holds the rows of each cells part from position 0. The bytes give the row count.
impl Coverage for Qwen38Engine {
    fn coverage(&self, _state: State, part: PartId, bytes: &[u8]) -> CoverageSpan {
        spec::coverage_of(part, bytes.len())
    }

    fn fits(&self, coverage: &CoverageSpan, slot: SlotId) -> Result<(), String> {
        spec::fits(coverage, slot)
    }
}

impl Seal for Qwen38Engine {
    fn seal(&mut self, state: State) -> Result<(), SealError> {
        let Some(id) = self.context(state) else {
            self.last_error = Some(format!("{state:?} is not in this engine"));
            return Err(SealError::Refused);
        };
        match self.worker.seal(id) {
            Call::Ok => Ok(()),
            Call::Refused(message) => {
                self.last_error = Some(message);
                Err(SealError::Refused)
            }
            Call::Failed(message) => {
                self.last_error = Some(message);
                Err(SealError::Failed)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use nagaya_abi::{
        ControlItem, ControlOp, Dtype, OutputsWanted, PassErrorKind, PassStatus, Segment, SegmentInputs, State, Tap,
    };

    use super::{SegmentContext, logit_outputs, plan_pass, tap_outputs, write_from};
    use crate::control::TapSet;
    use crate::spec;

    #[test]
    fn a_write_that_is_not_at_row_zero_is_refused() {
        assert_eq!(write_from(0), Ok(()));
        let error = write_from(2).expect_err("the engine cannot place rows at 2");
        assert!(error.contains("rows from 0"), "{error}");
    }

    /// A segment that asks for `tokens` at `start`.
    fn token_segment(tokens: &[u32], start: u64) -> Segment {
        Segment {
            control: Vec::new(),
            state: State::Slot(0),
            inputs: SegmentInputs::Tokens(tokens.to_vec()),
            positions: (start..start + tokens.len() as u64).collect(),
            meta: Vec::new(),
            outputs: OutputsWanted::default(),
        }
    }

    #[test]
    fn a_segment_that_breaks_an_engine_rule_is_refused_before_the_pass() {
        // The context holds three tokens, so the engine appends at position 3. plan_pass takes no worker, so these
        // checks need no GPU and no library.
        let contexts = [SegmentContext { id: 7, n: 3 }];

        let wrong = token_segment(&[11, 12], 4);
        let error = plan_pass(std::slice::from_ref(&wrong), &[], &contexts).expect_err("position 4 is not 3");
        assert_eq!(error.kind, PassErrorKind::Refused, "a refused pass did not run");
        assert!(error.message.contains("appends at 3"), "{}", error.message);

        // 248_320 is one past the last token id.
        let past = token_segment(&[spec::VOCAB], 3);
        let error =
            plan_pass(std::slice::from_ref(&past), &[], &contexts).expect_err("the token is past the vocabulary");
        assert_eq!(error.kind, PassErrorKind::Refused, "a refused pass did not run");
        assert!(error.message.contains("past the vocabulary"), "{}", error.message);

        // The engine takes token ids, not rows.
        let mut rows = token_segment(&[11], 3);
        rows.inputs = SegmentInputs::Rows(vec![0; 4]);
        let error = plan_pass(std::slice::from_ref(&rows), &[], &contexts).expect_err("rows are not token ids");
        assert_eq!(error.kind, PassErrorKind::Refused, "a refused pass did not run");
        assert!(error.message.contains("not rows"), "{}", error.message);
    }

    /// A segment of one token that asks for its logit row.
    fn logit_segment(token: u32, start: u64) -> Segment {
        let mut segment = token_segment(&[token], start);
        segment.outputs.logit_rows = vec![start];
        segment
    }

    fn item(index: u32, op: ControlOp, strength: f32, layers: Option<Vec<(u32, f32)>>) -> ControlItem {
        ControlItem {
            index,
            op,
            strength,
            layers,
        }
    }

    #[test]
    fn a_segment_with_a_control_list_that_the_table_refuses_is_refused_before_the_pass() {
        let contexts = [SegmentContext { id: 7, n: 3 }];
        // The table has 8 entries, 0 to 7.
        let mut segment = token_segment(&[11], 3);
        segment.control = vec![item(8, ControlOp::Add, 1.0, None)];
        let error = plan_pass(std::slice::from_ref(&segment), &[], &contexts).expect_err("entry 8 is not in the table");
        assert_eq!(error.kind, PassErrorKind::Refused, "a refused pass did not run");
        assert!(error.message.contains("segment 0"), "{}", error.message);
        assert!(error.message.contains("entry 8"), "{}", error.message);
        // A strength that is not finite.
        segment.control = vec![item(0, ControlOp::Add, f32::NAN, None)];
        let error = plan_pass(std::slice::from_ref(&segment), &[], &contexts).expect_err("NaN is not a strength");
        assert!(error.message.contains("finite"), "{}", error.message);
        // A layer that the model does not have: the message names the layer and the count.
        segment.control = vec![item(0, ControlOp::Add, 1.0, Some(vec![(48, 1.0)]))];
        let error = plan_pass(std::slice::from_ref(&segment), &[], &contexts).expect_err("layer 48 does not exist");
        assert!(error.message.contains("layer 48"), "{}", error.message);
        assert!(error.message.contains("48 layers"), "{}", error.message);
    }

    #[test]
    fn a_good_control_list_goes_into_the_plan_with_its_segment() {
        let contexts = [SegmentContext { id: 7, n: 3 }, SegmentContext { id: 8, n: 0 }];
        let plain = token_segment(&[11], 3);
        let mut steered = token_segment(&[12], 0);
        steered.control = vec![
            item(3, ControlOp::Clamp, 0.0, Some(vec![(8, 0.0)])),
            item(0, ControlOp::Ablate, 1.0, None),
        ];
        let plan = plan_pass(&[plain, steered.clone()], &[], &contexts).expect("the lists fit the table");
        assert_eq!(plan.control.len(), 2);
        assert!(plan.control[0].is_empty(), "segment 0 has no list");
        assert_eq!(plan.control[1], steered.control, "segment 1 keeps its list, in order");
    }

    #[test]
    fn the_control_list_of_a_later_segment_is_refused_by_its_segment_number() {
        let contexts = [SegmentContext { id: 7, n: 3 }, SegmentContext { id: 8, n: 0 }];
        let plain = token_segment(&[11], 3);
        let mut steered = token_segment(&[12], 0);
        steered.control = vec![item(9, ControlOp::Clamp, 0.0, None)];
        let error = plan_pass(&[plain, steered], &[], &contexts).expect_err("segment 1 has a bad list");
        assert!(error.message.contains("segment 1"), "{}", error.message);
    }

    #[test]
    fn a_pass_that_names_an_unknown_tap_is_refused_and_the_message_names_the_taps_of_the_engine() {
        let contexts = [SegmentContext { id: 7, n: 3 }];
        let segments = [logit_segment(11, 3)];
        for name in ["lens", "logits", ""] {
            let taps = [Tap { name: name.to_owned() }];
            let error = plan_pass(&segments, &taps, &contexts).expect_err("the engine has no such tap");
            assert_eq!(error.kind, PassErrorKind::Refused, "a refused pass did not run");
            assert!(
                error.message.contains(&format!("names the tap `{name}`")),
                "{}",
                error.message
            );
            assert!(error.message.contains("`residual` and `fold`"), "{}", error.message);
            assert!(error.message.contains("name one of them"), "{}", error.message);
        }
    }

    #[test]
    fn the_two_taps_of_the_engine_are_accepted_with_a_logit_row_and_refused_without() {
        let contexts = [SegmentContext { id: 7, n: 3 }];
        let both = [
            Tap {
                name: "residual".to_owned(),
            },
            Tap {
                name: "fold".to_owned(),
            },
        ];
        let plan = plan_pass(&[logit_segment(11, 3)], &both, &contexts).expect("both taps, one logit row");
        assert!(plan.taps.residual && plan.taps.fold);
        // A tap with no logit row would give nothing, so the pass is refused before it runs.
        let error = plan_pass(&[token_segment(&[11], 3)], &both, &contexts).expect_err("no logit row");
        assert_eq!(error.kind, PassErrorKind::Refused);
        assert!(
            error.message.contains("no segment asks for a logit row"),
            "{}",
            error.message
        );
        // The same tap twice.
        let twice = [both[0].clone(), both[0].clone()];
        let error = plan_pass(&[logit_segment(11, 3)], &twice, &contexts).expect_err("a tap comes twice");
        assert!(error.message.contains("twice"), "{}", error.message);
    }

    #[test]
    fn the_tap_outputs_have_the_names_the_dtype_the_count_and_the_sizes_of_the_layout() {
        let taps = TapSet {
            residual: true,
            fold: true,
        };
        // Segments 0 and 2 ask for a logit row, so each tap has 2 outputs: all `residual`, then all `fold`.
        let wants = [true, false, true];
        let bytes = |name: &'static str, size: usize, fill: u8| (name, vec![fill; size]);
        let given = vec![
            bytes("residual", 1_966_080, 1),
            bytes("residual", 1_966_080, 2),
            bytes("fold", 10_240, 3),
            bytes("fold", 10_240, 4),
        ];
        let outputs = tap_outputs(given.clone(), &wants, taps).expect("the bytes follow the layout");
        let names: Vec<&str> = outputs.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(names, ["residual", "residual", "fold", "fold"]);
        assert!(outputs.iter().all(|o| o.dtype == Dtype::F32));
        let sizes: Vec<usize> = outputs.iter().map(|o| o.bytes.len()).collect();
        assert_eq!(sizes, [1_966_080, 1_966_080, 10_240, 10_240]);
        assert_eq!(outputs[1].bytes[0], 2, "the outputs keep the order of the rows");
        // A pass with no tap gives no output.
        assert!(tap_outputs(Vec::new(), &wants, TapSet::default()).unwrap().is_empty());
        // One output short, one of the wrong size and one of the wrong name are faults after the pass.
        for broken in [
            given[..3].to_vec(),
            vec![
                bytes("residual", 100, 1),
                given[1].clone(),
                given[2].clone(),
                given[3].clone(),
            ],
            vec![given[2].clone(), given[1].clone(), given[0].clone(), given[3].clone()],
        ] {
            let error = tap_outputs(broken, &wants, taps).expect_err("the bytes break the layout");
            assert_eq!(error.kind, PassErrorKind::Ran(PassStatus::Poison));
        }
    }

    #[test]
    fn a_pass_that_the_engine_can_run_plans_its_call() {
        let contexts = [SegmentContext { id: 7, n: 3 }];
        let segments = [token_segment(&[11, 12], 3)];
        let plan = plan_pass(&segments, &[], &contexts).expect("the engine can run this pass");
        assert_eq!(plan.contexts, [7]);
        assert_eq!(plan.lens, [2]);
        assert_eq!(plan.tokens, [11, 12]);
        assert_eq!(plan.wants, [false]);
        assert!(!plan.want_logits());
    }

    #[test]
    fn a_short_logits_buffer_is_a_fault_after_the_pass() {
        let error = logit_outputs(&[0u8; 16], &[true, true], 16).expect_err("a short buffer must fail");
        assert_eq!(
            error.kind,
            PassErrorKind::Ran(PassStatus::Poison),
            "the pass ran, so a short buffer poisons the state"
        );
        assert_eq!(
            error.message,
            "the Qwen3.8 engine returned 16 logits bytes, but segment 1 needs bytes 16..32"
        );
    }

    #[test]
    fn a_complete_logits_buffer_gives_one_output_for_each_wanted_row() {
        let bytes: Vec<u8> = (0..48u8).collect();
        let outputs = logit_outputs(&bytes, &[true, false, true], 16).expect("the buffer holds every row");
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].bytes, bytes[0..16]);
        assert_eq!(outputs[1].bytes, bytes[32..48]);
        assert!(outputs.iter().all(|output| output.name == "logits"));
    }
}
