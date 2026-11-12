//! The `Engine` trait on top of the binding of the Qwen3.8 engine.

use std::sync::{Arc, OnceLock};

use nagaya_abi::{
    Coverage, CoverageSpan, Dtype, Engine, Identity, Limits, Output, PartError, PartId, Parts, PassError,
    PassErrorKind, PassOut, PassStatus, Seal, SealError, Segment, SegmentInputs, SlotId, Slots, State, Tap,
    WriteResult,
};

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
pub fn describe() -> Result<(Identity, Limits), Qwen38Error> {
    crate::worker::describe(&Qwen38Paths::from_env()?)
}

/// Open the Qwen3.8 engine, or make a new handle on the one that is open.
///
/// The first call loads the weights, and so does the first call after the last handle dropped. Every call creates a
/// new context pool for the handle. Every engine call goes to one worker thread, because the HIP device is the calling
/// thread's current device.
pub fn open() -> Result<Qwen38Engine, Qwen38Error> {
    let worker = worker()?;
    worker.acquire()?;
    let pool = match worker.open_states(&spec::CONTEXT_CLASSES) {
        Ok(pool) => pool,
        Err(error) => {
            worker.release();
            return Err(error);
        }
    };
    Ok(Qwen38Engine {
        worker,
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
    worker.acquire()?;
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
/// This build has no tap and no control table. The ABI says that an engine without `control` refuses a segment with a
/// control list, so the refusal is here and not in the process.
fn plan_pass(segments: &[Segment], taps: &[Tap], contexts: &[SegmentContext]) -> Result<PassPlan, PassError> {
    assert_eq!(segments.len(), contexts.len(), "every segment needs its context");
    if let Some(tap) = taps.first() {
        return Err(refused(format!(
            "the pass names the tap `{}`, but this build gives no tap yet; the engine will give the taps `residual` and `fold`: send the pass with no tap",
            tap.name
        )));
    }
    let mut plan = PassPlan::default();
    for (index, segment) in segments.iter().enumerate() {
        let context = contexts[index];
        if !segment.control.is_empty() {
            return Err(refused(format!(
                "segment {index} has {} control items, but this build of the engine has no control table yet; send the segment with no control list",
                segment.control.len()
            )));
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
    }
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
        ControlItem, ControlOp, OutputsWanted, PassErrorKind, PassStatus, Segment, SegmentInputs, State, Tap,
    };

    use super::{SegmentContext, logit_outputs, plan_pass, write_from};
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

    #[test]
    fn a_segment_with_a_control_list_is_refused_before_the_pass() {
        let contexts = [SegmentContext { id: 7, n: 3 }];
        let mut segment = token_segment(&[11], 3);
        segment.control = vec![ControlItem {
            index: 0,
            op: ControlOp::Add,
            strength: 1.0,
            layers: None,
        }];
        let error =
            plan_pass(std::slice::from_ref(&segment), &[], &contexts).expect_err("this build has no control table");
        assert_eq!(error.kind, PassErrorKind::Refused, "a refused pass did not run");
        // The message says what failed, why, and what the reader can do.
        assert!(
            error.message.contains("segment 0 has 1 control items"),
            "{}",
            error.message
        );
        assert!(error.message.contains("has no control table yet"), "{}", error.message);
        assert!(
            error.message.contains("send the segment with no control list"),
            "{}",
            error.message
        );
    }

    #[test]
    fn the_control_list_of_a_later_segment_is_refused_too() {
        let contexts = [SegmentContext { id: 7, n: 3 }, SegmentContext { id: 8, n: 0 }];
        let plain = token_segment(&[11], 3);
        let mut steered = token_segment(&[12], 0);
        steered.control = vec![ControlItem {
            index: 3,
            op: ControlOp::Clamp,
            strength: 0.0,
            layers: Some(vec![(8, 0.0)]),
        }];
        let error = plan_pass(&[plain, steered], &[], &contexts).expect_err("segment 1 has a list");
        assert!(
            error.message.contains("segment 1 has 1 control items"),
            "{}",
            error.message
        );
    }

    #[test]
    fn a_pass_that_names_a_tap_is_refused_and_the_message_names_the_taps_of_the_engine() {
        let contexts = [SegmentContext { id: 7, n: 3 }];
        let segments = [token_segment(&[11], 3)];
        for name in ["residual", "fold", "lens"] {
            let taps = [Tap { name: name.to_owned() }];
            let error = plan_pass(&segments, &taps, &contexts).expect_err("this build has no tap");
            assert_eq!(error.kind, PassErrorKind::Refused, "a refused pass did not run");
            assert!(
                error.message.contains(&format!("names the tap `{name}`")),
                "{}",
                error.message
            );
            assert!(error.message.contains("`residual` and `fold`"), "{}", error.message);
            assert!(error.message.contains("this build gives no tap"), "{}", error.message);
            assert!(error.message.contains("send the pass with no tap"), "{}", error.message);
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
