//! The `Engine` trait on top of the mk binding.

use std::sync::{Arc, OnceLock};

use nagaya_abi::{
    Coverage, Dtype, Engine, Identity, Limits, Output, PartError, Parts, PassError, PassErrorKind, PassOut, PassStatus,
    Seal, SealError, Segment, SegmentInputs, Slots, State, Tap, WriteResult,
};

use crate::spec;
use crate::worker::{Call, MkError, MkPaths, PassProblem, PassRequest, Worker, start};

/// The one engine in this process. mk aborts on close, so the engine stays open until the process exits (spec 3.3).
static WORKER: OnceLock<Result<Arc<Worker>, MkError>> = OnceLock::new();

/// Open the mk engine, or make a new handle on the one that is open.
///
/// The first call loads the weights. Every later call creates a new context pool for the handle. Every mk call goes
/// to one worker thread, because the HIP device is the calling thread's current device.
pub fn open() -> Result<MkEngine, MkError> {
    let paths = MkPaths::from_env()?;
    let worker = match WORKER.get_or_init(|| start(paths.clone())) {
        Ok(worker) => Arc::clone(worker),
        Err(error) => return Err(error.clone()),
    };
    if worker.paths() != &paths {
        return Err(MkError::new(
            "MK_ENGINE_LIB or MK_SHARDS changed after the engine opened".to_owned(),
        ));
    }
    let states = worker.open_states(spec::CONTEXT_COUNT, spec::CONTEXT_CAPACITY)?;
    Ok(MkEngine {
        worker,
        states,
        last_error: None,
    })
}

/// The mk engine: a context pool, the engine's declared facts, and the last error.
pub struct MkEngine {
    worker: Arc<Worker>,
    states: Vec<u64>,
    last_error: Option<String>,
}

impl MkEngine {
    /// The context id for one state.
    fn context(&self, state: State) -> Option<u64> {
        let State::Slot(slot) = state;
        self.states.get(slot as usize).copied()
    }

    /// The context's token count.
    pub fn n(&self, state: State) -> Result<u64, MkError> {
        let id = self
            .context(state)
            .ok_or_else(|| MkError::new(format!("{state:?} is not in this engine")))?;
        self.worker.n(id).map(u64::from)
    }

    /// The context's capacity in tokens.
    pub fn capacity(&self, state: State) -> Result<u64, MkError> {
        let id = self
            .context(state)
            .ok_or_else(|| MkError::new(format!("{state:?} is not in this engine")))?;
        self.worker.max(id).map(u64::from)
    }

    /// True when the context is poisoned.
    pub fn poisoned(&self, state: State) -> Result<bool, MkError> {
        let id = self
            .context(state)
            .ok_or_else(|| MkError::new(format!("{state:?} is not in this engine")))?;
        self.worker.poisoned(id)
    }

    /// True when the context owns its K/V. A fork does not.
    pub fn owns_kv(&self, state: State) -> Result<bool, MkError> {
        let id = self
            .context(state)
            .ok_or_else(|| MkError::new(format!("{state:?} is not in this engine")))?;
        self.worker.owns_kv(id)
    }

    /// mk's byte count for one stored part and token count.
    pub fn part_bytes(&self, part: u32, n: u32) -> Result<u64, MkError> {
        spec::stored_part(part)?;
        self.worker
            .part_bytes(part, n)
            .ok_or_else(|| MkError::new("the mk worker stopped"))
    }

    /// The pass scratch that the probe measured at open, in bytes (milestone 9c).
    pub fn pass_scratch(&self) -> u64 {
        self.worker.spec().pass_scratch
    }

    /// Record one refused pass and return its error. No context changed.
    fn failed(&mut self, message: String) -> PassError {
        self.last_error = Some(message.clone());
        refused(message)
    }
}

/// The first row of a cells write. mk writes rows from 0 and cannot place them later (ABI "Proposed: the first row
/// of a cells write").
fn write_from(from: u64) -> Result<(), String> {
    if from == 0 {
        Ok(())
    } else {
        Err(format!(
            "mk cannot place rows at position {from}; it writes rows from 0"
        ))
    }
}

impl Drop for MkEngine {
    fn drop(&mut self) {
        let ids = std::mem::take(&mut self.states);
        self.worker.free_states(ids);
    }
}

/// Build the logit outputs from mk's bytes.
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
                    "mk returned {} logits bytes; segment {index} needs bytes {start}..{end}",
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

/// The mk call for one pass.
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

/// Build the mk call for one pass, or say why the binding refuses it.
///
/// Every check here runs before the binding calls mk. Thus a refused pass changes no context.
fn plan_pass(segments: &[Segment], contexts: &[SegmentContext]) -> Result<PassPlan, PassError> {
    assert_eq!(segments.len(), contexts.len(), "every segment needs its context");
    let mut plan = PassPlan::default();
    for (index, segment) in segments.iter().enumerate() {
        let context = contexts[index];
        let segment_tokens = match &segment.inputs {
            SegmentInputs::Tokens(tokens) => tokens,
            _ => return Err(refused(format!("segment {index}: mk takes token ids, not rows"))),
        };
        if segment_tokens.is_empty() {
            return Err(refused(format!("segment {index} has no tokens; mk needs at least one")));
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
                    "segment {index} has position {position} where mk appends at {want}: the context holds {} tokens",
                    context.n
                )));
            }
        }
        if !segment.meta.is_empty() {
            return Err(refused(format!(
                "segment {index} has {} metadata bytes; mk reads no per-token metadata",
                segment.meta.len()
            )));
        }
        for token in segment_tokens {
            if *token >= spec::VOCAB {
                return Err(refused(format!(
                    "segment {index}: token {token} is past mk's vocabulary of {}",
                    spec::VOCAB
                )));
            }
        }
        let mut want_logits = false;
        if !segment.outputs.logit_rows.is_empty() {
            let last = *segment.positions.last().expect("the token count is not zero");
            if segment.outputs.logit_rows != [last] {
                return Err(refused(format!(
                    "segment {index} asks for logit rows {:?}; mk gives the row at its last token {last}",
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
            "the pass holds {} tokens; mk takes at most {}",
            plan.tokens.len(),
            spec::MAX_PASS_TOKENS
        )));
    }
    Ok(plan)
}

impl Engine for MkEngine {
    fn identity(&self) -> Identity {
        self.worker.spec().identity.clone()
    }

    fn limits(&self) -> Limits {
        self.worker.spec().limits.clone()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        if !taps.is_empty() {
            return Err(self.failed("mk taps need the unfused path and are not in this binding".to_owned()));
        }
        if segments.is_empty() {
            return Err(self.failed("a pass needs at least one segment".to_owned()));
        }
        if segments.len() > spec::MAX_SEGMENTS as usize {
            return Err(self.failed(format!(
                "a pass holds {} segments; mk takes at most {}",
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
        let plan = match plan_pass(segments, &contexts) {
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
        None
    }

    fn seal(&mut self) -> Option<&mut dyn Seal> {
        Some(self)
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        None
    }
}

impl MkEngine {
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

impl Parts for MkEngine {
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

    fn cells_read(
        &mut self,
        _state: State,
        part: usize,
        _from: u64,
        _n: u64,
        _out: &mut [u8],
    ) -> Result<(), PartError> {
        self.last_error = Some(format!(
            "mk has no range read for part {part}; read the whole part, which holds the context's n rows"
        ));
        Err(PartError::Refused)
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
            self.last_error = Some(format!("{n} rows do not fit mk's 32-bit count"));
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
            self.last_error = Some(format!("{n} tokens do not fit mk's 32-bit count"));
            return WriteResult::Refused;
        }
        let call = self.worker.set_n(id, n as u32);
        self.write_result(call)
    }
}

impl Seal for MkEngine {
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
    use nagaya_abi::{OutputsWanted, PassErrorKind, PassStatus, Segment, SegmentInputs, State};

    use super::{SegmentContext, logit_outputs, plan_pass, write_from};
    use crate::spec;

    #[test]
    fn a_write_that_is_not_at_row_zero_is_refused() {
        assert_eq!(write_from(0), Ok(()));
        let error = write_from(2).expect_err("mk cannot place rows at 2");
        assert!(error.contains("rows from 0"), "{error}");
    }

    /// A segment that asks for `tokens` at `start`.
    fn token_segment(tokens: &[u32], start: u64) -> Segment {
        Segment {
            state: State::Slot(0),
            inputs: SegmentInputs::Tokens(tokens.to_vec()),
            positions: (start..start + tokens.len() as u64).collect(),
            meta: Vec::new(),
            outputs: OutputsWanted::default(),
        }
    }

    #[test]
    fn a_segment_that_breaks_an_mk_rule_is_refused_before_the_pass() {
        // The context holds three tokens, so mk appends at position 3. plan_pass takes no worker, so these checks
        // need no GPU and no library.
        let contexts = [SegmentContext { id: 7, n: 3 }];

        let wrong = token_segment(&[11, 12], 4);
        let error = plan_pass(std::slice::from_ref(&wrong), &contexts).expect_err("position 4 is not 3");
        assert_eq!(error.kind, PassErrorKind::Refused, "a refused pass did not run");
        assert!(error.message.contains("appends at 3"), "{}", error.message);

        // 248_320 is one past the last token id.
        let past = token_segment(&[spec::VOCAB], 3);
        let error = plan_pass(std::slice::from_ref(&past), &contexts).expect_err("the token is past the vocabulary");
        assert_eq!(error.kind, PassErrorKind::Refused, "a refused pass did not run");
        assert!(error.message.contains("past mk's vocabulary"), "{}", error.message);

        // mk takes token ids, not rows.
        let mut rows = token_segment(&[11], 3);
        rows.inputs = SegmentInputs::Rows(vec![0; 4]);
        let error = plan_pass(std::slice::from_ref(&rows), &contexts).expect_err("rows are not token ids");
        assert_eq!(error.kind, PassErrorKind::Refused, "a refused pass did not run");
        assert!(error.message.contains("not rows"), "{}", error.message);
    }

    #[test]
    fn a_pass_that_mk_can_run_plans_its_call() {
        let contexts = [SegmentContext { id: 7, n: 3 }];
        let segments = [token_segment(&[11, 12], 3)];
        let plan = plan_pass(&segments, &contexts).expect("mk can run this pass");
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
            "mk returned 16 logits bytes; segment 1 needs bytes 16..32"
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
