//! The `Engine` trait on top of the mk binding.

use std::sync::{Arc, OnceLock};

use nagaya_abi::{
    Coverage, Engine, Identity, Limits, Output, PartError, Parts, PassError, PassOut, PassStatus, Seal, SealError,
    Segment, SegmentInputs, Slots, State, Tap, WriteResult,
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

    /// True when mk returns a pass that stopped on a spin-timeout instead of stopping the program.
    pub fn spin_recovery(&self) -> bool {
        self.worker.spec().recover_spin_timeouts
    }

    /// mk's byte count for one stored part and token count.
    pub fn part_bytes(&self, part: u32, n: u32) -> Result<u64, MkError> {
        spec::stored_part(part)?;
        self.worker
            .part_bytes(part, n)
            .ok_or_else(|| MkError::new("the mk worker stopped"))
    }

    /// Record one failed call and return its error.
    fn failed(&mut self, message: String) -> PassError {
        self.last_error = Some(message.clone());
        PassError {
            status: PassStatus::Ok,
            message,
        }
    }
}

impl Drop for MkEngine {
    fn drop(&mut self) {
        let ids = std::mem::take(&mut self.states);
        self.worker.free_states(ids);
    }
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
        let mut lens = Vec::with_capacity(segments.len());
        let mut tokens: Vec<i32> = Vec::new();
        let mut wants = Vec::with_capacity(segments.len());
        for (index, segment) in segments.iter().enumerate() {
            let Some(id) = self.context(segment.state) else {
                return Err(self.failed(format!("segment {index}: {:?} is not in this engine", segment.state)));
            };
            let n = match self.worker.n(id) {
                Ok(n) => u64::from(n),
                Err(error) => return Err(self.failed(format!("segment {index}: {error}"))),
            };
            let segment_tokens = match &segment.inputs {
                SegmentInputs::Tokens(tokens) => tokens,
                _ => return Err(self.failed(format!("segment {index}: mk takes token ids, not rows"))),
            };
            if segment_tokens.is_empty() {
                return Err(self.failed(format!("segment {index} has no tokens; mk needs at least one")));
            }
            if segment.positions.len() != segment_tokens.len() {
                return Err(self.failed(format!(
                    "segment {index} has {} tokens and {} positions",
                    segment_tokens.len(),
                    segment.positions.len()
                )));
            }
            for (offset, position) in segment.positions.iter().enumerate() {
                let want = n + offset as u64;
                if *position != want {
                    return Err(self.failed(format!(
                        "segment {index} has position {position} where mk appends at {want}: the context holds {n} tokens"
                    )));
                }
            }
            if !segment.meta.is_empty() {
                return Err(self.failed(format!(
                    "segment {index} has {} metadata bytes; mk reads no per-token metadata",
                    segment.meta.len()
                )));
            }
            for token in segment_tokens {
                if *token >= spec::VOCAB {
                    return Err(self.failed(format!(
                        "segment {index}: token {token} is past mk's vocabulary of {}",
                        spec::VOCAB
                    )));
                }
            }
            let mut want_logits = false;
            if !segment.outputs.logit_rows.is_empty() {
                let last = *segment.positions.last().expect("the token count is not zero");
                if segment.outputs.logit_rows != [last] {
                    return Err(self.failed(format!(
                        "segment {index} asks for logit rows {:?}; mk gives the row at its last token {last}",
                        segment.outputs.logit_rows
                    )));
                }
                want_logits = true;
            }
            contexts.push(id);
            lens.push(segment_tokens.len() as u32);
            tokens.extend(segment_tokens.iter().map(|token| *token as i32));
            wants.push(want_logits);
        }
        if tokens.len() > spec::MAX_PASS_TOKENS as usize {
            return Err(self.failed(format!(
                "the pass holds {} tokens; mk takes at most {}",
                tokens.len(),
                spec::MAX_PASS_TOKENS
            )));
        }

        let want_logits = wants.iter().any(|want| *want);
        let reply = match self.worker.pass(PassRequest {
            contexts,
            lens,
            tokens,
            want_logits,
        }) {
            Ok(reply) => reply,
            Err(PassProblem { message, poison }) => {
                self.last_error = Some(message.clone());
                return Err(PassError {
                    status: if poison { PassStatus::Poison } else { PassStatus::Ok },
                    message,
                });
            }
        };

        let mut outputs = Vec::new();
        if let Some(bytes) = reply.logits {
            let row = spec::VOCAB as usize * 4;
            for (index, want) in wants.iter().enumerate() {
                if !*want {
                    continue;
                }
                let start = index * row;
                let end = start + row;
                let Some(slice) = bytes.get(start..end) else {
                    return Err(self.failed(format!(
                        "mk returned {} logits bytes; segment {index} needs bytes {start}..{end}",
                        bytes.len()
                    )));
                };
                outputs.push(Output {
                    name: "logits".to_owned(),
                    bytes: slice.to_vec(),
                });
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

    fn cells_write(&mut self, state: State, part: usize, n: u64, bytes: &[u8]) -> WriteResult {
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
