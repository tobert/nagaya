//! The one thread that makes every mk call.
//!
//! mk's HIP device is the device that is current on the calling thread. One worker thread owns the engine, and every
//! call goes to it over a channel (spec 3.2). The engine is never closed: HIP teardown segfaults in the process (spec
//! 3.3).

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};

use nagaya_abi::{Identity, Limits};

use crate::binding::{self, Binding, PassStop};
use crate::gguf;
use crate::spec;
use crate::table::TensorTable;

/// An error from the mk binding. The message says what failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MkError {
    message: String,
}

impl MkError {
    /// Make an error with a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for MkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for MkError {}

impl From<String> for MkError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

/// Where the three weight shards are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MkPaths {
    /// The three shard paths (`MK_SHARDS`, with `:` between them).
    pub shards: Vec<PathBuf>,
}

/// Split `MK_SHARDS` on `:` as raw bytes, so a path that is not UTF-8 stays the same path.
fn parse_shards(raw: &[u8]) -> Result<Vec<PathBuf>, MkError> {
    let shards: Vec<PathBuf> = raw
        .split(|byte| *byte == b':')
        .map(|part| PathBuf::from(std::ffi::OsString::from_vec(part.to_vec())))
        .collect();
    if shards.len() != 3 || shards.iter().any(|path| path.as_os_str().is_empty()) {
        return Err(MkError::new(format!(
            "MK_SHARDS has {} paths; mk needs three, with : between them",
            shards.len()
        )));
    }
    Ok(shards)
}

impl MkPaths {
    /// Read the paths from `MK_SHARDS`.
    pub fn from_env() -> Result<Self, MkError> {
        let shards = std::env::var_os("MK_SHARDS")
            .ok_or_else(|| MkError::new("MK_SHARDS is not set; give three GGUF paths with : between them"))?;
        let shards = parse_shards(shards.as_os_str().as_bytes())?;
        Ok(Self { shards })
    }

    /// Check that every path exists and every shard has a GGUF v3 header.
    fn validate(&self) -> Result<Vec<gguf::ShardHeader>, MkError> {
        let mut headers = Vec::new();
        for path in &self.shards {
            if !path.is_file() {
                return Err(MkError::new(format!("the shard {} is missing", path.display())));
            }
            headers.push(gguf::read_header(path)?);
        }
        Ok(headers)
    }
}

/// What a fresh handle learns from the engine.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Spec {
    pub identity: Identity,
    pub limits: Limits,
    /// The pass scratch that the probe measured at open, in bytes.
    pub pass_scratch: u64,
}

/// Why a write or a seal refuses a context that was poisoned before the call. The call changed nothing (ABI "How mk
/// maps", part writes).
const POISONED: &str = "the context is poisoned; free it";

/// The result of a write or a seal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Call {
    /// The call ran and changed nothing unexpected.
    Ok,
    /// The call did not run, and nothing changed.
    Refused(String),
    /// The call changed the state and poisoned it.
    Failed(String),
}

/// A pass that did not run or ran and poisoned its contexts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PassProblem {
    /// Why the pass did not finish.
    pub message: String,
    /// True when mk ran the pass and poisoned its contexts.
    pub poison: bool,
}

/// One pass request, with the context ids in `lens` groups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PassRequest {
    /// The context ids, one for each segment.
    pub contexts: Vec<u64>,
    /// The token count of each segment.
    pub lens: Vec<u32>,
    /// The concatenated token ids.
    pub tokens: Vec<i32>,
    /// True when the caller wants one logits row for each segment.
    pub want_logits: bool,
}

/// The result of a pass that ran.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PassReply {
    /// The pass wall time in milliseconds.
    pub ms: f64,
    /// The logits, `[segments][vocab]` f32 little-endian, when asked.
    pub logits: Option<Vec<u8>>,
}

/// The worker thread's engine and its contexts.
struct ThreadState {
    binding: Binding,
    engine: *mut c_void,
    /// The reader's maps of the shards. The engine gathers PLE rows from them for its life. A stub has none.
    _model: Option<gguf::Model>,
    contexts: HashMap<u64, Context>,
    next_id: u64,
    fast_qmm: bool,
    tiles: bool,
}

/// One context that a handle owns.
struct Context {
    ptr: *mut c_void,
    capacity: u32,
    restored_kv: u32,
    restored_idx: u32,
    restored_ple: bool,
}

/// What one class of contexts took when mk made it. The host readings are `None` when `/proc/meminfo` fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassReading {
    /// The class.
    pub class: spec::ContextClass,
    /// The bytes that this binding declares for the class.
    pub declared: u64,
    /// The device free bytes that the class took (`hipMemGetInfo` before minus after).
    pub device: u64,
    /// The host's MemAvailable before the class, in bytes.
    pub available_before: Option<u64>,
    /// The host's MemAvailable after the class, in bytes.
    pub available_after: Option<u64>,
}

/// The contexts that one handle made, and what each class took.
#[derive(Clone, Debug)]
pub struct Pool {
    /// The context ids, in the order of the classes.
    pub ids: Vec<u64>,
    /// One reading for each class.
    pub readings: Vec<ClassReading>,
}

/// One command for the worker thread.
enum Command {
    Open {
        classes: Vec<spec::ContextClass>,
        reply: mpsc::Sender<Result<Pool, String>>,
    },
    Free {
        ids: Vec<u64>,
        reply: mpsc::Sender<()>,
    },
    N {
        id: u64,
        reply: mpsc::Sender<Result<u32, String>>,
    },
    Max {
        id: u64,
        reply: mpsc::Sender<Result<u32, String>>,
    },
    OwnsKv {
        id: u64,
        reply: mpsc::Sender<Result<bool, String>>,
    },
    Poisoned {
        id: u64,
        reply: mpsc::Sender<Result<bool, String>>,
    },
    Read {
        id: u64,
        part: u32,
        len: usize,
        reply: mpsc::Sender<Result<Vec<u8>, Call>>,
    },
    Write {
        id: u64,
        part: u32,
        bytes: Vec<u8>,
        reply: mpsc::Sender<Call>,
    },
    CellsWrite {
        id: u64,
        part: u32,
        n: u32,
        bytes: Vec<u8>,
        reply: mpsc::Sender<Call>,
    },
    SetN {
        id: u64,
        n: u32,
        reply: mpsc::Sender<Call>,
    },
    Seal {
        id: u64,
        reply: mpsc::Sender<Call>,
    },
    Clear {
        id: u64,
        reply: mpsc::Sender<Call>,
    },
    Pass {
        request: PassRequest,
        reply: mpsc::Sender<Result<PassReply, PassProblem>>,
    },
    PartBytes {
        part: u32,
        n: u32,
        reply: mpsc::Sender<u64>,
    },
}

/// The handle's side of the worker: a channel and the engine's declared facts.
pub(crate) struct Worker {
    sender: Mutex<mpsc::Sender<Command>>,
    spec: Spec,
    paths: MkPaths,
}

impl Worker {
    /// The identity and the limits of the open engine.
    pub fn spec(&self) -> &Spec {
        &self.spec
    }

    /// The paths that the engine was opened with.
    pub fn paths(&self) -> &MkPaths {
        &self.paths
    }

    /// Send one command and wait for its reply. `None` means the worker stopped.
    fn ask<T>(&self, make: impl FnOnce(mpsc::Sender<T>) -> Command) -> Option<T> {
        let (reply, answer) = mpsc::channel();
        let sender = self.sender.lock().ok()?;
        sender.send(make(reply)).ok()?;
        drop(sender);
        answer.recv().ok()
    }

    /// Create a context pool for one handle. The ids follow the order of the classes.
    pub fn open_states(&self, classes: &[spec::ContextClass]) -> Result<Pool, MkError> {
        let classes = classes.to_vec();
        self.ask(|reply| Command::Open { classes, reply })
            .unwrap_or_else(|| Err("the mk worker stopped".to_owned()))
            .map_err(MkError::from)
    }

    /// Free a handle's contexts.
    pub fn free_states(&self, ids: Vec<u64>) {
        if ids.is_empty() {
            return;
        }
        let _ = self.ask(|reply| Command::Free { ids, reply });
    }

    /// The context's token count.
    pub fn n(&self, id: u64) -> Result<u32, MkError> {
        self.ask(|reply| Command::N { id, reply })
            .unwrap_or_else(|| Err("the mk worker stopped".to_owned()))
            .map_err(MkError::from)
    }

    /// The context's capacity.
    pub fn max(&self, id: u64) -> Result<u32, MkError> {
        self.ask(|reply| Command::Max { id, reply })
            .unwrap_or_else(|| Err("the mk worker stopped".to_owned()))
            .map_err(MkError::from)
    }

    /// True when the context owns its K/V.
    pub fn owns_kv(&self, id: u64) -> Result<bool, MkError> {
        self.ask(|reply| Command::OwnsKv { id, reply })
            .unwrap_or_else(|| Err("the mk worker stopped".to_owned()))
            .map_err(MkError::from)
    }

    /// True when the context is poisoned.
    pub fn poisoned(&self, id: u64) -> Result<bool, MkError> {
        self.ask(|reply| Command::Poisoned { id, reply })
            .unwrap_or_else(|| Err("the mk worker stopped".to_owned()))
            .map_err(MkError::from)
    }

    /// Read one whole part into a new buffer of `len` bytes.
    pub fn read(&self, id: u64, part: usize, len: usize) -> Result<Vec<u8>, Call> {
        let part = part_id(part)?;
        self.ask(|reply| Command::Read { id, part, len, reply })
            .unwrap_or_else(|| Err(Call::Failed("the mk worker stopped".to_owned())))
    }

    /// Write one fixed part.
    pub fn write(&self, id: u64, part: usize, bytes: Vec<u8>) -> Call {
        let part = match part_id(part) {
            Ok(part) => part,
            Err(call) => return call,
        };
        self.ask(|reply| Command::Write { id, part, bytes, reply })
            .unwrap_or_else(|| Call::Failed("the mk worker stopped".to_owned()))
    }

    /// Write `n` rows of a cells part.
    pub fn cells_write(&self, id: u64, part: usize, n: u32, bytes: Vec<u8>) -> Call {
        let part = match part_id(part) {
            Ok(part) => part,
            Err(call) => return call,
        };
        self.ask(|reply| Command::CellsWrite {
            id,
            part,
            n,
            bytes,
            reply,
        })
        .unwrap_or_else(|| Call::Failed("the mk worker stopped".to_owned()))
    }

    /// Set the context's token count.
    pub fn set_n(&self, id: u64, n: u32) -> Call {
        self.ask(|reply| Command::SetN { id, n, reply })
            .unwrap_or_else(|| Call::Failed("the mk worker stopped".to_owned()))
    }

    /// Run mk's boundary rule.
    pub fn seal(&self, id: u64) -> Call {
        self.ask(|reply| Command::Seal { id, reply })
            .unwrap_or_else(|| Call::Failed("the mk worker stopped".to_owned()))
    }

    /// Empty the context: free it and make a new one of the same capacity under the same id.
    pub fn clear(&self, id: u64) -> Call {
        self.ask(|reply| Command::Clear { id, reply })
            .unwrap_or_else(|| Call::Failed("the mk worker stopped".to_owned()))
    }

    /// Run one pass.
    pub fn pass(&self, request: PassRequest) -> Result<PassReply, PassProblem> {
        self.ask(|reply| Command::Pass { request, reply }).unwrap_or_else(|| {
            Err(PassProblem {
                message: "the mk worker stopped".to_owned(),
                poison: false,
            })
        })
    }

    /// mk's byte count for one part and token count, or `None` when the worker stopped.
    pub fn part_bytes(&self, part: u32, n: u32) -> Option<u64> {
        self.ask(|reply| Command::PartBytes { part, n, reply })
    }
}

/// The 32-bit part id that mk takes, or a refusal. A cast would read part 2^32 + 3 as part 3.
fn part_id(part: usize) -> Result<u32, Call> {
    u32::try_from(part).map_err(|_| Call::Refused(format!("mk has no part {part}")))
}

/// Start the worker thread and open the engine on it.
pub(crate) fn start(paths: MkPaths) -> Result<Arc<Worker>, MkError> {
    let headers = paths.validate()?;
    let total: u64 = headers.iter().map(|header| header.size).sum();
    check_memory(total)?;

    let (sender, receiver) = mpsc::channel();
    let (ready, first) = mpsc::channel();
    let thread_paths = paths.clone();
    std::thread::Builder::new()
        .name("mk-engine".to_owned())
        .spawn(move || match ThreadState::open(&thread_paths) {
            Ok((state, spec)) => {
                if ready.send(Ok(spec)).is_ok() {
                    state.run(receiver);
                }
            }
            Err(error) => {
                let _ = ready.send(Err(error));
            }
        })
        .map_err(|error| MkError::new(format!("start the mk worker: {error}")))?;

    let spec = first
        .recv()
        .map_err(|error| MkError::new(format!("the mk worker did not answer: {error}")))??;
    Ok(Arc::new(Worker {
        sender: Mutex::new(sender),
        spec,
        paths,
    }))
}

/// Read `MK_FAST_QMM` as mk reads it: unset or 1 is fast, 0 is exact, and any other value stops mk.
fn fast_qmm() -> Result<bool, MkError> {
    match std::env::var("MK_FAST_QMM") {
        Ok(value) => fast_qmm_from(Some(&value)),
        Err(std::env::VarError::NotPresent) => fast_qmm_from(None),
        Err(error) => Err(MkError::new(format!("MK_FAST_QMM: {error}"))),
    }
}

/// Read `MK_TILES` as mk reads it: unset or 1 is tiled, 0 is the GGUF layout, and any other value stops mk.
fn tiles() -> Result<bool, MkError> {
    match std::env::var("MK_TILES") {
        Ok(value) => tiles_from(Some(&value)),
        Err(std::env::VarError::NotPresent) => tiles_from(None),
        Err(error) => Err(MkError::new(format!("MK_TILES: {error}"))),
    }
}

/// The value of one `MK_TILES` string.
fn tiles_from(value: Option<&str>) -> Result<bool, MkError> {
    match value {
        Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(other) => Err(MkError::new(format!("MK_TILES={other}: want 0 or 1"))),
        None => Ok(true),
    }
}

/// The value of one `MK_FAST_QMM` string.
fn fast_qmm_from(value: Option<&str>) -> Result<bool, MkError> {
    match value {
        Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(other) => Err(MkError::new(format!("MK_FAST_QMM={other}: want 0 or 1"))),
        None => Ok(true),
    }
}

/// The host safety floor. mk keeps 12 GiB of MemAvailable free (spec 4.7; mk `service/contexts.py`, `FLOOR_BYTES`).
const MEMORY_FLOOR: u64 = 12 << 30;

/// Refuse an allocation of `needed` bytes that leaves less than the floor free. `what` names the allocation.
fn memory_problem(what: &str, available: u64, needed: u64) -> Result<(), String> {
    if available < needed.saturating_add(MEMORY_FLOOR) {
        return Err(format!(
            "{what} takes {needed} bytes and mk puts them in memory; \
             {available} bytes are available and the floor is {MEMORY_FLOOR}"
        ));
    }
    Ok(())
}

/// The host's MemAvailable in bytes.
fn mem_available() -> Result<u64, MkError> {
    let text =
        std::fs::read_to_string("/proc/meminfo").map_err(|error| MkError::new(format!("/proc/meminfo: {error}")))?;
    text.lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .map(|kilobytes| kilobytes * 1024)
        .ok_or_else(|| MkError::new("/proc/meminfo has no MemAvailable"))
}

/// Refuse a load that cannot fit in the free memory. mk allocates the weight arena in full, and the floor stays free.
fn check_memory(needed: u64) -> Result<(), MkError> {
    memory_problem("the shards", mem_available()?, needed).map_err(MkError::new)
}

/// The device bytes that `what` took: the free bytes before minus the free bytes after.
///
/// Free memory that grew is an error with both readings. A saturating difference would read 0 bytes, and the class
/// check would pass (kaibo review, finding 7).
fn taken_bytes(before: u64, after: u64, what: &str) -> Result<u64, String> {
    before
        .checked_sub(after)
        .ok_or_else(|| format!("the free device memory grew from {before} to {after} bytes while mk made {what}"))
}

/// Free one context, then make a new one of `capacity` tokens. mk has no call that empties a context.
///
/// The free comes first, because the pool term holds the bytes of one context and not two (spec 4.2). When the new
/// context fails, the old one is already gone, and the error says so.
fn remake_context(
    old: *mut c_void,
    capacity: u32,
    free: impl Fn(*mut c_void),
    new: impl FnOnce(u32) -> *mut c_void,
    max: impl FnOnce(*mut c_void) -> u32,
) -> Result<*mut c_void, String> {
    free(old);
    let ptr = new(capacity);
    if ptr.is_null() {
        return Err(format!(
            "mk_ctx_new({capacity}) returned no context; the slot has no context now"
        ));
    }
    let made = max(ptr);
    if made != capacity {
        free(ptr);
        return Err(format!(
            "mk_ctx_new({capacity}) made a context of {made} tokens; the slot has no context now"
        ));
    }
    Ok(ptr)
}

/// Create `count` contexts of `capacity`, or free the ones that were made when a later call fails.
fn open_contexts(
    count: u32,
    capacity: u32,
    mut new: impl FnMut(u32) -> *mut c_void,
    mut max: impl FnMut(*mut c_void) -> u32,
    mut free: impl FnMut(*mut c_void),
) -> Result<Vec<(*mut c_void, u32)>, String> {
    let mut made: Vec<(*mut c_void, u32)> = Vec::new();
    for _ in 0..count {
        let ptr = new(capacity);
        if ptr.is_null() {
            for (ptr, _) in made.drain(..) {
                free(ptr);
            }
            return Err(format!("mk_ctx_new({capacity}) returned no context"));
        }
        let max_tokens = max(ptr);
        made.push((ptr, max_tokens));
    }
    Ok(made)
}

/// The two pass shapes of the scratch probe: the token count and whether the pass asks for logits.
///
/// The first shape has 1 token, so mk asks for `qmm_part2` too. The second shape asks for every other buffer at
/// `kMaxPassTokens`. The token counts add to the probe context's capacity, so one context serves both.
fn probe_shapes(capacity: u32) -> [(u32, bool); 2] {
    let rest = capacity
        .checked_sub(1)
        .expect("the probe context holds at least two tokens");
    [(1, false), (rest, true)]
}

impl ThreadState {
    /// Bind the linked engine and open it. This runs on the worker thread.
    fn open(paths: &MkPaths) -> Result<(Self, Spec), MkError> {
        let digest = gguf::spot_digest(&paths.shards)?;
        let fast_qmm = fast_qmm()?;
        let tiles = tiles()?;
        let (binding, build) = binding::linked().map_err(MkError::new)?;
        // Refuse a library whose parts are not the declared size, before it writes past a Rust buffer.
        if let Some(problem) = spec::parts_problem(
            |part| unsafe { (binding.state_bytes)(part) },
            |part, n| unsafe { (binding.part_bytes)(part, n) },
        ) {
            return Err(MkError::new(problem));
        }

        // The tokenizer is in the first shard. Hash it before the weights load, so a shard with no tokenizer fails fast.
        let text = gguf::text_keys(&paths.shards[0])?;

        // The Rust reader parses the shards, and the engine plans its arena from the reader's table (milestone 15).
        let model = gguf::Model::open(&paths.shards)?;
        let table = TensorTable::new(&model)?;
        let engine = unsafe { (binding.engine_open)(table.as_ptr()) };
        drop(table);
        if engine.is_null() {
            return Err(MkError::new("mk_engine_open returned no engine"));
        }
        let arena_bytes = unsafe { (binding.arena_bytes)(engine) };

        // mk stops the program on a pass that stops early by default (engine.h:215-220). Opt in, so a spin-timeout
        // returns with a last error and poisons the contexts, and the wrapper reports PassStatus::Poison (D6, 8.2).
        // mk has no query for this flag, so a test cannot observe the call. A forced spin-timeout needs contention.
        unsafe { (binding.recover_spin_timeouts)(engine, true) };

        let state = Self {
            binding,
            engine,
            _model: Some(model),
            contexts: HashMap::new(),
            next_id: 0,
            fast_qmm,
            tiles,
        };
        // mk has no call for the pool's size. Measure it here, before a caller's pass, so a larger pool is an open
        // error and the ledger's term is what the engine holds.
        let pass_scratch = state.measure_pass_scratch().map_err(MkError::new)?;
        spec::scratch_problem(pass_scratch).map_err(MkError::new)?;

        let identity = spec::identity(&paths.shards, &build, fast_qmm, tiles, &digest, &text);
        let limits = spec::limits(arena_bytes);
        Ok((
            state,
            Spec {
                identity,
                limits,
                pass_scratch,
            },
        ))
    }

    /// The free device memory in bytes, on the worker thread's current device.
    fn free_device_bytes(&self) -> Result<u64, String> {
        let mut free = 0usize;
        let mut total = 0usize;
        let code = unsafe { (self.binding.mem_get_info)(&mut free, &mut total) };
        if code != 0 {
            return Err(format!("hipMemGetInfo failed with HIP error {code}"));
        }
        Ok(free as u64)
    }

    /// The pass scratch in bytes: the device free memory that the pool's first asks take.
    ///
    /// A pass of 1 token asks for `qmm_part2`, and a pass of 17 or more tokens does not (mk `engine.hip:1187`). The
    /// second pass has the 255 tokens that remain, with logits, so it asks for every other buffer at `kMaxPassTokens`.
    /// mk keeps the pool for the engine's life. The probe context is freed before the second reading.
    fn measure_pass_scratch(&self) -> Result<u64, String> {
        let before = self.free_device_bytes()?;
        let context = unsafe { (self.binding.ctx_new)(self.engine, spec::PROBE_CAPACITY) };
        if context.is_null() {
            return Err(format!(
                "mk_ctx_new({}) returned no context for the pass scratch probe",
                spec::PROBE_CAPACITY
            ));
        }
        let result = self.probe_passes(context);
        unsafe { (self.binding.ctx_free)(self.engine, context) };
        let after = self.free_device_bytes()?;
        result?;
        before
            .checked_sub(after)
            .ok_or_else(|| format!("the free device memory grew from {before} to {after} bytes during the probe"))
    }

    /// Run the two probe passes on one context: 1 token, then the rest, with logits.
    fn probe_passes(&self, context: *mut c_void) -> Result<(), String> {
        let tokens: Vec<i32> = (1..=spec::PROBE_CAPACITY as i32).collect();
        let mut start = 0;
        for (count, want_logits) in probe_shapes(spec::PROBE_CAPACITY) {
            let end = start + count as usize;
            self.probe_pass(context, &tokens[start..end], want_logits)?;
            start = end;
        }
        Ok(())
    }

    /// Run one probe pass: `tokens` at the context's end, with the logits row when asked.
    fn probe_pass(&self, context: *mut c_void, tokens: &[i32], want_logits: bool) -> Result<(), String> {
        let segments = binding::segments(&[context], &[tokens.len() as u32], tokens);
        let rows = usize::from(want_logits);
        let mut logits = vec![0f32; rows * spec::VOCAB as usize];
        let logits = want_logits.then_some(logits.as_mut_slice());
        match unsafe { self.binding.pass_segments(self.engine, &segments, logits) } {
            Ok(_ms) => Ok(()),
            Err(PassStop::Refused(why)) => Err(format!("the pass scratch probe was refused: {why}")),
            Err(PassStop::Stopped(why)) => Err(format!("the pass scratch probe stopped: {why}")),
        }
    }

    /// Serve commands until the handle side closes the channel.
    fn run(mut self, receiver: mpsc::Receiver<Command>) {
        for command in receiver {
            self.handle(command);
        }
    }

    fn handle(&mut self, command: Command) {
        match command {
            Command::Open { classes, reply } => {
                let _ = reply.send(self.open_pool(&classes));
            }
            Command::Free { ids, reply } => {
                self.free_ids(&ids);
                let _ = reply.send(());
            }
            Command::N { id, reply } => {
                let _ =
                    reply.send(self.with_context(id, |state, context| unsafe { (state.binding.ctx_n)(context.ptr) }));
            }
            Command::Max { id, reply } => {
                let _ =
                    reply.send(self.with_context(id, |state, context| unsafe { (state.binding.ctx_max)(context.ptr) }));
            }
            Command::OwnsKv { id, reply } => {
                let _ = reply
                    .send(self.with_context(id, |state, context| unsafe { (state.binding.ctx_owns_kv)(context.ptr) }));
            }
            Command::Poisoned { id, reply } => {
                let _ = reply.send(self.with_context(id, |state, context| unsafe {
                    (state.binding.ctx_poisoned)(context.ptr)
                }));
            }
            Command::Read { id, part, len, reply } => {
                let _ = reply.send(self.read(id, part, len));
            }
            Command::Write { id, part, bytes, reply } => {
                let _ = reply.send(self.write(id, part, &bytes));
            }
            Command::CellsWrite {
                id,
                part,
                n,
                bytes,
                reply,
            } => {
                let _ = reply.send(self.cells_write(id, part, n, &bytes));
            }
            Command::SetN { id, n, reply } => {
                let _ = reply.send(self.set_n(id, n));
            }
            Command::Seal { id, reply } => {
                let _ = reply.send(self.seal(id));
            }
            Command::Clear { id, reply } => {
                let _ = reply.send(self.clear(id));
            }
            Command::Pass { request, reply } => {
                let _ = reply.send(self.pass(request));
            }
            Command::PartBytes { part, n, reply } => {
                let _ = reply.send(unsafe { (self.binding.part_bytes)(part, n) });
            }
        }
    }

    /// Create the contexts of every class, in order, and read what each class took.
    ///
    /// A later class that fails frees the contexts of the earlier ones. The pool is whole or not made.
    fn open_pool(&mut self, classes: &[spec::ContextClass]) -> Result<Pool, String> {
        // Admit the whole pool before mk makes any of it (spec 4.3, 4.7).
        let needed: u64 = classes
            .iter()
            .map(|class| u64::from(class.count) * spec::context_declared_bytes(class.capacity))
            .sum();
        memory_problem(
            "the context pool",
            mem_available().map_err(|error| error.to_string())?,
            needed,
        )?;
        let mut ids = Vec::new();
        let mut readings = Vec::new();
        for class in classes {
            let free_before = self.free_device_bytes();
            let available_before = mem_available();
            let made = self.open_states(*class);
            let free_after = self.free_device_bytes();
            let available_after = mem_available();
            match made {
                Ok(made) => ids.extend(made),
                Err(error) => {
                    self.free_ids(&ids);
                    return Err(error);
                }
            }
            // The ledger holds the declared bytes. A class that takes more is a defect, so refuse the pool.
            let taken = match (free_before, free_after) {
                (Ok(before), Ok(after)) => taken_bytes(
                    before,
                    after,
                    &format!("{} contexts of {}", class.count, class.capacity),
                ),
                (Err(error), _) | (_, Err(error)) => Err(error),
            };
            let device = match taken {
                Ok(device) => device,
                Err(error) => {
                    self.free_ids(&ids);
                    return Err(error);
                }
            };
            if let Err(problem) = spec::class_problem(*class, device) {
                self.free_ids(&ids);
                return Err(problem);
            }
            readings.push(ClassReading {
                class: *class,
                declared: u64::from(class.count) * spec::context_declared_bytes(class.capacity),
                device,
                available_before: available_before.ok(),
                available_after: available_after.ok(),
            });
        }
        Ok(Pool { ids, readings })
    }

    /// Free the contexts with these ids.
    fn free_ids(&mut self, ids: &[u64]) {
        for id in ids {
            if let Some(context) = self.contexts.remove(id) {
                unsafe { (self.binding.ctx_free)(self.engine, context.ptr) };
            }
        }
    }

    fn open_states(&mut self, class: spec::ContextClass) -> Result<Vec<u64>, String> {
        let spec::ContextClass { count, capacity } = class;
        if count == 0 {
            return Err("a handle needs at least one context".to_owned());
        }
        if capacity == 0 || capacity > spec::MAX_CONTEXT || !capacity.is_multiple_of(spec::CAPACITY_STEP) {
            return Err(format!(
                "a context capacity of {capacity} is not a multiple of {} in 1..={}",
                spec::CAPACITY_STEP,
                spec::MAX_CONTEXT
            ));
        }
        let made = open_contexts(
            count,
            capacity,
            |capacity| unsafe { (self.binding.ctx_new)(self.engine, capacity) },
            |ptr| unsafe { (self.binding.ctx_max)(ptr) },
            |ptr| unsafe { (self.binding.ctx_free)(self.engine, ptr) },
        )?;
        let mut ids = Vec::with_capacity(made.len());
        for (ptr, max) in made {
            let id = self.next_id;
            self.next_id += 1;
            self.contexts.insert(
                id,
                Context {
                    ptr,
                    capacity: max,
                    restored_kv: 0,
                    restored_idx: 0,
                    restored_ple: false,
                },
            );
            ids.push(id);
        }
        Ok(ids)
    }

    /// Run `call` with one context, or report that the id is unknown.
    fn with_context<T>(&self, id: u64, call: impl FnOnce(&Self, &Context) -> T) -> Result<T, String> {
        let context = self
            .contexts
            .get(&id)
            .ok_or_else(|| format!("context {id} is not in this engine"))?;
        Ok(call(self, context))
    }

    fn read(&self, id: u64, part: u32, len: usize) -> Result<Vec<u8>, Call> {
        let context = self.context(id)?;
        if unsafe { (self.binding.ctx_poisoned)(context.ptr) } {
            return Err(Call::Failed("the context is poisoned; free it".to_owned()));
        }
        let n = unsafe { (self.binding.ctx_n)(context.ptr) };
        let expected = match part {
            0 => spec::GDN_STATE_BYTES,
            1 => spec::GDN_CONV_BYTES,
            2 => spec::PLE_CONV_BYTES,
            3 => u64::from(n) * spec::KV_ROW_BYTES,
            4 => u64::from(n) * spec::IDX_ROW_BYTES,
            5 => spec::PLE_HIST_BYTES,
            other => return Err(Call::Refused(format!("mk has no part {other}"))),
        };
        if len as u64 != expected {
            return Err(Call::Refused(format!(
                "the buffer is {len} bytes; part {part} holds {expected} bytes at n = {n}"
            )));
        }
        let mut bytes = Aligned::zeroed(len);
        unsafe { (self.binding.state_read)(self.engine, context.ptr, part, bytes.as_mut_ptr()) };
        Ok(bytes.to_vec())
    }

    fn write(&mut self, id: u64, part: u32, bytes: &[u8]) -> Call {
        let context = match self.context(id) {
            Ok(context) => context,
            Err(call) => return call,
        };
        let ptr = context.ptr;
        let poisoned = unsafe { (self.binding.ctx_poisoned)(ptr) };
        let n = unsafe { (self.binding.ctx_n)(ptr) };
        let owns_kv = unsafe { (self.binding.ctx_owns_kv)(ptr) };
        // mk reads the part as f32 values, so the source must be aligned for them.
        let bytes = Aligned::copy(bytes);
        match part {
            0..=2 => {
                if poisoned {
                    return Call::Refused(POISONED.to_owned());
                }
                let expected = match part {
                    0 => spec::GDN_STATE_BYTES,
                    1 => spec::GDN_CONV_BYTES,
                    _ => spec::PLE_CONV_BYTES,
                };
                if bytes.len() as u64 != expected {
                    return Call::Refused(format!(
                        "part {part} holds {expected} bytes; the write has {}",
                        bytes.len()
                    ));
                }
                unsafe { (self.binding.state_write)(self.engine, ptr, part, bytes.as_ptr()) };
                if unsafe { (self.binding.ctx_poisoned)(ptr) } {
                    return Call::Failed("the write poisoned the context".to_owned());
                }
                Call::Ok
            }
            3 | 4 => Call::Refused(format!("part {part} is cells; use cells_write with the row count")),
            5 => {
                if poisoned {
                    return Call::Refused(POISONED.to_owned());
                }
                if n != 0 || !owns_kv {
                    return Call::Refused(format!(
                        "ple_hist needs an empty context that owns its K/V; this one has {n} tokens and owns_kv={owns_kv}"
                    ));
                }
                if bytes.len() as u64 != spec::PLE_HIST_BYTES {
                    return Call::Refused(format!(
                        "ple_hist holds {} bytes; the write has {}",
                        spec::PLE_HIST_BYTES,
                        bytes.len()
                    ));
                }
                let history = bytes.to_vec();
                let first = i32::from_ne_bytes([history[0], history[1], history[2], history[3]]);
                let second = i32::from_ne_bytes([history[4], history[5], history[6], history[7]]);
                for (index, value) in [first, second].into_iter().enumerate() {
                    if value < 0 || value as u32 >= spec::VOCAB {
                        return Call::Refused(format!(
                            "ple_hist[{index}] = {value} is not a token id in 0..{}",
                            spec::VOCAB
                        ));
                    }
                }
                unsafe { (self.binding.state_write)(self.engine, ptr, part, bytes.as_ptr()) };
                if unsafe { (self.binding.ctx_poisoned)(ptr) } {
                    return Call::Failed("the write poisoned the context".to_owned());
                }
                if let Some(context) = self.contexts.get_mut(&id) {
                    context.restored_ple = true;
                }
                Call::Ok
            }
            other => Call::Refused(format!("mk has no part {other}")),
        }
    }

    fn cells_write(&mut self, id: u64, part: u32, n: u32, bytes: &[u8]) -> Call {
        let context = match self.context(id) {
            Ok(context) => context,
            Err(call) => return call,
        };
        let ptr = context.ptr;
        let capacity = context.capacity;
        let poisoned = unsafe { (self.binding.ctx_poisoned)(ptr) };
        let current = unsafe { (self.binding.ctx_n)(ptr) };
        let owns_kv = unsafe { (self.binding.ctx_owns_kv)(ptr) };
        if poisoned {
            return Call::Refused(POISONED.to_owned());
        }
        // mk reads the rows as f32 values, so the source must be aligned for them.
        let bytes = Aligned::copy(bytes);
        let row = match part {
            3 => spec::KV_ROW_BYTES,
            4 => spec::IDX_ROW_BYTES,
            other => return Call::Refused(format!("mk has cells only in parts 3 and 4, not {other}")),
        };
        if current != 0 || !owns_kv {
            return Call::Refused(format!(
                "a cells write needs an empty context that owns its K/V; this one has {current} tokens and owns_kv={owns_kv}"
            ));
        }
        if n == 0 || n > capacity {
            return Call::Refused(format!("{n} cells do not fit a context of capacity {capacity}"));
        }
        if bytes.len() as u64 != u64::from(n) * row {
            return Call::Refused(format!(
                "{n} rows of {row} bytes are {} bytes; the write has {}",
                u64::from(n) * row,
                bytes.len()
            ));
        }
        unsafe { (self.binding.cells_write)(self.engine, ptr, part, n, bytes.as_ptr()) };
        if unsafe { (self.binding.ctx_poisoned)(ptr) } {
            return Call::Failed("the cells write poisoned the context".to_owned());
        }
        if let Some(context) = self.contexts.get_mut(&id) {
            if part == 3 {
                context.restored_kv = n;
            } else {
                context.restored_idx = n;
            }
        }
        Call::Ok
    }

    fn set_n(&mut self, id: u64, n: u32) -> Call {
        let context = match self.context(id) {
            Ok(context) => context,
            Err(call) => return call,
        };
        let ptr = context.ptr;
        let capacity = context.capacity;
        let poisoned = unsafe { (self.binding.ctx_poisoned)(ptr) };
        let current = unsafe { (self.binding.ctx_n)(ptr) };
        let owns_kv = unsafe { (self.binding.ctx_owns_kv)(ptr) };
        let restored = (context.restored_kv, context.restored_idx, context.restored_ple);
        if poisoned {
            return Call::Refused(POISONED.to_owned());
        }
        if current != 0 || !owns_kv {
            return Call::Refused(format!(
                "set_n needs an empty context that owns its K/V; this one has {current} tokens and owns_kv={owns_kv}"
            ));
        }
        if n == 0 || n > capacity {
            return Call::Refused(format!("set_n({n}) is not in 1..=the capacity {capacity}"));
        }
        if restored != (n, n, true) {
            return Call::Refused(format!(
                "set_n({n}): K/V written for {} cells, index keys for {}, PLE history written={}",
                restored.0, restored.1, restored.2
            ));
        }
        unsafe { (self.binding.set_n)(self.engine, ptr, n) };
        // mk returns nothing here. A call that left the context poisoned is a failure (spec 6.8), so check first.
        if unsafe { (self.binding.ctx_poisoned)(ptr) } {
            return Call::Failed("set_n poisoned the context".to_owned());
        }
        if unsafe { (self.binding.ctx_n)(ptr) } != n {
            return Call::Failed(format!("set_n({n}) did not set the token count"));
        }
        if let Some(context) = self.contexts.get_mut(&id) {
            context.restored_kv = 0;
            context.restored_idx = 0;
            context.restored_ple = false;
        }
        Call::Ok
    }

    /// Empty one context. A poisoned context can be cleared: a clear is how a slot leaves the poisoned state (8.2).
    fn clear(&mut self, id: u64) -> Call {
        let Some(context) = self.contexts.get(&id) else {
            return Call::Refused(format!("context {id} is not in this engine"));
        };
        let (old, capacity) = (context.ptr, context.capacity);
        let made = remake_context(
            old,
            capacity,
            |ptr| unsafe { (self.binding.ctx_free)(self.engine, ptr) },
            |capacity| unsafe { (self.binding.ctx_new)(self.engine, capacity) },
            |ptr| unsafe { (self.binding.ctx_max)(ptr) },
        );
        match made {
            Ok(ptr) => {
                self.contexts.insert(
                    id,
                    Context {
                        ptr,
                        capacity,
                        restored_kv: 0,
                        restored_idx: 0,
                        restored_ple: false,
                    },
                );
                Call::Ok
            }
            Err(message) => {
                self.contexts.remove(&id);
                Call::Failed(message)
            }
        }
    }

    fn seal(&mut self, id: u64) -> Call {
        let context = match self.context(id) {
            Ok(context) => context,
            Err(call) => return call,
        };
        let ptr = context.ptr;
        if unsafe { (self.binding.ctx_poisoned)(ptr) } {
            return Call::Refused(POISONED.to_owned());
        }
        // mk_ctx_round_gdn_bf16 stops the program on a non-finite value, or one that rounds to infinity. Check first.
        let mut state = Aligned::zeroed(spec::GDN_STATE_BYTES as usize);
        unsafe { (self.binding.state_read)(self.engine, ptr, 0, state.as_mut_ptr()) };
        for (index, value) in state.f32_values().enumerate() {
            if !value.is_finite() || !rounds_to_finite(value) {
                return Call::Refused(format!(
                    "gdn_state[{index}] = {value} is not finite after a bf16 round; mk stops on it"
                ));
            }
        }
        unsafe { (self.binding.round_gdn)(self.engine, ptr) };
        if unsafe { (self.binding.ctx_poisoned)(ptr) } {
            return Call::Failed("the seal poisoned the context".to_owned());
        }
        Call::Ok
    }

    fn pass(&mut self, request: PassRequest) -> Result<PassReply, PassProblem> {
        let problem = |message: String| PassProblem { message, poison: false };
        // mk reads MK_FAST_QMM at each pass build. A change after open would change the bits, and the identity
        // would be wrong, so refuse the pass.
        match fast_qmm() {
            Ok(value) if value == self.fast_qmm => {}
            Ok(value) => {
                return Err(problem(format!(
                    "MK_FAST_QMM changed from {} to {} after the engine opened",
                    u8::from(self.fast_qmm),
                    u8::from(value)
                )));
            }
            Err(error) => return Err(problem(error.to_string())),
        }
        // mk reads MK_TILES once, at load. A change after open makes the environment disagree with the identity,
        // so refuse the pass, as for MK_FAST_QMM.
        match tiles() {
            Ok(value) if value == self.tiles => {}
            Ok(value) => {
                return Err(problem(format!(
                    "MK_TILES changed from {} to {} after the engine opened",
                    u8::from(self.tiles),
                    u8::from(value)
                )));
            }
            Err(error) => return Err(problem(error.to_string())),
        }
        let count = request.contexts.len();
        if count == 0 || count > spec::MAX_SEGMENTS as usize {
            return Err(problem(format!(
                "a pass needs 1..={} segments; this one has {count}",
                spec::MAX_SEGMENTS
            )));
        }
        if request.lens.len() != count {
            return Err(problem("the segment and length counts differ".to_owned()));
        }
        let total: u32 = request.lens.iter().sum();
        if total as u64 != request.tokens.len() as u64 {
            return Err(problem("the lengths and the token count differ".to_owned()));
        }
        if total > spec::MAX_PASS_TOKENS {
            return Err(problem(format!(
                "{total} tokens are past mk's {}-token pass",
                spec::MAX_PASS_TOKENS
            )));
        }
        let mut pointers = Vec::with_capacity(count);
        for id in &request.contexts {
            let context = self
                .contexts
                .get(id)
                .ok_or_else(|| problem(format!("context {id} is not in this engine")))?;
            if unsafe { (self.binding.ctx_poisoned)(context.ptr) } {
                return Err(problem(format!("context {id} is poisoned; free it")));
            }
            pointers.push(context.ptr);
        }

        // The binding runs mk's segment check first, because mk stops the program on a bad segment.
        let segments = binding::segments(&pointers, &request.lens, &request.tokens);
        let rows = if request.want_logits { count } else { 0 };
        let mut logits = vec![0f32; rows * spec::VOCAB as usize];
        let buffer = request.want_logits.then_some(logits.as_mut_slice());
        let ms = match unsafe { self.binding.pass_segments(self.engine, &segments, buffer) } {
            Ok(ms) => ms,
            Err(PassStop::Refused(message)) => return Err(problem(message)),
            Err(PassStop::Stopped(message)) => return Err(PassProblem { message, poison: true }),
        };
        let logits = request
            .want_logits
            .then(|| logits.iter().flat_map(|value| value.to_le_bytes()).collect());
        Ok(PassReply { ms, logits })
    }
}

impl ThreadState {
    /// Borrow one context or refuse the call.
    fn context(&self, id: u64) -> Result<&Context, Call> {
        self.contexts
            .get(&id)
            .ok_or_else(|| Call::Refused(format!("context {id} is not in this engine")))
    }
}

/// A byte buffer that is aligned for mk's f32 stores. mk reads and writes a part as f32 values in host memory.
///
/// A `Vec<u8>` holds no alignment promise. This buffer holds its bytes in 8-byte words, so the pointer is 8-byte
/// aligned. It copies bytes into the words and out again, so it makes no assumption about the host byte order.
#[repr(align(8))]
#[derive(Clone, Copy)]
struct Word([u8; 8]);

struct Aligned {
    words: Vec<Word>,
    len: usize,
}

impl Aligned {
    /// A zeroed buffer of `len` bytes.
    fn zeroed(len: usize) -> Self {
        Self {
            words: vec![Word([0; 8]); len.div_ceil(8)],
            len,
        }
    }

    /// A buffer that holds a copy of `bytes`.
    fn copy(bytes: &[u8]) -> Self {
        let mut buffer = Self::zeroed(bytes.len());
        for (word, chunk) in buffer.words.iter_mut().zip(bytes.chunks(8)) {
            word.0[..chunk.len()].copy_from_slice(chunk);
        }
        buffer
    }

    /// The buffer's length in bytes.
    fn len(&self) -> usize {
        self.len
    }

    /// The bytes of the buffer, in memory order.
    fn to_vec(&self) -> Vec<u8> {
        self.words.iter().flat_map(|word| word.0).take(self.len).collect()
    }

    /// The buffer as f32 values, in order, for a buffer that holds f32 data.
    fn f32_values(&self) -> impl Iterator<Item = f32> + '_ {
        (0..self.len / 4).map(move |index| {
            let start = index * 4;
            let word = self.words[start / 8].0;
            let offset = start % 8;
            f32::from_ne_bytes([word[offset], word[offset + 1], word[offset + 2], word[offset + 3]])
        })
    }

    /// The pointer for mk to read.
    fn as_ptr(&self) -> *const c_void {
        self.words.as_ptr().cast()
    }

    /// The pointer for mk to write.
    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.words.as_mut_ptr().cast()
    }
}

/// True when a finite f32 rounds to a finite bf16 under round-to-nearest, ties-to-even.
fn rounds_to_finite(value: f32) -> bool {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded).is_finite()
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::ffi::c_void;
    use std::os::unix::ffi::OsStrExt;
    use std::path::PathBuf;

    use std::collections::HashMap;
    use std::sync::atomic::Ordering;

    use crate::binding::stub::{self, StubContext};

    use super::{
        Aligned, Call, Context, ThreadState, Word, fast_qmm_from, memory_problem, open_contexts, parse_shards,
        probe_shapes, remake_context, rounds_to_finite, taken_bytes, tiles_from,
    };

    /// A worker state with one stub context, id 1.
    fn stub_state(context: &StubContext) -> ThreadState {
        let contexts = HashMap::from([(
            1,
            Context {
                ptr: context as *const StubContext as *mut c_void,
                capacity: 256,
                restored_kv: 0,
                restored_idx: 0,
                restored_ple: false,
            },
        )]);
        ThreadState {
            binding: stub::binding(),
            engine: std::ptr::null_mut(),
            _model: None,
            contexts,
            next_id: 2,
            fast_qmm: true,
            tiles: false,
        }
    }

    fn is_refused(call: &Call) -> bool {
        matches!(call, Call::Refused(message) if message.contains("poisoned"))
    }

    #[test]
    fn every_write_to_a_poisoned_context_is_refused_and_changes_nothing() {
        let context = StubContext::new(true);
        let mut state = stub_state(&context);
        let history = [1u8, 0, 0, 0, 2, 0, 0, 0];
        let calls = [
            state.write(1, 0, &[0u8; 16]),
            state.write(1, 5, &history),
            state.cells_write(1, 3, 1, &[0u8; 8]),
            state.set_n(1, 1),
        ];
        for call in &calls {
            assert!(is_refused(call), "{call:?}");
        }
        assert_eq!(context.changes.load(Ordering::SeqCst), 0, "a refused call reached mk");
    }

    #[test]
    fn a_read_of_a_poisoned_context_is_still_an_error() {
        let context = StubContext::new(true);
        let state = stub_state(&context);
        assert!(matches!(state.read(1, 5, 8), Err(Call::Failed(_))));
    }

    #[test]
    fn a_write_that_poisons_the_context_is_failed() {
        let mut context = StubContext::new(false);
        context.write_poisons = true;
        let mut state = stub_state(&context);
        let history = [1u8, 0, 0, 0, 2, 0, 0, 0];
        let call = state.write(1, 5, &history);
        assert!(
            matches!(call, Call::Failed(ref message) if message.contains("poisoned")),
            "{call:?}"
        );
        assert_eq!(context.changes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn the_probe_asks_for_the_small_pass_and_then_the_logits_row() {
        // A pass of 1 token asks for qmm_part2 (mk engine.hip:1187). The second pass asks for the rest, with logits.
        assert_eq!(probe_shapes(256), [(1, false), (255, true)]);
        let total: u32 = probe_shapes(256).iter().map(|(tokens, _)| tokens).sum();
        assert_eq!(total, crate::spec::PROBE_CAPACITY);
    }

    #[test]
    fn a_context_pool_frees_what_it_made_when_a_later_new_fails() {
        let freed = RefCell::new(Vec::new());
        let mut calls = 0;
        let result = open_contexts(
            3,
            256,
            |_capacity| {
                calls += 1;
                if calls == 3 {
                    std::ptr::null_mut()
                } else {
                    (calls as usize) as *mut c_void
                }
            },
            |_ptr| 256,
            |ptr| freed.borrow_mut().push(ptr as usize),
        );
        assert!(result.is_err());
        assert_eq!(
            freed.into_inner(),
            vec![1usize, 2],
            "the made contexts were not all freed"
        );
    }

    #[test]
    fn a_clear_frees_the_context_and_then_makes_one_of_the_same_capacity() {
        let calls = RefCell::new(Vec::new());
        let made = remake_context(
            16usize as *mut c_void,
            4096,
            |ptr| calls.borrow_mut().push(format!("free {}", ptr as usize)),
            |capacity| {
                calls.borrow_mut().push(format!("new {capacity}"));
                2usize as *mut c_void
            },
            |_ptr| 4096,
        );
        assert_eq!(made, Ok(2usize as *mut c_void));
        // The free comes first: the pool term holds the bytes of one context, not two (spec 4.2).
        assert_eq!(calls.into_inner(), ["free 16", "new 4096"]);
    }

    #[test]
    fn a_clear_whose_new_fails_names_the_lost_context() {
        let error = remake_context(16usize as *mut c_void, 4096, |_| {}, |_| std::ptr::null_mut(), |_| 4096)
            .expect_err("a null context is an error");
        assert!(error.contains("mk_ctx_new(4096)"), "{error}");
        assert!(error.contains("no context"), "{error}");
    }

    #[test]
    fn a_clear_that_gets_another_capacity_frees_it_and_fails() {
        let freed = RefCell::new(Vec::new());
        let error = remake_context(
            16usize as *mut c_void,
            4096,
            |ptr| freed.borrow_mut().push(ptr as usize),
            |_| 2usize as *mut c_void,
            |_| 256,
        )
        .expect_err("another capacity is an error");
        assert!(error.contains("256"), "{error}");
        assert_eq!(freed.into_inner(), [16, 2]);
    }

    #[test]
    fn a_class_that_freed_device_memory_is_an_error_with_both_readings() {
        assert_eq!(taken_bytes(1000, 400, "the class of 12 contexts"), Ok(600));
        assert_eq!(taken_bytes(1000, 1000, "the class of 12 contexts"), Ok(0));
        let error = taken_bytes(400, 1000, "the class of 12 contexts").expect_err("free memory grew");
        assert!(error.contains("400") && error.contains("1000"), "{error}");
        assert!(error.contains("the class of 12 contexts"), "{error}");
    }

    #[test]
    fn tiles_has_two_values() {
        assert_eq!(tiles_from(None), Ok(true));
        assert_eq!(tiles_from(Some("1")), Ok(true));
        assert_eq!(tiles_from(Some("0")), Ok(false));
        assert!(tiles_from(Some("2")).is_err());
    }

    #[test]
    fn an_aligned_buffer_keeps_its_bytes_and_its_alignment() {
        let source: Vec<u8> = (0..19u8).collect();
        let buffer = Aligned::copy(&source);
        assert_eq!(buffer.to_vec(), source);
        assert_eq!(buffer.words.len(), 3, "19 bytes take three 8-byte words");
        assert_eq!(
            buffer.as_ptr() as usize % std::mem::align_of::<u64>(),
            0,
            "the buffer is not 8-byte aligned"
        );
        let zeroed = Aligned::zeroed(13);
        assert_eq!(zeroed.to_vec(), vec![0u8; 13]);
        let values: Vec<u8> = [1.0f32, -2.5, 3.25]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let floats: Vec<f32> = Aligned::copy(&values).f32_values().collect();
        assert_eq!(floats, vec![1.0, -2.5, 3.25]);
    }

    #[test]
    fn an_aligned_buffer_keeps_the_memory_order_of_its_words() {
        let mut buffer = Aligned::zeroed(16);
        buffer.words[0] = Word([1, 2, 3, 4, 5, 6, 7, 8]);
        buffer.words[1] = Word([9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(buffer.to_vec(), (1..=16u8).collect::<Vec<u8>>());
        assert_eq!(
            buffer.f32_values().next(),
            Some(f32::from_ne_bytes([1, 2, 3, 4])),
            "the first f32 is the first four bytes of the first word"
        );
        assert_eq!(
            buffer.f32_values().nth(1),
            Some(f32::from_ne_bytes([5, 6, 7, 8])),
            "the second f32 is the second four bytes of the first word"
        );
    }

    #[test]
    fn a_load_that_leaves_less_than_the_floor_is_refused() {
        // spec 4.7: mk keeps 12 GiB of MemAvailable free (mk service/contexts.py, FLOOR_BYTES).
        let floor: u64 = 12 << 30;
        assert_eq!(memory_problem("the shards", floor, 0), Ok(()));
        assert!(memory_problem("the shards", floor - 1, 0).is_err());
        assert!(memory_problem("the shards", floor, 1).is_err());
        assert_eq!(memory_problem("the shards", floor + 1, 1), Ok(()));
    }

    #[test]
    fn a_pool_that_leaves_less_than_the_floor_is_refused_and_the_message_names_the_pool() {
        // The pool is 12 198 215 680 bytes (spec.rs). With 20 GiB available, 12 GiB of floor leaves 8 GiB.
        let pool = crate::spec::pool_bytes();
        assert_eq!(pool, 12_131_106_816 + 67_108_864);
        let floor: u64 = 12 << 30;
        assert_eq!(memory_problem("the context pool", floor + pool, pool), Ok(()));
        let error = memory_problem("the context pool", floor + pool - 1, pool).expect_err("one byte short must refuse");
        assert!(error.contains("the context pool takes 12198215680 bytes"), "{error}");
        assert!(
            error.contains(&format!("{} bytes are available", floor + pool - 1)),
            "{error}"
        );
    }

    #[test]
    fn a_bf16_round_of_a_finite_value_can_be_infinite() {
        assert!(rounds_to_finite(1.0));
        assert!(rounds_to_finite(0.0));
        assert!(rounds_to_finite(-2.5));
        assert!(!rounds_to_finite(f32::NAN));
        assert!(!rounds_to_finite(f32::INFINITY));
        // f32::MAX rounds to 3.39e38, which bf16 cannot hold.
        assert!(!rounds_to_finite(f32::MAX));
    }

    #[test]
    fn a_shard_list_splits_on_colons_and_keeps_raw_bytes() {
        assert_eq!(
            parse_shards(b"/a:/b:/c").unwrap(),
            vec![PathBuf::from("/a"), PathBuf::from("/b"), PathBuf::from("/c")]
        );
        assert!(parse_shards(b"/a:/b").is_err());
        assert!(parse_shards(b"/a::/c").is_err());
        // A path that is not UTF-8 stays the same bytes.
        let paths = parse_shards(&[0xff, b':', b'/', b'b', b':', b'/', b'c']).unwrap();
        assert_eq!(paths[0].as_os_str().as_bytes(), &[0xff]);
    }

    #[test]
    fn fast_qmm_has_two_values() {
        assert_eq!(fast_qmm_from(None), Ok(true));
        assert_eq!(fast_qmm_from(Some("1")), Ok(true));
        assert_eq!(fast_qmm_from(Some("0")), Ok(false));
        assert!(fast_qmm_from(Some("2")).is_err());
    }
}
