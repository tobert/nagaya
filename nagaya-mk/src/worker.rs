//! The one thread that makes every mk call.
//!
//! mk's HIP device is the device that is current on the calling thread. One worker thread owns the engine, and every
//! call goes to it over a channel (spec 3.2). The engine is never closed: HIP teardown segfaults in the process (spec
//! 3.3).

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::fmt;
use std::io::Read;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};

use nagaya_abi::{Identity, Limits};
use sha2::{Digest, Sha256};

use crate::binding::{self, Binding};
use crate::gguf;
use crate::spec;

/// An error from the mk binding. The message says what failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MkError(String);

impl MkError {
    /// Make an error with a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for MkError {}

impl From<String> for MkError {
    fn from(message: String) -> Self {
        Self(message)
    }
}

/// Where the library and the three weight shards are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MkPaths {
    /// The path of `libmkengine.so` (`MK_ENGINE_LIB`).
    pub library: PathBuf,
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
    /// Read the paths from `MK_ENGINE_LIB` and `MK_SHARDS`.
    pub fn from_env() -> Result<Self, MkError> {
        let library = std::env::var_os("MK_ENGINE_LIB")
            .ok_or_else(|| MkError::new("MK_ENGINE_LIB is not set; set it to libmkengine.so"))?;
        let shards = std::env::var_os("MK_SHARDS")
            .ok_or_else(|| MkError::new("MK_SHARDS is not set; give three GGUF paths with : between them"))?;
        let shards = parse_shards(shards.as_os_str().as_bytes())?;
        Ok(Self {
            library: PathBuf::from(library),
            shards,
        })
    }

    /// Check that every path exists and every shard has a GGUF v3 header.
    fn validate(&self) -> Result<Vec<gguf::ShardHeader>, MkError> {
        if !self.library.is_file() {
            return Err(MkError::new(format!(
                "the mk engine library {} is missing",
                self.library.display()
            )));
        }
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
    /// True when mk returns a pass that stopped on a spin-timeout instead of stopping the program.
    pub recover_spin_timeouts: bool,
}

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

/// One command for the worker thread.
enum Command {
    Open {
        count: u32,
        capacity: u32,
        reply: mpsc::Sender<Result<Vec<u64>, String>>,
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

    /// Create a context pool for one handle.
    pub fn open_states(&self, count: u32, capacity: u32) -> Result<Vec<u64>, MkError> {
        self.ask(|reply| Command::Open { count, capacity, reply })
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
        self.ask(|reply| Command::Read {
            id,
            part: part as u32,
            len,
            reply,
        })
        .unwrap_or_else(|| Err(Call::Failed("the mk worker stopped".to_owned())))
    }

    /// Write one fixed part.
    pub fn write(&self, id: u64, part: usize, bytes: Vec<u8>) -> Call {
        self.ask(|reply| Command::Write {
            id,
            part: part as u32,
            bytes,
            reply,
        })
        .unwrap_or_else(|| Call::Failed("the mk worker stopped".to_owned()))
    }

    /// Write `n` rows of a cells part.
    pub fn cells_write(&self, id: u64, part: usize, n: u32, bytes: Vec<u8>) -> Call {
        self.ask(|reply| Command::CellsWrite {
            id,
            part: part as u32,
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

/// Refuse a load that leaves less than the floor free.
fn memory_problem(available: u64, needed: u64) -> Result<(), String> {
    if available < needed.saturating_add(MEMORY_FLOOR) {
        return Err(format!(
            "the shards take {needed} bytes and mk loads them into memory; \
             {available} bytes are available and the floor is {MEMORY_FLOOR}"
        ));
    }
    Ok(())
}

/// Refuse a load that cannot fit in the free memory. mk allocates the weight arena in full, and the floor stays free.
fn check_memory(needed: u64) -> Result<(), MkError> {
    let text =
        std::fs::read_to_string("/proc/meminfo").map_err(|error| MkError::new(format!("/proc/meminfo: {error}")))?;
    let available = text
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .map(|kilobytes| kilobytes * 1024)
        .ok_or_else(|| MkError::new("/proc/meminfo has no MemAvailable"))?;
    memory_problem(available, needed).map_err(MkError::new)
}

/// The SHA-256 of a file, as lowercase hex.
fn file_sha256(path: &Path) -> Result<String, MkError> {
    let mut file = std::fs::File::open(path).map_err(|error| MkError::new(format!("{}: {error}", path.display())))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| MkError::new(format!("{}: {error}", path.display())))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hash.finalize().iter().map(|byte| format!("{byte:02x}")).collect())
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

impl ThreadState {
    /// Load the library and open the engine. This runs on the worker thread.
    fn open(paths: &MkPaths) -> Result<(Self, Spec), MkError> {
        let digest = gguf::spot_digest(&paths.shards)?;
        let library_sha256 = file_sha256(&paths.library)?;
        let fast_qmm = fast_qmm()?;
        let tiles = tiles()?;
        let binding = binding::load(&paths.library).map_err(MkError::new)?;
        // Refuse a library whose parts are not the declared size, before it writes past a Rust buffer.
        if let Some(problem) = spec::parts_problem(
            |part| unsafe { (binding.state_bytes)(part) },
            |part, n| unsafe { (binding.part_bytes)(part, n) },
        ) {
            return Err(MkError::new(problem));
        }

        let shards: Vec<CString> = paths
            .shards
            .iter()
            .map(|path| {
                CString::new(path.as_os_str().as_bytes())
                    .map_err(|_| MkError::new(format!("the shard path {} holds a NUL", path.display())))
            })
            .collect::<Result<_, _>>()?;
        let engine = unsafe { (binding.engine_open)(shards[0].as_ptr(), shards[1].as_ptr(), shards[2].as_ptr()) };
        if engine.is_null() {
            return Err(MkError::new("mk_engine_open returned no engine"));
        }
        let mut fd = -1i32;
        let mut arena_bytes = 0u64;
        unsafe { (binding.arena)(engine, &mut fd, &mut arena_bytes) };

        // mk stops the program on a pass that stops early by default (engine.h:215-220). Opt in, so a spin-timeout
        // returns -1 and poisons the contexts, and the wrapper reports PassStatus::Poison (D6, spec 8.2).
        unsafe { (binding.recover_spin_timeouts)(engine, 1) };

        let identity = spec::identity(&paths.shards, &paths.library, &library_sha256, fast_qmm, tiles, &digest);
        let limits = spec::limits(arena_bytes);
        Ok((
            Self {
                binding,
                engine,
                contexts: HashMap::new(),
                next_id: 0,
                fast_qmm,
                tiles,
            },
            Spec {
                identity,
                limits,
                recover_spin_timeouts: true,
            },
        ))
    }

    /// Serve commands until the handle side closes the channel.
    fn run(mut self, receiver: mpsc::Receiver<Command>) {
        for command in receiver {
            self.handle(command);
        }
    }

    fn handle(&mut self, command: Command) {
        match command {
            Command::Open { count, capacity, reply } => {
                let _ = reply.send(self.open_states(count, capacity));
            }
            Command::Free { ids, reply } => {
                for id in ids {
                    if let Some(context) = self.contexts.remove(&id) {
                        unsafe { (self.binding.ctx_free)(self.engine, context.ptr) };
                    }
                }
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
                let _ = reply.send(self.with_context(id, |state, context| unsafe {
                    (state.binding.ctx_owns_kv)(context.ptr) != 0
                }));
            }
            Command::Poisoned { id, reply } => {
                let _ = reply.send(self.with_context(id, |state, context| unsafe {
                    (state.binding.ctx_poisoned)(context.ptr) != 0
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
            Command::Pass { request, reply } => {
                let _ = reply.send(self.pass(request));
            }
            Command::PartBytes { part, n, reply } => {
                let _ = reply.send(unsafe { (self.binding.part_bytes)(part, n) });
            }
        }
    }

    fn open_states(&mut self, count: u32, capacity: u32) -> Result<Vec<u64>, String> {
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
        if unsafe { (self.binding.ctx_poisoned)(context.ptr) } != 0 {
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
        let poisoned = unsafe { (self.binding.ctx_poisoned)(ptr) } != 0;
        let n = unsafe { (self.binding.ctx_n)(ptr) };
        let owns_kv = unsafe { (self.binding.ctx_owns_kv)(ptr) } != 0;
        // mk reads the part as f32 values, so the source must be aligned for them.
        let bytes = Aligned::copy(bytes);
        match part {
            0..=2 => {
                if poisoned {
                    return Call::Failed("the context is poisoned; free it".to_owned());
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
                if unsafe { (self.binding.ctx_poisoned)(ptr) } != 0 {
                    return Call::Failed("the write poisoned the context".to_owned());
                }
                Call::Ok
            }
            3 | 4 => Call::Refused(format!("part {part} is cells; use cells_write with the row count")),
            5 => {
                if poisoned {
                    return Call::Failed("the context is poisoned; free it".to_owned());
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
                let first = i32::from_le_bytes([history[0], history[1], history[2], history[3]]);
                let second = i32::from_le_bytes([history[4], history[5], history[6], history[7]]);
                for (index, value) in [first, second].into_iter().enumerate() {
                    if value < 0 || value as u32 >= spec::VOCAB {
                        return Call::Refused(format!(
                            "ple_hist[{index}] = {value} is not a token id in 0..{}",
                            spec::VOCAB
                        ));
                    }
                }
                unsafe { (self.binding.state_write)(self.engine, ptr, part, bytes.as_ptr()) };
                if unsafe { (self.binding.ctx_poisoned)(ptr) } != 0 {
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
        let poisoned = unsafe { (self.binding.ctx_poisoned)(ptr) } != 0;
        let current = unsafe { (self.binding.ctx_n)(ptr) };
        let owns_kv = unsafe { (self.binding.ctx_owns_kv)(ptr) } != 0;
        if poisoned {
            return Call::Failed("the context is poisoned; free it".to_owned());
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
        if unsafe { (self.binding.ctx_poisoned)(ptr) } != 0 {
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
        let poisoned = unsafe { (self.binding.ctx_poisoned)(ptr) } != 0;
        let current = unsafe { (self.binding.ctx_n)(ptr) };
        let owns_kv = unsafe { (self.binding.ctx_owns_kv)(ptr) } != 0;
        let restored = (context.restored_kv, context.restored_idx, context.restored_ple);
        if poisoned {
            return Call::Failed("the context is poisoned; free it".to_owned());
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

    fn seal(&mut self, id: u64) -> Call {
        let context = match self.context(id) {
            Ok(context) => context,
            Err(call) => return call,
        };
        let ptr = context.ptr;
        if unsafe { (self.binding.ctx_poisoned)(ptr) } != 0 {
            return Call::Refused("the context is poisoned; free it".to_owned());
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
        if unsafe { (self.binding.ctx_poisoned)(ptr) } != 0 {
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
            if unsafe { (self.binding.ctx_poisoned)(context.ptr) } != 0 {
                return Err(problem(format!("context {id} is poisoned; free it")));
            }
            pointers.push(context.ptr);
        }

        let mut why = [0 as c_char; 512];
        let code = unsafe {
            (self.binding.segments_problem)(
                self.engine,
                count as u32,
                pointers.as_ptr(),
                request.lens.as_ptr(),
                why.as_mut_ptr(),
                why.len() as u32,
            )
        };
        if code != 0 {
            let message = unsafe { CStr::from_ptr(why.as_ptr()) }.to_string_lossy().into_owned();
            return Err(problem(message));
        }

        let mut logits = if request.want_logits {
            vec![0f32; count * spec::VOCAB as usize]
        } else {
            Vec::new()
        };
        let logits_ptr = if request.want_logits {
            logits.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        };
        let ms = unsafe {
            (self.binding.pass_segments)(
                self.engine,
                count as u32,
                pointers.as_ptr(),
                request.lens.as_ptr(),
                request.tokens.as_ptr(),
                1,
                logits_ptr,
                std::ptr::null_mut(),
            )
        };
        if ms < 0.0 {
            let mut buffer = [0 as c_char; 1024];
            unsafe { (self.binding.last_error)(self.engine, buffer.as_mut_ptr(), buffer.len() as u32) };
            let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(PassProblem {
                message: if message.is_empty() {
                    "the pass stopped and mk gave no reason".to_owned()
                } else {
                    message
                },
                poison: true,
            });
        }
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
/// A `Vec<u8>` holds no alignment promise. This buffer holds its bytes in `u64` words, so the pointer is 8-byte
/// aligned, and it copies to and from bytes without unsafe code.
struct Aligned {
    words: Vec<u64>,
    len: usize,
}

impl Aligned {
    /// A zeroed buffer of `len` bytes.
    fn zeroed(len: usize) -> Self {
        Self {
            words: vec![0u64; len.div_ceil(8)],
            len,
        }
    }

    /// A buffer that holds a copy of `bytes`.
    fn copy(bytes: &[u8]) -> Self {
        let mut buffer = Self::zeroed(bytes.len());
        for (word, chunk) in buffer.words.iter_mut().zip(bytes.chunks(8)) {
            let mut piece = [0u8; 8];
            piece[..chunk.len()].copy_from_slice(chunk);
            *word = u64::from_le_bytes(piece);
        }
        buffer
    }

    /// The buffer's length in bytes.
    fn len(&self) -> usize {
        self.len
    }

    /// The bytes of the buffer.
    fn to_vec(&self) -> Vec<u8> {
        self.words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .take(self.len)
            .collect()
    }

    /// The buffer as f32 values, in order, for a buffer that holds f32 data.
    fn f32_values(&self) -> impl Iterator<Item = f32> + '_ {
        (0..self.len / 4).map(move |index| {
            let word = self.words[index / 2];
            let bits = if index % 2 == 0 {
                word as u32
            } else {
                (word >> 32) as u32
            };
            f32::from_bits(bits)
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

    use super::{Aligned, fast_qmm_from, memory_problem, open_contexts, parse_shards, rounds_to_finite, tiles_from};

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
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let floats: Vec<f32> = Aligned::copy(&values).f32_values().collect();
        assert_eq!(floats, vec![1.0, -2.5, 3.25]);
    }

    #[test]
    fn a_load_that_leaves_less_than_the_floor_is_refused() {
        // spec 4.7: mk keeps 12 GiB of MemAvailable free (mk service/contexts.py, FLOOR_BYTES).
        let floor: u64 = 12 << 30;
        assert_eq!(memory_problem(floor, 0), Ok(()));
        assert!(memory_problem(floor - 1, 0).is_err());
        assert!(memory_problem(floor, 1).is_err());
        assert_eq!(memory_problem(floor + 1, 1), Ok(()));
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
