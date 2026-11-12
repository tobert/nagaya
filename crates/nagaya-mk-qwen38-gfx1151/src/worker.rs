//! The one thread that makes every engine call.
//!
//! The HIP device is the device that is current on the calling thread. One worker thread owns the engine, and every
//! call goes to it over a channel (spec 3.2). The engine is never closed: HIP teardown segfaults in the process (spec
//! 3.3).

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};

use nagaya_abi::{ControlItem, Identity, Limits};

use crate::arena::{self, Arena};
use crate::binding::{self, Binding, PassStop, Qwen38Segment};
use crate::canary;
use crate::control::{SegmentShape, TapSet};
use crate::cuts::{ControlTable, Cuts};
use crate::gguf;
use crate::host::{self, Host, LaunchKnobs, PassIn};
use crate::pass::{self, Knobs, Weights};
use crate::ple::{self, PleTable};
use crate::spec;

/// What kind of failure a `Qwen38Error` is. The process reads the kind to choose its exit code (spec 10.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen38ErrorKind {
    /// Any failure that a restart can change.
    Other,
    /// The canary of the build does not match its record, or the record is missing (spec 8.1). A restart cannot change it.
    Canary,
}

/// An error from the engine binding. The message says what failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen38Error {
    message: String,
    kind: Qwen38ErrorKind,
}

impl Qwen38Error {
    /// Make an error with a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: Qwen38ErrorKind::Other,
        }
    }

    /// Make an error of the canary: a mismatch with the record, or no record (spec 8.1).
    pub fn canary(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: Qwen38ErrorKind::Canary,
        }
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The kind of the failure.
    pub fn kind(&self) -> Qwen38ErrorKind {
        self.kind
    }
}

impl fmt::Display for Qwen38Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Qwen38Error {}

impl From<String> for Qwen38Error {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

/// Where the three weight shards are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen38Paths {
    /// The three shard paths (`QWEN38_SHARDS`, with `:` between them).
    pub shards: Vec<PathBuf>,
}

/// Split `QWEN38_SHARDS` on `:` as raw bytes, so a path that is not UTF-8 stays the same path.
fn parse_shards(raw: &[u8]) -> Result<Vec<PathBuf>, Qwen38Error> {
    let shards: Vec<PathBuf> = raw
        .split(|byte| *byte == b':')
        .map(|part| PathBuf::from(std::ffi::OsString::from_vec(part.to_vec())))
        .collect();
    if shards.len() != 3 || shards.iter().any(|path| path.as_os_str().is_empty()) {
        return Err(Qwen38Error::new(format!(
            "QWEN38_SHARDS has {} paths, but the Qwen3.8 engine needs three; write three paths with : between them",
            shards.len()
        )));
    }
    Ok(shards)
}

impl Qwen38Paths {
    /// Read the paths from `QWEN38_SHARDS`.
    pub fn from_env() -> Result<Self, Qwen38Error> {
        let shards = std::env::var_os("QWEN38_SHARDS")
            .ok_or_else(|| Qwen38Error::new("QWEN38_SHARDS is not set; give three GGUF paths with : between them"))?;
        let shards = parse_shards(shards.as_os_str().as_bytes())?;
        Ok(Self { shards })
    }

    /// Check that every path exists and every shard has a GGUF v3 header.
    fn validate(&self) -> Result<Vec<gguf::ShardHeader>, Qwen38Error> {
        let mut headers = Vec::new();
        for path in &self.shards {
            if !path.is_file() {
                return Err(Qwen38Error::new(format!("the shard {} is missing", path.display())));
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
    /// The key of the canary record: the checkpoint and the two settings.
    pub canary_key: canary::Key,
}

/// Why a write or a seal refuses a context that was poisoned before the call. The call changed nothing (ABI, "How
/// the Qwen3.8 engine maps", part writes).
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
    /// True when the engine ran the pass and poisoned its contexts.
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
    /// True for each segment that asks for its logit row.
    pub wants: Vec<bool>,
    /// The control list of each segment.
    pub control: Vec<Vec<ControlItem>>,
    /// The taps that the pass names.
    pub taps: TapSet,
}

/// The result of a pass that ran.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PassReply {
    /// The pass wall time in milliseconds.
    pub ms: f64,
    /// The logits, `[segments][vocab]` f32 little-endian, when asked.
    pub logits: Option<Vec<u8>>,
    /// The tap outputs in the order of the layout (`control::tap_layout`): the name and the f32 little-endian bytes.
    pub taps: Vec<(&'static str, Vec<u8>)>,
}

/// The worker thread's engine and its contexts.
struct ThreadState {
    binding: Binding,
    engine: *mut c_void,
    /// The reader's maps of the shards, the PLE table in them, and the weight arena. The host gathers PLE rows from
    /// the maps for each pass, and the kernels read the arena. A stub has none of them.
    model: Option<gguf::Model>,
    ple: Option<PleTable>,
    _arena: Option<Arena>,
    /// The arena's tensors by role, and the host side of the pass: the scratch pool and the fixed buffers.
    weights: Option<Weights>,
    host: Option<Host>,
    contexts: HashMap<u64, Context>,
    next_id: u64,
    fast_qmm: bool,
    tiles: bool,
    /// What a load after a close needs (engine-abi "close"): the shards, the spot digest and the arena size of the
    /// first load. A load whose shards differ is refused, because the identity names the first ones. A stub has none.
    reload: Option<Reload>,
    /// What the canary of the last load gave: the report, or why it did not run. Each load sets it (spec 8.1).
    load_canary: Option<Result<canary::Report, String>>,
}

/// The facts of the first load, which a later load must give again.
struct Reload {
    paths: Qwen38Paths,
    digest: String,
    arena_bytes: u64,
}

/// One context that a handle owns.
struct Context {
    ptr: *mut c_void,
    capacity: u32,
    restored_kv: u32,
    restored_idx: u32,
    restored_ple: bool,
}

/// What one class of contexts took when the engine made it. The host readings are `None` when `/proc/meminfo` fails.
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
    /// Write one vector into an entry of the control table.
    ControlWrite {
        index: u32,
        bytes: Vec<u8>,
        reply: mpsc::Sender<Result<(), String>>,
    },
    /// Load the weights again after an unload. A loaded engine does nothing.
    Load {
        reply: mpsc::Sender<Result<(), String>>,
    },
    /// Free the weights: the arena, the shard maps, the PLE table and the weight table. Every context must be free.
    Unload {
        reply: mpsc::Sender<Result<(), String>>,
    },
    /// Give the result of the canary of the last load.
    LoadCanary {
        reply: mpsc::Sender<Result<canary::Report, String>>,
    },
    /// Run the canary (spec 8.1) in two contexts of its own, and free them.
    Canary {
        reply: mpsc::Sender<Result<canary::Report, String>>,
    },
}

/// The handle's side of the worker: a channel and the engine's declared facts.
pub(crate) struct Worker {
    sender: Mutex<mpsc::Sender<Command>>,
    spec: Spec,
    paths: Qwen38Paths,
    /// The handles and holds that use the weights. The last one to go unloads them; the first one loads them again.
    handles: Mutex<usize>,
}

impl Worker {
    /// The identity and the limits of the open engine.
    pub fn spec(&self) -> &Spec {
        &self.spec
    }

    /// The paths that the engine was opened with.
    pub fn paths(&self) -> &Qwen38Paths {
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

    /// Count one more user of the weights. The first user after an unload loads them again.
    pub fn acquire(&self) -> Result<(), Qwen38Error> {
        let mut handles = self.handles.lock().expect("the lock of the engine handles is poisoned");
        if *handles == 0 {
            self.ask(|reply| Command::Load { reply })
                .unwrap_or_else(|| Err("the engine worker stopped; restart the process".to_owned()))
                .map_err(Qwen38Error::from)?;
        }
        *handles += 1;
        Ok(())
    }

    /// Count one user more, as `acquire` does, and decide the open from the canary of the load (spec 8.1).
    ///
    /// The canary runs at each load of the weights, after the weights load. A handle that joins loaded weights is
    /// decided by the canary of that load. A refused open leaves no user counted. It gives the report of the load.
    pub fn acquire_checked(&self, check: canary::CanaryCheck) -> Result<canary::Report, Qwen38Error> {
        self.acquire()?;
        let run = self
            .ask(|reply| Command::LoadCanary { reply })
            .unwrap_or_else(|| Err("the engine worker stopped; restart the process".to_owned()));
        match canary::verify(check, &self.spec.canary_key, &run) {
            Ok(()) => Ok(run.expect("a verified canary has a report")),
            Err(error) => {
                self.release();
                Err(error)
            }
        }
    }

    /// Count one user less. The last user unloads the weights (engine-abi "close"). An unload that fails is a defect:
    /// the ledger would count memory as free that the engine still holds, so the program stops.
    pub fn release(&self) {
        let mut handles = self.handles.lock().expect("the lock of the engine handles is poisoned");
        *handles = handles.checked_sub(1).expect("an engine release without an acquire");
        if *handles == 0
            && let Some(Err(error)) = self.ask(|reply| Command::Unload { reply })
        {
            panic!("the Qwen3.8 engine could not free its weights: {error}");
        }
    }

    /// Create a context pool for one handle. The ids follow the order of the classes.
    pub fn open_states(&self, classes: &[spec::ContextClass]) -> Result<Pool, Qwen38Error> {
        let classes = classes.to_vec();
        self.ask(|reply| Command::Open { classes, reply })
            .unwrap_or_else(|| Err("the engine worker stopped; restart the process".to_owned()))
            .map_err(Qwen38Error::from)
    }

    /// Free a handle's contexts.
    pub fn free_states(&self, ids: Vec<u64>) {
        if ids.is_empty() {
            return;
        }
        let _ = self.ask(|reply| Command::Free { ids, reply });
    }

    /// The context's token count.
    pub fn n(&self, id: u64) -> Result<u32, Qwen38Error> {
        self.ask(|reply| Command::N { id, reply })
            .unwrap_or_else(|| Err("the engine worker stopped; restart the process".to_owned()))
            .map_err(Qwen38Error::from)
    }

    /// The context's capacity.
    pub fn max(&self, id: u64) -> Result<u32, Qwen38Error> {
        self.ask(|reply| Command::Max { id, reply })
            .unwrap_or_else(|| Err("the engine worker stopped; restart the process".to_owned()))
            .map_err(Qwen38Error::from)
    }

    /// True when the context owns its K/V.
    pub fn owns_kv(&self, id: u64) -> Result<bool, Qwen38Error> {
        self.ask(|reply| Command::OwnsKv { id, reply })
            .unwrap_or_else(|| Err("the engine worker stopped; restart the process".to_owned()))
            .map_err(Qwen38Error::from)
    }

    /// True when the context is poisoned.
    pub fn poisoned(&self, id: u64) -> Result<bool, Qwen38Error> {
        self.ask(|reply| Command::Poisoned { id, reply })
            .unwrap_or_else(|| Err("the engine worker stopped; restart the process".to_owned()))
            .map_err(Qwen38Error::from)
    }

    /// Read one whole part into a new buffer of `len` bytes.
    pub fn read(&self, id: u64, part: usize, len: usize) -> Result<Vec<u8>, Call> {
        let part = part_id(part)?;
        self.ask(|reply| Command::Read { id, part, len, reply })
            .unwrap_or_else(|| {
                Err(Call::Failed(
                    "the engine worker stopped; restart the process".to_owned(),
                ))
            })
    }

    /// Write one fixed part.
    pub fn write(&self, id: u64, part: usize, bytes: Vec<u8>) -> Call {
        let part = match part_id(part) {
            Ok(part) => part,
            Err(call) => return call,
        };
        self.ask(|reply| Command::Write { id, part, bytes, reply })
            .unwrap_or_else(|| Call::Failed("the engine worker stopped; restart the process".to_owned()))
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
        .unwrap_or_else(|| Call::Failed("the engine worker stopped; restart the process".to_owned()))
    }

    /// Set the context's token count.
    pub fn set_n(&self, id: u64, n: u32) -> Call {
        self.ask(|reply| Command::SetN { id, n, reply })
            .unwrap_or_else(|| Call::Failed("the engine worker stopped; restart the process".to_owned()))
    }

    /// Run the boundary rule of the engine.
    pub fn seal(&self, id: u64) -> Call {
        self.ask(|reply| Command::Seal { id, reply })
            .unwrap_or_else(|| Call::Failed("the engine worker stopped; restart the process".to_owned()))
    }

    /// Empty the context: free it and make a new one of the same capacity under the same id.
    pub fn clear(&self, id: u64) -> Call {
        self.ask(|reply| Command::Clear { id, reply })
            .unwrap_or_else(|| Call::Failed("the engine worker stopped; restart the process".to_owned()))
    }

    /// Run one pass.
    pub fn pass(&self, request: PassRequest) -> Result<PassReply, PassProblem> {
        self.ask(|reply| Command::Pass { request, reply }).unwrap_or_else(|| {
            Err(PassProblem {
                message: "the engine worker stopped; restart the process".to_owned(),
                poison: false,
            })
        })
    }

    /// Run the canary on the worker thread (spec 8.1). It uses two contexts of its own and touches no pool context.
    pub fn canary(&self) -> Result<canary::Report, Qwen38Error> {
        self.ask(|reply| Command::Canary { reply })
            .unwrap_or_else(|| Err("the engine worker stopped; restart the process".to_owned()))
            .map_err(Qwen38Error::from)
    }

    /// The engine's byte count for one part and token count, or `None` when the worker stopped.
    pub fn part_bytes(&self, part: u32, n: u32) -> Option<u64> {
        self.ask(|reply| Command::PartBytes { part, n, reply })
    }

    /// Write one vector into an entry of the control table.
    pub fn control_write(&self, index: u32, bytes: Vec<u8>) -> Result<(), String> {
        self.ask(|reply| Command::ControlWrite { index, bytes, reply })
            .unwrap_or_else(|| Err("the engine worker stopped; restart the process".to_owned()))
    }
}

/// The 32-bit part id that the engine takes, or a refusal. A cast would read part 2^32 + 3 as part 3.
fn part_id(part: usize) -> Result<u32, Call> {
    u32::try_from(part)
        .map_err(|_| Call::Refused(format!("the Qwen3.8 engine has no part {part}; use a part from 0 to 5")))
}

/// Start the worker thread and open the engine on it.
pub(crate) fn start(paths: Qwen38Paths) -> Result<Arc<Worker>, Qwen38Error> {
    let headers = paths.validate()?;
    let total: u64 = headers.iter().map(|header| header.size).sum();
    check_memory(total)?;

    let (sender, receiver) = mpsc::channel();
    let (ready, first) = mpsc::channel();
    let thread_paths = paths.clone();
    std::thread::Builder::new()
        .name("qwen38-engine".to_owned())
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
        .map_err(|error| Qwen38Error::new(format!("start the engine worker: {error}")))?;

    let spec = first
        .recv()
        .map_err(|error| Qwen38Error::new(format!("the engine worker did not answer: {error}")))??;
    Ok(Arc::new(Worker {
        sender: Mutex::new(sender),
        spec,
        paths,
        handles: Mutex::new(0),
    }))
}

/// The identity and the limits of the engine on `paths`, with no GPU and no weights in memory: the facts that `open`
/// gives after the load. The process serves an engine on demand from them before it loads (spec 11.1).
pub(crate) fn describe(paths: &Qwen38Paths, check: canary::CanaryCheck) -> Result<(Identity, Limits), Qwen38Error> {
    paths.validate()?;
    let digest = gguf::spot_digest(&paths.shards)?;
    let fast_qmm = fast_qmm()?;
    let tiles = tiles()?;
    let (_, build) = binding::linked().map_err(Qwen38Error::new)?;
    let text = gguf::text_keys(&paths.shards[0])?;
    let model = gguf::Model::open(&paths.shards)?;
    let (_, arena_bytes) = arena::plan(model.tensors(), tiles)?;
    let key = canary::Key {
        checkpoint: format!("spot-sha256:{digest}"),
        fast_qmm,
        tiles,
    };
    let canary_hash = canary::identity_hash(check, &key)?;
    Ok((
        spec::identity(&paths.shards, &build, fast_qmm, tiles, &digest, &text, &canary_hash),
        spec::limits(arena_bytes),
    ))
}

/// Read the shards, plan the arena and fill it (milestone 15 step 6): the shard maps, the PLE table, the arena and the
/// weight table.
fn load_weights(
    binding: &Binding,
    paths: &Qwen38Paths,
    tiles: bool,
) -> Result<(gguf::Model, PleTable, Arena, Weights), Qwen38Error> {
    let model = gguf::Model::open(&paths.shards)?;
    let ple = PleTable::new(&model)?;
    let (planned, arena_bytes) = arena::plan(model.tensors(), tiles)?;
    let arena = unsafe { arena::create(binding, &model, &planned, arena_bytes, tiles) }?;
    let tensors = planned
        .iter()
        .map(|p| {
            let tensor = pass::Tensor {
                kind: p.kind,
                ne: p.ne,
                row_bytes: p.row_bytes,
                d: arena.tensor(p),
            };
            (p.name.clone(), tensor)
        })
        .collect();
    let weights = Weights::new(&tensors)?;
    Ok((model, ple, arena, weights))
}

/// Read `QWEN38_FAST_QMM` as the engine reads it: unset or 1 is fast, 0 is exact, and any other value stops the engine.
fn fast_qmm() -> Result<bool, Qwen38Error> {
    match std::env::var("QWEN38_FAST_QMM") {
        Ok(value) => fast_qmm_from(Some(&value)),
        Err(std::env::VarError::NotPresent) => fast_qmm_from(None),
        Err(error) => Err(Qwen38Error::new(format!("QWEN38_FAST_QMM: {error}"))),
    }
}

/// Read `QWEN38_TILES` as the engine reads it: unset or 1 is tiled, 0 is the GGUF layout, and any other value stops the
/// engine.
fn tiles() -> Result<bool, Qwen38Error> {
    match std::env::var("QWEN38_TILES") {
        Ok(value) => tiles_from(Some(&value)),
        Err(std::env::VarError::NotPresent) => tiles_from(None),
        Err(error) => Err(Qwen38Error::new(format!("QWEN38_TILES: {error}"))),
    }
}

/// The value of one `QWEN38_TILES` string.
fn tiles_from(value: Option<&str>) -> Result<bool, Qwen38Error> {
    match value {
        Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(other) => Err(Qwen38Error::new(format!("QWEN38_TILES={other}: want 0 or 1"))),
        None => Ok(true),
    }
}

/// The value of one `QWEN38_FAST_QMM` string.
fn fast_qmm_from(value: Option<&str>) -> Result<bool, Qwen38Error> {
    match value {
        Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(other) => Err(Qwen38Error::new(format!("QWEN38_FAST_QMM={other}: want 0 or 1"))),
        None => Ok(true),
    }
}

/// The value of an environment variable as text. A value that is not UTF-8 stays visible, so a knob refuses it.
fn env_text(name: &str) -> Option<String> {
    std::env::var_os(name).map(|value| value.into_string().unwrap_or_else(|_| "<not UTF-8>".to_owned()))
}

/// The watchdog's deadline for each launch sequence: 10 s, or `QWEN38_WATCHDOG_DEADLINE_S` (a test knob) in (0, 3600].
/// The longest pass is a bit over a second (reference `PLAN.md`).
fn watchdog_deadline(value: Option<&str>) -> Result<f64, Qwen38Error> {
    let Some(value) = value else {
        return Ok(10.0);
    };
    match value.parse::<f64>() {
        Ok(seconds) if seconds > 0.0 && seconds <= 3600.0 => Ok(seconds),
        _ => Err(Qwen38Error::new(format!(
            "QWEN38_WATCHDOG_DEADLINE_S={value}: want seconds in (0, 3600]"
        ))),
    }
}

/// The host safety floor. The engine keeps 12 GiB of MemAvailable free (spec 4.7; reference `service/contexts.py`,
/// `FLOOR_BYTES`).
const MEMORY_FLOOR: u64 = 12 << 30;

/// Refuse an allocation of `needed` bytes that leaves less than the floor free. `what` names the allocation.
fn memory_problem(what: &str, available: u64, needed: u64) -> Result<(), String> {
    if available < needed.saturating_add(MEMORY_FLOOR) {
        return Err(format!(
            "{what} takes {needed} bytes in memory, but only {available} bytes are available and the floor is {MEMORY_FLOOR}; \
             free memory or load a smaller model"
        ));
    }
    Ok(())
}

/// The host's MemAvailable in bytes.
fn mem_available() -> Result<u64, Qwen38Error> {
    let text = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| Qwen38Error::new(format!("/proc/meminfo: {error}")))?;
    text.lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .map(|kilobytes| kilobytes * 1024)
        .ok_or_else(|| Qwen38Error::new("/proc/meminfo has no MemAvailable"))
}

/// Refuse a load that cannot fit in the free memory. The engine allocates the weight arena in full, and the floor stays
/// free.
fn check_memory(needed: u64) -> Result<(), Qwen38Error> {
    memory_problem("the shards", mem_available()?, needed).map_err(Qwen38Error::new)
}

/// Refuse an open that the free memory cannot hold, before it starts (spec 3.3b, "The memory check of an open").
///
/// `freed` is the memory that an unload before the open returns, at most. It counts as available. The rule is the one
/// of `check_memory`, which the open runs.
fn open_memory_problem(needed: u64, available: u64, freed: u64) -> Result<(), String> {
    if available.saturating_add(freed) >= needed.saturating_add(MEMORY_FLOOR) {
        return Ok(());
    }
    Err(format!(
        "the shards take {needed} bytes in memory, but only {available} bytes are available, the unload frees at most \
         {freed} bytes, and the floor is {MEMORY_FLOOR} bytes; name the loaded engines of the group in `unload`, or \
         free memory"
    ))
}

/// The problem that an open of the engine on `paths` would meet now, or `None`. It reads the sizes of the shards and
/// the free memory, as the open does.
pub(crate) fn open_problem(paths: &Qwen38Paths, freed: u64) -> Option<String> {
    let needed: u64 = match paths.validate() {
        Ok(headers) => headers.iter().map(|header| header.size).sum(),
        Err(error) => return Some(error.to_string()),
    };
    match mem_available() {
        Ok(available) => open_memory_problem(needed, available, freed).err(),
        Err(error) => Some(error.to_string()),
    }
}

/// The device bytes that `what` took: the free bytes before minus the free bytes after.
///
/// Free memory that grew is an error with both readings. A saturating difference would read 0 bytes, and the class
/// check would pass (kaibo review, finding 7).
fn taken_bytes(before: u64, after: u64, what: &str) -> Result<u64, String> {
    before
        .checked_sub(after)
        .ok_or_else(|| format!("the free device memory grew from {before} to {after} bytes while the engine made {what}; stop the other GPU client"))
}

/// Free one context, then make a new one of `capacity` tokens. The engine has no call that empties a context.
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
            "the engine made no context of {capacity} tokens; the slot has no context now"
        ));
    }
    let made = max(ptr);
    if made != capacity {
        free(ptr);
        return Err(format!(
            "the engine made a context of {made} tokens, not {capacity}; the slot has no context now"
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
            return Err(format!("the engine made no context of {capacity} tokens"));
        }
        let max_tokens = max(ptr);
        made.push((ptr, max_tokens));
    }
    Ok(made)
}

/// The two pass shapes of the scratch probe: the token count and whether the pass asks for logits.
///
/// The first shape has 1 token, so the engine asks for `qmm_part2` too. The second shape asks for every other buffer at
/// `kMaxPassTokens`. The token counts add to the probe context's capacity, so one context serves both.
fn probe_shapes(capacity: u32) -> [(u32, bool); 2] {
    let rest = capacity
        .checked_sub(1)
        .expect("the probe context holds at least two tokens");
    [(1, false), (rest, true)]
}

impl ThreadState {
    /// Bind the linked engine and open it. This runs on the worker thread.
    fn open(paths: &Qwen38Paths) -> Result<(Self, Spec), Qwen38Error> {
        let digest = gguf::spot_digest(&paths.shards)?;
        let fast_qmm = fast_qmm()?;
        let tiles = tiles()?;
        let (binding, build) = binding::linked().map_err(Qwen38Error::new)?;
        // Refuse a library whose parts are not the declared size, before it writes past a Rust buffer.
        if let Some(problem) = spec::parts_problem(
            |part| unsafe { (binding.state_bytes)(part) },
            |part, n| unsafe { (binding.part_bytes)(part, n) },
        ) {
            return Err(Qwen38Error::new(problem));
        }

        // The tokenizer is in the first shard. Hash it before the weights load, so a shard with no tokenizer fails fast.
        let text = gguf::text_keys(&paths.shards[0])?;

        let (model, ple, arena, weights) = load_weights(&binding, paths, tiles)?;
        let arena_bytes = arena.bytes;
        let deadline = watchdog_deadline(env_text("QWEN38_WATCHDOG_DEADLINE_S").as_deref())?;
        let engine = unsafe { (binding.engine_open)(deadline) };
        if engine.is_null() {
            return Err(Qwen38Error::new("qwen38_engine_open returned no engine"));
        }
        // The fixed buffers of a pass, as the engine made them at load: before the scratch probe.
        let host = unsafe { Host::new(&binding) };

        let mut state = Self {
            binding,
            engine,
            model: Some(model),
            ple: Some(ple),
            _arena: Some(arena),
            weights: Some(weights),
            host: Some(host),
            contexts: HashMap::new(),
            next_id: 0,
            fast_qmm,
            tiles,
            reload: Some(Reload {
                paths: paths.clone(),
                digest: digest.clone(),
                arena_bytes,
            }),
            load_canary: None,
        };
        // The engine has no call for the pool's size. Measure it here, before a caller's pass, so a larger pool is an
        // open error and the ledger's term is what the engine holds.
        let pass_scratch = state.measure_pass_scratch().map_err(Qwen38Error::new)?;
        spec::scratch_problem(pass_scratch).map_err(Qwen38Error::new)?;
        // The table of control vectors (milestone 34). It comes after the probe, so the probe measures the same pool.
        state.make_control_table();
        // The canary runs after the scratch probe, so that it does not change what the probe measures. The open
        // decides from its result (`acquire_checked`), so a mismatch does not poison the one worker of the process.
        state.run_load_canary();

        let canary_key = canary::Key {
            checkpoint: format!("spot-sha256:{digest}"),
            fast_qmm,
            tiles,
        };
        let canary_hash = canary::record(&canary_key).unwrap_or_default();
        let identity = spec::identity(&paths.shards, &build, fast_qmm, tiles, &digest, &text, canary_hash);
        let limits = spec::limits(arena_bytes);
        Ok((
            state,
            Spec {
                identity,
                limits,
                pass_scratch,
                canary_key,
            },
        ))
    }

    /// Load the weights again after an unload, from the same shards. A loaded engine does nothing.
    fn load_again(&mut self) -> Result<(), String> {
        if self.weights.is_some() {
            return Ok(());
        }
        let Some(reload) = &self.reload else {
            return Err("this engine has no shards to load".to_owned());
        };
        let headers = reload.paths.validate().map_err(|error| error.to_string())?;
        check_memory(headers.iter().map(|header| header.size).sum()).map_err(|error| error.to_string())?;
        let digest = gguf::spot_digest(&reload.paths.shards).map_err(|error| error.to_string())?;
        if digest != reload.digest {
            return Err(
                "the shards changed since the first load, and the identity names the first ones; restart the \
                        process"
                    .to_owned(),
            );
        }
        let (model, ple, arena, weights) =
            load_weights(&self.binding, &reload.paths, self.tiles).map_err(|error| error.to_string())?;
        if arena.bytes != reload.arena_bytes {
            return Err(format!(
                "the arena of the load has {} bytes, and the first load had {}; restart the process",
                arena.bytes, reload.arena_bytes
            ));
        }
        self.model = Some(model);
        self.ple = Some(ple);
        self._arena = Some(arena);
        self.weights = Some(weights);
        self.run_load_canary();
        Ok(())
    }

    /// Run the canary of a load and keep its result. The open decides from it (spec 8.1).
    fn run_load_canary(&mut self) {
        canary::count_load_run();
        self.load_canary = Some(self.run_canary());
    }

    /// Free the weights (engine-abi "close"). The weight table and the PLE table point into the arena and the shard
    /// maps, so they go first. The pass host, the C++ engine and the HIP runtime stay (Amy, 2026-11-01: "mk can stay
    /// loaded and use some memory").
    fn unload(&mut self) -> Result<(), String> {
        if !self.contexts.is_empty() {
            return Err(format!(
                "{} contexts are open; free them before the weights",
                self.contexts.len()
            ));
        }
        self.weights = None;
        self.ple = None;
        self._arena = None;
        self.model = None;
        Ok(())
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
    /// A pass of 1 token asks for `qmm_part2`, and a pass of 17 or more tokens does not (reference `engine.hip:1187`).
    /// The second pass has the 255 tokens that remain, with logits, so it asks for every other buffer at
    /// `kMaxPassTokens`. The engine keeps the pool for its life. The probe context is freed before the second reading.
    fn measure_pass_scratch(&mut self) -> Result<u64, String> {
        let before = self.free_device_bytes()?;
        let context = unsafe { (self.binding.ctx_new)(self.engine, spec::PROBE_CAPACITY) };
        if context.is_null() {
            return Err(format!(
                "the engine made no context of {} tokens for the pass scratch probe",
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
    fn probe_passes(&mut self, context: *mut c_void) -> Result<(), String> {
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
    fn probe_pass(&mut self, context: *mut c_void, tokens: &[i32], want_logits: bool) -> Result<(), String> {
        let segments = binding::segments(&[context], &[tokens.len() as u32], tokens);
        let rows = usize::from(want_logits);
        let mut logits = vec![0f32; rows * spec::VOCAB as usize];
        let logits = want_logits.then_some(logits.as_mut_slice());
        match self.run_pass(&segments, logits) {
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
            Command::ControlWrite { index, bytes, reply } => {
                let _ = reply.send(self.control_write(index, &bytes));
            }
            Command::Load { reply } => {
                let _ = reply.send(self.load_again());
            }
            Command::Unload { reply } => {
                let _ = reply.send(self.unload());
            }
            Command::LoadCanary { reply } => {
                let _ = reply.send(
                    self.load_canary
                        .clone()
                        .unwrap_or_else(|| Err("no load ran a canary; this is a defect of the engine".to_owned())),
                );
            }
            Command::Canary { reply } => {
                let _ = reply.send(self.run_canary());
            }
        }
    }

    /// Create the contexts of every class, in order, and read what each class took.
    ///
    /// A later class that fails frees the contexts of the earlier ones. The pool is whole or not made.
    fn open_pool(&mut self, classes: &[spec::ContextClass]) -> Result<Pool, String> {
        // Admit the whole pool before the engine makes any of it (spec 4.3, 4.7).
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

    /// Run the canary (spec 8.1) in two contexts that this call makes and frees.
    ///
    /// The contexts are not in any handle's pool, so the pool contexts keep their state. Each pass goes through
    /// `pass`, the call that a client pass takes: the same checks, the same fast or exact path, and the same join of
    /// segments. The contexts are freed on success and on every error. The run reports only: it checks no record.
    fn run_canary(&mut self) -> Result<canary::Report, String> {
        if self.weights.is_none() {
            return Err("the canary needs the weights, and they are not loaded; open a handle first".to_owned());
        }
        let capacity = |which: canary::Which| {
            which.tokens_needed().div_ceil(spec::CAPACITY_STEP as usize) as u32 * spec::CAPACITY_STEP
        };
        let classes = [canary::Which::Short, canary::Which::Long].map(|which| spec::ContextClass {
            count: 1,
            capacity: capacity(which),
        });
        let needed: u64 = classes
            .iter()
            .map(|class| spec::context_declared_bytes(class.capacity))
            .sum();
        memory_problem(
            "the canary contexts",
            mem_available().map_err(|error| error.to_string())?,
            needed,
        )?;
        let mut ids = Vec::with_capacity(classes.len());
        for class in classes {
            match self.open_states(class) {
                Ok(made) => ids.extend(made),
                Err(error) => {
                    self.free_ids(&ids);
                    return Err(error);
                }
            }
        }
        let result = canary::run(|call| {
            // The canary is a plain pass: each segment gives its logit row, with no control list and no tap. So it
            // runs as one launch, and a tap or a control list of a client cannot change its bits.
            let request = PassRequest {
                contexts: call.contexts.iter().map(|which| ids[*which as usize]).collect(),
                lens: call.lens.clone(),
                tokens: call.tokens.clone(),
                want_logits: true,
                wants: vec![true; call.lens.len()],
                control: vec![Vec::new(); call.lens.len()],
                taps: TapSet::default(),
            };
            let reply = self.pass(request).map_err(|problem| {
                format!(
                    "a canary pass failed: {}{}",
                    problem.message,
                    if problem.poison {
                        " (the pass poisoned its contexts)"
                    } else {
                        ""
                    }
                )
            })?;
            let bytes = reply.logits.ok_or_else(|| {
                "a canary pass gave no logits; the engine must give one row for each segment".to_owned()
            })?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .chunks(spec::VOCAB as usize)
                .map(|row| row.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect())
                .collect())
        });
        self.free_ids(&ids);
        result
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
            other => {
                return Err(Call::Refused(format!(
                    "the Qwen3.8 engine has no part {other}; use a part from 0 to 5"
                )));
            }
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
        // The engine reads the part as f32 values, so the source must be aligned for them.
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
            other => Call::Refused(format!(
                "the Qwen3.8 engine has no part {other}; use a part from 0 to 5"
            )),
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
        // The engine reads the rows as f32 values, so the source must be aligned for them.
        let bytes = Aligned::copy(bytes);
        let row = match part {
            3 => spec::KV_ROW_BYTES,
            4 => spec::IDX_ROW_BYTES,
            other => {
                return Call::Refused(format!(
                    "the Qwen3.8 engine has cells only in parts 3 and 4, not {other}; use part 3 or 4"
                ));
            }
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
        // The pooled keys are derived state: pool every complete block of the written index keys.
        if let (Some(host), Some(weights)) = (self.host.as_mut(), self.weights.as_ref()) {
            let context = unsafe { &*(ptr as *const crate::context::Qwen38Context) };
            unsafe { host::pool_context(&self.binding, self.engine, host, weights, context) };
        }
        // The engine returns nothing here. A call that left the context poisoned is a failure (spec 6.8), so check
        // first.
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
        // qwen38_ctx_round_gdn_bf16 stops the program on a non-finite value, or one that rounds to infinity. Check
        // first.
        let mut state = Aligned::zeroed(spec::GDN_STATE_BYTES as usize);
        unsafe { (self.binding.state_read)(self.engine, ptr, 0, state.as_mut_ptr()) };
        for (index, value) in state.f32_values().enumerate() {
            if !value.is_finite() || !rounds_to_finite(value) {
                return Call::Refused(format!(
                    "gdn_state[{index}] = {value} is not finite after a bf16 round, and the Qwen3.8 engine stops on it; write finite values only"
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
        // The engine reads QWEN38_FAST_QMM at each pass build. A change after open would change the bits, and the
        // identity would be wrong, so refuse the pass.
        match fast_qmm() {
            Ok(value) if value == self.fast_qmm => {}
            Ok(value) => {
                return Err(problem(format!(
                    "QWEN38_FAST_QMM changed from {} to {} after the engine opened",
                    u8::from(self.fast_qmm),
                    u8::from(value)
                )));
            }
            Err(error) => return Err(problem(error.to_string())),
        }
        // The engine reads QWEN38_TILES once, at load. A change after open makes the environment disagree with the
        // identity, so refuse the pass, as for QWEN38_FAST_QMM.
        match tiles() {
            Ok(value) if value == self.tiles => {}
            Ok(value) => {
                return Err(problem(format!(
                    "QWEN38_TILES changed from {} to {} after the engine opened",
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
                "{total} tokens are past the {}-token limit of a pass of the Qwen3.8 engine; split the pass",
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

        // The binding runs the segment check of the engine first, because the engine stops the program on a bad
        // segment.
        let segments = binding::segments(&pointers, &request.lens, &request.tokens);
        let mut cuts = self.plan_cuts(&request).map_err(problem)?;
        let rows = if request.want_logits { count } else { 0 };
        let mut logits = vec![0f32; rows * spec::VOCAB as usize];
        let buffer = request.want_logits.then_some(logits.as_mut_slice());
        let ms = match self.run_pass_cut(&segments, buffer, cuts.as_mut()) {
            Ok(ms) => ms,
            Err(PassStop::Refused(message)) => return Err(problem(message)),
            Err(PassStop::Stopped(message)) => return Err(PassProblem { message, poison: true }),
        };
        let logits = request
            .want_logits
            .then(|| logits.iter().flat_map(|value| value.to_le_bytes()).collect());
        // The pass ran, so a tap output of the wrong size is a fault of the engine: the contexts have advanced.
        let taps = match cuts.map(Cuts::into_outputs).transpose() {
            Ok(taps) => taps.unwrap_or_default(),
            Err(message) => return Err(PassProblem { message, poison: true }),
        };
        Ok(PassReply { ms, logits, taps })
    }
}

impl ThreadState {
    /// The cuts of a pass: `None` for a plain pass. The pass is refused when a list names an entry that no write
    /// filled, or when the plan refuses it (`control::plan_cuts`).
    fn plan_cuts(&self, request: &PassRequest) -> Result<Option<Cuts>, String> {
        let count = request.contexts.len();
        if request.wants.len() != count || request.control.len() != count {
            return Err("the segment, logit-row and control-list counts differ".to_owned());
        }
        let steered = request.control.iter().any(|list| !list.is_empty());
        if !steered && !request.taps.any() {
            return Ok(None);
        }
        let shapes: Vec<SegmentShape> = (0..count)
            .map(|i| SegmentShape {
                rows: request.lens[i] as usize,
                logit_row: request.wants[i],
                control: &request.control[i],
            })
            .collect();
        let cuts = Cuts::new(&shapes, request.taps)?;
        if steered {
            let table = self.host.as_ref().and_then(|host| host.control.as_ref());
            let Some(table) = table else {
                return Err("the pass has control items, but the engine has no control table".to_owned());
            };
            let lists: Vec<&[ControlItem]> = request.control.iter().map(Vec::as_slice).collect();
            if let Some(problem) = table.unwritten(&lists) {
                return Err(problem);
            }
        }
        Ok(Some(cuts))
    }

    /// Make the control table on the device. A stub has no pass host, so it gets no table.
    fn make_control_table(&mut self) {
        if let Some(host) = self.host.as_mut() {
            // SAFETY: the binding is the linked engine's, and this runs on the worker thread.
            host.control = Some(unsafe { ControlTable::new(&self.binding) });
        }
    }

    /// Write one vector into an entry of the control table. It makes no allocation.
    fn control_write(&mut self, index: u32, bytes: &[u8]) -> Result<(), String> {
        let Some(table) = self.host.as_mut().and_then(|host| host.control.as_mut()) else {
            return Err("the engine has no control table".to_owned());
        };
        // SAFETY: the binding is the linked engine's, and this runs on the worker thread.
        unsafe { table.write(&self.binding, index, bytes) }
    }

    /// The knobs of the pass build from the environment. A bad value refuses the pass.
    fn knobs(&self, want_logits: bool) -> Result<Knobs, String> {
        let knobs = Knobs::from(self.fast_qmm, env_text)?;
        if knobs.debug_layers > 0 && want_logits {
            return Err("QWEN38_DEBUG_LAYERS builds no head; ask for no logits".to_owned());
        }
        Ok(knobs)
    }

    /// One pass of `segments` with no cut (`run_pass_cut`).
    fn run_pass(&mut self, segments: &[Qwen38Segment], logits: Option<&mut [f32]>) -> Result<f64, PassStop> {
        self.run_pass_cut(segments, logits, None)
    }

    /// One pass of `segments` (`host.rs` `run_pass`), after the checks that refuse it: the segments, the poison, the
    /// knobs and the PLE rows. `logits` takes one row of `VOCAB` values for each segment. `cuts` is the plan of a pass
    /// with a tap or a control item.
    fn run_pass_cut(
        &mut self,
        segments: &[Qwen38Segment],
        logits: Option<&mut [f32]>,
        cuts: Option<&mut Cuts>,
    ) -> Result<f64, PassStop> {
        if let Some(problem) = host::segments_problem(segments) {
            return Err(PassStop::Refused(problem));
        }
        for (index, segment) in segments.iter().enumerate() {
            if unsafe { (self.binding.ctx_poisoned)(segment.context) } {
                return Err(PassStop::Refused(format!(
                    "segment {index}: the context is poisoned; free it"
                )));
            }
        }
        if let Some(logits) = &logits {
            assert_eq!(
                logits.len(),
                segments.len() * spec::VOCAB as usize,
                "the logits buffer must hold one row of VOCAB values for each segment"
            );
        }
        let knobs = self.knobs(logits.is_some()).map_err(PassStop::Refused)?;
        let launch = LaunchKnobs::from(env_text).map_err(PassStop::Refused)?;
        let ple = self.ple_rows(segments).map_err(PassStop::Refused)?;
        let (Some(host), Some(weights)) = (self.host.as_mut(), self.weights.as_ref()) else {
            return Err(PassStop::Refused("this engine has no pass host".to_owned()));
        };
        let input = PassIn {
            segments,
            ple: &ple,
            knobs,
            launch,
            cuts,
        };
        unsafe { host::run_pass(&self.binding, self.engine, host, weights, input, logits) }
    }

    /// The PLE rows of a pass, segment after segment: each segment's tokens after its context's PLE history.
    ///
    /// The engine stops the program on a read of a poisoned context, so a poisoned one is an error here.
    fn ple_rows(&self, segments: &[Qwen38Segment]) -> Result<Vec<f32>, String> {
        let (Some(model), Some(table)) = (&self.model, &self.ple) else {
            return Err("this engine has no PLE table".to_owned());
        };
        let tokens: usize = segments.iter().map(|segment| segment.t as usize).sum();
        let mut rows = vec![0f32; tokens * ple::EMBD];
        let mut at = 0;
        for (index, segment) in segments.iter().enumerate() {
            if segment.context.is_null() || segment.tokens.is_null() {
                return Err(format!("segment {index}: no context or no tokens pointer"));
            }
            if unsafe { (self.binding.ctx_poisoned)(segment.context) } {
                return Err(format!("segment {index}: the context is poisoned; free it"));
            }
            let mut history = [0i32; 2];
            unsafe { (self.binding.state_read)(self.engine, segment.context, 5, history.as_mut_ptr().cast()) };
            // SAFETY: the segment names `t` readable tokens (binding::segments).
            let tokens = unsafe { std::slice::from_raw_parts(segment.tokens, segment.t as usize) };
            let end = at + tokens.len() * ple::EMBD;
            table.rows(model, history, tokens, &mut rows[at..end]);
            at = end;
        }
        Ok(rows)
    }

    /// Borrow one context or refuse the call.
    fn context(&self, id: u64) -> Result<&Context, Call> {
        self.contexts
            .get(&id)
            .ok_or_else(|| Call::Refused(format!("context {id} is not in this engine")))
    }
}

/// A byte buffer that is aligned for the f32 stores of the engine. The engine reads and writes a part as f32 values in
/// host memory.
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

    /// The pointer for the engine to read.
    fn as_ptr(&self) -> *const c_void {
        self.words.as_ptr().cast()
    }

    /// The pointer for the engine to write.
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
        Aligned, Call, Context, MEMORY_FLOOR, ThreadState, Word, fast_qmm_from, memory_problem, open_contexts,
        open_memory_problem, parse_shards, probe_shapes, remake_context, rounds_to_finite, taken_bytes, tiles_from,
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
            model: None,
            ple: None,
            _arena: None,
            weights: None,
            host: None,
            contexts,
            next_id: 2,
            fast_qmm: true,
            tiles: false,
            reload: None,
            load_canary: None,
        }
    }

    /// A request for one segment of `lens` tokens at context 1, with `list` as its control list.
    fn steered_request(list: Vec<nagaya_abi::ControlItem>, taps: crate::control::TapSet) -> super::PassRequest {
        super::PassRequest {
            contexts: vec![1],
            lens: vec![2],
            tokens: vec![10, 11],
            want_logits: true,
            wants: vec![true],
            control: vec![list],
            taps,
        }
    }

    fn add_item(index: u32) -> nagaya_abi::ControlItem {
        nagaya_abi::ControlItem {
            index,
            op: nagaya_abi::ControlOp::Add,
            strength: 1.0,
            layers: None,
        }
    }

    fn vector_of_ones() -> Vec<u8> {
        (0..crate::control::LAYERS * crate::control::WIDTH)
            .flat_map(|_| 1.0f32.to_le_bytes())
            .collect()
    }

    #[test]
    fn a_plain_request_plans_no_cut_and_a_request_with_a_list_needs_a_written_entry() {
        let context = StubContext::new(false);
        let mut state = stub_state(&context);
        let none = crate::control::TapSet::default();
        // A plain pass makes no `Cuts`, with or without a table.
        assert!(state.plan_cuts(&steered_request(Vec::new(), none)).unwrap().is_none());
        // A list with no table is refused, and so is a write.
        let error = state
            .plan_cuts(&steered_request(vec![add_item(0)], none))
            .err()
            .expect("no table");
        assert!(error.contains("no control table"), "{error}");
        assert!(
            state
                .control_write(0, &vector_of_ones())
                .unwrap_err()
                .contains("no control table")
        );
        // With a table on the fake device, an entry that no write filled is refused by its segment and item.
        state.binding = crate::fake::binding();
        // SAFETY: the fake binding makes host memory.
        state.host = Some(unsafe { crate::host::Host::new(&state.binding) });
        state.make_control_table();
        let error = state
            .plan_cuts(&steered_request(vec![add_item(0)], none))
            .err()
            .expect("an empty entry");
        assert!(
            error.contains("segment 0: control item 0 names table entry 0"),
            "{error}"
        );
        assert!(error.contains("no control_write filled it"), "{error}");
        state.control_write(0, &vector_of_ones()).expect("a good write");
        let cuts = state
            .plan_cuts(&steered_request(vec![add_item(0)], none))
            .expect("entry 0 is filled");
        let cuts = cuts.expect("a list makes cuts");
        assert_eq!(cuts.layers().len(), 48, "an add on every layer cuts at every layer");
        assert!(cuts.fuse_hc_off());
        // A tap alone makes cuts too, and a tap keeps the fused combine.
        let tap = crate::control::TapSet {
            residual: true,
            fold: false,
        };
        let cuts = state
            .plan_cuts(&steered_request(Vec::new(), tap))
            .unwrap()
            .expect("a tap makes cuts");
        assert!(!cuts.fuse_hc_off());
        // The counts of the request must agree.
        let mut bad = steered_request(Vec::new(), tap);
        bad.wants = vec![true, false];
        assert!(
            state
                .plan_cuts(&bad)
                .err()
                .expect("counts differ")
                .contains("counts differ")
        );
    }

    #[test]
    fn a_worker_write_that_the_table_refuses_changes_nothing() {
        let context = StubContext::new(false);
        let mut state = stub_state(&context);
        state.binding = crate::fake::binding();
        // SAFETY: the fake binding makes host memory.
        state.host = Some(unsafe { crate::host::Host::new(&state.binding) });
        state.make_control_table();
        crate::fake::reset();
        let good = vector_of_ones();
        let mut not_finite = good.clone();
        not_finite[8..12].copy_from_slice(&f32::NEG_INFINITY.to_le_bytes());
        assert!(state.control_write(8, &good).unwrap_err().contains("entries"));
        assert!(
            state
                .control_write(0, &good[..100])
                .unwrap_err()
                .contains("send the whole vector")
        );
        assert!(state.control_write(0, &not_finite).unwrap_err().contains("value 2"));
        assert!(crate::fake::copies().is_empty(), "a refused write copies nothing");
        let none = crate::control::TapSet::default();
        assert!(
            state.plan_cuts(&steered_request(vec![add_item(0)], none)).is_err(),
            "no entry was filled"
        );
        state.control_write(0, &good).unwrap();
        assert_eq!(crate::fake::copies().len(), 1, "a good write is one copy");
    }

    #[test]
    fn an_unload_waits_until_every_context_is_free() {
        let context = StubContext::new(false);
        let mut state = stub_state(&context);
        let error = state.unload().unwrap_err();
        assert!(error.contains("1 contexts are open"), "{error}");
        state.contexts.clear();
        state.unload().unwrap();
        assert!(state.weights.is_none() && state.model.is_none() && state._arena.is_none());
    }

    #[test]
    fn a_load_of_a_stub_has_no_shards() {
        let context = StubContext::new(false);
        let mut state = stub_state(&context);
        assert_eq!(state.load_again().unwrap_err(), "this engine has no shards to load");
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
        assert_eq!(
            context.changes.load(Ordering::SeqCst),
            0,
            "a refused call reached the engine"
        );
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
        // A pass of 1 token asks for qmm_part2 (reference engine.hip:1187). The second pass asks for the rest, with
        // logits.
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
        assert!(error.contains("no context of 4096 tokens"), "{error}");
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
        // spec 4.7: the engine keeps 12 GiB of MemAvailable free (reference service/contexts.py, FLOOR_BYTES).
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

    const GIB: u64 = 1 << 30;

    /// The event of 2026-11-11: 90,237,583,360 bytes were available, and the shards needed 93,682,584,224 with the
    /// floor.
    #[test]
    fn an_open_that_the_free_memory_cannot_hold_is_a_problem_with_the_numbers() {
        let needed = 93_682_584_224 - MEMORY_FLOOR;
        let problem = open_memory_problem(needed, 90_237_583_360, 0).unwrap_err();
        for text in ["80797682336", "90237583360", "12884901888", "`unload`"] {
            assert!(problem.contains(text), "{text} is not in: {problem}");
        }
        assert!(
            memory_problem("the shards", 90_237_583_360, needed).is_err(),
            "the open refuses it too"
        );
    }

    #[test]
    fn the_bytes_of_an_unload_count_as_available() {
        let needed = 80 * GIB;
        assert!(open_memory_problem(needed, 90 * GIB, 0).is_err());
        assert!(open_memory_problem(needed, 90 * GIB, 3 * GIB).is_ok());
        // The limit is the floor: 92 GiB of 80 GiB and the 12 GiB floor holds.
        assert!(open_memory_problem(needed, 90 * GIB, 2 * GIB).is_ok());
        assert!(open_memory_problem(needed, 90 * GIB, 2 * GIB - 1).is_err());
    }

    #[test]
    fn the_check_of_the_open_and_the_check_before_it_agree() {
        for available in [0, 12 * GIB, 92 * GIB - 1, 92 * GIB, 200 * GIB] {
            assert_eq!(
                open_memory_problem(80 * GIB, available, 0).is_ok(),
                memory_problem("the shards", available, 80 * GIB).is_ok(),
                "available {available}"
            );
        }
    }
}
