//! Admission of a placement profile before any engine opens (spec 2.4).

use std::collections::BTreeMap;
use std::sync::Arc;

#[cfg(feature = "fake")]
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{ArenaFd, ArenaPart, Capability, CheckStrength, DeviceId, Engine, Identity, Limits, Tier};
use nagaya_ledger::{Admission, Budget, EnginePlan, Ledger, LedgerError, Profile};

use crate::arena::FdName;
use crate::inventory::{Device, Inventory};
use crate::placement::{DeviceError, check_devices, check_ignored};
use crate::profile::{EnginePlacement, PlacementProfile, Residency};

/// Why an engine cannot give its limits before it opens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanError {
    /// What the engine could not plan.
    pub message: String,
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for PlanError {}

/// Why an engine did not open (spec 10.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenErrorKind {
    /// The engine identity differs from the identity that the profile pins.
    Identity,
    /// A part size differs from the admitted plan.
    PartSize,
    /// Any other open failure.
    Other,
}

impl OpenErrorKind {
    /// True when the failure refuses the start (spec 10.4 row 3).
    pub fn refuses_start(self) -> bool {
        matches!(self, Self::Identity | Self::PartSize)
    }
}

/// Why an engine cannot open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenError {
    /// The kind of failure, for the exit code.
    pub kind: OpenErrorKind,
    /// What the engine could not open.
    pub message: String,
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for OpenError {}

/// A machine device of an engine, as the factory sees it (engine ABI "The factory gets its devices").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupDevice {
    /// The machine device id (spec 2.1).
    pub id: DeviceId,
    /// The device name from the probe.
    pub name: String,
    /// The PCI address, for example `0000:03:00.0`.
    pub pci_address: String,
    /// The device memory in bytes.
    pub memory_bytes: u64,
}

impl From<&Device> for GroupDevice {
    fn from(device: &Device) -> Self {
        Self {
            id: device.id,
            name: device.name.clone(),
            pci_address: device.pci_address.clone(),
            memory_bytes: device.memory_bytes,
        }
    }
}

/// The devices of an engine, in the order of the profile's `devices` list.
///
/// The position in the result is the group index (spec 2.4). A device that the inventory lacks is an error.
pub fn group_devices(engine: &EnginePlacement, inventory: &Inventory) -> Result<Vec<GroupDevice>, DeviceError> {
    engine
        .devices
        .iter()
        .map(|id| {
            inventory
                .device(*id)
                .map(GroupDevice::from)
                .ok_or_else(|| DeviceError::Unknown {
                    engine: engine.name.clone(),
                    device: *id,
                })
        })
        .collect()
}

/// An engine that opened on a stored arena, and the strength of the check of that arena (spec 10.2).
pub struct Attached {
    /// The engine.
    pub engine: Box<dyn Engine>,
    /// How much of the arena the check compared with the build.
    pub check: CheckStrength,
}

/// Why an engine did not open on a stored arena.
#[derive(Debug)]
pub enum AttachError {
    /// The arena does not match the build. Nothing is open, and the process can load the engine cold.
    Mismatch(String),
    /// The arena matched, and the open failed.
    Open(OpenError),
}

/// The factory of one engine kind (ABI "Rust shape", "Arena calls in v0").
pub trait Factory {
    /// Give the engine's limits for a config. `devices` are the machine devices of the engine.
    fn plan(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError>;

    /// Open the engine on the calling thread. The process calls this on the group thread.
    fn open(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError>;

    /// Give the identity and the limits that an open of this config would report, with no GPU and no load. The
    /// process serves a text engine on demand from them before it loads (spec 11.1). An open that reports other
    /// facts is an error. A kind that cannot tell them gives none, and its engine on demand is not a text model.
    fn describe(&self, _config: &toml::Table) -> Result<Option<(Identity, Limits)>, PlanError> {
        Ok(None)
    }

    /// Give the served model names of the engine `engine` with this config, with no open and no read of a file. The
    /// check of a profile refuses a name that a live peer serves (spec 2.3, "The check of a name"). A kind that cannot
    /// tell the names before the open gives none.
    fn served_names(&self, _engine: &str, _config: &toml::Table) -> Option<Vec<String>> {
        None
    }

    /// Give the parts of the arena that `arena_export` gives after an open. It reads no weights, as `plan` does. A
    /// kind with no arena gives no part.
    fn arena_parts(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Vec<ArenaPart>, PlanError> {
        Ok(Vec::new())
    }

    /// Check a stored arena against the build, and open the engine on it. The process calls this on the group
    /// thread. A mismatch opens nothing. A kind with no arena refuses each arena.
    fn attach(
        &self,
        _config: &toml::Table,
        _devices: &[GroupDevice],
        _arenas: Vec<ArenaFd>,
    ) -> Result<Attached, AttachError> {
        Err(AttachError::Mismatch("the engine kind keeps no arena".to_owned()))
    }
}

/// The factories by engine kind.
#[derive(Default)]
pub struct Registry {
    factories: BTreeMap<String, Arc<dyn Factory + Send + Sync>>,
}

impl Registry {
    /// Make an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a factory for an engine kind.
    pub fn register(&mut self, kind: &str, factory: Box<dyn Factory + Send + Sync>) {
        self.factories.insert(kind.to_owned(), Arc::from(factory));
    }

    /// The factory for an engine kind.
    pub fn factory(&self, kind: &str) -> Option<Arc<dyn Factory + Send + Sync>> {
        self.factories.get(kind).cloned()
    }

    /// True when the registry holds no factory.
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// Make a registry with the mk engine only. The tests of the mk engine use it.
    #[cfg(feature = "hip")]
    pub fn with_mk() -> Self {
        let mut registry = Self::new();
        registry.register(crate::mk::KIND, Box::new(crate::mk::MkFactory));
        registry
    }

    /// Make a registry with a factory for each engine kind that this build has: the mk engine with the feature
    /// `hip`, the Gemma 4, Qwen3-TTS, SDXL and EmbeddingGemma 2 engines with the feature `xe`, and the fake kinds with
    /// the feature `fake`.
    pub fn compiled_in() -> Self {
        #[allow(unused_mut)]
        let mut registry = Self::new();
        #[cfg(feature = "hip")]
        registry.register(crate::mk::KIND, Box::new(crate::mk::MkFactory));
        #[cfg(feature = "hip")]
        registry.register(
            crate::qwenimage21::KIND,
            Box::new(crate::qwenimage21::QwenImage21Factory),
        );
        #[cfg(feature = "xe")]
        registry.register(crate::gemma4::KIND, Box::new(crate::gemma4::Gemma4Factory));
        #[cfg(feature = "xe")]
        registry.register(crate::qwen3tts::KIND, Box::new(crate::qwen3tts::Qwen3TtsFactory));
        #[cfg(feature = "xe")]
        registry.register(crate::sdxl::KIND, Box::new(crate::sdxl::SdxlFactory));
        #[cfg(feature = "xe")]
        registry.register(
            crate::embeddinggemma2::KIND,
            Box::new(crate::embeddinggemma2::EmbeddingGemma2Factory),
        );
        #[cfg(feature = "fake")]
        registry.register_fakes();
        registry
    }

    /// Make a registry with the fake engine and the fake that panics on open. The tests use it.
    #[cfg(feature = "fake")]
    pub fn with_fake() -> Self {
        let mut registry = Self::new();
        registry.register_fakes();
        registry
    }

    #[cfg(feature = "fake")]
    fn register_fakes(&mut self) {
        self.register("fake", Box::new(FakeFactory));
        self.register("fake-panic", Box::new(PanicFactory));
        self.register(FAKE_TOY_KIND, Box::new(FakeToyFactory));
        self.register(FAKE_ARENA_KIND, Box::new(FakeArenaFactory));
    }
}

/// Why a profile cannot be admitted.
#[derive(Debug)]
pub enum AdmitError {
    /// The profile names an engine kind with no planner.
    UnknownKind {
        /// The engine name.
        engine: String,
        /// The engine kind.
        kind: String,
    },
    /// An engine cannot plan.
    Plan {
        /// The engine name.
        engine: String,
        /// The planner's message.
        message: String,
    },
    /// An engine declares a limit that no pass can use.
    Limits {
        /// The engine name.
        engine: String,
        /// The name of the field in `Limits`.
        field: &'static str,
        /// Why the value is invalid.
        why: &'static str,
    },
    /// The ledger refused the profile.
    Ledger(LedgerError),
    /// A profile device is missing or differs from the plan.
    Device(DeviceError),
    /// The profile needs more arena fds than the fd store keeps (spec 10.2).
    ArenaFds {
        /// The arena fds of the profile.
        needed: usize,
        /// The ceiling of the fd store, from `$FDSTORE`.
        ceiling: usize,
    },
    /// The process is stopping and takes no new work (spec 10.3).
    Stopping,
    /// A swap or a profile change runs, so the ledger takes no other change now (spec 3.3a).
    Busy(String),
    /// An entry breaks a rule of residency (spec 3.3a).
    Residency {
        /// The engine name.
        engine: String,
        /// Why the entry is invalid, and what to change.
        why: String,
    },
    /// An on-demand engine does not fit beside the resident engines of the other device groups (spec 3.3a).
    OnDemand {
        /// The on-demand engine.
        engine: String,
        /// The refusal of the ledger.
        error: LedgerError,
    },
}

impl std::fmt::Display for AdmitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdmitError::UnknownKind { engine, kind } => {
                write!(formatter, "engine `{engine}` has kind `{kind}`, which has no planner")
            }
            AdmitError::Plan { engine, message } => {
                write!(formatter, "engine `{engine}` cannot plan: {message}")
            }
            AdmitError::Limits { engine, field, why } => {
                write!(
                    formatter,
                    "engine `{engine}` declares `{field}` that is invalid: {why}; fix the engine's limits"
                )
            }
            AdmitError::Ledger(error) => {
                write!(formatter, "the profile cannot be admitted: {error}")
            }
            AdmitError::Device(error) => {
                write!(formatter, "the profile cannot be admitted: {error}")
            }
            AdmitError::ArenaFds { needed, ceiling } => write!(
                formatter,
                "the profile needs {needed} arena fds, and the fd store keeps {ceiling}; raise \
                 FileDescriptorStoreMax in the unit, or remove an engine"
            ),
            AdmitError::Stopping => write!(formatter, "the process is stopping"),
            AdmitError::Busy(why) => write!(formatter, "{why}"),
            AdmitError::Residency { engine, why } => write!(formatter, "engine `{engine}`: {why}"),
            AdmitError::OnDemand { engine, error } => write!(
                formatter,
                "the on-demand engine `{engine}` does not fit beside the resident engines of the other device \
                 groups: {error}"
            ),
        }
    }
}

impl std::error::Error for AdmitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AdmitError::Ledger(error) => Some(error),
            AdmitError::OnDemand { error, .. } => Some(error),
            AdmitError::Device(error) => Some(error),
            AdmitError::UnknownKind { .. }
            | AdmitError::Plan { .. }
            | AdmitError::Limits { .. }
            | AdmitError::ArenaFds { .. }
            | AdmitError::Residency { .. }
            | AdmitError::Stopping
            | AdmitError::Busy(_) => None,
        }
    }
}

/// One engine's admitted limits.
#[derive(Clone, Debug, PartialEq)]
pub struct AdmittedEngine {
    /// The engine name.
    pub name: String,
    /// The result of `plan(config)` that the ledger admitted.
    pub limits: Limits,
    /// The parts of the arena of the engine, from `arena_parts` (spec 10.2).
    pub arena_parts: Vec<ArenaPart>,
    /// The served model names of the engine from `served_names`, or `None` when its kind cannot tell them before the
    /// open (spec 2.3, "The check of a name").
    pub served: Option<Vec<String>>,
}

/// The result of admitting a profile.
#[derive(Clone, Debug, PartialEq)]
pub struct Admitted {
    /// The ledger's admission.
    pub admission: Admission,
    /// The planned limits, one per engine, in profile order.
    pub engines: Vec<AdmittedEngine>,
}

/// The fake engine's factory. The fake engine reads no weights.
#[cfg(feature = "fake")]
pub struct FakeFactory;

#[cfg(feature = "fake")]
impl Factory for FakeFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(FakeAlways::new().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(FakeAlways::new()))
    }
}

/// The kind of the fake engine that speaks the toy family (spec 11.3b). The process serves it as a council model.
#[cfg(feature = "fake")]
pub const FAKE_TOY_KIND: &str = "fake-toy";

/// The factory of the fake engine that speaks the toy family. Its logits cover the toy vocabulary with the speaking
/// bias of `nagaya-text-toy`. It has four slots of 4096 tokens: the context limit of a served model.
#[cfg(feature = "fake")]
pub struct FakeToyFactory;

#[cfg(feature = "fake")]
impl FakeToyFactory {
    /// The engine, with the model name of the config key `model` when the config has it.
    fn engine_of(config: &toml::Table) -> Result<FakeAlways, String> {
        match config.get("model") {
            None => Ok(Self::engine()),
            Some(toml::Value::String(model)) => Ok(Self::engine().with_model(model)),
            Some(other) => Err(format!(
                "the key `model` of a fake-toy engine is {other}; give a string"
            )),
        }
    }

    fn engine() -> FakeAlways {
        // A cells row of the fake has 8 bytes, so 32768 bytes hold 4096 tokens.
        let slots = vec![nagaya_abi::SlotLimit {
            bytes: 32768,
            count: 4,
            tokens: 4096,
        }];
        FakeAlways::with_slots(slots)
            .with_family("toy")
            .with_vocab(gogi_toy_vocab())
            .with_logit_bias(nagaya_text_toy::speaking_bias())
    }
}

/// The size of the toy vocabulary.
#[cfg(feature = "fake")]
fn gogi_toy_vocab() -> usize {
    nagaya_text_toy::speaking_bias().len()
}

#[cfg(feature = "fake")]
impl Factory for FakeToyFactory {
    fn plan(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        let engine = Self::engine_of(config).map_err(|message| PlanError { message })?;
        Ok(engine.limits())
    }

    fn open(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        let engine = Self::engine_of(config).map_err(|message| OpenError {
            kind: OpenErrorKind::Other,
            message,
        })?;
        // The key `open_ms` makes the open wait, so a test of the start can let gossip reach the node first.
        if let Some(ms) = config.get("open_ms").and_then(toml::Value::as_integer) {
            std::thread::sleep(std::time::Duration::from_millis(ms.max(0) as u64));
        }
        Ok(Box::new(engine))
    }

    fn served_names(&self, _engine: &str, config: &toml::Table) -> Option<Vec<String>> {
        Self::engine_of(config).ok().map(|engine| vec![engine.identity().model])
    }
}

/// The kind of the fake engine with an arena in a memfd (spec 10.2). The config key `arena` gives its bytes.
#[cfg(feature = "fake")]
pub const FAKE_ARENA_KIND: &str = "fake-arena";

/// The factory of the fake engine with an arena. The tests of the fd store use it.
#[cfg(feature = "fake")]
pub struct FakeArenaFactory;

#[cfg(feature = "fake")]
impl FakeArenaFactory {
    /// The bytes of the arena: the config key `arena`, or a fixed text.
    fn bytes(config: &toml::Table) -> Vec<u8> {
        match config.get("arena") {
            Some(value) => value
                .as_str()
                .expect("the key `arena` of the fake is a string")
                .as_bytes()
                .to_vec(),
            None => b"the arena of the fake".to_vec(),
        }
    }
}

#[cfg(feature = "fake")]
impl Factory for FakeArenaFactory {
    fn plan(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(FakeAlways::with_arena(&Self::bytes(config)).limits())
    }

    fn open(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(FakeAlways::with_arena(&Self::bytes(config))))
    }

    fn arena_parts(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Vec<ArenaPart>, PlanError> {
        Ok(FakeAlways::arena_parts())
    }

    fn attach(
        &self,
        config: &toml::Table,
        _devices: &[GroupDevice],
        arenas: Vec<ArenaFd>,
    ) -> Result<Attached, AttachError> {
        let (engine, check) = FakeAlways::attach_arena(arenas, &Self::bytes(config)).map_err(AttachError::Mismatch)?;
        Ok(Attached {
            engine: Box::new(engine),
            check,
        })
    }
}

/// The factory of a fake engine that panics on open. The tests use it for the panic path (spec 8.5).
#[cfg(feature = "fake")]
pub struct PanicFactory;

#[cfg(feature = "fake")]
impl Factory for PanicFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(FakeAlways::new().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        panic!("the fake engine cannot open")
    }
}

/// Admit a placement profile with the ledger, before any engine opens. A process with no fd store counts no arena
/// fd (spec 10.2).
pub fn admit(
    profile: &PlacementProfile,
    registry: &Registry,
    inventory: &Inventory,
    ledger: &mut Ledger,
) -> Result<Admitted, AdmitError> {
    admit_with_fd_ceiling(profile, registry, inventory, ledger, None)
}

/// The names of the stored fds of an engine's arena parts. A name that systemd refuses, or a part on a device
/// that the group lacks, is a plan error.
fn arena_names(
    engine: &EnginePlacement,
    parts: &[ArenaPart],
    devices: &[GroupDevice],
) -> Result<Vec<FdName>, AdmitError> {
    let fail = |message: String| AdmitError::Plan {
        engine: engine.name.clone(),
        message,
    };
    parts
        .iter()
        .map(|part| {
            let device = devices.get(part.device).ok_or_else(|| {
                fail(format!(
                    "the arena part `{}` is on device {} of the group, and the group has {} devices",
                    part.part,
                    part.device,
                    devices.len()
                ))
            })?;
            FdName::new(&engine.name, device.id, &part.part).map_err(fail)
        })
        .collect()
}

/// Admit a placement profile as [`admit`] does. When `fd_ceiling` is given, refuse a profile whose arena fds are
/// more than the ceiling (spec 10.2).
pub fn admit_with_fd_ceiling(
    profile: &PlacementProfile,
    registry: &Registry,
    inventory: &Inventory,
    ledger: &mut Ledger,
    fd_ceiling: Option<usize>,
) -> Result<Admitted, AdmitError> {
    admit_or_check(profile, registry, inventory, ledger, fd_ceiling, true)
}

/// Check a placement profile as [`admit_with_fd_ceiling`] does, with no change and no record (spec 3.3). A profile
/// change checks the whole new profile before it changes anything.
pub fn check_with_fd_ceiling(
    profile: &PlacementProfile,
    registry: &Registry,
    inventory: &Inventory,
    ledger: &mut Ledger,
    fd_ceiling: Option<usize>,
) -> Result<Admitted, AdmitError> {
    admit_or_check(profile, registry, inventory, ledger, fd_ceiling, false)
}

fn admit_or_check(
    profile: &PlacementProfile,
    registry: &Registry,
    inventory: &Inventory,
    ledger: &mut Ledger,
    fd_ceiling: Option<usize>,
    commit: bool,
) -> Result<Admitted, AdmitError> {
    let plans = plan_all(profile, registry, inventory, fd_ceiling)?;
    admit_planned(profile, &plans, ledger, commit)
}

/// Admit or check a profile from its plans, as [`admit_with_fd_ceiling`] does. No plan runs here.
pub fn admit_planned(
    profile: &PlacementProfile,
    plans: &Planned,
    ledger: &mut Ledger,
    commit: bool,
) -> Result<Admitted, AdmitError> {
    let Planned {
        budgets,
        engines,
        planned,
    } = plans;
    check_residency(profile, planned)?;
    // The resident engines take memory at start. Each on-demand engine must fit beside the resident engines of the
    // other device groups, because a swap unloads only engines of its own group (spec 3.3a). The checks change
    // nothing, so a refusal changes nothing.
    let resident = |index: usize| profile.engines[index].residency == Residency::Resident;
    let residents = charged(budgets, engines, resident);
    ledger.check(&residents).map_err(AdmitError::Ledger)?;
    let groups = crate::placement::device_groups(profile);
    for (index, placement) in profile.engines.iter().enumerate() {
        if resident(index) {
            continue;
        }
        let own = groups
            .iter()
            .find(|group| group.engines.contains(&index))
            .expect("each engine is in a device group");
        let beside = charged(budgets, engines, |other| {
            other == index || (resident(other) && !own.engines.contains(&other))
        });
        ledger.check(&beside).map_err(|error| AdmitError::OnDemand {
            engine: placement.name.clone(),
            error,
        })?;
    }
    let admission = if commit {
        ledger.admit(&residents)
    } else {
        ledger.check(&residents)
    }
    .map_err(AdmitError::Ledger)?;
    Ok(Admitted {
        admission,
        engines: planned.clone(),
    })
}

/// The ledger's profile of the engines that `loaded` names (spec 3.3a). Each other engine stays with its context terms
/// only and a share of 0: it takes no memory, and a request can charge its context before it waits for the load.
fn charged(budgets: &[Budget], engines: &[EnginePlan], loaded: impl Fn(usize) -> bool) -> Profile {
    Profile {
        budgets: budgets.to_vec(),
        engines: engines
            .iter()
            .enumerate()
            .map(|(index, engine)| {
                let mut plan = engine.clone();
                if !loaded(index) {
                    plan.limits
                        .memory
                        .mandatory
                        .retain(|term| term.lifetime == nagaya_abi::Lifetime::Context);
                    plan.limits.memory.elastic.clear();
                    plan.share = 0;
                }
                plan
            })
            .collect(),
    }
}

/// The plans of every engine of a profile, before the ledger sees them. The ledger keeps them for its swaps.
#[derive(Clone, Debug)]
pub struct Planned {
    budgets: Vec<Budget>,
    engines: Vec<EnginePlan>,
    planned: Vec<AdmittedEngine>,
}

impl Planned {
    /// The names of the engines, in profile order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.planned.iter().map(|engine| engine.name.as_str())
    }
}

/// Plan every engine of a profile, in profile order, and check its devices and its arena fds. No ledger runs here.
pub fn plan_all(
    profile: &PlacementProfile,
    registry: &Registry,
    inventory: &Inventory,
    fd_ceiling: Option<usize>,
) -> Result<Planned, AdmitError> {
    // An ignored device gets no work (spec 2.1), so refuse before any plan runs.
    check_ignored(profile, inventory).map_err(AdmitError::Device)?;
    let budgets: Vec<Budget> = profile
        .budgets
        .iter()
        .map(|budget| Budget {
            tiers: budget.tiers.iter().map(|tier| Tier::from(*tier)).collect(),
            bytes: budget.bytes,
        })
        .collect();
    let mut engines = Vec::with_capacity(profile.engines.len());
    let mut planned = Vec::with_capacity(profile.engines.len());
    for engine in &profile.engines {
        let factory = registry.factory(&engine.kind).ok_or_else(|| AdmitError::UnknownKind {
            engine: engine.name.clone(),
            kind: engine.kind.clone(),
        })?;
        let devices = group_devices(engine, inventory).map_err(AdmitError::Device)?;
        let limits = factory
            .plan(&engine.config, &devices)
            .map_err(|error| AdmitError::Plan {
                engine: engine.name.clone(),
                message: error.to_string(),
            })?;
        // The scheduler asserts this value, so a zero is refused here, before any thread starts. A speech, image or
        // embedding engine takes no pass (engine-abi "Speech", "The image capability", "Embedding"), so it declares no
        // segment.
        let passes = ![Capability::Speech, Capability::Image, Capability::Embedding]
            .iter()
            .any(|capability| limits.capabilities.contains(capability));
        if limits.max_segments == 0 && passes {
            return Err(AdmitError::Limits {
                engine: engine.name.clone(),
                field: "max_segments",
                why: "no pass can run with zero segments",
            });
        }
        let arena_parts = factory
            .arena_parts(&engine.config, &devices)
            .map_err(|error| AdmitError::Plan {
                engine: engine.name.clone(),
                message: error.to_string(),
            })?;
        arena_names(engine, &arena_parts, &devices)?;
        planned.push(AdmittedEngine {
            name: engine.name.clone(),
            limits: limits.clone(),
            arena_parts,
            served: factory.served_names(&engine.name, &engine.config),
        });
        engines.push(EnginePlan {
            name: engine.name.clone(),
            limits,
            devices: engine.devices.clone(),
            share: engine.share,
        });
    }
    check_devices(profile, &planned, inventory).map_err(AdmitError::Device)?;
    let needed: usize = planned.iter().map(|engine| engine.arena_parts.len()).sum();
    if let Some(ceiling) = fd_ceiling
        && needed > ceiling
    {
        return Err(AdmitError::ArenaFds { needed, ceiling });
    }
    Ok(Planned {
        budgets,
        engines,
        planned,
    })
}

/// Admit the engines of `profile` that `loaded` names, as a swap does (spec 3.3a). With `commit` false, the ledger
/// only checks, and nothing changes.
///
/// The residency of an entry does not count here: each named engine is charged, and each other engine is not.
pub fn admit_loaded(
    profile: &PlacementProfile,
    registry: &Registry,
    inventory: &Inventory,
    ledger: &mut Ledger,
    loaded: &std::collections::BTreeSet<String>,
    commit: bool,
) -> Result<Admitted, AdmitError> {
    let plans = plan_all(profile, registry, inventory, None)?;
    admit_loaded_planned(&plans, ledger, loaded, commit)
}

/// Admit the engines that `loaded` names from the plans of their profile, as [`admit_loaded`] does. No plan runs here,
/// so a swap does not wait for the plans of every engine of the profile.
pub fn admit_loaded_planned(
    plans: &Planned,
    ledger: &mut Ledger,
    loaded: &std::collections::BTreeSet<String>,
    commit: bool,
) -> Result<Admitted, AdmitError> {
    let Planned {
        budgets,
        engines,
        planned,
    } = plans;
    let charged = charged(budgets, engines, |index| loaded.contains(&engines[index].name));
    let admission = if commit {
        ledger.admit(&charged)
    } else {
        ledger.check(&charged)
    }
    .map_err(AdmitError::Ledger)?;
    Ok(Admitted {
        admission,
        engines: planned.clone(),
    })
}

/// The rules of residency (spec 3.3a): a pin only on a resident engine; no text model on demand; and `close` on each
/// engine of the device group of an on-demand engine.
fn check_residency(profile: &PlacementProfile, planned: &[AdmittedEngine]) -> Result<(), AdmitError> {
    let fail = |engine: &str, why: String| AdmitError::Residency {
        engine: engine.to_owned(),
        why,
    };
    let groups = crate::placement::device_groups(profile);
    // A group holds one engine with stages at a time, and its stages cover the group's devices (spec 3.2). Stage `s`
    // runs on group index `s`, which is machine device `devices[s]` of the profile (spec 2.4). Thus the list can have
    // any order, and it must hold each device of the group once. The serve loop cannot run another shape.
    for group in &groups {
        let mut devices = group.devices.clone();
        devices.sort_unstable();
        let mut resident: Option<&str> = None;
        for &member in &group.engines {
            let limits = &planned[member].limits;
            if !limits.capabilities.contains(&Capability::Stages) {
                continue;
            }
            let placement = &profile.engines[member];
            let name = placement.name.as_str();
            let stages: Vec<u32> = limits.stages.iter().map(|stage| stage.device).collect();
            let order: Vec<u32> = (0..devices.len() as u32).collect();
            let mut own = placement.devices.clone();
            own.sort_unstable();
            if own != devices || stages != order {
                return Err(fail(
                    name,
                    format!(
                        "an engine with stages spans its whole device group, and stage s runs on group index s; it \
                         names devices {:?} with stages on group indexes {stages:?}, and its group has devices \
                         {devices:?}; list each device of the group once, or move the other engines to other devices",
                        placement.devices
                    ),
                ));
            }
            if placement.residency != Residency::Resident {
                continue;
            }
            if let Some(first) = resident {
                return Err(fail(
                    name,
                    format!(
                        "a device group holds one engine with stages at a time, and `{first}` is resident in it too; \
                         make one of them on demand"
                    ),
                ));
            }
            resident = Some(name);
        }
    }
    for (index, placement) in profile.engines.iter().enumerate() {
        let name = placement.name.as_str();
        if placement.pinned && placement.residency != Residency::Resident {
            return Err(fail(
                name,
                "a pinned engine is resident, so it cannot be on demand; remove `pinned`, or remove \
                 `residency = \"on_demand\"`"
                    .to_owned(),
            ));
        }
        if placement.residency != Residency::OnDemand {
            continue;
        }
        // An image, speech or embedding engine builds its service from the profile, so its served model exists on
        // demand (spec 3.3a, "Council models on demand"). A text service attaches only to an engine that opens at the
        // start.
        let capabilities = &planned[index].limits.capabilities;
        let from_profile = [Capability::Image, Capability::Speech, Capability::Embedding]
            .iter()
            .any(|capability| capabilities.contains(capability));
        if !placement.family.is_empty() && !from_profile {
            return Err(fail(
                name,
                "a text model on demand is not built, so an on-demand engine of text has no `family` table; remove \
                 the table, or make the engine resident"
                    .to_owned(),
            ));
        }
        let own = groups
            .iter()
            .find(|group| group.engines.contains(&index))
            .expect("each engine is in a device group");
        for &member in &own.engines {
            if !planned[member].limits.capabilities.contains(&Capability::Close) {
                if member == index {
                    return Err(fail(
                        name,
                        "an on-demand engine needs `close`, because a swap closes it; use an engine that declares \
                         `close`, or make the engine resident"
                            .to_owned(),
                    ));
                }
                let other = &profile.engines[member].name;
                return Err(fail(
                    other,
                    format!(
                        "the engine has no `close`, and it shares a device group with the on-demand engine \
                         `{name}`; a swap must close it, so give it `close` or move `{name}` to other devices"
                    ),
                ));
            }
        }
    }
    Ok(())
}
