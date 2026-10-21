//! Admission of a placement profile before any engine opens (spec 2.4).

use std::collections::BTreeMap;
use std::sync::Arc;

#[cfg(feature = "fake")]
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{DeviceId, Engine, Limits, Tier};
use nagaya_ledger::{Admission, Budget, EnginePlan, Ledger, LedgerError, Profile};

use crate::inventory::{Device, Inventory};
use crate::placement::{DeviceError, check_devices};
use crate::profile::{EnginePlacement, PlacementProfile};

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
    /// The library differs from the pin in the profile.
    Pin,
    /// A part size differs from the admitted plan.
    PartSize,
    /// Any other open failure.
    Other,
}

impl OpenErrorKind {
    /// True when the failure refuses the start (spec 10.4 row 3).
    pub fn refuses_start(self) -> bool {
        matches!(self, Self::Identity | Self::Pin | Self::PartSize)
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

/// The factory of one engine kind (ABI "Rust shape").
pub trait Factory {
    /// Give the engine's limits for a config. `devices` are the machine devices of the engine.
    fn plan(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError>;

    /// Open the engine on the calling thread. The process calls this on the group thread.
    fn open(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError>;
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
    /// `hip`, the Gemma 4 engine with the feature `xe`, and the fake kinds with the feature `fake`.
    pub fn compiled_in() -> Self {
        #[allow(unused_mut)]
        let mut registry = Self::new();
        #[cfg(feature = "hip")]
        registry.register(crate::mk::KIND, Box::new(crate::mk::MkFactory));
        #[cfg(feature = "xe")]
        registry.register(crate::gemma4::KIND, Box::new(crate::gemma4::Gemma4Factory));
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
    /// The process is stopping and takes no new work (spec 10.3).
    Stopping,
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
            AdmitError::Stopping => write!(formatter, "the process is stopping"),
        }
    }
}

impl std::error::Error for AdmitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AdmitError::Ledger(error) => Some(error),
            AdmitError::Device(error) => Some(error),
            AdmitError::UnknownKind { .. }
            | AdmitError::Plan { .. }
            | AdmitError::Limits { .. }
            | AdmitError::Stopping => None,
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
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(Self::engine().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(Self::engine()))
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

/// Admit a placement profile with the ledger, before any engine opens.
pub fn admit(
    profile: &PlacementProfile,
    registry: &Registry,
    inventory: &Inventory,
    ledger: &mut Ledger,
) -> Result<Admitted, AdmitError> {
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
        // The scheduler asserts this value, so a zero is refused here, before any thread starts.
        if limits.max_segments == 0 {
            return Err(AdmitError::Limits {
                engine: engine.name.clone(),
                field: "max_segments",
                why: "no pass can run with zero segments",
            });
        }
        planned.push(AdmittedEngine {
            name: engine.name.clone(),
            limits: limits.clone(),
        });
        engines.push(EnginePlan {
            name: engine.name.clone(),
            limits,
            devices: engine.devices.clone(),
            share: engine.share,
        });
    }
    check_devices(profile, &planned, inventory).map_err(AdmitError::Device)?;
    let admission = ledger
        .admit(&Profile { budgets, engines })
        .map_err(AdmitError::Ledger)?;
    Ok(Admitted {
        admission,
        engines: planned,
    })
}
