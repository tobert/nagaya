//! Telemetry of the nagaya (spec 9).
//!
//! The crate has two layers.
//!
//! - The instruments (spec 9.4) need only the OpenTelemetry API. A library crate such as the ledger records on them.
//!   Without a provider they record nothing.
//! - The setup (feature `export`) starts the stderr log (spec 9.1), the OTLP exporters and the sampler (spec 9.3).
//!   Only the process and the tests turn it on.
//!
//! Spec 9.5: every instrument has a call site, and a test that runs that call site. Add an instrument to
//! [`Instruments`] in the same commit as its call site and its test.

mod instruments;

pub use instruments::{
    AlreadyInstalled, EngineState, EvictReason, Instruments, LookupOutcome, METER_NAME, QuantumLabels, RefusalGate,
    RestoreResult, ScopeGuard, TierLabel, install, instruments, scope,
};

#[cfg(feature = "export")]
mod context;
#[cfg(feature = "export")]
mod format;
#[cfg(feature = "export")]
mod sampler;
#[cfg(feature = "export")]
mod setup;
#[cfg(feature = "testing")]
pub mod testing;

#[cfg(feature = "export")]
pub use context::{INVALID_ATTRIBUTE, ParentStatus, adopt_parent};
#[cfg(feature = "export")]
pub use format::StderrFormat;
#[cfg(feature = "export")]
pub use sampler::{DEFAULT_RATE, FULL_RATE_PREFIXES, NagayaSampler, sampling_rate};
#[cfg(feature = "export")]
pub use setup::{Config, ResourceValue, SetupError, Telemetry, init};
