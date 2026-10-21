//! The process crate: bootstrap config, inventory, placement, admission, threads and lifecycle.

pub mod admission;
pub mod affinity;
pub mod bootstrap;
pub mod cpu_sets;
pub mod evidence;
pub mod exit;
pub mod gossip;
pub mod inventory;
pub mod lifecycle;
pub mod node;
pub mod notify;
pub mod park;
pub mod placement;
pub mod probe;
pub mod process;
pub mod profile;
pub mod signal;
pub mod swap;
pub mod telemetry;
