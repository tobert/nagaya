//! The nagaya process: bootstrap config, threads, HTTP, gossip, and systemd.

use std::panic::catch_unwind;
use std::path::PathBuf;
use std::time::Duration;

use nagaya::bootstrap::Bootstrap;
use nagaya::exit::{ExitCode, panic_text};
use nagaya::lifecycle;
use nagaya::notify::Notifier;
use nagaya::signal::{SignalFlag, install_terminate};

/// The environment variable that names the bootstrap config (`docs/spec.md` 2.1).
const CONFIG_ENV: &str = "NAGAYA_CONFIG";

/// The bootstrap config file when the variable is unset (`docs/spec.md` 2.1).
const DEFAULT_CONFIG: &str = "nagaya.toml";

/// The SIGTERM flag of the process (spec 10.3).
static SIGNALS: SignalFlag = SignalFlag::new();

fn main() {
    let code = match catch_unwind(run) {
        Ok(code) => code,
        Err(payload) => {
            eprintln!("the process panicked: {}", panic_text(payload.as_ref()));
            ExitCode::Crash
        }
    };
    std::process::exit(code.code());
}

fn run() -> ExitCode {
    let path = config_path();
    let config = match Bootstrap::read(&path) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("cannot use the bootstrap config `{}`: {error}", path.display());
            return ExitCode::RefusedStart;
        }
    };
    let profile = match lifecycle::active_profile(&config.store_dir) {
        Ok(profile) => profile,
        Err(error) => {
            eprintln!(
                "cannot use the placement profile in `{}`: {error}",
                config.store_dir.display()
            );
            return ExitCode::RefusedStart;
        }
    };
    let (registry, inventory) = match engine_machine() {
        Some(machine) => machine,
        None => {
            eprintln!("this build has no engine; build with the feature `fake` for a test engine");
            return ExitCode::RefusedStart;
        }
    };
    if let Err(error) = install_terminate(&SIGNALS) {
        eprintln!("cannot install the SIGTERM handler: {error}");
        return ExitCode::Crash;
    }
    let notifier = match Notifier::from_env() {
        Ok(notifier) => notifier,
        Err(error) => {
            eprintln!("cannot use the systemd notify socket: {error}");
            None
        }
    };
    let mut ready = match lifecycle::start(
        profile,
        registry,
        inventory,
        notifier.as_ref(),
        lifecycle::DEFAULT_OPEN_TIMEOUT,
    ) {
        Ok(ready) => ready,
        Err(error) => {
            eprintln!("{error}");
            return error.exit_code();
        }
    };
    eprintln!("ready with {} engines", ready.opened.len());
    let code = lifecycle::wait(&ready.process, &SIGNALS, notifier.as_ref(), lifecycle::WAIT_POLL);
    if code != ExitCode::Clean {
        // Spec 8.5: an unexpected failure stops the process. Ask the workers to stop, then do not join one.
        ready.process.abort();
        std::mem::forget(ready.process);
        return code;
    }
    let budget = Duration::from_secs(config.park_budget_secs);
    let report = ready.process.stop(budget, &SIGNALS);
    eprintln!(
        "stopped with {} states parked in {} bytes",
        report.parked.states, report.parked.bytes
    );
    report.exit_code()
}

/// The bootstrap config path (spec 2.1 note).
fn config_path() -> PathBuf {
    std::env::var_os(CONFIG_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG))
}

/// The engine factories and the inventory of this machine. Milestone 14 adds the mk engine.
#[cfg(feature = "fake")]
fn engine_machine() -> Option<(nagaya::admission::Registry, nagaya::inventory::Inventory)> {
    use nagaya::admission::Registry;
    use nagaya::inventory::{Device, Inventory};

    let device = Device {
        id: 0,
        name: "fake-0".to_owned(),
        memory_bytes: 1 << 30,
        unified: false,
    };
    let inventory = Inventory {
        devices: vec![device],
        host_bytes: 1 << 30,
    };
    Some((Registry::with_fake(), inventory))
}

/// The engine factories and the inventory of this machine. A build without an engine refuses to start.
#[cfg(not(feature = "fake"))]
fn engine_machine() -> Option<(nagaya::admission::Registry, nagaya::inventory::Inventory)> {
    None
}
