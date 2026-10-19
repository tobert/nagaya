//! The nagaya process: bootstrap config, threads, HTTP, gossip, and systemd.

use std::panic::catch_unwind;
use std::path::PathBuf;
use std::time::Duration;

use nagaya::bootstrap::Bootstrap;
use nagaya::exit::{ExitCode, panic_text};
use nagaya::gossip::{Gossip, GossipConfig, PublishError};
use nagaya::lifecycle;
use nagaya::node::Node;
use nagaya::notify::{EngineState, Notifier};
use nagaya::signal::{SignalFlag, install_terminate};
use nagaya::swap::{Outcome, Swap};

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
    let node_name = match config.node_name() {
        Ok(name) => name,
        Err(error) => {
            eprintln!("cannot use the node name: {error}");
            return ExitCode::RefusedStart;
        }
    };
    let mut gossip_config = GossipConfig::new(&node_name, config.gossip_bind);
    gossip_config.seeds = config.seed_peers.clone();
    let gossip = match Gossip::start(gossip_config) {
        Ok(gossip) => gossip,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::RefusedStart;
        }
    };
    eprintln!("gossip listening on {}", gossip.local_addr());
    gossip.watch_peers(|line| eprintln!("{line}"));
    let mut node = Node::new(gossip, config.http_bind);
    let code = serve(&config, &node_name, profile, registry, inventory, &mut node);
    node.shutdown();
    code
}

/// Log a gossip write that failed. Gossip is advice for peers, so a failed write does not stop the process.
fn published(result: Result<(), PublishError>) {
    if let Err(error) = result {
        eprintln!("cannot publish to gossip: {error}");
    }
}

/// Open the engines, run until a stop, and stop (spec 10.1, 10.3).
fn serve(
    config: &Bootstrap,
    node_name: &str,
    profile: nagaya::profile::PlacementProfile,
    registry: nagaya::admission::Registry,
    inventory: nagaya::inventory::Inventory,
    node: &mut Node,
) -> ExitCode {
    let notifier = match Notifier::from_env() {
        Ok(notifier) => notifier,
        Err(error) => {
            eprintln!("cannot use the systemd notify socket: {error}");
            None
        }
    };
    let mut swap = match Swap::open(&config.store_dir, node_name, profile.clone()) {
        Ok(swap) => swap,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::RefusedStart;
        }
    };
    let names: Vec<String> = profile.engines.iter().map(|engine| engine.name.clone()).collect();
    published(node.publish_machine(&inventory, &profile));
    published(node.engines_loading(&names));
    let mut ready = match lifecycle::start(
        profile,
        registry,
        inventory,
        &config.store_dir,
        notifier.as_ref(),
        lifecycle::DEFAULT_OPEN_TIMEOUT,
    ) {
        Ok(ready) => ready,
        Err(error) => {
            eprintln!("{error}");
            return error.exit_code();
        }
    };
    published(node.engines_ready(&ready.opened));
    eprintln!("ready with {} engines", ready.opened.len());
    let code = {
        // A desired profile from a peer arrives through gossip (spec 2.4 note, 3.3).
        let process = &ready.process;
        let mut tick = || {
            node.check_name();
            desired_profile(&mut swap, node, process)
        };
        lifecycle::wait_with(process, &SIGNALS, notifier.as_ref(), lifecycle::WAIT_POLL, &mut tick)
    };
    if code != ExitCode::Clean && code != ExitCode::PlannedRestart {
        // Spec 8.5: an unexpected failure stops the process. Ask the workers to stop, then do not join one.
        ready.process.abort();
        std::mem::forget(ready.process);
        return code;
    }
    let budget = Duration::from_secs(config.park_budget_secs);
    let opened: Vec<String> = ready.opened.iter().map(|engine| engine.name.clone()).collect();
    published(node.engines_state(&opened, EngineState::Draining));
    let report = lifecycle::stop(&mut ready, notifier.as_ref(), budget, &SIGNALS);
    published(node.engines_state(&opened, EngineState::Unloaded));
    for failure in &report.parked.failures {
        eprintln!("the park failed: {failure}");
    }
    eprintln!("{}", report.log_line());
    let stop_code = report.exit_code();
    // A planned restart stops as in 10.3, then exits 4 (spec 3.3). A failed stop is still a crash.
    let code = if code == ExitCode::PlannedRestart && stop_code == ExitCode::Clean {
        ExitCode::PlannedRestart
    } else {
        stop_code
    };
    // Spec 10.3: exit after the park budget. Do not join a group that is still parking.
    std::mem::forget(ready.process);
    code
}

/// Handle the newest desired profile for this host, when a peer offers one. Gives a code when the process must stop.
fn desired_profile(swap: &mut Swap, node: &mut Node, process: &nagaya::process::Process) -> Option<ExitCode> {
    let candidate = swap.candidate(&node.gossip().view())?;
    let version = candidate.version;
    match swap.handle(candidate, process) {
        Err(error) => {
            eprintln!("cannot handle desired profile version {version}: {error}");
            Some(ExitCode::Crash)
        }
        Ok(Outcome::Restart) => {
            eprintln!(
                "desired profile version {version} differs from the active profile; the process stops to restart"
            );
            node.clear_refusal();
            Some(ExitCode::PlannedRestart)
        }
        Ok(Outcome::Unchanged) => {
            eprintln!("desired profile version {version} equals the active profile; nothing changed");
            node.clear_refusal();
            None
        }
        Ok(Outcome::Refused(refusal)) => {
            eprintln!("refused desired profile version {version}: {}", refusal.reason);
            for entry in &refusal.shortfall {
                let tiers: Vec<String> = entry.tiers.iter().map(ToString::to_string).collect();
                eprintln!("  short by {} bytes: {}", entry.bytes, tiers.join(", "));
            }
            published(node.publish_refusal(&refusal));
            None
        }
    }
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
