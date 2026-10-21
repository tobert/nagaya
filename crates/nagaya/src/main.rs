//! The nagaya process: bootstrap config, threads, HTTP, gossip, and systemd.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use nagaya::bootstrap::Bootstrap;
use nagaya::cpu_sets::{self, CpuSets};
use nagaya::exit::{ExitCode, after_stop, leave, panic_text};
use nagaya::gossip::{Gossip, GossipConfig, PublishError};
use nagaya::lifecycle;
use nagaya::node::Node;
use nagaya::notify::{EngineState, Notifier};
use nagaya::probe::FileProbe;
use nagaya::signal::{SignalFlag, install_terminate};
use nagaya::swap::{Outcome, Swap};
use nagaya::telemetry::{host_name, resource};
use nagaya::{http, serving};
use nagaya_telemetry::{Config, Telemetry};

/// The environment variable that names the bootstrap config (`docs/spec.md` 2.1).
const CONFIG_ENV: &str = "NAGAYA_CONFIG";

/// The bootstrap config file when the variable is unset (`docs/spec.md` 2.1).
const DEFAULT_CONFIG: &str = "nagaya.toml";

/// How long a stop waits for the open HTTP requests to end, after the engines stopped.
const HTTP_STOP: Duration = Duration::from_secs(2);

/// The SIGTERM flag of the process (spec 10.3).
static SIGNALS: SignalFlag = SignalFlag::new();

fn main() {
    // The inventory is known before the config, so that the resource of the telemetry can carry its hardware.
    let machine = engine_machine();
    let telemetry = match start_telemetry(machine.as_ref().ok().map(|(_, inventory)| inventory)) {
        Ok(telemetry) => telemetry,
        Err(error) => {
            // The log is not up, so the message goes to stderr by hand.
            eprintln!("cannot start the telemetry: {error}");
            leave(ExitCode::RefusedStart);
        }
    };
    let span = tracing::info_span!(
        "nagaya.process",
        exit_code = tracing::field::Empty,
        otel.status_code = tracing::field::Empty
    );
    let code = span.in_scope(|| {
        let code = match catch_unwind(AssertUnwindSafe(|| run(machine))) {
            Ok(code) => code,
            Err(payload) => {
                tracing::error!("the process panicked: {}", panic_text(payload.as_ref()));
                ExitCode::Crash
            }
        };
        if code == ExitCode::Clean || code == ExitCode::PlannedRestart {
            tracing::info!(exit_code = code.code(), "the process exits");
        } else {
            tracing::error!(exit_code = code.code(), "the process exits");
        }
        code
    });
    span.record("exit_code", code.code());
    if code != ExitCode::Clean && code != ExitCode::PlannedRestart {
        span.record("otel.status_code", "ERROR");
    }
    drop(span);
    // `leave` runs no destructor, so the telemetry flushes here (spec 9.1).
    drop(telemetry);
    leave(code);
}

/// Start the stderr log and, when the environment asks for it, the OTLP export (spec 9.1).
fn start_telemetry(
    inventory: Option<&nagaya::inventory::Inventory>,
) -> Result<Telemetry, nagaya_telemetry::SetupError> {
    let host = host_name();
    let resource = resource(host.as_ref().ok().map(String::as_str), inventory);
    let telemetry = nagaya_telemetry::init(&Config::from_env(resource))?;
    if let Err(error) = host {
        tracing::warn!("cannot read the host name for the telemetry resource: {error}");
    }
    Ok(telemetry)
}

fn run(machine: Machine) -> ExitCode {
    let path = config_path();
    let config = match Bootstrap::read(&path) {
        Ok(config) => config,
        Err(error) => {
            tracing::error!("cannot use the bootstrap config `{}`: {error}", path.display());
            return ExitCode::RefusedStart;
        }
    };
    let profile = match lifecycle::active_profile(&config.store_dir) {
        Ok(profile) => profile,
        Err(error) => {
            tracing::error!(
                "cannot use the placement profile in `{}`: {error}",
                config.store_dir.display()
            );
            return ExitCode::RefusedStart;
        }
    };
    let (registry, inventory) = match machine {
        Ok(machine) => machine,
        Err(refusal) => {
            tracing::error!("{}", refusal.message);
            return refusal.code;
        }
    };
    if let Err(error) = install_terminate(&SIGNALS) {
        tracing::error!("cannot install the SIGTERM handler: {error}");
        return ExitCode::Crash;
    }
    let node_name = match config.node_name() {
        Ok(name) => name,
        Err(error) => {
            tracing::error!("cannot use the node name: {error}");
            return ExitCode::RefusedStart;
        }
    };
    // The CPU sets come from the online CPUs of the file probe. A machine that cannot split them refuses the start.
    let cpu_sets = match cpu_sets::for_machine(&FileProbe::new("/")) {
        Ok(sets) => sets,
        Err(error) => {
            tracing::error!("{error}");
            return error.exit_code();
        }
    };
    tracing::info!(
        "housekeeping CPUs {:?}; pool CPUs {} ({} threads)",
        cpu_sets.housekeeping,
        cpu_sets.pool.len(),
        cpu_sets.pool_size()
    );
    let mut gossip_config = GossipConfig::new(&node_name, config.gossip_bind);
    gossip_config.seeds = config.seed_peers.clone();
    gossip_config.housekeeping = Some(cpu_sets.housekeeping.clone());
    let gossip = match Gossip::start(gossip_config) {
        Ok(gossip) => gossip,
        Err(error) => {
            tracing::error!("{error}");
            return ExitCode::RefusedStart;
        }
    };
    tracing::info!("gossip listening on {}", gossip.local_addr());
    gossip.watch_peers(|line| tracing::info!("{line}"));
    let mut node = Node::new(gossip, config.http_bind);
    let code = serve(&config, &node_name, profile, registry, inventory, cpu_sets, &mut node);
    node.shutdown();
    code
}

/// Log a gossip write that failed. Gossip is advice for peers, so a failed write does not stop the process.
fn published(result: Result<(), PublishError>) {
    if let Err(error) = result {
        tracing::warn!("cannot publish to gossip: {error}");
    }
}

/// Open the engines, run until a stop, and stop (spec 10.1, 10.3).
fn serve(
    config: &Bootstrap,
    node_name: &str,
    profile: nagaya::profile::PlacementProfile,
    registry: nagaya::admission::Registry,
    inventory: nagaya::inventory::Inventory,
    cpu_sets: CpuSets,
    node: &mut Node,
) -> ExitCode {
    let notifier = match Notifier::from_env() {
        Ok(notifier) => notifier,
        Err(error) => {
            tracing::warn!("cannot use the systemd notify socket: {error}");
            None
        }
    };
    let mut swap = match Swap::open(&config.store_dir, node_name, profile.clone()) {
        Ok(swap) => swap,
        Err(error) => {
            tracing::error!("{error}");
            return ExitCode::RefusedStart;
        }
    };
    let names: Vec<String> = profile.engines.iter().map(|engine| engine.name.clone()).collect();
    let served_profile = profile.clone();
    let call_cpus = Some(cpu_sets.pool.clone());
    published(node.publish_machine(&inventory, &profile));
    published(node.engines_loading(&names));
    let options = lifecycle::Options::new(lifecycle::DEFAULT_OPEN_TIMEOUT).cpu_sets(cpu_sets);
    let mut ready = match lifecycle::start_with(
        profile,
        registry,
        inventory,
        &config.store_dir,
        notifier.as_ref(),
        options,
    ) {
        Ok(ready) => ready,
        // `lifecycle::start` logs the error with its span.
        Err(error) => return error.exit_code(),
    };
    published(node.engines_ready(&ready.opened));
    tracing::info!("ready with {} engines", ready.opened.len());
    // The council routes serve after READY (spec 11.1, 11.6). A failure here refuses the start (spec 10.4).
    let ports = ready.process.ports();
    let council = match serving::council(&served_profile, &ready.opened, &ports, &config.store_dir, call_cpus) {
        Ok(council) => Arc::new(council),
        Err(error) => {
            tracing::error!("{error}");
            ready.process.abort();
            std::mem::forget(ready.process);
            return ExitCode::RefusedStart;
        }
    };
    let server = match http::Server::start(&node.gossip().runtime(), config.http_bind, Arc::clone(&council)) {
        Ok(server) => server,
        Err(error) => {
            tracing::error!(
                "cannot bind the HTTP address {}: {error}; check `http_bind`",
                config.http_bind
            );
            ready.process.abort();
            std::mem::forget(ready.process);
            return ExitCode::RefusedStart;
        }
    };
    tracing::info!("http listening on {}", server.local_addr());
    let code = {
        // A desired profile from a peer arrives through gossip (spec 2.4 note, 3.3).
        let process = &ready.process;
        let council = &council;
        let mut tick = || {
            node.check_name();
            // An internal error of a route stops the process (spec 8.5, 11.5).
            if council.failure().is_some() {
                return Some(ExitCode::Crash);
            }
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
    // Spec 10.3 step 1: accept no new job. A request now gets a 503, and a job that the stop cancels ends its request.
    council.drain();
    let report = lifecycle::stop(&mut ready, notifier.as_ref(), budget, &SIGNALS);
    server.stop(HTTP_STOP);
    let (parked, unparked) = report.split_parked(&opened);
    published(node.engines_state(&parked, EngineState::Unloaded));
    published(node.engines_state(&unparked, EngineState::Fatal));
    for failure in &report.parked.failures {
        tracing::error!("the park failed: {failure}");
    }
    tracing::info!("{}", report.log_line());
    // A planned restart stops as in 10.3, then exits 4 (spec 3.3). A failed stop is still a crash. A route can fail
    // after the last wait tick, so the stop path checks the council again (spec 8.5, 11.7).
    let code = after_stop(code, report.exit_code(), council.failure().is_some());
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
            tracing::error!("cannot handle desired profile version {version}: {error}");
            Some(ExitCode::Crash)
        }
        Ok(Outcome::Restart) => {
            tracing::info!(
                "desired profile version {version} differs from the active profile; the process stops to restart"
            );
            node.clear_refusal();
            Some(ExitCode::PlannedRestart)
        }
        Ok(Outcome::Unchanged) => {
            tracing::info!("desired profile version {version} equals the active profile; nothing changed");
            node.clear_refusal();
            None
        }
        Ok(Outcome::Stopping) => {
            tracing::info!("desired profile version {version} waits: the process is stopping");
            None
        }
        Ok(Outcome::Refused(refusal)) => {
            tracing::warn!("refused desired profile version {version}: {}", refusal.reason);
            for entry in &refusal.shortfall {
                let tiers: Vec<String> = entry.tiers.iter().map(ToString::to_string).collect();
                tracing::warn!("  short by {} bytes: {}", entry.bytes, tiers.join(", "));
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

/// A refusal to start that is known before the telemetry starts. The log is not up then, so `run` logs the message.
struct Refusal {
    code: ExitCode,
    message: String,
}

/// The engine factories and the inventory of this machine, or the reason that the start is refused.
type Machine = Result<(nagaya::admission::Registry, nagaya::inventory::Inventory), Refusal>;

/// The engine factories and the inventory of this machine. The fake build has a fake engine and a made-up device.
#[cfg(feature = "fake")]
fn engine_machine() -> Machine {
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
    Ok((Registry::with_fake(), inventory))
}

/// The engine factories and the inventory of this machine. The inventory comes from the files of the machine (spec
/// 2.1). A probe error refuses the start (spec 10.4).
#[cfg(not(feature = "fake"))]
fn engine_machine() -> Machine {
    use nagaya::inventory::Probe;
    use nagaya::probe::FileProbe;

    let registry = engine_registry()?;
    let probe: &dyn Probe = &FileProbe::new("/");
    match probe.probe() {
        Ok(inventory) => Ok((registry, inventory)),
        Err(error) => Err(Refusal {
            code: error.exit_code(),
            message: format!("cannot read the inventory of this machine: {error}"),
        }),
    }
}

/// The engine factories of a build with the mk engine.
#[cfg(all(feature = "hip", not(feature = "fake")))]
fn engine_registry() -> Result<nagaya::admission::Registry, Refusal> {
    Ok(nagaya::admission::Registry::with_mk())
}

/// A build without an engine refuses to start.
#[cfg(all(not(feature = "hip"), not(feature = "fake")))]
fn engine_registry() -> Result<nagaya::admission::Registry, Refusal> {
    Err(Refusal {
        code: ExitCode::RefusedStart,
        message: "this build has no engine; build with the feature `hip` for the mk engine".to_owned(),
    })
}
