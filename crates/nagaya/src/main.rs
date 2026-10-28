//! The nagaya process: bootstrap config, threads, HTTP, gossip, and systemd.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use nagaya::arena::{ArenaSet, FdStore};
use nagaya::bootstrap::Bootstrap;
use nagaya::cpu_sets::{CpuSets, for_machine_with_devices_ignoring};
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
    // The stored arenas come first: the read changes the environment, so no thread may run yet (spec 10.2).
    // SAFETY: no thread runs yet, and nothing else owns the fds from 3 up.
    let arenas = unsafe { ArenaSet::from_env() };
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
        let code = match catch_unwind(AssertUnwindSafe(|| run(machine, arenas))) {
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

/// The fds that systemd passed back, or why the process cannot read them.
type Arenas = Result<ArenaSet, nagaya::arena::ArenaError>;

fn run(machine: Machine, arenas: Arenas) -> ExitCode {
    // A store that the process cannot read refuses the start, and systemd keeps each fd (spec 10.2, 10.4).
    let arenas = match arenas {
        Ok(arenas) => arenas,
        Err(error) => {
            tracing::error!("cannot read the fds that systemd passed: {error}");
            return ExitCode::RefusedStart;
        }
    };
    let fd_store = match FdStore::from_env() {
        Ok(store) => store,
        Err(error) => {
            tracing::error!("{error}");
            return ExitCode::RefusedStart;
        }
    };
    match &fd_store {
        Some(store) => tracing::info!(
            "the fd store keeps {} fds; systemd passed {} stored fds",
            store.ceiling(),
            arenas.len()
        ),
        None => tracing::info!("no fd store: the process keeps no arena through a restart"),
    }
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
    let (registry, mut inventory) = match machine {
        Ok(machine) => machine,
        Err(refusal) => {
            tracing::error!("{}", refusal.message);
            return refusal.code;
        }
    };
    // An address that the probe did not find stops the start, so a stale address cannot hide a device (spec 2.1).
    if let Err(error) = inventory.ignore(&config.ignore_devices) {
        tracing::error!("{error}");
        return error.exit_code();
    }
    for device in &inventory.devices {
        tracing::info!(
            "inventory: device {} `{}` at {} {}",
            device.id,
            device.name,
            device.pci_address,
            if device.ignored { "ignored" } else { "active" }
        );
    }
    // Admission checks this again. The early check stops the start before the CPU sets and the gossip (spec 2.1).
    if let Err(error) = nagaya::placement::check_ignored(&profile, &inventory) {
        tracing::error!("{error}");
        return ExitCode::RefusedStart;
    }
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
    // The CPU sets come from the CPU lists of the file probe (spec 3.2a, 3.2b). A machine that cannot split them refuses the start.
    let cpu_sets = match for_machine_with_devices_ignoring(&FileProbe::new("/"), &config.ignore_devices) {
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
    tracing::info!("device CPUs {:?}", cpu_sets.devices);
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
    let opening = Opening {
        cpu_sets,
        arenas,
        fd_store,
    };
    let code = serve(&config, &node_name, profile, registry, inventory, opening, &mut node);
    node.shutdown();
    code
}

/// Log a gossip write that failed. Gossip is advice for peers, so a failed write does not stop the process.
fn published(result: Result<(), PublishError>) {
    if let Err(error) = result {
        tracing::warn!("cannot publish to gossip: {error}");
    }
}

/// What the opens of the engines use: the CPU sets (spec 3.2a) and the stored arenas (spec 10.2).
struct Opening {
    cpu_sets: CpuSets,
    arenas: ArenaSet,
    fd_store: Option<FdStore>,
}

/// Open the engines, run until a stop, and stop (spec 10.1, 10.3).
fn serve(
    config: &Bootstrap,
    node_name: &str,
    profile: nagaya::profile::PlacementProfile,
    registry: nagaya::admission::Registry,
    inventory: nagaya::inventory::Inventory,
    opening: Opening,
    node: &mut Node,
) -> ExitCode {
    let Opening {
        cpu_sets,
        arenas,
        fd_store,
    } = opening;
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
    // The profile routes call the main loop, which owns the active profile (spec 3.3).
    let (mut changer, profile_calls) = nagaya::reprofile::Changer::new();
    let call_cpus = Some(cpu_sets.pool.clone());
    // Spec 10.1: the HTTP server listens before the engines open, and it answers 503 until the routes attach.
    let server = match http::Server::bind(&node.gossip().runtime(), config.http_bind) {
        Ok(server) => server,
        Err(error) => {
            tracing::error!(
                "cannot bind the HTTP address {}: {error}; check `http_bind`",
                config.http_bind
            );
            return ExitCode::RefusedStart;
        }
    };
    tracing::info!("http listening on {}", server.local_addr());
    published(node.publish_machine(&inventory, &profile));
    published(node.engines_loading(&names));
    let options = lifecycle::Options::new(lifecycle::DEFAULT_OPEN_TIMEOUT)
        .cpu_sets(cpu_sets)
        .arenas(arenas, fd_store)
        .hold_ready();
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
        Err(error) => {
            server.stop(HTTP_STOP);
            return error.exit_code();
        }
    };
    published(node.engines_ready(&ready.opened));
    // An on-demand engine stays unloaded until a load (spec 3.3a, 3.3b).
    let waiting: Vec<String> = names
        .iter()
        .filter(|name| ready.opened.iter().all(|engine| &engine.name != *name))
        .cloned()
        .collect();
    published(node.engines_state(&waiting, EngineState::Unloaded));
    // The council routes attach before READY (spec 10.1, 11.7). A failure here refuses the start (spec 10.4).
    let ports = ready.process.ports();
    let council = match serving::council(
        &serving::FamilyRegistry::compiled_in(),
        &served_profile,
        &ready.opened,
        &ports,
        &config.store_dir,
        &nagaya::qwen3tts::peers(&served_profile),
        call_cpus,
    )
    .and_then(|council| {
        Ok(council
            .with_speech(nagaya::qwen3tts::speech_services(
                &served_profile,
                &ports,
                &config.store_dir,
            )?)
            .with_loads(ports.clone())
            .with_profile(profile_calls.clone()))
    }) {
        Ok(council) => Arc::new(council),
        Err(error) => {
            tracing::error!("{error}");
            server.stop(HTTP_STOP);
            ready.process.abort();
            std::mem::forget(ready.process);
            return ExitCode::RefusedStart;
        }
    };
    server.attach(Arc::clone(&council));
    lifecycle::notify_ready(notifier.as_ref(), &ready.opened);
    tracing::info!("ready with {} engines", ready.opened.len());
    let code = {
        // A desired profile from a peer arrives through gossip (spec 2.4 note, 3.3).
        let process = &ready.process;
        let opened = &mut ready.opened;
        let council = &council;
        // The swaps in place that the wait heard of. The tick publishes them, because it holds the node (spec 3.3a).
        let swaps = std::cell::RefCell::new(Vec::new());
        let mut on_event = |event| swaps.borrow_mut().push(event);
        let mut tick = || {
            for event in swaps.borrow_mut().drain(..) {
                if let nagaya::process::Event::Swapped { unloaded, loaded, .. } = &event {
                    published(node.engines_state(unloaded, EngineState::Unloaded));
                    published(node.engines_ready(loaded));
                }
                lifecycle::apply_swap(opened, &event);
            }
            node.check_name();
            // An internal error of a route, or of a profile change, stops the process (spec 8.5, 11.5).
            if council.failure().is_some() || process.ports().failure().is_some() {
                return Some(ExitCode::Crash);
            }
            match changer.tick(&mut swap, process) {
                Ok(true) => {
                    tracing::info!("the profile of a PUT needs a restart; the process stops to restart");
                    return Some(ExitCode::PlannedRestart);
                }
                Ok(false) => {}
                Err(error) => {
                    tracing::error!("cannot persist the profile of a PUT: {error}");
                    return Some(ExitCode::Crash);
                }
            }
            desired_profile(&mut swap, node, process)
        };
        lifecycle::wait_with_events(
            process,
            &SIGNALS,
            notifier.as_ref(),
            lifecycle::WAIT_POLL,
            &mut tick,
            &mut on_event,
        )
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
        pci_address: "0000:00:00.0".to_owned(),
        memory_bytes: 1 << 30,
        unified: false,
        ignored: false,
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

/// The engine factories of this build. A build with no engine kind refuses to start (spec 10.4).
#[cfg(not(feature = "fake"))]
fn engine_registry() -> Result<nagaya::admission::Registry, Refusal> {
    let registry = nagaya::admission::Registry::compiled_in();
    if registry.is_empty() {
        return Err(Refusal {
            code: ExitCode::RefusedStart,
            message: "this build has no engine; build with the feature `hip` for the mk engine, or `xe` for the Gemma 4 engine".to_owned(),
        });
    }
    Ok(registry)
}
