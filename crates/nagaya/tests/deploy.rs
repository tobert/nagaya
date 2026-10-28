//! The files in `deploy/` for gfx1151host and b70host (spec 2.1, 2.4, 10.1). The real loaders read them, so a change to
//! a schema breaks these tests before it breaks a cutover.
//!
//! The admission of the b70host profile plans from the Gemma 4 checkpoint. That test skips when `GEMMA4_MODEL_DIR`
//! is not set, and fails then with `NAGAYA_REQUIRE_MODELS=1` (AGENTS "Tests that need a model file").

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry, admit};
use nagaya::bootstrap::Bootstrap;
use nagaya::inventory::Probe;
use nagaya::probe::FileProbe;
use nagaya::profile::{PROFILE_FILE, PlacementProfile, Residency, TierSpec};
use nagaya_abi::{Engine, Limits, Tier};
use nagaya_ledger::Ledger;

/// The host safety floor of spec 4.7: 12 GiB of MemAvailable stay free.
const FLOOR: u64 = 12 << 30;

/// The repository root, two levels above the crate.
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn deploy(name: &str) -> PathBuf {
    repo().join("deploy").join(name)
}

fn bootstrap() -> Bootstrap {
    Bootstrap::read(&deploy("gfx1151host/nagaya.toml")).unwrap()
}

fn profile() -> PlacementProfile {
    PlacementProfile::read(&deploy("gfx1151host").join(PROFILE_FILE)).unwrap()
}

fn b70host_bootstrap() -> Bootstrap {
    Bootstrap::read(&deploy("b70host/nagaya.toml")).unwrap()
}

fn b70host_profile() -> PlacementProfile {
    PlacementProfile::read(&deploy("b70host").join(PROFILE_FILE)).unwrap()
}

/// The value of a `Key=value` line in the unit. Panics when the key is missing or appears twice.
fn unit_value(unit: &str, key: &str) -> String {
    let prefix = format!("{key}=");
    let values: Vec<&str> = unit.lines().filter_map(|line| line.strip_prefix(&prefix)).collect();
    assert_eq!(
        values.len(),
        1,
        "the unit needs one `{key}=` line; it has {}",
        values.len()
    );
    values[0].to_owned()
}

fn unit() -> String {
    std::fs::read_to_string(deploy("nagaya.service")).unwrap()
}

fn b70host_unit() -> String {
    std::fs::read_to_string(deploy("b70host/nagaya.service")).unwrap()
}

/// The values of the `Environment=` lines of a unit.
fn environment(unit: &str) -> Vec<&str> {
    unit.lines()
        .filter_map(|line| line.strip_prefix("Environment="))
        .collect()
}

#[test]
fn the_bootstrap_config_binds_mks_address() {
    // Amy, 2026-10-18: the nagaya takes mk's address, and kaijutsu's gate config names it.
    let config = bootstrap();
    assert_eq!(config.http_bind, "198.51.100.20:8090".parse::<SocketAddr>().unwrap());
    assert_eq!(config.gossip_bind.ip(), config.http_bind.ip());
    assert_ne!(config.gossip_bind.port(), config.http_bind.port());
    assert_eq!(config.seed_peers, Vec::new(), "gfx1151host is one machine; it has no seeds");
    assert!(config.store_dir.is_absolute(), "the loader does not expand `~`");
    assert!(config.store_dir.ends_with(".local/share/nagaya/store"));
}

#[test]
fn the_profile_names_the_mk_engine_and_the_store_budgets() {
    let profile = profile();
    assert_eq!(profile.engines.len(), 1);
    let engine = &profile.engines[0];
    assert_eq!(engine.name, "mk");
    assert_eq!(engine.kind, "mk-qwen38-gfx1151");
    assert_eq!(engine.devices, vec![0]);
    // The mk factory plans from this key (`nagaya::mk`). It refuses any other key.
    let keys: Vec<&String> = engine.config.keys().collect();
    assert_eq!(keys, vec!["weights_bytes"]);
    assert!(engine.config["weights_bytes"].as_integer().unwrap() > 0);
    let store = profile.store.expect("an engine needs a [store] table (spec 4.1a)");
    assert!(store.nvme > 0);
    assert!(
        store.host.is_some(),
        "the host budget of the store is explicit on gfx1151host"
    );
}

#[test]
fn the_pooled_budget_leaves_the_host_safety_floor_free() {
    // Spec 4.1: device(0) and host are one pool on gfx1151host. Spec 4.7: 12 GiB of MemAvailable stay free.
    let profile = profile();
    let inventory = FileProbe::new(repo().join("crates/nagaya/tests/fixtures/probe/gfx1151host"))
        .probe()
        .unwrap();
    let pool = [TierSpec::from(Tier::Device(0)), TierSpec::from(Tier::Host)];
    let budget = profile
        .budgets
        .iter()
        .find(|budget| budget.tiers == pool)
        .expect("gfx1151host has one budget for device(0) and host");
    assert!(
        budget.bytes + FLOOR <= inventory.host_bytes,
        "the pool is {} bytes; host memory is {} bytes and the floor is {FLOOR} bytes",
        budget.bytes,
        inventory.host_bytes
    );
    let store = profile.store.unwrap();
    assert!(
        store.host.unwrap() < budget.bytes,
        "the store's host budget fits inside the pool"
    );
}

/// Check the restart policy of spec 10.4 and the start limits in a unit.
fn check_restart_policy(unit: &str) {
    assert_eq!(unit_value(unit, "Type"), "notify");
    assert_eq!(unit_value(unit, "Restart"), "on-failure");
    assert_eq!(unit_value(unit, "RestartSec"), "30");
    assert_eq!(unit_value(unit, "RestartPreventExitStatus"), "3");
    assert_eq!(unit_value(unit, "RestartForceExitStatus"), "4");
    assert_eq!(unit_value(unit, "StartLimitIntervalSec"), "600");
    assert_eq!(unit_value(unit, "StartLimitBurst"), "3");
    assert_eq!(unit_value(unit, "TimeoutStartSec"), "300");
    assert_eq!(unit_value(unit, "ExecStart"), "%h/src/nagaya/target/release/nagaya");
}

#[test]
fn the_unit_is_a_notify_unit_with_the_restart_policy_of_spec_10_4() {
    check_restart_policy(&unit());
}

#[test]
fn the_unit_reads_the_config_in_deploy() {
    let unit = unit();
    let environment: Vec<&str> = unit
        .lines()
        .filter_map(|line| line.strip_prefix("Environment="))
        .collect();
    let config = environment
        .iter()
        .find_map(|value| value.strip_prefix("NAGAYA_CONFIG=%h/src/nagaya/"))
        .expect("the unit sets NAGAYA_CONFIG under the repository");
    assert_eq!(config, "deploy/gfx1151host/nagaya.toml");
    assert!(repo().join(config).is_file(), "the file that the unit names exists");
    assert!(environment.contains(&"OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317"));
    assert_eq!(unit_value(&unit, "ExecStart"), "%h/src/nagaya/target/release/nagaya");
}

#[test]
fn the_unit_names_three_shards_and_no_mk_library() {
    let unit = unit();
    let environment: Vec<&str> = unit
        .lines()
        .filter_map(|line| line.strip_prefix("Environment="))
        .collect();
    // The binary links mk's engine, so the unit names no library and no library pin (AGENTS.md milestone 15).
    assert!(
        environment
            .iter()
            .all(|value| !value.trim_start_matches('"').starts_with("MK_ENGINE_LIB")),
        "{environment:?}"
    );
    let shards = environment
        .iter()
        .find_map(|value| {
            value
                .strip_prefix("\"MK_SHARDS=")
                .and_then(|rest| rest.strip_suffix('"'))
        })
        .expect("the unit sets MK_SHARDS in quotes");
    let paths: Vec<&str> = shards.split(':').collect();
    assert_eq!(paths.len(), 3, "mk needs three shards, with : between them");
    for (index, path) in paths.iter().enumerate() {
        assert!(path.ends_with(&format!("-0000{}-of-00003.gguf", index + 1)), "{path}");
    }
}

#[test]
fn the_b70host_config_binds_mkviolets_address_and_ignores_the_igpu() {
    // Amy, 2026-10-21: the nagaya replaces mkviolet and takes its address.
    let config = b70host_bootstrap();
    assert_eq!(config.http_bind, "198.51.100.10:8091".parse::<SocketAddr>().unwrap());
    assert_eq!(config.gossip_bind.ip(), config.http_bind.ip());
    assert_ne!(config.gossip_bind.port(), config.http_bind.port());
    assert_eq!(config.seed_peers, Vec::new());
    assert!(config.store_dir.is_absolute(), "the loader does not expand `~`");
    assert!(config.store_dir.ends_with(".local/share/nagaya/store"));
    // The Radeon 610M drives the display (spec 2.1, "Ignored devices").
    assert_eq!(config.ignore_devices, vec!["0000:80:00.0".to_owned()]);
}

#[test]
fn the_b70host_profile_names_the_gemma4_engine_on_both_cards() {
    let profile = b70host_profile();
    assert_eq!(
        profile.engines.len(),
        6,
        "gemma, then the speech engine and four image engines, on demand"
    );
    let engine = &profile.engines[0];
    assert_eq!(engine.residency, Residency::Resident);
    assert_eq!(engine.kind, nagaya::gemma4::KIND);
    assert_eq!(engine.devices, vec![0, 1], "stage 0 on device 0, stage 1 on device 1");
    // mkviolet serves one slot of 4096 positions and one of 65,536, with blocks of 512 (mkviolet `src/main.rs`).
    let slots: Vec<i64> = engine.config["slots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|slot| slot.as_integer().unwrap())
        .collect();
    assert_eq!(slots, vec![4096, 65_536]);
    assert_eq!(engine.config["max_block"].as_integer(), Some(512));
    assert!(
        !engine.config.contains_key("model_dir") && !engine.family.contains_key("model_dir"),
        "the checkpoint comes from GEMMA4_MODEL_DIR in the unit"
    );
    // Spec 4.1: one budget for each card, and one for host memory.
    let tiers: Vec<Vec<TierSpec>> = profile.budgets.iter().map(|budget| budget.tiers.clone()).collect();
    assert_eq!(
        tiers,
        vec![
            vec![TierSpec::from(Tier::Device(0))],
            vec![TierSpec::from(Tier::Device(1))],
            vec![TierSpec::from(Tier::Host)],
        ]
    );
    let inventory = FileProbe::new(repo().join("crates/nagaya/tests/fixtures/probe/b70host"))
        .probe()
        .unwrap();
    for (budget, id) in profile.budgets.iter().zip([0, 1]) {
        let device = inventory.device(id).unwrap();
        assert!(
            budget.bytes <= device.memory_bytes,
            "the budget of device {id} is {} bytes; the card has {} bytes",
            budget.bytes,
            device.memory_bytes
        );
    }
    let host = profile.budgets[2].bytes;
    assert!(
        host + FLOOR <= inventory.host_bytes,
        "the host budget is {host} bytes; host memory is {} bytes and the floor is {FLOOR} bytes",
        inventory.host_bytes
    );
    let store = profile.store.expect("an engine needs a [store] table (spec 4.1a)");
    assert!(store.nvme > 0);
    assert!(
        store.host.unwrap() < host,
        "the store's host budget fits inside the host budget"
    );
}

/// The Gemma 4 plan with no card, as the factory of the `xe` build plans. A deploy test opens no engine.
struct PlanOnly;

impl Factory for PlanOnly {
    fn plan(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        nagaya::gemma4::plan(config, devices)
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        panic!("a deploy test opens no engine")
    }
}

/// The plan of the speech engine, from its checkpoint and no card.
struct SpeechPlanOnly;

impl Factory for SpeechPlanOnly {
    fn plan(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        nagaya::qwen3tts::plan(config, devices)
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        panic!("a deploy test opens no engine")
    }
}

/// The plan of the SDXL engine, from the headers of its checkpoint and no card.
struct ImagePlanOnly;

impl Factory for ImagePlanOnly {
    fn plan(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        nagaya::sdxl::plan(config, devices)
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        panic!("a deploy test opens no engine")
    }
}

/// True when the SDXL checkpoint is named, as the unit names it. Without it, the test skips.
fn image_checkpoint() -> bool {
    if std::env::var_os(nagaya::sdxl::MODEL_VARIABLE).is_some() {
        return true;
    }
    if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") {
        panic!("SDXL_MODEL_DIR is not set and NAGAYA_REQUIRE_MODELS is set; name the SDXL checkpoint");
    }
    eprintln!("skipped: SDXL_MODEL_DIR is not set; set it to the SDXL base 1.0 checkpoint directory");
    false
}

/// True when the Qwen3-TTS checkpoint is named, as the unit names it. Without it, the test skips.
fn speech_checkpoint() -> bool {
    if std::env::var_os(nagaya::qwen3tts::MODEL_VARIABLE).is_some() {
        return true;
    }
    if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") {
        panic!("QWEN3TTS_MODEL_DIR is not set and NAGAYA_REQUIRE_MODELS is set; name the Qwen3-TTS checkpoint");
    }
    eprintln!("skipped: QWEN3TTS_MODEL_DIR is not set; set it to the Qwen3-TTS checkpoint directory");
    false
}

fn checkpoint() -> Option<PathBuf> {
    match std::env::var_os(nagaya::gemma4::MODEL_VARIABLE) {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("GEMMA4_MODEL_DIR is not set and NAGAYA_REQUIRE_MODELS is set; name the Gemma 4 checkpoint")
        }
        None => {
            eprintln!("skipped: GEMMA4_MODEL_DIR is not set; set it to the Gemma 4 checkpoint directory");
            None
        }
    }
}

#[test]
fn the_b70host_profile_names_the_speech_engine_on_demand_on_card_zero() {
    // Amy, 2026-10-24: the speech engine swaps in on demand, and the profile chooses its card (milestone 17).
    let profile = b70host_profile();
    let engine = &profile.engines[1];
    assert_eq!(
        (engine.name.as_str(), engine.kind.as_str()),
        ("voice", nagaya::qwen3tts::KIND)
    );
    assert_eq!(engine.devices, vec![0]);
    assert_eq!(engine.residency, Residency::OnDemand);
    assert_eq!(engine.idle_unload_s, None, "the default idle time of 60 s");
    assert!(
        engine.family.is_empty(),
        "an on-demand engine serves no council model (spec 3.3a)"
    );
    let config = nagaya::qwen3tts::config(&{
        // The checkpoint comes from QWEN3TTS_MODEL_DIR in the unit, so the check gives the table a directory.
        let mut table = engine.config.clone();
        assert!(
            !table.contains_key("model_dir"),
            "the checkpoint comes from QWEN3TTS_MODEL_DIR in the unit"
        );
        table.insert(
            "model_dir".to_owned(),
            toml::Value::String(std::env::temp_dir().display().to_string()),
        );
        table
    })
    .unwrap();
    assert_eq!(
        (
            config.slots,
            config.max_frames,
            config.max_prompt,
            config.frames_per_quantum
        ),
        (2, 1500, 512, 8)
    );
}

#[test]
fn the_b70host_profile_names_an_image_engine_on_demand_on_each_card() {
    // Amy, 2026-10-27: "one model per card", and each engine is its own served model (spec 11.9).
    let profile = b70host_profile();
    assert_eq!(
        profile.engines.len(),
        6,
        "the image engines follow gemma and the speech engine"
    );
    for (engine, (name, device)) in profile.engines[2..4].iter().zip([("sdxl-0", 0), ("sdxl-1", 1)]) {
        assert_eq!((engine.name.as_str(), engine.kind.as_str()), (name, nagaya::sdxl::KIND));
        assert_eq!(engine.devices, vec![device]);
        assert_eq!(engine.residency, Residency::OnDemand);
        assert_eq!(
            engine.idle_unload_s,
            Some(600),
            "{name}: an image session keeps its engine for 10 min"
        );
        assert!(
            engine.config.is_empty() && engine.family.is_empty(),
            "{name}: the checkpoint comes from SDXL_MODEL_DIR in the unit, and clip_skip is 1"
        );
    }
}

#[test]
fn the_b70host_profile_names_pony_on_card_zero_and_autismmix_on_card_one() {
    // Amy, 2026-10-28: Pony V6 and AutismMix from Civitai, beside SDXL base, one engine on each card. The profile
    // names the files; this test checks only their form.
    let profile = b70host_profile();
    assert_eq!(profile.engines.len(), 6);
    for (engine, (name, device)) in profile.engines[4..].iter().zip([("pony-0", 0), ("autismmix-1", 1)]) {
        assert_eq!((engine.name.as_str(), engine.kind.as_str()), (name, nagaya::sdxl::KIND));
        assert_eq!(engine.devices, vec![device]);
        assert_eq!(engine.residency, Residency::OnDemand);
        assert_eq!(engine.idle_unload_s, Some(600));
        assert!(
            !engine.config.contains_key("model_dir") && !engine.family.contains_key("model_dir"),
            "{name}: the tokenizer comes from SDXL_MODEL_DIR in the unit"
        );
        let config = nagaya::sdxl::config(&{
            let mut table = engine.config.clone();
            table.insert("model_dir".to_owned(), toml::Value::String("/sdxl".to_owned()));
            table
        })
        .unwrap();
        let file = config
            .checkpoint
            .unwrap_or_else(|| panic!("{name} names a single file"));
        assert!(
            file.is_absolute() && file.extension().is_some_and(|ext| ext == "safetensors"),
            "{name}: {}",
            file.display()
        );
        // Pony V6 asks for clip skip 2 (its Civitai page), and AutismMix is a Pony merge.
        let mut family = engine.family.clone();
        family.insert("model_dir".to_owned(), toml::Value::String("/sdxl".to_owned()));
        assert_eq!(nagaya::sdxl::family_config(&family).unwrap().1, 2, "{name}");
    }
}

/// True when each single file that the profile names exists. Without one, the test skips; a machine that does not
/// hold the files is normal (AGENTS.md, "Tests that need a model file").
fn named_checkpoints(profile: &PlacementProfile) -> bool {
    let absent: Vec<String> = profile
        .engines
        .iter()
        .filter_map(|engine| engine.config.get("checkpoint").and_then(|value| value.as_str()))
        .filter(|path| !std::path::Path::new(path).exists())
        .map(str::to_owned)
        .collect();
    if absent.is_empty() {
        return true;
    }
    if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") {
        panic!("NAGAYA_REQUIRE_MODELS is set, and the profile names absent files: {absent:?}");
    }
    eprintln!("skipped: the profile names checkpoints that this machine lacks: {absent:?}");
    false
}

#[test]
fn the_b70host_profile_admits_on_the_probed_b70host() {
    // The start path up to the open: the probe, the ignored iGPU, the plan and the ledger.
    if checkpoint().is_none()
        || !speech_checkpoint()
        || !image_checkpoint()
        || !named_checkpoints(&b70host_profile())
    {
        return;
    }
    let config = b70host_bootstrap();
    let mut inventory = FileProbe::new(repo().join("crates/nagaya/tests/fixtures/probe/b70host"))
        .probe()
        .unwrap();
    inventory.ignore(&config.ignore_devices).unwrap();
    for id in [0, 1] {
        assert!(!inventory.device(id).unwrap().ignored, "device {id} is a B70 card");
    }
    let mut registry = Registry::new();
    registry.register(nagaya::gemma4::KIND, Box::new(PlanOnly));
    registry.register(nagaya::qwen3tts::KIND, Box::new(SpeechPlanOnly));
    registry.register(nagaya::sdxl::KIND, Box::new(ImagePlanOnly));
    let admitted = admit(&b70host_profile(), &registry, &inventory, &mut Ledger::new()).expect("b70host admits");
    assert_eq!(admitted.engines.len(), 6);
    for engine in &admitted.engines[2..] {
        assert!(engine.limits.capabilities.contains(&nagaya_abi::Capability::Image));
    }
    assert!(
        admitted.engines[1]
            .limits
            .capabilities
            .contains(&nagaya_abi::Capability::Speech)
    );
    let limits = &admitted.engines[0].limits;
    for tier in [Tier::Device(0), Tier::Device(1), Tier::Host] {
        let bytes: u64 = limits
            .memory
            .mandatory
            .iter()
            .filter(|term| term.tier == tier)
            .map(|term| term.bytes)
            .sum();
        println!("{tier:?}: {bytes} bytes of mandatory terms");
    }
}

#[test]
fn the_b70host_unit_reads_its_config_and_the_checkpoint_and_hides_no_card() {
    let unit = b70host_unit();
    check_restart_policy(&unit);
    let environment = environment(&unit);
    let config = environment
        .iter()
        .find_map(|value| value.strip_prefix("NAGAYA_CONFIG=%h/src/nagaya/"))
        .expect("the unit sets NAGAYA_CONFIG under the repository");
    assert_eq!(config, "deploy/b70host/nagaya.toml");
    assert!(repo().join(config).is_file(), "the file that the unit names exists");
    assert!(environment.contains(&"GEMMA4_MODEL_DIR=%h/src/mkviolet/checkpoints/gemma-4-26B-A4B"));
    assert!(environment.contains(
        &"QWEN3TTS_MODEL_DIR=%h/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign/snapshots/\
          5ecdb67327fd37bb2e042aab12ff7391903235d3"
    ));
    assert!(environment.contains(
        &"SDXL_MODEL_DIR=%h/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/\
          462165984030d82259a11f4367a4eed129e94a7b"
    ));
    // The engine finds its cards by PCI address, so no Level Zero variable may hide one.
    assert!(
        environment.iter().all(|value| !value.starts_with("ZE_")),
        "{environment:?}"
    );
    assert!(
        environment.iter().all(|value| !value.starts_with("MK_")),
        "b70host has no mk engine"
    );
}

#[test]
fn the_b70host_unit_keeps_the_arenas_through_a_restart() {
    // Spec 10.2: systemd keeps the stored fds through a restart. The ceiling holds the 31 Gemma 4 slabs.
    let unit = b70host_unit();
    let ceiling: usize = unit_value(&unit, "FileDescriptorStoreMax").parse().unwrap();
    assert!(
        ceiling >= 31,
        "the ceiling {ceiling} is smaller than the 31 slabs of the Gemma 4 engine"
    );
    assert_eq!(unit_value(&unit, "FileDescriptorStorePreserve"), "restart");
}

/// The values of the `ExecStartPre=` lines of a unit, in order.
fn exec_start_pre(unit: &str) -> Vec<&str> {
    unit.lines()
        .filter_map(|line| line.strip_prefix("ExecStartPre="))
        .collect()
}

/// Check that a unit waits for the tailnet address of its bootstrap config before the process starts.
fn check_address_wait(unit: &str, bootstrap: &Bootstrap) {
    let address = bootstrap.http_bind.ip();
    assert_eq!(
        bootstrap.gossip_bind.ip(),
        address,
        "the wait covers one address, so both binds must use it"
    );
    let wait = format!("%h/src/nagaya/deploy/wait-address.sh {address} 120");
    assert!(
        exec_start_pre(unit).contains(&wait.as_str()),
        "the unit needs `ExecStartPre={wait}`"
    );
    // A user unit cannot order itself after a system unit, so such an `After=` only misleads the reader.
    assert!(
        !unit_value(unit, "After").contains("tailscaled.service"),
        "a user unit cannot wait for the system unit tailscaled.service"
    );
}

#[test]
fn the_units_wait_for_the_tailnet_address_of_their_config() {
    // On 2026-10-24 b70host booted after a power loss. The nagaya started before tailscaled set 198.51.100.10, the
    // gossip bind failed, and exit 3 kept the unit down.
    check_address_wait(&unit(), &bootstrap());
    check_address_wait(&b70host_unit(), &b70host_bootstrap());
}

#[test]
fn the_address_wait_passes_a_present_address_and_fails_an_absent_one() {
    let script = deploy("wait-address.sh");
    let present = std::process::Command::new(&script)
        .args(["127.0.0.1", "1"])
        .output()
        .unwrap();
    assert!(present.status.success(), "{present:?}");
    // 192.0.2.1 is TEST-NET-1 (RFC 5737), so no machine of ours holds it.
    let absent = std::process::Command::new(&script)
        .args(["192.0.2.1", "1"])
        .output()
        .unwrap();
    assert_eq!(absent.status.code(), Some(1), "{absent:?}");
    let message = String::from_utf8_lossy(&absent.stderr);
    assert!(
        message.contains("192.0.2.1"),
        "the message names the address: {message}"
    );
}
