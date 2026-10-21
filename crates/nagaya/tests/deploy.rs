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
use nagaya::profile::{PROFILE_FILE, PlacementProfile, TierSpec};
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
fn the_unit_names_the_mk_library_and_three_shards() {
    let unit = unit();
    let environment: Vec<&str> = unit
        .lines()
        .filter_map(|line| line.strip_prefix("Environment="))
        .collect();
    let library = environment
        .iter()
        .find_map(|value| value.strip_prefix("MK_ENGINE_LIB="))
        .expect("the unit sets MK_ENGINE_LIB");
    assert!(library.ends_with("/build/src/engine/libmkengine.so"), "{library}");
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
    assert_eq!(profile.engines.len(), 1);
    let engine = &profile.engines[0];
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
fn the_b70host_profile_admits_on_the_probed_b70host() {
    // The start path up to the open: the probe, the ignored iGPU, the plan and the ledger.
    if checkpoint().is_none() {
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
    let admitted = admit(&b70host_profile(), &registry, &inventory, &mut Ledger::new()).expect("b70host admits");
    assert_eq!(admitted.engines.len(), 1);
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
