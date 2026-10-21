//! The files in `deploy/` for gfx1151host (spec 2.1, 2.4, 10.1). The real loaders read them, so a change to a schema
//! breaks these tests before it breaks a cutover.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use nagaya::bootstrap::Bootstrap;
use nagaya::inventory::Probe;
use nagaya::probe::FileProbe;
use nagaya::profile::{PROFILE_FILE, PlacementProfile, TierSpec};
use nagaya_abi::Tier;

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
    const FLOOR: u64 = 12 << 30;
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

#[test]
fn the_unit_is_a_notify_unit_with_the_restart_policy_of_spec_10_4() {
    let unit = unit();
    assert_eq!(unit_value(&unit, "Type"), "notify");
    assert_eq!(unit_value(&unit, "Restart"), "on-failure");
    assert_eq!(unit_value(&unit, "RestartSec"), "30");
    assert_eq!(unit_value(&unit, "RestartPreventExitStatus"), "3");
    assert_eq!(unit_value(&unit, "RestartForceExitStatus"), "4");
    assert_eq!(unit_value(&unit, "StartLimitIntervalSec"), "600");
    assert_eq!(unit_value(&unit, "StartLimitBurst"), "3");
    assert_eq!(unit_value(&unit, "TimeoutStartSec"), "300");
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
