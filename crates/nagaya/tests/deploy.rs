//! The files in `deploy/` for gfx1151host and b70host (spec 2.1, 2.4, 10.1). The real loaders read them, so a change to
//! a schema breaks these tests before it breaks a cutover.
//!
//! The admission of the b70host profile plans from the Gemma 4 checkpoint. That test skips when `GEMMA4_MODEL_DIR`
//! is not set, and fails then with `NAGAYA_REQUIRE_MODELS=1` (AGENTS "Tests that need a model file").
//!
//! A test must not depend on the layout of one machine (AGENTS, milestone 31 step 4). The admission test of the
//! b70host profile reads no engine name and no count. The eight tests that named the engines of the lab layout
//! of two cards are gone since 2026-11-15.

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
    // Amy, 2026-10-18: the nagaya takes the address of the Qwen3.8 engine, and kaijutsu's gate config names it.
    let config = bootstrap();
    assert_eq!(config.http_bind, "198.51.100.20:8090".parse::<SocketAddr>().unwrap());
    assert_eq!(config.gossip_bind.ip(), config.http_bind.ip());
    assert_ne!(config.gossip_bind.port(), config.http_bind.port());
    assert!(config.store_dir.is_absolute(), "the loader does not expand `~`");
    assert!(config.store_dir.ends_with(".local/share/nagaya/store"));
}

#[test]
fn gfx1151host_and_b70host_seed_each_other_for_one_cluster() {
    // Milestone 23: each machine seeds the gossip address of the other, so they join one cluster (spec 2.1, 2.3).
    let gfx1151host = bootstrap();
    let b70host = b70host_bootstrap();
    assert_eq!(gfx1151host.seed_peers, vec![b70host.gossip_bind]);
    assert_eq!(b70host.seed_peers, vec![gfx1151host.gossip_bind]);
}

#[test]
fn the_profile_names_the_qwen38_engine_and_the_store_budgets() {
    let profile = profile();
    assert_eq!(
        profile.engines.len(),
        2,
        "the Qwen3.8 engine and Qwen-Image (milestone 22)"
    );
    let engine = &profile.engines[0];
    assert_eq!(engine.name, "qwen38");
    assert_eq!(engine.kind, "mk-qwen38-gfx1151");
    // Amy, 2026-11-01: Qwen-Image stays loaded on gfx1151host for a while, and the Qwen3.8 engine swaps in through the API
    // (spec 3.3b).
    assert_eq!(engine.residency, Residency::OnDemand);
    assert_eq!(engine.devices, vec![0]);
    // The factory of the Qwen3.8 engine plans from this key (`nagaya::qwen38`). It refuses any other key.
    let keys: Vec<&String> = engine.config.keys().collect();
    assert_eq!(keys, vec!["weights_bytes"]);
    assert!(engine.config["weights_bytes"].as_integer().unwrap() > 0);
    let store = profile.store.expect("an engine needs a [store] table (spec 4.1a)");
    assert!(store.nvme > 0);
    assert!(
        store.host.is_some(),
        "the host budget of the store is explicit on gfx1151host"
    );
    // Objects have their own budget of 200 GiB (spec 6.15; Amy, 2026-11-01).
    assert_eq!(store.objects, Some(200 << 30));
}

#[test]
fn the_profile_names_qwen_image_with_its_files_and_a_plan_table() {
    let profile = profile();
    let engine = &profile.engines[1];
    assert_eq!(engine.name, "qwen-image-fusal");
    assert_eq!(engine.kind, nagaya::qwenimage21::KIND);
    assert_eq!(engine.residency, Residency::Resident);
    assert_eq!(engine.devices, vec![0]);
    let config = nagaya::qwenimage21::config(&engine.config).unwrap();
    assert!(config.lora.is_some(), "the variant is FUSAL");
    assert!(
        config.plan_dir.is_some(),
        "the plan table is a file in the store directory (spec 12.1)"
    );
}

/// The plans of both engines and the store's host budget fit the pool, with room for a staging copy of a context of
/// 32,768 tokens (docs/deploy.md, "Two engines on gfx1151host"). The plan of Qwen-Image reads the headers of the files that
/// the profile names, so the test skips on a machine without them.
#[test]
fn both_engines_and_the_store_fit_the_pool_with_room_for_a_staging_copy() {
    let profile = profile();
    let config = nagaya::qwenimage21::config(&profile.engines[1].config).unwrap();
    if !config.encoder.is_file() {
        eprintln!("skipped: the Qwen-Image files of the profile are absent");
        return;
    }
    let qwen38_weights = profile.engines[0].config["weights_bytes"].as_integer().unwrap() as u64;
    let qwen38_terms: u64 = nagaya_mk_qwen38_gfx1151_plan(qwen38_weights);
    let qwen: u64 = nagaya_mk_qwenimage21_gfx1151::plan::plan(&config)
        .unwrap()
        .memory
        .mandatory
        .iter()
        .map(|term| term.bytes)
        .sum();
    let pool = [TierSpec::from(Tier::Device(0)), TierSpec::from(Tier::Host)];
    let budget = profile.budgets.iter().find(|budget| budget.tiers == pool).unwrap();
    let store = profile.store.unwrap().host.unwrap();
    let room = budget.bytes as i128 - (qwen38_terms + qwen + store) as i128;
    eprintln!("qwen38 {qwen38_terms}, qwen-image {qwen}, store {store}: {room} bytes stay free in the pool");
    // A staging copy of a context of 32,768 tokens is about 1.93 GB (docs/deploy.md, "The memory budgets").
    assert!(
        room >= 1_930_000_000,
        "the pool keeps {room} bytes free; a staging copy needs about 1.93 GB"
    );
}

/// The mandatory terms of the Qwen3.8 engine for its weights: the weights, the pass scratch and the context pool. They
/// are docs/deploy.md's numbers; the engine crate builds only with the feature `hip`.
fn nagaya_mk_qwen38_gfx1151_plan(weights: u64) -> u64 {
    assert_eq!(
        weights, 65_490_227_200,
        "a new weights size changes the sum of docs/deploy.md"
    );
    78_012_452_864
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
fn the_unit_names_three_shards() {
    let unit = unit();
    let environment: Vec<&str> = unit
        .lines()
        .filter_map(|line| line.strip_prefix("Environment="))
        .collect();
    let shards = environment
        .iter()
        .find_map(|value| {
            value
                .strip_prefix("\"QWEN38_SHARDS=")
                .and_then(|rest| rest.strip_suffix('"'))
        })
        .expect("the unit sets QWEN38_SHARDS in quotes");
    let paths: Vec<&str> = shards.split(':').collect();
    assert_eq!(paths.len(), 3, "the unit sets three shards, with : between them");
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
    assert!(config.store_dir.is_absolute(), "the loader does not expand `~`");
    assert!(config.store_dir.ends_with(".local/share/nagaya/store"));
    // The Radeon 610M drives the display (spec 2.1, "Ignored devices").
    assert_eq!(config.ignore_devices, vec!["0000:80:00.0".to_owned()]);
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

/// The plan of an LFM3 model, from its folder and no card.
struct Lfm3PlanOnly;

impl Factory for Lfm3PlanOnly {
    fn plan(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        nagaya::lfm3::plan(config, devices)
    }

    fn served_names(&self, engine: &str, config: &toml::Table) -> Option<Vec<String>> {
        nagaya::lfm3::served_names(engine, config)
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

/// The plan of the EmbeddingGemma 2 engine, from the headers of its checkpoint and no card.
struct EmbeddingPlanOnly;

impl Factory for EmbeddingPlanOnly {
    fn plan(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        nagaya::embeddinggemma2::plan(config, devices)
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        panic!("a deploy test opens no engine")
    }
}

/// True when the EmbeddingGemma 2 checkpoint is named, as the unit names it. Without it, the test skips.
fn embedding_checkpoint() -> bool {
    if std::env::var_os(nagaya::embeddinggemma2::MODEL_VARIABLE).is_some() {
        return true;
    }
    if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") {
        panic!("EMBEDDINGGEMMA2_MODEL_DIR is not set and NAGAYA_REQUIRE_MODELS is set; name the checkpoint");
    }
    eprintln!("skipped: EMBEDDINGGEMMA2_MODEL_DIR is not set; set it to the EmbeddingGemma 2 checkpoint directory");
    false
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
        || !embedding_checkpoint()
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
    registry.register(nagaya::embeddinggemma2::KIND, Box::new(EmbeddingPlanOnly));
    registry.register(nagaya::lfm3::KIND, Box::new(Lfm3PlanOnly));
    // The admission runs the start rules that need no card: the plans, the ledger and the residency (spec 3.3a).
    let profile = b70host_profile();
    let admitted = admit(&profile, &registry, &inventory, &mut Ledger::new()).expect("b70host admits");
    assert_eq!(admitted.engines.len(), profile.engines.len());
    check_served_names(&profile, &admitted.engines);
    // What the engines that load at the start take of each device. A term names a place in the device list of its
    // engine (spec 2.4). The test reads no engine name and no count, so a new layout needs no change here.
    let mut at_start: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    for (placement, engine) in profile.engines.iter().zip(&admitted.engines) {
        if placement.residency != Residency::Resident {
            continue;
        }
        for term in &engine.limits.memory.mandatory {
            if let Tier::Device(place) = term.tier {
                *at_start.entry(placement.devices[place as usize]).or_default() += term.bytes;
            }
        }
    }
    for (device, bytes) in &at_start {
        let budget = profile
            .budgets
            .iter()
            .find(|budget| budget.tiers.iter().any(|tier| tier.0 == Tier::Device(*device)))
            .unwrap_or_else(|| panic!("device {device} has a budget"));
        println!(
            "device {device}: {bytes} bytes at the start, of a budget of {}",
            budget.bytes
        );
        assert!(*bytes <= budget.bytes, "device {device}: {bytes} bytes at the start");
    }
    println!("unassigned at the start: {:?}", admitted.admission.unassigned);
}

/// Check the served names of the start (spec 2.3): each LFM3 engine serves its engine name, both in the profile check
/// and in the family plan of the council, and no two engines of the profile serve one name. The start of 12:59 on
/// 2026-11-10 exited 3, because `lfm3` and `lfm3-1` both served `lfm3`.
fn check_served_names(profile: &PlacementProfile, admitted: &[nagaya::admission::AdmittedEngine]) {
    let mut families = nagaya::serving::FamilyRegistry::new();
    families.register_with_objects("lfm3", nagaya::lfm3::open_family_with);
    let mut names: Vec<String> = Vec::new();
    for (placement, engine) in profile.engines.iter().zip(admitted) {
        assert_eq!(placement.name, engine.name);
        names.extend(engine.served.clone().unwrap_or_default());
        if placement.kind != nagaya::lfm3::KIND {
            continue;
        }
        assert_eq!(engine.served, Some(vec![engine.name.clone()]), "{}", engine.name);
        let dir = PathBuf::from(placement.family["model_dir"].as_str().unwrap());
        let family = nagaya_text_lfm3::Lfm3Family::from_dir(&dir).unwrap();
        let mut identity = nagaya_abi::fake::FakeAlways::new().with_family("lfm3").identity();
        identity.model = match family.kind() {
            nagaya_text_lfm3::Kind::Text => "lfm3",
            nagaya_text_lfm3::Kind::Omni => "lfm3-omni",
        }
        .to_owned();
        identity.tokenizer_hash = family.tokenizer_hash().to_owned();
        let opened = nagaya::process::OpenedEngine {
            name: engine.name.clone(),
            thread: std::thread::current().id(),
            identity,
            limits: engine.limits.clone(),
            handoff: None,
        };
        let plan = families
            .plan(&opened, &placement.family)
            .unwrap()
            .expect("the LFM3 family is registered");
        assert_eq!(
            plan.config.model, engine.name,
            "the council serves the name of the profile check"
        );
    }
    println!("the served names that the profile check knows: {names:?}");
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len(), "two engines serve one name: {names:?}");
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
    assert!(environment.contains(
        &"EMBEDDINGGEMMA2_MODEL_DIR=%h/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/\
          f9972ac22e696a105c482202646979e33e4b5a1e"
    ));
    // The engine finds its cards by PCI address, so no Level Zero variable may hide one.
    assert!(
        environment.iter().all(|value| !value.starts_with("ZE_")),
        "{environment:?}"
    );
    assert!(
        environment.iter().all(|value| !value.starts_with("MK_")),
        "b70host has no Qwen3.8 engine"
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
