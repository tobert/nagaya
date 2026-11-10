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
fn the_profile_names_the_mk_engine_and_the_store_budgets() {
    let profile = profile();
    assert_eq!(profile.engines.len(), 2, "mk and Qwen-Image (milestone 22)");
    let engine = &profile.engines[0];
    assert_eq!(engine.name, "mk");
    assert_eq!(engine.kind, "mk-qwen38-gfx1151");
    // Amy, 2026-11-01: Qwen-Image stays loaded on gfx1151host for a while, and mk swaps in through the API (spec 3.3b).
    assert_eq!(engine.residency, Residency::OnDemand);
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
    let mk = profile.engines[0].config["weights_bytes"].as_integer().unwrap() as u64;
    let mk_terms: u64 = nagaya_mk_qwen38_gfx1151_plan(mk);
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
    let room = budget.bytes as i128 - (mk_terms + qwen + store) as i128;
    eprintln!("mk {mk_terms}, qwen-image {qwen}, store {store}: {room} bytes stay free in the pool");
    // A staging copy of a context of 32,768 tokens is about 1.93 GB (docs/deploy.md, "The memory budgets").
    assert!(
        room >= 1_930_000_000,
        "the pool keeps {room} bytes free; a staging copy needs about 1.93 GB"
    );
}

/// The mandatory terms of the mk engine for its weights: the weights, the pass scratch and the context pool. They are
/// docs/deploy.md's numbers; the mk crate builds only with the feature `hip`.
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
        12,
        "gemma, the speech engine, five image engines on demand, the pinned embedding engine, inpaint-0 and the two \
         LFM3 models"
    );
    let engine = &profile.engines[0];
    // Amy, 2026-11-06 (milestone 24): Gemma 4 goes on demand, and it loads only through the load routes.
    assert_eq!(engine.residency, Residency::OnDemand);
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

/// The plan of an LFM3 model, from its folder and no card.
struct Lfm3PlanOnly;

impl Factory for Lfm3PlanOnly {
    fn plan(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        nagaya::lfm3::plan(config, devices)
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

#[test]
fn the_b70host_profile_names_the_speech_engine_on_demand_on_card_one() {
    // Amy, 2026-10-24: the speech engine swaps in on demand, and the profile chooses its card (milestone 17). Amy,
    // 2026-11-06 (milestone 24): card 1 holds the image engines, the embedding engine and speech.
    let profile = b70host_profile();
    let engine = &profile.engines[1];
    assert_eq!(
        (engine.name.as_str(), engine.kind.as_str()),
        ("voice", nagaya::qwen3tts::KIND)
    );
    assert_eq!(engine.devices, vec![1]);
    assert_eq!(engine.residency, Residency::OnDemand);
    assert!(
        !engine.pinned,
        "a client unloads the speech engine (spec 3.3a, \"Explicit layout\")"
    );
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
        12,
        "the image engines follow gemma and the speech engine"
    );
    for (engine, (name, device)) in profile.engines[2..4].iter().zip([("sdxl-0", 0), ("sdxl-1", 1)]) {
        assert_eq!((engine.name.as_str(), engine.kind.as_str()), (name, nagaya::sdxl::KIND));
        assert_eq!(engine.devices, vec![device]);
        assert_eq!(engine.residency, Residency::OnDemand);
        assert!(
            !engine.pinned,
            "{name}: a client unloads an image engine (spec 3.3a, \"Explicit layout\")"
        );
        assert!(
            engine.config.is_empty() && engine.family.is_empty(),
            "{name}: the checkpoint comes from SDXL_MODEL_DIR in the unit, and clip_skip is 2"
        );
    }
}

#[test]
fn the_b70host_profile_names_pony_on_card_zero_and_autismmix_on_card_one() {
    // Amy, 2026-10-28: Pony V6 and AutismMix from Civitai, beside SDXL base, one engine on each card. The profile
    // names the files; this test checks only their form.
    let profile = b70host_profile();
    assert_eq!(profile.engines.len(), 12);
    // Amy, 2026-10-29: SDXL-Lightning 4-step, a single file on card 0, shares that card with pony-0 (spec 11.9a).
    let singles = [("pony-0", 0), ("autismmix-1", 1), ("lightning-0", 0)];
    for (engine, (name, device)) in profile.engines[4..7].iter().zip(singles) {
        assert_eq!((engine.name.as_str(), engine.kind.as_str()), (name, nagaya::sdxl::KIND));
        assert_eq!(engine.devices, vec![device]);
        assert_eq!(engine.residency, Residency::OnDemand);
        assert!(!engine.pinned, "{name}");
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
        // Pony asks for "clip skip 2", the penultimate layer. `clip_skip` counts as A1111 does, so that is the default.
        assert!(engine.family.is_empty(), "{name}: the default clip_skip 2");
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

/// The embedding engine is resident and pinned on card 1 (spec 11.10, 3.3a "Explicit layout").
#[test]
fn the_b70host_profile_names_the_pinned_embedding_engine_on_card_one() {
    let profile = b70host_profile();
    let engine = &profile.engines[7];
    assert_eq!(
        (engine.name.as_str(), engine.kind.as_str()),
        ("embeddinggemma-2", nagaya::embeddinggemma2::KIND)
    );
    assert_eq!(engine.devices, vec![1]);
    assert_eq!(engine.residency, Residency::Resident);
    assert!(engine.pinned, "no load and no unload removes the embedding engine");
    assert!(
        engine.config.is_empty() && engine.family.is_empty(),
        "the checkpoint comes from EMBEDDINGGEMMA2_MODEL_DIR in the unit"
    );
}

/// The 9-channel inpainting UNet on card 1, on demand (milestone 20 step 7, spec 11.9b "The 9-channel UNet"). Its key
/// `model_dir` names the inpainting folder, and its family reads the tokenizer from `SDXL_MODEL_DIR` in the unit.
#[test]
fn the_b70host_profile_names_the_inpainting_engine_on_demand_on_card_one() {
    let profile = b70host_profile();
    let engine = &profile.engines[8];
    assert_eq!(
        (engine.name.as_str(), engine.kind.as_str()),
        ("inpaint-0", nagaya::sdxl::KIND)
    );
    assert_eq!(engine.devices, vec![1]);
    assert_eq!(engine.residency, Residency::OnDemand);
    assert!(!engine.pinned);
    assert!(
        engine.family.is_empty(),
        "the tokenizer comes from SDXL_MODEL_DIR, and clip_skip is the default"
    );
    let config = nagaya::sdxl::config(&engine.config).unwrap();
    assert_eq!(config.checkpoint, None, "the inpainting UNet is a diffusers folder");
    assert!(
        config.model_dir.is_absolute() && config.model_dir.ends_with("115134f363124c53c7d878647567d04daf26e41e"),
        "inpaint-0 names snapshot 115134f of the inpainting folder: {}",
        config.model_dir.display()
    );
}

/// Amy, 2026-11-06 (milestone 24): both LFM3 models are resident on card 0 of b70host. Each engine name is its
/// served name (spec 11.11), and each config and family table names its snapshot folder.
#[test]
fn the_b70host_profile_names_both_lfm3_models_resident_on_card_zero() {
    let profile = b70host_profile();
    assert_eq!(
        profile.engines.len(),
        12,
        "the two LFM3 models of card 0 follow inpaint-0"
    );
    for (engine, (name, repo, snapshot)) in profile.engines[9..].iter().zip([
        (
            "lfm3-omni",
            "models--LiquidAI--LFM3-Omni-2B",
            "21d75123f86c1f7ffb4bef1ea202710c4e7ebe67",
        ),
        (
            "lfm3",
            "models--LiquidAI--LFM3-1.7B",
            "0dc4268b561499c648c7ba7c9466bf3d8d01517a",
        ),
    ]) {
        assert_eq!((engine.name.as_str(), engine.kind.as_str()), (name, nagaya::lfm3::KIND));
        assert_eq!(engine.devices, vec![0], "{name}");
        assert_eq!(engine.residency, Residency::Resident, "{name}");
        assert!(!engine.pinned, "{name}: a load of an image engine on card 0 unloads it");
        for table in [&engine.config, &engine.family] {
            let dir = std::path::PathBuf::from(table["model_dir"].as_str().unwrap());
            assert!(dir.is_absolute(), "{name}: the loader does not expand `~`");
            assert!(
                dir.ends_with(format!("{repo}/snapshots/{snapshot}")),
                "{name}: {}",
                dir.display()
            );
        }
        // Eight slots, so a batched decode pass can hold eight contexts (milestone 27).
        assert_eq!(engine.config["slots"].as_integer(), Some(8), "{name}");
    }
    // The J-Lens of LFM3-1.7B, fitted on b70host's CPU (milestone 26 step 5, docs/lab.md). Omni has none yet.
    let lab = |i: usize| profile.engines[i].lab.clone();
    assert!(lab(9).is_empty(), "lfm3-omni names no lens");
    let jlens = std::path::PathBuf::from(lab(10)["jlens"].as_str().expect("lfm3 names its J-Lens"));
    assert_eq!(lab(10).len(), 1, "the table has one key");
    assert!(jlens.is_absolute(), "the loader does not expand `~`");
    assert!(
        jlens.ends_with(".local/share/nagaya/lens/lfm3.jlens.safetensors"),
        "{}",
        jlens.display()
    );
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
    let admitted = admit(&b70host_profile(), &registry, &inventory, &mut Ledger::new()).expect("b70host admits");
    assert_eq!(admitted.engines.len(), 11);
    // Both LFM3 models are resident on card 0 at the start (milestone 24): their terms fit the card together.
    let on_card_0: u64 = admitted.engines[9..]
        .iter()
        .flat_map(|engine| engine.limits.memory.mandatory.iter())
        .filter(|term| term.tier == Tier::Device(0))
        .map(|term| term.bytes)
        .sum();
    println!("the two LFM3 models take {on_card_0} bytes of card 0");
    assert!(on_card_0 < 32 << 30, "{on_card_0}");
    for engine in admitted.engines[2..7].iter().chain(&admitted.engines[8..9]) {
        assert!(engine.limits.capabilities.contains(&nagaya_abi::Capability::Image));
    }
    assert!(
        admitted.engines[7]
            .limits
            .capabilities
            .contains(&nagaya_abi::Capability::Embedding)
    );
    // The resident engines take memory at the start: gemma on both cards, and the embedding engine beside it on card 1.
    let unassigned = admitted.admission.unassigned.clone();
    println!("unassigned at the start: {unassigned:?}");
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

/// Amy, 2026-11-10 (milestone 29): a second `LFM3-1.7B` on card 1, so a swarm runs on both cards. It is its own
/// served model, and it reads the snapshot folder of `lfm3`. It is resident, because the process refuses a text model
/// on demand with a `family` table (`admission.rs`): the start of 12:49 on 2026-11-10 failed with exit 3.
#[test]
fn the_b70host_profile_names_lfm3_1_resident_on_card_one() {
    let profile = b70host_profile();
    let engine = profile.engines.last().unwrap();
    assert_eq!(
        (engine.name.as_str(), engine.kind.as_str()),
        ("lfm3-1", nagaya::lfm3::KIND)
    );
    assert_eq!(engine.devices, vec![1]);
    assert_eq!(
        engine.residency,
        Residency::Resident,
        "a text model on demand is not built"
    );
    assert!(!engine.pinned);
    let lfm3 = profile.engines.iter().find(|e| e.name == "lfm3").unwrap();
    assert_eq!(engine.config, lfm3.config, "the same weights and slots as lfm3");
    assert_eq!(engine.family, lfm3.family);
}
