use nagaya::gossip::MAX_KEY_BYTES;
use nagaya::profile::{
    BudgetSpec, EnginePlacement, MAX_ENGINE_NAME_BYTES, PlacementProfile, Residency, StoreSpec, TierSpec,
};
use nagaya_abi::Tier;

const EXAMPLE: &str = r#"
[[engines]]
name = "qwen"
kind = "fake"
devices = [0]
share = 1
pinned = true

[engines.config]
slots = 2

[[engines]]
name = "helper"
kind = "fake"
devices = [0]
share = 3
residency = "on_demand"

[[budgets]]
tiers = ["device(0)", "host"]
bytes = 1000

[store]
nvme = 4096
host = 1024
interval_copies = 3
"#;

fn example_profile() -> PlacementProfile {
    PlacementProfile {
        engines: vec![
            EnginePlacement {
                name: "qwen".to_owned(),
                kind: "fake".to_owned(),
                devices: vec![0],
                config: "slots = 2\n".parse::<toml::Table>().unwrap(),
                family: toml::Table::new(),
                lab: toml::Table::new(),
                share: 1,
                residency: Residency::Resident,
                pinned: true,
            },
            EnginePlacement {
                name: "helper".to_owned(),
                kind: "fake".to_owned(),
                devices: vec![0],
                config: toml::Table::new(),
                family: toml::Table::new(),
                lab: toml::Table::new(),
                share: 3,
                residency: Residency::OnDemand,
                pinned: false,
            },
        ],
        budgets: vec![BudgetSpec {
            tiers: vec![TierSpec::from(Tier::Device(0)), TierSpec::from(Tier::Host)],
            bytes: 1000,
        }],
        store: Some(StoreSpec {
            nvme: 4096,
            host: Some(1024),
            interval_copies: 3,
            objects: None,
        }),
        cpu: None,
    }
}

#[test]
fn the_profile_names_engines_devices_config_and_shares() {
    let profile = PlacementProfile::parse(EXAMPLE).unwrap();
    assert_eq!(profile.engines.len(), 2);
    assert_eq!(profile.engines[0].name, "qwen");
    assert_eq!(profile.engines[0].kind, "fake");
    assert_eq!(profile.engines[0].devices, vec![0]);
    assert_eq!(profile.engines[0].share, 1);
    assert_eq!(profile.engines[0].config.get("slots").unwrap().as_integer(), Some(2));
    assert_eq!(
        profile.budgets[0].tiers,
        vec![TierSpec::from(Tier::Device(0)), TierSpec::from(Tier::Host)]
    );
    assert_eq!(profile.budgets[0].bytes, 1000);
    assert_eq!(
        profile.store,
        Some(StoreSpec {
            nvme: 4096,
            host: Some(1024),
            interval_copies: 3,
            objects: None,
        })
    );
}

#[test]
fn a_profile_without_a_store_section_has_no_store_budget() {
    let text = EXAMPLE.replace("\n[store]\nnvme = 4096\nhost = 1024\ninterval_copies = 3\n", "");
    let profile = PlacementProfile::parse(&text).unwrap();
    assert_eq!(profile.store, None);
}

#[test]
fn interval_copies_is_zero_when_the_key_is_absent() {
    let text = EXAMPLE.replace("interval_copies = 3\n", "");
    let profile = PlacementProfile::parse(&text).unwrap();
    assert_eq!(profile.store.unwrap().interval_copies, 0);
}

#[test]
fn a_negative_interval_copies_is_refused() {
    let text = EXAMPLE.replace("interval_copies = 3", "interval_copies = -1");
    let error = PlacementProfile::parse(&text).unwrap_err();
    assert!(error.to_string().contains("interval_copies"), "{error}");
}

#[test]
fn a_store_section_without_nvme_names_the_key() {
    let text = EXAMPLE.replace("nvme = 4096\n", "");
    let error = PlacementProfile::parse(&text).unwrap_err();
    assert!(error.to_string().contains("nvme"), "{error}");
}

#[test]
fn the_optional_engine_fields_have_defaults() {
    let profile = PlacementProfile::parse(EXAMPLE).unwrap();
    assert_eq!(profile.engines[1].config, toml::Table::new());
    assert_eq!(profile.engines[1].family, toml::Table::new());
    assert_eq!(profile.engines[0].residency, Residency::Resident);
    assert!(!profile.engines[1].pinned);
}

#[test]
fn an_engine_names_its_pin() {
    let profile = PlacementProfile::parse(EXAMPLE).unwrap();
    assert!(profile.engines[0].pinned);
    assert_eq!(profile.engines[1].residency, Residency::OnDemand);
}

/// The idle time is gone (spec 3.3a, "Explicit layout"). A profile that names it does not parse, and the message
/// says that the key is gone.
#[test]
fn a_profile_that_names_an_idle_time_is_refused_and_says_the_key_is_gone() {
    let text = EXAMPLE.replace(
        "residency = \"on_demand\"\n",
        "residency = \"on_demand\"\nidle_unload_s = 30\n",
    );
    let error = PlacementProfile::parse(&text).unwrap_err().to_string();
    assert!(error.contains("`idle_unload_s`"), "{error}");
    assert!(error.contains("the key is gone"), "{error}");
    assert!(error.contains("helper"), "the message names the engine: {error}");
}

/// The pin of the identity is gone. The canary holds the bits now (spec 8.1). A profile that names the key does not
/// parse, and the message says that the key is gone, also when the value is `false`.
#[test]
fn a_profile_that_names_pin_identity_is_refused_and_says_the_key_is_gone() {
    for value in ["true", "false"] {
        let text = EXAMPLE.replace(
            "share = 3
",
            &format!("share = 3\npin_identity = {value}\n"),
        );
        let error = PlacementProfile::parse(&text).unwrap_err().to_string();
        assert!(error.contains("`pin_identity`"), "{error}");
        assert!(error.contains("the key is gone"), "{error}");
        assert!(error.contains("helper"), "the message names the engine: {error}");
    }
}

#[test]
fn a_residency_that_is_not_known_is_refused_with_its_text() {
    let text = EXAMPLE.replace("\"on_demand\"", "\"lazy\"");
    let error = PlacementProfile::parse(&text).unwrap_err();
    assert!(
        error.to_string().contains("lazy"),
        "the message must name the residency: {error}"
    );
}

#[test]
fn a_resident_engine_writes_no_residency_key() {
    let mut profile = example_profile();
    profile.engines.truncate(1);
    profile.engines[0].pinned = false;
    let text = profile.to_toml().unwrap();
    assert!(!text.contains("residency"), "{text}");
    assert!(
        !text.contains("pinned"),
        "an engine that is not pinned writes no key: {text}"
    );
}

#[test]
fn an_engine_entry_names_the_files_of_its_family() {
    let text = EXAMPLE.replace(
        "[engines.config]",
        "[engines.family]\ngguf = \"/models/a.gguf\"\n\n[engines.config]",
    );
    let profile = PlacementProfile::parse(&text).unwrap();
    assert_eq!(profile.engines[0].family["gguf"].as_str(), Some("/models/a.gguf"));
    assert_eq!(profile.engines[1].family, toml::Table::new());
}

#[test]
fn an_engine_entry_names_its_lab_table_and_an_entry_without_one_writes_no_key() {
    let text = EXAMPLE.replacen(
        "[engines.config]",
        "[engines.lab]\njlens = \"/lenses/lfm3.safetensors\"\n\n[engines.config]",
        1,
    );
    let profile = PlacementProfile::parse(&text).unwrap();
    assert_eq!(
        profile.engines[0].lab["jlens"].as_str(),
        Some("/lenses/lfm3.safetensors")
    );
    assert_eq!(profile.engines[1].lab, toml::Table::new());
    let written = profile.to_toml().unwrap();
    assert_eq!(written.matches("[engines.lab]").count(), 1, "{written}");
    assert_eq!(PlacementProfile::parse(&written).unwrap(), profile);
}

#[test]
fn an_unknown_key_is_refused() {
    let text = format!("unknown_key = 1\n{EXAMPLE}");
    let error = PlacementProfile::parse(&text).unwrap_err();
    assert!(
        error.to_string().contains("unknown_key"),
        "the message must name the unknown key: {error}"
    );
}

#[test]
fn a_bad_tier_names_the_text() {
    let text = EXAMPLE.replace("device(0)", "gpu0");
    let error = PlacementProfile::parse(&text).unwrap_err();
    assert!(
        error.to_string().contains("gpu0"),
        "the message must name the tier: {error}"
    );
}

#[test]
fn a_profile_round_trips_through_text() {
    let profile = example_profile();
    let text = profile.to_toml().unwrap();
    assert_eq!(PlacementProfile::parse(&text).unwrap(), profile);
}

#[test]
fn a_profile_round_trips_through_the_store_directory() {
    let directory = tempfile::tempdir().unwrap();
    let profile = example_profile();
    let path = profile.write(directory.path()).unwrap();
    assert_eq!(path, directory.path().join("profile.toml"));
    assert_eq!(PlacementProfile::read_active(directory.path()).unwrap(), profile);
    assert_eq!(PlacementProfile::read_active(directory.path()).unwrap(), profile);
}

fn with_name(name: &str) -> String {
    EXAMPLE.replacen("name = \"qwen\"", &format!("name = \"{name}\""), 1)
}

#[test]
fn an_engine_name_has_the_characters_of_a_node_name() {
    for name in ["qwen", "qwen-3.8_a", "A1.b"] {
        let profile = PlacementProfile::parse(&with_name(name)).unwrap();
        assert_eq!(profile.engines[0].name, name);
    }
    for name in ["", "a b", "a:b", "engine:x", "caf\u{e9}", "a/b"] {
        let error = PlacementProfile::parse(&with_name(name)).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("engine name"), "{name:?}: {text}");
        assert!(
            text.contains(&format!("`{name}`")),
            "the message names the name {name:?}: {text}"
        );
    }
}

#[test]
fn an_engine_name_has_a_length_bound_that_fits_a_gossip_key() {
    // The key is `engine:<name>`. It must stay within the key bound of gossip.
    assert!("engine:".len() + MAX_ENGINE_NAME_BYTES <= MAX_KEY_BYTES);
    let longest = "n".repeat(MAX_ENGINE_NAME_BYTES);
    assert!(PlacementProfile::parse(&with_name(&longest)).is_ok());
    let too_long = "n".repeat(MAX_ENGINE_NAME_BYTES + 1);
    let error = PlacementProfile::parse(&with_name(&too_long)).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("engine name"), "{text}");
    assert!(
        text.contains(&MAX_ENGINE_NAME_BYTES.to_string()),
        "the message gives the bound: {text}"
    );
}

#[test]
fn a_profile_from_a_peer_checks_the_engine_name_too() {
    // A desired profile arrives as JSON (spec 2.3). It uses the same rules as a file.
    let mut value = serde_json::to_value(example_profile()).unwrap();
    value["engines"][0]["name"] = serde_json::json!("a b");
    let error = serde_json::from_value::<PlacementProfile>(value).unwrap_err();
    assert!(error.to_string().contains("engine name"), "{error}");
}
