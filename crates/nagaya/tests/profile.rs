use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, StoreSpec, TierSpec};
use nagaya_abi::Tier;

const EXAMPLE: &str = r#"
[[engines]]
name = "qwen"
kind = "fake"
devices = [0]
share = 1
pin_identity = true

[engines.config]
slots = 2

[[engines]]
name = "helper"
kind = "fake"
devices = [0]
share = 3

[[budgets]]
tiers = ["device(0)", "host"]
bytes = 1000

[store]
nvme = 4096
host = 1024
"#;

fn example_profile() -> PlacementProfile {
    PlacementProfile {
        engines: vec![
            EnginePlacement {
                name: "qwen".to_owned(),
                kind: "fake".to_owned(),
                devices: vec![0],
                config: "slots = 2\n".parse::<toml::Table>().unwrap(),
                share: 1,
                pin_identity: true,
            },
            EnginePlacement {
                name: "helper".to_owned(),
                kind: "fake".to_owned(),
                devices: vec![0],
                config: toml::Table::new(),
                share: 3,
                pin_identity: false,
            },
        ],
        budgets: vec![BudgetSpec {
            tiers: vec![TierSpec::from(Tier::Device(0)), TierSpec::from(Tier::Host)],
            bytes: 1000,
        }],
        store: Some(StoreSpec {
            nvme: 4096,
            host: Some(1024),
        }),
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
    assert!(profile.engines[0].pin_identity);
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
            host: Some(1024)
        })
    );
}

#[test]
fn a_profile_without_a_store_section_has_no_store_budget() {
    let text = EXAMPLE.replace("\n[store]\nnvme = 4096\nhost = 1024\n", "");
    let profile = PlacementProfile::parse(&text).unwrap();
    assert_eq!(profile.store, None);
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
    assert!(!profile.engines[1].pin_identity);
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
