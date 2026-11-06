//! The LFM3 kind: the config of the profile, the plan with no card, and the family library (spec 2.4, 11.3, 11.11).
//!
//! A test that needs a model folder skips when `LFM3_MODEL_DIR` or `LFM3_OMNI_MODEL_DIR` is not set, and fails then
//! with `NAGAYA_REQUIRE_MODELS=1` (AGENTS "Tests that need a model file").

use std::path::PathBuf;

use nagaya::admission::{GroupDevice, Registry};
use nagaya::lfm3::{KIND, config, open_family, plan};
use nagaya::serving::FamilyRegistry;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Engine, Tier};

fn folder(variable: &str) -> Option<PathBuf> {
    match std::env::var_os(variable) {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("{variable} is not set and NAGAYA_REQUIRE_MODELS is set; name the LFM3 model folder")
        }
        None => {
            eprintln!("skipped: {variable} is not set; set it to the LFM3 model folder");
            None
        }
    }
}

fn table(text: &str) -> toml::Table {
    text.parse().expect("a TOML table")
}

fn card() -> Vec<GroupDevice> {
    vec![GroupDevice {
        id: 0,
        name: "xe".to_owned(),
        pci_address: "0000:03:00.0".to_owned(),
        memory_bytes: 32 << 30,
    }]
}

/// The config names the folder, the count of slots and the block. A key that the engine does not know is refused.
#[test]
fn the_config_gives_the_folder_the_slots_and_the_block() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().display();
    let parsed = config(&table(&format!("model_dir = \"{path}\"\nslots = 2"))).unwrap();
    assert_eq!((parsed.slots, parsed.max_block), (2, 512));
    assert_eq!(parsed.model_dir, dir.path());
    let block = config(&table(&format!("model_dir = \"{path}\"\nslots = 1\nmax_block = 256"))).unwrap();
    assert_eq!(block.max_block, 256);
    let unknown = config(&table(&format!("model_dir = \"{path}\"\nslots = 1\nseq = 1"))).unwrap_err();
    assert!(unknown.message.contains("`seq`"), "{}", unknown.message);
    let missing = config(&table(&format!("model_dir = \"{path}\""))).unwrap_err();
    assert!(missing.message.contains("`slots`"), "{}", missing.message);
    let list = config(&table(&format!("model_dir = \"{path}\"\nslots = [32768]"))).unwrap_err();
    assert!(list.message.contains("not a count"), "{}", list.message);
    let big = config(&table(&format!("model_dir = \"{path}\"\nslots = 1\nmax_block = 1024"))).unwrap_err();
    assert!(big.message.contains("max_block"), "{}", big.message);
    let none = config(&table("slots = 1")).unwrap_err();
    assert!(none.message.contains("`model_dir`"), "{}", none.message);
    let nowhere = config(&table("model_dir = \"/nonexistent/lfm3\"\nslots = 1")).unwrap_err();
    assert!(nowhere.message.contains("/nonexistent/lfm3"), "{}", nowhere.message);
}

#[test]
fn a_group_of_two_devices_is_refused() {
    let mut two = card();
    two.push(GroupDevice {
        id: 1,
        pci_address: "0000:08:00.0".to_owned(),
        ..two[0].clone()
    });
    let error = plan(&table("slots = 1"), &two).unwrap_err();
    assert!(error.message.contains("one card"), "{}", error.message);
}

/// The plan of each model reads the tensor headers with no card. Its terms fit a card of 32 GiB with two slots.
#[test]
fn the_plan_of_each_model_fits_one_card_with_no_card() {
    for variable in ["LFM3_MODEL_DIR", "LFM3_OMNI_MODEL_DIR"] {
        let Some(dir) = folder(variable) else {
            continue;
        };
        let limits = plan(
            &table(&format!("model_dir = \"{}\"\nslots = 2", dir.display())),
            &card(),
        )
        .expect("a plan");
        assert_eq!(limits.check(), Ok(()));
        assert_eq!(limits.devices, vec![0]);
        assert_eq!(limits.slots.iter().map(|class| class.tokens).max(), Some(32_768));
        let device: u64 = limits
            .memory
            .mandatory
            .iter()
            .filter(|term| term.tier == Tier::Device(0))
            .map(|term| term.bytes)
            .sum();
        assert!(device > 5 << 30 && device < 8 << 30, "{variable}: {device} bytes");
    }
}

/// The family opener reads the folder of the engine, and it refuses the folder of the other model.
#[test]
fn the_family_opens_from_the_folder_of_its_model() {
    let (Some(text), Some(omni)) = (folder("LFM3_MODEL_DIR"), folder("LFM3_OMNI_MODEL_DIR")) else {
        return;
    };
    let mut identity = FakeAlways::new().with_family("lfm3").identity();
    identity.model = "lfm3".to_owned();
    let opened = open_family(&identity, &table(&format!("model_dir = \"{}\"", text.display()))).expect("a family");
    let bytes = std::fs::read(text.join("tokenizer.json")).unwrap();
    assert_eq!(
        opened.tokenizer_hash.as_deref(),
        Some(nagaya_text_lfm3::tokenizer::sha256_of(&bytes).as_str())
    );
    let Err(other) = open_family(&identity, &table(&format!("model_dir = \"{}\"", omni.display()))) else {
        panic!("the folder of Omni opens a family for an engine of lfm3");
    };
    assert!(other.contains("lfm3-omni"), "{other}");
}

/// The kind and the family exist in a build with the feature `xe`, and in no other build.
#[test]
fn the_kind_and_the_family_exist_only_with_the_feature_xe() {
    let registry = Registry::compiled_in();
    assert_eq!(registry.factory(KIND).is_some(), cfg!(feature = "xe"));
    assert_eq!(KIND, "mk-lfm3-bmg");
    let families = FamilyRegistry::compiled_in();
    let engine = FakeAlways::new().with_family("lfm3");
    let opened = nagaya::process::OpenedEngine {
        name: "lfm3".to_owned(),
        thread: std::thread::current().id(),
        identity: engine.identity(),
        limits: engine.limits(),
        handoff: None,
    };
    let planned = families.plan(&opened, &table("model_dir = \"/nonexistent\""));
    if cfg!(feature = "xe") {
        assert!(planned.is_err(), "the opener of the build reads the table");
    } else {
        assert!(
            matches!(planned, Ok(None)),
            "a build with no feature xe has no lfm3 library"
        );
    }
}
