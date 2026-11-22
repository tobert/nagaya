//! The Inkling 2 kind: the config of the profile, the plan with no card, the served name and the family library (spec
//! 2.4, 11.3, 11.18).
//!
//! A test that needs a model folder skips when `INKLING2_MODEL_DIR` or `INKLING2_VARLOOP_MODEL_DIR` is not set, and
//! fails then with `NAGAYA_REQUIRE_MODELS=1` (AGENTS "Tests that need a model file").

use std::path::{Path, PathBuf};

use nagaya::admission::{GroupDevice, Registry};
use nagaya::inkling2::{KIND, config, open_family, plan, served_names};
use nagaya::process::OpenedEngine;
use nagaya::serving::{FamilyRegistry, ServedName};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Engine, Tier};

const VARIABLES: [&str; 2] = ["INKLING2_MODEL_DIR", "INKLING2_VARLOOP_MODEL_DIR"];

fn folder(variable: &str) -> Option<PathBuf> {
    match std::env::var_os(variable) {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("{variable} is not set and NAGAYA_REQUIRE_MODELS is set; name the Inkling 2 model folder")
        }
        None => {
            eprintln!("skipped: {variable} is not set; set it to the Inkling 2 model folder");
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

/// The config names the folder, the slots, the block and the reserve. A key that the engine does not know is refused.
#[test]
fn the_config_gives_the_folder_the_slots_the_block_and_the_reserve() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().display();
    let parsed = config(&table(&format!("model_dir = \"{path}\"\nslots = 2"))).unwrap();
    assert_eq!(parsed.model_dir, dir.path());
    assert_eq!(
        (parsed.slots, parsed.max_block, parsed.extra_runs),
        (vec![32_768, 32_768], 512, 24),
        "the defaults of the engine: slots of 32,768 positions, a block of 512, the reserve of 3 cycles"
    );
    let set = config(&table(&format!(
        "model_dir = \"{path}\"\nslots = [2048, 32768]\nmax_block = 256\nextra_runs = 48"
    )))
    .unwrap();
    assert_eq!(
        (set.slots, set.max_block, set.extra_runs),
        (vec![2048, 32_768], 256, 48)
    );
    for (text, named) in [
        (
            format!("model_dir = \"{path}\"\nslots = 1\nextra_runs = 12"),
            "extra_runs is 12",
        ),
        (
            format!("model_dir = \"{path}\"\nslots = 1\nextra_runs = -4"),
            "`extra_runs`",
        ),
        (format!("model_dir = \"{path}\"\nslots = 1\nseq = 1"), "`seq`"),
        (format!("model_dir = \"{path}\""), "`slots`"),
        (format!("model_dir = \"{path}\"\nslots = \"8\""), "a list of positions"),
        (
            format!("model_dir = \"{path}\"\nslots = [32768, 2048]"),
            "short to long",
        ),
        (
            format!("model_dir = \"{path}\"\nslots = 1\nmax_block = 1024"),
            "max_block",
        ),
        ("slots = 1".to_owned(), "`model_dir`"),
        ("model_dir = 3\nslots = 1".to_owned(), "not a string"),
        (
            "model_dir = \"/nonexistent/inkling2\"\nslots = 1".to_owned(),
            "/nonexistent/inkling2",
        ),
    ] {
        let error = config(&table(&text)).unwrap_err();
        assert!(error.message.contains(named), "{named}: {}", error.message);
    }
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
    for variable in VARIABLES {
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
        let device: u64 = limits
            .memory
            .mandatory
            .iter()
            .filter(|term| term.tier == Tier::Device(0))
            .map(|term| term.bytes)
            .sum();
        // 18,881,368,064 bytes of weights, two slots of 3,157,950,464 bytes, and the work: 25,543,784,132 bytes.
        assert!(
            device > 25_000_000_000 && device < 32 << 30,
            "{variable}: {device} bytes"
        );
    }
}

/// The served model of an engine of the kind is its engine name, before the open (spec 11.18).
#[test]
fn the_served_name_is_the_engine_name() {
    let config = table("model_dir = \"/nonexistent\"\nslots = 1");
    assert_eq!(
        served_names("inkling2-mini", &config),
        Some(vec!["inkling2-mini".to_owned()])
    );
    assert_eq!(served_names("ink-2", &config), Some(vec!["ink-2".to_owned()]));
}

/// The opened engine `name` with the identity that the engine reports for the folder `dir`.
fn opened(name: &str, dir: &Path) -> OpenedEngine {
    let fake = FakeAlways::new().with_family("inkling2");
    let mut identity = fake.identity();
    identity.model = nagaya_mk_inkling2_bmg::spec::Model::read(dir)
        .unwrap()
        .name()
        .to_owned();
    let facts = nagaya_mk_inkling2_bmg::spec::text_facts(dir).unwrap();
    identity.tokenizer_hash = facts.tokenizer_hash;
    identity.template = facts.template;
    OpenedEngine {
        name: name.to_owned(),
        thread: std::thread::current().id(),
        identity,
        limits: fake.limits(),
        handoff: None,
    }
}

/// The family of either folder serves either model, because the two share every file of the family. The tokenizer
/// hash of the engine's identity passes the check of the registry, and another hash fails it (spec 11.3).
#[test]
fn the_family_of_either_folder_passes_the_check_of_the_registry() {
    let (Some(mini), Some(varloop)) = (folder(VARIABLES[0]), folder(VARIABLES[1])) else {
        return;
    };
    let mut registry = FamilyRegistry::new();
    registry.register("inkling2", open_family);
    for (name, engine_dir) in [("inkling2-mini", &mini), ("inkling2-varloop", &varloop)] {
        for family_dir in [&mini, &varloop] {
            let family_table = table(&format!("model_dir = \"{}\"", family_dir.display()));
            let plan = registry
                .plan(&opened(name, engine_dir), &family_table)
                .unwrap()
                .expect("the Inkling 2 family is registered");
            assert_eq!(plan.config.model, name, "the served model is the engine name");
        }
        let mut other = opened(name, engine_dir);
        other.identity.tokenizer_hash = "blake3:00".to_owned();
        let family_table = table(&format!("model_dir = \"{}\"", mini.display()));
        let Err(error) = registry.plan(&other, &family_table) else {
            panic!("a family with another tokenizer hash serves the engine");
        };
        assert!(error.contains("blake3:00"), "{error}");
    }
    let opened = open_family(
        &opened("inkling2-mini", &mini).identity,
        &table(&format!("model_dir = \"{}\"", mini.display())),
    )
    .expect("a family");
    assert_eq!(opened.served, ServedName::Engine);
    let bytes = std::fs::read(mini.join("tokenizer.json")).unwrap();
    assert_eq!(
        opened.tokenizer_hash.as_deref(),
        Some(nagaya_text_inkling2::tokenizer::sha256_of(&bytes).as_str())
    );
}

/// The family's template must be the template of the identity, so the council renders what the engine names (spec
/// 11.3). A family of another template is refused, and the error names both folders' roles.
#[test]
fn a_family_of_another_template_is_refused() {
    let Some(mini) = folder(VARIABLES[0]) else {
        return;
    };
    let mut identity = opened("inkling2-mini", &mini).identity;
    let family_table = table(&format!("model_dir = \"{}\"", mini.display()));
    assert!(open_family(&identity, &family_table).is_ok());
    identity.template.push_str("{# another #}");
    let Err(error) = open_family(&identity, &family_table) else {
        panic!("a family with another template serves the engine");
    };
    assert!(error.contains("the chat template of the family"), "{error}");
}

/// With no key `model_dir`, the family reads `INKLING2_MODEL_DIR`, as an engine on demand needs (spec 11.18). A key
/// that names no directory is refused by name.
#[test]
fn the_family_with_no_table_reads_the_variable() {
    let Some(mini) = folder(VARIABLES[0]) else {
        return;
    };
    let identity = opened("inkling2-mini", &mini).identity;
    let family = open_family(&identity, &toml::Table::new()).expect("the family of INKLING2_MODEL_DIR");
    assert!(family.tokenizer_hash.is_some_and(|hash| hash.starts_with("sha256:")));
    let Err(error) = open_family(&identity, &table("model_dir = \"/nonexistent/family\"")) else {
        panic!("a family opens from a folder that does not exist");
    };
    assert!(error.contains("/nonexistent/family"), "{error}");
}

/// The kind and the family exist in a build with the feature `xe`, and in no other build.
#[test]
fn the_kind_and_the_family_exist_only_with_the_feature_xe() {
    let registry = Registry::compiled_in();
    assert_eq!(registry.factory(KIND).is_some(), cfg!(feature = "xe"));
    assert_eq!(KIND, "mk-inkling2-bmg");
    let families = FamilyRegistry::compiled_in();
    let engine = FakeAlways::new().with_family("inkling2");
    let opened = OpenedEngine {
        name: "inkling2-mini".to_owned(),
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
            "a build with no feature xe has no inkling2 library"
        );
    }
}
