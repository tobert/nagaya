//! Host tests for the `mk-inkling2-sm121` factory, config validation, and `cuda` registry wiring.

use std::path::PathBuf;

use nagaya::admission::{GroupDevice, Registry};
use nagaya::inkling2_sm121::{self, KIND};
use nagaya::process::OpenedEngine;
use nagaya::serving::FamilyRegistry;
use nagaya_abi::Engine;
use nagaya_abi::fake::FakeAlways;

fn device(id: u32, pci_address: &str) -> GroupDevice {
    GroupDevice {
        id,
        name: "NVIDIA GB10".to_owned(),
        pci_address: pci_address.to_owned(),
        memory_bytes: 128 << 30,
    }
}

fn one_device() -> Vec<GroupDevice> {
    vec![device(0, "0000:0a:00.0")]
}

#[test]
fn config_refuses_unknown_key_missing_keys_and_two_devices() {
    let dir = tempfile::tempdir().unwrap();
    let mut table = toml::Table::new();
    table.insert(
        "model_dir".to_owned(),
        toml::Value::String(dir.path().display().to_string()),
    );
    table.insert("slots".to_owned(), toml::Value::Integer(1));
    table.insert("bogus".to_owned(), toml::Value::Integer(1));
    let err = inkling2_sm121::config(&table).unwrap_err().message;
    assert!(err.contains("bogus"), "{err}");

    table.remove("bogus");
    assert!(inkling2_sm121::config(&table).is_ok());

    let two = vec![device(0, "0000:0a:00.0"), device(1, "0000:0b:00.0")];
    let dev_err = inkling2_sm121::plan(&table, &two).unwrap_err().message;
    assert!(dev_err.contains("one GPU"), "{dev_err}");

    table.remove("slots");
    assert!(inkling2_sm121::config(&table).is_err());

    let mut no_dir = toml::Table::new();
    no_dir.insert("slots".to_owned(), toml::Value::Integer(1));
    assert!(inkling2_sm121::config(&no_dir).is_err());
}

#[test]
fn config_reads_the_cache_format_and_refuses_a_bad_one() {
    let dir = tempfile::tempdir().unwrap();
    let mut table = toml::Table::new();
    table.insert(
        "model_dir".to_owned(),
        toml::Value::String(dir.path().display().to_string()),
    );
    table.insert("slots".to_owned(), toml::Value::Integer(1));
    let cache = |table: &toml::Table| {
        inkling2_sm121::config(table).map(|config| nagaya_mk_inkling2_sm121::spec::cache_name(config.cache))
    };
    assert_eq!(cache(&table).unwrap(), "f32", "the default is the oracle form");
    for name in ["f32", "bf16", "e4m3"] {
        table.insert("cache".to_owned(), toml::Value::String(name.to_owned()));
        assert_eq!(cache(&table).unwrap(), name);
    }
    table.insert("cache".to_owned(), toml::Value::String("fp8".to_owned()));
    let error = cache(&table).unwrap_err().message;
    assert!(error.contains("`fp8`") && error.contains("`bf16`"), "{error}");
    table.insert("cache".to_owned(), toml::Value::Integer(16));
    let error = cache(&table).unwrap_err().message;
    assert!(error.contains("`cache`") && error.contains("16"), "{error}");
}

#[cfg(feature = "cuda")]
#[test]
fn open_says_which_milestone_steps_are_missing() {
    use nagaya::admission::Factory;

    let dir = tempfile::tempdir().unwrap();
    let mut table = toml::Table::new();
    table.insert(
        "model_dir".to_owned(),
        toml::Value::String(dir.path().display().to_string()),
    );
    table.insert("slots".to_owned(), toml::Value::Integer(1));
    let error = match inkling2_sm121::Inkling2Sm121Factory.open(&table, &one_device()) {
        Ok(_) => panic!("open must refuse until the engine exists"),
        Err(error) => error.message,
    };
    assert!(error.contains("step 9"), "{error}");
    assert!(error.contains("step 11"), "{error}");
    assert!(!error.contains("step 8"), "{error}");
}

#[test]
fn served_names_and_compiled_in_registries_match_features() {
    let empty = toml::Table::new();
    assert_eq!(
        inkling2_sm121::served_names("inkling2-mini", &empty),
        Some(vec!["inkling2-mini".to_owned()])
    );
    let registry = Registry::compiled_in();
    assert_eq!(registry.factory(KIND).is_some(), cfg!(feature = "cuda"));
    let families = FamilyRegistry::compiled_in();
    let engine = FakeAlways::new().with_family(nagaya_mk_inkling2_sm121::spec::FAMILY);
    let opened = OpenedEngine {
        name: "inkling2-mini".to_owned(),
        thread: std::thread::current().id(),
        identity: engine.identity(),
        limits: engine.limits(),
        handoff: None,
    };
    let mut bad_family = toml::Table::new();
    bad_family.insert("model_dir".to_owned(), toml::Value::String("/nonexistent".to_owned()));
    let planned = families.plan(&opened, &bad_family);
    if cfg!(any(feature = "xe", feature = "cuda")) {
        assert!(planned.is_err(), "inkling2 family opener reads the table");
    } else {
        assert!(matches!(planned, Ok(None)));
    }
}

#[test]
fn real_model_plans_512k_slot_through_factory() {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    let Some(path) = std::env::var_os("INKLING2_MODEL_DIR") else {
        if require {
            panic!("INKLING2_MODEL_DIR is not set and NAGAYA_REQUIRE_MODELS is set");
        }
        eprintln!("skipping inkling2_sm121_factory: INKLING2_MODEL_DIR is not set");
        return;
    };
    let dir = PathBuf::from(path);
    if !dir.exists() {
        if require {
            panic!("INKLING2_MODEL_DIR points to absent path `{}`", dir.display());
        }
        eprintln!("skipping inkling2_sm121_factory: `{}` is absent", dir.display());
        return;
    }
    let mut table = toml::Table::new();
    table.insert("model_dir".to_owned(), toml::Value::String(dir.display().to_string()));
    table.insert(
        "slots".to_owned(),
        toml::Value::Array(vec![toml::Value::Integer(524_288)]),
    );
    let lim = inkling2_sm121::plan(&table, &one_device()).unwrap();
    lim.check().unwrap();
    let kv = lim.memory.mandatory.iter().find(|t| t.name == "kv").unwrap().bytes;
    assert_eq!(kv, 34_866_888_704);
    assert_eq!(lim.slots.len(), 1);
    assert_eq!((lim.slots[0].tokens, lim.slots[0].bytes), (524_288, 34_866_888_704));
}
