//! The mk engine as a factory kind (spec 2.4, 10.4; docs/engine-abi.md "v0 mk binding choices").
//!
//! No test here opens the engine, loads the library or touches the GPU. `plan` reads the environment and checks that
//! files exist. A test that sets the environment holds `ENV`, because the environment belongs to the whole process.
#![cfg(feature = "hip")]

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use nagaya::admission::{Factory, OpenErrorKind, Registry, admit};
use nagaya::exit::ExitCode;
use nagaya::inventory::{Device, Inventory, Probe};
use nagaya::lifecycle;
use nagaya::mk::{self, MkFactory};
use nagaya::probe::FileProbe;
use nagaya::profile::{PROFILE_FILE, PlacementProfile};
use nagaya_abi::Tier;
use nagaya_ledger::Ledger;
use nagaya_mk_qwen38_gfx1151::MkError;

static ENV: Mutex<()> = Mutex::new(());

const LIB: &str = "MK_ENGINE_LIB";
const SHARDS: &str = "MK_SHARDS";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn config(text: &str) -> toml::Table {
    text.parse().expect("a TOML table")
}

/// Set or remove a variable. The caller holds `ENV`.
fn set(name: &str, value: Option<&Path>) {
    // SAFETY: every test that reads or writes these variables holds `ENV`.
    unsafe {
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }
}

/// A library file and three shard files. They are empty: `plan` checks that they exist and reads nothing.
struct Install {
    _directory: tempfile::TempDir,
    library: PathBuf,
    shards: [PathBuf; 3],
}

impl Install {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let file = |name: &str| {
            let path = directory.path().join(name);
            std::fs::write(&path, b"").unwrap();
            path
        };
        Self {
            library: file("libmkengine.so"),
            shards: [file("one.gguf"), file("two.gguf"), file("three.gguf")],
            _directory: directory,
        }
    }

    fn export(&self) {
        let shards = self.shards.iter().map(|path| path.to_str().unwrap()).collect::<Vec<_>>();
        set(LIB, Some(&self.library));
        set(SHARDS, Some(Path::new(&shards.join(":"))));
    }
}

#[test]
fn a_hip_build_registers_the_mk_kind() {
    let registry = Registry::with_mk();
    assert!(registry.factory(mk::KIND).is_some(), "the registry has no factory for `{}`", mk::KIND);
    assert!(registry.factory("fake").is_none(), "the mk registry holds the real kind only");
}

#[test]
fn the_registered_kind_is_the_kind_in_the_gfx1151host_profile() {
    let profile = PlacementProfile::read(&repo().join("deploy/gfx1151host").join(PROFILE_FILE)).unwrap();
    assert_eq!(profile.engines[0].kind, mk::KIND);
    let registry = Registry::with_mk();
    for engine in &profile.engines {
        assert!(registry.factory(&engine.kind).is_some(), "no factory for `{}`", engine.kind);
    }
}

#[test]
fn a_missing_variable_refuses_the_start_with_exit_3_and_names_the_variable() {
    let _env = ENV.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let install = Install::new();
    let profile = PlacementProfile::read(&repo().join("deploy/gfx1151host").join(PROFILE_FILE)).unwrap();
    let inventory = Inventory {
        devices: vec![Device {
            id: 0,
            name: "gfx1151".to_owned(),
            memory_bytes: 133_143_986_176,
            unified: true,
        }],
        host_bytes: 131_156_408 * 1024,
    };
    for missing in [LIB, SHARDS] {
        install.export();
        set(missing, None);
        let store = tempfile::tempdir().unwrap();
        let error = match lifecycle::start(
            profile.clone(),
            Registry::with_mk(),
            inventory.clone(),
            store.path(),
            None,
            Duration::from_secs(5),
        ) {
            Ok(_) => panic!("the start succeeded with {missing} unset"),
            Err(error) => error,
        };
        assert_eq!(error.exit_code(), ExitCode::RefusedStart, "{error}");
        assert!(error.to_string().contains(missing), "the message does not name {missing}: {error}");
    }
}

#[test]
fn plan_refuses_a_library_or_a_shard_that_is_not_a_file() {
    let _env = ENV.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let install = Install::new();
    let table = config("weights_bytes = 1000");
    install.export();
    set(LIB, Some(&install.library.with_extension("gone")));
    let message = MkFactory.plan(&table).unwrap_err().to_string();
    assert!(message.contains(LIB) && message.contains("gone"), "{message}");
    install.export();
    std::fs::remove_file(&install.shards[1]).unwrap();
    let message = MkFactory.plan(&table).unwrap_err().to_string();
    assert!(message.contains(SHARDS) && message.contains("two.gguf"), "{message}");
}

#[test]
fn plan_declares_the_weights_term_from_the_config() {
    let _env = ENV.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let install = Install::new();
    install.export();
    let limits = MkFactory.plan(&config("weights_bytes = 65498251264")).unwrap();
    let weights = limits
        .memory
        .mandatory
        .iter()
        .find(|term| term.name == "weights")
        .expect("a weights term");
    assert_eq!(weights.bytes, 65_498_251_264);
    assert_eq!(weights.tier, Tier::Device(0));
    assert_eq!(limits, nagaya_mk_qwen38_gfx1151::plan(65_498_251_264));
}

#[test]
fn plan_refuses_a_config_without_a_usable_weights_size_or_with_an_unknown_key() {
    let _env = ENV.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let install = Install::new();
    install.export();
    for (text, needle) in [
        ("", "weights_bytes"),
        ("weights_bytes = 0", "weights_bytes"),
        ("weights_bytes = -5", "weights_bytes"),
        ("weights_bytes = \"big\"", "weights_bytes"),
        ("weights_bytes = 1000\nslots = 4", "slots"),
    ] {
        let message = MkFactory.plan(&config(text)).unwrap_err().to_string();
        assert!(message.contains(needle), "config `{text}` gave `{message}`");
    }
}

#[test]
fn a_pin_refusal_refuses_the_start_and_any_other_open_error_is_a_crash() {
    let pin = mk::open_error(MkError::pin("the mk engine library /x has SHA-256 aa; MK_ENGINE_LIB_SHA256 pins bb"));
    assert_eq!(pin.kind, OpenErrorKind::Pin);
    assert!(pin.kind.refuses_start());
    assert!(pin.message.contains("MK_ENGINE_LIB_SHA256"));
    let other = mk::open_error(MkError::new("mk_engine_open returned no engine"));
    assert_eq!(other.kind, OpenErrorKind::Other);
    assert!(!other.kind.refuses_start());
}

#[test]
fn an_arena_that_differs_from_the_plan_refuses_the_start_and_gives_the_real_size() {
    // The ledger admitted `weights_bytes`. A real arena of another size means the ledger holds the wrong bytes.
    let planned = nagaya_mk_qwen38_gfx1151::plan(65_498_251_264);
    assert_eq!(mk::weights_problem(&planned, &planned), Ok(()));
    let opened = nagaya_mk_qwen38_gfx1151::plan(65_000_000_000);
    let error = mk::weights_problem(&planned, &opened).unwrap_err();
    assert_eq!(error.kind, OpenErrorKind::PartSize);
    assert!(error.kind.refuses_start());
    assert!(error.message.contains("65000000000"), "the message gives the real size: {}", error.message);
    assert!(error.message.contains("65498251264"), "the message gives the planned size: {}", error.message);
    assert!(error.message.contains("weights_bytes"), "{}", error.message);
}

#[test]
fn the_gfx1151host_profile_admits_on_the_probed_gfx1151host_with_the_mk_factory() {
    // The whole start path up to the open: the probe, the plan and the ledger. Nothing here opens the engine.
    let _env = ENV.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let install = Install::new();
    install.export();
    let profile = PlacementProfile::read(&repo().join("deploy/gfx1151host").join(PROFILE_FILE)).unwrap();
    let inventory = FileProbe::new(repo().join("crates/nagaya/tests/fixtures/probe/gfx1151host"))
        .probe()
        .unwrap();
    let admitted = admit(&profile, &Registry::with_mk(), &inventory, &mut Ledger::new()).expect("gfx1151host admits");
    assert_eq!(admitted.engines.len(), 1);
    assert_eq!(admitted.engines[0].name, "mk");
}
