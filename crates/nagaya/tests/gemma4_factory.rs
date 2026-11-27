//! The Gemma 4 kind: the config of the profile, the plan with no card, and the family library (spec 2.4, 11.3).
//!
//! A test that needs the checkpoint skips when `GEMMA4_MODEL_DIR` is not set, and fails then with
//! `NAGAYA_REQUIRE_MODELS=1` (AGENTS "Tests that need a model file").

use std::path::PathBuf;

use nagaya::admission::{GroupDevice, Registry};
use nagaya::gemma4::{KIND, config, open_family, plan};
use nagaya::serving::FamilyRegistry;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Engine, Tier};

fn checkpoint() -> Option<PathBuf> {
    match std::env::var_os("GEMMA4_MODEL_DIR") {
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

fn table(text: &str) -> toml::Table {
    text.parse().expect("a TOML table")
}

fn device(id: u32, pci: &str) -> GroupDevice {
    GroupDevice {
        id,
        name: "xe".to_owned(),
        pci_address: pci.to_owned(),
        memory_bytes: 32 << 30,
    }
}

fn two_cards() -> Vec<GroupDevice> {
    vec![device(0, "0000:03:00.0"), device(1, "0000:08:00.0")]
}

/// The config names the slots, the block and the checkpoint. A key that the engine does not know is refused.
///
/// Falsified by a parse that ignores an unknown key, and by a default block other than 512.
#[test]
fn the_config_gives_the_slots_the_block_and_the_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().display();
    let parsed = config(&table(&format!("model_dir = \"{path}\"\nslots = [4096, 65536]"))).unwrap();
    assert_eq!(parsed.slots, vec![4096, 65_536]);
    assert_eq!(parsed.max_block, 512);
    assert_eq!(parsed.model_dir, dir.path());
    let block = config(&table(&format!(
        "model_dir = \"{path}\"\nslots = [4096]\nmax_block = 256"
    )))
    .unwrap();
    assert_eq!(block.max_block, 256);
    let unknown = config(&table(&format!("model_dir = \"{path}\"\nslots = [4096]\nseq = 1"))).unwrap_err();
    assert!(unknown.message.contains("`seq`"), "{}", unknown.message);
    let missing = config(&table(&format!("model_dir = \"{path}\""))).unwrap_err();
    assert!(missing.message.contains("`slots`"), "{}", missing.message);
    let big = config(&table(&format!("model_dir = \"{path}\"\nslots = [65537]"))).unwrap_err();
    assert!(big.message.contains("65537"), "{}", big.message);
    let nowhere = config(&table("model_dir = \"/nonexistent/gemma\"\nslots = [4096]")).unwrap_err();
    assert!(nowhere.message.contains("/nonexistent/gemma"), "{}", nowhere.message);
}

#[test]
fn a_group_of_one_device_is_refused() {
    let error = plan(&table("slots = [4096]"), &two_cards()[..1]).unwrap_err();
    assert!(error.message.contains("two cards"), "{}", error.message);
}

/// The plan reads the tensor headers with no card. It gives group indexes, the feed rule, and terms on both cards.
#[test]
fn the_plan_gives_the_limits_of_two_stages_with_no_card() {
    let Some(dir) = checkpoint() else {
        return;
    };
    let config = table(&format!("model_dir = \"{}\"\nslots = [4096, 65536]", dir.display()));
    let limits = plan(&config, &two_cards()).expect("a plan");
    assert_eq!(limits.check(), Ok(()));
    assert_eq!(limits.devices, vec![0, 1]);
    assert_eq!(limits.slots.iter().map(|class| class.tokens).max(), Some(65_536));
    let on = |tier: Tier| -> u64 {
        limits
            .memory
            .mandatory
            .iter()
            .filter(|term| term.tier == tier)
            .map(|term| term.bytes)
            .sum()
    };
    // Each B70 holds 32 GiB. The weights of half the model and the slots must fit each one.
    for card in 0..2 {
        let bytes = on(Tier::Device(card));
        assert!(bytes > 20 << 30 && bytes < 32 << 30, "card {card}: {bytes} bytes");
    }
}

/// The family opener reads the tokenizer of the checkpoint, and its hash is the SHA-256 of `tokenizer.json`.
#[test]
fn the_family_opens_from_the_checkpoint_with_the_hash_of_its_tokenizer() {
    let Some(dir) = checkpoint() else {
        return;
    };
    let identity = FakeAlways::new().with_family("gemma4").identity();
    let opened = open_family(&identity, &table(&format!("model_dir = \"{}\"", dir.display()))).expect("a family");
    let bytes = std::fs::read(dir.join("tokenizer.json")).unwrap();
    assert_eq!(
        opened.tokenizer_hash.as_deref(),
        Some(nagaya_family_gemma4::tokenizer::sha256_of(&bytes).as_str())
    );
}

/// The kind and the family exist in a build with the feature `xe`, and in no other build.
#[test]
fn the_kind_and_the_family_exist_only_with_the_feature_xe() {
    let registry = Registry::compiled_in();
    assert_eq!(registry.factory(KIND).is_some(), cfg!(feature = "xe"));
    let families = FamilyRegistry::compiled_in();
    let engine = FakeAlways::new().with_family("gemma4");
    let opened = nagaya::process::OpenedEngine {
        name: "gemma".to_owned(),
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
            "a build with no feature xe has no gemma4 library"
        );
    }
}

#[test]
fn the_arena_parts_are_the_slabs_of_the_engine_on_two_cards() {
    let parts = nagaya::gemma4::arena_parts(&two_cards()).expect("the parts of two cards");
    assert_eq!(parts, nagaya_mk_gemma4_bmg::spec::arena_parts());
    assert_eq!(parts.len(), 31);
    let error = nagaya::gemma4::arena_parts(&two_cards()[..1]).unwrap_err();
    assert!(error.message.contains("two cards"), "{}", error.message);
}
