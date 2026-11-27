//! The registry of family libraries (spec 11.3, "The engine registry").

use std::sync::Arc;

use nagaya::process::OpenedEngine;
use nagaya::serving::{FamilyRegistry, OpenedFamily, ServedName};
use nagaya_abi::SlotLimit;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Engine, Identity};
use nagaya_family_toy::ToyFamily;

fn opened(family: &str, slots: Vec<SlotLimit>) -> OpenedEngine {
    let engine = FakeAlways::with_slots(slots).with_family(family);
    OpenedEngine {
        name: "engine-one".to_owned(),
        thread: std::thread::current().id(),
        identity: engine.identity(),
        limits: engine.limits(),
        handoff: None,
    }
}

fn toy_opener(_: &Identity, _: &toml::Table) -> Result<OpenedFamily, String> {
    Ok(OpenedFamily {
        family: Arc::new(ToyFamily::new()),
        tokenizer_hash: None,
        speech: None,
        served: ServedName::Identity,
    })
}

fn slots() -> Vec<SlotLimit> {
    vec![
        SlotLimit {
            bytes: 64,
            count: 1,
            tokens: 100,
        },
        SlotLimit {
            bytes: 256,
            count: 1,
            tokens: 700,
        },
    ]
}

#[test]
fn the_identity_selects_the_family_and_the_largest_slot_sets_the_context() {
    let mut registry = FamilyRegistry::new();
    registry.register("toy", toy_opener);
    let plan = registry
        .plan(&opened("toy", slots()), &toml::Table::new())
        .unwrap()
        .expect("the toy family is registered");
    assert_eq!(plan.config.context_tokens, 700);
    // A decision reads its context in the same slots, so its limit is the context limit (spec 11.3c).
    assert_eq!(plan.config.decision_context_tokens, 700);
    assert_eq!(plan.config.model, "fake-always");
}

#[test]
fn a_family_that_names_by_engine_serves_the_engine_name() {
    // Spec 11.11, "The served name of an LFM3 engine": two engines of one checkpoint serve two names.
    let mut registry = FamilyRegistry::new();
    registry.register("toy", |identity, table| {
        Ok(OpenedFamily {
            served: ServedName::Engine,
            ..toy_opener(identity, table)?
        })
    });
    let plan = registry
        .plan(&opened("toy", slots()), &toml::Table::new())
        .unwrap()
        .expect("the toy family is registered");
    assert_eq!(plan.config.model, "engine-one", "not the `model` of the identity");
    assert_eq!(plan.config.engine, "engine-one");
}

#[test]
fn a_family_with_no_library_is_not_served() {
    let mut registry = FamilyRegistry::new();
    registry.register("toy", toy_opener);
    assert!(
        registry
            .plan(&opened("gemma4", slots()), &toml::Table::new())
            .unwrap()
            .is_none()
    );
}

#[test]
fn the_opener_gets_the_table_of_the_profile() {
    let mut registry = FamilyRegistry::new();
    registry.register("toy", |identity, table| {
        assert_eq!(table.get("gguf").and_then(|value| value.as_str()), Some("/x.gguf"));
        toy_opener(identity, table)
    });
    let table: toml::Table = "gguf = \"/x.gguf\"".parse().unwrap();
    registry.plan(&opened("toy", slots()), &table).unwrap().unwrap();
}

#[test]
fn a_tokenizer_hash_that_the_identity_does_not_have_refuses_the_family() {
    let mut registry = FamilyRegistry::new();
    registry.register("toy", |_, _| {
        Ok(OpenedFamily {
            family: Arc::new(ToyFamily::new()),
            tokenizer_hash: Some("sha256:other".to_owned()),
            speech: None,
            served: ServedName::Identity,
        })
    });
    let error = registry
        .plan(&opened("toy", slots()), &toml::Table::new())
        .err()
        .expect("the hash differs");
    assert!(error.contains("sha256:other") && error.contains("fake:0"), "{error}");
}

#[test]
fn an_engine_with_no_slot_is_refused() {
    let mut registry = FamilyRegistry::new();
    registry.register("toy", toy_opener);
    let error = registry
        .plan(&opened("toy", Vec::new()), &toml::Table::new())
        .err()
        .expect("no slot");
    assert!(error.contains("slot"), "{error}");
}
