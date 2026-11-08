//! The actions of a profile change (spec 3.3, "A profile change in place" and "Which actions run in place").

use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya::reprofile::{Action, Change, change};
use nagaya_abi::Tier;

fn engine(name: &str, residency: Residency) -> EnginePlacement {
    let mut config = toml::Table::new();
    config.insert("bytes".to_owned(), toml::Value::Integer(600));
    EnginePlacement {
        name: name.to_owned(),
        kind: "counted".to_owned(),
        devices: vec![0],
        config,
        family: toml::Table::new(),
        lab: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        pinned: false,
    }
}

fn profile(engines: Vec<EnginePlacement>) -> PlacementProfile {
    PlacementProfile {
        engines,
        budgets: vec![BudgetSpec {
            tiers: vec![TierSpec::from(Tier::Device(0))],
            bytes: 1000,
        }],
        store: Some(StoreSpec {
            nvme: 1 << 20,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
    }
}

fn active() -> PlacementProfile {
    profile(vec![
        engine("chat", Residency::Resident),
        engine("voice", Residency::OnDemand),
    ])
}

/// Every engine declares `close`.
fn closes(_: &str) -> bool {
    true
}

fn actions(change: &Change) -> Vec<(&str, Action)> {
    change
        .actions
        .iter()
        .map(|entry| (entry.engine.as_str(), entry.action))
        .collect()
}

#[test]
fn an_equal_profile_keeps_each_engine_in_place() {
    let change = change(&active(), &active(), closes);
    assert_eq!(actions(&change), [("chat", Action::Keep), ("voice", Action::Keep)]);
    assert!(!change.restart);
}

#[test]
fn a_new_share_and_new_budgets_keep_the_engine_in_place() {
    let mut new = active();
    new.engines[0].share = 3;
    new.budgets[0].bytes = 2000;
    let change = change(&active(), &new, closes);
    assert_eq!(actions(&change), [("chat", Action::Keep), ("voice", Action::Keep)]);
    assert!(!change.restart);
}

#[test]
fn a_new_residency_or_pin_is_a_residency_action_in_place() {
    let mut new = active();
    new.engines[0].residency = Residency::OnDemand;
    new.engines[1].pinned = true;
    let change = change(&active(), &new, closes);
    assert_eq!(
        actions(&change),
        [("chat", Action::Residency), ("voice", Action::Residency)]
    );
    assert!(!change.restart);
}

#[test]
fn a_gone_entry_unloads_in_place_when_it_closes() {
    let new = profile(vec![engine("chat", Residency::Resident)]);
    let change = change(&active(), &new, closes);
    assert_eq!(actions(&change), [("chat", Action::Keep), ("voice", Action::Unload)]);
    assert!(!change.restart);
}

#[test]
fn a_gone_entry_without_close_restarts() {
    let new = profile(vec![engine("voice", Residency::OnDemand)]);
    let change = change(&active(), &new, |name| name != "chat");
    assert_eq!(actions(&change), [("voice", Action::Keep), ("chat", Action::Unload)]);
    assert!(change.restart);
}

#[test]
fn a_new_entry_is_a_load_and_restarts() {
    let mut new = active();
    new.engines.push(engine("paint", Residency::OnDemand));
    let change = change(&active(), &new, closes);
    assert_eq!(
        actions(&change),
        [("chat", Action::Keep), ("voice", Action::Keep), ("paint", Action::Load)]
    );
    assert!(change.restart);
}

#[test]
fn each_change_of_kind_devices_config_family_lab_or_pin_is_a_reload_and_restarts() {
    let edits: [fn(&mut EnginePlacement); 6] = [
        |e| e.kind = "other".to_owned(),
        |e| e.devices = vec![1],
        |e| {
            e.config.insert("bytes".to_owned(), toml::Value::Integer(700));
        },
        |e| {
            e.family
                .insert("chat_template".to_owned(), toml::Value::String("x".to_owned()));
        },
        |e| e.pin_identity = true,
        // The lab reads the lens of the profile, so a new lens reloads the engine (docs/lab.md, "The rules of the
        // J-Lens").
        |e| {
            e.lab.insert(
                "jlens".to_owned(),
                toml::Value::String("/lenses/a.safetensors".to_owned()),
            );
        },
    ];
    for (index, edit) in edits.iter().enumerate() {
        let mut new = active();
        edit(&mut new.engines[1]);
        // A reload outranks a residency change of the same entry.
        new.engines[1].residency = Residency::Resident;
        let change = change(&active(), &new, closes);
        assert_eq!(
            actions(&change),
            [("chat", Action::Keep), ("voice", Action::Reload)],
            "edit {index}"
        );
        assert!(change.restart, "edit {index}");
    }
}

#[test]
fn a_new_store_table_restarts() {
    let mut new = active();
    new.store.as_mut().unwrap().nvme = 1 << 21;
    let change = change(&active(), &new, closes);
    assert_eq!(actions(&change), [("chat", Action::Keep), ("voice", Action::Keep)]);
    assert!(change.restart);
}

#[test]
fn an_action_is_written_as_its_word() {
    let words: Vec<String> = [
        Action::Keep,
        Action::Residency,
        Action::Load,
        Action::Unload,
        Action::Reload,
    ]
    .iter()
    .map(|action| serde_json::to_string(action).unwrap())
    .collect();
    assert_eq!(
        words,
        ["\"keep\"", "\"residency\"", "\"load\"", "\"unload\"", "\"reload\""]
    );
}
