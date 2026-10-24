//! The admission of engines on demand (spec 3.3a): only the resident engines take memory at start, and each on-demand
//! engine must fit beside the resident engines of the other device groups.

use nagaya::admission::{AdmitError, Factory, GroupDevice, OpenError, PlanError, Registry, admit};
use nagaya::inventory::{Device, Inventory};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Capability, Engine, Lifetime, Limits, MandatoryTerm, Tier};
use nagaya_ledger::{Ledger, LedgerError, Refusal};

/// An engine whose weights take `bytes` of its first device, from its config. It declares `close` unless the config
/// has `close = false`.
struct Weights;

fn weights_limits(config: &toml::Table) -> Limits {
    let mut limits = FakeAlways::new().limits();
    let bytes = config.get("bytes").and_then(toml::Value::as_integer).unwrap_or(0) as u64;
    limits.memory.mandatory = vec![MandatoryTerm {
        name: "weights".to_owned(),
        tier: Tier::Device(0),
        bytes,
        lifetime: Lifetime::Engine,
        kinds: Vec::new(),
    }];
    limits.memory.elastic = Vec::new();
    if config.get("close").and_then(toml::Value::as_bool) != Some(false) {
        limits.capabilities.push(Capability::Close);
    }
    if config.get("stages").and_then(toml::Value::as_bool) == Some(true) {
        limits.capabilities.push(Capability::Stages);
    }
    limits
}

impl Factory for Weights {
    fn plan(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(weights_limits(config))
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        panic!("an admission opens no engine")
    }
}

fn registry() -> Registry {
    let mut registry = Registry::new();
    registry.register("weights", Box::new(Weights));
    registry
}

fn engine(name: &str, device: u32, bytes: u64, residency: Residency) -> EnginePlacement {
    let mut config = toml::Table::new();
    config.insert("bytes".to_owned(), toml::Value::Integer(bytes as i64));
    EnginePlacement {
        name: name.to_owned(),
        kind: "weights".to_owned(),
        devices: vec![device],
        config,
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        idle_unload_s: None,
    }
}

fn profile(engines: Vec<EnginePlacement>) -> PlacementProfile {
    PlacementProfile {
        engines,
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: 1000,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(1))],
                bytes: 1000,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes: 1000,
            },
        ],
        store: None,
    }
}

fn inventory() -> Inventory {
    Inventory {
        devices: (0..2)
            .map(|id| Device {
                id,
                name: format!("device-{id}"),
                pci_address: format!("0000:0{id}:00.0"),
                memory_bytes: 1000,
                unified: false,
                ignored: false,
            })
            .collect(),
        host_bytes: 1000,
    }
}

fn refused(profile: &PlacementProfile) -> (AdmitError, Ledger, Ledger) {
    let mut ledger = Ledger::new();
    let before = ledger.clone();
    let error = admit(profile, &registry(), &inventory(), &mut ledger).unwrap_err();
    (error, before, ledger)
}

#[test]
fn an_on_demand_engine_takes_no_memory_at_start() {
    let profile = profile(vec![
        engine("chat", 0, 600, Residency::Resident),
        engine("voice", 0, 600, Residency::OnDemand),
    ]);
    let mut ledger = Ledger::new();
    let admitted = admit(&profile, &registry(), &inventory(), &mut ledger).unwrap();
    assert_eq!(
        ledger.unassigned()[0],
        (vec![Tier::Device(0)], 400),
        "only the resident engine is charged"
    );
    let names: Vec<&str> = admitted.engines.iter().map(|engine| engine.name.as_str()).collect();
    assert_eq!(
        names,
        ["chat", "voice"],
        "the admission plans every engine, in profile order"
    );
}

#[test]
fn an_on_demand_engine_that_does_not_fit_alone_is_refused_with_its_name_and_changes_nothing() {
    let profile = profile(vec![
        engine("chat", 0, 600, Residency::Resident),
        engine("voice", 0, 1200, Residency::OnDemand),
    ]);
    let (error, before, after) = refused(&profile);
    assert_eq!(after, before);
    match &error {
        AdmitError::OnDemand { engine, error } => {
            assert_eq!(engine, "voice");
            assert_eq!(
                error,
                &LedgerError::Refused(Refusal {
                    shortfall: vec![(vec![Tier::Device(0)], 200)],
                })
            );
        }
        other => panic!("expected the refusal of `voice`, got {other:?}"),
    }
    assert!(error.to_string().contains("voice"), "{error}");
}

#[test]
fn an_on_demand_engine_must_fit_beside_the_resident_engines_of_other_groups() {
    let mut chat = engine("chat", 0, 0, Residency::Resident);
    let mut voice = engine("voice", 1, 0, Residency::OnDemand);
    // Both engines take host memory, and they are in two device groups.
    for (placement, bytes) in [(&mut chat, 600), (&mut voice, 600)] {
        placement.config.insert("bytes".to_owned(), toml::Value::Integer(0));
        placement.config.insert("host".to_owned(), toml::Value::Integer(bytes));
    }
    struct Host;
    impl Factory for Host {
        fn plan(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
            let mut limits = weights_limits(config);
            limits.memory.mandatory[0].tier = Tier::Host;
            limits.memory.mandatory[0].bytes = config["host"].as_integer().unwrap() as u64;
            Ok(limits)
        }
        fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            panic!("an admission opens no engine")
        }
    }
    chat.kind = "host".to_owned();
    voice.kind = "host".to_owned();
    let mut registry = Registry::new();
    registry.register("host", Box::new(Host));
    let mut ledger = Ledger::new();
    let error = admit(&profile(vec![chat, voice]), &registry, &inventory(), &mut ledger).unwrap_err();
    assert!(
        matches!(&error, AdmitError::OnDemand { engine, .. } if engine == "voice"),
        "{error:?}"
    );
    assert_eq!(ledger, Ledger::new());
}

#[test]
fn an_on_demand_engine_unloads_the_engines_of_its_own_group_in_the_worst_case() {
    // The two engines do not fit together, and they share device 0. The swap unloads `chat` for `voice`.
    let profile = profile(vec![
        engine("chat", 0, 900, Residency::Resident),
        engine("voice", 0, 900, Residency::OnDemand),
    ]);
    let mut ledger = Ledger::new();
    admit(&profile, &registry(), &inventory(), &mut ledger).unwrap();
}

#[test]
fn an_on_demand_engine_needs_close() {
    let mut voice = engine("voice", 0, 100, Residency::OnDemand);
    voice.config.insert("close".to_owned(), toml::Value::Boolean(false));
    let (error, before, after) = refused(&profile(vec![voice]));
    assert_eq!(after, before);
    assert!(
        matches!(&error, AdmitError::Residency { engine, .. } if engine == "voice"),
        "{error:?}"
    );
    assert!(error.to_string().contains("close"), "{error}");
}

#[test]
fn each_engine_of_the_group_of_an_on_demand_engine_needs_close() {
    let mut chat = engine("chat", 0, 100, Residency::Resident);
    chat.config.insert("close".to_owned(), toml::Value::Boolean(false));
    let voice = engine("voice", 0, 100, Residency::OnDemand);
    let (error, before, after) = refused(&profile(vec![chat, voice]));
    assert_eq!(after, before);
    assert!(
        matches!(&error, AdmitError::Residency { engine, .. } if engine == "chat"),
        "{error:?}"
    );
}

#[test]
fn an_engine_without_close_in_another_group_does_not_matter() {
    let mut chat = engine("chat", 1, 100, Residency::Resident);
    chat.config.insert("close".to_owned(), toml::Value::Boolean(false));
    let voice = engine("voice", 0, 100, Residency::OnDemand);
    let mut ledger = Ledger::new();
    admit(&profile(vec![chat, voice]), &registry(), &inventory(), &mut ledger).unwrap();
}

#[test]
fn an_on_demand_engine_serves_no_council_model() {
    let mut voice = engine("voice", 0, 100, Residency::OnDemand);
    voice
        .family
        .insert("model_dir".to_owned(), toml::Value::String("/models/voice".to_owned()));
    let (error, before, after) = refused(&profile(vec![voice]));
    assert_eq!(after, before);
    assert!(
        matches!(&error, AdmitError::Residency { engine, .. } if engine == "voice"),
        "{error:?}"
    );
    assert!(error.to_string().contains("family"), "{error}");
}

#[test]
fn an_idle_time_on_a_resident_engine_is_refused() {
    let mut chat = engine("chat", 0, 100, Residency::Resident);
    chat.idle_unload_s = Some(30);
    let (error, before, after) = refused(&profile(vec![chat]));
    assert_eq!(after, before);
    assert!(
        matches!(&error, AdmitError::Residency { engine, .. } if engine == "chat"),
        "{error:?}"
    );
    assert!(error.to_string().contains("idle_unload_s"), "{error}");
}

#[test]
fn an_idle_time_of_zero_is_refused() {
    let mut voice = engine("voice", 0, 100, Residency::OnDemand);
    voice.idle_unload_s = Some(0);
    let (error, _, _) = refused(&profile(vec![voice]));
    assert!(
        matches!(&error, AdmitError::Residency { engine, .. } if engine == "voice"),
        "{error:?}"
    );
}

#[test]
fn an_engine_with_stages_shares_its_group_only_with_on_demand_engines() {
    let mut gemma = engine("gemma", 0, 100, Residency::Resident);
    gemma.config.insert("stages".to_owned(), toml::Value::Boolean(true));
    let helper = engine("helper", 0, 100, Residency::Resident);
    let (error, before, after) = refused(&profile(vec![gemma.clone(), helper]));
    assert_eq!(after, before);
    assert!(
        matches!(&error, AdmitError::Residency { engine, .. } if engine == "helper"),
        "{error:?}"
    );
    assert!(error.to_string().contains("stages"), "{error}");
    // With the other engine on demand, the profile is admitted: a swap unloads one for the other.
    let voice = engine("voice", 0, 100, Residency::OnDemand);
    let mut ledger = Ledger::new();
    admit(&profile(vec![gemma, voice]), &registry(), &inventory(), &mut ledger).unwrap();
}
