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
    if config.get("image").and_then(toml::Value::as_bool) == Some(true) {
        limits.capabilities.push(Capability::Image);
    }
    if config.get("wide").and_then(toml::Value::as_bool) == Some(true) {
        limits.devices = vec![0, 1];
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
        pinned: false,
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

/// A text model on demand is not built: its service attaches only to an engine that opens at the start (spec 3.3a,
/// "Council models on demand"). Its family table would name a served model that never comes, so it is refused.
#[test]
fn an_on_demand_text_engine_takes_no_family_table() {
    let mut chat = engine("chat", 0, 100, Residency::OnDemand);
    chat.family
        .insert("model_dir".to_owned(), toml::Value::String("/models/chat".to_owned()));
    let (error, before, after) = refused(&profile(vec![chat]));
    assert_eq!(after, before);
    assert!(
        matches!(&error, AdmitError::Residency { engine, .. } if engine == "chat"),
        "{error:?}"
    );
    assert!(
        error.to_string().contains("family") && error.to_string().contains("text model"),
        "{error}"
    );
}

/// An image engine builds its service from the profile, so its served model exists on demand, and its family table
/// gives the defaults of that model, such as `clip_skip` (spec 11.9a).
#[test]
fn an_on_demand_image_engine_takes_its_family_table() {
    let mut pony = engine("pony", 0, 100, Residency::OnDemand);
    pony.config.insert("image".to_owned(), toml::Value::Boolean(true));
    pony.family.insert("clip_skip".to_owned(), toml::Value::Integer(2));
    admit(&profile(vec![pony]), &registry(), &inventory(), &mut Ledger::new())
        .expect("an image engine on demand with a family table");
}

/// A pinned engine is resident (spec 3.3a, "Pins"), so a pin on an on-demand engine is refused.
#[test]
fn a_pin_on_an_on_demand_engine_is_refused() {
    let mut voice = engine("voice", 0, 100, Residency::OnDemand);
    voice.pinned = true;
    let (error, before, after) = refused(&profile(vec![voice]));
    assert_eq!(after, before);
    assert!(
        matches!(&error, AdmitError::Residency { engine, .. } if engine == "voice"),
        "{error:?}"
    );
    assert!(error.to_string().contains("pinned"), "{error}");
}

#[test]
fn a_pin_on_a_resident_engine_is_admitted() {
    let mut chat = engine("chat", 0, 100, Residency::Resident);
    chat.pinned = true;
    admit(&profile(vec![chat]), &registry(), &inventory(), &mut Ledger::new()).expect("a pinned resident engine");
}

/// No swap unloads an engine by itself now (spec 3.3a, "Explicit layout"), so a resident engine can share its group
/// with a resident engine with stages. Both must fit together, as every resident engine must.
#[test]
fn a_resident_engine_shares_its_group_with_a_resident_engine_with_stages() {
    let mut gemma = engine("gemma", 0, 100, Residency::Resident);
    gemma.config.insert("stages".to_owned(), toml::Value::Boolean(true));
    let mut helper = engine("helper", 0, 100, Residency::Resident);
    helper.pinned = true;
    admit(
        &profile(vec![gemma.clone(), helper]),
        &registry(),
        &inventory(),
        &mut Ledger::new(),
    )
    .expect("two resident engines that fit together");
    let too_big = engine("big", 0, 950, Residency::Resident);
    let (error, _, _) = refused(&profile(vec![gemma, too_big]));
    assert!(matches!(&error, AdmitError::Ledger(_)), "{error:?}");
}

/// A device group holds one engine with stages at a time (spec 3.2), so two resident engines with stages in one group
/// are refused at admission. They both opened at start before, and the serve loop then broke. An on-demand engine with
/// stages beside a resident one is admitted: its load names the resident one in `unload`.
#[test]
fn two_resident_engines_with_stages_in_one_group_are_refused() {
    let with_stages = |name: &str, residency: Residency| {
        let mut placement = engine(name, 0, 100, residency);
        placement.config.insert("stages".to_owned(), toml::Value::Boolean(true));
        placement
    };
    let (error, before, after) = refused(&profile(vec![
        with_stages("gemma", Residency::Resident),
        with_stages("paint", Residency::Resident),
    ]));
    match &error {
        AdmitError::Residency { engine, why } => {
            assert_eq!(engine, "paint");
            assert!(why.contains("`gemma`"), "{why}");
        }
        other => panic!("two resident engines with stages in one group: {other:?}"),
    }
    assert_eq!(before, after, "a refusal changes nothing");
    admit(
        &profile(vec![
            with_stages("gemma", Residency::Resident),
            with_stages("paint", Residency::OnDemand),
        ]),
        &registry(),
        &inventory(),
        &mut Ledger::new(),
    )
    .expect("an on-demand engine with stages beside a resident one");
}

/// An engine with stages spans its whole group, with one stage on each device in order (spec 3.2). Another shape is
/// refused at admission. It stopped the serve loop with a panic before.
#[test]
fn an_engine_with_stages_that_does_not_span_its_group_is_refused() {
    let mut gemma = engine("gemma", 0, 100, Residency::Resident);
    gemma.config.insert("stages".to_owned(), toml::Value::Boolean(true));
    let mut wide = engine("wide", 0, 100, Residency::Resident);
    wide.devices = vec![0, 1];
    wide.config.insert("wide".to_owned(), toml::Value::Boolean(true));
    let (error, _, _) = refused(&profile(vec![gemma, wide]));
    match &error {
        AdmitError::Residency { engine, why } => {
            assert_eq!(engine, "gemma");
            assert!(why.contains("[0, 1]"), "{why}");
        }
        other => panic!("an engine with stages on device 0 of a group of devices 0 and 1: {other:?}"),
    }
}

#[test]
fn an_on_demand_engine_takes_no_share_of_the_spare_memory_at_start() {
    struct Elastic;
    impl Factory for Elastic {
        fn plan(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
            let mut limits = weights_limits(config);
            limits.memory.elastic = vec![nagaya_abi::ElasticTerm {
                name: "cache".to_owned(),
                tier: Tier::Device(0),
                unit_bytes: 1,
                // The ledger divides the spare bytes in proportion to the minimums (spec 4.4).
                minimum: 1,
                maximum: 1 << 20,
                kinds: Vec::new(),
            }];
            Ok(limits)
        }
        fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
            panic!("an admission opens no engine")
        }
    }
    let mut chat = engine("chat", 0, 200, Residency::Resident);
    let mut voice = engine("voice", 0, 200, Residency::OnDemand);
    chat.kind = "elastic".to_owned();
    voice.kind = "elastic".to_owned();
    let mut registry = Registry::new();
    registry.register("elastic", Box::new(Elastic));
    let mut ledger = Ledger::new();
    let admitted = admit(&profile(vec![chat, voice]), &registry, &inventory(), &mut ledger).unwrap();
    let chat = admitted
        .admission
        .engines
        .iter()
        .find(|engine| engine.name == "chat")
        .unwrap();
    // Device 0 has 1000 bytes. `chat` takes 200 of them, and its cache takes the other 800.
    assert_eq!(chat.elastic, vec![("cache".to_owned(), 800)]);
}
