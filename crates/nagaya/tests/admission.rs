use nagaya::admission::{AdmitError, Factory, GroupDevice, OpenError, OpenErrorKind, PlanError, Registry, admit};
use nagaya::inventory::{Device, Inventory};
use nagaya::placement::DeviceError;
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Engine, Limits, Tier};
use nagaya_ledger::{Ledger, LedgerError};

fn engine(name: &str, kind: &str, share: u64) -> EnginePlacement {
    EnginePlacement {
        name: name.to_owned(),
        kind: kind.to_owned(),
        devices: vec![0],
        config: toml::Table::new(),
        share,
        pin_identity: false,
    }
}

fn host_budget(bytes: u64) -> BudgetSpec {
    BudgetSpec {
        tiers: vec![TierSpec::from(Tier::Host)],
        bytes,
    }
}

fn device_budget(device: nagaya_abi::DeviceId, bytes: u64) -> BudgetSpec {
    BudgetSpec {
        tiers: vec![TierSpec::from(Tier::Device(device))],
        bytes,
    }
}

fn inventory(ids: &[nagaya_abi::DeviceId]) -> Inventory {
    Inventory {
        devices: ids
            .iter()
            .map(|id| Device {
                id: *id,
                name: format!("device-{id}"),
                pci_address: "0000:00:00.0".to_owned(),
                memory_bytes: 1024,
                unified: false,
            })
            .collect(),
        host_bytes: 1024,
    }
}

struct FailingPlan;

impl Factory for FailingPlan {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Err(PlanError {
            message: "the weights are absent".to_owned(),
        })
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn nagaya_abi::Engine>, OpenError> {
        Err(OpenError {
            kind: OpenErrorKind::Other,
            message: "the weights are absent".to_owned(),
        })
    }
}

#[test]
fn the_admission_carries_the_planned_limits() {
    let profile = PlacementProfile {
        engines: vec![engine("fake-one", "fake", 1)],
        budgets: vec![device_budget(0, 4096), host_budget(256)],
        store: None,
    };
    let mut ledger = Ledger::new();
    let admitted = admit(&profile, &Registry::with_fake(), &inventory(&[0]), &mut ledger).unwrap();
    assert_eq!(admitted.engines.len(), 1);
    assert_eq!(admitted.engines[0].name, "fake-one");
    assert_eq!(admitted.engines[0].limits, FakeAlways::new().limits());
    assert_eq!(admitted.admission.engines.len(), 1);
}

#[test]
fn the_mandatory_bytes_must_fit() {
    let profile = PlacementProfile {
        engines: vec![engine("fits", "fake", 1)],
        budgets: vec![device_budget(0, 4096), host_budget(256)],
        store: None,
    };
    let mut ledger = Ledger::new();
    let admitted = admit(&profile, &Registry::with_fake(), &inventory(&[0]), &mut ledger).unwrap();
    assert_eq!(admitted.admission.engines[0].mandatory, vec![(Tier::Device(0), 4096)]);
    assert_eq!(
        admitted.admission.unassigned,
        vec![(vec![Tier::Device(0)], 0), (vec![Tier::Host], 0)]
    );

    let profile = PlacementProfile {
        engines: vec![engine("short", "fake", 1)],
        budgets: vec![device_budget(0, 4000), host_budget(256)],
        store: None,
    };
    let mut ledger = Ledger::new();
    let before = ledger.clone();
    let error = admit(&profile, &Registry::with_fake(), &inventory(&[0]), &mut ledger).unwrap_err();
    match error {
        AdmitError::Ledger(LedgerError::Refused(refusal)) => {
            assert_eq!(refusal.shortfall, vec![(vec![Tier::Device(0)], 96)]);
        }
        other => panic!("expected a ledger refusal, got {other:?}"),
    }
    assert_eq!(ledger, before);
}

#[test]
fn the_elastic_minimum_must_fit() {
    let profile = PlacementProfile {
        engines: vec![engine("fits", "fake", 1)],
        budgets: vec![device_budget(0, 4096), host_budget(256)],
        store: None,
    };
    let mut ledger = Ledger::new();
    let admitted = admit(&profile, &Registry::with_fake(), &inventory(&[0]), &mut ledger).unwrap();
    assert_eq!(
        admitted.admission.engines[0].elastic,
        vec![("workspace".to_owned(), 256)]
    );
    assert_eq!(
        admitted.admission.unassigned,
        vec![(vec![Tier::Device(0)], 0), (vec![Tier::Host], 0)]
    );

    let profile = PlacementProfile {
        engines: vec![engine("short", "fake", 1)],
        budgets: vec![device_budget(0, 4096), host_budget(200)],
        store: None,
    };
    let mut ledger = Ledger::new();
    let before = ledger.clone();
    let error = admit(&profile, &Registry::with_fake(), &inventory(&[0]), &mut ledger).unwrap_err();
    match error {
        AdmitError::Ledger(LedgerError::Refused(refusal)) => {
            assert_eq!(refusal.shortfall, vec![(vec![Tier::Host], 56)]);
        }
        other => panic!("expected a ledger refusal, got {other:?}"),
    }
    assert_eq!(ledger, before);
}

#[test]
fn an_unknown_kind_is_refused() {
    let profile = PlacementProfile {
        engines: vec![engine("ghost", "absent-kind", 1)],
        budgets: vec![host_budget(100)],
        store: None,
    };
    let mut ledger = Ledger::new();
    let error = admit(&profile, &Registry::new(), &inventory(&[0]), &mut ledger).unwrap_err();
    assert!(matches!(error, AdmitError::UnknownKind { .. }), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("ghost"), "{message}");
    assert!(message.contains("absent-kind"), "{message}");
}

#[test]
fn a_failed_plan_is_refused_and_names_the_engine() {
    let mut registry = Registry::new();
    registry.register("broken", Box::new(FailingPlan));
    let profile = PlacementProfile {
        engines: vec![engine("ghost", "broken", 1)],
        budgets: vec![host_budget(100)],
        store: None,
    };
    let mut ledger = Ledger::new();
    let error = admit(&profile, &registry, &inventory(&[0]), &mut ledger).unwrap_err();
    assert!(matches!(error, AdmitError::Plan { .. }), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("ghost"), "{message}");
    assert!(message.contains("the weights are absent"), "{message}");
}

#[test]
fn the_fake_engine_admits_with_its_limits() {
    let profile = PlacementProfile {
        engines: vec![engine("fake-one", "fake", 1)],
        budgets: vec![device_budget(0, 4096), host_budget(4096)],
        store: None,
    };
    let mut ledger = Ledger::new();
    let admitted = admit(&profile, &Registry::with_fake(), &inventory(&[0]), &mut ledger).unwrap();
    assert_eq!(admitted.admission.engines[0].mandatory, vec![(Tier::Device(0), 4096)]);
    assert_eq!(
        admitted.admission.engines[0].elastic,
        vec![("workspace".to_owned(), 2048)]
    );
    assert_eq!(
        admitted.admission.unassigned,
        vec![(vec![Tier::Device(0)], 0), (vec![Tier::Host], 2048)]
    );
}

#[test]
fn a_device_refusal_changes_nothing() {
    let mut placement = engine("ghost", "fake", 1);
    placement.devices = vec![7];
    let profile = PlacementProfile {
        engines: vec![placement],
        budgets: vec![host_budget(100)],
        store: None,
    };
    let mut ledger = Ledger::new();
    let before = ledger.clone();
    let error = admit(&profile, &Registry::with_fake(), &inventory(&[0]), &mut ledger).unwrap_err();
    assert!(
        matches!(error, AdmitError::Device(DeviceError::Unknown { .. })),
        "{error:?}"
    );
    assert_eq!(ledger, before);
}

struct FixedLimits(Limits);

impl Factory for FixedLimits {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(self.0.clone())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn nagaya_abi::Engine>, OpenError> {
        panic!("an engine that admission refused is never opened")
    }
}

#[test]
fn an_engine_with_no_segments_is_refused_with_its_name_and_the_field() {
    // Review finding 8: the scheduler asserts `max_segments > 0`, so admission refuses a zero first.
    let mut limits = FakeAlways::new().limits();
    limits.max_segments = 0;
    let mut registry = Registry::new();
    registry.register("zero", Box::new(FixedLimits(limits)));
    let profile = PlacementProfile {
        engines: vec![engine("no-passes", "zero", 1)],
        budgets: vec![device_budget(0, 4096), host_budget(256)],
        store: None,
    };
    let mut ledger = Ledger::new();
    let before = ledger.clone();
    let error = admit(&profile, &registry, &inventory(&[0]), &mut ledger).unwrap_err();
    match &error {
        AdmitError::Limits { engine, field, .. } => {
            assert_eq!(engine, "no-passes");
            assert_eq!(*field, "max_segments");
        }
        other => panic!("expected a limits refusal, got {other:?}"),
    }
    let text = error.to_string();
    assert!(
        text.contains("`no-passes`") && text.contains("`max_segments`"),
        "{text}"
    );
    assert_eq!(ledger, before, "a refusal changes nothing (spec 4.3)");
}

#[test]
fn one_segment_is_admitted() {
    let mut limits = FakeAlways::new().limits();
    limits.max_segments = 1;
    let mut registry = Registry::new();
    registry.register("one", Box::new(FixedLimits(limits)));
    let profile = PlacementProfile {
        engines: vec![engine("one-pass", "one", 1)],
        budgets: vec![device_budget(0, 4096), host_budget(256)],
        store: None,
    };
    let mut ledger = Ledger::new();
    admit(&profile, &registry, &inventory(&[0]), &mut ledger).unwrap();
}
