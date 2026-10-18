use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{ElasticTerm, Engine, Lifetime, MandatoryTerm, MemoryLimits, Tier};
use nagaya_ledger::{Budget, EnginePlan, Ledger, LedgerError, Profile, Refusal};

fn budget(tiers: &[Tier], bytes: u64) -> Budget {
    Budget {
        tiers: tiers.to_vec(),
        bytes,
    }
}

fn mandatory(name: &str, tier: Tier, bytes: u64) -> MandatoryTerm {
    MandatoryTerm {
        name: name.to_owned(),
        tier,
        bytes,
        lifetime: Lifetime::Engine,
        kinds: Vec::new(),
    }
}

fn elastic(name: &str, tier: Tier, unit_bytes: u64, minimum: u64, maximum: u64) -> ElasticTerm {
    ElasticTerm {
        name: name.to_owned(),
        tier,
        unit_bytes,
        minimum,
        maximum,
        kinds: Vec::new(),
    }
}

fn plan(name: &str, share: u64, memory: MemoryLimits) -> EnginePlan {
    let mut limits = FakeAlways::new().limits();
    limits.memory = memory;
    EnginePlan {
        name: name.to_owned(),
        limits,
        share,
    }
}

#[test]
fn mandatory_and_minimums_must_fit() {
    let memory = MemoryLimits {
        mandatory: vec![mandatory("m", Tier::Host, 60)],
        elastic: vec![elastic("e", Tier::Host, 10, 6, 10)],
    };
    let profile = Profile {
        budgets: vec![budget(&[Tier::Host], 100)],
        engines: vec![plan("a", 1, memory)],
    };
    let mut ledger = Ledger::new();
    let error = ledger.admit(&profile).unwrap_err();
    assert_eq!(
        error,
        LedgerError::Refused(Refusal {
            shortfall: vec![(vec![Tier::Host], 20)],
        })
    );
}

#[test]
fn spare_is_divided_by_share() {
    let memory_a = MemoryLimits {
        mandatory: Vec::new(),
        elastic: vec![elastic("a", Tier::Host, 1, 1, 1000)],
    };
    let memory_b = MemoryLimits {
        mandatory: Vec::new(),
        elastic: vec![elastic("b", Tier::Host, 1, 1, 1000)],
    };
    let profile = Profile {
        budgets: vec![budget(&[Tier::Host], 1000)],
        engines: vec![plan("a", 1, memory_a), plan("b", 3, memory_b)],
    };
    let mut ledger = Ledger::new();
    let admission = ledger.admit(&profile).unwrap();
    assert_eq!(admission.engines[0].elastic, vec![("a".to_owned(), 250)]);
    assert_eq!(admission.engines[1].elastic, vec![("b".to_owned(), 749)]);
    assert_eq!(admission.unassigned, vec![(vec![Tier::Host], 1)]);
}

#[test]
fn units_round_down_and_cap() {
    let memory = MemoryLimits {
        mandatory: Vec::new(),
        elastic: vec![elastic("e", Tier::Host, 10, 1, 5)],
    };
    let profile = Profile {
        budgets: vec![budget(&[Tier::Host], 105)],
        engines: vec![plan("a", 1, memory)],
    };
    let mut ledger = Ledger::new();
    let admission = ledger.admit(&profile).unwrap();
    assert_eq!(admission.engines[0].elastic, vec![("e".to_owned(), 50)]);
    assert_eq!(admission.unassigned, vec![(vec![Tier::Host], 55)]);
}

#[test]
fn a_refusal_changes_nothing() {
    let memory = MemoryLimits {
        mandatory: vec![mandatory("m", Tier::Host, 120)],
        elastic: Vec::new(),
    };
    let profile = Profile {
        budgets: vec![budget(&[Tier::Host], 100)],
        engines: vec![plan("a", 1, memory)],
    };
    let mut ledger = Ledger::new();
    let before = ledger.clone();
    let error = ledger.admit(&profile).unwrap_err();
    assert_eq!(
        error,
        LedgerError::Refused(Refusal {
            shortfall: vec![(vec![Tier::Host], 20)],
        })
    );
    assert_eq!(ledger, before);
}

#[test]
fn each_tier_has_its_own_shortfall() {
    let memory = MemoryLimits {
        mandatory: vec![mandatory("h", Tier::Host, 120), mandatory("d", Tier::Device(0), 50)],
        elastic: Vec::new(),
    };
    let profile = Profile {
        budgets: vec![budget(&[Tier::Host], 100), budget(&[Tier::Device(0)], 100)],
        engines: vec![plan("a", 1, memory)],
    };
    let mut ledger = Ledger::new();
    let error = ledger.admit(&profile).unwrap_err();
    assert_eq!(
        error,
        LedgerError::Refused(Refusal {
            shortfall: vec![(vec![Tier::Host], 20)],
        })
    );
}

#[test]
fn the_fake_limits_admit() {
    let limits = FakeAlways::new().limits();
    let profile = Profile {
        budgets: vec![budget(&[Tier::Host], 100)],
        engines: vec![EnginePlan {
            name: "fake".to_owned(),
            limits,
            share: 1,
        }],
    };
    let mut ledger = Ledger::new();
    let admission = ledger.admit(&profile).unwrap();
    assert_eq!(admission.engines[0].mandatory, Vec::new());
    assert_eq!(admission.engines[0].elastic, Vec::new());
    assert_eq!(admission.unassigned, vec![(vec![Tier::Host], 100)]);
}

#[test]
fn a_term_without_a_budget_is_invalid() {
    let memory = MemoryLimits {
        mandatory: vec![mandatory("m", Tier::Device(1), 10)],
        elastic: Vec::new(),
    };
    let profile = Profile {
        budgets: vec![budget(&[Tier::Host], 100)],
        engines: vec![plan("a", 1, memory)],
    };
    let mut ledger = Ledger::new();
    assert!(matches!(ledger.admit(&profile), Err(LedgerError::Invalid(_))));
}

#[test]
fn a_pooled_budget_reports_one_shortfall() {
    let memory = MemoryLimits {
        mandatory: vec![mandatory("m", Tier::Device(0), 150)],
        elastic: Vec::new(),
    };
    let profile = Profile {
        budgets: vec![budget(&[Tier::Device(0), Tier::Host], 100)],
        engines: vec![plan("a", 1, memory)],
    };
    let mut ledger = Ledger::new();
    let error = ledger.admit(&profile).unwrap_err();
    assert_eq!(
        error,
        LedgerError::Refused(Refusal {
            shortfall: vec![(vec![Tier::Device(0), Tier::Host], 50)],
        })
    );
}

#[test]
fn two_short_budgets_report_two_entries() {
    let memory = MemoryLimits {
        mandatory: vec![mandatory("h", Tier::Host, 120), mandatory("d", Tier::Device(1), 130)],
        elastic: Vec::new(),
    };
    let profile = Profile {
        budgets: vec![
            budget(&[Tier::Host, Tier::Device(0)], 100),
            budget(&[Tier::Device(1)], 100),
        ],
        engines: vec![plan("a", 1, memory)],
    };
    let mut ledger = Ledger::new();
    let error = ledger.admit(&profile).unwrap_err();
    assert_eq!(
        error,
        LedgerError::Refused(Refusal {
            shortfall: vec![(vec![Tier::Host, Tier::Device(0)], 20), (vec![Tier::Device(1)], 30),],
        })
    );
}

#[test]
fn an_engine_without_elastic_terms_leaves_its_share() {
    let memory_a = MemoryLimits {
        mandatory: Vec::new(),
        elastic: vec![elastic("a", Tier::Host, 1, 1, 1000)],
    };
    let profile = Profile {
        budgets: vec![budget(&[Tier::Host], 101)],
        engines: vec![plan("a", 1, memory_a), plan("b", 1, MemoryLimits::default())],
    };
    let mut ledger = Ledger::new();
    let admission = ledger.admit(&profile).unwrap();
    assert_eq!(admission.engines[0].elastic, vec![("a".to_owned(), 51)]);
    assert_eq!(admission.engines[1].elastic, Vec::new());
    assert_eq!(admission.unassigned, vec![(vec![Tier::Host], 50)]);
}

#[test]
fn a_zero_share_leaves_the_spare_unassigned() {
    let memory = MemoryLimits {
        mandatory: Vec::new(),
        elastic: vec![elastic("e", Tier::Host, 1, 1, 1000)],
    };
    let profile = Profile {
        budgets: vec![budget(&[Tier::Host], 100)],
        engines: vec![plan("a", 0, memory)],
    };
    let mut ledger = Ledger::new();
    let admission = ledger.admit(&profile).unwrap();
    assert_eq!(admission.engines[0].elastic, vec![("e".to_owned(), 1)]);
    assert_eq!(admission.unassigned, vec![(vec![Tier::Host], 99)]);
}

#[test]
fn a_budget_with_no_tiers_is_invalid() {
    let profile = Profile {
        budgets: vec![Budget {
            tiers: Vec::new(),
            bytes: 100,
        }],
        engines: Vec::new(),
    };
    let mut ledger = Ledger::new();
    assert!(matches!(ledger.admit(&profile), Err(LedgerError::Invalid(_))));
}
