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
fn only_the_short_tier_is_reported() {
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
        budgets: vec![budget(&[Tier::Device(0)], 4096), budget(&[Tier::Host], 4096)],
        engines: vec![EnginePlan {
            name: "fake".to_owned(),
            limits,
            share: 1,
        }],
    };
    let mut ledger = Ledger::new();
    let admission = ledger.admit(&profile).unwrap();
    assert_eq!(admission.engines[0].mandatory, vec![(Tier::Device(0), 4096)]);
    assert_eq!(admission.engines[0].elastic, vec![("workspace".to_owned(), 2048)]);
    assert_eq!(
        admission.unassigned,
        vec![(vec![Tier::Device(0)], 0), (vec![Tier::Host], 2048)]
    );
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

#[test]
fn a_refusal_prints_each_short_budget_with_its_tiers_and_bytes() {
    let error = LedgerError::Refused(Refusal {
        shortfall: vec![(vec![Tier::Device(0), Tier::Host], 20), (vec![Tier::Device(1)], 5)],
    });
    assert_eq!(
        error.to_string(),
        "the profile does not fit: the budget of device 0, host is short by 20 bytes; \
         the budget of device 1 is short by 5 bytes"
    );
}

fn term_with(name: &str, tier: Tier, bytes: u64, lifetime: Lifetime) -> MandatoryTerm {
    MandatoryTerm {
        lifetime,
        ..mandatory(name, tier, bytes)
    }
}

/// One host budget of 100 bytes, one engine `a` with a 30 byte engine term and a 50 byte context term.
fn context_profile() -> Profile {
    let memory = MemoryLimits {
        mandatory: vec![
            term_with("weights", Tier::Host, 30, Lifetime::Engine),
            term_with("context", Tier::Host, 50, Lifetime::Context),
        ],
        elastic: Vec::new(),
    };
    Profile {
        budgets: vec![budget(&[Tier::Host], 100)],
        engines: vec![plan("a", 1, memory)],
    }
}

#[test]
fn open_does_not_charge_a_context_term() {
    let mut ledger = Ledger::new();
    let admission = ledger.admit(&context_profile()).unwrap();
    assert_eq!(admission.unassigned, vec![(vec![Tier::Host], 70)]);
    assert_eq!(admission.engines[0].mandatory, vec![(Tier::Host, 30)]);
    assert_eq!(ledger.unassigned(), vec![(vec![Tier::Host], 70)]);
}

#[test]
fn open_charges_a_loan_term_once() {
    let memory = MemoryLimits {
        mandatory: vec![term_with("scratch", Tier::Host, 10, Lifetime::Loan)],
        elastic: Vec::new(),
    };
    let profile = Profile {
        budgets: vec![budget(&[Tier::Host], 100)],
        engines: vec![plan("a", 1, memory)],
    };
    let mut ledger = Ledger::new();
    let admission = ledger.admit(&profile).unwrap();
    assert_eq!(admission.unassigned, vec![(vec![Tier::Host], 90)]);
}

#[test]
fn ctx_new_charges_the_context_terms_from_the_unassigned_bytes() {
    let mut ledger = Ledger::new();
    ledger.admit(&context_profile()).unwrap();
    ledger.admit_context("a").unwrap();
    assert_eq!(ledger.unassigned(), vec![(vec![Tier::Host], 20)]);
}

#[test]
fn a_refused_ctx_new_names_the_shortfall_and_changes_nothing() {
    let mut ledger = Ledger::new();
    ledger.admit(&context_profile()).unwrap();
    ledger.admit_context("a").unwrap();
    let before = ledger.clone();
    let error = ledger.admit_context("a").unwrap_err();
    assert_eq!(
        error,
        LedgerError::ContextRefused(Refusal {
            shortfall: vec![(vec![Tier::Host], 30)],
        })
    );
    assert_eq!(ledger, before);
}

#[test]
fn a_ctx_new_that_is_short_in_one_tier_charges_no_tier() {
    let memory = MemoryLimits {
        mandatory: vec![
            term_with("host", Tier::Host, 10, Lifetime::Context),
            term_with("device", Tier::Device(0), 50, Lifetime::Context),
        ],
        elastic: Vec::new(),
    };
    let profile = Profile {
        budgets: vec![budget(&[Tier::Host], 100), budget(&[Tier::Device(0)], 40)],
        engines: vec![plan("a", 1, memory)],
    };
    let mut ledger = Ledger::new();
    ledger.admit(&profile).unwrap();
    let error = ledger.admit_context("a").unwrap_err();
    assert!(matches!(error, LedgerError::ContextRefused(_)), "{error:?}");
    assert_eq!(
        ledger.unassigned(),
        vec![(vec![Tier::Host], 100), (vec![Tier::Device(0)], 40)]
    );
}

#[test]
fn the_end_of_a_context_gives_its_bytes_back() {
    let mut ledger = Ledger::new();
    ledger.admit(&context_profile()).unwrap();
    ledger.admit_context("a").unwrap();
    ledger.release_context("a").unwrap();
    assert_eq!(ledger.unassigned(), vec![(vec![Tier::Host], 70)]);
    ledger.admit_context("a").unwrap();
}

#[test]
fn a_release_with_no_context_is_an_error() {
    let mut ledger = Ledger::new();
    ledger.admit(&context_profile()).unwrap();
    assert!(matches!(ledger.release_context("a"), Err(LedgerError::Invalid(_))));
    assert_eq!(ledger.unassigned(), vec![(vec![Tier::Host], 70)]);
}

#[test]
fn ctx_new_for_an_engine_that_is_not_admitted_is_an_error() {
    let mut ledger = Ledger::new();
    ledger.admit(&context_profile()).unwrap();
    assert!(matches!(ledger.admit_context("b"), Err(LedgerError::Invalid(_))));
}

#[test]
fn a_context_that_fits_exactly_is_admitted() {
    let memory = MemoryLimits {
        mandatory: vec![term_with("context", Tier::Host, 50, Lifetime::Context)],
        elastic: Vec::new(),
    };
    let profile = Profile {
        budgets: vec![budget(&[Tier::Host], 100)],
        engines: vec![plan("a", 1, memory)],
    };
    let mut ledger = Ledger::new();
    ledger.admit(&profile).unwrap();
    ledger.admit_context("a").unwrap();
    ledger.admit_context("a").unwrap();
    assert_eq!(ledger.unassigned(), vec![(vec![Tier::Host], 0)]);
    assert!(ledger.admit_context("a").is_err());
}
