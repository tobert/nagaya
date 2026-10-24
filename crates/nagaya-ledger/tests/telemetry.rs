//! The ledger records `nagaya.ledger.bytes` and `nagaya.ledger.refusals` from `admit` and `admit_context` (spec 9.4, 9.5).

use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{ElasticTerm, Engine, Lifetime, MandatoryTerm, MemoryLimits, Tier};
use nagaya_ledger::{Budget, EnginePlan, Ledger, LedgerError, Profile};
use nagaya_telemetry::testing::{Capture, Point};

fn plan(name: &str, memory: MemoryLimits) -> EnginePlan {
    let mut limits = FakeAlways::new().limits();
    limits.memory = memory;
    EnginePlan {
        name: name.to_owned(),
        limits,
        devices: vec![0],
        share: 1,
    }
}

fn mandatory(name: &str, tier: Tier, bytes: u64, lifetime: Lifetime) -> MandatoryTerm {
    MandatoryTerm {
        name: name.to_owned(),
        tier,
        bytes,
        lifetime,
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

fn series(points: &[Point]) -> Vec<(String, String, String, f64)> {
    let mut rows: Vec<(String, String, String, f64)> = points
        .iter()
        .map(|point| {
            (
                point.attributes["tier"].clone(),
                point.attributes["engine"].clone(),
                point.attributes["term"].clone(),
                point.value,
            )
        })
        .collect();
    rows.sort_by(|left, right| left.partial_cmp(right).unwrap());
    rows
}

fn row(tier: &str, engine: &str, term: &str, bytes: f64) -> (String, String, String, f64) {
    (tier.to_owned(), engine.to_owned(), term.to_owned(), bytes)
}

#[test]
fn admit_records_the_bytes_of_each_term_of_each_engine() {
    let capture = Capture::start(Vec::new());
    let memory = MemoryLimits {
        mandatory: vec![
            mandatory("weights", Tier::Device(0), 600, Lifetime::Engine),
            // A context term is charged at `ctx_new`, not at `open`. It is not in the admission (spec 4.2).
            mandatory("per_context", Tier::Host, 5, Lifetime::Context),
        ],
        elastic: vec![elastic("cache", Tier::Host, 10, 1, 1000)],
    };
    let profile = Profile {
        budgets: vec![
            Budget {
                tiers: vec![Tier::Device(0)],
                bytes: 1000,
            },
            Budget {
                tiers: vec![Tier::Host],
                bytes: 200,
            },
        ],
        engines: vec![plan("qwen", memory)],
    };
    Ledger::new().admit(&profile).unwrap();

    let points = capture.metric("nagaya.ledger.bytes");
    assert_eq!(
        series(&points),
        [
            row("device0", "qwen", "weights", 600.0),
            row("host", "qwen", "cache", 200.0)
        ]
    );
}

#[test]
fn a_refused_profile_counts_one_refusal_for_each_short_budget() {
    let capture = Capture::start(Vec::new());
    let memory = MemoryLimits {
        mandatory: vec![
            mandatory("weights", Tier::Device(0), 600, Lifetime::Engine),
            mandatory("pinned", Tier::Host, 300, Lifetime::Engine),
        ],
        elastic: Vec::new(),
    };
    let profile = Profile {
        budgets: vec![
            Budget {
                tiers: vec![Tier::Device(0)],
                bytes: 100,
            },
            Budget {
                tiers: vec![Tier::Host],
                bytes: 100,
            },
        ],
        engines: vec![plan("qwen", memory)],
    };
    let error = Ledger::new().admit(&profile).unwrap_err();
    assert!(matches!(error, LedgerError::Refused(_)), "{error:?}");

    let mut refusals = capture.metric("nagaya.ledger.refusals");
    refusals.sort_by_key(|point| point.attributes["tier"].clone());
    assert_eq!(refusals.len(), 2, "{refusals:?}");
    for (point, tier) in refusals.iter().zip(["device0", "host"]) {
        assert_eq!(point.value, 1.0);
        assert_eq!(point.attributes["tier"], tier);
        assert_eq!(point.attributes["gate"], "budget");
        assert_eq!(point.attributes["engine"], "all");
    }
    assert!(
        capture.metric("nagaya.ledger.bytes").is_empty(),
        "a refusal records no bytes"
    );
}

#[test]
fn a_budget_of_two_tiers_is_one_refusal_with_both_tiers_in_its_label() {
    let capture = Capture::start(Vec::new());
    let memory = MemoryLimits {
        mandatory: vec![mandatory("weights", Tier::Host, 300, Lifetime::Engine)],
        elastic: Vec::new(),
    };
    let profile = Profile {
        budgets: vec![Budget {
            tiers: vec![Tier::Host, Tier::Device(0)],
            bytes: 100,
        }],
        engines: vec![plan("qwen", memory)],
    };
    Ledger::new().admit(&profile).unwrap_err();

    let refusals = capture.metric("nagaya.ledger.refusals");
    assert_eq!(refusals.len(), 1, "{refusals:?}");
    assert_eq!(refusals[0].value, 1.0);
    assert_eq!(refusals[0].attributes["tier"], "device0+host");
}

#[test]
fn a_refused_context_counts_a_refusal_for_its_engine() {
    let capture = Capture::start(Vec::new());
    let memory = MemoryLimits {
        mandatory: vec![mandatory("per_context", Tier::Host, 60, Lifetime::Context)],
        elastic: Vec::new(),
    };
    let profile = Profile {
        budgets: vec![Budget {
            tiers: vec![Tier::Host],
            bytes: 100,
        }],
        engines: vec![plan("qwen", memory)],
    };
    let mut ledger = Ledger::new();
    ledger.admit(&profile).unwrap();
    ledger.admit_context("qwen").unwrap();
    assert!(
        capture.metric("nagaya.ledger.refusals").is_empty(),
        "a fitting context is no refusal"
    );
    let error = ledger.admit_context("qwen").unwrap_err();
    assert!(matches!(error, LedgerError::ContextRefused(_)), "{error:?}");

    let refusals = capture.metric("nagaya.ledger.refusals");
    assert_eq!(refusals.len(), 1, "{refusals:?}");
    assert_eq!(refusals[0].value, 1.0);
    assert_eq!(refusals[0].attributes["engine"], "qwen");
    assert_eq!(refusals[0].attributes["tier"], "host");
    assert_eq!(refusals[0].attributes["gate"], "budget");
}

#[test]
fn admit_context_and_release_context_record_the_context_term_bytes() {
    let capture = Capture::start(Vec::new());
    let memory = MemoryLimits {
        mandatory: vec![mandatory("per_context", Tier::Host, 60, Lifetime::Context)],
        elastic: Vec::new(),
    };
    let profile = Profile {
        budgets: vec![Budget {
            tiers: vec![Tier::Host],
            bytes: 200,
        }],
        engines: vec![plan("qwen", memory)],
    };
    let mut ledger = Ledger::new();
    ledger.admit(&profile).unwrap();
    assert!(capture.metric("nagaya.ledger.bytes").is_empty());

    ledger.admit_context("qwen").unwrap();
    assert_eq!(
        series(&capture.metric("nagaya.ledger.bytes")),
        [row("host", "qwen", "per_context", 60.0)]
    );

    ledger.admit_context("qwen").unwrap();
    assert_eq!(
        series(&capture.metric("nagaya.ledger.bytes")),
        [row("host", "qwen", "per_context", 120.0)]
    );

    ledger.release_context("qwen").unwrap();
    assert_eq!(
        series(&capture.metric("nagaya.ledger.bytes")),
        [row("host", "qwen", "per_context", 60.0)]
    );

    ledger.release_context("qwen").unwrap();
    assert_eq!(
        series(&capture.metric("nagaya.ledger.bytes")),
        [row("host", "qwen", "per_context", 0.0)]
    );
}

#[test]
fn an_invalid_profile_is_no_refusal() {
    let capture = Capture::start(Vec::new());
    let profile = Profile {
        budgets: vec![Budget {
            tiers: Vec::new(),
            bytes: 100,
        }],
        engines: Vec::new(),
    };
    let error = Ledger::new().admit(&profile).unwrap_err();
    assert!(matches!(error, LedgerError::Invalid(_)), "{error:?}");
    assert!(capture.metric("nagaya.ledger.refusals").is_empty());
}

#[test]
fn a_refused_staging_counts_a_refusal_for_its_engine() {
    let capture = Capture::start(Vec::new());
    let profile = Profile {
        budgets: vec![Budget {
            tiers: vec![Tier::Host],
            bytes: 100,
        }],
        engines: vec![plan(
            "qwen",
            MemoryLimits {
                mandatory: Vec::new(),
                elastic: Vec::new(),
            },
        )],
    };
    let mut ledger = Ledger::new();
    ledger.admit(&profile).unwrap();
    ledger.admit_staging("qwen", Tier::Host, 100).unwrap();
    assert!(capture.metric("nagaya.ledger.refusals").is_empty());
    let error = ledger.admit_staging("qwen", Tier::Host, 1).unwrap_err();
    assert!(matches!(error, LedgerError::StagingRefused(_)), "{error:?}");
    let refusals = capture.metric("nagaya.ledger.refusals");
    assert_eq!(refusals.len(), 1, "{refusals:?}");
    assert_eq!(refusals[0].attributes["engine"], "qwen");
    assert_eq!(refusals[0].attributes["tier"], "host");
    assert_eq!(refusals[0].attributes["gate"], "budget");
}
