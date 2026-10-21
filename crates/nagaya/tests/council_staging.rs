//! A copy of engine state in host memory is charged on the ledger (spec 4.2). The engine is the fake engine over the toy
//! vocabulary. Kaibo review 2026-10-20, finding 3.

mod council_support;

use council_support::{ENGINE, Fixture};
use gogi_proto::context::Context;
use gogi_proto::reads::ReadsRequest;
use nagaya::council::error::ServiceError;
use nagaya::council::{Schedule, ServiceConfig};
use nagaya_abi::Tier;
use nagaya_ledger::LedgerError;
use serde_json::{Value, json};

fn context(text: &str) -> Context {
    let value: Value = json!({"messages": [{"role": "user", "content": text, "snap": true}]});
    serde_json::from_value(value).unwrap()
}

/// A read with two suffixes: the job marks the state after the context, and rewinds to it for the second suffix.
fn two_suffix_read(text: &str) -> ReadsRequest {
    let body = json!({"model": "toy", "contexts": [{"messages": [{"role": "user", "content": text, "snap": true}]}],
        "suffixes": ["\nAnswer: ", "\nVerdict: "], "letters": ["A", "B", "C"]});
    serde_json::from_value(body).unwrap()
}

fn host_unassigned(f: &Fixture) -> u64 {
    f.process
        .ports()
        .unassigned()
        .into_iter()
        .find(|(tiers, _)| tiers.contains(&Tier::Host))
        .map(|(_, bytes)| bytes)
        .expect("a host budget")
}

/// Take all the host bytes but `leave`. Give the bytes that were taken.
fn leave_only(f: &Fixture, leave: u64) -> u64 {
    let take = host_unassigned(f) - leave;
    f.process.ports().admit_staging(ENGINE, Tier::Host, take).unwrap();
    take
}

fn is_staging_refusal(error: &ServiceError) -> bool {
    matches!(error, ServiceError::ContextRefused(LedgerError::StagingRefused(_)))
}

#[test]
fn a_build_whose_snapshot_copy_does_not_fit_is_refused_and_changes_nothing() {
    let f = Fixture::new();
    let free = host_unassigned(&f);
    // A snapshot of the toy engine is a few thousand bytes. 100 bytes cannot hold the copy.
    let taken = leave_only(&f, 100);
    let before = f.process.ports().unassigned();
    let error = f
        .service
        .build(&context("hello"), false, Schedule::default())
        .unwrap_err();
    assert!(is_staging_refusal(&error), "{error:?}");
    assert!(error.to_string().contains("short by"), "{error}");
    assert_eq!(f.process.ports().unassigned(), before, "the ledger is as it was");
    assert_eq!(f.entries(), 0, "no snapshot");
    assert_eq!(f.service.free_slots(), 3, "no slot is held");
    // With the bytes back, the same build works, and it gives back what it took.
    f.process.ports().release_staging(Tier::Host, taken).unwrap();
    f.service.build(&context("hello"), false, Schedule::default()).unwrap();
    assert_eq!(host_unassigned(&f), free, "a build releases its copy");
}

#[test]
fn a_mark_that_does_not_fit_refuses_the_read_and_changes_nothing() {
    let f = Fixture::new();
    let request = two_suffix_read("hello");
    // The build holds the head, so the read restores it and needs no snapshot copy. Only the mark is left.
    f.service.build(&context("hello"), false, Schedule::default()).unwrap();
    let free = host_unassigned(&f);
    let entries = f.entries();
    let taken = leave_only(&f, 100);
    let before = f.process.ports().unassigned();
    let error = f.service.read(&request).unwrap_err();
    assert!(is_staging_refusal(&error), "{error:?}");
    assert_eq!(f.process.ports().unassigned(), before, "the ledger is as it was");
    assert_eq!(f.entries(), entries, "no snapshot");
    assert_eq!(f.service.free_slots(), 3, "no slot is held");
    f.process.ports().release_staging(Tier::Host, taken).unwrap();
    f.service.read(&request).unwrap();
    assert_eq!(host_unassigned(&f), free, "a read releases its mark");
}

#[test]
fn a_read_releases_its_copies() {
    let f = Fixture::with(0, 1 << 20, None, ServiceConfig::new("toy", ENGINE));
    let free = host_unassigned(&f);
    for text in ["one", "two"] {
        f.service.read(&two_suffix_read(text)).unwrap();
        assert_eq!(host_unassigned(&f), free, "the read of {text} released its copies");
    }
}
