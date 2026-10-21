//! The ledger charges the contexts of a service (spec 4.2, 4.3).

mod council_support;

use council_support::{ENGINE, TermFactory, roomy_slots, start};
use nagaya::process::ContextChargeError;
use nagaya_abi::Tier;
use nagaya_ledger::LedgerError;

fn host_unassigned(ports: &nagaya::process::Ports) -> u64 {
    ports
        .unassigned()
        .into_iter()
        .find(|(tiers, _)| tiers.contains(&Tier::Host))
        .map(|(_, bytes)| bytes)
        .expect("a host budget")
}

#[test]
fn a_context_takes_bytes_until_it_is_released_and_a_refusal_changes_nothing() {
    let directory = tempfile::tempdir().unwrap();
    // The fake's workspace term takes 2048 of the 4096 host bytes, so 2048 bytes stay unassigned.
    let term = 2048;
    let (process, _) = start(
        directory.path(),
        TermFactory {
            bytes: term,
            slots: roomy_slots(2),
            vocab: None,
            ..TermFactory::toy()
        },
        4096,
        None,
    );
    let ports = process.ports();
    assert_eq!(
        host_unassigned(&ports),
        term,
        "the test expects 2048 unassigned host bytes"
    );

    ports.admit_context(ENGINE).unwrap();
    assert_eq!(host_unassigned(&ports), 0);

    let before = ports.unassigned();
    match ports.admit_context(ENGINE) {
        Err(ContextChargeError::Ledger(LedgerError::ContextRefused(refusal))) => {
            assert_eq!(refusal.shortfall.iter().map(|(_, bytes)| *bytes).sum::<u64>(), term);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(ports.unassigned(), before, "a refusal changes nothing");

    ports.release_context(ENGINE).unwrap();
    assert_eq!(host_unassigned(&ports), term);
    ports.admit_context(ENGINE).unwrap();
}

#[test]
fn a_release_with_no_live_context_is_an_error() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _) = start(
        directory.path(),
        TermFactory {
            bytes: 100,
            slots: roomy_slots(2),
            vocab: None,
            ..TermFactory::toy()
        },
        4096,
        None,
    );
    let ports = process.ports();
    assert!(matches!(
        ports.release_context(ENGINE),
        Err(ContextChargeError::Ledger(LedgerError::Invalid(_)))
    ));
}

#[test]
fn a_stopping_process_charges_no_context() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, _) = start(
        directory.path(),
        TermFactory {
            bytes: 100,
            slots: roomy_slots(2),
            vocab: None,
            ..TermFactory::toy()
        },
        4096,
        None,
    );
    let ports = process.ports();
    process.stop(std::time::Duration::from_secs(5), &nagaya::signal::SignalFlag::new());
    assert!(matches!(ports.admit_context(ENGINE), Err(ContextChargeError::Stopping)));
}

#[test]
fn staging_takes_host_bytes_until_it_is_released_and_a_refusal_changes_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let (process, _) = start(
        directory.path(),
        TermFactory {
            slots: roomy_slots(2),
            vocab: None,
            ..TermFactory::toy()
        },
        4096,
        None,
    );
    let ports = process.ports();
    let free = host_unassigned(&ports);
    ports.admit_staging(ENGINE, Tier::Host, free - 10).unwrap();
    assert_eq!(host_unassigned(&ports), 10);

    let before = ports.unassigned();
    match ports.admit_staging(ENGINE, Tier::Host, 15) {
        Err(ContextChargeError::Ledger(LedgerError::StagingRefused(refusal))) => {
            assert_eq!(refusal.shortfall.iter().map(|(_, bytes)| *bytes).sum::<u64>(), 5);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(ports.unassigned(), before, "a refusal changes nothing");

    ports.release_staging(Tier::Host, free - 10).unwrap();
    assert_eq!(host_unassigned(&ports), free);
    assert!(matches!(
        ports.release_staging(Tier::Host, 1),
        Err(ContextChargeError::Ledger(LedgerError::Invalid(_)))
    ));
}

#[test]
fn a_stopping_process_charges_no_staging() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, _) = start(directory.path(), TermFactory::toy(), 4096, None);
    let ports = process.ports();
    process.stop(std::time::Duration::from_secs(5), &nagaya::signal::SignalFlag::new());
    assert!(matches!(
        ports.admit_staging(ENGINE, Tier::Host, 1),
        Err(ContextChargeError::Stopping)
    ));
}
