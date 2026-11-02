use std::net::SocketAddr;
use std::path::Path;

use nagaya::admission::{AdmitError, Registry, admit};
use nagaya::gossip::{ClusterView, NodeView};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, StoreSpec, TierSpec};
use nagaya::reprofile::Changer;
use nagaya::swap::{Candidate, Outcome, Swap, SwapError, VERSION_FILE, desired_value};
use nagaya_abi::Tier;
use nagaya_ledger::{Ledger, LedgerError};
use serde_json::json;

fn engine(name: &str) -> EnginePlacement {
    EnginePlacement {
        name: name.to_owned(),
        kind: "fake".to_owned(),
        devices: vec![0],
        config: toml::Table::new(),
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency: Default::default(),
        pinned: false,
    }
}

/// A profile with `engines` fake engines and `device` bytes of device budget.
fn profile(engines: &[&str], device: u64) -> PlacementProfile {
    PlacementProfile {
        engines: engines.iter().map(|name| engine(name)).collect(),
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: device,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes: 1 << 20,
            },
        ],
        store: Some(StoreSpec {
            nvme: 65536,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
    }
}

fn inventory() -> Inventory {
    Inventory {
        devices: vec![Device {
            id: 0,
            name: "fake-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 1 << 30,
            unified: false,
            ignored: false,
        }],
        host_bytes: 1 << 30,
    }
}

/// A running process that holds `active` in its ledger. The store is in `dir`.
fn running(dir: &Path, active: &PlacementProfile) -> Process {
    active.write(dir).unwrap();
    let mut process = Start::new(active.clone(), Registry::with_fake(), inventory())
        .store(dir)
        .start()
        .unwrap();
    process.wait_opens(std::time::Duration::from_secs(10)).unwrap();
    process
}

/// A view in which each `(peer, key, value)` is a key of a live peer.
fn view(entries: &[(&str, &str, serde_json::Value)]) -> ClusterView {
    let mut nodes: Vec<NodeView> = Vec::new();
    for (peer, key, value) in entries {
        if !nodes.iter().any(|node| node.name == *peer) {
            nodes.push(NodeView {
                name: (*peer).to_owned(),
                id: format!("{peer}:1:127.0.0.1:1"),
                generation: 1,
                addr: "127.0.0.1:1".parse::<SocketAddr>().unwrap(),
                is_self: false,
                keys: Default::default(),
            });
        }
        let node = nodes.iter_mut().find(|node| node.name == *peer).unwrap();
        node.keys.insert((*key).to_owned(), value.to_string());
    }
    ClusterView {
        live: nodes,
        dead: Vec::new(),
    }
}

const KEY: &str = "profile:desired:gfx1151host";

/// Handle a desired profile with the code of a local `PUT` (spec 2.4, "The routes of a profile for another host").
fn desire(swap: &mut Swap, candidate: Candidate, process: &Process) -> Result<Outcome, SwapError> {
    Changer::new().0.desire(swap, process, candidate)
}

/// True for an accepted profile whose change restarts the process, or for one that runs in place when `restart` is
/// false.
fn accepted(outcome: &Outcome, restart: bool) -> bool {
    matches!(outcome, Outcome::Accepted(change) if change.restart == restart)
}

fn swap(dir: &Path, active: &PlacementProfile) -> Swap {
    Swap::open(dir, "gfx1151host", active.clone()).unwrap()
}

#[test]
fn an_older_version_never_replaces_a_newer_one() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    let process = running(directory.path(), &active);
    let mut swap = swap(directory.path(), &active);

    // Peer b holds the older version and sorts last. Peer a holds the newer one.
    let newer = profile(&["fake-one"], 8192);
    let older = profile(&["fake-one", "fake-two"], 8192);
    let both = view(&[
        ("a", KEY, desired_value(5, &newer)),
        ("b", KEY, desired_value(3, &older)),
    ]);
    let candidate = swap.candidate(&both).expect("a version above 0 is a candidate");
    assert_eq!((candidate.version, candidate.from.as_str()), (5, "a"));
    let outcome = desire(&mut swap, candidate, &process).unwrap();
    assert!(accepted(&outcome, false), "a new budget runs in place: {outcome:?}");
    assert_eq!(swap.handled(), 5);

    assert_eq!(swap.candidate(&both), None, "versions 5 and 3 are not above 5");
    let only_older = view(&[("b", KEY, desired_value(3, &older))]);
    assert_eq!(swap.candidate(&only_older), None);
    let newest = view(&[
        ("a", KEY, desired_value(5, &newer)),
        ("c", KEY, desired_value(6, &newer)),
    ]);
    assert_eq!(swap.candidate(&newest).unwrap().version, 6);
    assert_eq!(PlacementProfile::read_active(directory.path()).unwrap(), newer);
}

#[test]
fn the_handled_version_survives_a_restart() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    let process = running(directory.path(), &active);
    let mut first = swap(directory.path(), &active);
    let newer = profile(&["fake-one"], 8192);
    let offered = view(&[("a", KEY, desired_value(7, &newer))]);
    let candidate = first.candidate(&offered).unwrap();
    desire(&mut first, candidate, &process).unwrap();
    assert_eq!(
        std::fs::read_to_string(directory.path().join(VERSION_FILE))
            .unwrap()
            .trim(),
        "7"
    );

    let second = Swap::open(directory.path(), "gfx1151host", newer).unwrap();
    assert_eq!(second.handled(), 7);
    let mut second = second;
    assert_eq!(second.candidate(&offered), None);
}

#[test]
fn a_profile_that_adds_an_engine_is_persisted_and_restarts() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    let process = running(directory.path(), &active);
    let mut swap = swap(directory.path(), &active);
    let wanted = profile(&["fake-one", "fake-two"], 8192);
    let offered = view(&[("a", KEY, desired_value(1, &wanted))]);
    let candidate = swap.candidate(&offered).unwrap();
    let outcome = desire(&mut swap, candidate, &process).unwrap();
    assert!(accepted(&outcome, true), "a new engine restarts: {outcome:?}");
    assert_eq!(PlacementProfile::read_active(directory.path()).unwrap(), wanted);
    assert_eq!(
        std::fs::read_to_string(directory.path().join(VERSION_FILE))
            .unwrap()
            .trim(),
        "1"
    );
}

/// Offer `wanted` as version 1 and give the outcome of the swap.
fn offer(dir: &Path, active: &PlacementProfile, wanted: &PlacementProfile, process: &Process) -> Outcome {
    let mut swap = swap(dir, active);
    let offered = view(&[("a", KEY, desired_value(1, wanted))]);
    let candidate = swap.candidate(&offered).unwrap();
    desire(&mut swap, candidate, process).unwrap()
}

#[test]
fn a_profile_that_changes_only_a_share_runs_in_place() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    let process = running(directory.path(), &active);
    let mut wanted = active.clone();
    wanted.engines[0].share = 2;
    let outcome = offer(directory.path(), &active, &wanted, &process);
    assert!(accepted(&outcome, false), "{outcome:?}");
    assert_eq!(PlacementProfile::read_active(directory.path()).unwrap(), wanted);
}

#[test]
fn a_profile_that_changes_only_a_store_budget_restarts() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    let process = running(directory.path(), &active);
    let mut wanted = active.clone();
    wanted.store = Some(StoreSpec {
        nvme: 131072,
        host: None,
        interval_copies: 0,
        objects: None,
    });
    let outcome = offer(directory.path(), &active, &wanted, &process);
    assert!(
        accepted(&outcome, true),
        "a change of [store] restarts (spec 3.3): {outcome:?}"
    );
    assert_eq!(PlacementProfile::read_active(directory.path()).unwrap(), wanted);
}

#[test]
fn an_identical_profile_changes_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    let mut process = running(directory.path(), &active);
    let file = directory.path().join("profile.toml");
    let bytes = std::fs::read(&file).unwrap();
    let modified = std::fs::metadata(&file).unwrap().modified().unwrap();
    // A stopped process refuses each admission. So a handle that asks the ledger would not give `Unchanged`.
    process.abort();
    let mut swap = swap(directory.path(), &active);
    let offered = view(&[("a", KEY, desired_value(3, &active))]);
    let candidate = swap.candidate(&offered).unwrap();
    assert_eq!(desire(&mut swap, candidate, &process).unwrap(), Outcome::Unchanged);
    assert_eq!(swap.handled(), 3, "the version is recorded");
    assert_eq!(
        std::fs::read_to_string(directory.path().join(VERSION_FILE))
            .unwrap()
            .trim(),
        "3"
    );
    assert_eq!(std::fs::read(&file).unwrap(), bytes);
    assert_eq!(std::fs::metadata(&file).unwrap().modified().unwrap(), modified);
    assert_eq!(swap.candidate(&offered), None);
}

#[test]
fn a_profile_that_does_not_fit_is_refused_and_changes_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    let process = running(directory.path(), &active);
    let mut swap = swap(directory.path(), &active);
    let too_small = profile(&["fake-one", "fake-two"], 1);
    let offered = view(&[("a", KEY, desired_value(2, &too_small))]);

    // The ledger gives the shortfall that the refusal must report.
    let expected = match admit(&too_small, &Registry::with_fake(), &inventory(), &mut Ledger::new()) {
        Err(AdmitError::Ledger(LedgerError::Refused(refusal))) => refusal.shortfall,
        other => panic!("the test profile must not fit: {other:?}"),
    };
    assert!(!expected.is_empty());

    let candidate = swap.candidate(&offered).unwrap();
    let Outcome::Refused(refusal) = desire(&mut swap, candidate, &process).unwrap() else {
        panic!("the profile does not fit")
    };
    assert_eq!(refusal.version, 2);
    let reported: Vec<(Vec<Tier>, u64)> = refusal
        .shortfall
        .iter()
        .map(|entry| (entry.tiers.iter().map(|tier| Tier::from(*tier)).collect(), entry.bytes))
        .collect();
    assert_eq!(reported, expected);
    assert_eq!(
        refusal.to_value()["shortfall"][0]["tiers"],
        json!(["device(0)"]),
        "the value names the tiers"
    );
    assert_eq!(
        PlacementProfile::read_active(directory.path()).unwrap(),
        active,
        "the file keeps the active profile"
    );
    assert_eq!(
        std::fs::read_to_string(directory.path().join(VERSION_FILE))
            .unwrap()
            .trim(),
        "2",
        "a refused version is a handled version"
    );
    assert_eq!(swap.active(), &active);
    assert_eq!(
        swap.candidate(&offered),
        None,
        "the node does not handle version 2 twice"
    );
}

#[test]
fn a_profile_that_does_not_parse_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    let process = running(directory.path(), &active);
    let mut swap = swap(directory.path(), &active);
    let offered = view(&[("a", KEY, json!({"version": 4, "profile": {"engines": 3}}))]);
    let candidate = swap.candidate(&offered).unwrap();
    let Outcome::Refused(refusal) = desire(&mut swap, candidate, &process).unwrap() else {
        panic!("the profile does not parse")
    };
    assert_eq!(refusal.version, 4);
    assert!(refusal.reason.contains("profile"), "{}", refusal.reason);
    assert!(refusal.shortfall.is_empty());
    assert_eq!(PlacementProfile::read_active(directory.path()).unwrap(), active);
}

#[test]
fn a_profile_with_engines_and_no_store_budget_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    let process = running(directory.path(), &active);
    let mut swap = swap(directory.path(), &active);
    let mut wanted = profile(&["fake-one"], 8192);
    wanted.store = None;
    let offered = view(&[("a", KEY, desired_value(1, &wanted))]);
    let candidate = swap.candidate(&offered).unwrap();
    let Outcome::Refused(refusal) = desire(&mut swap, candidate, &process).unwrap() else {
        panic!("the profile has no store budget")
    };
    assert!(refusal.reason.contains("store.nvme"), "{}", refusal.reason);
    assert_eq!(PlacementProfile::read_active(directory.path()).unwrap(), active);
}

#[test]
fn a_desired_profile_for_another_host_is_not_a_candidate() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    let mut swap = swap(directory.path(), &active);
    let offered = view(&[("a", "profile:desired:b70host", desired_value(9, &active))]);
    assert_eq!(swap.candidate(&offered), None);
}

#[test]
fn a_version_file_that_is_not_a_number_stops_the_open() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join(VERSION_FILE), "seven\n").unwrap();
    let error = Swap::open(directory.path(), "gfx1151host", profile(&["fake-one"], 4096))
        .err()
        .expect("the open fails");
    assert!(
        error.to_string().contains(VERSION_FILE),
        "the message names the file: {error}"
    );
}

#[test]
fn a_stopping_process_is_not_a_refusal() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    let mut process = running(directory.path(), &active);
    process.abort();
    let mut swap = swap(directory.path(), &active);
    let wanted = profile(&["fake-one", "fake-two"], 8192);
    let offered = view(&[("a", KEY, desired_value(4, &wanted))]);
    let candidate = swap.candidate(&offered).unwrap();
    assert_eq!(desire(&mut swap, candidate, &process).unwrap(), Outcome::Stopping);
    assert_eq!(swap.handled(), 0, "the node did not handle the version");
    assert!(!directory.path().join(VERSION_FILE).exists(), "no version is written");
    assert_eq!(PlacementProfile::read_active(directory.path()).unwrap(), active);
    assert_eq!(swap.active(), &active);
    assert_eq!(
        swap.candidate(&offered).map(|candidate| candidate.version),
        Some(4),
        "the version stays a candidate"
    );
}

/// Refuse `wanted` as `version` with a running process, and give the swap that refused it.
fn refuse(dir: &Path, active: &PlacementProfile, version: u64) -> Swap {
    let process = running(dir, active);
    let mut swap = swap(dir, active);
    let too_small = profile(&["fake-one", "fake-two"], 1);
    let offered = view(&[("a", KEY, desired_value(version, &too_small))]);
    let candidate = swap.candidate(&offered).unwrap();
    assert!(matches!(
        desire(&mut swap, candidate, &process).unwrap(),
        Outcome::Refused(_)
    ));
    swap
}

#[test]
fn a_refused_version_survives_a_restart() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    refuse(directory.path(), &active, 5);
    let mut second = Swap::open(directory.path(), "gfx1151host", active.clone()).unwrap();
    assert_eq!(second.handled(), 5);
    let too_small = profile(&["fake-one", "fake-two"], 1);
    let offered = view(&[("a", KEY, desired_value(5, &too_small))]);
    assert_eq!(
        second.candidate(&offered),
        None,
        "the restart does not handle version 5 again"
    );
    assert_eq!(
        PlacementProfile::read_active(directory.path()).unwrap(),
        active,
        "a refusal keeps profile.toml"
    );
}

#[test]
fn a_lower_version_after_a_refused_one_is_ignored_after_a_restart() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    // Version 3 was applied earlier. Version 5 is refused.
    std::fs::write(directory.path().join(VERSION_FILE), "3\n").unwrap();
    refuse(directory.path(), &active, 5);
    let mut second = Swap::open(directory.path(), "gfx1151host", active.clone()).unwrap();
    let lower = profile(&["fake-one", "fake-two"], 8192);
    let offered = view(&[("a", KEY, desired_value(4, &lower))]);
    assert_eq!(
        second.candidate(&offered),
        None,
        "version 4 is below the refused version 5"
    );
}

#[test]
fn a_peer_that_churns_malformed_values_costs_one_entry() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    let mut swap = swap(directory.path(), &active);
    for round in 0..50 {
        // Each value is malformed in a new way: the version is not a whole number.
        let offered = view(&[("a", KEY, json!({"version": format!("v{round}"), "profile": {}}))]);
        assert_eq!(swap.candidate(&offered), None);
    }
    assert_eq!(swap.reported_len(), 1, "one entry for the peer, not one for each value");
    let two = view(&[("a", KEY, json!({"version": "x"})), ("b", KEY, json!({"version": "y"}))]);
    assert_eq!(swap.candidate(&two), None);
    assert_eq!(swap.reported_len(), 2, "one entry for each peer id");
    // A peer that left the view costs nothing.
    let alone = view(&[("b", KEY, json!({"version": "y"}))]);
    assert_eq!(swap.candidate(&alone), None);
    assert_eq!(swap.reported_len(), 1);
}

/// A version that comes while a change waits is refused with the reason `busy`, and it counts as handled (spec 2.4).
#[test]
fn a_desired_profile_during_a_change_is_refused_as_busy() {
    let directory = tempfile::tempdir().unwrap();
    let active = profile(&["fake-one"], 4096);
    let process = running(directory.path(), &active);
    let mut swap = swap(directory.path(), &active);
    // A change of another client holds the ledger.
    process.begin_change(&profile(&["fake-one"], 16384)).unwrap();
    let wanted = profile(&["fake-one"], 8192);
    let offered = view(&[("a", KEY, desired_value(6, &wanted))]);
    let candidate = swap.candidate(&offered).unwrap();
    let Outcome::Refused(refusal) = desire(&mut swap, candidate, &process).unwrap() else {
        panic!("a change runs")
    };
    assert_eq!((refusal.version, refusal.reason.as_str()), (6, "busy"));
    assert_eq!(swap.handled(), 6, "a refused version is a handled version");
    assert_eq!(PlacementProfile::read_active(directory.path()).unwrap(), active);
}
