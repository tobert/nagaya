//! Pins and the snapshot info of a service (gogi "Snapshots and pins", spec 6.10).

mod council_support;

use council_support::{ENGINE, Fixture};
use gogi_proto::SnapshotId;
use gogi_proto::context::{Context, Tier};
use gogi_proto::pins::PinsPut;
use nagaya::council::error::ServiceError;
use nagaya::council::{Schedule, ServiceConfig};
use serde_json::{Value, json};

fn context(text: &str) -> Context {
    let value: Value = json!({"messages": [{"role": "user", "content": text, "snap": true}]});
    serde_json::from_value(value).unwrap()
}

fn built(f: &Fixture, text: &str) -> SnapshotId {
    f.service
        .build(&context(text), false, Schedule::default())
        .unwrap()
        .head
        .unwrap()
}

fn put(ids: &[&SnapshotId], ttl_s: u64) -> PinsPut {
    PinsPut {
        snapshots: ids.iter().map(|id| id.to_string()).collect(),
        ttl_s,
    }
}

fn pinned_by_the_store(f: &Fixture, id: &SnapshotId) -> usize {
    let ports = f.process.ports();
    let store = ports.store().unwrap().lock().unwrap();
    let content = nagaya_store::ContentId::from_hex(id.as_str().trim_start_matches("blake3:")).unwrap();
    store.pins().get(&content).map_or(0, Vec::len)
}

#[test]
fn a_snapshot_reports_its_model_tokens_tier_size_and_pinning_owners() {
    let f = Fixture::new();
    let id = built(&f, "hello");
    let detail = f.service.snapshot(&id).unwrap();
    assert_eq!(detail.id, id);
    assert_eq!(detail.model, "toy");
    assert!(detail.tokens > 0);
    assert_eq!(detail.tier, Tier::Host);
    assert!(detail.bytes.unwrap() >= 4096);
    assert!(detail.pins.is_empty());
    f.service.put_pins("kernel-1", &put(&[&id], 3600)).unwrap();
    f.service.put_pins("kernel-2", &put(&[&id], 600)).unwrap();
    let detail = f.service.snapshot(&id).unwrap();
    let mut pins: Vec<(String, u64)> = detail.pins.into_iter().map(|p| (p.owner, p.expires_in_s)).collect();
    pins.sort();
    // The clock reads a minute later at each call, so the ends are a little under the ttl.
    assert_eq!(pins.len(), 2);
    assert_eq!(pins[0].0, "kernel-1");
    assert!(pins[0].1 > 3000 && pins[0].1 <= 3600, "{pins:?}");
    assert_eq!(pins[1].0, "kernel-2");
    assert!(pins[1].1 > 0 && pins[1].1 <= 600, "{pins:?}");
}

#[test]
fn the_tokens_of_a_snapshot_are_its_position() {
    let f = Fixture::new();
    let built = f.service.build(&context("hello"), false, Schedule::default()).unwrap();
    let detail = f.service.snapshot(&built.head.unwrap()).unwrap();
    assert_eq!(detail.tokens, built.usage.prompt);
}

#[test]
fn an_unknown_snapshot_is_not_found() {
    let f = Fixture::new();
    for id in [
        format!("blake3:{}", "0".repeat(64)),
        format!("snap:{}", "0".repeat(64)),
        format!("blake3:{}", "f".repeat(40)),
    ] {
        let id = SnapshotId::parse(id).unwrap();
        assert!(
            matches!(f.service.snapshot(&id), Err(ServiceError::NotFound(_))),
            "{id}"
        );
    }
}

#[test]
fn a_put_pins_the_snapshots_that_the_store_holds_and_reports_the_others() {
    let f = Fixture::new();
    let id = built(&f, "hello");
    let unknown = SnapshotId::parse(format!("blake3:{}", "0".repeat(64))).unwrap();
    let result = f.service.put_pins("kernel-1", &put(&[&id, &unknown], 3600)).unwrap();
    assert_eq!(result.owner, "kernel-1");
    assert_eq!(result.expires_in_s, Some(3600));
    assert_eq!(
        result
            .snapshots
            .iter()
            .map(|s| (s.id.as_str(), s.held))
            .collect::<Vec<_>>(),
        [(id.as_str(), true), (unknown.as_str(), false)]
    );
    assert_eq!(pinned_by_the_store(&f, &id), 1);
    let listing = f.service.get_pins("kernel-1").unwrap();
    assert_eq!(
        listing.snapshots,
        std::slice::from_ref(&id),
        "only the held snapshot is listed"
    );
    assert!(listing.expires_in_s > 3000 && listing.expires_in_s <= 3600);
}

#[test]
fn a_put_replaces_the_whole_set_of_the_owner() {
    let f = Fixture::new();
    let a = built(&f, "a");
    let b = built(&f, "b");
    f.service.put_pins("kernel-1", &put(&[&a], 3600)).unwrap();
    f.service.put_pins("kernel-1", &put(&[&b], 3600)).unwrap();
    assert_eq!(pinned_by_the_store(&f, &a), 0, "a left the set");
    assert_eq!(pinned_by_the_store(&f, &b), 1);
    assert_eq!(f.service.get_pins("kernel-1").unwrap().snapshots, [b]);
}

#[test]
fn an_empty_set_pins_nothing_and_has_no_end() {
    let f = Fixture::new();
    let a = built(&f, "a");
    f.service.put_pins("kernel-1", &put(&[&a], 3600)).unwrap();
    let result = f.service.put_pins("kernel-1", &put(&[], 3600)).unwrap();
    assert_eq!(result.expires_in_s, None);
    assert_eq!(pinned_by_the_store(&f, &a), 0);
    assert!(matches!(f.service.get_pins("kernel-1"), Err(ServiceError::NotFound(_))));
}

#[test]
fn two_owners_pin_one_snapshot_and_one_delete_leaves_the_other() {
    let f = Fixture::new();
    let a = built(&f, "a");
    f.service.put_pins("one", &put(&[&a], 3600)).unwrap();
    f.service.put_pins("two", &put(&[&a], 3600)).unwrap();
    assert_eq!(pinned_by_the_store(&f, &a), 2);
    f.service.delete_pins("one").unwrap();
    assert_eq!(pinned_by_the_store(&f, &a), 1);
    assert!(matches!(f.service.get_pins("one"), Err(ServiceError::NotFound(_))));
    assert!(f.service.get_pins("two").is_ok());
    assert!(matches!(f.service.delete_pins("one"), Err(ServiceError::NotFound(_))));
}

#[test]
fn pins_end_with_their_ttl() {
    let f = Fixture::new();
    let a = built(&f, "a");
    f.service.put_pins("one", &put(&[&a], 100)).unwrap();
    f.advance(200_000);
    assert!(matches!(f.service.get_pins("one"), Err(ServiceError::NotFound(_))));
    assert!(
        f.service.snapshot(&a).unwrap().pins.is_empty(),
        "an ended pin names no owner"
    );
}

#[test]
fn a_bad_ttl_or_a_bad_owner_is_a_client_error_and_changes_nothing() {
    let f = Fixture::new();
    let a = built(&f, "a");
    for ttl in [0, 86_401] {
        let error = f.service.put_pins("one", &put(&[&a], ttl)).unwrap_err();
        assert!(
            matches!(&error, ServiceError::Invalid(i) if i.param.as_deref() == Some("ttl_s")),
            "{error:?}"
        );
    }
    for owner in ["", "a/b", &"k".repeat(129)] {
        assert!(matches!(
            f.service.put_pins(owner, &put(&[&a], 60)),
            Err(ServiceError::Invalid(_))
        ));
    }
    assert_eq!(pinned_by_the_store(&f, &a), 0);
}

#[test]
fn a_set_past_pin_bytes_is_too_large_and_pins_nothing() {
    let mut config = ServiceConfig::new("toy", ENGINE);
    config.pin_bytes = 5000;
    let f = Fixture::with(0, 1 << 20, None, config);
    let a = built(&f, "a");
    let b = built(&f, "b");
    f.service.put_pins("one", &put(&[&a], 3600)).unwrap();
    // One snapshot is about 4.6 KB. A second one for another owner is over 5000 bytes together.
    let error = f.service.put_pins("two", &put(&[&b], 3600)).unwrap_err();
    assert!(
        matches!(&error, ServiceError::TooLarge { param, .. } if param == "pin_bytes"),
        "{error:?}"
    );
    assert_eq!(pinned_by_the_store(&f, &b), 0);
    assert!(matches!(f.service.get_pins("two"), Err(ServiceError::NotFound(_))));
    // The set that the owner had before stays when a replacement is refused.
    let error = f.service.put_pins("one", &put(&[&a, &b], 3600)).unwrap_err();
    assert!(matches!(error, ServiceError::TooLarge { .. }));
    assert_eq!(f.service.get_pins("one").unwrap().snapshots, [a]);
}

#[test]
fn the_pins_are_in_a_file_that_a_second_service_reads() {
    let f = Fixture::new();
    let a = built(&f, "a");
    f.service.put_pins("kernel-1", &put(&[&a], 3600)).unwrap();
    let second = f.attach_again();
    assert_eq!(second.get_pins("kernel-1").unwrap().snapshots, [a]);
}

#[test]
fn a_pin_on_a_snapshot_that_the_store_lost_is_dropped_when_a_service_attaches() {
    let f = Fixture::new();
    let a = built(&f, "a");
    f.service.put_pins("kernel-1", &put(&[&a], 3600)).unwrap();
    // The store forgets the entry, as a restart does for the host tier (spec 6.11). Its pin goes with it.
    {
        let ports = f.process.ports();
        let mut store = ports.store().unwrap().lock().unwrap();
        let content = nagaya_store::ContentId::from_hex(a.as_str().trim_start_matches("blake3:")).unwrap();
        store
            .unpin(&content, nagaya::council::pins::owner_key("kernel-1"))
            .unwrap();
        let size = store.entries()[&content].size;
        let gone = store.evict(1, nagaya_store::Tier::Host, size, 1_000_000).unwrap();
        assert_eq!(gone, [content]);
    }
    let second = f.attach_again();
    assert!(matches!(second.get_pins("kernel-1"), Err(ServiceError::NotFound(_))));
}

#[test]
fn a_snapshot_of_another_engine_is_not_held_by_this_service() {
    use nagaya_abi::fake::FakeAlways;
    use nagaya_abi::{Engine, State};
    use nagaya_store::{EngineFacts, Put, Reason, Tier as StoreTier, recipe_root, snapshot_from_engine};
    let f = Fixture::new();
    let mut other = FakeAlways::new();
    let limits = other.limits();
    let snapshot = snapshot_from_engine(&mut other, State::Slot(1), &[(1, 0, 0), (2, 0, 0)]).unwrap();
    let id = {
        let ports = f.process.ports();
        let mut store = ports.store().unwrap().lock().unwrap();
        store
            .put(Put {
                snapshot: &snapshot,
                recipe: recipe_root(&[9; 32], b"", &[], b""),
                previous_recipe: None,
                engine: EngineFacts::of("another-engine", &limits),
                identity: [9; 32],
                owner: 1,
                tier: StoreTier::Host,
                reason: Reason::Anchor,
                sealed_recipe: None,
                provenance: None,
                now_ms: 1,
            })
            .unwrap()
    };
    let id = SnapshotId::parse(format!("blake3:{}", id.hex())).unwrap();
    assert!(matches!(f.service.snapshot(&id), Err(ServiceError::NotFound(_))));
    let result = f.service.put_pins("one", &put(&[&id], 60)).unwrap();
    assert!(!result.snapshots[0].held);
    assert_eq!(pinned_by_the_store(&f, &id), 0);
}
