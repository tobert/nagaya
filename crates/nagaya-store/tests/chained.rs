//! The index by `chains_from` (spec 11.3c): a build finds a sealed snapshot inside a message from the boundary below
//! it. The index must name the same entries as the main index through each change of the store.

use nagaya_abi::{Determinism, Dtype, PartKind};
use nagaya_store::{
    ContentId, EngineFacts, Put, Reason, RecipeId, Snapshot, Store, Tensor, Tier, recipe_boundary, recipe_root,
};

fn snapshot(seed: u8) -> Snapshot {
    let tensor = |name: &str, kind, shape: Vec<u64>, data: Vec<u8>, coverage| Tensor {
        name: name.to_owned(),
        kind,
        dtype: Dtype::U8,
        shape,
        data,
        coverage,
        block_format: None,
    };
    Snapshot {
        tensors: vec![
            tensor("state", PartKind::Fixed, vec![16], vec![seed; 16], None),
            tensor("cells", PartKind::Cells, vec![2, 8], vec![seed; 16], Some((0, 2))),
        ],
        parent: None,
    }
}

fn engine() -> EngineFacts {
    EngineFacts {
        name: "fake".to_owned(),
        determinism: Determinism::Always,
        bit_dependence: Vec::new(),
        chain: nagaya_abi::ChainLimits {
            max_depth: u32::MAX,
            delta_fraction: 1.0,
        },
    }
}

fn put(store: &mut Store, seed: u8, from: Option<RecipeId>, tier: Tier, now_ms: u64) -> ContentId {
    store
        .put(Put {
            snapshot: &snapshot(seed),
            recipe: recipe_boundary(&recipe_root(&[1; 32], b"", &[], b""), &[seed as u32], 2),
            previous_recipe: None,
            chains_from: from,
            engine: engine(),
            identity: [1; 32],
            owner: 1,
            tier,
            reason: Reason::PrefillDone,
            sealed_recipe: None,
            provenance: None,
            now_ms,
        })
        .unwrap()
}

fn chained(store: &Store, from: &RecipeId) -> Vec<ContentId> {
    let mut ids: Vec<ContentId> = store.chained_from(from).map(|entry| entry.content_id).collect();
    ids.sort();
    ids
}

fn sorted(mut ids: Vec<ContentId>) -> Vec<ContentId> {
    ids.sort();
    ids
}

#[test]
fn the_index_by_chains_from_follows_put_park_evict_and_restart() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = (
        recipe_root(&[2; 32], b"", &[], b""),
        recipe_root(&[3; 32], b"", &[], b""),
    );
    let mut store = Store::open_with_limits(dir.path(), 1 << 20, 1 << 20, u64::MAX, 0).unwrap();
    let host = put(&mut store, 1, Some(a), Tier::Host, 10);
    let nvme = put(&mut store, 2, Some(a), Tier::Nvme, 20);
    let other = put(&mut store, 3, Some(b), Tier::Nvme, 30);
    let none = put(&mut store, 4, None, Tier::Nvme, 40);
    store.check_chained().unwrap();
    assert_eq!(chained(&store, &a), sorted(vec![host, nvme]));
    assert_eq!(chained(&store, &b), [other]);
    assert!(store.entries().contains_key(&none));

    // A park moves the bytes to NVMe and keeps the entry.
    store.park(&host, 50).unwrap();
    store.check_chained().unwrap();
    assert_eq!(chained(&store, &a), sorted(vec![host, nvme]));

    // An eviction of all the NVMe tier removes the chained entries too.
    let bytes: u64 = store
        .entries()
        .values()
        .filter(|entry| entry.tier == Tier::Nvme)
        .map(|entry| entry.size)
        .sum();
    let evicted = store.evict(1, Tier::Nvme, bytes, 60).unwrap();
    assert_eq!(sorted(evicted), sorted(vec![host, nvme, other, none]));
    store.check_chained().unwrap();
    assert_eq!((chained(&store, &a), chained(&store, &b)), (vec![], vec![]));

    // A restart reads the index again, and makes the index by chains_from from it. The host tier does not survive.
    let kept = put(&mut store, 5, Some(a), Tier::Nvme, 70);
    let lost = put(&mut store, 6, Some(a), Tier::Host, 80);
    assert_eq!(chained(&store, &a), sorted(vec![kept, lost]));
    drop(store);
    let store = Store::open_with_limits(dir.path(), 1 << 20, 1 << 20, u64::MAX, 0).unwrap();
    store.check_chained().unwrap();
    assert_eq!(chained(&store, &a), [kept]);
}

#[test]
fn a_put_that_fails_leaves_the_index_by_chains_from_as_it_was() {
    let dir = tempfile::tempdir().unwrap();
    let from = recipe_root(&[2; 32], b"", &[], b"");
    let mut store = Store::open_with_limits(dir.path(), 1 << 20, 1 << 20, u64::MAX, 0).unwrap();
    put(&mut store, 1, Some(from), Tier::Nvme, 10);
    store.fail_next(nagaya_store::FailPoint::IndexWrite);
    let failed = store.put(Put {
        snapshot: &snapshot(2),
        recipe: recipe_boundary(&from, &[2], 4),
        previous_recipe: None,
        chains_from: Some(from),
        engine: engine(),
        identity: [1; 32],
        owner: 1,
        tier: Tier::Nvme,
        reason: Reason::PrefillDone,
        sealed_recipe: None,
        provenance: None,
        now_ms: 20,
    });
    assert!(failed.is_err());
    store.check_chained().unwrap();
    assert_eq!(store.chained_from(&from).count(), 1);
}
