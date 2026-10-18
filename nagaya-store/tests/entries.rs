use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Determinism, Dtype, Engine, OutputsWanted, PartKind, Segment, SegmentInputs, State, WriteResult};
use nagaya_store::{
    Check, ContentId, EntryState, PartCoverage, Put, Reason, RecipeId, Snapshot, Store, Tensor, Tier, Tolerance,
    recipe_root, snapshot_from_engine,
};

fn tensor(name: &str, kind: PartKind, shape: Vec<u64>, data: Vec<u8>, coverage: Option<(u64, u64)>) -> Tensor {
    Tensor {
        name: name.to_owned(),
        kind,
        dtype: Dtype::U8,
        shape,
        data,
        coverage,
        block_format: None,
    }
}

fn base_snapshot(seed: u8) -> Snapshot {
    Snapshot {
        tensors: vec![
            tensor("state", PartKind::Fixed, vec![16], vec![seed; 16], None),
            tensor(
                "cells",
                PartKind::Cells,
                vec![2, 8],
                (0u8..16).map(|value| value.wrapping_add(seed)).collect(),
                Some((0, 2)),
            ),
        ],
        parent: None,
    }
}

fn wide_snapshot(seed: u8) -> Snapshot {
    Snapshot {
        tensors: vec![
            tensor("state", PartKind::Fixed, vec![16], vec![seed; 16], None),
            tensor(
                "cells",
                PartKind::Cells,
                vec![3, 8],
                (0u8..24).map(|value| value.wrapping_add(seed)).collect(),
                Some((0, 3)),
            ),
        ],
        parent: None,
    }
}

fn delta_at(parent: ContentId, start: u64) -> Snapshot {
    let seed = start as u8;
    Snapshot {
        tensors: vec![
            tensor("state", PartKind::Fixed, vec![16], vec![seed; 16], None),
            tensor(
                "cells",
                PartKind::Cells,
                vec![2, 8],
                (seed..seed + 16).collect(),
                Some((start, start + 2)),
            ),
        ],
        parent: Some(parent),
    }
}

fn delta_snapshot(parent: ContentId) -> Snapshot {
    delta_at(parent, 2)
}

fn recipe(seed: u8) -> RecipeId {
    recipe_root(&[seed; 32], b"", &[], b"")
}

fn put(store: &mut Store, snapshot: &Snapshot, recipe: RecipeId, owner: u64) -> ContentId {
    store
        .put(Put {
            snapshot,
            recipe,
            identity: [1u8; 32],
            determinism: Determinism::Always,
            owner,
            tier: Tier::Host,
            reason: Reason::Anchor,
            provenance: None,
            now_ms: 10,
        })
        .expect("put")
}

#[test]
fn an_entry_records_its_facts() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let snapshot = base_snapshot(1);
    let recipe = recipe(1);
    let id = store
        .put(Put {
            snapshot: &snapshot,
            recipe,
            identity: [9u8; 32],
            determinism: Determinism::Always,
            owner: 7,
            tier: Tier::Device(0),
            reason: Reason::Interval,
            provenance: Some("exact".to_owned()),
            now_ms: 10,
        })
        .unwrap();
    let entry = store.entries().get(&id).unwrap();
    assert_eq!(entry.content_id, id);
    assert_eq!(entry.recipe, recipe);
    assert_eq!(entry.parent, None);
    assert_eq!(
        entry.coverage,
        vec![PartCoverage {
            part: "cells".to_owned(),
            start: 0,
            end: 2,
        }]
    );
    assert_eq!(entry.tier, Tier::Device(0));
    assert!(entry.size > 0);
    assert_eq!(entry.owner, 7);
    assert_eq!(entry.created_ms, 10);
    assert_eq!(entry.hits, 0);
    assert_eq!(entry.reason, Reason::Interval);
    assert_eq!(entry.provenance.as_deref(), Some("exact"));
    assert_eq!(entry.check, Check::Full);
    assert_eq!(entry.identity, [9u8; 32]);
    assert_eq!(entry.state, EntryState::Live);
}

#[test]
fn a_delta_keeps_its_parent_and_the_base() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(2);
    let base = put(&mut store, &base_snapshot(1), recipe, 0);
    let delta = put(&mut store, &delta_snapshot(base), recipe, 0);
    assert_eq!(store.entries().get(&delta).unwrap().parent, Some(base));
    let chain = store.lookup(recipe, &Tolerance::Exact, 20).unwrap();
    assert_eq!(chain, vec![base, delta]);
    let victims = store.evict(0, 1, 20).unwrap();
    assert!(!victims.contains(&base));
    assert!(victims.contains(&delta));
    assert!(store.entries().contains_key(&base));
}

#[test]
fn lookup_returns_the_deepest_chain() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(3);
    let short = put(&mut store, &base_snapshot(1), recipe, 0);
    let wide = put(&mut store, &wide_snapshot(2), recipe, 0);
    let delta = put(&mut store, &delta_at(short, 2), recipe, 0);
    let chain = store.lookup(recipe, &Tolerance::Exact, 20).unwrap();
    assert_eq!(chain, vec![short, delta]);
    assert!(!chain.contains(&wide));
}

#[test]
fn the_tripwire_quarantines_both() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(4);
    let first = put(&mut store, &base_snapshot(1), recipe, 0);
    let second = store.put(Put {
        snapshot: &base_snapshot(2),
        recipe,
        identity: [1u8; 32],
        determinism: Determinism::Always,
        owner: 0,
        tier: Tier::Host,
        reason: Reason::Anchor,
        provenance: None,
        now_ms: 10,
    });
    assert!(second.is_err());
    assert_eq!(store.entries().get(&first).unwrap().state, EntryState::Quarantined);
    assert!(store.lookup(recipe, &Tolerance::Exact, 20).is_err());
}

#[test]
fn restore_writes_the_parts_and_verifies_the_content_id() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let mut source = FakeAlways::new();
    source.pass(&[segment(&[1, 2, 3, 4, 5, 6])], &[]).unwrap();
    let snapshot = snapshot_from_engine(&mut source, State::Slot(0), &[(1, 0, 6), (2, 2, 4)]).unwrap();
    let id = put(&mut store, &snapshot, recipe(5), 0);
    let mut target = FakeAlways::new();
    let results = store.restore(&id, &mut target, State::Slot(0), &[1u8; 32], 20).unwrap();
    assert_eq!(results, vec![WriteResult::Ok, WriteResult::Ok, WriteResult::Ok]);
    for tensor in &snapshot.tensors {
        let index = target
            .limits()
            .parts
            .iter()
            .position(|part| part.name == tensor.name)
            .unwrap();
        let mut bytes = vec![0u8; tensor.data.len()];
        match tensor.kind {
            PartKind::Fixed => {
                target
                    .parts()
                    .unwrap()
                    .part_read(State::Slot(0), index, &mut bytes)
                    .unwrap();
            }
            PartKind::Cells | PartKind::Ring => {
                let (start, end) = tensor.coverage.unwrap();
                let from = if tensor.kind == PartKind::Ring { 0 } else { start };
                target
                    .parts()
                    .unwrap()
                    .cells_read(State::Slot(0), index, from, end - start, &mut bytes)
                    .unwrap();
            }
            PartKind::Derived => continue,
        }
        assert_eq!(bytes, tensor.data);
    }

    drop(store);
    let blob = dir.path().join("blobs").join(id.hex());
    std::fs::write(&blob, b"corrupt").unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let mut target = FakeAlways::new();
    assert!(store.restore(&id, &mut target, State::Slot(0), &[1u8; 32], 30).is_err());
    assert_eq!(store.entries().get(&id).unwrap().state, EntryState::Quarantined);
    assert!(blob.exists());
}

#[test]
fn eviction_skips_pins_and_thins_interiors() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(6);
    let root = put(&mut store, &base_snapshot(1), recipe, 1);
    let middle = put(&mut store, &delta_at(root, 2), recipe, 1);
    let tip = put(&mut store, &delta_at(middle, 4), recipe, 1);
    store.pin(root, 1, 100).unwrap();
    let victims = store.evict(1, 20_000, 10).unwrap();
    assert!(!victims.contains(&root));
    assert_eq!(victims.first(), Some(&middle));
    assert!(victims.contains(&tip));
    assert!(store.entries().contains_key(&root));
}

#[test]
fn an_owner_does_not_evict_another() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(7), 1);
    let second = put(&mut store, &wide_snapshot(2), recipe(8), 2);
    let victims = store.evict(1, 1, 10).unwrap();
    assert_eq!(victims, vec![first]);
    assert!(store.entries().contains_key(&second));
}

#[test]
fn a_pin_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let mut store = Store::open(dir.path()).unwrap();
        id = put(&mut store, &base_snapshot(1), recipe(8), 1);
        store.pin(id, 1, 100).unwrap();
    }
    let mut store = Store::open(dir.path()).unwrap();
    assert!(store.pins().contains_key(&id));
    let victims = store.evict(1, 1, 10).unwrap();
    assert!(victims.is_empty());
    assert!(store.entries().contains_key(&id));
}

#[test]
fn a_restart_rebuilds_the_index_and_checks_entries() {
    let dir = tempfile::tempdir().unwrap();
    let (first, second);
    {
        let mut store = Store::open(dir.path()).unwrap();
        first = put(&mut store, &base_snapshot(1), recipe(9), 1);
        second = put(&mut store, &base_snapshot(2), recipe(10), 1);
    }
    let mut store = Store::open(dir.path()).unwrap();
    assert_eq!(store.entries().len(), 2);
    let blob = dir.path().join("blobs").join(first.hex());
    std::fs::write(&blob, b"corrupt").unwrap();
    assert!(store.get(&first, 10).is_err());
    assert_eq!(store.entries().get(&first).unwrap().state, EntryState::Quarantined);
    assert!(store.get(&second, 10).unwrap().is_some());
}

fn segment(tokens: &[u32]) -> Segment {
    Segment {
        state: State::Slot(0),
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (0..tokens.len() as u64).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted::default(),
    }
}

fn dual_snapshot(cells_end: u64, ring_end: u64, seed: u8) -> Snapshot {
    let rows =
        |count: u64, value: u8| -> Vec<u8> { (0..count * 8).map(|index| value.wrapping_add(index as u8)).collect() };
    Snapshot {
        tensors: vec![
            tensor("state", PartKind::Fixed, vec![16], vec![seed; 16], None),
            tensor(
                "cells",
                PartKind::Cells,
                vec![cells_end, 8],
                rows(cells_end, seed),
                Some((0, cells_end)),
            ),
            tensor(
                "ring",
                PartKind::Ring,
                vec![ring_end, 8],
                rows(ring_end, seed.wrapping_add(1)),
                Some((0, ring_end)),
            ),
        ],
        parent: None,
    }
}

#[test]
fn a_sum_does_not_hide_a_lower_part() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(20);
    put(&mut store, &dual_snapshot(1, 5, 1), recipe, 0);
    put(&mut store, &dual_snapshot(3, 2, 2), recipe, 0);
    let error = store.lookup(recipe, &Tolerance::Exact, 10).unwrap_err();
    assert!(matches!(error, nagaya_store::StoreError::Ambiguous));
}

#[test]
fn a_chain_higher_in_every_part_is_chosen() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(21);
    let deeper = put(&mut store, &dual_snapshot(5, 5, 1), recipe, 0);
    put(&mut store, &dual_snapshot(3, 4, 2), recipe, 0);
    assert_eq!(store.lookup(recipe, &Tolerance::Exact, 10).unwrap(), vec![deeper]);
}

#[test]
fn a_quarantined_entry_refuses_its_recipe_only() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let bad_recipe = recipe(22);
    let first = put(&mut store, &base_snapshot(1), bad_recipe, 0);
    assert!(
        store
            .put(Put {
                snapshot: &base_snapshot(2),
                recipe: bad_recipe,
                identity: [1u8; 32],
                determinism: Determinism::Always,
                owner: 0,
                tier: Tier::Host,
                reason: Reason::Anchor,
                provenance: None,
                now_ms: 10,
            })
            .is_err()
    );
    put(&mut store, &base_snapshot(3), bad_recipe, 0);
    let good_recipe = recipe(23);
    let good = put(&mut store, &base_snapshot(4), good_recipe, 0);
    assert!(matches!(
        store.lookup(bad_recipe, &Tolerance::Exact, 10),
        Err(nagaya_store::StoreError::Tripwire)
    ));
    assert_eq!(store.lookup(good_recipe, &Tolerance::Exact, 10).unwrap(), vec![good]);
    assert_eq!(store.entries().get(&first).unwrap().state, EntryState::Quarantined);
}

#[test]
fn a_put_that_fits_writes_no_victims() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_budget(dir.path(), 10_000).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(24), 1);
    let second = put(&mut store, &base_snapshot(2), recipe(35), 1);
    assert!(store.entries().contains_key(&first));
    assert!(store.entries().contains_key(&second));
    assert_eq!(store.entries().len(), 2);
}

#[test]
fn a_put_over_budget_evicts_first() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_budget(dir.path(), 5_000).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(25), 1);
    let second = put(&mut store, &base_snapshot(2), recipe(26), 1);
    assert!(!store.entries().contains_key(&first));
    assert!(store.entries().contains_key(&second));
    assert!(!dir.path().join("blobs").join(first.hex()).exists());
    assert!(dir.path().join("blobs").join(second.hex()).exists());
}

#[test]
fn a_put_that_cannot_fit_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_budget(dir.path(), 10_000).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(27), 1);
    let second = put(&mut store, &base_snapshot(2), recipe(28), 1);
    store.pin(second, 1, 100).unwrap();
    store.set_budget(5_000);
    let before: Vec<String> = std::fs::read_dir(dir.path().join("blobs"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    let error = store.put(Put {
        snapshot: &base_snapshot(3),
        recipe: recipe(29),
        identity: [1u8; 32],
        determinism: Determinism::Always,
        owner: 1,
        tier: Tier::Host,
        reason: Reason::Anchor,
        provenance: None,
        now_ms: 10,
    });
    assert!(matches!(error, Err(nagaya_store::StoreError::NoSpace)));
    let after: Vec<String> = std::fs::read_dir(dir.path().join("blobs"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(before, after);
    assert!(store.entries().contains_key(&first));
    assert!(store.entries().contains_key(&second));
    assert!(dir.path().join("blobs").join(first.hex()).exists());
    assert!(dir.path().join("blobs").join(second.hex()).exists());
}

#[test]
fn a_failed_put_keeps_the_index_and_blobs() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(30), 1);
    for point in [
        nagaya_store::FailPoint::BlobWrite,
        nagaya_store::FailPoint::BlobRename,
        nagaya_store::FailPoint::IndexWrite,
        nagaya_store::FailPoint::IndexRename,
    ] {
        let before: Vec<String> = std::fs::read_dir(dir.path().join("blobs"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        store.fail_next(point);
        let error = store.put(Put {
            snapshot: &base_snapshot(2),
            recipe: recipe(31),
            identity: [1u8; 32],
            determinism: Determinism::Always,
            owner: 1,
            tier: Tier::Host,
            reason: Reason::Anchor,
            provenance: None,
            now_ms: 10,
        });
        assert!(error.is_err());
        assert_eq!(store.entries().len(), 1);
        assert!(store.entries().contains_key(&first));
        let after: Vec<String> = std::fs::read_dir(dir.path().join("blobs"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(before, after);
    }
}

#[test]
fn a_pin_larger_than_the_budget_loses_protection() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_budget(dir.path(), 10_000).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(32), 1);
    store.pin(id, 1, 100).unwrap();
    store.set_budget(100);
    let victims = store.evict(1, 1, 10).unwrap();
    assert_eq!(victims, vec![id]);
    assert!(!store.entries().contains_key(&id));
}

#[test]
fn a_request_pin_protects_until_its_expiry() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(33);
    let id = put(&mut store, &base_snapshot(1), recipe, 1);
    store.lookup_and_pin(recipe, &Tolerance::Exact, 1, 100, 10).unwrap();
    assert!(store.evict(1, 1, 10).unwrap().is_empty());
    assert_eq!(store.evict(1, 1, 101).unwrap(), vec![id]);
}

#[test]
fn another_owners_pin_protects() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(34), 1);
    store.pin(id, 2, 100).unwrap();
    assert!(store.evict(1, 1, 10).unwrap().is_empty());
    assert!(store.entries().contains_key(&id));
}
