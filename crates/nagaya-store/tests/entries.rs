use std::path::Path;

use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    BitDependence, ChainLimits, Dtype, Engine, OutputsWanted, PartKind, Segment, SegmentInputs, SlotLimit, State,
    WriteResult,
};
use nagaya_store::{
    Check, ContentId, EntryState, PartCoverage, Pin, Put, Quarantine, Reason, RecipeId, Snapshot, Store, StoreError,
    Tensor, Tier, Tolerance, recipe_boundary, recipe_root, snapshot_from_engine,
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
    delta_with(parent, start, start as u8)
}

fn delta_with(parent: ContentId, start: u64, seed: u8) -> Snapshot {
    Snapshot {
        tensors: vec![
            tensor("state", PartKind::Fixed, vec![16], vec![seed; 16], None),
            tensor(
                "cells",
                PartKind::Cells,
                vec![2, 8],
                (seed..seed.wrapping_add(16)).collect(),
                Some((start, start + 2)),
            ),
        ],
        parent: Some(parent),
    }
}

fn recipe(seed: u8) -> RecipeId {
    recipe_root(&[seed; 32], b"", &[], b"")
}

fn next_recipe(previous: RecipeId, seed: u8) -> RecipeId {
    recipe_boundary(&previous, &[seed as u32], 2)
}

fn open_chain() -> ChainLimits {
    ChainLimits {
        max_depth: u32::MAX,
        delta_fraction: 1.0,
    }
}

fn try_put(
    store: &mut Store,
    snapshot: &Snapshot,
    recipe: RecipeId,
    previous_recipe: Option<RecipeId>,
    owner: u64,
    tier: Tier,
    now_ms: u64,
) -> Result<ContentId, StoreError> {
    store.put(Put {
        snapshot,
        recipe,
        previous_recipe,
        identity: [1u8; 32],
        owner,
        tier,
        reason: Reason::Anchor,
        provenance: None,
        now_ms,
    })
}

fn put(store: &mut Store, snapshot: &Snapshot, recipe: RecipeId, owner: u64) -> ContentId {
    try_put(store, snapshot, recipe, None, owner, Tier::Host, 10).expect("put")
}

fn put_delta(store: &mut Store, snapshot: &Snapshot, recipe: RecipeId, previous: RecipeId, owner: u64) -> ContentId {
    try_put(store, snapshot, recipe, Some(previous), owner, Tier::Host, 10).expect("put delta")
}

fn put_provenance(
    store: &mut Store,
    snapshot: &Snapshot,
    recipe: RecipeId,
    previous: Option<RecipeId>,
    provenance: &str,
) -> ContentId {
    store
        .put(Put {
            snapshot,
            recipe,
            previous_recipe: previous,
            identity: [1u8; 32],
            owner: 1,
            tier: Tier::Host,
            reason: Reason::Anchor,
            provenance: Some(provenance.to_owned()),
            now_ms: 10,
        })
        .expect("put provenance")
}

fn lookup(store: &mut Store, recipe: RecipeId) -> Result<Vec<ContentId>, StoreError> {
    let mut engine = FakeAlways::new();
    store.lookup(recipe, &Tolerance::Exact, &mut engine, 0, 10)
}

fn lookup_with(store: &mut Store, recipe: RecipeId, tolerance: &Tolerance) -> Result<Vec<ContentId>, StoreError> {
    let mut engine = FakeAlways::new();
    store.lookup(recipe, tolerance, &mut engine, 0, 10)
}

fn edit_index(dir: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let path = dir.join("index.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
    edit(&mut value);
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
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
            previous_recipe: None,
            identity: [9u8; 32],
            owner: 7,
            tier: Tier::Nvme,
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
    assert_eq!(entry.tier, Tier::Nvme);
    assert!(entry.size > 0);
    assert_eq!(entry.owner, 7);
    assert_eq!(entry.created_ms, 10);
    assert_eq!(entry.hits, 0);
    assert_eq!(entry.reason, Reason::Interval);
    assert_eq!(entry.provenance.as_deref(), Some("exact"));
    assert_eq!(entry.check, Check::Full);
    assert_eq!(entry.identity, [9u8; 32]);
    assert_eq!(entry.state, EntryState::Live);
    assert_eq!(entry.quarantine, None);
}

#[test]
fn a_delta_keeps_its_parent_and_the_base() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let base_recipe = recipe(2);
    let base = put(&mut store, &base_snapshot(1), base_recipe, 0);
    let delta_recipe = next_recipe(base_recipe, 2);
    let delta = put_delta(&mut store, &delta_at(base, 2), delta_recipe, base_recipe, 0);
    assert_eq!(store.entries().get(&delta).unwrap().parent, Some(base));
    let chain = lookup(&mut store, delta_recipe).unwrap();
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
    let root_recipe = recipe(3);
    let short = put(&mut store, &base_snapshot(1), root_recipe, 0);
    let wide = put(&mut store, &wide_snapshot(2), root_recipe, 0);
    let delta_recipe = next_recipe(root_recipe, 2);
    let delta = put_delta(&mut store, &delta_at(short, 2), delta_recipe, root_recipe, 0);
    let chain = lookup(&mut store, delta_recipe).unwrap();
    assert_eq!(chain, vec![short, delta]);
    assert!(!chain.contains(&wide));
}

#[test]
fn the_tripwire_quarantines_both() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(4);
    let first = put(&mut store, &base_snapshot(1), recipe, 0);
    let error = try_put(&mut store, &base_snapshot(2), recipe, None, 0, Tier::Host, 10).unwrap_err();
    let ids = match error {
        StoreError::Tripwire { ids, .. } => ids,
        other => panic!("expected the tripwire, got {other:?}"),
    };
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&first));
    for entry in store.entries().values() {
        assert_eq!(entry.state, EntryState::Quarantined);
        assert_eq!(entry.quarantine, Some(Quarantine::Tripwire));
    }
    assert_eq!(store.entries().len(), 2);
    assert!(store.refused().contains(&recipe));
    assert!(matches!(lookup(&mut store, recipe), Err(StoreError::Tripwire { .. })));
    assert!(matches!(
        try_put(&mut store, &base_snapshot(3), recipe, None, 0, Tier::Host, 10),
        Err(StoreError::Tripwire { .. })
    ));
}

#[test]
fn a_person_clears_a_tripwire() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(40);
    let first = put(&mut store, &base_snapshot(1), recipe, 0);
    assert!(try_put(&mut store, &base_snapshot(2), recipe, None, 0, Tier::Host, 10).is_err());
    assert!(store.clear_entry(&first).is_err());
    store.clear_refusal(recipe).unwrap();
    assert!(!store.entries().contains_key(&first));
    let fresh = put(&mut store, &base_snapshot(3), recipe, 0);
    assert_eq!(lookup(&mut store, recipe).unwrap(), vec![fresh]);
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
    let entry = store.entries().get(&id).unwrap();
    assert_eq!(entry.state, EntryState::Quarantined);
    assert_eq!(entry.quarantine, Some(Quarantine::ContentId));
    assert!(blob.exists());
}

#[test]
fn eviction_skips_pins_and_thins_interiors() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let root_recipe = recipe(6);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let middle_recipe = next_recipe(root_recipe, 2);
    let middle = put_delta(&mut store, &delta_at(root, 2), middle_recipe, root_recipe, 1);
    let tip_recipe = next_recipe(middle_recipe, 4);
    let tip = put_delta(&mut store, &delta_at(middle, 4), tip_recipe, middle_recipe, 1);
    store.pin(root, 1, 100, 10).unwrap();
    let victims = store.evict(1, 1, 10).unwrap();
    assert!(!victims.contains(&root));
    assert_eq!(victims.first(), Some(&middle));
    assert!(victims.contains(&tip));
    assert!(store.entries().contains_key(&root));
}

#[test]
fn an_interior_eviction_takes_every_child() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let root_recipe = recipe(60);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let middle_recipe = next_recipe(root_recipe, 2);
    let middle = put_delta(&mut store, &delta_at(root, 2), middle_recipe, root_recipe, 1);
    let first_recipe = next_recipe(middle_recipe, 4);
    let first = put_delta(&mut store, &delta_with(middle, 4, 4), first_recipe, middle_recipe, 1);
    let second_recipe = next_recipe(middle_recipe, 5);
    let second = put_delta(&mut store, &delta_with(middle, 4, 5), second_recipe, middle_recipe, 1);
    let victims = store.evict(1, 1, 10).unwrap();
    assert!(victims.contains(&middle));
    assert!(victims.contains(&first));
    assert!(victims.contains(&second));
    assert!(store.entries().contains_key(&root));
    assert!(lookup(&mut store, first_recipe).unwrap().is_empty());
    assert!(lookup(&mut store, second_recipe).unwrap().is_empty());
}

#[test]
fn a_pin_on_a_child_protects_its_parent() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let root_recipe = recipe(61);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let child_recipe = next_recipe(root_recipe, 2);
    let child = put_delta(&mut store, &delta_at(root, 2), child_recipe, root_recipe, 1);
    store.pin(child, 1, 100, 10).unwrap();
    assert!(matches!(store.evict(1, 1, 10), Err(StoreError::NoSpace)));
    assert!(store.entries().contains_key(&root));
    assert!(store.entries().contains_key(&child));
}

#[test]
fn a_cross_tier_child_keeps_the_parent() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_limits(dir.path(), 5_000, 5_000, u64::MAX, open_chain()).unwrap();
    let root_recipe = recipe(162);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let child_recipe = next_recipe(root_recipe, 2);
    let child = try_put(
        &mut store,
        &delta_at(root, 2),
        child_recipe,
        Some(root_recipe),
        1,
        Tier::Nvme,
        10,
    )
    .unwrap();
    let error = try_put(&mut store, &base_snapshot(3), recipe(163), None, 1, Tier::Host, 10).unwrap_err();
    assert!(matches!(error, StoreError::NoSpace));
    assert!(store.entries().contains_key(&root));
    assert!(store.entries().contains_key(&child));
}

#[test]
fn another_owners_child_keeps_the_parent() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let root_recipe = recipe(62);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let child_recipe = next_recipe(root_recipe, 2);
    let child = put_delta(&mut store, &delta_at(root, 2), child_recipe, root_recipe, 2);
    assert!(matches!(store.evict(1, 1, 10), Err(StoreError::NoSpace)));
    assert!(store.entries().contains_key(&root));
    assert!(store.entries().contains_key(&child));
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
        store.pin(id, 1, 100, 10).unwrap();
    }
    let mut store = Store::open(dir.path()).unwrap();
    assert!(store.pins().contains_key(&id));
    assert!(matches!(store.evict(1, 1, 10), Err(StoreError::NoSpace)));
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
    assert_eq!(store.entries().get(&first).unwrap().check, Check::None);
    let blob = dir.path().join("blobs").join(first.hex());
    std::fs::write(&blob, b"corrupt").unwrap();
    assert!(store.get(&first, 10).is_err());
    assert_eq!(store.entries().get(&first).unwrap().state, EntryState::Quarantined);
    assert!(store.get(&second, 10).unwrap().is_some());
    assert_eq!(store.entries().get(&second).unwrap().check, Check::Full);
}

#[test]
fn a_quarantined_parent_skips_only_its_chain() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let first_root_recipe = recipe(160);
    let second_root_recipe = recipe(161);
    let first_root = put(&mut store, &base_snapshot(1), first_root_recipe, 1);
    let second_root = put(&mut store, &base_snapshot(2), second_root_recipe, 1);
    let first_delta_recipe = next_recipe(first_root_recipe, 2);
    put_delta(
        &mut store,
        &delta_at(first_root, 2),
        first_delta_recipe,
        first_root_recipe,
        1,
    );
    let second_delta_recipe = next_recipe(second_root_recipe, 3);
    let second_delta = put_delta(
        &mut store,
        &delta_at(second_root, 2),
        second_delta_recipe,
        second_root_recipe,
        1,
    );
    let blob = dir.path().join("blobs").join(first_root.hex());
    std::fs::write(&blob, b"corrupt").unwrap();
    assert!(store.get(&first_root, 10).is_err());
    assert_eq!(store.entries().get(&first_root).unwrap().state, EntryState::Quarantined);
    assert!(lookup(&mut store, first_delta_recipe).unwrap().is_empty());
    assert_eq!(
        lookup(&mut store, second_delta_recipe).unwrap(),
        vec![second_root, second_delta]
    );
}

#[test]
fn a_chain_with_a_missing_parent_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let root_recipe = recipe(63);
    let delta_recipe;
    let (root, delta);
    {
        let mut store = Store::open(dir.path()).unwrap();
        root = put(&mut store, &base_snapshot(1), root_recipe, 1);
        delta_recipe = next_recipe(root_recipe, 2);
        delta = put_delta(&mut store, &delta_at(root, 2), delta_recipe, root_recipe, 1);
    }
    edit_index(dir.path(), |index| {
        index["entries"].as_object_mut().unwrap().remove(&root.hex());
    });
    let mut store = Store::open(dir.path()).unwrap();
    let error = lookup(&mut store, delta_recipe).unwrap_err();
    assert!(format!("{error}").contains("missing parent"), "{error}");
    let mut target = FakeAlways::new();
    let error = store
        .restore(&delta, &mut target, State::Slot(0), &[1u8; 32], 10)
        .unwrap_err();
    assert!(format!("{error}").contains("missing parent"), "{error}");
}

#[test]
fn a_chain_cycle_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let root_recipe = recipe(64);
    let delta_recipe;
    let (root, delta);
    {
        let mut store = Store::open(dir.path()).unwrap();
        root = put(&mut store, &base_snapshot(1), root_recipe, 1);
        delta_recipe = next_recipe(root_recipe, 2);
        delta = put_delta(&mut store, &delta_at(root, 2), delta_recipe, root_recipe, 1);
    }
    edit_index(dir.path(), |index| {
        index["entries"][root.hex()]["parent"] = serde_json::json!(delta.hex());
    });
    let mut store = Store::open(dir.path()).unwrap();
    let error = lookup(&mut store, delta_recipe).unwrap_err();
    assert!(format!("{error}").contains("cycle"), "{error}");
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
    let error = lookup(&mut store, recipe).unwrap_err();
    assert!(matches!(error, nagaya_store::StoreError::Ambiguous));
}

#[test]
fn a_chain_higher_in_every_part_is_chosen() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(21);
    let deeper = put(&mut store, &dual_snapshot(5, 5, 1), recipe, 0);
    put(&mut store, &dual_snapshot(3, 4, 2), recipe, 0);
    assert_eq!(lookup(&mut store, recipe).unwrap(), vec![deeper]);
}

#[test]
fn a_quarantined_entry_refuses_its_recipe_only() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let bad_recipe = recipe(22);
    let first = put(&mut store, &base_snapshot(1), bad_recipe, 0);
    assert!(try_put(&mut store, &base_snapshot(2), bad_recipe, None, 0, Tier::Host, 10).is_err());
    let good_recipe = recipe(23);
    let good = put(&mut store, &base_snapshot(4), good_recipe, 0);
    assert!(matches!(
        lookup(&mut store, bad_recipe),
        Err(StoreError::Tripwire { .. })
    ));
    assert_eq!(lookup(&mut store, good_recipe).unwrap(), vec![good]);
    assert_eq!(store.entries().get(&first).unwrap().state, EntryState::Quarantined);
}

#[test]
fn a_content_id_quarantine_leaves_the_recipe_servable() {
    let dir = tempfile::tempdir().unwrap();
    let recipe = recipe(120);
    let (a, b);
    {
        let mut store = Store::open(dir.path()).unwrap();
        a = put(&mut store, &base_snapshot(1), recipe, 1);
        b = put(&mut store, &wide_snapshot(2), recipe, 1);
    }
    let blob = dir.path().join("blobs").join(a.hex());
    std::fs::write(&blob, b"corrupt").unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let chain = lookup(&mut store, recipe).unwrap();
    assert_eq!(chain, vec![b]);
    let entry = store.entries().get(&a).unwrap();
    assert_eq!(entry.state, EntryState::Quarantined);
    assert_eq!(entry.quarantine, Some(Quarantine::ContentId));
    assert!(!store.refused().contains(&recipe));
    store.clear_refusal(recipe).unwrap();
    assert_eq!(store.entries().get(&a).unwrap().state, EntryState::Quarantined);

    std::fs::write(&blob, base_snapshot(1).encode().unwrap()).unwrap();
    store.clear_entry(&a).unwrap();
    assert_eq!(store.entries().get(&a).unwrap().state, EntryState::Live);
    assert_eq!(store.entries().get(&a).unwrap().quarantine, None);
    assert_eq!(lookup(&mut store, recipe).unwrap(), vec![b]);
}

#[test]
fn a_put_does_not_store_a_quarantined_id_again() {
    let dir = tempfile::tempdir().unwrap();
    let recipe = recipe(121);
    let id;
    {
        let mut store = Store::open(dir.path()).unwrap();
        id = put(&mut store, &base_snapshot(1), recipe, 1);
    }
    let blob = dir.path().join("blobs").join(id.hex());
    std::fs::write(&blob, b"corrupt").unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    assert!(store.get(&id, 10).is_err());
    let error = try_put(&mut store, &base_snapshot(1), recipe, None, 1, Tier::Host, 10).unwrap_err();
    assert!(matches!(error, StoreError::Quarantined(quarantined) if quarantined == id));
}

#[test]
fn an_identity_mismatch_quarantines_the_entry() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(122), 1);
    let mut target = FakeAlways::new();
    let error = store
        .restore(&id, &mut target, State::Slot(0), &[9u8; 32], 10)
        .unwrap_err();
    assert!(matches!(error, StoreError::Quarantined(quarantined) if quarantined == id));
    let entry = store.entries().get(&id).unwrap();
    assert_eq!(entry.quarantine, Some(Quarantine::Identity));
}

#[test]
fn a_failed_fit_does_not_touch_the_entry() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(123), 1);
    let mut target = FakeAlways::with_slots(vec![SlotLimit { bytes: 8, count: 1 }]);
    assert!(matches!(
        store.restore(&id, &mut target, State::Slot(0), &[1u8; 32], 30),
        Err(StoreError::Fit(_))
    ));
    let entry = store.entries().get(&id).unwrap();
    assert_eq!(entry.hits, 0);
    assert_eq!(entry.last_used_ms, 10);
}

#[test]
fn a_refused_restore_does_not_touch_the_entry() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let snapshot = Snapshot {
        tensors: vec![
            tensor("state", PartKind::Fixed, vec![16], vec![1; 16], None),
            // The fake's cells part is not writable as a whole, so this write refuses.
            tensor("cells", PartKind::Fixed, vec![16], vec![0; 16], None),
        ],
        parent: None,
    };
    let id = put(&mut store, &snapshot, recipe(153), 1);
    let mut target = FakeAlways::new();
    let results = store.restore(&id, &mut target, State::Slot(0), &[1u8; 32], 30).unwrap();
    assert_eq!(results, vec![WriteResult::Refused, WriteResult::Ok]);
    let entry = store.entries().get(&id).unwrap();
    assert_eq!(entry.hits, 0);
    assert_eq!(entry.last_used_ms, 10);
}

#[test]
fn restore_writes_a_chain_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let root_recipe = recipe(124);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let delta_recipe = next_recipe(root_recipe, 2);
    let delta = put_delta(&mut store, &delta_at(root, 2), delta_recipe, root_recipe, 1);
    let mut target = FakeAlways::new();
    let results = store
        .restore(&delta, &mut target, State::Slot(0), &[1u8; 32], 20)
        .unwrap();
    assert_eq!(
        results,
        vec![WriteResult::Ok, WriteResult::Ok, WriteResult::Ok, WriteResult::Ok]
    );
    let mut rows = vec![0u8; 32];
    target
        .parts()
        .unwrap()
        .cells_read(State::Slot(0), 1, 0, 4, &mut rows)
        .unwrap();
    let base = base_snapshot(1);
    let with_delta = delta_at(root, 2);
    let mut wanted = base.tensors[1].data.clone();
    wanted.extend_from_slice(&with_delta.tensors[1].data);
    assert_eq!(rows, wanted);
    assert_eq!(store.entries().get(&delta).unwrap().hits, 1);
}

#[test]
fn a_restore_checks_the_chain_fit() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let root_recipe = recipe(125);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let delta_recipe = next_recipe(root_recipe, 2);
    let delta = put_delta(&mut store, &delta_at(root, 2), delta_recipe, root_recipe, 1);
    let mut target = FakeAlways::with_slots(vec![SlotLimit { bytes: 24, count: 1 }]);
    assert!(
        store
            .restore(&root, &mut target, State::Slot(0), &[1u8; 32], 20)
            .is_ok()
    );
    assert!(matches!(
        store.restore(&delta, &mut target, State::Slot(0), &[1u8; 32], 20),
        Err(StoreError::Fit(_))
    ));
}

#[test]
fn a_lookup_keeps_only_chains_that_fit_the_slot() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let root_recipe = recipe(126);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let delta_recipe = next_recipe(root_recipe, 2);
    let delta = put_delta(&mut store, &delta_at(root, 2), delta_recipe, root_recipe, 1);
    let mut small = FakeAlways::with_slots(vec![SlotLimit { bytes: 24, count: 1 }]);
    assert_eq!(
        store.lookup(root_recipe, &Tolerance::Exact, &mut small, 0, 10).unwrap(),
        vec![root]
    );
    assert!(
        store
            .lookup(delta_recipe, &Tolerance::Exact, &mut small, 0, 10)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        store
            .lookup(delta_recipe, &Tolerance::Exact, &mut FakeAlways::new(), 0, 10)
            .unwrap(),
        vec![root, delta]
    );
}

#[test]
fn an_exact_lookup_rejects_an_approximate_base() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let root_recipe = recipe(127);
    let root = put_provenance(&mut store, &base_snapshot(1), root_recipe, None, "lossy");
    let delta_recipe = next_recipe(root_recipe, 2);
    let delta = put_delta(&mut store, &delta_at(root, 2), delta_recipe, root_recipe, 1);
    assert!(lookup(&mut store, delta_recipe).unwrap().is_empty());
    assert_eq!(
        lookup_with(&mut store, delta_recipe, &Tolerance::Approx(vec!["lossy".to_owned()])).unwrap(),
        vec![root, delta]
    );
    assert!(
        lookup_with(&mut store, delta_recipe, &Tolerance::Approx(vec!["other".to_owned()]))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn a_restart_makes_a_per_process_entry_not_exact() {
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let mut store = Store::open(dir.path()).unwrap();
        store.set_determinism(nagaya_abi::Determinism::PerProcess, Vec::new());
        id = put(&mut store, &base_snapshot(1), recipe(128), 1);
        assert_eq!(lookup(&mut store, recipe(128)).unwrap(), vec![id]);
    }
    let mut store = Store::open(dir.path()).unwrap();
    assert!(lookup(&mut store, recipe(128)).unwrap().is_empty());
    assert_eq!(
        lookup_with(&mut store, recipe(128), &Tolerance::Approx(vec!["x".to_owned()])).unwrap(),
        vec![id]
    );
}

#[test]
fn a_per_process_engine_does_not_trip_the_wire() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    store.set_determinism(nagaya_abi::Determinism::PerProcess, Vec::new());
    put(&mut store, &base_snapshot(1), recipe(129), 1);
    put(&mut store, &base_snapshot(2), recipe(129), 1);
    assert_eq!(store.entries().len(), 2);
    assert!(store.refused().is_empty());

    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    store.set_determinism(nagaya_abi::Determinism::Always, vec![BitDependence::Residency]);
    put(&mut store, &base_snapshot(1), recipe(130), 1);
    put(&mut store, &base_snapshot(2), recipe(130), 1);
    assert_eq!(store.entries().len(), 2);
}

#[test]
fn a_tie_serves_the_newest_then_the_smaller_content_id() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    store.set_determinism(nagaya_abi::Determinism::PerProcess, Vec::new());
    let new_recipe = recipe(165);
    let first = try_put(&mut store, &base_snapshot(1), new_recipe, None, 1, Tier::Host, 10).unwrap();
    let second = try_put(&mut store, &base_snapshot(2), new_recipe, None, 1, Tier::Host, 20).unwrap();
    assert_ne!(first, second);
    assert_eq!(lookup(&mut store, new_recipe).unwrap(), vec![second]);

    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    store.set_determinism(nagaya_abi::Determinism::PerProcess, Vec::new());
    let old_recipe = recipe(166);
    let first = try_put(&mut store, &base_snapshot(3), old_recipe, None, 1, Tier::Host, 10).unwrap();
    let second = try_put(&mut store, &base_snapshot(4), old_recipe, None, 1, Tier::Host, 10).unwrap();
    assert_eq!(lookup(&mut store, old_recipe).unwrap(), vec![first.min(second)]);
}

#[test]
fn a_lookup_miss_is_empty() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    assert!(lookup(&mut store, recipe(131)).unwrap().is_empty());
}

#[test]
fn a_duplicate_put_under_another_recipe_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    put(&mut store, &base_snapshot(1), recipe(132), 1);
    let error = try_put(&mut store, &base_snapshot(1), recipe(133), None, 1, Tier::Host, 10).unwrap_err();
    assert!(matches!(error, StoreError::Invalid(_)));
}

#[test]
fn a_delta_checks_its_parent() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let root_recipe = recipe(134);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let delta_recipe = next_recipe(root_recipe, 2);
    let missing = nagaya_store::content_id(b"no such snapshot");
    let error = try_put(
        &mut store,
        &delta_at(missing, 2),
        delta_recipe,
        Some(root_recipe),
        1,
        Tier::Host,
        10,
    )
    .unwrap_err();
    assert!(format!("{error}").contains("parent"), "{error}");

    let error = try_put(
        &mut store,
        &delta_at(root, 3),
        delta_recipe,
        Some(root_recipe),
        1,
        Tier::Host,
        10,
    )
    .unwrap_err();
    assert!(format!("{error}").contains("parent's end"), "{error}");

    let error = try_put(
        &mut store,
        &delta_at(root, 2),
        delta_recipe,
        Some(recipe(135)),
        1,
        Tier::Host,
        10,
    )
    .unwrap_err();
    assert!(format!("{error}").contains("previous recipe"), "{error}");
}

#[test]
fn a_delta_respects_the_chain_limits() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_limits(
        dir.path(),
        u64::MAX,
        u64::MAX,
        u64::MAX,
        ChainLimits {
            max_depth: 1,
            delta_fraction: 1.0,
        },
    )
    .unwrap();
    let root_recipe = recipe(136);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let first_recipe = next_recipe(root_recipe, 2);
    let first = put_delta(&mut store, &delta_at(root, 2), first_recipe, root_recipe, 1);
    let second_recipe = next_recipe(first_recipe, 4);
    let error = try_put(
        &mut store,
        &delta_at(first, 4),
        second_recipe,
        Some(first_recipe),
        1,
        Tier::Host,
        10,
    )
    .unwrap_err();
    assert!(format!("{error}").contains("deeper"), "{error}");

    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_limits(
        dir.path(),
        u64::MAX,
        u64::MAX,
        u64::MAX,
        ChainLimits {
            max_depth: 0,
            delta_fraction: 1.0,
        },
    )
    .unwrap();
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let error = try_put(
        &mut store,
        &delta_at(root, 2),
        first_recipe,
        Some(root_recipe),
        1,
        Tier::Host,
        10,
    )
    .unwrap_err();
    assert!(format!("{error}").contains("deeper"), "{error}");

    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_limits(
        dir.path(),
        u64::MAX,
        u64::MAX,
        u64::MAX,
        ChainLimits {
            max_depth: 10,
            delta_fraction: 0.0,
        },
    )
    .unwrap();
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let error = try_put(
        &mut store,
        &delta_at(root, 2),
        first_recipe,
        Some(root_recipe),
        1,
        Tier::Host,
        10,
    )
    .unwrap_err();
    assert!(format!("{error}").contains("larger"), "{error}");
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
    store.pin(second, 1, 100, 10).unwrap();
    store.set_budget(Tier::Host, 5_000);
    let before: Vec<String> = std::fs::read_dir(dir.path().join("blobs"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    let error = try_put(&mut store, &base_snapshot(3), recipe(29), None, 1, Tier::Host, 10);
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
        let error = try_put(&mut store, &base_snapshot(2), recipe(31), None, 1, Tier::Host, 10);
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
    store.pin(id, 1, 100, 10).unwrap();
    store.set_budget(Tier::Host, 100);
    let victims = store.evict(1, 1, 10).unwrap();
    assert_eq!(victims, vec![id]);
    assert!(!store.entries().contains_key(&id));
}

#[test]
fn a_pin_longer_than_a_day_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(170);
    let id = put(&mut store, &base_snapshot(1), recipe, 1);
    let now = 10;
    let one_day = 86_400_000;
    let error = store.pin(id, 1, now + one_day + 1, now).unwrap_err();
    assert!(format!("{error}").contains("one day"), "{error}");
    assert!(store.pins().is_empty());
    store.pin(id, 1, now + one_day, now).unwrap();
    assert!(store.pins().contains_key(&id));
    store.unpin(&id).unwrap();

    let mut engine = FakeAlways::new();
    let pin = Pin {
        owner: 1,
        expires_ms: now + one_day + 1,
    };
    let error = store
        .lookup_and_pin(recipe, &Tolerance::Exact, &mut engine, 0, pin, now)
        .unwrap_err();
    assert!(format!("{error}").contains("one day"), "{error}");
    assert!(store.pins().is_empty());
}

#[test]
fn a_request_pin_protects_until_its_expiry() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(33);
    let id = put(&mut store, &base_snapshot(1), recipe, 1);
    let mut engine = FakeAlways::new();
    store
        .lookup_and_pin(
            recipe,
            &Tolerance::Exact,
            &mut engine,
            0,
            Pin {
                owner: 1,
                expires_ms: 100,
            },
            10,
        )
        .unwrap();
    assert!(matches!(store.evict(1, 1, 10), Err(StoreError::NoSpace)));
    assert_eq!(store.evict(1, 1, 101).unwrap(), vec![id]);
    assert!(store.pins().is_empty());
}

#[test]
fn another_owners_pin_protects() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(34), 1);
    store.pin(id, 2, 100, 10).unwrap();
    assert!(matches!(store.evict(1, 1, 10), Err(StoreError::NoSpace)));
    assert!(store.entries().contains_key(&id));
}

#[test]
fn an_eviction_without_a_tier_skips_a_mixed_group() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_limits(dir.path(), 100_000, 100_000, u64::MAX, open_chain()).unwrap();
    let root_recipe = recipe(171);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let middle_recipe = next_recipe(root_recipe, 2);
    let middle = put_delta(&mut store, &delta_at(root, 2), middle_recipe, root_recipe, 1);
    let tip_recipe = next_recipe(middle_recipe, 4);
    let tip = try_put(
        &mut store,
        &delta_at(middle, 4),
        tip_recipe,
        Some(middle_recipe),
        1,
        Tier::Nvme,
        10,
    )
    .unwrap();
    // The middle group spans host and NVMe. Only the NVMe tip can go.
    let victims = store.evict(1, 1, 10).unwrap();
    assert_eq!(victims, vec![tip]);
    assert!(store.entries().contains_key(&root));
    assert!(store.entries().contains_key(&middle));
}

#[test]
fn an_eviction_of_zero_bytes_takes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(140), 1);
    assert!(store.evict(1, 0, 10).unwrap().is_empty());
    assert!(store.entries().contains_key(&id));
}

#[test]
fn a_put_does_not_evict_another_owners_entry() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_budget(dir.path(), 5_000).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(141), 1);
    let error = try_put(&mut store, &base_snapshot(2), recipe(142), None, 2, Tier::Host, 10);
    assert!(matches!(error, Err(StoreError::NoSpace)));
    assert!(store.entries().contains_key(&first));
    assert_eq!(store.entries().len(), 1);
}

#[test]
fn a_put_over_the_owner_quota_evicts_that_owner() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_limits(dir.path(), 100_000, 100_000, 5_000, open_chain()).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(143), 1);
    let second = put(&mut store, &base_snapshot(2), recipe(144), 1);
    assert!(!store.entries().contains_key(&first));
    assert!(store.entries().contains_key(&second));
    let other = put(&mut store, &base_snapshot(3), recipe(145), 2);
    assert!(store.entries().contains_key(&second));
    assert!(store.entries().contains_key(&other));
}

#[test]
fn a_put_does_not_take_a_budget_from_another_tier() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_limits(dir.path(), 5_000, 5_000, u64::MAX, open_chain()).unwrap();
    let host_first = put(&mut store, &base_snapshot(1), recipe(146), 1);
    let nvme = try_put(&mut store, &base_snapshot(2), recipe(147), None, 1, Tier::Nvme, 10).unwrap();
    let host_second = put(&mut store, &base_snapshot(3), recipe(148), 1);
    assert!(!store.entries().contains_key(&host_first));
    assert!(store.entries().contains_key(&host_second));
    assert!(store.entries().contains_key(&nvme));
}

#[test]
fn a_failed_blob_delete_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_budget(dir.path(), 10_000).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(149), 1);
    let second = put(&mut store, &base_snapshot(2), recipe(150), 1);
    // A directory at the blob path makes the delete fail.
    for id in [first, second] {
        let path = dir.path().join("blobs").join(id.hex());
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
    }
    let error = try_put(&mut store, &base_snapshot(3), recipe(151), None, 1, Tier::Host, 10).unwrap_err();
    assert!(matches!(error, StoreError::Io(_)), "{error}");
    assert_eq!(store.entries().len(), 2);
    let victim = if store.entries().contains_key(&first) {
        second
    } else {
        first
    };
    assert!(!store.entries().contains_key(&victim));
    assert!(dir.path().join("blobs").join(victim.hex()).is_dir());
}

#[test]
fn pins_protect_one_by_one() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(156), 1);
    let second = put(&mut store, &base_snapshot(2), recipe(157), 1);
    store.pin(first, 1, 100, 10).unwrap();
    store.pin(second, 1, 100, 10).unwrap();
    // Each pin alone fits the budget. Their sum does not.
    store.set_budget(Tier::Host, 7_000);
    assert!(matches!(store.evict(1, 1, 10), Err(StoreError::NoSpace)));
    assert!(store.entries().contains_key(&first));
    assert!(store.entries().contains_key(&second));
}

#[test]
fn an_index_mismatch_quarantines_the_entry() {
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let mut store = Store::open(dir.path()).unwrap();
        id = put(&mut store, &base_snapshot(1), recipe(158), 1);
    }
    edit_index(dir.path(), |index| {
        index["entries"][id.hex()]["coverage"][0]["end"] = serde_json::json!(3);
    });
    let mut store = Store::open(dir.path()).unwrap();
    let mut target = FakeAlways::new();
    let error = store
        .restore(&id, &mut target, State::Slot(0), &[1u8; 32], 10)
        .unwrap_err();
    assert!(format!("{error}").contains("disagrees"), "{error}");
    assert_eq!(store.entries().get(&id).unwrap().quarantine, Some(Quarantine::Index));
}

#[test]
fn an_index_mismatch_in_a_middle_entry_is_found() {
    let dir = tempfile::tempdir().unwrap();
    let (middle, tip);
    {
        let mut store = Store::open(dir.path()).unwrap();
        let root_recipe = recipe(164);
        let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
        let middle_recipe = next_recipe(root_recipe, 2);
        middle = put_delta(&mut store, &delta_at(root, 2), middle_recipe, root_recipe, 1);
        let tip_recipe = next_recipe(middle_recipe, 4);
        tip = put_delta(&mut store, &delta_at(middle, 4), tip_recipe, middle_recipe, 1);
    }
    edit_index(dir.path(), |index| {
        index["entries"][middle.hex()]["coverage"][0]["end"] = serde_json::json!(5);
    });
    let mut store = Store::open(dir.path()).unwrap();
    let mut target = FakeAlways::new();
    let error = store
        .restore(&tip, &mut target, State::Slot(0), &[1u8; 32], 10)
        .unwrap_err();
    assert!(format!("{error}").contains("disagrees"), "{error}");
    assert_eq!(
        store.entries().get(&middle).unwrap().quarantine,
        Some(Quarantine::Index)
    );
}

#[test]
fn an_unreadable_blob_quarantines_with_read() {
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let mut store = Store::open(dir.path()).unwrap();
        id = put(&mut store, &base_snapshot(1), recipe(159), 1);
    }
    let path = dir.path().join("blobs").join(id.hex());
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    assert!(matches!(store.get(&id, 10), Err(StoreError::Io(_))));
    assert_eq!(store.entries().get(&id).unwrap().quarantine, Some(Quarantine::Read));
}

#[test]
fn clearing_a_tripwire_removes_its_entries() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(167);
    let first = put(&mut store, &base_snapshot(1), recipe, 1);
    let error = try_put(&mut store, &base_snapshot(2), recipe, None, 1, Tier::Host, 10).unwrap_err();
    let second = match error {
        StoreError::Tripwire { ids, .. } => ids.into_iter().find(|id| *id != first).unwrap(),
        other => panic!("expected the tripwire, got {other:?}"),
    };
    store.clear_refusal(recipe).unwrap();
    assert!(!store.entries().contains_key(&first));
    assert!(!store.entries().contains_key(&second));
    assert!(!dir.path().join("blobs").join(first.hex()).exists());
    assert!(!dir.path().join("blobs").join(second.hex()).exists());
    assert!(dir.path().join("quarantine").join(first.hex()).exists());
    assert!(dir.path().join("quarantine").join(second.hex()).exists());
    assert!(store.refused().is_empty());
    let fresh = put(&mut store, &base_snapshot(3), recipe, 1);
    assert_eq!(lookup(&mut store, recipe).unwrap(), vec![fresh]);
}

#[test]
fn a_tripwire_quarantine_clears_with_its_recipe() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(152);
    let first = put(&mut store, &base_snapshot(1), recipe, 1);
    assert!(try_put(&mut store, &base_snapshot(2), recipe, None, 1, Tier::Host, 10).is_err());
    assert!(store.clear_entry(&first).is_err());
    store.clear_refusal(recipe).unwrap();
    assert!(!store.entries().contains_key(&first));
    let fresh = put(&mut store, &base_snapshot(3), recipe, 1);
    assert_eq!(lookup(&mut store, recipe).unwrap(), vec![fresh]);
}

fn quarantine_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir.join("quarantine"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

#[test]
fn a_second_quarantine_of_one_id_keeps_both_blobs() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(170);
    let first = put(&mut store, &base_snapshot(1), recipe, 1);
    assert!(try_put(&mut store, &base_snapshot(2), recipe, None, 1, Tier::Host, 10).is_err());
    store.clear_refusal(recipe).unwrap();
    assert_eq!(quarantine_names(dir.path()).len(), 2);

    let again = put(&mut store, &base_snapshot(1), recipe, 1);
    assert_eq!(again, first, "the same bytes give the same content id");
    assert!(try_put(&mut store, &base_snapshot(2), recipe, None, 1, Tier::Host, 10).is_err());
    store.clear_refusal(recipe).unwrap();

    let names = quarantine_names(dir.path());
    assert_eq!(names.len(), 4, "{names:?}");
    assert!(names.contains(&first.hex()), "{names:?}");
    assert!(names.contains(&format!("{}.1", first.hex())), "{names:?}");
}

#[test]
fn quarantined_blobs_are_not_in_the_budget() {
    let dir = tempfile::tempdir().unwrap();
    let size = base_snapshot(1).encode().unwrap().len() as u64;
    let mut store = Store::open_with_budget(dir.path(), size).unwrap();
    let recipe = recipe(171);
    put(&mut store, &base_snapshot(1), recipe, 1);
    assert!(try_put(&mut store, &base_snapshot(2), recipe, None, 1, Tier::Host, 10).is_err());
    store.clear_refusal(recipe).unwrap();

    let fresh = put(&mut store, &base_snapshot(3), recipe, 1);
    assert_eq!(lookup(&mut store, recipe).unwrap(), vec![fresh]);
}

#[test]
fn a_failed_move_reports_an_error_and_leaves_the_blob_in_blobs() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(172);
    let first = put(&mut store, &base_snapshot(1), recipe, 1);
    let error = try_put(&mut store, &base_snapshot(2), recipe, None, 1, Tier::Host, 10).unwrap_err();
    let second = match error {
        StoreError::Tripwire { ids, .. } => ids.into_iter().find(|id| *id != first).unwrap(),
        other => panic!("expected the tripwire, got {other:?}"),
    };
    let quarantine = dir.path().join("quarantine");
    std::fs::set_permissions(&quarantine, std::fs::Permissions::from_mode(0o500)).unwrap();

    let error = store.clear_refusal(recipe).unwrap_err();
    assert!(matches!(error, StoreError::Io(_)), "{error:?}");
    assert!(dir.path().join("blobs").join(first.hex()).exists());
    assert!(dir.path().join("blobs").join(second.hex()).exists());
    assert_eq!(quarantine_names(dir.path()).len(), 0);
    std::fs::set_permissions(&quarantine, std::fs::Permissions::from_mode(0o700)).unwrap();

    // A crash between the index write and the move leaves the same state. The next open reads the index,
    // which does not name the blob, so the blob stays and no entry serves it.
    let mut store = Store::open(dir.path()).unwrap();
    assert!(dir.path().join("blobs").join(first.hex()).exists());
    assert!(store.entries().is_empty());
    let again = put(&mut store, &base_snapshot(1), recipe, 1);
    assert_eq!(again, first);
}
