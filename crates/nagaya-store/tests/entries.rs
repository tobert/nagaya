use std::path::Path;

use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    BitDependence, ChainLimits, Dtype, Engine, Identity, Limits, OutputsWanted, PartError, PartId, PartKind, Parts,
    PassError, PassOut, Segment, SegmentInputs, SlotLimit, State, Tap, WriteResult,
};
use nagaya_store::{
    Check, ContentId, EngineFacts, EntryState, PartCoverage, Pin, Put, Quarantine, Reason, RecipeId, Restored,
    Snapshot, Store, StoreError, Tensor, Tier, Tolerance, recipe_boundary, recipe_not_sealed, recipe_root,
    snapshot_from_engine,
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

fn facts(determinism: nagaya_abi::Determinism, bit_dependence: Vec<BitDependence>, chain: ChainLimits) -> EngineFacts {
    EngineFacts {
        name: "fake".to_owned(),
        determinism,
        bit_dependence,
        chain,
    }
}

/// The facts of an engine with no chain limit that always gives the same bits.
fn open_engine() -> EngineFacts {
    facts(nagaya_abi::Determinism::Always, Vec::new(), open_chain())
}

/// Put with the facts of one engine. The default helpers use `open_engine`.
fn put_as(
    store: &mut Store,
    snapshot: &Snapshot,
    recipe: RecipeId,
    previous_recipe: Option<RecipeId>,
    engine: &EngineFacts,
) -> Result<ContentId, StoreError> {
    put_as_at(store, snapshot, recipe, previous_recipe, engine, 10)
}

fn put_as_at(
    store: &mut Store,
    snapshot: &Snapshot,
    recipe: RecipeId,
    previous_recipe: Option<RecipeId>,
    engine: &EngineFacts,
    now_ms: u64,
) -> Result<ContentId, StoreError> {
    store.put(Put {
        snapshot,
        recipe,
        previous_recipe,
        chains_from: None,
        engine: engine.clone(),
        identity: [1u8; 32],
        owner: 1,
        tier: Tier::Nvme,
        reason: Reason::Anchor,
        sealed_recipe: None,
        provenance: None,
        now_ms,
    })
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
        chains_from: None,
        engine: open_engine(),
        identity: [1u8; 32],
        owner,
        tier,
        reason: Reason::Anchor,
        sealed_recipe: None,
        provenance: None,
        now_ms,
    })
}

fn put(store: &mut Store, snapshot: &Snapshot, recipe: RecipeId, owner: u64) -> ContentId {
    try_put(store, snapshot, recipe, None, owner, Tier::Nvme, 10).expect("put")
}

fn put_delta(store: &mut Store, snapshot: &Snapshot, recipe: RecipeId, previous: RecipeId, owner: u64) -> ContentId {
    try_put(store, snapshot, recipe, Some(previous), owner, Tier::Nvme, 10).expect("put delta")
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
            chains_from: None,
            engine: open_engine(),
            identity: [1u8; 32],
            owner: 1,
            tier: Tier::Nvme,
            reason: Reason::Anchor,
            sealed_recipe: None,
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
            chains_from: None,
            engine: open_engine(),
            identity: [9u8; 32],
            owner: 7,
            tier: Tier::Nvme,
            reason: Reason::PrefillDone,
            sealed_recipe: None,
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
    assert_eq!(entry.reason, Reason::PrefillDone);
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
    // The base is hotter than the delta, so by value (spec 6.9 step 4) the delta goes first and the base stays.
    for _ in 0..3 {
        store.get(&base, 15).unwrap();
    }
    let victims = store.evict(0, Tier::Nvme, 1, 20).unwrap();
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
    let error = try_put(&mut store, &base_snapshot(2), recipe, None, 0, Tier::Nvme, 10).unwrap_err();
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
        try_put(&mut store, &base_snapshot(3), recipe, None, 0, Tier::Nvme, 10),
        Err(StoreError::Tripwire { .. })
    ));
}

#[test]
fn a_person_clears_a_tripwire() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(40);
    let first = put(&mut store, &base_snapshot(1), recipe, 0);
    assert!(try_put(&mut store, &base_snapshot(2), recipe, None, 0, Tier::Nvme, 10).is_err());
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
    assert_eq!(results, Restored::Done);
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
                let from = start;
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
fn a_restore_keeps_a_ring_base() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let mut source = FakeAlways::new();
    source.pass(&[segment(&[1, 2, 3, 4, 5, 6])], &[]).unwrap();
    let snapshot = snapshot_from_engine(&mut source, State::Slot(0), &[(1, 0, 6), (2, 2, 4)]).unwrap();
    let ring = snapshot.tensors.iter().find(|tensor| tensor.name == "ring").unwrap();
    assert_eq!(ring.coverage, Some((2, 6)), "the ring window moved");
    assert_eq!(ring.data.len(), 32, "four rows of eight bytes");

    let id = put(&mut store, &snapshot, recipe(30), 0);
    let mut target = FakeAlways::new();
    let results = store.restore(&id, &mut target, State::Slot(0), &[1u8; 32], 20).unwrap();
    assert_eq!(results, Restored::Done);

    let index = target
        .limits()
        .parts
        .iter()
        .position(|part| part.name == "ring")
        .unwrap();
    let span = target.coverage().unwrap().coverage(State::Slot(0), index, &ring.data);
    assert_eq!((span.start, span.end), (2, 6), "the restore must keep the ring base");
    let mut bytes = vec![0u8; ring.data.len()];
    target
        .parts()
        .unwrap()
        .cells_read(State::Slot(0), index, 2, 4, &mut bytes)
        .unwrap();
    assert_eq!(bytes, ring.data);
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
    let victims = store.evict(1, Tier::Nvme, 1, 10).unwrap();
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
    let victims = store.evict(1, Tier::Nvme, 1, 10).unwrap();
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
    assert!(matches!(store.evict(1, Tier::Nvme, 1, 10), Err(StoreError::NoSpace)));
    assert!(store.entries().contains_key(&root));
    assert!(store.entries().contains_key(&child));
}

#[test]
fn a_cross_tier_child_keeps_the_parent() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_limits(dir.path(), 5_000, 5_000, u64::MAX, 0).unwrap();
    let root_recipe = recipe(162);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let child_recipe = next_recipe(root_recipe, 2);
    let child = try_put(
        &mut store,
        &delta_at(root, 2),
        child_recipe,
        Some(root_recipe),
        1,
        Tier::Host,
        10,
    )
    .unwrap();
    let error = try_put(&mut store, &base_snapshot(3), recipe(163), None, 1, Tier::Nvme, 10).unwrap_err();
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
    assert!(matches!(store.evict(1, Tier::Nvme, 1, 10), Err(StoreError::NoSpace)));
    assert!(store.entries().contains_key(&root));
    assert!(store.entries().contains_key(&child));
}

#[test]
fn an_owner_does_not_evict_another() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(7), 1);
    let second = put(&mut store, &wide_snapshot(2), recipe(8), 2);
    let victims = store.evict(1, Tier::Nvme, 1, 10).unwrap();
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
    assert!(matches!(store.evict(1, Tier::Nvme, 1, 10), Err(StoreError::NoSpace)));
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
    let first = put(&mut store, &dual_snapshot(1, 5, 1), recipe, 0);
    let second = put(&mut store, &dual_snapshot(3, 2, 2), recipe, 0);
    let error = lookup(&mut store, recipe).unwrap_err();
    let StoreError::Ambiguous { left, right } = &error else {
        panic!("{error:?}");
    };
    let mut named = [*left, *right];
    let mut tips = [first, second];
    named.sort();
    tips.sort();
    assert_eq!(named, tips, "the error names both chain tips");
    let message = error.to_string();
    assert!(
        message.contains(&first.hex()) && message.contains(&second.hex()),
        "{message}"
    );
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
    assert!(try_put(&mut store, &base_snapshot(2), bad_recipe, None, 0, Tier::Nvme, 10).is_err());
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
    let error = try_put(&mut store, &base_snapshot(1), recipe, None, 1, Tier::Nvme, 10).unwrap_err();
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
    let mut target = FakeAlways::with_slots(vec![SlotLimit {
        bytes: 8,
        count: 1,
        tokens: 4096,
    }]);
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
    assert_eq!(
        results,
        Restored::Refused {
            part: "cells".to_owned()
        }
    );
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
    assert_eq!(results, Restored::Done);
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
    let mut target = FakeAlways::with_slots(vec![SlotLimit {
        bytes: 24,
        count: 1,
        tokens: 4096,
    }]);
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
    let mut small = FakeAlways::with_slots(vec![SlotLimit {
        bytes: 24,
        count: 1,
        tokens: 4096,
    }]);
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
        id = put_as(
            &mut store,
            &base_snapshot(1),
            recipe(128),
            None,
            &facts(nagaya_abi::Determinism::PerProcess, Vec::new(), open_chain()),
        )
        .unwrap();
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
    let engine = facts(nagaya_abi::Determinism::PerProcess, Vec::new(), open_chain());
    put_as(&mut store, &base_snapshot(1), recipe(129), None, &engine).unwrap();
    put_as(&mut store, &base_snapshot(2), recipe(129), None, &engine).unwrap();
    assert_eq!(store.entries().len(), 2);
    assert!(store.refused().is_empty());

    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let engine = facts(
        nagaya_abi::Determinism::Always,
        vec![BitDependence::Residency],
        open_chain(),
    );
    put_as(&mut store, &base_snapshot(1), recipe(130), None, &engine).unwrap();
    put_as(&mut store, &base_snapshot(2), recipe(130), None, &engine).unwrap();
    assert_eq!(store.entries().len(), 2);
}

#[test]
fn a_tie_serves_the_newest_then_the_smaller_content_id() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let engine = facts(nagaya_abi::Determinism::PerProcess, Vec::new(), open_chain());
    let new_recipe = recipe(165);
    let first = put_as_at(&mut store, &base_snapshot(1), new_recipe, None, &engine, 10).unwrap();
    let second = put_as_at(&mut store, &base_snapshot(2), new_recipe, None, &engine, 20).unwrap();
    assert_ne!(first, second);
    assert_eq!(lookup(&mut store, new_recipe).unwrap(), vec![second]);

    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let engine = facts(nagaya_abi::Determinism::PerProcess, Vec::new(), open_chain());
    let old_recipe = recipe(166);
    let first = put_as_at(&mut store, &base_snapshot(3), old_recipe, None, &engine, 10).unwrap();
    let second = put_as_at(&mut store, &base_snapshot(4), old_recipe, None, &engine, 10).unwrap();
    assert_eq!(lookup(&mut store, old_recipe).unwrap(), vec![first.min(second)]);
}

#[test]
fn an_index_entry_without_its_engine_facts_does_not_open() {
    // An index from before the engine facts has no `engine` field. The store refuses it and says why.
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let mut store = Store::open(dir.path()).unwrap();
        id = put(&mut store, &base_snapshot(1), recipe(196), 1);
    }
    edit_index(dir.path(), |index| {
        index["entries"][id.hex()].as_object_mut().unwrap().remove("engine");
    });
    let error = Store::open(dir.path()).unwrap_err();
    assert!(format!("{error}").contains("engine"), "{error}");
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
    let error = try_put(&mut store, &base_snapshot(1), recipe(133), None, 1, Tier::Nvme, 10).unwrap_err();
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
        Tier::Nvme,
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
        Tier::Nvme,
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
        Tier::Nvme,
        10,
    )
    .unwrap_err();
    assert!(format!("{error}").contains("previous recipe"), "{error}");
}

#[test]
fn a_delta_respects_the_chain_limits_of_its_own_put() {
    let limits = |max_depth, delta_fraction| {
        facts(
            nagaya_abi::Determinism::Always,
            Vec::new(),
            ChainLimits {
                max_depth,
                delta_fraction,
            },
        )
    };
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let engine = limits(1, 1.0);
    let root_recipe = recipe(136);
    let root = put_as(&mut store, &base_snapshot(1), root_recipe, None, &engine).unwrap();
    let first_recipe = next_recipe(root_recipe, 2);
    let first = put_as(&mut store, &delta_at(root, 2), first_recipe, Some(root_recipe), &engine).unwrap();
    let second_recipe = next_recipe(first_recipe, 4);
    let error = put_as(
        &mut store,
        &delta_at(first, 4),
        second_recipe,
        Some(first_recipe),
        &engine,
    )
    .unwrap_err();
    assert!(format!("{error}").contains("deeper"), "{error}");

    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let engine = limits(0, 1.0);
    let root = put_as(&mut store, &base_snapshot(1), root_recipe, None, &engine).unwrap();
    let error = put_as(&mut store, &delta_at(root, 2), first_recipe, Some(root_recipe), &engine).unwrap_err();
    assert!(format!("{error}").contains("deeper"), "{error}");

    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let engine = limits(10, 0.0);
    let root = put_as(&mut store, &base_snapshot(1), root_recipe, None, &engine).unwrap();
    let error = put_as(&mut store, &delta_at(root, 2), first_recipe, Some(root_recipe), &engine).unwrap_err();
    assert!(format!("{error}").contains("larger"), "{error}");
}

#[test]
fn each_put_carries_the_chain_limits_of_its_engine() {
    // Spec 6.1 and 6.5: one store serves every engine. Two engines with different limits share it.
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let shallow = facts(
        nagaya_abi::Determinism::Always,
        Vec::new(),
        ChainLimits {
            max_depth: 0,
            delta_fraction: 0.0,
        },
    );
    let deep = open_engine();

    let root_recipe = recipe(185);
    let root = put_as(&mut store, &base_snapshot(1), root_recipe, None, &shallow).unwrap();
    let delta_recipe = next_recipe(root_recipe, 2);
    let error = put_as(
        &mut store,
        &delta_at(root, 2),
        delta_recipe,
        Some(root_recipe),
        &shallow,
    )
    .unwrap_err();
    assert!(format!("{error}").contains("deeper"), "{error}");
    // The deep engine puts into the same store after the shallow engine.
    let delta = put_as(&mut store, &delta_at(root, 2), delta_recipe, Some(root_recipe), &deep).unwrap();
    assert_eq!(store.entries()[&delta].chain.max_depth, u32::MAX);
    assert_eq!(store.entries()[&root].chain.max_depth, 0);
    assert_eq!(store.entries()[&root].engine, "fake");
}

#[test]
fn the_tripwire_reads_the_recorded_class_of_each_entry() {
    // Spec 6.7: the tripwire is on for an entry only when its engine is deterministic across processes.
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let wandering = facts(nagaya_abi::Determinism::PerProcess, Vec::new(), open_chain());
    let steady = open_engine();
    let resident = facts(
        nagaya_abi::Determinism::Always,
        vec![BitDependence::Residency],
        open_chain(),
    );

    // A per-process entry and a steady put under one recipe do not trip.
    let mixed = recipe(186);
    put_as(&mut store, &base_snapshot(1), mixed, None, &wandering).unwrap();
    put_as(&mut store, &base_snapshot(2), mixed, None, &steady).unwrap();
    assert!(store.refused().is_empty());
    // The same holds the other way round, and for an engine whose bits depend on residency.
    put_as(&mut store, &base_snapshot(3), mixed, None, &wandering).unwrap();
    put_as(&mut store, &base_snapshot(4), mixed, None, &resident).unwrap();
    assert!(store.refused().is_empty());
    assert_eq!(store.entries().len(), 4);

    // Two steady puts under one recipe still trip, whatever the engines before them.
    let pure = recipe(187);
    put_as(&mut store, &base_snapshot(5), pure, None, &steady).unwrap();
    let error = put_as(&mut store, &base_snapshot(6), pure, None, &steady).unwrap_err();
    assert!(matches!(error, StoreError::Tripwire { .. }), "{error:?}");
    assert!(store.refused().contains(&pure));
}

fn rows_delta(parent: ContentId, start: u64, rows: u64) -> Snapshot {
    Snapshot {
        tensors: vec![
            tensor("state", PartKind::Fixed, vec![16], vec![9; 16], None),
            tensor(
                "cells",
                PartKind::Cells,
                vec![rows, 8],
                vec![7; (rows * 8) as usize],
                Some((start, start + rows)),
            ),
        ],
        parent: Some(parent),
    }
}

#[test]
fn the_delta_fraction_compares_with_the_base_of_the_chain() {
    // Spec 6.5 check 3: the limit is a fraction of the full base snapshot, not of the parent (a delta).
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let root_recipe = recipe(180);
    let root = put(&mut store, &wide_snapshot(1), root_recipe, 1);
    let first_recipe = next_recipe(root_recipe, 2);
    let first_snapshot = rows_delta(root, 3, 1);
    let first = put_delta(&mut store, &first_snapshot, first_recipe, root_recipe, 1);
    let base_size = wide_snapshot(1).encode().unwrap().len();
    let parent_size = first_snapshot.encode().unwrap().len();
    let second_snapshot = rows_delta(first, 4, 2);
    let second_size = second_snapshot.encode().unwrap().len();
    assert!(
        parent_size < second_size && second_size <= base_size,
        "the sizes set the test"
    );

    let second_recipe = next_recipe(first_recipe, 4);
    let second = try_put(
        &mut store,
        &second_snapshot,
        second_recipe,
        Some(first_recipe),
        1,
        Tier::Nvme,
        10,
    )
    .expect("a delta larger than its parent and no larger than the base fits");

    let third_snapshot = rows_delta(second, 6, 4);
    assert!(
        third_snapshot.encode().unwrap().len() > base_size,
        "the sizes set the test"
    );
    let error = try_put(
        &mut store,
        &third_snapshot,
        next_recipe(second_recipe, 6),
        Some(second_recipe),
        1,
        Tier::Nvme,
        10,
    )
    .unwrap_err();
    assert!(format!("{error}").contains("larger"), "{error}");
}

#[test]
fn eviction_ranks_tips_roots_and_standalone_entries_by_value_together() {
    // Spec 6.9 step 4: after the interiors, one ranking by value per byte covers every remaining entry.
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let root_recipe = recipe(181);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let tip_recipe = next_recipe(root_recipe, 2);
    let tip = put_delta(&mut store, &delta_at(root, 2), tip_recipe, root_recipe, 1);
    let cold = put(&mut store, &base_snapshot(5), recipe(182), 1);
    for _ in 0..5 {
        store.get(&tip, 1_000).unwrap();
    }
    let victims = store.evict(1, Tier::Nvme, 1, 1_000).unwrap();
    assert_eq!(victims, vec![cold], "the cold standalone entry goes before the hot tip");
    assert!(store.entries().contains_key(&tip));
}

#[test]
fn eviction_takes_a_cold_tip_before_a_hot_standalone_entry() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let root_recipe = recipe(183);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let tip_recipe = next_recipe(root_recipe, 2);
    let tip = put_delta(&mut store, &delta_at(root, 2), tip_recipe, root_recipe, 1);
    let hot = put(&mut store, &base_snapshot(5), recipe(184), 1);
    for _ in 0..5 {
        store.get(&hot, 1_000).unwrap();
    }
    let victims = store.evict(1, Tier::Nvme, 1, 1_000).unwrap();
    assert_eq!(victims, vec![tip]);
    assert!(store.entries().contains_key(&hot));
}

#[test]
fn a_put_that_fits_writes_no_victims() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_nvme_budget(dir.path(), 10_000).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(24), 1);
    let second = put(&mut store, &base_snapshot(2), recipe(35), 1);
    assert!(store.entries().contains_key(&first));
    assert!(store.entries().contains_key(&second));
    assert_eq!(store.entries().len(), 2);
}

#[test]
fn a_put_over_budget_evicts_first() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_nvme_budget(dir.path(), 5_000).unwrap();
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
    let mut store = Store::open_with_nvme_budget(dir.path(), 10_000).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(27), 1);
    let second = put(&mut store, &base_snapshot(2), recipe(28), 1);
    store.pin(second, 1, 100, 10).unwrap();
    store.set_budget(Tier::Nvme, 5_000);
    let before: Vec<String> = std::fs::read_dir(dir.path().join("blobs"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    let error = try_put(&mut store, &base_snapshot(3), recipe(29), None, 1, Tier::Nvme, 10);
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
        let error = try_put(&mut store, &base_snapshot(2), recipe(31), None, 1, Tier::Nvme, 10);
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
    let mut store = Store::open_with_nvme_budget(dir.path(), 10_000).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(32), 1);
    store.pin(id, 1, 100, 10).unwrap();
    store.set_budget(Tier::Nvme, 100);
    let victims = store.evict(1, Tier::Nvme, 1, 10).unwrap();
    assert_eq!(victims, vec![id]);
    assert!(!store.entries().contains_key(&id));
}

#[test]
fn an_entry_stays_pinned_while_any_owner_pin_is_live() {
    // Spec 6.10: each owner holds its own pin with its own expiry.
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(190), 1);
    store.pin(id, 10, 100_000, 10).unwrap();
    store.pin(id, 20, 50_000, 10).unwrap();
    // The pin of owner 20 has run out. The pin of owner 10 has not.
    assert_eq!(
        store.evict(1, Tier::Nvme, 1, 60_000).unwrap_err().to_string(),
        StoreError::NoSpace.to_string()
    );
    assert!(store.entries().contains_key(&id));
    // Both have run out.
    assert_eq!(store.evict(1, Tier::Nvme, 1, 100_000).unwrap(), vec![id]);
}

#[test]
fn a_later_live_pin_protects_after_an_earlier_pin_ran_out() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(195), 1);
    store.pin(id, 10, 50_000, 10).unwrap();
    store.pin(id, 20, 100_000, 10).unwrap();
    assert!(store.evict(1, Tier::Nvme, 1, 60_000).is_err());
    assert!(store.entries().contains_key(&id));
}

#[test]
fn unpin_removes_only_the_pin_of_its_owner() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(191), 1);
    store.pin(id, 10, 100_000, 10).unwrap();
    store.pin(id, 20, 50_000, 10).unwrap();
    // A second pin by the same owner replaces that owner's pin and no other.
    store.pin(id, 20, 70_000, 10).unwrap();
    let mut owners: Vec<(u64, u64)> = store.pins()[&id]
        .iter()
        .map(|pin| (pin.owner, pin.expires_ms))
        .collect();
    owners.sort();
    assert_eq!(owners, vec![(10, 100_000), (20, 70_000)]);

    store.unpin(&id, 10).unwrap();
    assert_eq!(store.pins()[&id].len(), 1);
    assert_eq!(store.pins()[&id][0].owner, 20);
    assert!(
        store.evict(1, Tier::Nvme, 1, 60_000).is_err(),
        "the pin of owner 20 still protects the entry"
    );
    store.unpin(&id, 20).unwrap();
    assert!(store.pins().is_empty());
    // An owner with no pin changes nothing.
    store.unpin(&id, 30).unwrap();
}

#[test]
fn two_owner_pins_survive_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let mut store = Store::open(dir.path()).unwrap();
        id = put(&mut store, &base_snapshot(1), recipe(192), 1);
        store.pin(id, 10, 100_000, 10).unwrap();
        store.pin(id, 20, 50_000, 10).unwrap();
    }
    let mut store = Store::open(dir.path()).unwrap();
    let mut owners: Vec<(u64, u64)> = store.pins()[&id]
        .iter()
        .map(|pin| (pin.owner, pin.expires_ms))
        .collect();
    owners.sort();
    assert_eq!(owners, vec![(10, 100_000), (20, 50_000)]);
    assert!(
        store.evict(1, Tier::Nvme, 1, 60_000).is_err(),
        "the pin of owner 10 protects the entry after the restart"
    );
    store.unpin(&id, 10).unwrap();
    assert_eq!(
        store.evict(1, Tier::Nvme, 1, 40_000).unwrap_err().to_string(),
        StoreError::NoSpace.to_string()
    );
}

#[test]
fn an_index_with_one_pin_for_each_entry_still_opens() {
    // The old index held one pin object for each entry. The store reads it as a list of one pin.
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let mut store = Store::open(dir.path()).unwrap();
        id = put(&mut store, &base_snapshot(1), recipe(193), 1);
        store.pin(id, 10, 100_000, 10).unwrap();
    }
    edit_index(dir.path(), |index| {
        let list = index["pins"][id.hex()].take();
        assert!(list.is_array(), "the new format is a list");
        index["pins"][id.hex()] = list[0].clone();
    });
    let mut store = Store::open(dir.path()).unwrap();
    assert_eq!(store.pins()[&id].len(), 1);
    assert_eq!(store.pins()[&id][0].owner, 10);
    assert!(store.evict(1, Tier::Nvme, 1, 60_000).is_err());
}

#[test]
fn a_lookup_pin_of_a_second_owner_keeps_the_first_pin() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(194);
    let id = put(&mut store, &base_snapshot(1), recipe, 1);
    store.pin(id, 10, 100_000, 10).unwrap();
    let mut engine = FakeAlways::new();
    store
        .lookup_and_pin(
            recipe,
            &Tolerance::Exact,
            &mut engine,
            0,
            Pin {
                owner: 20,
                expires_ms: 20_000,
            },
            10,
        )
        .unwrap();
    assert_eq!(store.pins()[&id].len(), 2);
    assert!(store.evict(1, Tier::Nvme, 1, 60_000).is_err());
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
    store.unpin(&id, 1).unwrap();

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
    assert!(matches!(store.evict(1, Tier::Nvme, 1, 10), Err(StoreError::NoSpace)));
    assert_eq!(store.evict(1, Tier::Nvme, 1, 101).unwrap(), vec![id]);
    assert!(store.pins().is_empty());
}

#[test]
fn another_owners_pin_protects() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(34), 1);
    store.pin(id, 2, 100, 10).unwrap();
    assert!(matches!(store.evict(1, Tier::Nvme, 1, 10), Err(StoreError::NoSpace)));
    assert!(store.entries().contains_key(&id));
}

#[test]
fn an_eviction_in_one_tier_skips_a_group_that_spans_tiers() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_limits(dir.path(), 100_000, 100_000, u64::MAX, 0).unwrap();
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
        Tier::Host,
        10,
    )
    .unwrap();
    // The groups of the root and the middle span NVMe and host, so an NVMe eviction takes nothing.
    assert!(matches!(store.evict(1, Tier::Nvme, 1, 10), Err(StoreError::NoSpace)));
    // Only the host tip can go.
    let victims = store.evict(1, Tier::Host, 1, 10).unwrap();
    assert_eq!(victims, vec![tip]);
    assert!(store.entries().contains_key(&root));
    assert!(store.entries().contains_key(&middle));
}

#[test]
fn an_eviction_of_zero_bytes_takes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(140), 1);
    assert!(store.evict(1, Tier::Nvme, 0, 10).unwrap().is_empty());
    assert!(store.entries().contains_key(&id));
}

#[test]
fn a_put_does_not_evict_another_owners_entry() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_nvme_budget(dir.path(), 5_000).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(141), 1);
    let error = try_put(&mut store, &base_snapshot(2), recipe(142), None, 2, Tier::Nvme, 10);
    assert!(matches!(error, Err(StoreError::NoSpace)));
    assert!(store.entries().contains_key(&first));
    assert_eq!(store.entries().len(), 1);
}

#[test]
fn a_put_over_the_owner_quota_evicts_that_owner() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_limits(dir.path(), 100_000, 100_000, 5_000, 0).unwrap();
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
    // NVMe has room for two entries, so the host victim moves there (6.9) and evicts nothing on NVMe.
    let mut store = Store::open_with_limits(dir.path(), 5_000, 10_000, u64::MAX, 0).unwrap();
    let host_first = put_in(&mut store, &base_snapshot(1), recipe(146), 1, Tier::Host);
    let nvme = try_put(&mut store, &base_snapshot(2), recipe(147), None, 1, Tier::Nvme, 10).unwrap();
    let host_second = put_in(&mut store, &base_snapshot(3), recipe(148), 1, Tier::Host);
    assert_eq!(store.entries()[&host_first].tier, Tier::Nvme);
    assert!(store.entries().contains_key(&host_second));
    assert!(store.entries().contains_key(&nvme));
}

#[test]
fn a_failed_blob_delete_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_nvme_budget(dir.path(), 10_000).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(149), 1);
    let second = put(&mut store, &base_snapshot(2), recipe(150), 1);
    // A directory at the blob path makes the delete fail.
    for id in [first, second] {
        let path = dir.path().join("blobs").join(id.hex());
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
    }
    let error = try_put(&mut store, &base_snapshot(3), recipe(151), None, 1, Tier::Nvme, 10).unwrap_err();
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
    store.set_budget(Tier::Nvme, 7_000);
    assert!(matches!(store.evict(1, Tier::Nvme, 1, 10), Err(StoreError::NoSpace)));
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

/// Put one entry, then make its blob unreadable: a directory takes the place of the file.
fn store_with_unreadable_blob(dir: &Path) -> (Store, ContentId, RecipeId, Vec<u8>) {
    let recipe = recipe(159);
    let id;
    {
        let mut store = Store::open(dir).unwrap();
        id = put(&mut store, &base_snapshot(1), recipe, 1);
    }
    let path = dir.join("blobs").join(id.hex());
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    (Store::open(dir).unwrap(), id, recipe, bytes)
}

fn repair_blob(dir: &Path, id: &ContentId, bytes: &[u8]) {
    let path = dir.join("blobs").join(id.hex());
    std::fs::remove_dir(&path).unwrap();
    std::fs::write(&path, bytes).unwrap();
}

#[test]
fn a_read_error_in_a_lookup_is_a_miss_and_not_a_quarantine() {
    // Spec 6.8: only a content id or an identity that fails verification quarantines an entry.
    let dir = tempfile::tempdir().unwrap();
    let (mut store, id, recipe, bytes) = store_with_unreadable_blob(dir.path());
    assert_eq!(lookup(&mut store, recipe).unwrap(), Vec::<ContentId>::new());
    let entry = store.entries().get(&id).expect("the entry stays in the index");
    assert_eq!(entry.state, EntryState::Live);
    assert_eq!(entry.quarantine, None);

    // The next use checks the entry again. The blob is back, so the lookup hits.
    repair_blob(dir.path(), &id, &bytes);
    assert_eq!(lookup(&mut store, recipe).unwrap(), vec![id]);
}

#[test]
fn a_read_error_in_a_restore_or_a_get_is_an_error_and_not_a_quarantine() {
    let dir = tempfile::tempdir().unwrap();
    let (mut store, id, _, bytes) = store_with_unreadable_blob(dir.path());
    assert!(matches!(store.get(&id, 10), Err(StoreError::Io(_))));
    let mut target = FakeAlways::new();
    let error = store.restore(&id, &mut target, State::Slot(0), &[1u8; 32], 10);
    assert!(matches!(error, Err(StoreError::Io(_))), "{error:?}");
    let entry = store.entries().get(&id).unwrap();
    assert_eq!(entry.state, EntryState::Live);
    assert_eq!(entry.quarantine, None);

    repair_blob(dir.path(), &id, &bytes);
    assert!(store.get(&id, 10).unwrap().is_some());
}

#[test]
fn clearing_a_tripwire_removes_its_entries() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(167);
    let first = put(&mut store, &base_snapshot(1), recipe, 1);
    let error = try_put(&mut store, &base_snapshot(2), recipe, None, 1, Tier::Nvme, 10).unwrap_err();
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
    assert!(try_put(&mut store, &base_snapshot(2), recipe, None, 1, Tier::Nvme, 10).is_err());
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
    assert!(try_put(&mut store, &base_snapshot(2), recipe, None, 1, Tier::Nvme, 10).is_err());
    store.clear_refusal(recipe).unwrap();
    assert_eq!(quarantine_names(dir.path()).len(), 2);

    let again = put(&mut store, &base_snapshot(1), recipe, 1);
    assert_eq!(again, first, "the same bytes give the same content id");
    assert!(try_put(&mut store, &base_snapshot(2), recipe, None, 1, Tier::Nvme, 10).is_err());
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
    let mut store = Store::open_with_nvme_budget(dir.path(), size).unwrap();
    let recipe = recipe(171);
    put(&mut store, &base_snapshot(1), recipe, 1);
    assert!(try_put(&mut store, &base_snapshot(2), recipe, None, 1, Tier::Nvme, 10).is_err());
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
    let error = try_put(&mut store, &base_snapshot(2), recipe, None, 1, Tier::Nvme, 10).unwrap_err();
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

/// An engine that answers one numbered part write with a chosen result. It counts the writes.
struct Scripted {
    inner: FakeAlways,
    answer_at: usize,
    answer: WriteResult,
    writes: usize,
}

impl Scripted {
    fn new(answer_at: usize, answer: WriteResult) -> Self {
        Scripted {
            inner: FakeAlways::new(),
            answer_at,
            answer,
            writes: 0,
        }
    }

    fn next_write(&mut self) -> Option<WriteResult> {
        self.writes += 1;
        (self.writes == self.answer_at).then_some(self.answer)
    }
}

impl Engine for Scripted {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }
}

impl Parts for Scripted {
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        self.inner.parts().unwrap().part_read(state, part, out)
    }

    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        match self.next_write() {
            Some(answer) => answer,
            None => self.inner.parts().unwrap().part_write(state, part, bytes),
        }
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        self.inner.parts().unwrap().cells_read(state, part, from, n, out)
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        match self.next_write() {
            Some(answer) => answer,
            None => self.inner.parts().unwrap().cells_write(state, part, from, n, bytes),
        }
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        self.inner.parts().unwrap().set_n(state, n)
    }
}

/// Put a root and a delta. The chain has four part writes: two for the root, two for the delta.
fn chain_of_two(store: &mut Store, seed: u8) -> ContentId {
    let root_recipe = recipe(seed);
    let root = put(store, &base_snapshot(1), root_recipe, 1);
    let delta_recipe = next_recipe(root_recipe, 2);
    put_delta(store, &delta_at(root, 2), delta_recipe, root_recipe, 1)
}

#[test]
fn a_refused_base_stops_the_restore_before_the_delta() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let delta = chain_of_two(&mut store, 140);
    let mut target = Scripted::new(1, WriteResult::Refused);
    let result = store
        .restore(&delta, &mut target, State::Slot(0), &[1u8; 32], 20)
        .unwrap();
    assert_eq!(
        result,
        Restored::Refused {
            part: "cells".to_owned()
        }
    );
    assert_eq!(target.writes, 1, "no later part is written");
    assert_eq!(store.entries().get(&delta).unwrap().hits, 0);
}

#[test]
fn a_failed_part_stops_the_restore() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let delta = chain_of_two(&mut store, 141);
    let mut target = Scripted::new(2, WriteResult::Failed);
    let result = store
        .restore(&delta, &mut target, State::Slot(0), &[1u8; 32], 20)
        .unwrap();
    assert_eq!(
        result,
        Restored::Failed {
            part: "state".to_owned()
        }
    );
    assert_eq!(target.writes, 2, "no later part is written");
    assert_eq!(store.entries().get(&delta).unwrap().hits, 0);
}

#[test]
fn a_refusal_after_a_write_is_reported_as_failed() {
    // The first part changed the state, so "nothing changed" (6.8) is false. The state is poisoned.
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let delta = chain_of_two(&mut store, 142);
    let mut target = Scripted::new(3, WriteResult::Refused);
    let result = store
        .restore(&delta, &mut target, State::Slot(0), &[1u8; 32], 20)
        .unwrap();
    assert_eq!(
        result,
        Restored::Failed {
            part: "cells".to_owned()
        }
    );
    assert_eq!(target.writes, 3, "no later part is written");
}

fn blob_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir.join("blobs"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn put_in(store: &mut Store, snapshot: &Snapshot, recipe: RecipeId, owner: u64, tier: Tier) -> ContentId {
    try_put(store, snapshot, recipe, None, owner, tier, 10).expect("put")
}

#[test]
fn a_host_entry_keeps_its_bytes_in_memory_and_not_in_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put_in(&mut store, &base_snapshot(1), recipe(60), 1, Tier::Host);
    assert!(blob_names(dir.path()).is_empty(), "a host entry writes no file");
    let size = store.entries().get(&id).unwrap().size;
    assert_eq!(store.host_resident_bytes(), size);
    assert_eq!(store.get(&id, 11).unwrap().unwrap().check, Check::Full);
}

#[test]
fn an_nvme_entry_keeps_its_bytes_in_a_file_and_not_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put_in(&mut store, &base_snapshot(1), recipe(61), 1, Tier::Nvme);
    assert_eq!(blob_names(dir.path()), vec![id.hex()]);
    assert_eq!(store.host_resident_bytes(), 0);
}

#[test]
fn the_content_id_is_the_same_in_either_tier() {
    let host_dir = tempfile::tempdir().unwrap();
    let nvme_dir = tempfile::tempdir().unwrap();
    let mut host = Store::open(host_dir.path()).unwrap();
    let mut nvme = Store::open(nvme_dir.path()).unwrap();
    let snapshot = base_snapshot(3);
    let in_host = put_in(&mut host, &snapshot, recipe(62), 1, Tier::Host);
    let in_nvme = put_in(&mut nvme, &snapshot, recipe(62), 1, Tier::Nvme);
    assert_eq!(in_host, in_nvme);
    assert_eq!(in_host, snapshot.content_id().unwrap());
}

#[test]
fn a_lookup_and_a_restore_read_a_host_entry() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let mut source = FakeAlways::new();
    source.pass(&[segment(&[1, 2, 3, 4, 5, 6])], &[]).unwrap();
    let snapshot = snapshot_from_engine(&mut source, State::Slot(0), &[(1, 0, 6), (2, 2, 4)]).unwrap();
    let id = put_in(&mut store, &snapshot, recipe(63), 0, Tier::Host);
    assert_eq!(lookup(&mut store, recipe(63)).unwrap(), vec![id]);
    let mut target = FakeAlways::new();
    let result = store.restore(&id, &mut target, State::Slot(0), &[1u8; 32], 20).unwrap();
    assert_eq!(result, Restored::Done);
}

#[test]
fn a_restart_removes_host_entries_and_keeps_nvme_entries() {
    let dir = tempfile::tempdir().unwrap();
    let (host, nvme);
    {
        let mut store = Store::open(dir.path()).unwrap();
        host = put_in(&mut store, &base_snapshot(1), recipe(64), 1, Tier::Host);
        nvme = put_in(&mut store, &base_snapshot(2), recipe(65), 1, Tier::Nvme);
        store.pin(host, 1, 100, 10).unwrap();
    }
    let mut store = Store::open(dir.path()).unwrap();
    assert!(!store.entries().contains_key(&host));
    assert!(!store.pins().contains_key(&host), "a pin on a removed entry goes too");
    assert!(store.entries().contains_key(&nvme));
    assert_eq!(store.host_resident_bytes(), 0);
    assert!(lookup(&mut store, recipe(64)).unwrap().is_empty());
    // The index file on disk has no host entry either.
    let text = std::fs::read_to_string(dir.path().join("index.json")).unwrap();
    assert!(!text.contains(&host.hex()));
}

#[test]
fn a_restart_removes_the_blob_file_of_an_old_host_entry() {
    // An index from before the host tier held host entries as files.
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let mut store = Store::open(dir.path()).unwrap();
        id = put_in(&mut store, &base_snapshot(1), recipe(66), 1, Tier::Nvme);
    }
    edit_index(dir.path(), |value| {
        value["entries"][id.hex()]["tier"] = serde_json::json!("Host");
    });
    let store = Store::open(dir.path()).unwrap();
    assert!(!store.entries().contains_key(&id));
    assert!(
        blob_names(dir.path()).is_empty(),
        "no file stays for a removed host entry"
    );
}

#[test]
fn an_nvme_delta_needs_an_nvme_parent() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let base_recipe = recipe(67);
    let base = put_in(&mut store, &base_snapshot(1), base_recipe, 1, Tier::Host);
    let delta_recipe = next_recipe(base_recipe, 2);
    let error = try_put(
        &mut store,
        &delta_at(base, 2),
        delta_recipe,
        Some(base_recipe),
        1,
        Tier::Nvme,
        10,
    )
    .unwrap_err();
    assert!(matches!(error, StoreError::Invalid(_)), "{error:?}");
    assert!(error.to_string().contains("host"), "{error}");
    assert_eq!(store.entries().len(), 1);
    // A host delta on a host parent is fine.
    try_put(
        &mut store,
        &delta_at(base, 2),
        delta_recipe,
        Some(base_recipe),
        1,
        Tier::Host,
        10,
    )
    .unwrap();
}

#[test]
fn the_tiers_have_separate_budgets_and_eviction_runs_in_one_tier() {
    let dir = tempfile::tempdir().unwrap();
    // NVMe has room for two entries, so the old host entry moves there (6.9).
    let mut store = Store::open_with_limits(dir.path(), 5_000, 10_000, u64::MAX, 0).unwrap();
    let size = base_snapshot(1).encode().unwrap().len() as u64;
    assert!(
        size > 2_500 && size <= 5_000,
        "the fixture fills most of one tier: {size}"
    );
    let nvme = put_in(&mut store, &base_snapshot(1), recipe(68), 1, Tier::Nvme);
    let host_old = put_in(&mut store, &base_snapshot(2), recipe(69), 1, Tier::Host);
    // A new host entry moves the old host entry out of memory. The NVMe entry stays, though it is the older one.
    let host_new = put_in(&mut store, &base_snapshot(3), recipe(70), 1, Tier::Host);
    assert!(store.entries().contains_key(&nvme));
    assert_eq!(store.entries()[&host_old].tier, Tier::Nvme);
    assert!(store.entries().contains_key(&host_new));
    assert_eq!(store.host_resident_bytes(), size, "the moved bytes leave memory");
    let mut on_disk = vec![nvme.hex(), host_old.hex()];
    on_disk.sort();
    assert_eq!(blob_names(dir.path()), on_disk);
}

#[test]
fn an_evict_call_names_its_tier() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let nvme = put_in(&mut store, &base_snapshot(1), recipe(71), 1, Tier::Nvme);
    let host = put_in(&mut store, &base_snapshot(2), recipe(72), 1, Tier::Host);
    let victims = store.evict(1, Tier::Host, 1, 20).unwrap();
    assert_eq!(victims, vec![host]);
    assert!(store.entries().contains_key(&nvme));
    assert_eq!(store.host_resident_bytes(), 0);
}

#[test]
fn clearing_a_refusal_keeps_the_bytes_of_a_host_entry_in_quarantine() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let shared = recipe(73);
    let first = put_in(&mut store, &base_snapshot(1), shared, 1, Tier::Host);
    let second = match try_put(&mut store, &base_snapshot(2), shared, None, 1, Tier::Host, 10) {
        Err(StoreError::Tripwire { ids, .. }) => ids.into_iter().find(|id| *id != first).unwrap(),
        other => panic!("expected the tripwire, got {other:?}"),
    };
    store.clear_refusal(shared).unwrap();
    assert!(store.entries().is_empty());
    assert_eq!(store.host_resident_bytes(), 0);
    for (id, seed) in [(first, 1u8), (second, 2u8)] {
        let kept = std::fs::read(dir.path().join("quarantine").join(id.hex())).unwrap();
        assert_eq!(kept, base_snapshot(seed).encode().unwrap());
    }
}

/// Put an interval copy: the host tier, and the recipe of the sealed snapshot with the item "not sealed".
fn put_copy(
    store: &mut Store,
    snapshot: &Snapshot,
    sealed: RecipeId,
    owner: u64,
    now_ms: u64,
) -> Result<ContentId, StoreError> {
    store.put(Put {
        snapshot,
        recipe: recipe_not_sealed(&sealed),
        previous_recipe: None,
        chains_from: None,
        engine: open_engine(),
        identity: [1u8; 32],
        owner,
        tier: Tier::Host,
        reason: Reason::Interval,
        sealed_recipe: Some(sealed),
        provenance: None,
        now_ms,
    })
}

fn copy_store(dir: &Path, copies: u32) -> Store {
    Store::open_with_limits(dir, u64::MAX, u64::MAX, u64::MAX, copies).unwrap()
}

#[test]
fn a_store_with_no_interval_copies_refuses_an_interval_put() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let error = put_copy(&mut store, &base_snapshot(1), recipe(80), 1, 10).unwrap_err();
    assert!(matches!(error, StoreError::Invalid(_)), "{error:?}");
    assert!(error.to_string().contains("interval_copies"), "{error}");
    assert!(store.entries().is_empty());
    assert_eq!(store.host_resident_bytes(), 0);
}

#[test]
fn an_interval_put_in_nvme_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = copy_store(dir.path(), 4);
    let sealed = recipe(81);
    let error = store
        .put(Put {
            snapshot: &base_snapshot(1),
            recipe: recipe_not_sealed(&sealed),
            previous_recipe: None,
            chains_from: None,
            engine: open_engine(),
            identity: [1u8; 32],
            owner: 1,
            tier: Tier::Nvme,
            reason: Reason::Interval,
            sealed_recipe: Some(sealed),
            provenance: None,
            now_ms: 10,
        })
        .unwrap_err();
    assert!(matches!(error, StoreError::Invalid(_)), "{error:?}");
    assert!(error.to_string().contains("host"), "{error}");
    assert!(store.entries().is_empty());
    assert!(blob_names(dir.path()).is_empty());
}

#[test]
fn an_interval_put_needs_the_not_sealed_recipe() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = copy_store(dir.path(), 4);
    let sealed = recipe(82);
    let put_with = |store: &mut Store, recipe: RecipeId, sealed_recipe: Option<RecipeId>| {
        store.put(Put {
            snapshot: &base_snapshot(1),
            recipe,
            previous_recipe: None,
            chains_from: None,
            engine: open_engine(),
            identity: [1u8; 32],
            owner: 1,
            tier: Tier::Host,
            reason: Reason::Interval,
            sealed_recipe,
            provenance: None,
            now_ms: 10,
        })
    };
    // The sealed recipe itself, with no tag 8.
    let error = put_with(&mut store, sealed, Some(sealed)).unwrap_err();
    assert!(error.to_string().contains("not sealed"), "{error}");
    // No sealed recipe to check against.
    assert!(put_with(&mut store, recipe_not_sealed(&sealed), None).is_err());
    // A not sealed recipe of another position.
    assert!(put_with(&mut store, recipe_not_sealed(&recipe(83)), Some(sealed)).is_err());
    assert!(store.entries().is_empty());
    put_with(&mut store, recipe_not_sealed(&sealed), Some(sealed)).unwrap();
}

#[test]
fn only_an_interval_put_may_carry_the_not_sealed_recipe() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = copy_store(dir.path(), 4);
    let sealed = recipe(84);
    let error = store
        .put(Put {
            snapshot: &base_snapshot(1),
            recipe: recipe_not_sealed(&sealed),
            previous_recipe: None,
            chains_from: None,
            engine: open_engine(),
            identity: [1u8; 32],
            owner: 1,
            tier: Tier::Nvme,
            reason: Reason::Anchor,
            sealed_recipe: Some(sealed),
            provenance: None,
            now_ms: 10,
        })
        .unwrap_err();
    assert!(matches!(error, StoreError::Invalid(_)), "{error:?}");
    assert!(store.entries().is_empty());
}

#[test]
fn an_interval_copy_has_no_parent_and_no_delta_names_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = copy_store(dir.path(), 4);
    let base_recipe = recipe(85);
    let base = put(&mut store, &base_snapshot(1), base_recipe, 1);
    let error = put_copy(&mut store, &delta_at(base, 2), base_recipe, 1, 10).unwrap_err();
    assert!(error.to_string().contains("parent"), "{error}");
    let copy_recipe = recipe(86);
    let copy = put_copy(&mut store, &base_snapshot(2), copy_recipe, 1, 10).unwrap();
    let error = try_put(
        &mut store,
        &delta_at(copy, 2),
        next_recipe(recipe_not_sealed(&copy_recipe), 2),
        Some(recipe_not_sealed(&copy_recipe)),
        1,
        Tier::Host,
        10,
    )
    .unwrap_err();
    assert!(error.to_string().contains("interval copy"), "{error}");
    assert_eq!(store.entries().len(), 2);
}

#[test]
fn a_lookup_with_the_item_finds_the_copy_and_one_without_never_does() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = copy_store(dir.path(), 4);
    let sealed = recipe(87);
    let copy = put_copy(&mut store, &base_snapshot(1), sealed, 1, 10).unwrap();
    assert_eq!(lookup(&mut store, recipe_not_sealed(&sealed)).unwrap(), vec![copy]);
    assert!(
        lookup(&mut store, sealed).unwrap().is_empty(),
        "a sealed lookup misses a copy"
    );
    // A sealed snapshot at the same position is a different entry with a different recipe.
    let snapshot = put(&mut store, &base_snapshot(2), sealed, 1);
    assert_eq!(lookup(&mut store, sealed).unwrap(), vec![snapshot]);
    assert_eq!(lookup(&mut store, recipe_not_sealed(&sealed)).unwrap(), vec![copy]);
}

#[test]
fn above_the_cap_the_store_removes_the_least_recently_used_copy_at_once() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = copy_store(dir.path(), 2);
    let first = put_copy(&mut store, &base_snapshot(1), recipe(88), 1, 10).unwrap();
    let second = put_copy(&mut store, &base_snapshot(2), recipe(89), 1, 11).unwrap();
    // A use makes the first copy the more recent one. The second is the least recently used.
    store.get(&first, 20).unwrap();
    let third = put_copy(&mut store, &base_snapshot(3), recipe(90), 1, 30).unwrap();
    assert!(store.entries().contains_key(&first));
    assert!(!store.entries().contains_key(&second));
    assert!(store.entries().contains_key(&third));
    let size = store.entries().get(&first).unwrap().size;
    assert_eq!(store.host_resident_bytes(), 2 * size, "the removed copy leaves memory");
}

#[test]
fn a_cap_of_pinned_copies_refuses_the_put_and_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = copy_store(dir.path(), 1);
    let first = put_copy(&mut store, &base_snapshot(1), recipe(91), 1, 10).unwrap();
    store.pin(first, 1, 1_000, 10).unwrap();
    let error = put_copy(&mut store, &base_snapshot(2), recipe(92), 1, 20).unwrap_err();
    assert!(matches!(error, StoreError::NoSpace), "{error:?}");
    assert_eq!(store.entries().len(), 1);
    assert!(store.entries().contains_key(&first));
}

#[test]
fn a_put_that_needs_bytes_takes_interval_copies_before_the_ranking() {
    let dir = tempfile::tempdir().unwrap();
    // The host budget holds two entries of the fixture.
    let size = base_snapshot(1).encode().unwrap().len() as u64;
    let mut store = Store::open_with_limits(dir.path(), 2 * size, u64::MAX, u64::MAX, 4).unwrap();
    // A cold sealed entry: the value ranking would take it first.
    let cold = put_in(&mut store, &base_snapshot(1), recipe(93), 1, Tier::Host);
    // A hot copy: many hits and a late use.
    let copy = put_copy(&mut store, &base_snapshot(2), recipe(94), 1, 10).unwrap();
    for time in 90..100 {
        store.get(&copy, time).unwrap();
    }
    let new = put_in(&mut store, &base_snapshot(3), recipe(95), 1, Tier::Host);
    assert!(!store.entries().contains_key(&copy), "the copy goes first");
    assert!(store.entries().contains_key(&cold));
    assert!(store.entries().contains_key(&new));
}

#[test]
fn copies_go_least_recently_used_first_not_by_value() {
    let dir = tempfile::tempdir().unwrap();
    let size = base_snapshot(1).encode().unwrap().len() as u64;
    let mut store = Store::open_with_limits(dir.path(), 2 * size, u64::MAX, u64::MAX, 4).unwrap();
    // The older use has many hits. The newer use has none. By value the newer copy would go first.
    let old_use = put_copy(&mut store, &base_snapshot(1), recipe(96), 1, 10).unwrap();
    for _ in 0..20 {
        store.get(&old_use, 20).unwrap();
    }
    let new_use = put_copy(&mut store, &base_snapshot(2), recipe(97), 1, 30).unwrap();
    let new = put_in(&mut store, &base_snapshot(3), recipe(98), 1, Tier::Host);
    assert!(
        !store.entries().contains_key(&old_use),
        "the least recently used copy goes"
    );
    assert!(store.entries().contains_key(&new_use));
    assert!(store.entries().contains_key(&new));
}

#[test]
fn remove_drops_one_interval_copy_and_its_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = copy_store(dir.path(), 4);
    let keep = put_copy(&mut store, &base_snapshot(1), recipe(99), 1, 10).unwrap();
    let drop_id = put_copy(&mut store, &base_snapshot(2), recipe(100), 1, 10).unwrap();
    assert!(store.remove(&drop_id, 20).unwrap());
    assert!(!store.entries().contains_key(&drop_id));
    assert!(store.entries().contains_key(&keep));
    let size = store.entries().get(&keep).unwrap().size;
    assert_eq!(store.host_resident_bytes(), size);
    assert!(lookup(&mut store, recipe_not_sealed(&recipe(100))).unwrap().is_empty());
    // The index on disk agrees.
    let text = std::fs::read_to_string(dir.path().join("index.json")).unwrap();
    assert!(!text.contains(&drop_id.hex()));
    // A second remove of the same id finds nothing. The cap can have removed the copy already.
    assert!(!store.remove(&drop_id, 21).unwrap());
}

#[test]
fn remove_refuses_a_sealed_entry_and_a_pinned_copy() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = copy_store(dir.path(), 4);
    let sealed = put(&mut store, &base_snapshot(1), recipe(101), 1);
    let error = store.remove(&sealed, 20).unwrap_err();
    assert!(matches!(error, StoreError::Invalid(_)), "{error:?}");
    assert!(store.entries().contains_key(&sealed));
    let copy = put_copy(&mut store, &base_snapshot(2), recipe(102), 1, 10).unwrap();
    store.pin(copy, 2, 1_000, 10).unwrap();
    assert!(matches!(store.remove(&copy, 20), Err(StoreError::Invalid(_))));
    assert!(store.entries().contains_key(&copy));
    // After the lease ends, the remove works.
    assert!(store.remove(&copy, 2_000).unwrap());
}

#[test]
fn a_restart_loses_the_interval_copies() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut store = copy_store(dir.path(), 4);
        put_copy(&mut store, &base_snapshot(1), recipe(103), 1, 10).unwrap();
    }
    let store = copy_store(dir.path(), 4);
    assert!(store.entries().is_empty());
}

/// An engine whose `cells_write` leaves the logical row count alone, as mk does. It records each `set_n`.
struct CountSpy {
    inner: Scripted,
    set_n_calls: Vec<u64>,
    set_n_answer: WriteResult,
    log: Vec<&'static str>,
}

impl CountSpy {
    fn new(set_n_answer: WriteResult) -> Self {
        CountSpy {
            inner: Scripted::new(usize::MAX, WriteResult::Ok),
            set_n_calls: Vec::new(),
            set_n_answer,
            log: Vec::new(),
        }
    }
}

impl Engine for CountSpy {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }
}

impl Parts for CountSpy {
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        self.inner.part_read(state, part, out)
    }

    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        self.log.push("part_write");
        self.inner.part_write(state, part, bytes)
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        self.inner.cells_read(state, part, from, n, out)
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        self.log.push("cells_write");
        self.inner.cells_write(state, part, from, n, bytes)
    }

    fn set_n(&mut self, _state: State, n: u64) -> WriteResult {
        self.log.push("set_n");
        self.set_n_calls.push(n);
        self.set_n_answer
    }
}

#[test]
fn a_restore_sets_n_to_the_end_of_the_cells_after_every_part() {
    // Kaibo job 2, finding 1 (spec 6.8, ABI): the caller calls `set_n` after it writes every part.
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let delta = chain_of_two(&mut store, 150);
    let mut target = CountSpy::new(WriteResult::Ok);
    let result = store
        .restore(&delta, &mut target, State::Slot(0), &[1u8; 32], 20)
        .unwrap();
    assert_eq!(result, Restored::Done);
    assert_eq!(
        target.set_n_calls,
        vec![4],
        "the root covers 0..2 and the delta covers 2..4"
    );
    assert_eq!(target.log.last(), Some(&"set_n"), "set_n comes after every part write");
    assert_eq!(target.log.iter().filter(|call| **call == "set_n").count(), 1);
}

#[test]
fn a_refused_set_n_after_the_writes_is_a_failed_restore() {
    // Every part is written, so the state changed. A refusal is a failure (6.8), and the entry is not touched.
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let delta = chain_of_two(&mut store, 151);
    for answer in [WriteResult::Refused, WriteResult::Failed] {
        let mut target = CountSpy::new(answer);
        let result = store
            .restore(&delta, &mut target, State::Slot(0), &[1u8; 32], 20)
            .unwrap();
        assert_eq!(
            result,
            Restored::Failed {
                part: "cells".to_owned()
            },
            "{answer:?}"
        );
        assert_eq!(store.entries().get(&delta).unwrap().hits, 0);
    }
}

/// Put a root and a delta in a store whose budget holds exactly them. Returns their ids and the budget.
fn full_chain_of_two(dir: &Path, seed: u8) -> (Store, ContentId, ContentId, RecipeId) {
    let mut store = Store::open(dir).unwrap();
    let root_recipe = recipe(seed);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let delta_recipe = next_recipe(root_recipe, 2);
    let d1 = put_delta(&mut store, &delta_at(root, 2), delta_recipe, root_recipe, 1);
    let used = store.entries().get(&root).unwrap().size + store.entries().get(&d1).unwrap().size;
    store.set_budget(Tier::Nvme, used);
    (store, root, d1, delta_recipe)
}

#[test]
fn a_put_never_evicts_the_ancestors_of_its_delta() {
    // Kaibo job 2, finding 3 (6.5, 6.9): the only victims are the root and D1, which are the ancestors of D2.
    let dir = tempfile::tempdir().unwrap();
    let (mut store, root, d1, d1_recipe) = full_chain_of_two(dir.path(), 160);
    let d2_recipe = next_recipe(d1_recipe, 3);
    let error = try_put(
        &mut store,
        &delta_at(d1, 4),
        d2_recipe,
        Some(d1_recipe),
        1,
        Tier::Nvme,
        10,
    );
    assert!(matches!(error, Err(StoreError::NoSpace)), "{error:?}");
    assert!(store.entries().contains_key(&root));
    assert!(store.entries().contains_key(&d1));
    assert_eq!(store.entries().len(), 2);
}

#[test]
fn a_put_under_pressure_evicts_another_entry_and_keeps_the_chain_whole() {
    let dir = tempfile::tempdir().unwrap();
    let (mut store, root, d1, d1_recipe) = full_chain_of_two(dir.path(), 161);
    // A hot standalone entry. By value it ranks above the cold chain.
    store.set_budget(Tier::Nvme, u64::MAX);
    let hot = put(&mut store, &wide_snapshot(9), recipe(162), 1);
    for time in 0..5 {
        store.get(&hot, 10 + time).unwrap();
    }
    let used: u64 = store.entries().values().map(|entry| entry.size).sum();
    store.set_budget(Tier::Nvme, used);
    let d2_recipe = next_recipe(d1_recipe, 3);
    let d2 = try_put(
        &mut store,
        &delta_at(d1, 4),
        d2_recipe,
        Some(d1_recipe),
        1,
        Tier::Nvme,
        20,
    )
    .expect("put D2");
    assert!(store.entries().contains_key(&root));
    assert!(store.entries().contains_key(&d1));
    assert!(store.entries().contains_key(&d2));
    assert!(!store.entries().contains_key(&hot), "the other entry made room");
    assert_eq!(lookup(&mut store, d2_recipe).unwrap().len(), 3);
}

#[test]
fn a_put_of_an_existing_id_in_another_tier_is_refused() {
    // Kaibo job 2, finding 10: Ok for a host entry would be a lie to a put that asked for NVMe.
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let snapshot = base_snapshot(4);
    let id = put_in(&mut store, &snapshot, recipe(170), 1, Tier::Host);
    let before = store.entries().clone();
    let error = try_put(&mut store, &snapshot, recipe(170), None, 1, Tier::Nvme, 20);
    let Err(StoreError::Invalid(message)) = error else {
        panic!("expected a refusal, got {error:?}");
    };
    assert!(message.contains("tier"), "{message}");
    assert_eq!(store.entries(), &before);
    assert_eq!(store.entries().get(&id).unwrap().tier, Tier::Host);
    assert!(blob_names(dir.path()).is_empty(), "no file appeared");
    // The same tier is still a no-op.
    assert_eq!(
        try_put(&mut store, &snapshot, recipe(170), None, 1, Tier::Host, 20).unwrap(),
        id
    );
}

/// A store with a pinned entry that lost its protection (it is larger than the budget).
fn store_with_an_unprotected_pin(dir: &Path) -> (Store, ContentId) {
    let mut store = Store::open_with_nvme_budget(dir, u64::MAX).unwrap();
    let id = put(&mut store, &wide_snapshot(1), recipe(171), 1);
    store.pin(id, 1, 100, 10).unwrap();
    let size = store.entries().get(&id).unwrap().size;
    store.set_budget(Tier::Nvme, size - 1);
    (store, id)
}

#[test]
fn an_eviction_by_a_put_removes_the_pins_of_its_victims() {
    // Kaibo job 2, finding 11: a re-put of the same content id must not inherit a stale pin.
    let dir = tempfile::tempdir().unwrap();
    let (mut store, victim) = store_with_an_unprotected_pin(dir.path());
    put(&mut store, &base_snapshot(2), recipe(172), 1);
    assert!(
        !store.entries().contains_key(&victim),
        "the put evicted the pinned entry"
    );
    assert!(!store.pins().contains_key(&victim));
}

#[test]
fn evict_removes_the_pins_of_its_victims() {
    let dir = tempfile::tempdir().unwrap();
    let (mut store, victim) = store_with_an_unprotected_pin(dir.path());
    assert_eq!(store.evict(1, Tier::Nvme, 1, 10).unwrap(), vec![victim]);
    assert!(!store.pins().contains_key(&victim));
}

#[test]
fn a_failed_quarantine_write_loses_no_host_bytes() {
    // Kaibo job 2, finding 4 (6.8): a failure never deletes content. A host entry has only its bytes in memory.
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let recipe = recipe(180);
    let first = put_in(&mut store, &base_snapshot(1), recipe, 1, Tier::Host);
    let error = try_put(&mut store, &base_snapshot(2), recipe, None, 1, Tier::Host, 10).unwrap_err();
    assert!(matches!(error, StoreError::Tripwire { .. }));
    let held = store.host_resident_bytes();
    assert!(held > 0);
    let quarantine = dir.path().join("quarantine");
    std::fs::set_permissions(&quarantine, std::fs::Permissions::from_mode(0o500)).unwrap();
    assert!(
        std::fs::File::create(quarantine.join("probe")).is_err(),
        "this test needs a directory that refuses writes; run it as a user other than root"
    );
    let result = store.clear_refusal(recipe);
    std::fs::set_permissions(&quarantine, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(result.is_err(), "the write failed, so clear_refusal fails");
    assert_eq!(store.host_resident_bytes(), held, "the bytes are still in memory");
    assert!(store.entries().contains_key(&first));
    assert!(store.refused().contains(&recipe));
    // The retry saves both entries.
    store.clear_refusal(recipe).unwrap();
    assert_eq!(store.host_resident_bytes(), 0);
    assert!(quarantine.join(first.hex()).exists());
    assert_eq!(std::fs::read_dir(&quarantine).unwrap().count(), 2);
}

/// The index as a person sees it: the entries, the pins and the refused recipes.
fn view(store: &Store) -> String {
    format!("{:?} {:?} {:?}", store.entries(), store.pins(), store.refused())
}

#[test]
fn a_failed_pin_changes_nothing() {
    // Kaibo job 2, finding 5 (6.10): a refused pin changes nothing.
    for point in [
        nagaya_store::FailPoint::IndexWrite,
        nagaya_store::FailPoint::IndexRename,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path()).unwrap();
        let id = put(&mut store, &base_snapshot(1), recipe(181), 1);
        let before = view(&store);
        store.fail_next(point);
        assert!(store.pin(id, 1, 100, 10).is_err());
        assert_eq!(view(&store), before, "{point:?}");
        assert!(store.pins().is_empty());
    }
}

#[test]
fn a_failed_unpin_changes_nothing() {
    for point in [
        nagaya_store::FailPoint::IndexWrite,
        nagaya_store::FailPoint::IndexRename,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path()).unwrap();
        let id = put(&mut store, &base_snapshot(1), recipe(182), 1);
        store.pin(id, 1, 100, 10).unwrap();
        store.pin(id, 2, 100, 10).unwrap();
        let before = view(&store);
        store.fail_next(point);
        assert!(store.unpin(&id, 1).is_err());
        assert_eq!(view(&store), before, "{point:?}");
        assert_eq!(store.pins().get(&id).unwrap().len(), 2);
    }
}

#[test]
fn a_failed_lookup_and_pin_adds_no_pin() {
    for point in [
        nagaya_store::FailPoint::IndexWrite,
        nagaya_store::FailPoint::IndexRename,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path()).unwrap();
        let recipe = recipe(183);
        put(&mut store, &base_snapshot(1), recipe, 1);
        let mut engine = FakeAlways::new();
        store.fail_next(point);
        let pin = Pin {
            owner: 1,
            expires_ms: 100,
        };
        let result = store.lookup_and_pin(recipe, &Tolerance::Exact, &mut engine, 0, pin, 10);
        assert!(result.is_err());
        assert!(store.pins().is_empty(), "{point:?}");
    }
}

#[test]
fn a_failed_clear_entry_leaves_the_entry_quarantined() {
    for point in [
        nagaya_store::FailPoint::IndexWrite,
        nagaya_store::FailPoint::IndexRename,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path()).unwrap();
        let id = put(&mut store, &base_snapshot(1), recipe(184), 1);
        let mut target = FakeAlways::new();
        assert!(store.restore(&id, &mut target, State::Slot(0), &[9u8; 32], 10).is_err());
        assert_eq!(store.entries().get(&id).unwrap().state, EntryState::Quarantined);
        let before = view(&store);
        store.fail_next(point);
        assert!(store.clear_entry(&id).is_err());
        assert_eq!(view(&store), before, "{point:?}");
        // The entry is still quarantined, so a retry does the work.
        store.clear_entry(&id).unwrap();
        assert_eq!(store.entries().get(&id).unwrap().state, EntryState::Live);
    }
}

// A park moves a host entry to NVMe, bit for bit (spec 10.3, ontology "Park").

#[test]
fn a_park_moves_a_host_entry_to_nvme_and_a_restart_keeps_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put_in(&mut store, &base_snapshot(1), recipe(170), 7, Tier::Host);
    let before = store.entries()[&id].clone();
    store.park(&id, 20).unwrap();
    let after = store.entries()[&id].clone();
    assert_eq!(after.tier, Tier::Nvme);
    assert_eq!(
        (after.recipe, after.owner, after.reason, after.size),
        (before.recipe, before.owner, before.reason, before.size),
        "a park changes only the tier"
    );
    assert_eq!(blob_names(dir.path()), vec![id.hex()]);
    assert_eq!(store.host_resident_bytes(), 0, "the host tier frees the bytes");
    drop(store);

    let mut store = Store::open(dir.path()).unwrap();
    assert_eq!(store.entries()[&id].tier, Tier::Nvme, "6.11 removes host entries only");
    assert_eq!(lookup(&mut store, recipe(170)).unwrap(), vec![id]);
}

#[test]
fn a_park_of_an_nvme_entry_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put_in(&mut store, &base_snapshot(1), recipe(171), 1, Tier::Nvme);
    let before = store.entries().clone();
    store.park(&id, 20).unwrap();
    assert_eq!(store.entries(), &before);
}

#[test]
fn a_park_refuses_an_interval_copy_and_an_absent_entry() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = copy_store(dir.path(), 4);
    let copy = put_copy(&mut store, &base_snapshot(1), recipe(172), 1, 10).unwrap();
    let before = store.entries().clone();
    let error = store.park(&copy, 20).unwrap_err();
    assert!(matches!(error, StoreError::Invalid(_)), "{error:?}");
    assert!(error.to_string().contains("6.13"), "{error}");
    assert_eq!(store.entries(), &before, "a refused park changes nothing");
    let absent = base_snapshot(2).content_id().unwrap();
    assert!(matches!(store.park(&absent, 20), Err(StoreError::Invalid(_))));
}

#[test]
fn a_park_with_no_room_on_nvme_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let size = base_snapshot(1).encode().unwrap().len() as u64;
    let mut store = Store::open_with_limits(dir.path(), u64::MAX, size - 1, u64::MAX, 0).unwrap();
    let id = put_in(&mut store, &base_snapshot(1), recipe(173), 1, Tier::Host);
    let before = store.entries().clone();
    assert!(matches!(store.park(&id, 20), Err(StoreError::NoSpace)));
    assert_eq!(store.entries(), &before);
    assert!(blob_names(dir.path()).is_empty());
    assert_eq!(store.host_resident_bytes(), size);
}

#[test]
fn a_park_evicts_an_nvme_entry_of_the_same_owner_for_room() {
    let dir = tempfile::tempdir().unwrap();
    let size = base_snapshot(1).encode().unwrap().len() as u64;
    let mut store = Store::open_with_limits(dir.path(), u64::MAX, size, u64::MAX, 0).unwrap();
    let old = put_in(&mut store, &base_snapshot(2), recipe(174), 1, Tier::Nvme);
    let id = put_in(&mut store, &base_snapshot(1), recipe(175), 1, Tier::Host);
    store.park(&id, 20).unwrap();
    assert!(!store.entries().contains_key(&old), "6.9: the park made room");
    assert_eq!(store.entries()[&id].tier, Tier::Nvme);
    assert_eq!(blob_names(dir.path()), vec![id.hex()]);
}

#[test]
fn a_park_refuses_a_delta_whose_parent_is_in_the_host_tier() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let base = put_in(&mut store, &base_snapshot(1), recipe(176), 1, Tier::Host);
    let next = next_recipe(recipe(176), 1);
    let delta = try_put(
        &mut store,
        &delta_at(base, 2),
        next,
        Some(recipe(176)),
        1,
        Tier::Host,
        10,
    )
    .unwrap();
    let before = store.entries().clone();
    let error = store.park(&delta, 20).unwrap_err();
    assert!(error.to_string().contains("park the parent first"), "{error}");
    assert_eq!(store.entries(), &before);
    store.park(&base, 20).unwrap();
    store.park(&delta, 20).unwrap();
    assert_eq!(
        store.entries()[&delta].tier,
        Tier::Nvme,
        "a parked parent lets its delta park"
    );
}

#[test]
fn eviction_takes_a_chain_root_after_its_descendants_when_more_space_is_needed() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let root_recipe = recipe(240);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let middle_recipe = next_recipe(root_recipe, 2);
    let middle = put_delta(&mut store, &delta_at(root, 2), middle_recipe, root_recipe, 1);
    let tip_recipe = next_recipe(middle_recipe, 4);
    let tip = put_delta(&mut store, &delta_at(middle, 4), tip_recipe, middle_recipe, 1);
    let total: u64 = store.entries().values().map(|entry| entry.size).sum();
    let victims = store.evict(1, Tier::Nvme, total, 10).unwrap();
    assert_eq!(victims, vec![middle, tip, root]);
    assert!(store.entries().is_empty());

    let root2_recipe = recipe(241);
    let root2 = put(&mut store, &base_snapshot(2), root2_recipe, 1);
    let tip2_recipe = next_recipe(root2_recipe, 2);
    let tip2 = put_delta(&mut store, &delta_at(root2, 2), tip2_recipe, root2_recipe, 1);
    store.get(&root2, 20).unwrap();
    let total2: u64 = store.entries().values().map(|entry| entry.size).sum();
    let victims2 = store.evict(1, Tier::Nvme, total2, 20).unwrap();
    assert_eq!(victims2, vec![tip2, root2]);
    assert!(store.entries().is_empty());
}

#[test]
fn a_pinned_chain_larger_than_the_budget_loses_protection() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_nvme_budget(dir.path(), 10_000).unwrap();
    let root_recipe = recipe(242);
    let root = put(&mut store, &base_snapshot(1), root_recipe, 1);
    let tip_recipe = next_recipe(root_recipe, 2);
    let tip = put_delta(&mut store, &delta_at(root, 2), tip_recipe, root_recipe, 1);
    let root_size = store.entries()[&root].size;
    let tip_size = store.entries()[&tip].size;
    let budget = root_size.max(tip_size) + 1;
    assert!(root_size + tip_size > budget, "the chain exceeds the budget");

    store.pin(tip, 1, 100, 10).unwrap();
    store.set_budget(Tier::Nvme, budget);
    let victims = store.evict(1, Tier::Nvme, 1, 10).unwrap();
    assert!(victims.contains(&tip), "the oversize pinned chain loses protection");

    store.set_budget(Tier::Nvme, 10_000);
    let root2_recipe = recipe(243);
    let root2 = put(&mut store, &base_snapshot(3), root2_recipe, 1);
    let tip2_recipe = next_recipe(root2_recipe, 2);
    let tip2 = put_delta(&mut store, &delta_at(root2, 2), tip2_recipe, root2_recipe, 1);
    let mut engine = FakeAlways::new();
    store
        .lookup_and_pin(
            tip2_recipe,
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
    store.set_budget(Tier::Nvme, budget);
    let victims2 = store.evict(1, Tier::Nvme, root_size + tip_size, 10).unwrap();
    assert!(victims2.contains(&root2) && victims2.contains(&tip2));
}

// The stop parks the pinned entries of the host tier, so a pin stays through a restart (spec 6.11, Decided
// 2026-10-23).

#[test]
fn a_park_of_pins_keeps_each_pinned_host_entry_and_its_parents_through_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let alone = put_in(&mut store, &base_snapshot(1), recipe(200), 1, Tier::Host);
    let parent = put_in(&mut store, &base_snapshot(2), recipe(201), 1, Tier::Host);
    let next = next_recipe(recipe(201), 1);
    let child = try_put(
        &mut store,
        &delta_at(parent, 2),
        next,
        Some(recipe(201)),
        1,
        Tier::Host,
        10,
    )
    .unwrap();
    let unpinned = put_in(&mut store, &base_snapshot(3), recipe(202), 1, Tier::Host);
    let ended = put_in(&mut store, &wide_snapshot(4), recipe(203), 1, Tier::Host);
    store.pin(alone, 5, 1_000, 10).unwrap();
    store.pin(child, 6, 1_000, 10).unwrap();
    store.pin(ended, 7, 15, 10).unwrap();
    let sizes: u64 = [alone, parent, child].iter().map(|id| store.entries()[id].size).sum();

    let report = store.park_pinned(20);
    assert!(report.failures.is_empty(), "{:?}", report.failures);
    let mut parked = report.parked.clone();
    parked.sort();
    let mut wanted = vec![alone, parent, child];
    wanted.sort();
    assert_eq!(parked, wanted, "the pinned entries and the parent under a pin park");
    assert_eq!(report.bytes, sizes);
    let position = |id| report.parked.iter().position(|parked| *parked == id).unwrap();
    assert!(position(parent) < position(child), "a parent parks before its child");
    assert_eq!(store.entries()[&unpinned].tier, Tier::Host, "an unpinned entry stays");
    assert_eq!(store.entries()[&ended].tier, Tier::Host, "an ended pin holds nothing");
    drop(store);

    let mut store = Store::open(dir.path()).unwrap();
    for id in [alone, parent, child] {
        assert_eq!(store.entries()[&id].tier, Tier::Nvme);
    }
    assert!(!store.entries().contains_key(&unpinned));
    assert!(!store.entries().contains_key(&ended));
    assert_eq!(
        store.pins()[&alone],
        vec![Pin {
            owner: 5,
            expires_ms: 1_000
        }]
    );
    assert_eq!(
        store.pins()[&child],
        vec![Pin {
            owner: 6,
            expires_ms: 1_000
        }]
    );
    assert_eq!(lookup(&mut store, next).unwrap(), vec![parent, child]);
}

#[test]
fn a_park_of_pins_leaves_a_pinned_interval_copy() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = copy_store(dir.path(), 4);
    let copy = put_copy(&mut store, &base_snapshot(1), recipe(204), 1, 10).unwrap();
    store.pin(copy, 5, 1_000, 10).unwrap();
    let report = store.park_pinned(20);
    assert!(report.parked.is_empty());
    assert!(
        report.failures.is_empty(),
        "6.13: a copy stays in the host tier, and that is no failure"
    );
    assert_eq!(store.entries()[&copy].tier, Tier::Host);
}

#[test]
fn a_park_of_pins_reports_a_park_with_no_room() {
    let dir = tempfile::tempdir().unwrap();
    let size = base_snapshot(1).encode().unwrap().len() as u64;
    let mut store = Store::open_with_limits(dir.path(), u64::MAX, size - 1, u64::MAX, 0).unwrap();
    let id = put_in(&mut store, &base_snapshot(1), recipe(205), 1, Tier::Host);
    store.pin(id, 5, 1_000, 10).unwrap();
    let report = store.park_pinned(20);
    assert!(report.parked.is_empty());
    assert_eq!(report.failures.len(), 1, "{:?}", report.failures);
    assert!(report.failures[0].contains(&id.hex()), "{:?}", report.failures);
    assert_eq!(store.entries()[&id].tier, Tier::Host);
}

// A host victim moves to NVMe (spec 6.9, "A host victim moves to NVMe", Decided 2026-11-01).

fn host_put(store: &mut Store, snapshot: &Snapshot, recipe: RecipeId, now_ms: u64) -> Result<ContentId, StoreError> {
    try_put(store, snapshot, recipe, None, 1, Tier::Host, now_ms)
}

fn base_size() -> u64 {
    base_snapshot(1).encode().unwrap().len() as u64
}

#[test]
fn a_put_that_needs_host_room_parks_the_victim_on_nvme() {
    let dir = tempfile::tempdir().unwrap();
    let size = base_size();
    let mut store = Store::open_with_limits(dir.path(), 2 * size, u64::MAX, u64::MAX, 0).unwrap();
    let ids: Vec<ContentId> = (0..3u8)
        .map(|i| host_put(&mut store, &base_snapshot(i + 1), recipe(180 + i), 10 + u64::from(i)).unwrap())
        .collect();
    let moved: Vec<ContentId> = ids
        .iter()
        .copied()
        .filter(|id| store.entries()[id].tier == Tier::Nvme)
        .collect();
    assert_eq!(moved.len(), 1, "one victim moves, and no entry goes");
    assert_eq!(
        store.entries()[&ids[2]].tier,
        Tier::Host,
        "the new entry is in the host tier"
    );
    assert_eq!(store.host_resident_bytes(), 2 * size);
    assert_eq!(
        blob_names(dir.path()),
        vec![moved[0].hex()],
        "the moved entry has its file"
    );
    let moved_recipe = store.entries()[&moved[0]].recipe;
    drop(store);

    let mut store = Store::open(dir.path()).unwrap();
    assert_eq!(
        lookup(&mut store, moved_recipe).unwrap(),
        vec![moved[0]],
        "a restart keeps the moved entry"
    );
}

#[test]
fn a_host_victim_with_no_room_on_nvme_goes_as_before() {
    let dir = tempfile::tempdir().unwrap();
    let size = base_size();
    let mut store = Store::open_with_limits(dir.path(), 2 * size, size - 1, u64::MAX, 0).unwrap();
    for i in 0..3u8 {
        host_put(&mut store, &base_snapshot(i + 1), recipe(185 + i), 10 + u64::from(i)).unwrap();
    }
    assert_eq!(store.entries().len(), 2, "the victim goes");
    assert!(store.entries().values().all(|entry| entry.tier == Tier::Host));
    assert!(blob_names(dir.path()).is_empty());
}

#[test]
fn an_interval_copy_never_moves_to_nvme() {
    let dir = tempfile::tempdir().unwrap();
    let size = base_size();
    let mut store = Store::open_with_limits(dir.path(), 2 * size, u64::MAX, u64::MAX, 4).unwrap();
    let copy = put_copy(&mut store, &base_snapshot(1), recipe(190), 1, 10).unwrap();
    let a = host_put(&mut store, &base_snapshot(2), recipe(191), 11).unwrap();
    let b = host_put(&mut store, &base_snapshot(3), recipe(192), 12).unwrap();
    assert!(!store.entries().contains_key(&copy), "the copy goes first (6.13)");
    assert_eq!(
        (store.entries()[&a].tier, store.entries()[&b].tier),
        (Tier::Host, Tier::Host)
    );
    assert!(blob_names(dir.path()).is_empty());
}

/// A host tier that holds a root and its tip, and nothing more.
fn chain_in_host(dir: &Path) -> (Store, ContentId, ContentId) {
    let root_snapshot = base_snapshot(1);
    let root_id = root_snapshot.content_id().unwrap();
    let budget = root_snapshot.encode().unwrap().len() + delta_at(root_id, 2).encode().unwrap().len();
    let mut store = Store::open_with_limits(dir, budget as u64, u64::MAX, u64::MAX, 0).unwrap();
    let root = host_put(&mut store, &root_snapshot, recipe(195), 10).unwrap();
    let tip_recipe = next_recipe(recipe(195), 2);
    let tip = try_put(
        &mut store,
        &delta_at(root, 2),
        tip_recipe,
        Some(recipe(195)),
        1,
        Tier::Host,
        11,
    )
    .unwrap();
    (store, root, tip)
}

#[test]
fn a_chain_moves_to_nvme_parent_first() {
    let dir = tempfile::tempdir().unwrap();
    let (mut store, root, tip) = chain_in_host(dir.path());
    let tip_recipe = store.entries()[&tip].recipe;
    // The wide put needs more room than the tip alone, so the root goes and takes its chain (6.9).
    host_put(&mut store, &wide_snapshot(3), recipe(196), 12).unwrap();
    assert_eq!(store.entries()[&root].tier, Tier::Nvme);
    assert_eq!(store.entries()[&tip].tier, Tier::Nvme);
    assert_eq!(store.entries()[&tip].parent, Some(root));
    drop(store);
    let mut store = Store::open(dir.path()).unwrap();
    assert_eq!(
        lookup(&mut store, tip_recipe).unwrap(),
        vec![root, tip],
        "a restart keeps the chain"
    );
}

#[test]
fn a_tip_whose_parent_stays_in_the_host_tier_goes_as_before() {
    let dir = tempfile::tempdir().unwrap();
    let (mut store, root, tip) = chain_in_host(dir.path());
    // A put of one base's size takes only the tip. A park would cut the chain at a restart, so the tip goes.
    host_put(&mut store, &base_snapshot(3), recipe(197), 12).unwrap();
    assert!(!store.entries().contains_key(&tip));
    assert_eq!(store.entries()[&root].tier, Tier::Host);
    assert!(blob_names(dir.path()).is_empty());
}

#[test]
fn a_refused_host_put_moves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let size = base_size();
    let mut store = Store::open_with_limits(dir.path(), 2 * size, u64::MAX, u64::MAX, 0).unwrap();
    let a = host_put(&mut store, &base_snapshot(1), recipe(201), 10).unwrap();
    host_put(&mut store, &base_snapshot(2), recipe(198), 11).unwrap();
    store.pin(a, 1, 1000, 11).unwrap();
    let before = store.entries().clone();
    // One victim can move, but the pin keeps the other, so the put cannot get its room.
    let big = wide_snapshot(3);
    assert!(
        big.encode().unwrap().len() as u64 > size,
        "the wide snapshot needs more than one entry's room"
    );
    assert!(matches!(
        host_put(&mut store, &big, recipe(199), 12),
        Err(StoreError::NoSpace)
    ));
    assert_eq!(store.entries(), &before, "a refused put changes nothing");
    assert!(blob_names(dir.path()).is_empty());
}
