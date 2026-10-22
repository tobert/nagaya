//! The store records its holdings, lookups, evictions and restores from the real call sites (spec 9.4, 9.5).

use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    BitDependence, ChainLimits, Dtype, Engine, Identity, Limits, PartError, PartId, PartKind, Parts, PassError,
    PassOut, Segment, State, Tap, WriteResult,
};
use nagaya_store::{
    ContentId, EngineFacts, Put, Reason, RecipeId, Restored, Snapshot, Store, StoreError, Tensor, Tier, Tolerance,
    recipe_boundary, recipe_not_sealed, recipe_root,
};
use nagaya_telemetry::testing::{Capture, Point};

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

fn delta_at(parent: ContentId, start: u64) -> Snapshot {
    let seed = start as u8;
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

fn engine_facts() -> EngineFacts {
    EngineFacts {
        name: "fake".to_owned(),
        determinism: nagaya_abi::Determinism::Always,
        bit_dependence: Vec::<BitDependence>::new(),
        chain: ChainLimits {
            max_depth: u32::MAX,
            delta_fraction: 1.0,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn put_full(
    store: &mut Store,
    snapshot: &Snapshot,
    recipe: RecipeId,
    previous: Option<RecipeId>,
    tier: Tier,
    reason: Reason,
    sealed: Option<RecipeId>,
    provenance: Option<&str>,
    now_ms: u64,
) -> Result<ContentId, StoreError> {
    store.put(Put {
        snapshot,
        recipe,
        previous_recipe: previous,
        chains_from: None,
        engine: engine_facts(),
        identity: [1u8; 32],
        owner: 1,
        tier,
        reason,
        sealed_recipe: sealed,
        provenance: provenance.map(str::to_owned),
        now_ms,
    })
}

fn put(store: &mut Store, snapshot: &Snapshot, recipe: RecipeId) -> ContentId {
    put_full(
        store,
        snapshot,
        recipe,
        None,
        Tier::Nvme,
        Reason::Anchor,
        None,
        None,
        10,
    )
    .expect("put")
}

fn put_copy(store: &mut Store, snapshot: &Snapshot, sealed: RecipeId, now_ms: u64) -> ContentId {
    put_full(
        store,
        snapshot,
        recipe_not_sealed(&sealed),
        None,
        Tier::Host,
        Reason::Interval,
        Some(sealed),
        None,
        now_ms,
    )
    .expect("put copy")
}

fn lookup(store: &mut Store, recipe: RecipeId, tolerance: &Tolerance) -> Result<Vec<ContentId>, StoreError> {
    let mut engine = FakeAlways::new();
    store.lookup(recipe, tolerance, &mut engine, 0, 10)
}

fn find(points: &[Point], wanted: &[(&str, &str)]) -> Option<Point> {
    points
        .iter()
        .find(|point| {
            wanted
                .iter()
                .all(|(key, value)| point.attributes.get(*key).map(String::as_str) == Some(*value))
        })
        .cloned()
}

fn lookups(capture: &Capture) -> Vec<(String, f64)> {
    let mut rows: Vec<(String, f64)> = capture
        .metric("nagaya.store.lookups")
        .iter()
        .map(|point| {
            assert_eq!(
                point.attributes["engine"], "fake-always",
                "the label is the model of the engine"
            );
            (point.attributes["outcome"].clone(), point.value)
        })
        .collect();
    rows.sort_by(|left, right| left.partial_cmp(right).unwrap());
    rows
}

fn restores(capture: &Capture) -> Vec<(String, f64)> {
    let mut rows: Vec<(String, f64)> = capture
        .metric("nagaya.store.restores")
        .iter()
        .map(|point| (point.attributes["result"].clone(), point.value))
        .collect();
    rows.sort_by(|left, right| left.partial_cmp(right).unwrap());
    rows
}

fn evictions(capture: &Capture) -> Vec<(String, String, f64)> {
    let mut rows: Vec<(String, String, f64)> = capture
        .metric("nagaya.store.evictions")
        .iter()
        .map(|point| {
            (
                point.attributes["tier"].clone(),
                point.attributes["reason"].clone(),
                point.value,
            )
        })
        .collect();
    rows.sort_by(|left, right| left.partial_cmp(right).unwrap());
    rows
}

fn held(capture: &Capture, tier: &str) -> Option<(f64, f64)> {
    let bytes = find(
        &capture.metric("nagaya.store.bytes"),
        &[("tier", tier), ("engine", "fake")],
    )?;
    let entries = find(
        &capture.metric("nagaya.store.entries"),
        &[("tier", tier), ("engine", "fake")],
    )?;
    Some((bytes.value, entries.value))
}

fn size_of(snapshot: &Snapshot) -> f64 {
    snapshot.encode().unwrap().len() as f64
}

#[test]
fn a_put_records_the_bytes_and_the_entries_held_in_its_tier() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    put(&mut store, &base_snapshot(1), recipe(1));
    assert_eq!(held(&capture, "nvme"), Some((size_of(&base_snapshot(1)), 1.0)));
    put(&mut store, &base_snapshot(2), recipe(2));
    assert_eq!(
        held(&capture, "nvme"),
        Some((size_of(&base_snapshot(1)) + size_of(&base_snapshot(2)), 2.0))
    );
    assert_eq!(held(&capture, "host"), None, "no entry lives in the host tier");
}

#[test]
fn a_series_that_empties_reads_zero() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(1));
    store.evict(1, Tier::Nvme, 1, 10).unwrap();
    assert!(!store.entries().contains_key(&id));
    assert_eq!(held(&capture, "nvme"), Some((0.0, 0.0)));
}

#[test]
fn a_reopened_store_records_what_it_holds() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut store = Store::open(dir.path()).unwrap();
        put(&mut store, &base_snapshot(1), recipe(1));
    }
    let capture = Capture::start(Vec::new());
    let _store = Store::open(dir.path()).unwrap();
    assert_eq!(held(&capture, "nvme"), Some((size_of(&base_snapshot(1)), 1.0)));
}

#[test]
fn a_lookup_that_finds_a_chain_counts_exact() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    put(&mut store, &base_snapshot(1), recipe(1));
    assert_eq!(lookup(&mut store, recipe(1), &Tolerance::Exact).unwrap().len(), 1);
    assert_eq!(lookups(&capture), [("exact".to_owned(), 1.0)]);
}

#[test]
fn a_lookup_that_finds_an_approximate_chain_counts_approx() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    put_full(
        &mut store,
        &base_snapshot(1),
        recipe(1),
        None,
        Tier::Nvme,
        Reason::Anchor,
        None,
        Some("lossy"),
        10,
    )
    .unwrap();
    let tolerance = Tolerance::Approx(vec!["lossy".to_owned()]);
    assert_eq!(lookup(&mut store, recipe(1), &tolerance).unwrap().len(), 1);
    assert_eq!(lookups(&capture), [("approx".to_owned(), 1.0)]);
}

#[test]
fn a_lookup_that_finds_nothing_counts_a_miss() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    put(&mut store, &base_snapshot(1), recipe(1));
    // The recipe is unknown.
    assert!(lookup(&mut store, recipe(9), &Tolerance::Exact).unwrap().is_empty());
    // The recipe is known, and an exact lookup rejects its approximate base.
    let root = put_full(
        &mut store,
        &base_snapshot(3),
        recipe(3),
        None,
        Tier::Nvme,
        Reason::Anchor,
        None,
        Some("lossy"),
        10,
    )
    .unwrap();
    let next = recipe_boundary(&recipe(3), &[7], 2);
    put_full(
        &mut store,
        &delta_at(root, 2),
        next,
        Some(recipe(3)),
        Tier::Nvme,
        Reason::Anchor,
        None,
        None,
        10,
    )
    .unwrap();
    assert!(lookup(&mut store, next, &Tolerance::Exact).unwrap().is_empty());
    assert_eq!(lookups(&capture), [("miss".to_owned(), 2.0)]);
}

#[test]
fn a_lookup_that_the_store_refuses_counts_refused() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    put(&mut store, &base_snapshot(1), recipe(4));
    // A second content id under one recipe with equal coverage is a tripwire (spec 6.7).
    assert!(
        put_full(
            &mut store,
            &base_snapshot(2),
            recipe(4),
            None,
            Tier::Nvme,
            Reason::Anchor,
            None,
            None,
            10
        )
        .is_err()
    );
    let error = lookup(&mut store, recipe(4), &Tolerance::Exact).unwrap_err();
    assert!(matches!(error, StoreError::Tripwire { .. }), "{error:?}");
    assert_eq!(lookups(&capture), [("refused".to_owned(), 1.0)]);
}

#[test]
fn evict_counts_the_entries_that_it_removes_by_tier() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let one = put(&mut store, &base_snapshot(1), recipe(1));
    let two = put(&mut store, &base_snapshot(2), recipe(2));
    let victims = store
        .evict(1, Tier::Nvme, size_of(&base_snapshot(1)) as u64 * 2, 10)
        .unwrap();
    assert_eq!(victims.len(), 2, "{one:?} {two:?}");
    assert_eq!(evictions(&capture), [("nvme".to_owned(), "evict".to_owned(), 2.0)]);
}

#[test]
fn a_put_that_makes_room_counts_put_room() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_nvme_budget(dir.path(), 5_000).unwrap();
    let first = put(&mut store, &base_snapshot(1), recipe(25));
    put(&mut store, &base_snapshot(2), recipe(26));
    assert!(!store.entries().contains_key(&first));
    assert_eq!(evictions(&capture), [("nvme".to_owned(), "put_room".to_owned(), 1.0)]);
}

#[test]
fn an_interval_put_over_the_cap_counts_interval_cap() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_limits(dir.path(), u64::MAX, u64::MAX, u64::MAX, 1).unwrap();
    let old = put_copy(&mut store, &base_snapshot(1), recipe(1), 10);
    put_copy(&mut store, &base_snapshot(2), recipe(2), 20);
    assert!(!store.entries().contains_key(&old));
    assert_eq!(
        evictions(&capture),
        [("host".to_owned(), "interval_cap".to_owned(), 1.0)]
    );
}

#[test]
fn remove_counts_interval_replaced() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open_with_limits(dir.path(), u64::MAX, u64::MAX, u64::MAX, 4).unwrap();
    let copy = put_copy(&mut store, &base_snapshot(1), recipe(1), 10);
    assert!(store.remove(&copy, 20).unwrap());
    assert!(!store.remove(&copy, 21).unwrap(), "a second remove finds nothing");
    assert_eq!(
        evictions(&capture),
        [("host".to_owned(), "interval_replaced".to_owned(), 1.0)]
    );
}

#[test]
fn a_restore_that_writes_every_part_counts_done() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(1));
    let mut target = FakeAlways::new();
    let result = store.restore(&id, &mut target, State::Slot(0), &[1u8; 32], 20).unwrap();
    assert_eq!(result, Restored::Done);
    assert_eq!(restores(&capture), [("done".to_owned(), 1.0)]);
    let point = &capture.metric("nagaya.store.restores")[0];
    assert_eq!(point.attributes["engine"], "fake-always");
}

#[test]
fn a_restore_that_the_engine_refuses_counts_refused() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &chain_root(), recipe(1));
    let mut target = Scripted::new(1, WriteResult::Refused);
    let result = store.restore(&id, &mut target, State::Slot(0), &[1u8; 32], 20).unwrap();
    assert!(matches!(result, Restored::Refused { .. }), "{result:?}");
    assert_eq!(restores(&capture), [("refused".to_owned(), 1.0)]);
}

#[test]
fn a_restore_that_fails_after_a_write_counts_failed() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &chain_root(), recipe(1));
    let mut target = Scripted::new(2, WriteResult::Failed);
    let result = store.restore(&id, &mut target, State::Slot(0), &[1u8; 32], 20).unwrap();
    assert!(matches!(result, Restored::Failed { .. }), "{result:?}");
    assert_eq!(restores(&capture), [("failed".to_owned(), 1.0)]);
}

#[test]
fn a_restore_that_the_store_refuses_counts_error() {
    let capture = Capture::start(Vec::new());
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).unwrap();
    let id = put(&mut store, &base_snapshot(1), recipe(1));
    let mut target = FakeAlways::new();
    // The identity digest does not match the entry, so the store quarantines it (spec 6.11).
    let error = store
        .restore(&id, &mut target, State::Slot(0), &[2u8; 32], 20)
        .unwrap_err();
    assert!(matches!(error, StoreError::Quarantined(_)), "{error:?}");
    assert_eq!(restores(&capture), [("error".to_owned(), 1.0)]);
}

/// A snapshot with two parts, so that a restore writes twice.
fn chain_root() -> Snapshot {
    base_snapshot(1)
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
