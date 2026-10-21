//! A stop of the process with the mk engine on gfx1151 parks each state that holds tokens (spec 10.3).
//!
//! The test opens mk, so it needs the GPU, `MK_ENGINE_LIB` and `MK_SHARDS`. It loads the weights once (about 25 s).
//! The process holds the only engine handle. The test does not close mk (spec 3.3).
#![cfg(feature = "hip")]

use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use nagaya::admission::Registry;
use nagaya::exit::ExitCode;
use nagaya::inventory::Probe;
use nagaya::lifecycle;
use nagaya::probe::FileProbe;
use nagaya::process::{Quantum, QuantumOutcome};
use nagaya::profile::{PROFILE_FILE, PlacementProfile};
use nagaya::signal::SignalFlag;
use nagaya_abi::{Engine, OutputsWanted, Segment, SegmentInputs, State};
use nagaya_store::Store;

/// Feed `count` token ids to one slot, in passes of at most 256 tokens. Each pass starts at the token count.
fn feed(engine: &mut dyn Engine, slot: u32, count: usize) {
    let ids: Vec<u32> = (0..count).map(|index| ((index * 7919 + 13) % 248_320) as u32).collect();
    let mut done = 0;
    while done < ids.len() {
        let end = (done + 256).min(ids.len());
        let segment = Segment {
            state: State::Slot(slot),
            inputs: SegmentInputs::Tokens(ids[done..end].to_vec()),
            positions: (done as u64..end as u64).collect(),
            meta: Vec::new(),
            outputs: OutputsWanted::default(),
        };
        engine
            .pass(&[segment], &[])
            .unwrap_or_else(|error| panic!("the pass at token {done} of slot {slot} failed: {error:?}"));
        done = end;
    }
}

#[test]
fn a_stop_with_mk_parks_the_states_that_hold_tokens_and_is_clean() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let profile = PlacementProfile::read(&repo.join("deploy/gfx1151host").join(PROFILE_FILE)).unwrap();
    let inventory = FileProbe::new("/").probe().expect("the inventory of this machine");
    let store = tempfile::tempdir().unwrap();
    let mut ready = lifecycle::start(
        profile,
        Registry::with_mk(),
        inventory,
        store.path(),
        None,
        Duration::from_secs(120),
    )
    .expect("the process starts with mk");

    // Slot 0 is small (4096 tokens) and slot 12 is large (32 768). The other 14 slots stay empty.
    let (done, ran) = mpsc::channel();
    let (quantum, outcome) = Quantum::with_outcome(move |engines| {
        let engine = engines[0].as_mut();
        feed(engine, 0, 300);
        feed(engine, 12, 5);
        let slots = engine.slots().expect("mk has the slots capability");
        let _ = done.send((slots.slot_high(0), slots.slot_high(12), slots.slot_high(1)));
    });
    ready.process.submit(0, quantum).unwrap();
    assert_eq!(outcome.recv_timeout(Duration::from_secs(60)), Ok(QuantumOutcome::Ran));
    assert_eq!(
        ran.recv().unwrap(),
        (300, 5, 0),
        "the token counts of slots 0, 12 and 1"
    );

    let report = lifecycle::stop(&mut ready, None, Duration::from_secs(10), &SignalFlag::new());
    println!("{}", report.log_line());
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
    assert!(report.parked.failures.is_empty(), "{report:?}");
    assert_eq!(report.parked.states, 2, "two slots hold tokens: {report:?}");
    assert_eq!(report.parked.entries, 2, "{report:?}");
    // Each state has the fixed parts 0-2 and 5, and the cells parts 3 and 4. `idx_pooled` is derived.
    assert_eq!(report.parked.parts, 12, "{report:?}");
    // Spec 10.3: the process exits after the park. Do not join the group thread: its drop would free mk's contexts.
    std::mem::forget(ready);

    let store = Store::open(store.path()).unwrap();
    let mut ends: Vec<u64> = store
        .entries()
        .values()
        .map(|entry| entry.coverage.iter().map(|part| part.end).max().unwrap_or(0))
        .collect();
    ends.sort_unstable();
    assert_eq!(ends, [5, 300], "the entries hold the rows of each state");
}
