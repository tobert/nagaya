//! The gates of the calls around a pass on a B70 card (milestone 35 step 12, engine-abi "Inkling 2"): a snapshot of
//! the parts and its restore.
//!
//! - The parts of a run that the count does not reach read as zeros. Thus a slot that held another count before gives
//!   the bytes of a fresh slot, and one recipe gets one content id.
//! - The part `cycles` holds the count. A restore keeps it, so the next pass runs at the count of the snapshot.
//!
//! A test skips when the model is absent. Release the card first, and run alone: `NAGAYA_TEST_CARDS=<pci> cargo test
//! --release -p nagaya-mk-inkling2-bmg --features xe --test xe_calls -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use common::card;
use nagaya_abi::{Engine, PartKind, Segment, State, WriteResult};
use nagaya_mk_inkling2_bmg::Inkling2Engine;

/// `n` token ids from a seeded generator, below 150,000.
fn ids(n: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % 150_000) as u32
        })
        .collect()
}

/// Feed `ids` into `slot` from position `from` at `cycles`, in blocks of 512, and give the logits of the last
/// position.
fn feed(engine: &mut Inkling2Engine, slot: u32, from: u64, ids: &[u32], cycles: Option<u8>) -> Vec<u8> {
    let mut last = Vec::new();
    for (i, chunk) in ids.chunks(512).enumerate() {
        let pos = from + (i * 512) as u64;
        let end = pos + chunk.len() as u64;
        let segment = Segment {
            cycles,
            ..card::segment(slot, pos, chunk, &[end - 1])
        };
        let out = engine
            .pass(&[segment], &[])
            .unwrap_or_else(|e| panic!("a pass at {pos}: {}", e.message));
        last = out.outputs[0].bytes.clone();
    }
    last
}

/// Every part of `slot`, as a snapshot reads it (`nagaya::park::park_snapshot`): a fixed part whole, and a row part
/// from its base to the token count.
fn snapshot(engine: &mut Inkling2Engine, slot: u32) -> Vec<(String, u64, Vec<u8>)> {
    let limits = engine.limits();
    let state = State::Slot(slot);
    let high = engine.slots().unwrap().slot_high(slot);
    let mut out = Vec::new();
    for (id, part) in limits.parts.iter().enumerate() {
        let row: u64 = part.shape.iter().product::<u64>() * 4;
        if part.kind == PartKind::Fixed {
            let mut bytes = vec![0u8; row as usize];
            engine
                .parts()
                .unwrap()
                .part_read(state, id, &mut bytes)
                .unwrap_or_else(|e| panic!("a read of {}: {e:?}", part.name));
            out.push((part.name.clone(), 0, bytes));
            continue;
        }
        let start = engine.coverage().unwrap().coverage(state, id, &[]).start;
        let n = high - start;
        let mut bytes = vec![0u8; (n * row) as usize];
        engine
            .parts()
            .unwrap()
            .cells_read(state, id, start, n, &mut bytes)
            .unwrap_or_else(|e| panic!("a read of {}: {e:?}", part.name));
        out.push((part.name.clone(), start, bytes));
    }
    out
}

/// Write the parts of `snapshot` into `slot`, then set its token count to `n`.
fn restore(engine: &mut Inkling2Engine, slot: u32, snapshot: &[(String, u64, Vec<u8>)], n: u64) {
    let limits = engine.limits();
    let state = State::Slot(slot);
    let parts = engine.parts().unwrap();
    for (id, (name, start, bytes)) in snapshot.iter().enumerate() {
        let part = &limits.parts[id];
        assert_eq!(&part.name, name);
        let result = if part.kind == PartKind::Fixed {
            parts.part_write(state, id, bytes)
        } else {
            let row: u64 = part.shape.iter().product::<u64>() * 4;
            parts.cells_write(state, id, *start, bytes.len() as u64 / row, bytes)
        };
        assert_eq!(result, WriteResult::Ok, "the write of {name}");
    }
    assert_eq!(parts.set_n(state, n), WriteResult::Ok);
}

/// A slot that held 3 cycles and was cleared gives, at 1 cycle, the bytes of a fresh slot. Its restore keeps the
/// count, and a pass after it gives the bits of a pass with no snapshot.
#[test]
fn a_snapshot_at_one_cycle_is_the_snapshot_of_a_fresh_slot_and_keeps_its_count() {
    if common::model_absent(Which::Mini) {
        return;
    }
    let mut engine = card::open(Which::Mini, vec![4096, 4096, 4096], 512);
    let prompt = ids(1300, 7);
    let next = ids(40, 11);
    // Slot 0 holds the reserve rows of 3 cycles, then a clear, then the prompt at 1 cycle.
    feed(&mut engine, 0, 0, &ids(2600, 3), None);
    card::clear(&mut engine, 0);
    feed(&mut engine, 0, 0, &prompt, Some(1));
    // Slot 1 is fresh.
    feed(&mut engine, 1, 0, &prompt, Some(1));
    let (a, b) = (snapshot(&mut engine, 0), snapshot(&mut engine, 1));
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert!(x == y, "part {} differs between a used slot and a fresh one", x.0);
    }
    let (_, _, count) = a.last().unwrap();
    assert_eq!(
        (a.last().unwrap().0.as_str(), count.as_slice()),
        ("cycles", &1u32.to_le_bytes()[..])
    );
    let reserve = a.iter().find(|p| p.0 == "k.x00").unwrap();
    assert!(
        !reserve.2.is_empty() && reserve.2.iter().all(|b| *b == 0),
        "an inactive part reads as zeros"
    );
    // The restore of the snapshot into slot 2 keeps the count: a pass at the default is refused.
    restore(&mut engine, 2, &a, prompt.len() as u64);
    let pos = prompt.len() as u64;
    let refused = engine.pass(&[card::segment(2, pos, &next, &[])], &[]).unwrap_err();
    assert!(
        refused.message.contains("of 1 cycles, and the pass runs 3"),
        "{}",
        refused.message
    );
    let restored = feed(&mut engine, 2, pos, &next, Some(1));
    let fed = feed(&mut engine, 0, pos, &next, Some(1));
    assert!(
        restored == fed,
        "the restored slot gives the bits of the slot that was fed"
    );
}
