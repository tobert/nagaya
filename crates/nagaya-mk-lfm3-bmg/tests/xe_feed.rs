//! The feed rule, the rings and the bound of a quantum on a B70 card (engine-abi D8, D11, "The first LFM3 binding").
//!
//! - The same split of a feed gives the same bits each time.
//! - Other splits of the same feed: the test prints their gaps. Lane C found that block attention and one-token
//!   attention give other bits, so the engine declares a feed rule.
//! - A ring slot that has wrapped continues with the same bits after a restore of its rows, also of only the last 256.
//! - The last block of a full slot fits `MAX_QUANTUM_MS`.
//! - A long slot gives the bits of a short slot, and its last block fits `MAX_QUANTUM_MS` too (engine-abi "Slot
//!   sizes").
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-lfm3-bmg --features xe --test xe_feed -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use nagaya_abi::{Engine, PartKind, State, WriteResult};
use nagaya_mk_lfm3_bmg::spec::{MAX_QUANTUM_MS, MAX_SLOT_TOKENS, SLOT_TOKENS};

const BLOCK: usize = 512;

#[test]
fn the_same_split_gives_the_same_bits_and_other_splits_stay_close() {
    if common::absent(Which::Text) {
        return;
    }
    let mut engine = common::open(Which::Text, 2, BLOCK);
    let ids = common::ids(608, 1);
    let rule = common::blocks(ids.len(), BLOCK);
    let first = common::feed_split(&mut engine, 0, 0, &ids, &rule);
    let second = common::feed_split(&mut engine, 1, 0, &ids, &rule);
    assert_eq!(
        common::first_other_bits(&first, &second),
        None,
        "two slots with one split differ"
    );
    engine.slots().unwrap().slot_clear(0);
    let again = common::feed_split(&mut engine, 0, 0, &ids, &rule);
    assert_eq!(
        common::first_other_bits(&first, &again),
        None,
        "a feed after a clear differs"
    );
    let mut decode = vec![BLOCK, 80];
    decode.extend(std::iter::repeat_n(1, 16));
    for split in [vec![304, 304], vec![BLOCK, 95, 1], vec![1, 511, 96], decode] {
        engine.slots().unwrap().slot_clear(1);
        let got = common::feed_split(&mut engine, 1, 0, &ids, &split);
        let gap = common::max_abs(&got, &first);
        println!(
            "split {:?}: largest difference {gap:.3e} from the feed rule, bits {}",
            &split[..split.len().min(4)],
            if common::first_other_bits(&got, &first).is_none() {
                "equal"
            } else {
                "differ"
            }
        );
        assert!(gap < common::LOGIT_TOL, "split {split:?}: a logit differs by {gap}");
    }
}

/// The parts of `slot` as the store parks them: each fixed part whole, and each row part from `from(part)` to the
/// token count.
fn park(engine: &mut dyn Engine, slot: u32, from: &dyn Fn(usize, u64) -> u64) -> Vec<(usize, u64, Vec<u8>)> {
    let limits = engine.limits();
    let state = State::Slot(slot);
    let n = engine.slots().unwrap().slot_high(slot);
    let mut out = Vec::new();
    for (id, part) in limits.parts.iter().enumerate() {
        let size = (part.shape.iter().product::<u64>() * 4) as usize;
        if part.kind == PartKind::Fixed {
            let mut bytes = vec![0u8; size];
            engine.parts().unwrap().part_read(state, id, &mut bytes).unwrap();
            out.push((id, 0, bytes));
        } else {
            let base = engine.coverage().unwrap().coverage(state, id, &[]).start;
            let start = from(id, base).max(base);
            let mut bytes = vec![0u8; size * (n - start) as usize];
            engine
                .parts()
                .unwrap()
                .cells_read(state, id, start, n - start, &mut bytes)
                .unwrap();
            out.push((id, start, bytes));
        }
    }
    out
}

fn restore(engine: &mut dyn Engine, slot: u32, n: u64, parked: &[(usize, u64, Vec<u8>)]) {
    let limits = engine.limits();
    let state = State::Slot(slot);
    assert_eq!(engine.slots().unwrap().slot_clear(slot), WriteResult::Ok);
    for (id, start, bytes) in parked {
        let part = &limits.parts[*id];
        let result = if part.kind == PartKind::Fixed {
            engine.parts().unwrap().part_write(state, *id, bytes)
        } else {
            let rows = bytes.len() as u64 / (part.shape.iter().product::<u64>() * 4);
            engine.parts().unwrap().cells_write(state, *id, *start, rows, bytes)
        };
        assert_eq!(
            result,
            WriteResult::Ok,
            "the write of part {}: {:?}",
            part.name,
            engine.last_error()
        );
    }
    assert_eq!(
        engine.parts().unwrap().set_n(state, n),
        WriteResult::Ok,
        "{:?}",
        engine.last_error()
    );
}

#[test]
fn a_wrapped_ring_continues_with_the_same_bits_after_a_restore() {
    if common::absent(Which::Text) {
        return;
    }
    let mut engine = common::open(Which::Text, 2, BLOCK);
    // 2,600 positions wrap each ring of 2,048 rows. The next 300 positions cross the end of the ring again.
    let (head, tail) = (common::ids(2600, 2), common::ids(300, 3));
    common::feed_split(&mut engine, 0, 0, &head, &common::blocks(head.len(), BLOCK));
    let ring = engine
        .limits()
        .parts
        .iter()
        .position(|part| part.kind == PartKind::Ring)
        .unwrap();
    let base = engine.coverage().unwrap().coverage(State::Slot(0), ring, &[]).start;
    assert_eq!(base, 2600 - 2048, "a ring holds its last 2,048 positions");
    let whole = park(&mut engine, 0, &|_, base| base);
    // A restore of the last 256 positions of each ring, and every position of each cells part.
    let rings: Vec<usize> = engine
        .limits()
        .parts
        .iter()
        .enumerate()
        .filter(|(_, part)| part.kind == PartKind::Ring)
        .map(|(id, _)| id)
        .collect();
    let least = park(&mut engine, 0, &|id, base| {
        if rings.contains(&id) { 2600 - 256 } else { base }
    });
    let tail_split = common::blocks(tail.len(), BLOCK);
    let want = common::feed_split(&mut engine, 0, 2600, &tail, &tail_split);
    for (what, parked) in [("whole", &whole), ("last 256", &least)] {
        restore(&mut engine, 1, 2600, parked);
        let got = common::feed_split(&mut engine, 1, 2600, &tail, &tail_split);
        assert_eq!(
            common::first_other_bits(&got, &want),
            None,
            "a restore of the {what} rows continues with other bits"
        );
        let after = park(&mut engine, 1, &|_, _| 2600);
        let kept = park(&mut engine, 0, &|_, _| 2600);
        for ((id, _, a), (_, _, b)) in after.iter().zip(&kept) {
            assert!(
                a == b,
                "part {id} differs after the tail of a restore of the {what} rows"
            );
        }
    }
    // A restore with fewer than the last 256 rows of a ring is refused, and the slot holds no token.
    let short = park(&mut engine, 0, &|id, base| if id == ring { 2900 - 255 } else { base });
    let limits = engine.limits();
    engine.slots().unwrap().slot_clear(1);
    for (id, start, bytes) in &short {
        let part = &limits.parts[*id];
        if part.kind == PartKind::Fixed {
            engine.parts().unwrap().part_write(State::Slot(1), *id, bytes);
        } else {
            let rows = bytes.len() as u64 / (part.shape.iter().product::<u64>() * 4);
            engine
                .parts()
                .unwrap()
                .cells_write(State::Slot(1), *id, *start, rows, bytes);
        }
    }
    assert_eq!(
        engine.parts().unwrap().set_n(State::Slot(1), 2900),
        WriteResult::Refused
    );
    assert_eq!(engine.slots().unwrap().slot_high(1), 0);
}

#[test]
fn the_longest_quantum_fits_its_bound() {
    if common::absent(Which::Text) {
        return;
    }
    let mut engine = common::open(Which::Text, 1, BLOCK);
    let ids = common::ids(SLOT_TOKENS, 4);
    let split = common::blocks(SLOT_TOKENS - 1, BLOCK);
    let mut at = 0usize;
    let mut longest = 0u64;
    for n in &split {
        let out = engine
            .pass(&[common::segment(0, at as u64, &ids[at..at + n], &[])], &[])
            .unwrap();
        longest = longest.max(out.ms);
        at += n;
    }
    let last_block = longest;
    let out = engine
        .pass(&[common::segment(0, at as u64, &ids[at..], &[at as u64])], &[])
        .unwrap();
    println!(
        "the longest block of {BLOCK} in a slot of {SLOT_TOKENS}: {last_block} ms; the last decode step: {} ms",
        out.ms
    );
    assert!(
        last_block.max(out.ms) < u64::from(MAX_QUANTUM_MS),
        "a quantum took {} ms; the engine declares {MAX_QUANTUM_MS}",
        last_block.max(out.ms)
    );
}

#[test]
fn a_long_slot_gives_the_bits_of_a_short_slot_and_its_last_block_fits_the_bound() {
    if common::absent(Which::Text) {
        return;
    }
    let mut engine = common::open_sized(Which::Text, vec![SLOT_TOKENS, MAX_SLOT_TOKENS], BLOCK);
    // 4,096 tokens wrap the rings of the sliding layers in both slots.
    let ids = common::ids(MAX_SLOT_TOKENS, 5);
    let rule = common::blocks(4096, BLOCK);
    let short = common::feed_split(&mut engine, 0, 0, &ids[..4096], &rule);
    let long = common::feed_split(&mut engine, 1, 0, &ids[..4096], &rule);
    assert_eq!(
        common::first_other_bits(&short, &long),
        None,
        "the long slot gives other bits than the short slot"
    );
    let mut at = 4096usize;
    let mut longest = 0u64;
    for n in common::blocks(MAX_SLOT_TOKENS - 1 - at, BLOCK) {
        let out = engine
            .pass(&[common::segment(1, at as u64, &ids[at..at + n], &[])], &[])
            .unwrap();
        longest = longest.max(out.ms);
        at += n;
    }
    let out = engine
        .pass(&[common::segment(1, at as u64, &ids[at..], &[at as u64])], &[])
        .unwrap();
    println!(
        "the longest block of {BLOCK} in a slot of {MAX_SLOT_TOKENS}: {longest} ms; the last decode step: {} ms",
        out.ms
    );
    assert!(
        longest.max(out.ms) < u64::from(MAX_QUANTUM_MS),
        "a quantum took {} ms; the engine declares {MAX_QUANTUM_MS}",
        longest.max(out.ms)
    );
}
