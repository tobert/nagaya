//! Batched decode passes of both LFM3 models on a B70 card (milestone 27 step 3, engine-abi "LFM3", "Batched
//! passes"). Each row of a batch gives the bits of a pass of that row alone, for several decode steps, so its K, V and
//! convolution state are right too. A pass that breaks a rule of a batch is refused before it changes a slot.
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-lfm3-bmg --features xe --test xe_batch -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use nagaya_abi::{ControlItem, ControlOp, Engine, PassErrorKind, Tap};

/// The prompts of the rows: lengths on both sides of the window of 256 and of a feed block.
const PROMPTS: [usize; 8] = [7, 33, 300, 1, 129, 256, 517, 64];
const BLOCK: usize = 512;
const STEPS: usize = 4;

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best as u32
}

/// Feed the prompts of `rows` rows into slots 0.. alone and into slots `rows`.. for the batch. Then take `STEPS`
/// greedy decode steps: one pass for each alone slot, and one pass of all the batch slots. Each row of the batch gives
/// the bits of its alone pass.
fn rows_match(which: Which, rows: usize) {
    let mut engine = common::open(which, 2 * rows, BLOCK);
    let mut next = Vec::new();
    for (i, n) in PROMPTS.iter().take(rows).enumerate() {
        let ids = common::ids(*n, 40 + i as u64);
        let alone = common::feed_split(&mut engine, i as u32, 0, &ids, &common::blocks(*n, BLOCK));
        let batch = common::feed_split(&mut engine, (rows + i) as u32, 0, &ids, &common::blocks(*n, BLOCK));
        assert_eq!(
            common::first_other_bits(&alone, &batch),
            None,
            "the twin feeds of row {i}"
        );
        next.push(argmax(&alone));
    }
    for step in 0..STEPS {
        let mut alone = Vec::new();
        for (i, n) in PROMPTS.iter().take(rows).enumerate() {
            let pos = (n + step) as u64;
            let out = engine
                .pass(&[common::segment(i as u32, pos, &[next[i]], &[pos])], &[])
                .unwrap_or_else(|e| panic!("step {step} row {i} alone: {}", e.message));
            alone.push(f32s(&out.outputs[0].bytes));
        }
        let segments: Vec<_> = PROMPTS
            .iter()
            .take(rows)
            .enumerate()
            .map(|(i, n)| {
                let pos = (n + step) as u64;
                common::segment((rows + i) as u32, pos, &[next[i]], &[pos])
            })
            .collect();
        let out = engine
            .pass(&segments, &[])
            .unwrap_or_else(|e| panic!("step {step}, a batch of {rows}: {}", e.message));
        assert_eq!(
            out.outputs.len(),
            rows,
            "one logit row for each segment, in segment order"
        );
        for (i, want) in alone.iter().enumerate() {
            assert_eq!(out.outputs[i].name, "logits");
            let got = f32s(&out.outputs[i].bytes);
            assert_eq!(
                common::first_other_bits(want, &got),
                None,
                "{which:?}: step {step}, row {i} of a batch of {rows} differs from its pass alone"
            );
            next[i] = argmax(want);
        }
    }
    println!("{which:?}: a batch of {rows} gave the bits of each row alone for {STEPS} steps");
}

#[test]
fn each_row_of_a_batch_gives_the_bits_of_its_pass_alone() {
    for which in [Which::Text, Which::Omni] {
        if common::absent(which) {
            continue;
        }
        for rows in [2, 3, 8] {
            rows_match(which, rows);
        }
    }
}

#[test]
fn a_batch_that_breaks_a_rule_is_refused_and_changes_no_slot() {
    if common::absent(Which::Text) {
        return;
    }
    let mut engine = common::open(Which::Text, 3, BLOCK);
    for slot in 0..3u32 {
        common::feed_split(&mut engine, slot, 0, &common::ids(5, u64::from(slot)), &[5]);
    }
    let one = |slot: u32| common::segment(slot, 5, &[17], &[5]);
    let two_tokens = common::segment(2, 5, &[17, 18], &[6]);
    let cases: Vec<(Vec<nagaya_abi::Segment>, Vec<Tap>, &str)> = vec![
        (vec![one(0), one(0)], Vec::new(), "slot 0 twice"),
        (vec![one(0), two_tokens], Vec::new(), "2 tokens"),
        (
            vec![one(0), one(1)],
            vec![Tap {
                name: "residual".to_owned(),
            }],
            "tap",
        ),
        (
            vec![one(0), common::segment(1, 5, &[17], &[])],
            Vec::new(),
            "no logit row",
        ),
        ((0..9).map(|_| one(0)).collect(), Vec::new(), "9 segments"),
    ];
    for (segments, taps, why) in cases {
        let e = engine.pass(&segments, &taps).expect_err(why);
        assert_eq!(e.kind, PassErrorKind::Refused, "{why}: {}", e.message);
        println!("{why}: {}", e.message);
    }
    // A control list too long for one ring is refused, in a batch and in a pass alone.
    let entry = vec![0u8; 24 * 2048 * 4];
    engine
        .control()
        .expect("lfm3 has a table")
        .control_write(0, &entry)
        .unwrap();
    let item = ControlItem {
        index: 0,
        op: ControlOp::Add,
        strength: 1.0,
        layers: None,
    };
    let long = |slot: u32, items: usize| {
        let mut segment = one(slot);
        segment.control = vec![item.clone(); items];
        segment
    };
    for (segments, why) in [
        // Each item adds a task after each of 24 layers: 120 items pass the ring's limit of 2,047 at any row count.
        (vec![long(0, 120), long(1, 120), long(2, 120)], "a batch of 120 items"),
        (vec![long(0, 200)], "a pass of 200 items"),
    ] {
        let e = engine.pass(&segments, &[]).expect_err(why);
        assert_eq!(e.kind, PassErrorKind::Refused, "{why}: {}", e.message);
        assert!(e.message.contains("send fewer control items"), "{why}: {}", e.message);
        println!("{why}: {}", e.message);
    }
    // No slot moved: a pass at position 5 of each slot still runs.
    let out = engine
        .pass(&[one(0), one(1), one(2)], &[])
        .expect("the batch after the refusals");
    assert_eq!(out.outputs.len(), 3);
}

/// A batch has one row for each segment, and the buffers hold `max_block` rows. An engine with a block of 4 takes at
/// most 4 segments.
#[test]
fn a_small_block_takes_at_most_that_many_segments() {
    if common::absent(Which::Text) {
        return;
    }
    let mut engine = common::open(Which::Text, 6, 4);
    assert_eq!(engine.limits().max_segments, 4);
    for slot in 0..6u32 {
        common::feed_split(&mut engine, slot, 0, &common::ids(3, u64::from(slot)), &[3]);
    }
    let segments: Vec<_> = (0..6).map(|slot| common::segment(slot, 3, &[17], &[3])).collect();
    let e = engine.pass(&segments, &[]).expect_err("6 segments with a block of 4");
    assert_eq!(e.kind, PassErrorKind::Refused, "{}", e.message);
    let out = engine.pass(&segments[..4], &[]).expect("4 segments with a block of 4");
    assert_eq!(out.outputs.len(), 4);
}
