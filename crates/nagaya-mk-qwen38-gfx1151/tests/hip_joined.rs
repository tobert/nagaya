//! The joined decode of mk on gfx1151 (milestone 30 step 2): a decode row gives the bits that it gives alone, also at
//! deep positions, in a pass of 16 segments, and beside a long prefill.
//!
//! mk declares `segments_exact` (engine-abi D11). The conformance gate `segments_exact_bits` checks it at position 0
//! with segments of 1 and 2 tokens. This gate checks the decode rows that a joined quantum of the process makes: one
//! token for each context, after a prefill of thousands of tokens, past the indexer budget of the attention layers.
//! Run it with `QWEN38_FAST_QMM` unset and with `QWEN38_FAST_QMM=0`, and with `--test-threads=1`.

#![cfg(feature = "hip")]

use nagaya_abi::{Engine, OutputsWanted, Segment, SegmentInputs, State};
use nagaya_mk_qwen38_gfx1151::{Hold, Qwen38Engine, Qwen38Error, hold};

/// The hold of this binary on mk's weights.
static HOLD: std::sync::OnceLock<Hold> = std::sync::OnceLock::new();

/// Open a handle, with the weights held for the binary. Each open makes a new context pool.
fn open() -> Result<Qwen38Engine, Qwen38Error> {
    HOLD.get_or_init(|| hold().expect("hold mk's weights"));
    nagaya_mk_qwen38_gfx1151::open()
}

/// The slots of mk: 12 small slots of 4096 tokens, then 4 large slots of 32 768 (`spec.rs` `CONTEXT_CLASSES`).
const SLOTS: u32 = 16;
/// The first large slot.
const LARGE: u32 = 12;
/// The tokens of one pass (`spec.rs` `MAX_PASS_TOKENS`).
const PASS: usize = 256;
/// The decode steps that the gate compares.
const STEPS: usize = 3;
/// The prefill tokens beside the decode rows in step 1: with the 15 decode rows, the pass holds 255 tokens.
const BESIDE: usize = 240;

/// The token ids of the prompt of `slot`: each slot has its own ids and its own length.
fn prompt(slot: u32) -> Vec<u32> {
    // Small slots take 2300 to 3955 tokens, past the indexer budget of 2048. Large slots take 5000 to 6500.
    let count = if slot < LARGE {
        2300 + 151 * slot as usize
    } else {
        5000 + 500 * (slot - LARGE) as usize
    };
    (0..count)
        .map(|index| ((index * 7919 + 13 + 104_729 * slot as usize) % 248_320) as u32)
        .collect()
}

/// The decode token of `slot` at `step`. The gate feeds fixed ids, so both engines feed the same tokens.
fn decode_token(slot: u32, step: usize) -> u32 {
    ((slot as usize * 31_337 + step * 7_717 + 911) % 248_320) as u32
}

/// The tokens that slot 15 takes beside the decode rows of step 1.
fn beside_tokens() -> Vec<u32> {
    (0..BESIDE)
        .map(|index| ((index * 4_099 + 77) % 248_320) as u32)
        .collect()
}

/// A segment of `tokens` at the slot's next position, with the logit row of its last token when `logits` is true.
fn segment(engine: &Qwen38Engine, slot: u32, tokens: &[u32], logits: bool) -> Segment {
    let state = State::Slot(slot);
    let start = engine.n(state).expect("the slot's token count");
    let mut outputs = OutputsWanted::default();
    if logits {
        outputs.logit_rows = vec![start + tokens.len() as u64 - 1];
    }
    Segment {
        control: Vec::new(),
        state,
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (start..start + tokens.len() as u64).collect(),
        meta: Vec::new(),
        outputs,
    }
}

/// Run one pass, and give the bit patterns of each logit row, in segment order.
fn run(engine: &mut Qwen38Engine, segments: &[Segment], what: &str) -> Vec<Vec<u32>> {
    let out = engine
        .pass(segments, &[])
        .unwrap_or_else(|error| panic!("{what}: the pass failed: {error:?}"));
    out.outputs
        .iter()
        .map(|output| {
            let (chunks, rest) = output.bytes.as_chunks::<4>();
            assert!(rest.is_empty(), "{what}: a logit row of {} bytes", output.bytes.len());
            chunks.iter().map(|chunk| u32::from_le_bytes(*chunk)).collect()
        })
        .collect()
}

/// Feed the prompt of each slot in passes of one segment.
fn prefill(engine: &mut Qwen38Engine) {
    for slot in 0..SLOTS {
        let ids = prompt(slot);
        for chunk in ids.chunks(PASS) {
            let wanted = segment(engine, slot, chunk, false);
            run(engine, &[wanted], &format!("the prefill of slot {slot}"));
        }
        assert_eq!(engine.n(State::Slot(slot)).unwrap(), ids.len() as u64);
    }
}

/// The slots that decode at `step`, and the prefill of slot 15 beside them in step 1.
fn plan(step: usize) -> (Vec<u32>, bool) {
    match step {
        // Step 0: 16 decode rows. Step 1: 15 decode rows beside a prefill of 240. Step 2: 2 decode rows.
        0 => ((0..SLOTS).collect(), false),
        1 => ((0..SLOTS - 1).collect(), true),
        _ => (vec![3, LARGE + 1], false),
    }
}

#[test]
fn a_decode_row_gives_its_bits_alone_and_joined_at_deep_positions() {
    // The reference: each segment alone, in a pass of one segment.
    let mut alone = open().expect("open mk");
    prefill(&mut alone);
    let mut wanted: Vec<Vec<Vec<u32>>> = Vec::new();
    for step in 0..STEPS {
        let (slots, beside) = plan(step);
        let mut rows = Vec::new();
        for &slot in &slots {
            let seg = segment(&alone, slot, &[decode_token(slot, step)], true);
            rows.extend(run(&mut alone, &[seg], &format!("step {step}, slot {slot} alone")));
        }
        if beside {
            let seg = segment(&alone, SLOTS - 1, &beside_tokens(), false);
            run(
                &mut alone,
                &[seg],
                &format!("step {step}, the prefill of slot 15 alone"),
            );
        }
        wanted.push(rows);
    }
    let counts: Vec<u64> = (0..SLOTS).map(|slot| alone.n(State::Slot(slot)).unwrap()).collect();
    // Two pools of 16 contexts take about 24 GB beside the weights, so the reference pool goes first.
    drop(alone);

    // The same feeds on a new pool: each step is one pass of all its segments.
    let mut joined = open().expect("open mk");
    prefill(&mut joined);
    for (step, rows) in wanted.iter().enumerate() {
        let (slots, beside) = plan(step);
        let mut segments: Vec<Segment> = slots
            .iter()
            .map(|&slot| segment(&joined, slot, &[decode_token(slot, step)], true))
            .collect();
        if beside {
            // The prefill goes in the middle of the pass, so the decode rows after it start past row 0 of a tile.
            segments.insert(7, segment(&joined, SLOTS - 1, &beside_tokens(), false));
        }
        let found = run(&mut joined, &segments, &format!("step {step}, joined"));
        assert_eq!(found.len(), rows.len(), "step {step}: the count of logit rows");
        for (index, (left, right)) in rows.iter().zip(&found).enumerate() {
            let slot = slots[index];
            let differ = left.iter().zip(right).filter(|(a, b)| a != b).count();
            assert_eq!(
                differ,
                0,
                "step {step}, slot {slot} at position {}: {differ} of {} logits differ between alone and joined",
                joined.n(State::Slot(slot)).unwrap() - 1,
                left.len()
            );
        }
    }
    for slot in 0..SLOTS {
        assert_eq!(
            counts[slot as usize],
            joined.n(State::Slot(slot)).unwrap(),
            "slot {slot}: the token counts"
        );
    }
}
