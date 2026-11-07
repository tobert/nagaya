//! Measure whether the split of tokens into passes changes the bits (spec open question 15.4).
//!
//! For a fixed token sequence, the probe runs the sequence as one pass, as one pass for each token, and as uneven
//! chunks. Each run starts in a new context. The probe then compares every stored part and the logits of the last
//! position as integer bit patterns. It reports the first differing byte, the count of differing elements, the
//! largest logit difference and whether the argmax agrees. The probe never closes the engine.

use std::time::Instant;

use nagaya_abi::{Engine, OutputsWanted, Segment, SegmentInputs, State};

/// The first token id. Ids below it are mostly special or byte tokens.
const FIRST_ID: u32 = 1_000;
/// The number of ids that the sequence draws from.
const ID_SPAN: u32 = 100_000;
/// The number of f32 values in a logit row.
const VOCAB: usize = 248_320;

/// What one run left behind.
struct Run {
    label: String,
    millis: f64,
    /// The bytes of each stored part, in part order.
    parts: Vec<Vec<u8>>,
    /// The logits of the last position.
    logits: Vec<f32>,
}

/// A fixed sequence of real vocabulary ids from a linear congruential generator.
fn sequence(length: usize) -> Vec<u32> {
    let mut state = 12_345u64;
    (0..length)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            FIRST_ID + ((state >> 33) as u32) % ID_SPAN
        })
        .collect()
}

/// Run the tokens in passes of the given sizes in a new context, and read the state.
fn run(label: &str, tokens: &[u32], sizes: &[usize]) -> Run {
    assert_eq!(
        sizes.iter().sum::<usize>(),
        tokens.len(),
        "the chunk sizes must cover the tokens"
    );
    let mut engine = nagaya_mk_qwen38_gfx1151::open().expect("open mk");
    let state = State::Slot(0);
    let started = Instant::now();
    let mut start = 0usize;
    let mut logits = Vec::new();
    for (index, size) in sizes.iter().enumerate() {
        let last = index + 1 == sizes.len();
        let chunk = &tokens[start..start + size];
        let segment = Segment {
            control: Vec::new(),
            state,
            inputs: SegmentInputs::Tokens(chunk.to_vec()),
            positions: (start as u64..(start + size) as u64).collect(),
            meta: Vec::new(),
            outputs: OutputsWanted {
                logit_rows: if last {
                    vec![(start + size) as u64 - 1]
                } else {
                    Vec::new()
                },
            },
        };
        let out = engine
            .pass(std::slice::from_ref(&segment), &[])
            .unwrap_or_else(|error| panic!("{label}: pass at {start} failed: {error:?}"));
        if last {
            let bytes = &out.outputs.first().expect("the last pass gives logits").bytes;
            logits = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect();
        }
        start += size;
    }
    let millis = started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(engine.n(state).expect("n"), tokens.len() as u64);
    assert_eq!(logits.len(), VOCAB);
    let limits = engine.limits();
    let mut parts = Vec::new();
    for (part, spec) in limits.parts.iter().enumerate() {
        if !spec.is_stored() {
            parts.push(Vec::new());
            continue;
        }
        let bytes = engine.part_bytes(part as u32, tokens.len() as u32).expect("part bytes") as usize;
        let mut out = vec![0u8; bytes];
        engine
            .parts()
            .expect("parts")
            .part_read(state, part, &mut out)
            .expect("part_read");
        parts.push(out);
    }
    let names: Vec<String> = limits.parts.iter().map(|spec| spec.name.clone()).collect();
    println!("ran {label}: {} passes, {millis:.1} ms, parts {names:?}", sizes.len());
    // Dropping the engine frees its contexts. It does not close the engine.
    drop(engine);
    Run {
        label: label.to_owned(),
        millis,
        parts,
        logits,
    }
}

/// The element size in bytes. Every stored part of mk is f32 or u32.
fn element_size(_part: usize) -> usize {
    4
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// Compare two runs and print one block.
fn compare(names: &[String], left: &Run, right: &Run) {
    println!("--- {} vs {}", left.label, right.label);
    let mut all_equal = true;
    for (part, name) in names.iter().enumerate() {
        let (a, b) = (&left.parts[part], &right.parts[part]);
        if a.is_empty() && b.is_empty() {
            println!("  part {part} {name}: derived, skipped");
            continue;
        }
        if a == b {
            println!("  part {part} {name}: equal ({} bytes)", a.len());
            continue;
        }
        all_equal = false;
        let width = element_size(part);
        let first = a.iter().zip(b).position(|(x, y)| x != y);
        let differing = a.chunks(width).zip(b.chunks(width)).filter(|(x, y)| x != y).count();
        println!(
            "  part {part} {name}: DIFFERENT, {} bytes, first differing byte {first:?}, {differing} of {} elements differ",
            a.len(),
            a.len() / width,
        );
    }
    let (a, b) = (&left.logits, &right.logits);
    let bits_differ = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let max_abs = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let (arg_a, arg_b) = (argmax(a), argmax(b));
    if bits_differ == 0 {
        println!("  logits: equal ({} values), argmax {arg_a}", a.len());
    } else {
        all_equal = false;
        println!(
            "  logits: DIFFERENT, {bits_differ} of {} values differ, max abs diff {max_abs:e}, argmax {arg_a} vs {arg_b} ({})",
            a.len(),
            if arg_a == arg_b { "agrees" } else { "DISAGREES" },
        );
    }
    println!("  verdict: {}", if all_equal { "EQUAL" } else { "NOT EQUAL" });
}

fn main() {
    let probe = nagaya_mk_qwen38_gfx1151::open().expect("open mk");
    let names: Vec<String> = probe.limits().parts.iter().map(|spec| spec.name.clone()).collect();
    drop(probe);
    for (length, chunks) in [(64usize, vec![17usize, 47]), (200, vec![17, 47, 136])] {
        println!("=== N = {length}");
        let tokens = sequence(length);
        let whole = run(&format!("N{length} one pass"), &tokens, &[length]);
        let single = run(&format!("N{length} one token each"), &tokens, &vec![1; length]);
        let chunked = run(&format!("N{length} chunks {chunks:?}"), &tokens, &chunks);
        // The same split twice shows whether a run repeats its own bits.
        let again = run(&format!("N{length} one pass again"), &tokens, &[length]);
        compare(&names, &whole, &again);
        compare(&names, &whole, &single);
        compare(&names, &whole, &chunked);
        compare(&names, &single, &chunked);
        println!(
            "timings ms: one pass {:.1}, one token each {:.1}, chunks {:.1}, one pass again {:.1}",
            whole.millis, single.millis, chunked.millis, again.millis
        );
    }
}
