//! Benchmark: the gap between mk on gfx1151 and the CPU reference (milestone 11.5, spec 8.3).
//!
//! The fixture `tests/fixtures/oracle/qwen38.json` holds the token ids and the top-64 logits of the last position
//! from mk's CPU reference. The bench opens mk once and runs each prompt in its own context, in passes of at most 256
//! tokens. It prints one row for each prompt and exits with status 1 when a tolerance breaks. The tolerances live in
//! `tests/support/oracle.rs`, which a test shares.
//!
//! The bench never closes the engine (spec 3.3). It ends the process with `exit`.

#[path = "../tests/support/oracle.rs"]
mod oracle;

use std::path::Path;
use std::time::Instant;

use nagaya_abi::{Engine, OutputsWanted, Segment, SegmentInputs, State};

/// The most tokens in one pass (mk `kMaxPassTokens`).
const MAX_PASS_TOKENS: usize = 256;

/// The capacity of the small context class, which holds slots 0-11 (`spec.rs` `CONTEXT_CLASSES`).
const SMALL_CONTEXT: usize = 4096;

/// Run the ids in context `slot`, and return the logits of the last position and the wall time.
///
/// A prompt of more than `MAX_PASS_TOKENS` tokens takes several passes, each of at most that many tokens. Only the
/// last pass asks for logits.
fn run(engine: &mut impl Engine, slot: u32, prompt: &oracle::Prompt) -> (Vec<f32>, f64) {
    let length = prompt.ids.len();
    let started = Instant::now();
    let mut logits = Vec::new();
    let mut done = 0;
    while done < length {
        let end = (done + MAX_PASS_TOKENS).min(length);
        let last = end == length;
        let segment = Segment {
            state: State::Slot(slot),
            inputs: SegmentInputs::Tokens(prompt.ids[done..end].to_vec()),
            positions: (done as u64..end as u64).collect(),
            meta: Vec::new(),
            outputs: OutputsWanted {
                logit_rows: if last { vec![(end - 1) as u64] } else { Vec::new() },
            },
        };
        let out = engine
            .pass(std::slice::from_ref(&segment), &[])
            .unwrap_or_else(|error| panic!("{}: the pass at token {done} failed: {error:?}", prompt.name));
        if last {
            let bytes = &out.outputs.first().expect("the pass gives logits").bytes;
            logits = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect();
        }
        done = end;
    }
    (logits, started.elapsed().as_secs_f64())
}

fn main() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/oracle/qwen38.json");
    let fixture = oracle::load(&path);
    if let Some(problem) = oracle::fixture_problem(&fixture, SMALL_CONTEXT) {
        panic!("the fixture is bad: {problem}");
    }
    let path = oracle::QmmPath::from_env();
    let mut engine = nagaya_mk_qwen38_gfx1151::open().expect("open mk");
    println!(
        "oracle: qmm path {}; reference {} layers, transformers {}, reference_sha {}, {} s",
        path.name(),
        fixture.layers,
        fixture.transformers,
        fixture.reference_sha,
        fixture.seconds
    );
    println!(
        "{:<8} {:>6} {:>6} {:>3} {:>3} {:>3} {:>5} {:>10} {:>10} {:>10} {:>8}",
        "prompt", "tokens", "argmax", "t1", "t8", "t64", "t8/16", "max|dlogit|", "KL", "dlogsumexp", "wall_s"
    );
    let mut failed = false;
    for (slot, prompt) in fixture.prompts.iter().enumerate() {
        let (logits, seconds) = run(&mut engine, slot as u32, prompt);
        let gap = oracle::measure(prompt, &logits);
        println!(
            "{:<8} {:>6} {:>6} {:>3} {:>3} {:>3} {:>5} {:>10.6} {:>10.3e} {:>10.6} {:>8.3}",
            prompt.name,
            prompt.ids.len(),
            gap.argmax_agrees,
            gap.overlap_1,
            gap.overlap_8,
            gap.overlap_64,
            gap.top8_inside,
            gap.max_abs_diff,
            gap.kl,
            gap.lse_diff,
            seconds
        );
        for breach in oracle::breaches(&gap, path) {
            failed = true;
            println!("  BREACH {}: {breach}", prompt.name);
        }
    }
    let gate = |limit: Option<f64>| limit.map_or("not a gate".to_owned(), |v| format!("<= {v:e}"));
    println!(
        "gates: margin>{} argmax; ref top-8 in top-{}; max|dlogit| {}; KL {}; |dlogsumexp| {}",
        oracle::MARGIN,
        oracle::TOP8_INSIDE,
        gate(path.max_abs_diff()),
        gate(path.max_kl()),
        gate(path.max_lse_diff())
    );
    std::process::exit(i32::from(failed));
}
