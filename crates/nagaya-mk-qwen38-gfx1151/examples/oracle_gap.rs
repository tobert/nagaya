//! Measure the gap between the CPU reference and mk on gfx1151 (milestone 11.5 spike, spec 8.3).
//!
//! The fixture `tests/fixtures/oracle/qwen38.json` holds, for each prompt, the token ids and the top-64 logits of the
//! last position from the CPU reference. This example runs the same ids through mk and prints, for each prompt:
//!
//! - whether the argmax agrees;
//! - the overlap of the two top-k sets, for k = 1, 8 and 64;
//! - the largest absolute logit difference over the reference top-64;
//! - KL(ref || mk) over the reference top-64, with both sides renormalised over those 64 ids;
//! - the same KL with the reference over the full vocabulary, through the reference logsumexp (a lower bound on
//!   the true gap, because the tail is not seen).
//!
//! The example never closes the engine.

use std::path::Path;

use nagaya_abi::{Engine, OutputsWanted, Segment, SegmentInputs, State};

/// The size of the reference's top list.
const TOP: usize = 64;

/// One prompt of the fixture.
struct Prompt {
    name: String,
    ids: Vec<u32>,
    top_ids: Vec<usize>,
    top_logits: Vec<f64>,
    logsumexp: f64,
}

fn load(path: &Path) -> Vec<Prompt> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let document: serde_json::Value = serde_json::from_str(&text).expect("the fixture is JSON");
    document["prompts"]
        .as_array()
        .expect("prompts")
        .iter()
        .map(|entry| {
            let list = |key: &str| {
                entry[key]
                    .as_array()
                    .unwrap_or_else(|| panic!("{key} is a list"))
                    .clone()
            };
            let prompt = Prompt {
                name: entry["name"].as_str().expect("name").to_owned(),
                ids: list("ids").iter().map(|v| v.as_u64().expect("id") as u32).collect(),
                top_ids: list("top_ids")
                    .iter()
                    .map(|v| v.as_u64().expect("id") as usize)
                    .collect(),
                top_logits: list("top_logits").iter().map(|v| v.as_f64().expect("logit")).collect(),
                logsumexp: entry["logsumexp"].as_f64().expect("logsumexp"),
            };
            assert_eq!(
                prompt.top_ids.len(),
                TOP,
                "{}: the fixture needs {TOP} top ids",
                prompt.name
            );
            prompt
        })
        .collect()
}

/// Run the ids as one pass in a new context, and return the logits of the last position.
fn run(prompt: &Prompt) -> Vec<f32> {
    let mut engine = nagaya_mk_qwen38_gfx1151::open().expect("open mk");
    let length = prompt.ids.len() as u64;
    let segment = Segment {
        state: State::Slot(0),
        inputs: SegmentInputs::Tokens(prompt.ids.clone()),
        positions: (0..length).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted {
            logit_rows: vec![length - 1],
        },
    };
    let out = engine
        .pass(std::slice::from_ref(&segment), &[])
        .unwrap_or_else(|error| panic!("{}: the pass failed: {error:?}", prompt.name));
    let bytes = &out.outputs.first().expect("the pass gives logits").bytes;
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// The ids of the k largest values, largest first.
fn top_k(values: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*b].total_cmp(&values[*a]));
    order.truncate(k);
    order
}

fn overlap(a: &[usize], b: &[usize], k: usize) -> usize {
    a[..k].iter().filter(|id| b[..k].contains(id)).count()
}

/// KL(p || q) for two log-probability lists over the same ids.
fn kl(p_log: &[f64], q_log: &[f64]) -> f64 {
    p_log.iter().zip(q_log).map(|(p, q)| p.exp() * (p - q)).sum()
}

/// Log-softmax of a list, over the list alone.
fn log_softmax(values: &[f64]) -> Vec<f64> {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values.iter().map(|v| (v - max).exp()).sum();
    values.iter().map(|v| v - max - sum.ln()).collect()
}

fn main() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/oracle/qwen38.json");
    for prompt in load(&path) {
        let logits = run(&prompt);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "{}: mk gave a non-finite logit",
            prompt.name
        );
        let mk_top = top_k(&logits, TOP);
        let mk_at: Vec<f64> = prompt.top_ids.iter().map(|id| f64::from(logits[*id])).collect();
        let max_diff = mk_at
            .iter()
            .zip(&prompt.top_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let kl_top = kl(&log_softmax(&prompt.top_logits), &log_softmax(&mk_at));
        // The reference over the full vocabulary, against mk over the full vocabulary, on the reference top-64 only.
        let mk_lse = {
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            max + logits.iter().map(|v| (f64::from(*v) - max).exp()).sum::<f64>().ln()
        };
        let ref_full: Vec<f64> = prompt.top_logits.iter().map(|v| v - prompt.logsumexp).collect();
        let mk_full: Vec<f64> = mk_at.iter().map(|v| v - mk_lse).collect();
        let kl_full: f64 = ref_full.iter().zip(&mk_full).map(|(p, q)| p.exp() * (p - q)).sum();
        println!("prompt {} ({} tokens)", prompt.name, prompt.ids.len());
        println!(
            "  argmax: ref {} mk {} agree {}",
            prompt.top_ids[0],
            mk_top[0],
            prompt.top_ids[0] == mk_top[0]
        );
        println!(
            "  top-k overlap: k=1 {} k=8 {}/8 k=64 {}/64",
            overlap(&prompt.top_ids, &mk_top, 1),
            overlap(&prompt.top_ids, &mk_top, 8),
            overlap(&prompt.top_ids, &mk_top, TOP)
        );
        println!("  max |logit diff| over ref top-64: {max_diff:.6}");
        println!("  KL(ref || mk) over top-64, renormalised: {kl_top:.3e}");
        println!("  KL(ref || mk) over top-64 terms, full-vocab normalisers: {kl_full:.3e}");
        println!("  logsumexp: ref {:.6} mk {mk_lse:.6}", prompt.logsumexp);
    }
}
