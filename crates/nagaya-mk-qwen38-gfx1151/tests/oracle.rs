//! The oracle's fixture and its metric code, with no GPU (milestone 11.5).

#[path = "support/oracle.rs"]
mod oracle;

use std::path::Path;

use oracle::{Fixture, Prompt, TOP};

/// The most tokens that one mk context takes (`spec.rs` `CONTEXT_CAPACITY`).
const MAX_TOKENS: usize = 256;

fn real() -> Fixture {
    oracle::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/oracle/qwen38.json"))
}

#[test]
fn the_real_fixture_is_sound() {
    let fixture = real();
    assert_eq!(fixture.layers, 48, "the reference ran every layer");
    assert!(fixture.prompts.len() >= 4, "the fixture holds the long prompt too");
    assert_eq!(oracle::fixture_problem(&fixture, MAX_TOKENS), None);
}

#[test]
fn the_fixture_records_the_tool_and_the_reference() {
    let fixture = real();
    assert!(!fixture.transformers.is_empty());
    assert!(!fixture.reference_sha.is_empty());
    assert!(fixture.seconds > 0);
    let long = fixture
        .prompts
        .iter()
        .find(|p| p.name == "long")
        .expect("a long prompt");
    assert!(
        long.ids.len() >= 200,
        "the long prompt is long: {} tokens",
        long.ids.len()
    );
    assert!(long.source.contains("written by"), "the long prompt records its source");
}

#[test]
fn a_fixture_with_a_missing_field_is_refused() {
    let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/oracle/qwen38.json"))
        .expect("read");
    for key in ["transformers", "reference_sha", "seconds", "layers", "vocab"] {
        let broken = text.replacen(&format!("\"{key}\""), "\"other\"", 1);
        let error = oracle::parse(&broken).expect_err(key);
        assert!(error.contains(key), "{key}: {error}");
    }
    let broken = text.replacen("\"source\"", "\"other\"", 1);
    assert!(oracle::parse(&broken).expect_err("source").contains("source"));
}

fn mutated(change: impl Fn(&mut Prompt, &mut Fixture)) -> Option<String> {
    let mut fixture = real();
    let mut prompt = fixture.prompts[0].clone();
    change(&mut prompt, &mut fixture);
    fixture.prompts[0] = prompt;
    oracle::fixture_problem(&fixture, MAX_TOKENS)
}

#[test]
fn a_bad_fixture_is_refused() {
    let unsorted = mutated(|p, _| p.top_logits.swap(0, 1));
    assert!(unsorted.expect("unsorted").contains("not sorted"));
    let above = mutated(|p, _| p.logsumexp = p.top_logits[0] - 1.0);
    assert!(above.expect("above").contains("above the logsumexp"));
    let outside = mutated(|p, f| p.top_ids[5] = f.vocab);
    assert!(outside.expect("outside").contains("outside the vocabulary"));
    let token = mutated(|p, f| p.ids[0] = f.vocab as u32);
    assert!(token.expect("token").contains("outside the vocabulary"));
    let short = mutated(|p, _| {
        p.top_ids.pop();
        p.top_logits.pop();
    });
    assert!(short.expect("short").contains("does not hold"));
    let repeat = mutated(|p, _| p.top_ids[1] = p.top_ids[0]);
    assert!(repeat.expect("repeat").contains("repeats"));
    let long = mutated(|p, _| p.ids = vec![1; MAX_TOKENS + 1]);
    assert!(long.expect("long").contains("the limit"));
}

// Synthetic data for the metric code. The vocabulary is 500 ids. Rank r in the top list has logit 12 - 0.2 r.

const VOCAB: usize = 500;

fn engine_logits() -> Vec<f32> {
    (0..VOCAB)
        .map(|id| {
            if id < TOP {
                12.0 - 0.2 * id as f32
            } else {
                -3.0 - 0.001 * id as f32
            }
        })
        .collect()
}

fn reference_of(logits: &[f32]) -> Prompt {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    order.truncate(TOP);
    let max = f64::from(logits[order[0]]);
    let sum: f64 = logits.iter().map(|v| (f64::from(*v) - max).exp()).sum();
    Prompt {
        name: "synthetic".to_owned(),
        source: "test".to_owned(),
        text: "t".to_owned(),
        ids: vec![1],
        top_logits: order.iter().map(|id| f64::from(logits[*id])).collect(),
        top_ids: order,
        logsumexp: max + sum.ln(),
    }
}

fn breaches_of(logits: &[f32]) -> Vec<String> {
    let prompt = reference_of(&engine_logits());
    oracle::breaches(&oracle::measure(&prompt, logits))
}

#[test]
fn equal_logits_break_no_tolerance() {
    let prompt = reference_of(&engine_logits());
    let gap = oracle::measure(&prompt, &engine_logits());
    assert!(gap.argmax_agrees);
    assert_eq!(
        (gap.overlap_1, gap.overlap_8, gap.overlap_64, gap.top8_inside),
        (1, 8, 64, 8)
    );
    assert_eq!(gap.max_abs_diff, 0.0);
    assert!(gap.kl.abs() < 1e-12 && gap.lse_diff.abs() < 1e-9);
    assert_eq!(oracle::breaches(&gap), Vec::<String>::new());
}

#[test]
fn a_shift_inside_the_tolerance_is_accepted() {
    let mut logits = engine_logits();
    logits[40] += 0.09;
    assert_eq!(breaches_of(&logits), Vec::<String>::new());
}

#[test]
fn one_logit_shifted_past_the_tolerance_is_a_breach() {
    let mut logits = engine_logits();
    logits[40] += 0.11;
    let list = breaches_of(&logits);
    assert_eq!(list.len(), 1, "{list:?}");
    assert!(list[0].contains("max abs diff"), "{list:?}");
}

#[test]
fn a_shift_of_the_top_logit_breaks_the_divergence_and_the_sum() {
    let mut logits = engine_logits();
    logits[0] -= 0.5;
    let list = breaches_of(&logits).join("; ");
    assert!(
        list.contains("KL") && list.contains("logsumexp") && list.contains("max abs diff"),
        "{list}"
    );
}

#[test]
fn a_small_shift_of_the_top_logit_breaks_only_the_divergence() {
    // The shift is inside the diff and sum tolerances, but the top id holds much of the mass.
    let mut logits = engine_logits();
    logits[0] -= 0.09;
    let list = breaches_of(&logits);
    assert_eq!(list.len(), 1, "{list:?}");
    assert!(list[0].contains("KL"), "{list:?}");
}

#[test]
fn two_swapped_ids_are_a_breach() {
    // The engine gives the logit of id 0 to id 1 and the logit of id 1 to id 0. The reference margin is 0.2 here, so
    // the argmax rule does not apply. The diff rule does.
    let mut logits = engine_logits();
    logits.swap(0, 1);
    let list = breaches_of(&logits).join("; ");
    assert!(list.contains("max abs diff"), "{list}");
    assert!(!list.contains("argmax"), "{list}");
}

#[test]
fn a_dropped_argmax_is_a_breach_when_the_margin_is_wide() {
    let mut logits = engine_logits();
    logits[0] = 20.0; // The reference top-1 is id 0 with logit 12.
    let mut prompt = reference_of(&engine_logits());
    prompt.top_logits[0] = 13.0; // A margin of 1.0 over the second.
    prompt.top_logits[1] = 12.0;
    let gap = oracle::measure(&prompt, &[&[0.0f32][..], &logits[1..]].concat());
    assert!(!gap.argmax_agrees);
    let list = oracle::breaches(&gap).join("; ");
    assert!(list.contains("argmax"), "{list}");
}

#[test]
fn a_dropped_argmax_is_no_breach_when_the_margin_is_narrow() {
    let prompt = reference_of(&engine_logits());
    assert!(prompt.top_logits[0] - prompt.top_logits[1] < oracle::MARGIN);
    // Ids 0 and 1 swap in rank: the engine puts id 1 on top by a hair. The diffs stay small.
    let mut logits = engine_logits();
    logits[0] = 11.85;
    logits[1] = 11.95;
    let gap = oracle::measure(&prompt, &logits);
    assert!(!gap.argmax_agrees);
    assert!(!oracle::breaches(&gap).iter().any(|b| b.contains("argmax")));
}

#[test]
fn a_reference_top8_id_outside_the_engine_top16_is_a_breach() {
    let mut logits = engine_logits();
    logits[3] = -50.0;
    let list = breaches_of(&logits).join("; ");
    assert!(list.contains("top-16"), "{list}");
}

#[test]
fn a_non_finite_logit_is_a_breach() {
    let mut logits = engine_logits();
    logits[400] = f32::NAN;
    assert!(breaches_of(&logits).iter().any(|b| b.contains("non-finite")));
}
