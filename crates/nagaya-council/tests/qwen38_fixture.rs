//! The sampler against fixed outputs of `service/sampling.py` of the reference (tools/sampler_fixture.py).
//!
//! Tolerance for probabilities: 1e-12 absolute. Both sides use float64. They differ only in the order of the
//! summation (numpy's pairwise sum, a plain loop here) and in `exp`. Each differs by a few units of 2^-53
//! (1.1e-16) for each term. A vocabulary of 1000 terms stays under 1e-13, so 1e-12 leaves room, and still
//! fails for any change of a filter, the temperature, or the order of the filters.
//! The entropy `h` is a sum of 1000 terms of size up to a few nats, so it gets 1e-9 (as the test of `h` in the reference).

use nagaya_council::sampler::{Settings, pick_with, survivors};
use serde_json::Value;

const PROB_TOLERANCE: f64 = 1e-12;
const ENTROPY_TOLERANCE: f64 = 1e-9;

fn fixture() -> Value {
    let text = include_str!("fixtures/sampler.json");
    serde_json::from_str(text).expect("the fixture is JSON")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

struct Case {
    name: String,
    logits: Vec<f32>,
    settings: Settings,
    ids: Vec<u32>,
    probs: Vec<f64>,
    uniform: f64,
    token: u32,
    p: f64,
    h: f64,
}

fn cases() -> Vec<Case> {
    let doc = fixture();
    let rows = doc["rows"].as_object().unwrap();
    doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let s = &c["settings"];
            let row = floats(&rows[c["row"].as_str().unwrap()]);
            // Each fixture logit is a float32 value, so the narrowing is exact.
            let logits: Vec<f32> = row
                .iter()
                .map(|&x| {
                    assert_eq!(f64::from(x as f32), x, "the fixture logit is not a float32 value");
                    x as f32
                })
                .collect();
            Case {
                name: c["name"].as_str().unwrap().to_string(),
                logits,
                settings: Settings {
                    temperature: s["temperature"].as_f64().unwrap(),
                    top_p: s["top_p"].as_f64().unwrap(),
                    top_k: s["top_k"].as_u64().unwrap(),
                    min_p: s["min_p"].as_f64().unwrap(),
                    seed: s["seed"].as_u64().unwrap(),
                },
                ids: c["survivor_ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect(),
                probs: floats(&c["survivor_probs"]),
                uniform: c["uniform"].as_f64().unwrap(),
                token: c["token"].as_u64().unwrap() as u32,
                p: c["p"].as_f64().unwrap(),
                h: c["h"].as_f64().unwrap(),
            }
        })
        .collect()
}

#[test]
fn the_fixture_has_many_cases_and_varied_survivors() {
    let all = cases();
    assert!(all.len() > 1000, "only {} cases", all.len());
    let sizes: std::collections::BTreeSet<usize> = all.iter().map(|c| c.ids.len()).collect();
    assert!(sizes.len() > 8, "the survivor counts are too alike: {sizes:?}");
}

#[test]
fn survivors_and_probabilities_match_the_reference() {
    for c in cases() {
        let got = survivors(&c.logits, &c.settings).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        assert_eq!(got.ids, c.ids, "{}: survivor ids", c.name);
        assert_eq!(got.probs.len(), c.probs.len(), "{}", c.name);
        for (a, b) in got.probs.iter().zip(&c.probs) {
            assert!(
                (a - b).abs() <= PROB_TOLERANCE,
                "{}: probability {a} against the reference {b}",
                c.name
            );
        }
    }
}

#[test]
fn the_pick_with_mks_uniform_value_gives_mks_token() {
    for c in cases() {
        let got = pick_with(&c.logits, &c.settings, c.uniform).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        assert_eq!(got.token, c.token, "{}: token", c.name);
    }
}

#[test]
fn p_and_h_match_the_reference() {
    for c in cases() {
        let got = pick_with(&c.logits, &c.settings, c.uniform).unwrap();
        assert!(
            (got.p - c.p).abs() <= PROB_TOLERANCE,
            "{}: p {} against the reference {}",
            c.name,
            got.p,
            c.p
        );
        assert!(
            (got.h - c.h).abs() <= ENTROPY_TOLERANCE,
            "{}: h {} against the reference {}",
            c.name,
            got.h,
            c.h
        );
    }
}
