//! The tokens that a reply never picks (spec 11.11, "The generation modes"): the sampler drops them before each
//! filter, as the reference sets their logits to minus infinity.

use nagaya_council::sampler::{Settings, pick, pick_except, pick_with, pick_with_except};

fn settings(temperature: f64, top_k: u64, top_p: f64, min_p: f64) -> Settings {
    Settings {
        temperature,
        top_p,
        top_k,
        min_p,
        seed: 7,
    }
}

/// A row in which tokens 2 and 5 lead by far.
fn row() -> Vec<f32> {
    vec![0.0, 0.5, 9.0, 1.0, 0.2, 8.5, 0.1]
}

#[test]
fn greedy_never_picks_a_dropped_token() {
    let s = settings(0.0, 0, 1.0, 0.0);
    assert_eq!(pick(&row(), &s, 3).unwrap().token, 2);
    assert_eq!(pick_except(&row(), &s, 3, &[2]).unwrap().token, 5);
    assert_eq!(pick_except(&row(), &s, 3, &[2, 5]).unwrap().token, 3);
}

#[test]
fn a_draw_never_picks_a_dropped_token_at_any_value() {
    for s in [
        settings(1.0, 0, 1.0, 0.0),
        settings(1.0, 2, 1.0, 0.0),
        settings(0.7, 0, 0.9, 0.0),
        settings(1.0, 0, 1.0, 0.05),
    ] {
        for step in 0..1000 {
            let u = step as f64 / 1000.0;
            let token = pick_with_except(&row(), &s, u, &[2, 5]).unwrap().token;
            assert!(
                token != 2 && token != 5,
                "{s:?} at {u} picked the dropped token {token}"
            );
        }
    }
}

#[test]
fn the_survivors_after_a_drop_are_the_survivors_of_the_row_without_the_token() {
    // A top_k of 2 keeps the two best of the tokens that are left: 5 and 3, close in logit.
    let row = vec![0.0, 0.5, 9.0, 2.0, 0.2, 2.2, 0.1];
    let s = settings(1.0, 2, 1.0, 0.0);
    let mut seen = std::collections::BTreeSet::new();
    for step in 0..1000 {
        seen.insert(pick_with_except(&row, &s, step as f64 / 1000.0, &[2]).unwrap().token);
    }
    assert_eq!(seen.into_iter().collect::<Vec<_>>(), vec![3, 5]);
}

#[test]
fn no_drop_gives_the_pick_of_the_plain_sampler() {
    let s = settings(1.0, 0, 1.0, 0.0);
    for step in 0..100 {
        let u = step as f64 / 100.0;
        assert_eq!(
            pick_with_except(&row(), &s, u, &[]).unwrap(),
            pick_with(&row(), &s, u).unwrap()
        );
    }
}

#[test]
fn the_stats_of_a_pick_stay_those_of_the_whole_row() {
    // The probability and the entropy are of the untempered softmax over the whole vocabulary (gogi "stats").
    let s = settings(0.0, 0, 1.0, 0.0);
    let plain = pick_with(&row(), &settings(0.0, 0, 1.0, 0.0), 0.0).unwrap();
    let dropped = pick_except(&row(), &s, 0, &[2]).unwrap();
    assert_eq!(dropped.h, plain.h);
    assert!(
        dropped.p < 0.5,
        "token 5 has less than half of the mass of the row: {}",
        dropped.p
    );
}

#[test]
fn a_row_with_every_token_dropped_is_an_error() {
    let s = settings(0.0, 0, 1.0, 0.0);
    assert!(pick_except(&[1.0, 2.0], &s, 0, &[0, 1]).is_err());
}
