//! The rules of the `embed` call on a B70 card (engine-abi "Embedding"): the same ids give the same bits, a call does
//! not depend on the calls before it, a bad input is refused with nothing run, and a drop frees the card for a reopen.
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test --release -p nagaya-mk-embeddinggemma2-bmg --features xe --test xe_embed -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use nagaya_abi::{Embedding, Engine, PassErrorKind};
use nagaya_mk_embeddinggemma2_bmg::spec::{MAX_TOKENS, VOCAB};
use nagaya_mk_embeddinggemma2_bmg::{Config, EmbeddingGemma2Engine};

fn open() -> EmbeddingGemma2Engine {
    let card = common::cards()[1].clone();
    EmbeddingGemma2Engine::open(
        Config {
            model_dir: common::model_dir(),
        },
        &card,
    )
    .expect("open EmbeddingGemma 2 on card 1")
}

/// Ids of `n` tokens: `<bos>`, a run of words, `<eos>`.
fn ids(n: usize, seed: u32) -> Vec<u32> {
    let mut out = vec![2u32];
    out.extend((0..n - 2).map(|i| 1000 + (i as u32 * 7919 + seed * 104_729) % 200_000));
    out.push(1);
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn a_call_gives_the_same_bits_alone_and_after_other_calls() {
    if common::absent() {
        return;
    }
    let mut engine = open();
    let (a, b) = (ids(37, 1), ids(MAX_TOKENS, 2));
    let first = engine.embed(&a).unwrap();
    // A long call fills every buffer; the short call after it must not see its rows.
    let long = engine.embed(&b).unwrap();
    let again = engine.embed(&a).unwrap();
    assert_eq!(bits(&first), bits(&again), "a call depends on the call before it");
    assert_ne!(bits(&first), bits(&long));
    let mut other = open();
    drop(engine);
    assert_eq!(
        bits(&other.embed(&a).unwrap()),
        bits(&first),
        "two opens give other bits"
    );
}

#[test]
fn a_bad_input_is_refused_and_nothing_runs() {
    if common::absent() {
        return;
    }
    let mut engine = open();
    let good = engine.embed(&ids(9, 3)).unwrap();
    for (what, bad) in [
        ("no ids", Vec::new()),
        ("too many ids", ids(MAX_TOKENS + 1, 4)),
        ("an id past the vocabulary", vec![2, VOCAB as u32, 1]),
    ] {
        let error = engine.embed(&bad).expect_err(what);
        assert_eq!(error.kind, PassErrorKind::Refused, "{what}");
        assert!(error.message.starts_with("nothing ran"), "{what}: {}", error.message);
        assert_eq!(engine.last_error(), Some(error.message.clone()));
    }
    assert_eq!(bits(&engine.embed(&ids(9, 3)).unwrap()), bits(&good));
    assert!(engine.pass(&[], &[]).is_err(), "an embedding engine takes no pass");
}

#[test]
fn the_time_of_a_short_and_a_full_input() {
    if common::absent() {
        return;
    }
    let mut engine = open();
    for n in [19usize, 512, 2040, MAX_TOKENS] {
        let input = ids(n, 5);
        engine.embed(&input).unwrap();
        let runs = 5;
        let start = std::time::Instant::now();
        for _ in 0..runs {
            engine.embed(&input).unwrap();
        }
        println!(
            "{n:5} tokens: {:.1} ms per embed",
            start.elapsed().as_secs_f64() * 1000.0 / runs as f64
        );
    }
}

#[test]
fn a_drop_frees_the_card_for_the_next_open() {
    if common::absent() {
        return;
    }
    let input = ids(64, 6);
    let mut want = None;
    for round in 0..4 {
        let mut engine = open();
        let got = bits(&engine.embed(&input).unwrap());
        assert_eq!(want.get_or_insert_with(|| got.clone()), &got, "round {round}");
    }
}
