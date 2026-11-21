//! The arena of the Gemma 4 engine on the two B70 cards (spec 10.2): the export of 31 DMA-BUF slabs, an attach
//! from them with the spot check, and the bits after the attach. Run alone, with `--test-threads=1`.
//!
//! The attach imports the slabs of the cold load in this process, so the weights take their device memory once.
#![cfg(feature = "xe")]

mod common;

use std::os::fd::OwnedFd;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use nagaya_abi::{Engine, OutputsWanted, Segment, SegmentInputs, State};
use nagaya_mk_gemma4_bmg::engine::{LoadError, Loaded, open};
use nagaya_mk_gemma4_bmg::spec::{Config, arena_parts};
use sha2::{Digest, Sha256};

/// The two cards of the test: the first two cards of `NAGAYA_TEST_CARDS`, stage 0 on the first.
fn cards() -> Vec<String> {
    nagaya_mk_bmg::test_cards::cards(2)
}

/// The cold load that the tests share, and its time in ms. The weights of one load fill most of both cards.
static COLD: OnceLock<(Arc<Loaded>, u128)> = OnceLock::new();

fn cold() -> &'static (Arc<Loaded>, u128) {
    COLD.get_or_init(|| {
        let started = Instant::now();
        let loaded = Loaded::load(&common::model_dir(), &cards()).expect("a cold load");
        (loaded, started.elapsed().as_millis())
    })
}

fn config() -> Config {
    Config {
        model_dir: common::model_dir(),
        slots: vec![2048],
        max_block: 512,
    }
}

fn plain(from: u64, tokens: &[u32]) -> Segment {
    let to = from + tokens.len() as u64;
    Segment {
        control: Vec::new(),
        state: State::Slot(0),
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (from..to).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted {
            logit_rows: vec![to - 1],
        },
        layer_loop: None,
        cycles: None,
    }
}

/// The SHA-256 of the logits of a prefill of 600 tokens in two blocks and four decode steps.
fn logit_hash(loaded: &Arc<Loaded>) -> String {
    let mut engine = open(loaded, &config()).expect("open an engine");
    let prompt: Vec<u32> = (0..600u32).map(|i| (1000 + i * 7919) % 200_000).collect();
    let mut plan = vec![plain(0, &prompt[..512]), plain(512, &prompt[512..])];
    plan.extend((0..4u32).map(|i| plain(600 + u64::from(i), &[5000 + i])));
    let mut hasher = Sha256::new();
    for segment in &plan {
        let out = engine.pass(std::slice::from_ref(segment), &[]).expect("a pass");
        for output in out.outputs {
            hasher.update(&output.bytes);
        }
    }
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// The slabs of an export, by stage.
fn by_stage(loaded: &Arc<Loaded>) -> [Vec<(String, OwnedFd)>; 2] {
    let mut out: [Vec<(String, OwnedFd)>; 2] = [Vec::new(), Vec::new()];
    for fd in loaded.arena_export().expect("the export") {
        out[fd.device].push((fd.part, fd.fd));
    }
    out
}

/// Falsified by a head that stays out of the export, and by an attach that skips the spot check.
#[test]
fn an_attach_from_the_exported_slabs_gives_the_bits_of_the_cold_load() {
    if common::absent(&[]) {
        return;
    }
    let (cold, cold_ms) = cold();
    let mut engine = open(cold, &config()).expect("open an engine");
    let exported = engine
        .arena()
        .expect("the arena capability")
        .arena_export()
        .expect("the export");
    let mut names: Vec<(usize, String)> = exported.iter().map(|fd| (fd.device, fd.part.clone())).collect();
    names.sort();
    let mut expected: Vec<(usize, String)> = arena_parts().into_iter().map(|part| (part.device, part.part)).collect();
    expected.sort();
    assert_eq!(names, expected, "the export gives the arena parts");
    drop(exported);
    drop(engine);
    let cold_hash = logit_hash(cold);

    let started = Instant::now();
    let hot = Loaded::attach(&common::model_dir(), &cards(), by_stage(cold)).expect("an attach");
    let hot_ms = started.elapsed().as_millis();
    assert_eq!(
        logit_hash(&hot),
        cold_hash,
        "the logits after the attach differ from the cold load"
    );
    eprintln!("Loaded: cold {cold_ms} ms, attach {hot_ms} ms");
}

#[test]
fn an_attach_from_slabs_under_the_wrong_names_or_with_a_slab_missing_is_a_mismatch() {
    if common::absent(&[]) {
        return;
    }
    let (cold, _) = cold();
    let mut swapped = by_stage(cold);
    let first = swapped[0].iter().position(|(part, _)| part == "w00").unwrap();
    let second = swapped[0].iter().position(|(part, _)| part == "w01").unwrap();
    swapped[0][first].0 = "w01".to_owned();
    swapped[0][second].0 = "w00".to_owned();
    match Loaded::attach(&common::model_dir(), &cards(), swapped) {
        Err(LoadError::Mismatch(why)) => assert!(why.contains("differs"), "{why}"),
        Err(LoadError::Other(why)) => panic!("an error that is not a mismatch: {why}"),
        Ok(_) => panic!("slabs under the wrong names attached"),
    }
    let mut missing = by_stage(cold);
    missing[1].retain(|(part, _)| part != "head");
    match Loaded::attach(&common::model_dir(), &cards(), missing) {
        Err(LoadError::Mismatch(why)) => assert!(why.contains("head"), "{why}"),
        Err(LoadError::Other(why)) => panic!("an error that is not a mismatch: {why}"),
        Ok(_) => panic!("an arena with no head attached"),
    }
}
