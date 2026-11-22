//! The bits of the logits of the Inkling 2 engine on a B70 card, against a fixture of their SHA-256 (milestone 36,
//! "The rules of each item"). A change that keeps the order of each sum keeps each hash. A change that moves bits
//! changes the engine identity (spec 12.1): measure the gap again, then write the new hashes into the fixture.
//!
//! A driver or a compiler update can also move the bits of the card's functions, such as `exp` and `tanh`. Then the
//! canary decides (spec 12.1), and the new hashes go into the fixture with the reason.
//!
//! The test skips when the model or the oracle file is absent. Release the card first, and run alone:
//! `NAGAYA_TEST_CARDS=<pci> cargo test -p nagaya-mk-inkling2-bmg --release --features xe --test xe_bits --
//! --test-threads=1`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use common::card;
use nagaya_abi::Engine;
use nagaya_mk_inkling2_bmg::Inkling2Engine;
use nagaya_mk_inkling2_bmg::spec::{Config, DEFAULT_MAX_BLOCK};
use nagaya_text_inkling2::tokenizer::sha256_of;

/// The fixture: one line for each case, its name and the SHA-256 of the bits of its logits.
const FIXTURE: &str = include_str!("fixtures/logit_bits.txt");
/// A reserve of 4 cycles past the first, so a pass can run 1 to 5 cycles.
const EXTRA_RUNS: usize = 48;
/// The decode steps of each count of cycles.
const STEPS: usize = 4;

fn absent() -> bool {
    common::model_absent(Which::Mini) || common::golden_absent("inkling2-mini.oracle.json")
}

/// The prompt "longer" of the oracle, 637 tokens of prose, repeated to `n` ids.
fn prose(n: usize) -> Vec<u32> {
    let oracle = common::golden_json("inkling2-mini.oracle.json");
    let prompt = oracle["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "longer")
        .expect("the oracle holds the prompt longer")
        .clone();
    let ids: Vec<u32> = prompt["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    ids.iter().copied().cycle().take(n).collect()
}

fn hash(logits: &[f32]) -> String {
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    sha256_of(&bytes)
}

fn argmax(v: &[f32]) -> u32 {
    (0..v.len()).max_by(|a, b| v[*a].total_cmp(&v[*b])).unwrap() as u32
}

/// The logits of the rows `rows` of a pass of `tokens` from `from` at `cycles` cycles.
fn pass(engine: &mut Inkling2Engine, from: usize, tokens: &[u32], rows: &[u64], cycles: u8) -> Vec<Vec<f32>> {
    let mut segment = card::segment(0, from as u64, tokens, rows);
    segment.cycles = Some(cycles);
    let out = engine
        .pass(&[segment], &[])
        .unwrap_or_else(|e| panic!("a pass at {from} of {} tokens: {}", tokens.len(), e.message));
    assert_eq!(
        out.outputs.len(),
        rows.len(),
        "the pass gives one output for each logit row"
    );
    out.outputs.iter().map(|o| card::floats(&o.bytes)).collect()
}

/// The hash of each case at `cycles` cycles: three rows of a block of 512 from position 0, then the greedy decode
/// steps after it.
fn cases(engine: &mut Inkling2Engine, cycles: u8) -> Vec<(String, String)> {
    card::clear(engine, 0);
    let ids = prose(DEFAULT_MAX_BLOCK);
    let rows = [0u64, 255, 511];
    let block = pass(engine, 0, &ids, &rows, cycles);
    let mut out: Vec<(String, String)> = rows
        .iter()
        .zip(&block)
        .map(|(r, l)| (format!("cycles-{cycles}-block-row-{r}"), hash(l)))
        .collect();
    let mut logits = block.last().unwrap().clone();
    for k in 0..STEPS {
        let at = DEFAULT_MAX_BLOCK + k;
        logits = pass(engine, at, &[argmax(&logits)], &[at as u64], cycles).remove(0);
        out.push((format!("cycles-{cycles}-step-{k}"), hash(&logits)));
    }
    out
}

/// Each block row and each decode step gives the hash of the fixture.
#[test]
fn the_logits_keep_the_bits_of_the_fixture() {
    if absent() {
        return;
    }
    let config = Config {
        model_dir: card::model_dir(Which::Mini),
        slots: vec![4_096],
        max_block: DEFAULT_MAX_BLOCK,
        extra_runs: EXTRA_RUNS,
    };
    let cards = nagaya_mk_bmg::test_cards::card();
    let mut engine = Inkling2Engine::open(config, &cards).unwrap_or_else(|e| panic!("the open failed: {e}"));
    let mut got = Vec::new();
    for cycles in [1u8, 3, 5] {
        got.extend(cases(&mut engine, cycles));
    }
    let want: Vec<(String, String)> = FIXTURE
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (name, sha) = l.split_once(' ').expect("a line of the fixture is a name and a hash");
            (name.to_owned(), sha.to_owned())
        })
        .collect();
    for (name, sha) in &got {
        eprintln!("{name} {sha}");
    }
    let moved: Vec<&String> = got.iter().filter(|g| !want.contains(g)).map(|(name, _)| name).collect();
    assert!(
        moved.is_empty() && got.len() == want.len(),
        "{} of {} cases give other bits than tests/fixtures/logit_bits.txt: {moved:?}. A change of bits changes the \
         engine identity (spec 12.1). Measure the gap, then write the lines above into the fixture.",
        moved.len().max(got.len().abs_diff(want.len())),
        got.len()
    );
}
