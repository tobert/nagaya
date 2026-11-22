//! The bits of a block of up to 8 tokens of the Inkling 2 engine on a B70 card (milestone 36 item 10, engine-abi "The
//! bits of a decode step"). A block of 2 to 8 tokens gets the bits of the same tokens decoded one at a time.
//!
//! The test skips when the model or the oracle file is absent. Release the card first, and run alone:
//! `NAGAYA_TEST_CARDS=<pci> cargo test -p nagaya-mk-inkling2-bmg --release --features xe --test xe_batch --
//! --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use common::card;
use nagaya_mk_inkling2_bmg::Inkling2Engine;
use nagaya_mk_inkling2_bmg::spec::DEFAULT_MAX_BLOCK;

fn absent() -> bool {
    common::model_absent(Which::Mini) || common::golden_absent("inkling2-mini.oracle.json")
}

/// The prompt "longer" of the oracle, 637 tokens of prose, repeated to `n` ids from the offset `skip`.
fn prose(skip: usize, n: usize) -> Vec<u32> {
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
    ids.iter().copied().cycle().skip(skip).take(n).collect()
}

/// A block of 2 to 8 tokens gets the bits of the same tokens decoded one at a time, at each position of the block
/// (milestone 36 item 10). A block of 9 tokens or more takes other paths of the matmuls and the experts.
#[test]
fn a_block_of_up_to_8_tokens_gets_the_bits_of_its_decode_steps() {
    if absent() {
        return;
    }
    let mut engine = card::open(Which::Mini, vec![2_048], DEFAULT_MAX_BLOCK);
    let ids = prose(0, 700);
    let (prefix, tail) = ids.split_at(640);
    for n in [2usize, 5, 8] {
        let feed_prefix = |engine: &mut Inkling2Engine| {
            card::clear(engine, 0);
            card::feed_split(engine, 0, 0, prefix, &card::blocks(prefix.len(), DEFAULT_MAX_BLOCK));
        };
        feed_prefix(&mut engine);
        let rows: Vec<u64> = (640..640 + n as u64).collect();
        let block = card::pass(&mut engine, 0, 640, &tail[..n], &rows);
        feed_prefix(&mut engine);
        for (t, want) in block.iter().enumerate() {
            let step = card::pass(&mut engine, 0, 640 + t as u64, &tail[t..t + 1], &[640 + t as u64]).remove(0);
            let (count, gap) = card::other_bits(&step, want);
            eprintln!("a block of {n}, position {t}: {count} values of other bits, gap {gap:.3e}");
            assert_eq!(
                count, 0,
                "a block of {n}: the decode step of position {t} differs by {gap:.3e}"
            );
        }
    }
}
