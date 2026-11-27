//! GPU report: what a block of 1, 2, 3, 4, 5, 8 and 16 rows costs against 1 row, for the checked drafts of D1
//! (docs/gb10-research.md). It asserts nothing about speed: it prints the median launch time of each block size at a short
//! context (16 positions behind the block), at 3 cycles, with the f32 and the e4m3 cache, with the logits of the last row
//! only (a plain decode) and of every row (a check of drafts needs the logit of each row). The ratio to the 1-row step
//! is the `cost(rows)` of the doc.
//!
//! `INKLING2_MODEL_DIR=<folder> INKLING2_GOLDEN_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test --release -p
//!  nagaya-mk-inkling2-sm121 --features cuda --test cuda_block_cost -- --ignored --test-threads=1 --nocapture`
//!
//! A report test: no mutation applies, since nothing is asserted but that each call ran.

#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;

use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_inkling2_sm121::pass::step::Pass;
use nagaya_mk_inkling2_sm121::pass::weights::DeviceWeights;
use nagaya_mk_inkling2_sm121::spec;

const SEQ: usize = 256;
/// The rows of the blocks that the report times.
const ROWS: [usize; 7] = [1, 2, 3, 4, 5, 8, 16];
/// The positions that a slot holds before the timed block: a short context.
const CONTEXT: usize = 16;
/// Calls that do not count, then calls that do; the report takes the median of the counted ones.
const WARM: usize = 2;
const COUNTED: usize = 9;

/// The ids of the prompt `long` of the golden: more than 200 ids to feed the calls.
fn long_ids() -> Vec<u32> {
    let dir = std::env::var_os("INKLING2_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/inkling2-golden")));
    let path = dir.join("inkling2-mini.quant.oracle.json");
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap();
    json["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "long")
        .expect("the golden has the prompt long")["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// The median launch time in ms of a block of `rows` rows after `CONTEXT` positions, with the logits of `logit_rows`.
fn time_block(pass: &mut Pass, ids: &[u32], rows: usize, logit_rows: usize) -> f64 {
    let mut walls = Vec::new();
    for i in 0..WARM + COUNTED {
        pass.slots[0].zero().unwrap();
        pass.run(&ids[..CONTEXT], 1).unwrap_or_else(|e| panic!("{e}"));
        pass.run(&ids[CONTEXT..CONTEXT + rows], logit_rows)
            .unwrap_or_else(|e| panic!("{e}"));
        if i >= WARM {
            walls.push(pass.launches[0].wall.as_secs_f64() * 1e3);
        }
    }
    median(walls)
}

#[test]
#[ignore = "a GPU report; run with --ignored"]
fn the_cost_of_a_block_of_rows_against_one_row() {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    let dir = match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(d) if PathBuf::from(&d).join(spec::WEIGHTS_FILE).exists() => PathBuf::from(d),
        _ => {
            assert!(
                !require,
                "INKLING2_MODEL_DIR has no checkpoint and NAGAYA_REQUIRE_MODELS is set"
            );
            eprintln!("cuda_block_cost: skipped, INKLING2_MODEL_DIR has no checkpoint");
            return;
        }
    };
    let ids = long_ids();
    assert!(ids.len() >= CONTEXT + 16);
    for format in [CacheFormat::F32, CacheFormat::E4m3] {
        let dev = common::gpu::open_cache(format);
        let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
        let mut pass = Pass::with_cache_format(&dev, &weights, SEQ, 16, 3, format).unwrap_or_else(|e| panic!("{e}"));
        eprintln!(
            "cache {format:?}, 3 cycles, context {CONTEXT}, median launch of {COUNTED} calls (ms, ratio to 1 row)"
        );
        eprintln!("  rows | logits of the last row only | logits of every row");
        let base = time_block(&mut pass, &ids, 1, 1);
        for rows in ROWS {
            let last = time_block(&mut pass, &ids, rows, 1);
            let all = time_block(&mut pass, &ids, rows, rows);
            eprintln!(
                "  {rows:>4} | {last:>8.3} ms  x{:>5.2}        | {all:>8.3} ms  x{:>5.2}",
                last / base,
                all / base
            );
        }
    }
}
