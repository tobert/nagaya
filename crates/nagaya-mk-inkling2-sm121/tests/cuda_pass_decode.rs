//! GPU gate: a decode step equals the same row in a block. The prompt of the chat
//! (17 ids) runs as one block, as 16 rows then 1 row (a decode step), as 9 rows then 8 rows, and row by row. The logits
//! of the last row, every plane of the cache and every conv state must have the same bits in each run, and at each
//! width of the grid (48, 24, 7 and 1). The comparison is of the whole slot, so it covers the plane rows of position 16
//! and the 3 conv rows of each of the 4 streams of each of the 48 states.
//!
//! `INKLING2_MODEL_DIR=<folder> INKLING2_GOLDEN_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p
//!  nagaya-mk-inkling2-sm121 --features cuda --test cuda_pass_decode -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;

use nagaya_mk_inkling2_sm121::pass::step::Pass;
use nagaya_mk_inkling2_sm121::pass::weights::DeviceWeights;
use nagaya_mk_inkling2_sm121::spec;

const SEQ: usize = 1024;

/// One hash for each buffer of the slot: the planes and the conv state of each run.
fn snapshot(pass: &Pass<'_>) -> Vec<(String, [u8; 32])> {
    let mut out = Vec::new();
    for run in &pass.slots[0].runs {
        for (name, buf) in [("k", &run.k_plane), ("v", &run.v_plane), ("conv", &run.conv)] {
            let bytes = buf.read(0, buf.bytes()).unwrap();
            out.push((
                format!("state {} layer {} {name}", run.state, run.layer),
                *blake3::hash(&bytes).as_bytes(),
            ));
        }
    }
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn chat_ids() -> Vec<u32> {
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
        .find(|p| p["name"] == "chat")
        .expect("the golden has the prompt chat")["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

fn run_split(pass: &mut Pass<'_>, ids: &[u32], split: &[usize], grid: u32) -> Vec<f32> {
    pass.slots[0].zero().unwrap();
    pass.grid = grid;
    let mut at = 0;
    let mut out = Vec::new();
    for &n in split {
        out = pass.run(&ids[at..at + n], 1).unwrap_or_else(|e| panic!("{e}"));
        // A block of 17 rows or more runs PROJ in the block form, a block of 16 or fewer in the decode form. The bits of the
        // two forms are the same, and this gate is what shows it.
        let want = if n > 16 { 48 } else { 0 };
        assert_eq!(
            pass.proj_block_form, want,
            "a call of {n} rows: the PROJ tasks in the block form"
        );
        at += n;
    }
    assert_eq!(at, ids.len());
    out
}

#[test]
fn a_decode_step_and_a_block_give_the_same_bits_at_each_grid() {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    let dir = match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(d) if PathBuf::from(&d).join(spec::WEIGHTS_FILE).exists() => PathBuf::from(d),
        _ => {
            assert!(
                !require,
                "INKLING2_MODEL_DIR has no checkpoint and NAGAYA_REQUIRE_MODELS is set"
            );
            eprintln!("cuda_pass_decode: skipped, INKLING2_MODEL_DIR has no checkpoint");
            return;
        }
    };
    let dev = common::gpu::open();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let mut pass = Pass::new(&dev, &weights, SEQ, 512, 3).unwrap_or_else(|e| panic!("{e}"));
    let ids = chat_ids();
    assert_eq!(ids.len(), 17);
    let width = dev.sm_count();
    let reference = run_split(&mut pass, &ids, &[17], width);
    let ref_snap = snapshot(&pass);
    assert!(reference.iter().all(|v| v.is_finite()));
    let mut cases: Vec<(Vec<usize>, u32)> = Vec::new();
    if let Ok(list) = std::env::var("PASS_SPLITS") {
        // For a diagnosis: splits such as "9,8;8,9;5,12", all at the width of the GPU.
        for split in list.split(';') {
            cases.push((split.split(',').map(|n| n.parse().unwrap()).collect(), width));
        }
    } else {
        for grid in [width, 24, 7, 1] {
            cases.push((vec![16, 1], grid));
            cases.push((vec![9, 8], grid));
        }
        cases.push((vec![1; 17], width));
        cases.push((vec![17], 7));
    }
    for (split, grid) in cases {
        let started = std::time::Instant::now();
        let logits = run_split(&mut pass, &ids, &split, grid);
        let snap = snapshot(&pass);
        let same_logits = bits(&logits) == bits(&reference);
        let differing: Vec<&str> = snap
            .iter()
            .zip(&ref_snap)
            .filter(|(a, b)| a.1 != b.1)
            .map(|(a, _)| a.0.as_str())
            .collect();
        eprintln!(
            "split {split:?} at grid {grid}: logits {} | {} of {} buffers differ | {:.1} s",
            if same_logits { "equal" } else { "DIFFER" },
            differing.len(),
            snap.len(),
            started.elapsed().as_secs_f64()
        );
        assert!(
            same_logits,
            "split {split:?} at grid {grid}: the last logits differ from the block of 17 rows"
        );
        assert!(
            differing.is_empty(),
            "split {split:?} at grid {grid}: {} buffers of the slot differ from the block of 17 rows, the first is {}",
            differing.len(),
            differing[0]
        );
    }
}
