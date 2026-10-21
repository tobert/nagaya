//! The logits of the KV layout against a baseline file, bit for bit.
//!
//! mkviolet's `src/golden.rs`, copied with no change in behavior (milestone 15s, step 4).
#![cfg(feature = "xe")]
// `!(err < tol)` fails on NaN too, as mkviolet meant. The other lints wait for the port, as in lib.rs.
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::chunks_exact_to_as_chunks,
    clippy::needless_range_loop
)]

mod common;

use common::*;
use nagaya_mk_gemma4_bmg::gemma4::*;

/// A change to how KV rows are stored must not change one bit of the logits. Records a
/// prefill whose blocks cross row 4096, greedy decode, a rewind of 1032 positions, a snapshot
/// reloaded over a clobbered slot, and decode across row 8192, and compares every logit's bits
/// with kv_layout_baseline.bin in the reference directory. `GEMMA4_KV_BASELINE=write` writes that file
/// instead: write it with the layout being replaced, then test the new one.
#[test]
fn golden_kv_layout_matches_baseline() {
    let write = std::env::var("GEMMA4_KV_BASELINE").as_deref() == Ok("write");
    let baseline: &[&str] = if write { &[] } else { &["kv_layout_baseline.bin"] };
    if common::absent(baseline) {
        return;
    }
    let path = common::golden_dir().join("kv_layout_baseline.bin");
    let cl = common::cluster();
    let engine = Gemma4Engine::load_slots(&cl, &common::model_dir(), &[16384], 512).unwrap();
    let toks =
        |seed: u32, n: usize| -> Vec<u32> { (0..n as u32).map(|i| (1000 + (i + seed) * 7919) % 200000).collect() };
    let mut rows: Vec<Vec<f32>> = Vec::new();
    // Greedy one-token steps from `logits` at position `pos`; returns the last logits.
    let decode = |mut logits: Vec<f32>, pos: usize, steps: usize, rows: &mut Vec<Vec<f32>>| -> Vec<f32> {
        for i in 0..steps {
            logits = engine.forward(&cl, &[top1(&logits) as u32], pos + i).unwrap().0;
            rows.push(logits.clone());
        }
        logits
    };
    let p = toks(0, 6000);
    rows.push(engine.forward(&cl, &p[..300], 0).unwrap().0);
    // Blocks of 512 from 300: the one at 3884..4396 crosses row 4096.
    let l = engine.forward(&cl, &p[300..], 300).unwrap().0;
    rows.push(l.clone());
    decode(l, 6000, 32, &mut rows);
    // Rewind to 5000 and feed other tokens over the rows past it.
    let l = engine.forward(&cl, &toks(5, 1500), 5000).unwrap().0;
    rows.push(l.clone());
    let l = decode(l, 6500, 8, &mut rows);
    let snap = engine.save_kv(&cl, 6508);
    rows.push(engine.forward(&cl, &toks(9, 7000), 0).unwrap().0);
    engine.load_kv(&cl, &snap).unwrap();
    decode(l, 6508, 8, &mut rows);
    // Feed to 8186, then decode across row 8192.
    let l2 = engine.forward(&cl, &toks(13, 1670), 6516).unwrap().0;
    rows.push(l2.clone());
    decode(l2, 8186, 16, &mut rows);
    let bits: Vec<u32> = rows.iter().flatten().map(|v| v.to_bits()).collect();
    println!("{} logit rows, {} values", rows.len(), bits.len());
    if write {
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        std::fs::create_dir_all(common::golden_dir()).unwrap();
        std::fs::write(&path, bytes).unwrap();
        println!("wrote {}", path.display());
        return;
    }
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("{}: {} (write it with GEMMA4_KV_BASELINE=write)", path.display(), e));
    let want: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        want.len(),
        bits.len(),
        "the baseline holds a different number of logits"
    );
    let diff: Vec<usize> = (0..bits.len()).filter(|&i| bits[i] != want[i]).collect();
    if let Some(&first) = diff.first() {
        panic!(
            "{} of {} logits differ; first at row {} token {}: {} vs {}",
            diff.len(),
            bits.len(),
            first / VOCAB,
            first % VOCAB,
            f32::from_bits(bits[first]),
            f32::from_bits(want[first])
        );
    }
}
