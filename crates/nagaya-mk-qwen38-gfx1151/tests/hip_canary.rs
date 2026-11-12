//! The canary of the Qwen3.8 engine in its own process (spec 8.1). It opens the engine in the measure form, which
//! compares nothing, so it works for a checkpoint and settings that the table does not hold. It runs the canary once
//! and prints one line, and the line of the table to add. Run it in several processes and compare the lines: each run must give the same hashes.
//!
//! It needs the GPU and `QWEN38_SHARDS`, and `nagaya.service` stopped (spec 3.1). It loads the weights once.
//! Set `QWEN38_FAST_QMM` and `QWEN38_TILES` as the run needs. Run:
//! `cargo test -p nagaya-mk-qwen38-gfx1151 --features hip --test hip_canary -- --nocapture`.

#![cfg(feature = "hip")]

use nagaya_abi::Engine;
use nagaya_mk_qwen38_gfx1151::{CanaryCheck, canary, open_with};

/// The line of a report. The main session compares it across processes, so its form is fixed.
fn line(report: &canary::Report, fast_qmm: &str, tiles: &str) -> String {
    format!(
        "canary: hash={} fast_qmm={fast_qmm} tiles={tiles} short_ms={:.0} long_ms={:.0} decode_ms={:.0} \
         total_ms={:.0} row0={}",
        canary::hex(&report.canary_hash),
        report.short_ms,
        report.long_ms,
        report.decode_ms,
        report.total_ms,
        canary::hex(&report.row_hashes[0]),
    )
}

#[test]
fn the_canary_prints_its_report() {
    let engine = open_with(CanaryCheck::Measure).expect("open the engine");
    let identity = engine.identity();
    let numbers = &identity.numbers;
    let report = engine.canary().expect("run the canary");
    assert_eq!(report.row_hashes.len(), canary::ROWS);
    // The canary of the load and the canary of this run are one hash.
    assert_eq!(engine.load_canary().canary_hash, report.canary_hash);
    eprintln!("{}", line(&report, &numbers["fast_qmm"], &numbers["tiles"]));
    let key = canary::Key {
        checkpoint: identity.checkpoint_digest.clone(),
        fast_qmm: numbers["fast_qmm"] == "1",
        tiles: numbers["tiles"] == "1",
    };
    match canary::record(&key) {
        Some(recorded) if recorded == canary::hex(&report.canary_hash) => eprintln!("the table holds this hash"),
        Some(recorded) => eprintln!("the table holds {recorded}, which differs from this hash"),
        None => eprintln!(
            "the table has no line for this key; add: {}",
            canary::table_line(&key, &report.canary_hash)
        ),
    }
}
