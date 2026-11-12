//! The canary of the Qwen3.8 engine in its own process (spec 8.1). It opens the engine, runs the canary once and
//! prints one line. Run it in several processes and compare the lines: each run must give the same hashes.
//!
//! It needs the GPU and `QWEN38_SHARDS`, and `nagaya.service` stopped (spec 3.1). It loads the weights once.
//! Set `QWEN38_FAST_QMM` and `QWEN38_TILES` as the run needs. Run:
//! `cargo test -p nagaya-mk-qwen38-gfx1151 --features hip --test hip_canary -- --nocapture`.

#![cfg(feature = "hip")]

use nagaya_abi::Engine;
use nagaya_mk_qwen38_gfx1151::{canary, open};

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
    let engine = open().expect("open the engine");
    let numbers = engine.identity().numbers;
    let report = engine.canary().expect("run the canary");
    assert_eq!(report.row_hashes.len(), canary::ROWS);
    eprintln!("{}", line(&report, &numbers["fast_qmm"], &numbers["tiles"]));
}
