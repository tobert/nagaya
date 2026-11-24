//! GPU gate of the latency of the engine by position and by cache format (milestone 37 step 11). A slot is filled to
//! a position with seeded finite rows, with no prefill, through the part calls of the engine. Then the gate times, by
//! the wall clock of the call, a step of 1 row, a call of 16 rows and a block of 512 rows, at 3 cycles and at 5.
//!
//! `INKLING2_MODEL_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-inkling2-sm121 --features
//!  cuda --test cuda_latency -- --test-threads=1 --nocapture`
//!
//! `LATENCY_POSITIONS=1024,32768` limits the positions of a run that iterates.

#![cfg(feature = "cuda")]

mod engine_support;
mod latency_support;
mod latency_table;

use std::time::Duration;

use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_inkling2_sm121::{bound, spec};

use engine_support::{config, model_dir, open, watchdog};
use latency_support::*;

fn positions() -> Vec<usize> {
    std::env::var("LATENCY_POSITIONS").map_or_else(
        |_| POSITIONS.to_vec(),
        |s| s.split(',').map(|x| x.parse().unwrap()).collect(),
    )
}

/// The calls of one position: rows, and the rows that give logits. A step, a call of 16, a block of 128 and a block of
/// 512. The block of 512 gives the logits of its last row, as a prefill does.
const CALLS: [(usize, usize); 4] = [(1, 1), (16, 16), (128, 1), (512, 1)];

/// One measured call.
struct Line {
    positions: usize,
    /// The position of the first row of the call.
    at: usize,
    cycles: u8,
    rows: usize,
    ms: u64,
    bound_ms: Option<u32>,
}

/// Measure each position of a format and give the lines of the table.
fn measure(format: CacheFormat, positions: &[usize], extra: usize, cycles_list: &[u8]) -> Option<Vec<Line>> {
    let dir = model_dir("cuda_latency")?;
    let _guard = watchdog(Duration::from_secs(2400), "the latency gate");
    let top = *positions.iter().max().unwrap();
    let seq = slot_len(top);
    let mut cfg = config(&dir, &[seq], format);
    cfg.extra_runs = extra;
    let mut engine = open(cfg);
    let fill = Fill::new(format, top);
    let mut table = Vec::new();
    // A warm-up call at a short position: the first launch of a process loads the module and the weights.
    fill_slot(&mut engine, &fill, format, seq, extra, 1024, 3);
    timed(&mut engine, 1024, 16, 1, 3);
    for &p in positions {
        for &cycles in cycles_list {
            let fill_ms = fill_slot(&mut engine, &fill, format, seq, extra, p, cycles as usize);
            let mut at = p;
            // A warm-up step at this position, then the calls.
            timed(&mut engine, at, 1, 1, cycles);
            at += 1;
            for (rows, logits) in CALLS {
                let t = timed(&mut engine, at, rows, logits, cycles);
                let call_at = at;
                at += rows;
                let ms = (t.us as f64 / 1000.0).round() as u64;
                eprintln!(
                    "cuda_latency: {format:?} p={p} cycles={cycles} rows={rows}: {:.1} ms, bound {:?} (fill {fill_ms} ms)",
                    t.us as f64 / 1000.0,
                    t.bound_ms
                );
                table.push(Line {
                    positions: p,
                    at: call_at,
                    cycles,
                    rows,
                    ms,
                    bound_ms: t.bound_ms,
                });
            }
        }
    }
    Some(table)
}

/// The rules of each line: the ceiling of its cell, the estimate of the model, and the bound.
fn check(format: CacheFormat, lines: &[Line]) {
    let record = std::env::var_os("LATENCY_RECORD").is_some();
    let mut broken = Vec::new();
    for l in lines {
        let what = format!(
            "{format:?} at {} positions, {} cycles, {} rows: {} ms",
            l.positions, l.cycles, l.rows, l.ms
        );
        let cell = latency_table::CELLS
            .iter()
            .find(|c| c.format == format && c.positions == l.positions && c.cycles == l.cycles && c.rows == l.rows);
        match cell {
            Some(cell) if l.ms > latency_table::ceiling_ms(cell.ms) => {
                broken.push(format!(
                    "{what}, above the ceiling {} ms",
                    latency_table::ceiling_ms(cell.ms)
                ));
            }
            None if !record => broken.push(format!("{what}, and the table has no cell for it")),
            _ => {}
        }
        let estimate = bound::estimate_ms(l.at, l.rows, l.cycles as usize);
        if l.ms > estimate {
            broken.push(format!("{what}, above the model of time, {estimate} ms"));
        }
        // The engine declares the bound of the model, and a call that takes more than the quantum has one.
        let want = bound::pass_bound_ms(l.at, l.rows, l.cycles as usize).map(|b| b.get());
        if l.bound_ms != want {
            broken.push(format!(
                "{what}, and the engine declared {:?}, not {want:?}",
                l.bound_ms
            ));
        }
        if l.ms > u64::from(spec::MAX_QUANTUM_MS) && l.bound_ms.is_none() {
            broken.push(format!("{what}, past the quantum, and the call has no bound"));
        }
        if let Some(b) = l.bound_ms
            && l.ms >= u64::from(b)
        {
            broken.push(format!("{what}, past its bound {b} ms"));
        }
    }
    assert!(
        broken.is_empty(),
        "the latency gate broke these rules:\n{}",
        broken.join("\n")
    );
}

fn run(format: CacheFormat, positions: &[usize], extra: usize, cycles_list: &[u8]) {
    let Some(lines) = measure(format, positions, extra, cycles_list) else {
        return;
    };
    for l in &lines {
        println!("TABLE {format:?} {} {} {} {}", l.positions, l.cycles, l.rows, l.ms);
    }
    check(format, &lines);
}

/// The positions of the gate that are at most `top`.
fn up_to(top: usize) -> Vec<usize> {
    positions().into_iter().filter(|p| *p <= top).collect()
}

#[test]
fn latency_f32() {
    // An engine of 5 cycles at 524,288 positions plans 71.8 GB of the 121 GB, and the page cache of the machine can
    // hold the free memory below that. So the gate splits: 5 cycles up to 409,600 positions, 3 cycles at 524,288.
    run(CacheFormat::F32, &up_to(409_600), EXTRA, &[3, 5]);
    let deep: Vec<usize> = positions().into_iter().filter(|p| *p > 409_600).collect();
    if !deep.is_empty() {
        run(CacheFormat::F32, &deep, EXTRA / 2, &[3]);
    }
}

/// The full run of `f32`: 5 cycles at 524,288 positions. It needs 71.8 GB of free GPU memory.
#[test]
#[ignore = "needs 71.8 GB of free GPU memory"]
fn latency_f32_most_cycles_at_the_deepest_position() {
    run(CacheFormat::F32, &[524_288], EXTRA, &[5]);
}

#[test]
fn latency_bf16() {
    run(CacheFormat::Bf16, &positions(), EXTRA, &[3, 5]);
}

#[test]
fn latency_e4m3() {
    run(CacheFormat::E4m3, &positions(), EXTRA, &[3, 5]);
}
