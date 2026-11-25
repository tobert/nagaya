//! GPU report: the time of each op of the pass, and the tokens a second at 1 row and at 16 rows. It asserts nothing about speed. It prints, for a decode step of 1 row and for a block of 16 rows, one line for each
//! op: the count of tasks, the mean time of a task from its first wait that ended to the last end of its items, the bytes of
//! weights that a task reads at 1 row, the rate against the 252 GB/s that the probe measured, and the time that the design
//! gives. The wall of the call is the launch of the ring, and the rest of the call is the host (the tasks, the plan).
//!
//! `INKLING2_MODEL_DIR=<folder> INKLING2_GOLDEN_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p
//!  nagaya-mk-inkling2-sm121 --features cuda --test cuda_pass_profile -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use nagaya_mk_inkling2_sm121::pass::step::{OpSpan, Pass};
use nagaya_mk_inkling2_sm121::pass::weights::DeviceWeights;
use nagaya_mk_inkling2_sm121::spec;

const PEAK_GBPS: f64 = 252.0;
const SEQ: usize = 1024;

/// The time of an op in the design (the table of the fused tasks), in microseconds, for an NVFP4 layer
/// and an MXFP8 layer. The attention is bound by the cache, and QUANT has no figure.
fn budget_us(name: &str) -> Option<(f64, f64)> {
    Some(match name {
        "EMBED" => (1.0, 1.0),
        "MIX" => (2.0, 2.0),
        "GATE_DOWN" | "GATE_UP" => (7.4, 7.4),
        "GATE_DOWN_e" | "GATE_UP_e" => (5.0, 5.0),
        "PROJ" => (165.0, 165.0),
        "OUT_MIX" => (61.0, 61.0),
        "ROUTE" => (3.5, 3.5),
        "UP" => (100.0, 182.0),
        "DOWN" => (50.0, 91.0),
        "HEAD" => (3840.0, 3840.0),
        _ => return None,
    })
}

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

fn table(title: &str, spans: &[OpSpan], wall: Duration, rate: bool) {
    // The kinds in the order of their first task.
    let mut order: Vec<String> = Vec::new();
    let mut groups: BTreeMap<String, Vec<&OpSpan>> = BTreeMap::new();
    for s in spans {
        let key = if s.tag.name == "UP" || s.tag.name == "DOWN" {
            // An NVFP4 layer and an MXFP8 layer have different bytes and different budgets.
            let mxfp8 = s.tag.layer.is_some_and(|l| l >= spec::NVFP4_LAYERS);
            format!("{}{}", s.tag.name, if mxfp8 { " (MXFP8)" } else { " (NVFP4)" })
        } else {
            s.tag.name.to_owned()
        };
        if !order.contains(&key) {
            order.push(key.clone());
        }
        groups.entry(key).or_default().push(s);
    }
    eprintln!("{title}");
    eprintln!(
        "  {:<14} {:>5} {:>10} {:>10} {:>10} {:>8} {:>8} {:>10}",
        "op", "tasks", "mean us", "total ms", "weights MB", "GB/s", "% of 252", "budget us"
    );
    let mut sum_ns = 0u64;
    for key in &order {
        let g = &groups[key];
        let total: u64 = g.iter().map(|s| s.ns).sum();
        sum_ns += total;
        let mean_us = total as f64 / g.len() as f64 / 1e3;
        let bytes = g[0].tag.weight_bytes as f64;
        let name = g[0].tag.name;
        let budget = budget_us(name).map(|(n, m)| if key.contains("MXFP8") { m } else { n });
        let gbps = if rate && bytes > 0.0 {
            bytes / (mean_us * 1e3)
        } else {
            f64::NAN
        };
        eprintln!(
            "  {:<14} {:>5} {:>10.1} {:>10.3} {:>10.2} {:>8} {:>8} {:>10}",
            key,
            g.len(),
            mean_us,
            total as f64 / 1e6,
            bytes / 1e6,
            if gbps.is_nan() {
                "-".to_owned()
            } else {
                format!("{gbps:.0}")
            },
            if gbps.is_nan() {
                "-".to_owned()
            } else {
                format!("{:.0}%", 100.0 * gbps / PEAK_GBPS)
            },
            budget.map_or("-".to_owned(), |b| format!("{b:.1}")),
        );
    }
    let wall_ms = wall.as_secs_f64() * 1e3;
    eprintln!(
        "  sum of the spans {:.3} ms; wall of the launch {:.3} ms; the rest ({:.3} ms) is the boundaries between tasks",
        sum_ns as f64 / 1e6,
        wall_ms,
        wall_ms - sum_ns as f64 / 1e6
    );
}

#[test]
fn the_time_of_each_op_and_the_tokens_a_second_at_1_and_16_rows() {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    let dir = match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(d) if PathBuf::from(&d).join(spec::WEIGHTS_FILE).exists() => PathBuf::from(d),
        _ => {
            assert!(
                !require,
                "INKLING2_MODEL_DIR has no checkpoint and NAGAYA_REQUIRE_MODELS is set"
            );
            eprintln!("cuda_pass_profile: skipped, INKLING2_MODEL_DIR has no checkpoint");
            return;
        }
    };
    let dev = common::gpu::open_profile();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let mut pass = Pass::with_profile(&dev, &weights, SEQ, 512, 3, true).unwrap_or_else(|e| panic!("{e}"));
    let ids = long_ids();
    assert!(ids.len() >= 200);
    let mean = |v: &[Duration]| v.iter().sum::<Duration>().as_secs_f64() / v.len() as f64;

    // A block of 16 rows, 6 times at the positions 0, 16, ... 80. The first call is a warm-up and does not count.
    pass.slots[0].zero().unwrap();
    let (mut walls, mut calls) = (Vec::new(), Vec::new());
    let mut last_spans = Vec::new();
    for i in 0..6 {
        let started = Instant::now();
        pass.run(&ids[16 * i..16 * (i + 1)], 1)
            .unwrap_or_else(|e| panic!("{e}"));
        let call = started.elapsed();
        if i > 0 {
            walls.push(pass.launches[0].wall);
            calls.push(call);
        }
        last_spans = pass.spans().unwrap();
    }
    table(
        "a block of 16 rows (the last of 5 measured):",
        &last_spans,
        *walls.last().unwrap(),
        false,
    );
    eprintln!(
        "  16 rows: launch {:.3} ms ({:.0} tokens/s), call with the host {:.3} ms ({:.0} tokens/s)",
        mean(&walls) * 1e3,
        16.0 / mean(&walls),
        mean(&calls) * 1e3,
        16.0 / mean(&calls)
    );

    // Decode steps of 1 row after the 96 rows above. The first 3 steps are a warm-up.
    let (mut walls, mut calls) = (Vec::new(), Vec::new());
    let mut last_spans = Vec::new();
    for i in 0..8 {
        let started = Instant::now();
        pass.run(&ids[96 + i..97 + i], 1).unwrap_or_else(|e| panic!("{e}"));
        let call = started.elapsed();
        if i >= 3 {
            walls.push(pass.launches[0].wall);
            calls.push(call);
        }
        last_spans = pass.spans().unwrap();
    }
    table(
        "a decode step of 1 row (the last of 5 measured):",
        &last_spans,
        *walls.last().unwrap(),
        true,
    );
    eprintln!(
        "  1 row: launch {:.3} ms ({:.1} tokens/s), call with the host {:.3} ms ({:.1} tokens/s)",
        mean(&walls) * 1e3,
        1.0 / mean(&walls),
        mean(&calls) * 1e3,
        1.0 / mean(&calls)
    );
    let bytes: usize = last_spans.iter().map(|s| s.tag.weight_bytes).sum();
    eprintln!(
        "  weights of a decode step: {:.2} GB; at 252 GB/s: {:.2} ms",
        bytes as f64 / 1e9,
        bytes as f64 / 252e9 * 1e3
    );
    assert!(last_spans.iter().all(|s| s.ns > 0));
}
