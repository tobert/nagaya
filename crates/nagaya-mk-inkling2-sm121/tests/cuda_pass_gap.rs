//! GPU report: where the time of a decode step of 1 row goes (milestone 37 step 10, lane G). For each kind of op it prints
//! the mean span of a task, the gap between the end of the task before and the first wait that ended, the tail (first block
//! end to last block end), the mean time that a block waited for FULL (the scoreboard and the producer), the mean busy time
//! of a block, and the time of the weights at 252 GB/s. It asserts only that the numbers exist.
//!
//! `INKLING2_MODEL_DIR=<folder> INKLING2_GOLDEN_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p
//!  nagaya-mk-inkling2-sm121 --features cuda --test cuda_pass_gap -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use nagaya_mk_inkling2_sm121::pass::step::{Pass, TaskTimes};
use nagaya_mk_inkling2_sm121::pass::weights::DeviceWeights;
use nagaya_mk_inkling2_sm121::spec;

const PEAK_GBPS: f64 = 252.0;

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

#[derive(Default)]
struct Sum {
    tasks: usize,
    span: f64,
    gap: f64,
    tail: f64,
    wait: f64,
    busy: f64,
    blocks: usize,
    weights_us: f64,
}

#[test]
fn where_the_time_of_a_decode_step_goes() {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    let dir = match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(d) if PathBuf::from(&d).join(spec::WEIGHTS_FILE).exists() => PathBuf::from(d),
        _ => {
            assert!(
                !require,
                "INKLING2_MODEL_DIR has no checkpoint and NAGAYA_REQUIRE_MODELS is set"
            );
            eprintln!("cuda_pass_gap: skipped, INKLING2_MODEL_DIR has no checkpoint");
            return;
        }
    };
    let dev = common::gpu::open_profile();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let mut pass = Pass::with_profile(&dev, &weights, 1024, 512, 3, true).unwrap_or_else(|e| panic!("{e}"));
    let ids = long_ids();
    pass.slots[0].zero().unwrap();
    pass.run(&ids[0..16], 1).unwrap_or_else(|e| panic!("{e}"));
    for i in 0..8 {
        pass.run(&ids[96 + i..97 + i], 1).unwrap_or_else(|e| panic!("{e}"));
    }
    let tasks = pass.block_times().unwrap();
    let wall = pass.launches[0].wall.as_secs_f64() * 1e3;

    // The mean busy time of each block over the tasks of one kind: is a slow block the same block each time?
    for name in ["PROJ", "OUT_MIX", "UP", "DOWN"] {
        let mut sum = [0.0f64; 48];
        let mut n = 0.0;
        for TaskTimes { tag, items, blocks } in &tasks {
            if tag.name == name && tag.layer.is_some_and(|l| l < spec::NVFP4_LAYERS) && *items as usize >= blocks.len()
            {
                for (i, b) in blocks.iter().enumerate() {
                    sum[i] += (b[2] - b[1]) as f64 / 1e3;
                }
                n += 1.0;
            }
        }
        let mean: Vec<String> = sum.iter().map(|v| format!("{:.0}", v / n)).collect();
        eprintln!(
            "  {name}: mean busy of block 0..47 over {n} tasks, us: {}",
            mean.join(" ")
        );
    }
    let mut order: Vec<String> = Vec::new();
    let mut sums: BTreeMap<String, Sum> = BTreeMap::new();
    let mut prev_end: Option<u64> = None;
    let first = tasks.iter().flat_map(|t| t.blocks.iter().map(|b| b[0])).min().unwrap();
    let last = tasks.iter().flat_map(|t| t.blocks.iter().map(|b| b[2])).max().unwrap();
    let mut first_gap = 0.0;
    let mut shown: Vec<String> = Vec::new();
    for TaskTimes { tag, items, blocks } in &tasks {
        if blocks.is_empty() {
            continue;
        }
        let key = if tag.name == "UP" || tag.name == "DOWN" {
            let mx = tag.layer.is_some_and(|l| l >= spec::NVFP4_LAYERS);
            format!("{}{}", tag.name, if mx { " (MXFP8)" } else { " (NVFP4)" })
        } else {
            tag.name.to_owned()
        };
        if !order.contains(&key) {
            order.push(key.clone());
        }
        if tag.layer == Some(10) && !shown.contains(&key) {
            shown.push(key.clone());
            let mut busy: Vec<u64> = blocks.iter().map(|b| (b[2] - b[1]) / 100).collect();
            busy.sort_unstable();
            let begins: Vec<u64> = blocks
                .iter()
                .map(|b| (b[1] - blocks.iter().map(|c| c[1]).min().unwrap()) / 100)
                .collect();
            eprintln!(
                "  {key}: {items} items, {} blocks, busy of each block (x100 ns, sorted) {busy:?}; start offsets {begins:?}",
                blocks.len()
            );
        }
        let begin = blocks.iter().map(|b| b[1]).min().unwrap();
        let end = blocks.iter().map(|b| b[2]).max().unwrap();
        let min_end = blocks.iter().map(|b| b[2]).min().unwrap();
        let s = sums.entry(key).or_default();
        s.tasks += 1;
        s.span += (end - begin) as f64 / 1e3;
        s.tail += (end - min_end) as f64 / 1e3;
        match prev_end {
            Some(p) => s.gap += begin.saturating_sub(p) as f64 / 1e3,
            None => first_gap = (begin - first) as f64 / 1e3,
        }
        for b in blocks {
            s.wait += (b[1] - b[0]) as f64 / 1e3;
            s.busy += (b[2] - b[1]) as f64 / 1e3;
            s.blocks += 1;
        }
        s.weights_us += tag.weight_bytes as f64 / PEAK_GBPS / 1e3;
        prev_end = Some(end);
    }
    eprintln!("a decode step of 1 row, microseconds, the sum over the tasks of each kind (last of the warm calls):");
    eprintln!(
        "  {:<14} {:>5} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11}",
        "op", "tasks", "span", "gap", "tail", "weights", "wait/block", "busy/block"
    );
    let (mut t_span, mut t_gap, mut t_tail, mut t_w) = (0.0, 0.0, 0.0, 0.0);
    for key in &order {
        let s = &sums[key];
        eprintln!(
            "  {:<14} {:>5} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>11.1} {:>11.1}",
            key,
            s.tasks,
            s.span,
            s.gap,
            s.tail,
            s.weights_us,
            s.wait / s.blocks as f64,
            s.busy / s.blocks as f64
        );
        t_span += s.span;
        t_gap += s.gap;
        t_tail += s.tail;
        t_w += s.weights_us;
    }
    eprintln!(
        "  total: spans {:.3} ms (tails inside: {:.3} ms), gaps {:.3} ms, weights at 252 GB/s {:.3} ms, first block starts {:.1} us after the first timestamp, \
         first to last timestamp {:.3} ms, wall of the launch {:.3} ms",
        t_span / 1e3,
        t_tail / 1e3,
        t_gap / 1e3,
        t_w / 1e3,
        first_gap,
        (last - first) as f64 / 1e6,
        wall
    );
    // The report must agree with `Pass::spans`, the other reader of the same profile: the same tasks and the same spans.
    let spans = pass.spans().unwrap();
    let report: Vec<(u32, u64)> = tasks
        .iter()
        .filter(|t| !t.blocks.is_empty())
        .map(|TaskTimes { items, blocks, .. }| {
            let begin = blocks.iter().map(|b| b[1]).min().unwrap();
            (*items, blocks.iter().map(|b| b[2]).max().unwrap() - begin)
        })
        .collect();
    let known: Vec<(u32, u64)> = spans.iter().map(|s| (s.items, s.ns)).collect();
    assert_eq!(report, known, "block_times and spans read different profiles");
    assert!(t_span > 0.0);
}
