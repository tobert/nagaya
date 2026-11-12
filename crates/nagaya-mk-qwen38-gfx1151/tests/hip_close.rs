//! The close of the engine on gfx1151 (engine-abi "close"): the drop of the last handle frees the weights, and the next open
//! loads them again with the same logits. Its own binary, so no hold keeps the weights.
//!
//! It needs the GPU and `QWEN38_SHARDS`, and `nagaya.service` stopped (spec 3.1). It loads the weights three times.
//! Run: `cargo test -p nagaya-mk-qwen38-gfx1151 --features hip --test hip_close -- --nocapture`.

#![cfg(feature = "hip")]

use nagaya_abi::{Engine, OutputsWanted, Segment, SegmentInputs, State};
use nagaya_mk_qwen38_gfx1151::open;

/// The host's MemAvailable in bytes. The arena is a memfd, so its pages are not available while it lives.
fn mem_available() -> u64 {
    let text = std::fs::read_to_string("/proc/meminfo").expect("read /proc/meminfo");
    let line = text
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))
        .expect("MemAvailable in /proc/meminfo");
    let kb: u64 = line.split_whitespace().nth(1).unwrap().parse().unwrap();
    kb * 1024
}

/// The memory of this process in bytes: its anonymous pages and its shared memory, which holds the memfd of the arena.
/// Another process does not change it, as it changes MemAvailable.
fn own_bytes() -> u64 {
    let text = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let field = |name: &str| -> u64 {
        let line = text
            .lines()
            .find(|line| line.starts_with(name))
            .expect("a field of /proc/self/status");
        line.split_whitespace().nth(1).unwrap().parse::<u64>().unwrap() * 1024
    };
    field("RssAnon:") + field("RssShmem:")
}

/// The GTT bytes in use on the amdgpu devices.
fn gtt_used() -> u64 {
    std::fs::read_dir("/sys/class/drm")
        .expect("read /sys/class/drm")
        .filter_map(|entry| std::fs::read_to_string(entry.ok()?.path().join("device/mem_info_gtt_used")).ok())
        .map(|text| text.trim().parse::<u64>().expect("a GTT count"))
        .sum()
}

/// The logits of the last of three tokens on slot 0, as bytes.
fn logits(engine: &mut dyn Engine) -> Vec<u8> {
    let segment = Segment {
        control: Vec::new(),
        state: State::Slot(0),
        inputs: SegmentInputs::Tokens(vec![11, 12, 13]),
        positions: vec![0, 1, 2],
        meta: Vec::new(),
        outputs: OutputsWanted { logit_rows: vec![2] },
    };
    let out = engine.pass(std::slice::from_ref(&segment), &[]).expect("a pass");
    out.outputs[0].bytes.clone()
}

const GIB: f64 = (1u64 << 30) as f64;

#[test]
fn the_last_drop_frees_the_weights_and_the_next_open_gives_the_same_logits() {
    let (before, before_gtt) = (own_bytes(), gtt_used());
    let mut first: Option<Vec<u8>> = None;
    let mut after_drop = Vec::new();
    for round in 0..3 {
        let mut engine = open().expect("open the engine");
        let (loaded, loaded_gtt) = (own_bytes(), gtt_used());
        let bytes = logits(&mut engine);
        drop(engine);
        let (freed, freed_gtt) = (own_bytes(), gtt_used());
        eprintln!(
            "round {round}: the process holds {:.2} GiB before the first open, {:.2} GiB open, {:.2} GiB after the \
             drop; GTT {:.2}, {:.2}, {:.2} GiB; MemAvailable {:.2} GiB",
            before as f64 / GIB,
            loaded as f64 / GIB,
            freed as f64 / GIB,
            before_gtt as f64 / GIB,
            loaded_gtt as f64 / GIB,
            freed_gtt as f64 / GIB,
            mem_available() as f64 / GIB
        );
        // The arena is 61 GiB (spec.rs); the contexts are 11.4 GiB more.
        assert!(
            loaded > freed + (60u64 << 30),
            "round {round}: the drop gave back {:.2} GiB; the weights hold 61 GiB",
            (loaded as f64 - freed as f64) / GIB
        );
        after_drop.push((freed, freed_gtt));
        match &first {
            None => first = Some(bytes),
            Some(want) => assert!(&bytes == want, "round {round}: the logits differ from the first open"),
        }
    }
    // Each close leaves the same residue: the runtime, the pass host and the C++ engine (Amy, 2026-11-01). A close
    // that leaks a part of the weights makes the residue grow from round to round.
    for (what, values) in [
        ("the process", after_drop.iter().map(|x| x.0).collect::<Vec<u64>>()),
        ("the GTT", after_drop.iter().map(|x| x.1).collect()),
    ] {
        let spread = values.iter().max().unwrap() - values.iter().min().unwrap();
        assert!(
            spread < (256u64 << 20),
            "{what} after each drop moved by {:.3} GiB",
            spread as f64 / GIB
        );
    }
    let (last, last_gtt) = *after_drop.last().unwrap();
    eprintln!(
        "the residue after a close: the process {:.2} GiB, the GTT {:.2} GiB more than before the first open",
        (last as f64 - before as f64) / GIB,
        (last_gtt as f64 - before_gtt as f64) / GIB
    );
}
