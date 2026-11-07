//! `close` of the Gemma 4 engine on the two B70 cards (engine-abi "`close` in Rust", spec 3.3a): a drop of the engine
//! frees its device memory, so a swap can load another engine on the cards. Run alone, with `--test-threads=1`.
//!
//! The test reads the device memory of this process from DRM fdinfo (`drm-total-vram0` of each `xe` client). The
//! Level Zero driver keeps some device memory for the process after its first init: 400,707,584 bytes on b70host
//! on 2026-10-24, the same after each of three inits. The test measures that first, with a cluster and no weights.
//! Then it loads, runs a pass and drops three times. A drop that leaks fails the first check, and a leak that grows
//! fails a later one.
#![cfg(feature = "xe")]

mod common;

use nagaya_abi::{Engine, OutputsWanted, Segment, SegmentInputs, State};
use nagaya_mk_gemma4_bmg::engine::{Loaded, open};
use nagaya_mk_gemma4_bmg::spec::Config;

/// The device bytes above the driver's own that may stay after a drop. A load holds about 50.9 GB; on 2026-10-24 a
/// drop left 196,608 bytes above the driver's own.
const SLACK: u64 = 1 << 20;

/// The PCI addresses of the `xe` cards, in PCI order.
fn cards() -> Vec<String> {
    let mut found: Vec<String> = std::fs::read_dir("/sys/bus/pci/drivers/xe")
        .expect("the xe driver directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.len() == 12 && name.as_bytes()[4] == b':')
        .collect();
    found.sort();
    found
}

/// The device bytes of this process on every `xe` card: the sum of `drm-total-vram0` over its DRM fds.
fn process_vram() -> u64 {
    let mut total = 0;
    for entry in std::fs::read_dir("/proc/self/fdinfo").expect("the fdinfo of this process") {
        let entry = entry.expect("an fdinfo entry");
        // An fd can close between the listing and the read.
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        if !text
            .lines()
            .any(|line| line.starts_with("drm-driver:") && line.ends_with("xe"))
        {
            continue;
        }
        for line in text.lines() {
            if let Some(value) = line.strip_prefix("drm-total-vram0:") {
                let kib = value
                    .trim()
                    .strip_suffix(" KiB")
                    .unwrap_or_else(|| panic!("`drm-total-vram0` is `{value}`, not a count of KiB"));
                total += kib.trim().parse::<u64>().expect("a count of KiB") * 1024;
            }
        }
    }
    total
}

#[test]
fn a_drop_frees_the_device_memory_of_each_load() {
    if common::absent(&[]) {
        return;
    }
    let config = Config {
        model_dir: common::model_dir(),
        slots: vec![2048],
        max_block: 512,
    };
    drop(common::cluster());
    let driver = process_vram();
    let mut first = None;
    for round in 1..=3 {
        let loaded = Loaded::load(&common::model_dir(), &cards()).expect("a cold load");
        let mut engine = open(&loaded, &config).expect("open an engine");
        drop(loaded);
        let held = process_vram();
        let tokens: Vec<u32> = (0..16u32).map(|i| 1000 + i * 7919).collect();
        let segment = Segment {
            control: Vec::new(),
            state: State::Slot(0),
            inputs: SegmentInputs::Tokens(tokens),
            positions: (0..16).collect(),
            meta: Vec::new(),
            outputs: OutputsWanted { logit_rows: vec![15] },
        };
        engine.pass(&[segment], &[]).expect("a pass");
        drop(engine);
        let left = process_vram();
        eprintln!("round {round}: {held} device bytes with the engine, {left} after its drop, {driver} the driver's");
        assert!(
            left <= driver + SLACK,
            "round {round}: {left} device bytes stay after the drop of the engine, and the driver keeps {driver}; \
             the bound is {SLACK} bytes above the driver's. The engine held {held}"
        );
        // A leak smaller than the slack still grows with each load.
        let first = *first.get_or_insert(left);
        assert_eq!(
            left, first,
            "round {round}: {left} device bytes stay after the drop, and round 1 left {first}; each load leaks"
        );
    }
}

/// The slabs of an export, by stage, as `Loaded::attach` takes them.
fn by_stage(loaded: &Loaded) -> [Vec<(String, std::os::fd::OwnedFd)>; 2] {
    let mut out: [Vec<(String, std::os::fd::OwnedFd)>; 2] = [Vec::new(), Vec::new()];
    for arena in loaded.arena_export().expect("an export of the slabs") {
        out[arena.device].push((arena.part, arena.fd));
    }
    out
}

/// The path of a start with a stored arena (spec 10.2): the engine attaches the slabs that systemd holds. The swap
/// removes those fds from systemd's store (spec 3.3a), so the memory is free when the engine drops and the fds close.
#[test]
fn an_attached_engine_frees_its_memory_when_its_fds_close() {
    if common::absent(&[]) {
        return;
    }
    let config = Config {
        model_dir: common::model_dir(),
        slots: vec![2048],
        max_block: 512,
    };
    drop(common::cluster());
    let driver = process_vram();
    let cold = Loaded::load(&common::model_dir(), &cards()).expect("a cold load");
    let stored = by_stage(&cold);
    drop(cold);
    let attached = Loaded::attach(&common::model_dir(), &cards(), stored).expect("an attach of the stored slabs");
    let engine = open(&attached, &config).expect("open an engine");
    drop(attached);
    let held = process_vram();
    drop(engine);
    let left = process_vram();
    eprintln!("attach: {held} device bytes with the engine, {left} after its drop, {driver} the driver's");
    assert!(
        left <= driver + SLACK,
        "{left} device bytes stay after the drop of an attached engine, and the driver keeps {driver}; the bound is \
         {SLACK} bytes above the driver's. The engine held {held}"
    );
}
