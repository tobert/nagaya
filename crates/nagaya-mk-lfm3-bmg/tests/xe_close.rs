//! `close` of the LFM3 engine (engine-abi "`close` in Rust", spec 3.3a): a drop frees its device memory.
//!
//! The test reads the device memory of this process from DRM fdinfo (`drm-total-vram0` of each `xe` client), as the
//! close test of the speech engine does. The driver keeps some device memory for the process after its first init and
//! its first launch, so the test measures that first. Then it opens, runs a pass and drops three times.
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-lfm3-bmg --features xe --test xe_close -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use nagaya_mk_bmg::level_zero::L0Cluster;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::{KERNEL_ENTRY, LFM3_KERNEL};

/// The device bytes above the driver's own that may stay after a drop.
const SLACK: u64 = 1 << 20;

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
fn a_drop_frees_the_device_memory_of_each_open() {
    if common::absent(Which::Text) {
        return;
    }
    let addresses = vec![common::cards()[0].clone()];
    let cluster = L0Cluster::init_at(LFM3_KERNEL, KERNEL_ENTRY, &addresses).expect("open card 0");
    let card = &cluster.cards[0];
    let mut ring = Ring::new(card, 1);
    ring.set(&[]);
    // SAFETY: the ring holds only its halt.
    unsafe { ring.run(card) };
    ring.free(card);
    drop(cluster);
    let driver = process_vram();
    let mut first = None;
    for round in 1..=3 {
        let start = std::time::Instant::now();
        let mut engine = common::open(Which::Text, 1, 512);
        let open_ms = start.elapsed().as_millis();
        let ids = common::ids(40, round);
        let logits = common::feed_split(&mut engine, 0, 0, &ids, &[40]);
        assert_eq!(logits.len(), engine.vocab());
        let loaded = process_vram();
        let start = std::time::Instant::now();
        drop(engine);
        let drop_ms = start.elapsed().as_millis();
        let left = process_vram();
        println!(
            "round {round}: open {open_ms} ms, {loaded} bytes loaded, drop {drop_ms} ms, {left} bytes left, driver \
             {driver}"
        );
        assert!(loaded > driver + (3 << 30), "the open holds {loaded} bytes");
        assert!(
            left <= driver + SLACK,
            "round {round}: {left} bytes stay after the drop"
        );
        assert_eq!(*first.get_or_insert(left), left, "round {round}: the leftover grows");
    }
}
