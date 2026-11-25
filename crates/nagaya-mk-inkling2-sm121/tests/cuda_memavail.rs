//! GPU gate of the memory rule of the open. On a GPU with unified memory the page cache holds
//! the checkpoint after a read, and `cuMemGetInfo` leaves it out of the free bytes. The kernel reclaims it for a GPU
//! allocation (a probe of 2026-11-24). The gate reads the checkpoint, and then it checks two facts:
//! - the available bytes of the rule are at least the free bytes of the driver, and
//! - an open of a plan that is larger than the free bytes of the driver, and smaller than the available bytes less the
//!   floor, succeeds.
//!
//! The gate writes a filler file of up to 160 GiB to fill the page cache, so it runs only with `--ignored`:
//! `INKLING2_MODEL_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-inkling2-sm121 --features cuda
//!  --test cuda_memavail -- --ignored --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod engine_support;

use std::fs::File;
use std::io;
use std::time::Duration;

use nagaya_abi::Tier;
use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::memory::Count;
use nagaya_mk_cuda::{INKLING2_CUBIN, test_cards};
use nagaya_mk_inkling2_sm121::health::MIN_FREE_AFTER_BYTES;
use nagaya_mk_inkling2_sm121::spec::{self, WEIGHTS_FILE};

use engine_support::{config, model_dir, open, watchdog};

/// The plan must pass the free bytes of the driver by this much, and the available bytes less the floor must pass the
/// plan by this much. Other work moves both counts by some hundreds of megabytes.
const MARGIN: u64 = 1 << 30;
/// The filler grows until the driver reports fewer free bytes than this.
const FILL_TO: usize = 40 << 30;

fn planned_bytes(dir: &std::path::Path, slots: &[usize]) -> Option<u64> {
    let limits = spec::plan(&config(dir, slots, CacheFormat::F32)).ok()?;
    Some(
        limits
            .memory
            .mandatory
            .iter()
            .filter(|term| matches!(term.tier, Tier::Device(_)))
            .map(|term| term.bytes)
            .sum(),
    )
}

#[test]
#[ignore = "it writes up to 160 GiB of filler to the disk; run it with --ignored"]
fn an_open_that_the_driver_count_refuses_but_the_host_count_allows_succeeds() {
    let Some(dir) = model_dir("cuda_memavail") else {
        return;
    };
    let _guard = watchdog(Duration::from_secs(900), "the memavail gate");
    let device = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the probe device");
    let started = std::time::Instant::now();
    let read = io::copy(&mut File::open(dir.join(WEIGHTS_FILE)).unwrap(), &mut io::sink()).unwrap();
    // The checkpoint alone leaves a gap of some GB. Write a temporary file until the driver reports little free memory:
    // the written pages stay in the page cache, which the driver leaves out. The file is removed at the end.
    let scratch = tempfile::tempdir().unwrap();
    let mut filler = File::create(scratch.path().join("filler")).unwrap();
    let chunk = vec![0x5Au8; 1 << 30];
    for written in 0..160 {
        if device.memory_info().unwrap().0 < FILL_TO {
            break;
        }
        io::Write::write_all(&mut filler, &chunk).unwrap();
        eprintln!("wrote {} GiB of filler", written + 1);
    }
    let (available, total) = device.available_bytes().unwrap();
    let (free, _) = device.memory_info().unwrap();
    let (free, total) = (free as u64, total as u64);
    eprintln!(
        "read {read} bytes in {:.1} s; driver free {free}, available {} ({}), total {total}",
        started.elapsed().as_secs_f64(),
        available.bytes,
        available.count.label()
    );
    assert_eq!(available.count, Count::MemAvailable, "the GB10 is an integrated GPU");
    assert!(
        available.bytes >= free,
        "the available bytes {} are less than the free bytes {free} of the driver",
        available.bytes
    );
    // The smallest plan above the free bytes of the driver, by one slot of growing positions.
    let planned = (1..=64usize)
        .map(|k| vec![8192 * k])
        .chain((2..=8usize).map(|n| vec![65536; n]))
        .chain((2..=8usize).map(|n| vec![131072; n]))
        .find_map(|slots| {
            let bytes = planned_bytes(&dir, &slots)?;
            (bytes > free + MARGIN).then_some((slots, bytes))
        });
    let Some((slots, planned)) = planned else {
        eprintln!("cuda_memavail: skipped, no plan passes the {free} free bytes of the driver");
        return;
    };
    if planned + MIN_FREE_AFTER_BYTES + MARGIN > available.bytes {
        eprintln!(
            "cuda_memavail: skipped, the gap is too small: plan {planned}, driver free {free}, available {}",
            available.bytes
        );
        return;
    }
    eprintln!("the plan of slots {slots:?} takes {planned} bytes: more than the {free} free, less than the available");
    let engine = open(config(&dir, &slots, CacheFormat::F32));
    assert!(engine.weights_bytes() > 0);
    assert_eq!(engine.close(), 0, "the close frees each buffer");
    drop(filler);
}
