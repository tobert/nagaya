//! An open on one card beside work on another card (spec 3.3a, "Single-card swaps"). Card A runs convolutions in a
//! loop, and card B opens in the middle. The open of a single-card engine must not stall the launches of card A.
//!
//! A second test uploads `XE_UPLOAD_GB` GB (4 by default) to card B in 64 MiB copies through one staging buffer, as
//! the SDXL engine does, while card A runs.
//!
//! It needs two cards. Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run
//! alone: `cargo test --release -p nagaya-mk-bmg --features xe --test xe_open_beside -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::*;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::{KERNEL_ENTRY, SDXL_KERNEL};

/// The quiet time before and after the open.
const QUIET: Duration = Duration::from_millis(1500);

/// The launches of card A: the start of each launch after the start of the run, and its duration.
type Launches = Vec<(Duration, Duration)>;

/// Run launches of 3 x 3 convolutions on the card at `address` until `stop` is set. `begin` is the clock of the run.
fn work(address: String, begin: Instant, ready: Arc<AtomicBool>, stop: Arc<AtomicBool>) -> Launches {
    let cluster = L0Cluster::init_at(SDXL_KERNEL, KERNEL_ENTRY, &[address]).expect("card A opens");
    let card: &B70Card = &cluster.cards[0];
    let (b, h, w, cin, cout) = (1usize, 64usize, 64usize, 320usize, 320usize);
    let zeros = |bytes: usize| {
        let ptr = card.alloc_device_usm::<u8>(bytes.max(64));
        card.fill_zero_sync(ptr as *mut c_void, bytes.max(64));
        ptr as u64
    };
    let (x, wt, y) = (
        zeros(b * h * w * cin * 4),
        zeros(cout * cin * 9 * 2),
        zeros(b * h * w * cout * 4),
    );
    let conv = MkTaskDescriptor {
        opcode: OP_CONV2D,
        flags: FLAG_F16,
        n: (h | w << 16) as u32,
        m: cout as u32,
        k: cin as u32,
        p0: b as u32,
        layer: (3 | 3 << 8 | 1 << 16 | 1 << 24) as u32,
        a: x,
        b: wt,
        c: y,
        ..Default::default()
    };
    let mut ring = Ring::new(card, 1);
    ring.set(&[conv; 4]);
    let mut launches = Vec::new();
    while !stop.load(Ordering::SeqCst) {
        let at = Instant::now();
        // SAFETY: the buffers hold the sizes of the convolution.
        unsafe { ring.run(card) };
        launches.push((at - begin, at.elapsed()));
        ready.store(true, Ordering::SeqCst);
    }
    // A destroyed context does not free its memory, so the buffers go first.
    ring.free(card);
    for ptr in [x, wt, y] {
        card.free(ptr as *mut c_void);
    }
    launches
}

/// Run card A, open card B with `open` in the middle, and give the median launch of card A, the longest launch of card
/// A that overlaps the open, and the time of the open.
fn beside(open: impl FnOnce() -> L0Cluster) -> (Duration, Duration, Duration) {
    let addresses = L0Cluster::addresses();
    assert!(
        addresses.len() >= 2,
        "the test needs two cards, and the driver has {}",
        addresses.len()
    );
    let begin = Instant::now();
    let (ready, stop) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let worker = {
        let (address, ready, stop) = (addresses[0].clone(), ready.clone(), stop.clone());
        thread::spawn(move || work(address, begin, ready, stop))
    };
    while !ready.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(10));
    }
    thread::sleep(QUIET);
    let from = begin.elapsed();
    drop(open());
    let to = begin.elapsed();
    thread::sleep(QUIET);
    stop.store(true, Ordering::SeqCst);
    let launches = worker.join().expect("the work of card A panicked");
    let mut all: Vec<Duration> = launches.iter().map(|(_, took)| *took).collect();
    all.sort();
    let median = all[all.len() / 2];
    let longest = launches
        .iter()
        .filter(|(at, took)| *at + *took >= from && *at <= to)
        .map(|(_, took)| *took)
        .max()
        .unwrap_or_default();
    (median, longest, to - from)
}

#[test]
fn an_open_of_one_card_does_not_stall_the_launches_of_another_card() {
    let other = L0Cluster::addresses()[1].clone();
    // The old open made a context, queues and the module on every card, then dropped the others. Print it to compare.
    let (median, longest, took) = beside(|| L0Cluster::init(SDXL_KERNEL, KERNEL_ENTRY));
    eprintln!("open of every card: {took:?}; card A median launch {median:?}, longest beside the open {longest:?}");
    let (median, longest, took) =
        beside(|| L0Cluster::init_at(SDXL_KERNEL, KERNEL_ENTRY, &[other]).expect("card B opens"));
    eprintln!("open of card B only: {took:?}; card A median launch {median:?}, longest beside the open {longest:?}");
    assert!(
        longest <= median * 2 + Duration::from_millis(50),
        "an open of card B stalled a launch of card A: {longest:?} against a median of {median:?}"
    );
}

/// Open card B and upload `bytes` bytes to it as the SDXL engine does: an allocation and a zero fill for each 256 MiB
/// tensor, then copies of 64 MiB through one buffer of host memory.
fn open_and_upload(address: String, bytes: usize) -> L0Cluster {
    const STAGING: usize = 64 << 20;
    const TENSOR: usize = 256 << 20;
    let cluster = L0Cluster::init_at(SDXL_KERNEL, KERNEL_ENTRY, &[address]).expect("card B opens");
    let card = &cluster.cards[0];
    let staging = card.alloc_host_usm::<u8>(STAGING);
    // SAFETY: the staging buffer holds STAGING bytes of host memory.
    unsafe { std::ptr::write_bytes(staging, 0x3c, STAGING) };
    let mut tensors = Vec::new();
    for _ in 0..bytes.div_ceil(TENSOR) {
        let at = card.alloc_device_usm::<u8>(TENSOR);
        card.fill_zero_sync(at as *mut c_void, TENSOR);
        for offset in (0..TENSOR).step_by(STAGING) {
            card.memcpy_sync((at as usize + offset) as *mut c_void, staging as *const c_void, STAGING);
        }
        tensors.push(at);
    }
    for at in tensors {
        card.free(at as *mut c_void);
    }
    card.free(staging as *mut c_void);
    cluster
}

#[test]
fn an_upload_of_gigabytes_to_one_card_does_not_stall_the_launches_of_another_card() {
    let other = L0Cluster::addresses()[1].clone();
    let gb: usize = std::env::var("XE_UPLOAD_GB")
        .ok()
        .map(|value| value.parse().expect("XE_UPLOAD_GB is a whole number of GB"))
        .unwrap_or(4);
    let (median, longest, took) = beside(|| open_and_upload(other, gb << 30));
    eprintln!("upload of {gb} GB to card B: {took:?}; card A median launch {median:?}, longest beside it {longest:?}");
    assert!(
        longest <= median * 2 + Duration::from_millis(50),
        "an upload to card B stalled a launch of card A: {longest:?} against a median of {median:?}"
    );
}
