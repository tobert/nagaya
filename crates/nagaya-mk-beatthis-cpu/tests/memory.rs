//! The host bytes that the engine declares (`engine::host_bytes`) against the bytes that a run of one whole chunk
//! allocates. A counting allocator holds the peak of the live bytes.

#[path = "../../nagaya-family-beatthis/tests/common/mod.rs"]
mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use nagaya_family_beatthis::chunks::CHUNK;
use nagaya_family_beatthis::mel::MELS;
use nagaya_family_beatthis::plan::{self, Config};
use nagaya_family_beatthis::weights::{self, Tensors};
use nagaya_mk_beatthis_cpu as cpu;

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: each call goes to the system allocator with the same layout; the counters do not touch the memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static A: Counting = Counting;

/// A run of one whole chunk of final0 on 8 threads allocates at most the declared bytes of 8 threads less the
/// weights, and at least 99% of them, so the declared bytes are a bound that is not too large.
#[test]
fn a_chunk_of_final0_allocates_the_declared_bytes() {
    let Some(bytes) = common::weights("final0") else { return };
    let config = Config::from_json(&Tensors::hyper_parameters(&bytes).unwrap()).unwrap();
    let t = Tensors::parse(&bytes).unwrap();
    let p = plan::build(&config, CHUNK);
    let w = weights::prepare(&p.weights, &t).unwrap();
    drop(t);
    drop(bytes);
    let input = vec![0.5f32; CHUNK * MELS];
    let threads = 8;
    cpu::kernels::start_workers(threads);
    let weights_bytes: usize = w.iter().map(|v| 4 * v.len()).sum();
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let out = cpu::run(&p, &w, &input, threads, |_, _| {});
    let used = PEAK.load(Ordering::Relaxed) - base;
    drop(out);
    let declared = cpu::engine::host_bytes_for(&config, threads) as usize - weights_bytes;
    eprintln!("declared {declared} bytes past the weights, used {used}");
    assert!(
        used <= declared,
        "a chunk used {used} bytes, and the engine declares {declared}"
    );
    assert!(
        used as f64 >= 0.99 * declared as f64,
        "a chunk used {used} bytes, and the engine declares {declared}: the declared bytes are too large"
    );
}
