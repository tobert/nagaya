//! GPU stress gate of the signal (Milestone 37 step 7, substep 6): a block signals once for each task, with one
//! release add. A chain of copy stages, each one reading data that blocks of the stage before wrote, runs many
//! times at three grid widths. Every result must be exact, and each event must hold the items of its task.
//!
//! The runs for each grid come from `NAGAYA_STRESS_RUNS` (default 10,000).
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-cuda --features cuda --test cuda_stress -- --test-threads=1`

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, OP_COPY};
use nagaya_mk_cuda::ring::{Ring, Task};
use nagaya_mk_cuda::{INKLING2_CUBIN, test_cards};

/// The grid widths of the gate: every SM of the GB10, half of them, and a count that divides no count of items.
const GRIDS: [u32; 3] = [48, 24, 7];
/// Words in an item of a stage.
const ITEM_WORDS: u32 = 1000;
/// The stage that reads the stage before it reads at this offset in words. It is not a multiple of `ITEM_WORDS`, so
/// an item reads words of two items of its producer, and those items belong to two blocks.
const SHIFT_WORDS: u32 = 300;
/// Items of stage 0. Each stage has one item less than the stage before it, so that its last item stays in range.
const FIRST_ITEMS: u32 = 400;
const STAGES: u32 = 300;
/// A run of the whole ring must end within this time, or the test ends the process.
const WATCHDOG: Duration = Duration::from_secs(20);

fn items_of(stage: u32) -> u32 {
    FIRST_ITEMS - stage
}

fn words_of(stage: u32) -> usize {
    items_of(stage) as usize * ITEM_WORDS as usize
}

fn runs() -> u64 {
    std::env::var("NAGAYA_STRESS_RUNS")
        .ok()
        .map(|v| v.parse().expect("NAGAYA_STRESS_RUNS is a count of runs"))
        .unwrap_or(10_000)
}

fn pattern(index: u32) -> u32 {
    index.wrapping_mul(0x9E37_79B1) ^ (index >> 7) ^ 0x1234_5678
}

fn download(dev: &CudaDevice, ptr: CUdeviceptr, words: usize) -> Vec<u32> {
    let mut out = vec![0u32; words];
    unsafe { dev.copy_dtoh(out.as_mut_ptr().cast::<c_void>(), ptr, words * 4) }.unwrap();
    out
}

#[test]
fn a_chain_of_stages_is_exact_at_each_grid_and_each_event_counts_its_items() {
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU");
    let mut ring = Ring::new(&dev, 1024, false).unwrap();
    // Buffer `s` holds the output of stage `s`. Buffer 0 holds the input, with room for the shift of stage 1.
    let buffers: Vec<CUdeviceptr> = (0..=STAGES)
        .map(|s| dev.alloc_device((words_of(s) + SHIFT_WORDS as usize) * 4).unwrap())
        .collect();
    let input: Vec<u32> = (0..words_of(0) as u32 + SHIFT_WORDS).map(pattern).collect();
    let tasks: Vec<Task> = (1..=STAGES)
        .map(|s| {
            let src = buffers[s as usize - 1] + u64::from(SHIFT_WORDS) * 4;
            let dst = buffers[s as usize];
            let bytes = words_of(s) * 4;
            Task::with_items(MkTaskDescriptor {
                opcode: OP_COPY,
                items: items_of(s),
                a: dst,
                b: src,
                n: ITEM_WORDS,
                ..Default::default()
            })
            .reads(src, bytes)
            .writes(dst, bytes)
        })
        .collect();
    ring.set(&tasks);

    // The stage `s` holds the input shifted by `s * SHIFT_WORDS`.
    let want_last: Vec<u32> = (0..words_of(STAGES))
        .map(|j| input[j + (STAGES * SHIFT_WORDS) as usize])
        .collect();

    let progress = AtomicU64::new(0);
    let finished = AtomicBool::new(false);
    let runs = runs();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let (mut seen, mut since) = (0, Instant::now());
            while !finished.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(100));
                let now = progress.load(Ordering::SeqCst);
                if now != seen {
                    (seen, since) = (now, Instant::now());
                } else if since.elapsed() > WATCHDOG {
                    eprintln!("cuda_stress: no run ended within {WATCHDOG:?} after run {now}; the ring hangs");
                    std::process::exit(2);
                }
            }
        });
        let started = Instant::now();
        for &grid in &GRIDS {
            for run in 0..runs {
                unsafe { dev.copy_htod(buffers[0], input.as_ptr().cast::<c_void>(), input.len() * 4) }.unwrap();
                // A stage that reads before its producer wrote reads zeros, so a wrong order shows in the result.
                for (stage, &buffer) in buffers.iter().enumerate().skip(1) {
                    dev.memset_zero(buffer, (words_of(stage as u32) + SHIFT_WORDS as usize) * 4)
                        .unwrap();
                }
                unsafe { ring.run_grid(grid, 256, 0).unwrap() };
                progress.fetch_add(1, Ordering::SeqCst);

                let got = download(&dev, buffers[STAGES as usize], want_last.len());
                if let Some(at) = got.iter().zip(&want_last).position(|(g, w)| g != w) {
                    panic!(
                        "grid {grid}, run {run}: the last stage is wrong at word {at}: {:#x}, wanted {:#x}",
                        got[at], want_last[at]
                    );
                }
                // The events: each holds the items of its task, so that a wait on it is the same at each grid.
                let events = ring.read_scoreboard().unwrap();
                for s in 1..=STAGES {
                    assert_eq!(
                        events[s as usize - 1],
                        items_of(s),
                        "grid {grid}, run {run}: the event of stage {s}"
                    );
                }
            }
            eprintln!(
                "cuda_stress: grid {grid}: {runs} runs of {STAGES} stages are exact ({:.1} s so far)",
                started.elapsed().as_secs_f64()
            );
        }
        finished.store(true, Ordering::SeqCst);
    });

    ring.free().unwrap();
    for buffer in buffers {
        dev.free_device(buffer).unwrap();
    }
}
