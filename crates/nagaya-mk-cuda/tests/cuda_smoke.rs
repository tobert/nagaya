//! GPU smoke gate (Milestone 37 step 5): verify `sm_121a` cubin load, persistent interpreter execution,
//! `OP_NOP`/`OP_COPY`/`OP_HALT`, scoreboard synchronization, and `%globaltimer` profiling.
//!
//! Run with one thread and a free GPU PCI address:
//! `NAGAYA_TEST_CARDS=0000:0a:00.0 cargo test -p nagaya-mk-cuda --features cuda --test cuda_smoke -- --test-threads=1`

#![cfg(feature = "cuda")]

use std::ffi::c_void;

use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::protocol::{
    MkTaskDescriptor, OP_COPY, OP_NOP, OP_SELFTEST_CHECK, OP_SELFTEST_FILL, PROFILE_WORDS, selftest_value,
};
use nagaya_mk_cuda::ring::{Ring, Task, chained};
use nagaya_mk_cuda::{INKLING2_CUBIN, INKLING2_PROFILE_CUBIN, test_cards};

#[test]
fn persistent_interpreter_runs_copy_and_profile_on_gb10() {
    let pci = test_cards::card();
    let dev = CudaDevice::open_by_pci(&pci, INKLING2_CUBIN).expect("open GB10 with INKLING2_CUBIN");
    eprintln!(
        "cuda_smoke: opened {} (SMs={}, driver={})",
        dev.pci_address(),
        dev.sm_count(),
        dev.driver_version()
    );
    assert!(dev.sm_count() >= 1);

    let n: usize = 16_384;
    let bytes = n * 4;
    let host_src: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 1024.5).collect();
    let mut host_dst = vec![0.0f32; n];

    let dev_src = dev.alloc_device(bytes).unwrap();
    let dev_dst = dev.alloc_device(bytes).unwrap();
    unsafe {
        dev.copy_htod(dev_src, host_src.as_ptr().cast::<c_void>(), bytes)
            .unwrap()
    };
    dev.memset_zero(dev_dst, bytes).unwrap();

    let mut ring = Ring::new(&dev, 1024, false).unwrap();
    ring.zero_device().unwrap();
    // 16 items of 1,024 floats. The items run on whatever blocks the assignment gives.
    ring.set(&[
        Task::per_block(MkTaskDescriptor::default()),
        Task::with_items(MkTaskDescriptor {
            opcode: OP_COPY,
            items: 16,
            a: dev_dst,
            b: dev_src,
            n: 1024,
            ..Default::default()
        })
        .after(0),
        Task::per_block(MkTaskDescriptor::default()).after(1),
    ]);
    unsafe { ring.run().unwrap() };
    unsafe {
        dev.copy_dtoh(host_dst.as_mut_ptr().cast::<c_void>(), dev_dst, bytes)
            .unwrap()
    };

    for (i, (&got, &want)) in host_dst.iter().zip(&host_src).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "mismatch at index {i}: got {got}, want {want}"
        );
    }

    let stats = ring.read_stats().unwrap();
    assert_eq!(stats.len(), dev.sm_count() as usize);
    for (bid, st) in stats.iter().enumerate() {
        assert_eq!(st.tasks_run, 3, "block {bid} ran {} tasks", st.tasks_run);
        assert!(st.exec_ns > 0, "block {bid} reported 0 exec_ns");
    }
    ring.free().unwrap();
    dev.free_device(dev_src).unwrap();
    dev.free_device(dev_dst).unwrap();
    drop(dev);

    // Verify the MK_PROFILE cubin records monotonic %globaltimer timestamps.
    let prof_dev = CudaDevice::open_by_pci(&pci, INKLING2_PROFILE_CUBIN).expect("open GB10 with profile cubin");
    let mut prof_ring = Ring::new(&prof_dev, 1024, true).unwrap();
    prof_ring.zero_device().unwrap();
    prof_ring.set(&chained(&[
        MkTaskDescriptor {
            opcode: OP_NOP,
            ..Default::default()
        },
        MkTaskDescriptor {
            opcode: OP_NOP,
            ..Default::default()
        },
    ]));
    unsafe { prof_ring.run().unwrap() };
    let prof = prof_ring.read_profile().unwrap();
    let blocks = prof_ring.grid_blocks() as usize;
    // 2 tasks + 1 trailing OP_HALT = 3 entries per block.
    assert_eq!(prof.len(), 3 * blocks * PROFILE_WORDS);
    for task in 0..3 {
        for bid in 0..blocks {
            let base = (task * blocks + bid) * PROFILE_WORDS;
            let (t_start, t_wait, t_end) = (prof[base], prof[base + 1], prof[base + 2]);
            assert!(t_start > 0, "task {task} block {bid}: t_start=0");
            assert!(
                t_start <= t_wait && t_wait <= t_end,
                "task {task} block {bid}: non-monotonic ({t_start}, {t_wait}, {t_end})"
            );
        }
    }
    prof_ring.free().unwrap();
}

#[test]
fn a_device_moves_to_another_thread_and_works_there() {
    let pci = test_cards::card();
    let dev = CudaDevice::open_by_pci(&pci, INKLING2_CUBIN).expect("open GB10");
    // The thread that opened the device holds the context. The new thread holds none.
    std::thread::spawn(move || {
        let ptr = dev.alloc_device(4096).expect("alloc on the second thread");
        dev.memset_zero(ptr, 4096).expect("memset on the second thread");
        dev.sync().expect("sync on the second thread");
        dev.free_device(ptr).expect("free on the second thread");
    })
    .join()
    .expect("the second thread did not panic");
}

#[test]
fn a_ring_that_fails_to_allocate_frees_what_it_holds() {
    let pci = test_cards::card();
    let dev = CudaDevice::open_by_pci(&pci, INKLING2_CUBIN).expect("open GB10");
    let overflow = Ring::new(&dev, usize::MAX, false).expect_err("an overflowing scratch is an error");
    assert!(overflow.contains("overflows"), "{overflow}");
    // 2^50 floats are 4 PiB. The first four buffers succeed, and the scratch fails.
    let too_big = Ring::new(&dev, 1 << 50, false).expect_err("4 PiB of scratch is an error");
    assert!(too_big.contains("cuMemAlloc failed"), "{too_big}");
    assert!(too_big.contains("OUT_OF_MEMORY"), "{too_big}");
    assert_eq!(dev.live_buffers(), 0, "the failed rings leaked buffers");
    // The device still works, and a later ring runs.
    let ring = Ring::new(&dev, 1024, false).expect("a later ring allocates");
    assert_eq!(dev.live_buffers(), 5, "a ring holds 5 buffers with no profile");
    ring.free().expect("the ring frees all of its buffers");
    assert_eq!(dev.live_buffers(), 0, "free left buffers");
    drop(Ring::new(&dev, 1024, true).expect("a profile ring allocates"));
    assert_eq!(dev.live_buffers(), 0, "Drop left buffers");
}

#[test]
fn a_grid_that_is_not_resident_is_refused_before_the_launch() {
    let pci = test_cards::card();
    let dev = CudaDevice::open_by_pci(&pci, INKLING2_CUBIN).expect("open GB10");
    let resident = dev.max_resident_blocks(256).expect("occupancy");
    assert!(
        resident >= dev.sm_count(),
        "resident={resident}, SMs={}",
        dev.sm_count()
    );
    // No pointer is valid here. The call must refuse before the launch reads one.
    let error = unsafe { dev.launch_megakernel(resident + 1, 256, 0, 1, 0, 0, 0, 0, 0) }
        .expect_err("a grid that is not resident is an error");
    assert!(error.contains("hang"), "{error}");
    assert!(
        error.contains(&format!("Use a grid of {resident} blocks or fewer")),
        "{error}"
    );
}

/// Words that each block writes in the scoreboard test.
const WORDS_PER_BLOCK: usize = 1024;
/// Delay of the slow block of a fill, in ns. It is long enough for the other blocks to reach the next task.
const DELAY_NS: u32 = 300_000;

/// The tasks of one round: fill, check, fill again with a new seed and a new slow block, check again, then two nops.
/// The tasks declare the ranges that they touch, and the builder orders them from the ranges. The nops follow each
/// other by an explicit order, because they touch nothing.
fn scoreboard_tasks(data: u64, out1: u64, out2: u64, slow_a: u32, slow_b: u32, blocks: usize) -> Vec<Task> {
    let data_bytes = blocks * WORDS_PER_BLOCK * 4;
    let fill = |seed: u32, slow: u32| {
        Task::per_block(MkTaskDescriptor {
            opcode: OP_SELFTEST_FILL,
            a: data,
            n: WORDS_PER_BLOCK as u32,
            k: DELAY_NS,
            kt: slow,
            group_items: seed,
            ..Default::default()
        })
        .writes(data, data_bytes)
    };
    let check = |seed: u32, out: u64| {
        Task::per_block(MkTaskDescriptor {
            opcode: OP_SELFTEST_CHECK,
            a: data,
            e: out,
            n: WORDS_PER_BLOCK as u32,
            group_items: seed,
            ..Default::default()
        })
        .reads(data, data_bytes)
        .writes(out, blocks * 4)
    };
    let nop = || Task::per_block(MkTaskDescriptor::default());
    vec![
        fill(1, slow_a),
        check(1, out1),
        fill(2, slow_b),
        check(2, out2),
        nop().after(3),
        nop().after(4),
    ]
}

/// Run the ring in a thread with a watchdog, so that a hang fails the test.
fn run_with_watchdog(ring: &Ring<'_>) {
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for _ in 0..300 {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if done.load(std::sync::atomic::Ordering::SeqCst) {
                    return;
                }
            }
            eprintln!("cuda_smoke: the ring did not finish within 30 s; it hangs");
            std::process::exit(2);
        });
        unsafe { ring.run().unwrap() };
        done.store(true, std::sync::atomic::Ordering::SeqCst);
    });
}

#[test]
fn the_scoreboard_orders_the_tasks_of_all_blocks() {
    let pci = test_cards::card();
    let dev = CudaDevice::open_by_pci(&pci, INKLING2_PROFILE_CUBIN).expect("open GB10 with the profile cubin");
    let mut ring = Ring::new(&dev, 1024, true).unwrap();
    let blocks = ring.grid_blocks() as usize;
    let data_words = blocks * WORDS_PER_BLOCK;
    let data = dev.alloc_device(data_words * 4).unwrap();
    let out1 = dev.alloc_device(blocks * 4).unwrap();
    let out2 = dev.alloc_device(blocks * 4).unwrap();

    for round in 0..6 {
        // A word that no block writes stays zero, and the check counts it as wrong.
        dev.memset_zero(data, data_words * 4).unwrap();
        dev.memset_zero(out1, blocks * 4).unwrap();
        dev.memset_zero(out2, blocks * 4).unwrap();
        let slow_a = (blocks - 1 - round % blocks) as u32;
        let slow_b = (round * 7 % blocks) as u32;
        ring.set(&scoreboard_tasks(data, out1, out2, slow_a, slow_b, blocks));
        run_with_watchdog(&ring);

        // Each block read the words of all other blocks, and found each one right.
        for (name, out) in [("first", out1), ("second", out2)] {
            let mut wrong = vec![u32::MAX; blocks];
            unsafe {
                dev.copy_dtoh(wrong.as_mut_ptr().cast::<c_void>(), out, blocks * 4)
                    .unwrap()
            };
            assert!(
                wrong.iter().all(|&w| w == 0),
                "round {round}: a block read stale words in the {name} check: {wrong:?}"
            );
        }
        // The check is not empty: the words hold the pattern of the second seed.
        let mut words = vec![0u32; data_words];
        unsafe {
            dev.copy_dtoh(words.as_mut_ptr().cast::<c_void>(), data, data_words * 4)
                .unwrap()
        };
        for (k, &got) in words.iter().enumerate() {
            let want = selftest_value(2, (k / WORDS_PER_BLOCK) as u32, (k % WORDS_PER_BLOCK) as u32);
            assert_eq!(got, want, "round {round}: word {k}");
        }

        // Each task signals its own event, one count for each block. The halt signals no event.
        let tasks = 6;
        let scoreboard = ring.read_scoreboard().unwrap();
        assert_eq!(scoreboard.len(), tasks + 1);
        for (event, &count) in scoreboard.iter().enumerate().take(tasks) {
            assert_eq!(count, blocks as u32, "round {round}: event {event}");
        }
        assert_eq!(scoreboard[tasks], 0, "round {round}: the halt signals no event");

        // Task i + 1 leaves its wait in each block no earlier than the last end of task i in any block. The halt
        // waits on nothing, so the pairs end with the last task.
        let prof = ring.read_profile().unwrap();
        let slot = |task: usize, block: usize| {
            let base = (task * blocks + block) * PROFILE_WORDS;
            (prof[base], prof[base + 1], prof[base + 2])
        };
        for task in 0..tasks - 1 {
            let last_end = (0..blocks).map(|b| slot(task, b).2).max().unwrap();
            for block in 0..blocks {
                let after_wait = slot(task + 1, block).1;
                assert!(
                    after_wait >= last_end,
                    "round {round}: block {block} left the wait of task {} {} ns before task {task} ended",
                    task + 1,
                    last_end - after_wait
                );
            }
        }
    }
    ring.free().unwrap();
    dev.free_device(data).unwrap();
    dev.free_device(out1).unwrap();
    dev.free_device(out2).unwrap();
}
