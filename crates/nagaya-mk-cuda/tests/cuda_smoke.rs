//! GPU smoke gate (Milestone 37 step 5): verify `sm_121a` cubin load, persistent interpreter execution,
//! `OP_NOP`/`OP_COPY`/`OP_HALT`, scoreboard synchronization, and `%globaltimer` profiling.
//!
//! Run with one thread and a free GPU PCI address:
//! `NAGAYA_TEST_CARDS=000f:01:00.0 cargo test -p nagaya-mk-cuda --features cuda --test cuda_smoke -- --test-threads=1`

#![cfg(feature = "cuda")]

use std::ffi::c_void;

use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, OP_COPY, OP_NOP, PROFILE_WORDS};
use nagaya_mk_cuda::ring::Ring;
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
    assert!(dev.driver_version() >= 13_000, "driver={}", dev.driver_version());

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
    ring.zero_device(&dev).unwrap();
    ring.set(&[
        MkTaskDescriptor {
            opcode: OP_NOP,
            ..Default::default()
        },
        MkTaskDescriptor {
            opcode: OP_COPY,
            a: dev_dst,
            b: dev_src,
            p0: n as u32,
            ..Default::default()
        },
        MkTaskDescriptor {
            opcode: OP_NOP,
            ..Default::default()
        },
    ]);
    unsafe { ring.run(&dev).unwrap() };
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

    let stats = ring.read_stats(&dev).unwrap();
    assert_eq!(stats.len(), dev.sm_count() as usize);
    for (bid, st) in stats.iter().enumerate() {
        assert_eq!(st.tasks_run, 3, "block {bid} ran {} tasks", st.tasks_run);
        assert!(st.exec_ns > 0, "block {bid} reported 0 exec_ns");
    }
    ring.free(&dev).unwrap();
    dev.free_device(dev_src).unwrap();
    dev.free_device(dev_dst).unwrap();
    drop(dev);

    // Verify the MK_PROFILE cubin records monotonic %globaltimer timestamps.
    let prof_dev = CudaDevice::open_by_pci(&pci, INKLING2_PROFILE_CUBIN).expect("open GB10 with profile cubin");
    let mut prof_ring = Ring::new(&prof_dev, 1024, true).unwrap();
    prof_ring.zero_device(&prof_dev).unwrap();
    prof_ring.set(&[
        MkTaskDescriptor {
            opcode: OP_NOP,
            ..Default::default()
        },
        MkTaskDescriptor {
            opcode: OP_NOP,
            ..Default::default()
        },
    ]);
    unsafe { prof_ring.run(&prof_dev).unwrap() };
    let prof = prof_ring.read_profile(&prof_dev).unwrap();
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
    prof_ring.free(&prof_dev).unwrap();
}
