//! A wait that never ends must trap, not hang (Milestone 37 step 6). A trap leaves the primary context of the process
//! unusable, so this test lives in its own test binary.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-cuda --features cuda --test cuda_spin`

#![cfg(feature = "cuda")]

use std::ffi::c_void;

use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, NO_EVENT, OP_HALT, OP_NOP};
use nagaya_mk_cuda::{INKLING2_SPIN_TEST_CUBIN, test_cards};

#[test]
fn a_wait_that_never_ends_traps_instead_of_hanging() {
    // A watchdog ends the process if the trap does not fire, so a hang fails the test.
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(60));
        eprintln!("cuda_smoke: the launch did not trap within 60 s; it hangs");
        std::process::exit(2);
    });
    let pci = test_cards::card();
    let dev = CudaDevice::open_by_pci(&pci, INKLING2_SPIN_TEST_CUBIN).expect("open GB10");
    // Task 0 waits for event 7. No task signals it.
    let tasks = [
        MkTaskDescriptor {
            opcode: OP_NOP,
            wait_event: 7,
            signal_event: NO_EVENT,
            ..Default::default()
        },
        MkTaskDescriptor {
            opcode: OP_HALT,
            ..Default::default()
        },
    ];
    let dev_tasks = dev.alloc_device(128).unwrap();
    let scoreboard = dev.alloc_device(64).unwrap();
    dev.memset_zero(scoreboard, 64).unwrap();
    unsafe { dev.copy_htod(dev_tasks, tasks.as_ptr().cast::<c_void>(), 128).unwrap() };
    let error = unsafe { dev.launch_megakernel(dev.sm_count(), 256, dev_tasks, 2, scoreboard, 0, 0, 0, 0) }
        .expect_err("the wait traps");
    assert!(error.contains("cuCtxSynchronize failed"), "{error}");
}
