//! The affinity calls of spec 3.2a. Each test runs on its own thread, so a mask does not reach the test runner.

use std::thread;

use nagaya::affinity::{current_affinity, current_policy, pin_current_thread, set_batch_policy};

#[test]
fn a_one_cpu_mask_reads_back_on_its_thread() {
    let read_back = thread::spawn(|| {
        pin_current_thread(&[0]).expect("CPU 0 is online on every test machine");
        current_affinity().expect("the mask reads back")
    })
    .join()
    .expect("the thread did not panic");
    assert_eq!(read_back, vec![0]);
}

#[test]
fn a_pool_thread_runs_at_sched_batch() {
    let policy = thread::spawn(|| {
        set_batch_policy().expect("SCHED_BATCH needs no privilege");
        current_policy().expect("the policy reads back")
    })
    .join()
    .expect("the thread did not panic");
    assert_eq!(policy, libc::SCHED_BATCH);
}

#[test]
fn a_cpu_past_the_mask_size_is_refused() {
    let result = thread::spawn(|| pin_current_thread(&[5000]))
        .join()
        .expect("the thread did not panic");
    let error = result.expect_err("a CPU past CPU_SETSIZE is an error");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
}
