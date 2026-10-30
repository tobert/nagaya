//! The CPU set of each stage thread (spec 3.2b). The test reads the mask of each thread with `sched_getaffinity`.

mod feed_support;
mod task_support;

use std::time::Duration;

use feed_support::{FeedFactory, FeedFixture};
use nagaya::affinity::current_affinity;
use nagaya::cpu_sets::CpuSets;
use task_support::threads_named;

/// The mask of the thread of this process that has `name`.
fn mask_named(name: &str) -> Vec<u32> {
    let Some(&tid) = threads_named(name).first() else {
        panic!("no thread has the name {name}");
    };
    // SAFETY: an all-zero `cpu_set_t` is an empty mask.
    let mut mask: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: `mask` is writable and has the size that the call is given.
    let result = unsafe { libc::sched_getaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &mut mask) };
    assert_eq!(result, 0, "sched_getaffinity failed for thread {tid}");
    // SAFETY: each index is below the mask size.
    (0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
        .map(|cpu| cpu as u32)
        .collect()
}

#[test]
fn each_stage_thread_runs_on_the_set_of_its_device_and_the_group_thread_on_the_pool() {
    let allowed = current_affinity().unwrap();
    if allowed.len() < 6 {
        eprintln!(
            "skipped: this test needs six CPUs, and the process may use {}",
            allowed.len()
        );
        return;
    }
    let sets = CpuSets {
        housekeeping: allowed[..2].to_vec(),
        pool: allowed[2..6].to_vec(),
        devices: vec![allowed[2..4].to_vec(), allowed[4..6].to_vec()],
    };
    let fixture = FeedFixture::with_sets(FeedFactory::new(2), Duration::from_secs(10), Some(sets.clone()));
    assert_eq!(mask_named("nagaya-group-0"), sets.pool);
    assert_eq!(mask_named("nagaya-stg-0-0"), sets.devices[0]);
    assert_eq!(mask_named("nagaya-stg-0-1"), sets.devices[1]);
    drop(fixture);
}
