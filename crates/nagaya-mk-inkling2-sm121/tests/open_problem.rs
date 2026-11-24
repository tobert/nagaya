//! The memory rule of `open_problem` (spec 3.3b): the planned device bytes against the free bytes of the GPU and the
//! bytes that the unload of a swap frees, with the floor of 2 GiB of the engine's memory rule. It needs no GPU.

use nagaya_mk_cuda::memory::Count;
use nagaya_mk_inkling2_sm121::describe::open_memory_problem;
use nagaya_mk_inkling2_sm121::health::MIN_FREE_AFTER_BYTES;

const GIB: u64 = 1 << 30;
const HOST: Count = Count::MemAvailable;

#[test]
fn an_open_fits_when_the_free_bytes_hold_the_plan_and_the_floor() {
    let planned = 20 * GIB;
    let total = 121 * GIB;
    assert_eq!(
        open_memory_problem(planned, planned + MIN_FREE_AFTER_BYTES, total, 0, HOST),
        None
    );
    assert!(open_memory_problem(planned, planned + MIN_FREE_AFTER_BYTES - 1, total, 0, HOST).is_some());
}

#[test]
fn the_freed_bytes_count_as_free() {
    let planned = 20 * GIB;
    let total = 121 * GIB;
    let free = 5 * GIB;
    assert!(open_memory_problem(planned, free, total, 0, HOST).is_some());
    let need = planned + MIN_FREE_AFTER_BYTES - free;
    assert_eq!(open_memory_problem(planned, free, total, need, HOST), None);
    assert!(open_memory_problem(planned, free, total, need - 1, HOST).is_some());
}

#[test]
fn a_huge_count_of_freed_bytes_does_not_wrap() {
    assert_eq!(open_memory_problem(40 * GIB, 30 * GIB, 121 * GIB, u64::MAX, HOST), None);
    assert!(open_memory_problem(u64::MAX, 30 * GIB, 121 * GIB, 0, HOST).is_some());
}

#[test]
fn the_message_names_each_number_and_what_to_do() {
    let message = open_memory_problem(20_000_000_000, 7_000_000_000, 121_000_000_000, 3_000_000_000, HOST).unwrap();
    for number in [
        "20000000000",
        "7000000000",
        "121000000000",
        "3000000000",
        &MIN_FREE_AFTER_BYTES.to_string(),
    ] {
        assert!(message.contains(number), "{number} is not in: {message}");
    }
    assert!(message.contains("MemAvailable of the host"), "{message}");
    let gpu = open_memory_problem(1, 0, 2, 0, Count::GpuFree).unwrap();
    assert!(gpu.contains("free bytes of the GPU"), "{gpu}");
    assert!(message.contains("`slots`") && message.contains("`unload`"), "{message}");
}
