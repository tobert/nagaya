//! GPU gate of `CudaDevice::memory_info`: the driver reports the free and the total bytes, and the
//! free bytes fall by the bytes of an allocation that the GPU touched, and rise again when it is freed.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-cuda --features cuda --test cuda_mem_info -- --test-threads=1`

#![cfg(feature = "cuda")]

use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::{INKLING2_CUBIN, test_cards};

const ALLOCATION: usize = 2 << 30;
/// Other work on the machine moves the free bytes a little. The gate allows this much in each direction.
const SLACK: usize = 512 << 20;

#[test]
fn the_free_bytes_fall_with_an_allocation_and_rise_when_it_is_freed() {
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU");
    let (before, total) = dev.memory_info().expect("memory_info");
    assert!(total > 0 && before <= total, "free {before} of total {total}");
    let ptr = dev.alloc_device(ALLOCATION).expect("allocate");
    dev.memset_zero(ptr, ALLOCATION).expect("touch");
    let (during, _) = dev.memory_info().expect("memory_info");
    assert!(
        during + ALLOCATION <= before + SLACK,
        "free bytes fell from {before} to {during} for an allocation of {ALLOCATION}"
    );
    dev.free_device(ptr).expect("free");
    let (after, _) = dev.memory_info().expect("memory_info");
    assert!(
        after + SLACK >= before,
        "free bytes were {before} before and are {after} after the free"
    );
    assert_eq!(dev.live_buffers(), 0);
}
