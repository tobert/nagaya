//! GPU gate of the close of the engine (milestone 37 step 11): an open, a pass and a drop, three times. After each drop
//! the engine has freed every device buffer, and the free memory of the GPU is what it was before the open.
//!
//! `INKLING2_MODEL_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-inkling2-sm121 --features cuda
//!  --test cuda_engine_close -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod engine_support;

use std::time::Duration;

use nagaya_abi::Engine;
use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::{INKLING2_CUBIN, test_cards};

use engine_support::{chat_ids, config, ids_of, model_dir, open, segment, watchdog};

/// On a GPU with unified memory the page cache of the 19 GB checkpoint read moves the free bytes by some gigabytes, so
/// the free bytes after a drop do not equal the free bytes before the open. The gate asserts what a leak breaks. After
/// each drop the free bytes are within `DRIFT` of the free bytes before the first open. And they do not fall from one
/// round to the next by more than `ROUND_DRIFT`. An open takes about 20 GB, so a leak of one engine fails both. The
/// count of live buffers is the hard rule: it proves that the engine freed each allocation.
const DRIFT: usize = 4 << 30;
const ROUND_DRIFT: usize = 1 << 30;

#[test]
fn three_opens_and_drops_free_every_buffer_and_all_the_memory() {
    let Some(dir) = model_dir("cuda_engine_close") else {
        return;
    };
    let _guard = watchdog(Duration::from_secs(1800), "the close gate");
    let probe = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the probe device");
    let (before, _) = probe.memory_info().unwrap();
    let chat = chat_ids();
    let mut last_after: Option<usize> = None;
    for round in 0..3 {
        let mut engine = open(config(&dir, &[1024, 1024], CacheFormat::F32));
        assert!(engine.live_buffers() > 0, "an open engine holds buffers");
        let (during, _) = probe.memory_info().unwrap();
        assert!(
            during + (10 << 30) < before,
            "the open engine must hold memory: {before} free before, {during} during (round {round})"
        );
        let out = engine
            .pass(&[segment(0, 0, &ids_of(&chat, 0, 0, 17), vec![16], None)], &[])
            .unwrap_or_else(|e| panic!("{}", e.message));
        assert_eq!(out.outputs.len(), 1);
        if round < 2 {
            // `close` gives the count of buffers that the engine left behind.
            assert_eq!(engine.close(), 0, "round {round}: the close left device buffers");
        } else {
            // The drop panics when a buffer is left, so a plain drop that returns is a pass.
            drop(engine);
        }
        let (after, _) = probe.memory_info().unwrap();
        assert!(
            after + DRIFT >= before,
            "round {round}: {before} bytes were free before the first open, and {after} are free after the drop"
        );
        if let Some(last) = last_after {
            assert!(
                after + ROUND_DRIFT >= last,
                "round {round}: the free bytes fell from {last} to {after} over one round"
            );
        }
        last_after = Some(after);
        assert_eq!(probe.live_buffers(), 0);
    }
}
