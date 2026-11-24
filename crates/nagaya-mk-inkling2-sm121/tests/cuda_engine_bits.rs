//! GPU gate of the bits of the engine (milestone 37 step 11): the engine adds no numeric step. A pass through the engine
//! gives the logits that `Pass::run_slot` gives for the same ids, for each cache format.
//!
//! `INKLING2_MODEL_DIR=<folder> INKLING2_GOLDEN_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p
//!  nagaya-mk-inkling2-sm121 --features cuda --test cuda_engine_bits -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod engine_support;

use std::time::Duration;

use nagaya_abi::Engine;
use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::test_cards;
use nagaya_mk_inkling2_sm121::pass::step::{Pass, PassConfig};
use nagaya_mk_inkling2_sm121::pass::weights::DeviceWeights;
use nagaya_mk_inkling2_sm121::spec::{self, LOGITS};

use engine_support::{chat_ids, config, ids_of, logit_bits, model_dir, open, segment, watchdog};

const SEQ: usize = 1024;

#[test]
fn a_pass_through_the_engine_gives_the_bits_of_the_pass() {
    let Some(dir) = model_dir("cuda_engine_bits") else {
        return;
    };
    let _guard = watchdog(Duration::from_secs(3600), "the bits gate");
    let chat = chat_ids();
    // The calls: a block with logits for three rows inside a span of 16, one row, one row with no logits, and a block of 40 rows.
    let calls: [(usize, usize, Vec<usize>); 4] = [
        (0, 17, vec![1, 5, 16]),
        (17, 1, vec![0]),
        (18, 1, vec![]),
        (19, 40, vec![39]),
    ];
    for format in [CacheFormat::F32, CacheFormat::Bf16, CacheFormat::E4m3] {
        let dev = CudaDevice::open_by_pci(&test_cards::card(), format.cubin()).unwrap();
        let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap();
        let mut pass = Pass::open(
            &dev,
            &weights,
            &PassConfig {
                seqs: vec![SEQ],
                max_rows: spec::DEFAULT_MAX_BLOCK,
                max_cycles: 3,
                profile: false,
                format,
            },
        )
        .unwrap();
        let mut engine = open(config(&dir, &[SEQ], format));
        for (at, rows, logit_rows) in &calls {
            let ids = ids_of(&chat, 2, *at, *rows);
            let want = pass.run_slot(0, 3, &ids, logit_rows).unwrap();
            let positions: Vec<u64> = logit_rows.iter().map(|r| (*at + r) as u64).collect();
            let got = engine
                .pass(&[segment(0, *at as u64, &ids, positions, None)], &[])
                .unwrap_or_else(|e| panic!("{}", e.message));
            let got = logit_bits(&got);
            assert_eq!(got.len(), logit_rows.len(), "{format:?} at {at}");
            for (row, bits) in got.iter().enumerate() {
                let want_bits: Vec<u32> = want[row * LOGITS..(row + 1) * LOGITS]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                assert_eq!(
                    bits, &want_bits,
                    "{format:?}: the engine differs from the pass at {at}, row {row}"
                );
            }
        }
    }
}
