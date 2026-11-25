//! GPU gate of the part calls of the engine: a slot that a restore wrote into another slot
//! through the engine gives the bits of the slot that was read, in each cache format. A slot of 2,048 positions fed
//! with 1,600 tokens holds a wrapped ring, so the restore covers the start of the rings.
//!
//! `INKLING2_MODEL_DIR=<folder> INKLING2_GOLDEN_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p
//!  nagaya-mk-inkling2-sm121 --features cuda --test cuda_engine_parts -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod engine_support;

use std::time::Duration;

use nagaya_abi::{Engine, State, WriteResult};
use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_inkling2_sm121::spec;

use engine_support::{chat_ids, config, ids_of, logit_bits, model_dir, open, segment, watchdog};

const FED: usize = 1600;

#[test]
fn a_restored_slot_gives_the_bits_of_the_slot_that_was_read_in_each_format() {
    let Some(dir) = model_dir("cuda_engine_parts") else {
        return;
    };
    let _guard = watchdog(Duration::from_secs(3600), "the parts gate");
    let chat = chat_ids();
    for format in [CacheFormat::F32, CacheFormat::Bf16, CacheFormat::E4m3] {
        let mut engine = open(config(&dir, &[2048, 2048], format));
        let limits = engine.limits();
        assert_eq!(limits.parts[0].dtype, spec::part_dtype(format));
        let mut at = 0;
        while at < FED {
            let rows = (FED - at).min(512);
            engine
                .pass(
                    &[segment(0, at as u64, &ids_of(&chat, 3, at, rows), Vec::new(), None)],
                    &[],
                )
                .unwrap_or_else(|e| panic!("{}", e.message));
            at += rows;
        }
        // Read every part of slot 0.
        let mut parts = Vec::new();
        for (id, part) in limits.parts.iter().enumerate() {
            let row = (part.shape.iter().product::<u64>() * part.dtype.element_bytes()) as usize;
            let (start, len) = if part.is_row() {
                let start = engine.coverage().unwrap().coverage(State::Slot(0), id, &[]).start;
                (start, (FED as u64 - start) as usize * row)
            } else {
                (0, row)
            };
            let mut bytes = vec![0u8; len];
            engine
                .parts()
                .unwrap()
                .part_read(State::Slot(0), id, &mut bytes)
                .unwrap();
            parts.push((id, start, bytes));
        }
        assert!(
            parts.iter().any(|(_, start, _)| *start > 0),
            "{format:?}: no ring wrapped"
        );
        // A value that the format cannot hold is refused, and the refusal changes nothing.
        let (key, _, key_bytes) = parts
            .iter()
            .find(|(id, start, _)| limits.parts[*id].is_row() && *start == 0)
            .unwrap();
        let mut bad = key_bytes[..8 * 128 * format_bytes(format)].to_vec();
        bad[..format_bytes(format)].fill(0xff);
        assert_eq!(
            engine.parts().unwrap().cells_write(State::Slot(1), *key, 0, 1, &bad),
            WriteResult::Refused,
            "{format:?}"
        );
        assert!(engine.last_error().unwrap().contains("NaN or an infinity"));
        // Restore into slot 1: each part, then the count.
        for (id, start, bytes) in &parts {
            let part = &limits.parts[*id];
            let result = if part.is_row() {
                let row = (part.shape.iter().product::<u64>() * part.dtype.element_bytes()) as usize;
                engine
                    .parts()
                    .unwrap()
                    .cells_write(State::Slot(1), *id, *start, (bytes.len() / row) as u64, bytes)
            } else {
                engine.parts().unwrap().part_write(State::Slot(1), *id, bytes)
            };
            assert_eq!(
                result,
                WriteResult::Ok,
                "{format:?}: part {}: {:?}",
                part.name,
                engine.last_error()
            );
        }
        assert_eq!(
            engine.parts().unwrap().set_n(State::Slot(1), FED as u64),
            WriteResult::Ok
        );
        assert_eq!(engine.slots().unwrap().slot_high(1), FED as u64);
        // The two slots continue with the same ids and give the same bits.
        let next = ids_of(&chat, 3, FED, 16);
        let rows = vec![FED as u64 + 15];
        let a = engine
            .pass(&[segment(0, FED as u64, &next, rows.clone(), None)], &[])
            .unwrap();
        let b = engine.pass(&[segment(1, FED as u64, &next, rows, None)], &[]).unwrap();
        assert_eq!(logit_bits(&a), logit_bits(&b), "{format:?}: the restored slot differs");
    }
}

fn format_bytes(format: CacheFormat) -> usize {
    format.row_bytes() / 128
}
