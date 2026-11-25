//! GPU gates of the state rules of the engine:
//!
//! - The limits of the engine are the plan of the factory, and an open refuses slots that do not fit the memory.
//! - A refused pass changes nothing. A poisoned slot refuses each call that changes it until a clear, and the other
//!   slots stay usable. A clear then a pass give the bits of a fresh slot.
//! - A slot keeps its count of cycles, and another count on a slot that holds positions is refused.
//!
//! `INKLING2_MODEL_DIR=<folder> INKLING2_GOLDEN_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p
//!  nagaya-mk-inkling2-sm121 --features cuda --test cuda_engine_state -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod engine_support;

use std::time::Duration;

use nagaya_abi::{Engine, PartError, PassErrorKind, PassStatus, State, WriteResult};
use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_cuda::test_cards;
use nagaya_mk_inkling2_sm121::engine::Inkling2Sm121Engine;
use nagaya_mk_inkling2_sm121::spec;

use engine_support::{chat_ids, config, ids_of, logit_bits, model_dir, open, segment, watchdog};

/// Every stored part of `slot`, whole: the rows of a row part from its base, and a fixed part.
fn read_all(engine: &mut Inkling2Sm121Engine, slot: u32) -> Vec<Vec<u8>> {
    let limits = engine.limits();
    let high = engine.slots().unwrap().slot_high(slot);
    let mut out = Vec::new();
    for (id, part) in limits.parts.iter().enumerate() {
        let row = part.shape.iter().product::<u64>() * part.dtype.element_bytes();
        let len = if part.is_row() {
            let start = engine.coverage().unwrap().coverage(State::Slot(slot), id, &[]).start;
            (high - start) * row
        } else {
            row
        };
        let mut bytes = vec![0u8; len as usize];
        engine
            .parts()
            .unwrap()
            .part_read(State::Slot(slot), id, &mut bytes)
            .unwrap_or_else(|e| panic!("part {} of slot {slot}: {e:?}", part.name));
        out.push(bytes);
    }
    out
}

#[test]
fn the_limits_are_the_plan_and_an_open_refuses_slots_that_do_not_fit() {
    let Some(dir) = model_dir("cuda_engine_state") else {
        return;
    };
    let _guard = watchdog(Duration::from_secs(1800), "the limits gate");
    let slots = [1024, 2048];
    let cfg = config(&dir, &slots, CacheFormat::F32);
    let engine = open(cfg.clone());
    let plan = spec::plan(&cfg).unwrap();
    assert_eq!(engine.limits(), plan, "the limits of the engine are not the plan");
    let identity = engine.identity();
    assert_eq!(identity.numbers["cache"], "f32");
    assert!(identity.checkpoint_digest.starts_with("blake3:"));
    assert_eq!(identity.bit_settings, spec::bit_settings(&cfg));
    assert_eq!(
        identity.numbers["cubin_blake3"],
        spec::identity_numbers(&cfg)["cubin_blake3"]
    );
    let planned_weights = plan
        .memory
        .mandatory
        .iter()
        .find(|t| t.name == "weights")
        .unwrap()
        .bytes;
    println!(
        "open: the digest took {} ms; the weights take {} device bytes, and the plan counts {planned_weights}",
        engine.digest_millis(),
        engine.weights_bytes()
    );
    assert!(engine.weights_bytes() <= planned_weights, "the weights pass the plan");
    drop(engine);

    // Four slots of the longest context hold more bytes than the GPU has. The open refuses before it reads the weights.
    let big = config(&dir, &[spec::MAX_SLOT_TOKENS; 4], CacheFormat::F32);
    let started = std::time::Instant::now();
    let error = match Inkling2Sm121Engine::open(big, &test_cards::card()) {
        Ok(_) => panic!("the open must refuse slots that do not fit the memory"),
        Err(error) => error,
    };
    assert!(
        error.contains("bytes of GPU memory") && error.contains("MemAvailable of the host"),
        "{error}"
    );
    assert!(error.contains("`slots`") && error.contains("`cache`"), "{error}");
    assert!(
        started.elapsed() < Duration::from_secs(60),
        "the refusal must come before the load of the weights: {:?}",
        started.elapsed()
    );
    // The refusal freed what it took: the next open works.
    drop(open(cfg));
}

#[test]
fn a_refused_pass_changes_nothing_and_a_poisoned_slot_refuses_until_a_clear() {
    let Some(dir) = model_dir("cuda_engine_state") else {
        return;
    };
    let _guard = watchdog(Duration::from_secs(1800), "the poison gate");
    let mut engine = open(config(&dir, &[1024, 1024], CacheFormat::F32));
    let chat = chat_ids();
    let first = ids_of(&chat, 0, 0, 17);
    let next = ids_of(&chat, 0, 17, 1);
    let fresh = engine
        .pass(&[segment(0, 0, &first, vec![16], None)], &[])
        .unwrap_or_else(|e| panic!("{}", e.message));
    let fresh_bits = logit_bits(&fresh);
    let before = read_all(&mut engine, 0);
    let live = engine.live_buffers();

    // Each refusal leaves the parts, the count and the buffers as they were.
    let refusals = [
        ("a position that does not follow", segment(0, 3, &next, vec![3], None)),
        (
            "a token past the vocabulary",
            segment(0, 17, &[spec::LOGITS as u32], vec![17], None),
        ),
        (
            "logit rows that do not rise",
            segment(0, 17, &first[..4], vec![18, 18], None),
        ),
        (
            "logit rows past a span of 16",
            segment(0, 17, &ids_of(&chat, 0, 17, 40), vec![17, 40], None),
        ),
        ("another count of cycles", segment(0, 17, &next, vec![17], Some(2))),
        ("a slot that is not there", segment(2, 0, &next, vec![0], None)),
    ];
    for (what, bad) in &refusals {
        let error = engine.pass(std::slice::from_ref(bad), &[]).expect_err(what);
        assert_eq!(error.kind, PassErrorKind::Refused, "{what}: {}", error.message);
        assert_eq!(engine.last_error().as_deref(), Some(error.message.as_str()), "{what}");
        assert_eq!(engine.slots().unwrap().slot_high(0), 17, "{what}");
        assert_eq!(
            read_all(&mut engine, 0),
            before,
            "{what}: a refused pass changed the parts"
        );
        assert_eq!(engine.live_buffers(), live, "{what}");
    }

    // A failed launch poisons the slot. The test sets the same state that the pass sets after a failed launch.
    engine.poison_for_test(0);
    let step = segment(0, 17, &next, vec![17], None);
    let error = engine
        .pass(std::slice::from_ref(&step), &[])
        .expect_err("a poisoned slot refuses a pass");
    assert_eq!(error.kind, PassErrorKind::Refused);
    assert!(
        error.message.contains("poisoned") && error.message.contains("slot_clear"),
        "{}",
        error.message
    );
    let limits = engine.limits();
    let conv = limits.parts.iter().position(|p| p.name == "conv.00").unwrap();
    let key = limits.parts.iter().position(|p| p.name == "k.00").unwrap();
    let mut conv_bytes = vec![0u8; 3 * 6656 * 4];
    engine
        .parts()
        .unwrap()
        .part_read(State::Slot(0), conv, &mut conv_bytes)
        .unwrap();
    assert_eq!(
        engine.parts().unwrap().part_write(State::Slot(0), conv, &conv_bytes),
        WriteResult::Refused
    );
    assert!(engine.last_error().unwrap().contains("poisoned"));
    let row = 8 * 128 * 4;
    assert_eq!(
        engine
            .parts()
            .unwrap()
            .cells_write(State::Slot(0), key, 0, 1, &vec![0u8; row]),
        WriteResult::Refused
    );
    assert_eq!(engine.parts().unwrap().set_n(State::Slot(0), 5), WriteResult::Refused);
    // A read still works, so a caller can look at the poisoned state, and the other slot is not touched.
    assert_eq!(read_all(&mut engine, 0), before);
    let other = engine
        .pass(&[segment(1, 0, &first, vec![16], None)], &[])
        .unwrap_or_else(|e| panic!("{}", e.message));
    assert_eq!(logit_bits(&other), fresh_bits, "slot 1 differs from a fresh slot 0");
    let error = engine.pass(&[step], &[]).expect_err("the slot is still poisoned");
    assert_eq!(error.kind, PassErrorKind::Refused);

    // A clear takes the poison and empties the slot. A pass then gives the bits of a fresh slot.
    assert_eq!(engine.slots().unwrap().slot_clear(0), WriteResult::Ok);
    assert_eq!(engine.slots().unwrap().slot_high(0), 0);
    let again = engine
        .pass(&[segment(0, 0, &first, vec![16], None)], &[])
        .unwrap_or_else(|e| panic!("{}", e.message));
    assert_eq!(again.status, PassStatus::Ok);
    assert_eq!(
        logit_bits(&again),
        fresh_bits,
        "a pass after a clear differs from a fresh slot"
    );
    assert_eq!(
        read_all(&mut engine, 0),
        before,
        "the parts after a clear differ from a fresh slot"
    );
    // A conv state that a restore wrote with no count does not reach a pass at position 0: the pass starts from zeros.
    let junk: Vec<u8> = (0..3 * 6656)
        .flat_map(|i| (((i % 13) as f32) * 0.5 - 3.0).to_le_bytes())
        .collect();
    assert_eq!(
        engine.parts().unwrap().part_write(State::Slot(0), conv, &junk),
        WriteResult::Ok
    );
    let from_junk = engine
        .pass(&[segment(0, 0, &first, vec![16], None)], &[])
        .unwrap_or_else(|e| panic!("{}", e.message));
    assert_eq!(
        logit_bits(&from_junk),
        fresh_bits,
        "a written conv state reached a pass at position 0"
    );
    // A part read of a part that does not exist is refused, and it is not a failure.
    let mut bytes = [0u8; 4];
    assert_eq!(
        engine.parts().unwrap().part_read(State::Slot(0), 10_000, &mut bytes),
        Err(PartError::Refused)
    );
}

#[test]
fn a_slot_keeps_its_count_of_cycles() {
    let Some(dir) = model_dir("cuda_engine_state") else {
        return;
    };
    let _guard = watchdog(Duration::from_secs(1800), "the cycles gate");
    let mut engine = open(config(&dir, &[1024, 1024], CacheFormat::F32));
    let chat = chat_ids();
    let ids = ids_of(&chat, 1, 0, 17);
    let cycles_part = engine.limits().parts.len() - 1;
    let count = |engine: &mut Inkling2Sm121Engine, slot: u32| {
        let mut bytes = [0u8; 4];
        engine
            .parts()
            .unwrap()
            .part_read(State::Slot(slot), cycles_part, &mut bytes)
            .unwrap();
        u32::from_le_bytes(bytes)
    };
    assert_eq!(
        (count(&mut engine, 0), count(&mut engine, 1)),
        (0, 0),
        "an empty slot has no count"
    );
    // Slot 0 runs the default count, 3. Slot 1 runs 2.
    engine.pass(&[segment(0, 0, &ids, vec![16], None)], &[]).unwrap();
    engine.pass(&[segment(1, 0, &ids, vec![16], Some(2))], &[]).unwrap();
    assert_eq!((count(&mut engine, 0), count(&mut engine, 1)), (3, 2));
    let next = ids_of(&chat, 1, 17, 1);
    for (slot, wrong) in [(0u32, 2u8), (1, 3)] {
        let error = engine
            .pass(&[segment(slot, 17, &next, vec![17], Some(wrong))], &[])
            .expect_err("another count on a slot that holds positions");
        assert_eq!(error.kind, PassErrorKind::Refused);
        assert!(error.message.contains("cycles"), "{}", error.message);
        assert_eq!(engine.slots().unwrap().slot_high(slot), 17);
    }
    assert_eq!((count(&mut engine, 0), count(&mut engine, 1)), (3, 2));
    // The kept count passes, with or without a named count.
    engine.pass(&[segment(0, 17, &next, vec![17], Some(3))], &[]).unwrap();
    engine.pass(&[segment(1, 17, &next, vec![17], Some(2))], &[]).unwrap();
    // A clear frees the slot for another count.
    assert_eq!(engine.slots().unwrap().slot_clear(0), WriteResult::Ok);
    assert_eq!(count(&mut engine, 0), 0);
    engine.pass(&[segment(0, 0, &ids, vec![16], Some(2))], &[]).unwrap();
    assert_eq!(count(&mut engine, 0), 2);
}
