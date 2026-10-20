//! The mk engine on gfx1151: the binding calls, the write rules, one pass, and the conformance suite.
//!
//! Every test calls `open()`. The first call loads the weights; every later call makes a new context pool on the same
//! engine. All mk calls go to one worker thread. Run with `--test-threads=1`.

#![cfg(feature = "hip")]

use nagaya_abi::{
    Engine, OutputsWanted, PartError, PassErrorKind, SealError, Segment, SegmentInputs, State, Tap, WriteResult,
};
use nagaya_conformance::{compare_f32_rows, run_all};
use nagaya_mk_qwen38_gfx1151::open;

/// A finite f32 pattern as bytes.
fn finite_bytes(len: usize, seed: f32) -> Vec<u8> {
    (0..len / 4)
        .flat_map(|index| (((index % 97) as f32) * 0.25 - 7.0 + seed).to_le_bytes())
        .collect()
}

/// Read bytes as f32 values.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// The row length of a part in f32 values, for `compare_f32_rows`.
fn row_len(part: usize) -> usize {
    match part {
        0 => 128 * 128,
        1 | 2 => 10_240,
        3 => 2 * 2 * 256,
        4 => 128,
        _ => 1,
    }
}

#[test]
fn the_binding_opens_mk_and_declares_its_parts() {
    let engine = open().expect("open mk");
    let limits = engine.limits();
    assert_eq!(limits.parts.len(), 7);
    assert_eq!(limits.parts[0].name, "gdn_state");
    assert_eq!(limits.parts[3].name, "kv");
    assert_eq!(limits.parts[5].name, "ple_hist");
    assert_eq!(limits.parts[6].name, "idx_pooled");
    assert_eq!(limits.slots.len(), 1);
    assert_eq!(limits.slots[0].count, 16);
    assert_eq!(limits.max_segments, 16);
    assert_eq!(limits.max_pass_inputs, 256);
    let weights = limits
        .memory
        .mandatory
        .iter()
        .find(|term| term.name == "weights")
        .expect("a weights term");
    assert!(weights.bytes > 0, "mk_engine_arena gave no arena bytes");
    let context = limits
        .memory
        .mandatory
        .iter()
        .find(|term| term.name == "context")
        .expect("a context term");
    assert_eq!(context.bytes, 16 * 132_587_520);
    // The pass scratch term holds the device cost that the probe measured at open.
    let scratch = limits
        .memory
        .mandatory
        .iter()
        .find(|term| term.name == "pass_scratch")
        .expect("a pass_scratch term");
    assert!(engine.pass_scratch() > 0, "the probe measured no pass scratch");
    assert!(
        engine.pass_scratch() <= scratch.bytes,
        "measured {} bytes, declared {}",
        engine.pass_scratch(),
        scratch.bytes
    );

    // mk's own byte counts agree with the declared shapes.
    assert_eq!(engine.part_bytes(0, 0).expect("part 0"), 113_246_208);
    assert_eq!(engine.part_bytes(1, 0).expect("part 1"), 4_423_680);
    assert_eq!(engine.part_bytes(2, 0).expect("part 2"), 368_640);
    assert_eq!(engine.part_bytes(3, 5).expect("part 3"), 5 * 49_152);
    assert_eq!(engine.part_bytes(4, 5).expect("part 4"), 5 * 6_144);
    assert_eq!(engine.part_bytes(5, 0).expect("part 5"), 8);
    // Part 6 is derived: mk_ctx_part_bytes would stop the program, so the binding refuses it first.
    assert!(engine.part_bytes(6, 0).is_err());

    let identity = engine.identity();
    assert_eq!(identity.model, "Qwen3.8-Flash-Next-UD-IQ4_XS");
    assert!(
        identity.checkpoint_digest.starts_with("spot-sha256:"),
        "{}",
        identity.checkpoint_digest
    );
    // The identity holds the hash of the tokenizer keys in the first shard, as the text library computes it.
    let shards = std::env::var("MK_SHARDS").expect("MK_SHARDS");
    let first = shards.split(':').next().expect("a first shard");
    let metadata = nagaya_text_qwen38::gguf::Metadata::open(std::path::Path::new(first)).expect("the first shard");
    assert_eq!(
        identity.tokenizer_hash,
        nagaya_text_qwen38::tokenizer_hash(&metadata).expect("a tokenizer hash")
    );
    assert_eq!(
        identity.tokenizer_hash,
        "sha256:9724dd72094b6311776a1f5a8a49d283f676a5f0f56d8e0ba9671adb18df06ad"
    );
    // The template is the text of `tokenizer.chat_template` in the first shard.
    match metadata.get("tokenizer.chat_template") {
        Some(nagaya_text_qwen38::gguf::Value::Str(text)) => {
            assert!(!text.is_empty());
            assert_eq!(&identity.template, text);
        }
        other => panic!("the first shard has no string chat template: {other:?}"),
    }
    assert_eq!(identity.canary_hash, "");
    assert_eq!(identity.numbers.get("fused").map(String::as_str), Some("1"));
    assert_eq!(identity.numbers.get("fast_qmm").map(String::as_str), Some("1"));

    assert_eq!(engine.n(State::Slot(0)).unwrap(), 0);
    assert_eq!(engine.capacity(State::Slot(0)).unwrap(), 256);
    assert!(engine.owns_kv(State::Slot(0)).unwrap());
}

#[test]
fn a_fixed_part_round_trips() {
    let mut engine = open().expect("open mk");
    let limits = engine.limits();
    let state = State::Slot(0);
    for part in [0usize, 1, 2] {
        let bytes = finite_bytes(
            limits.parts[part].shape.iter().product::<u64>() as usize * 4,
            part as f32,
        );
        assert_eq!(
            engine.parts().unwrap().part_write(state, part, &bytes),
            WriteResult::Ok,
            "part {part}"
        );
        let mut out = vec![0u8; bytes.len()];
        engine.parts().unwrap().part_read(state, part, &mut out).unwrap();
        let expected = floats(&bytes);
        let actual = floats(&out);
        compare_f32_rows("a_fixed_part_round_trips", &expected, &actual, row_len(part)).unwrap();
    }
}

#[test]
fn a_cell_part_round_trips_and_sets_n() {
    let mut engine = open().expect("open mk");
    let state = State::Slot(0);
    let kv = finite_bytes(2 * 49_152, 1.0);
    let keys = finite_bytes(2 * 6_144, 2.0);

    assert_eq!(
        engine.parts().unwrap().cells_write(state, 3, 0, 2, &kv),
        WriteResult::Ok
    );
    assert_eq!(
        engine.parts().unwrap().cells_write(state, 4, 0, 2, &keys),
        WriteResult::Ok
    );
    // A failed set_n leaves the context empty.
    assert_eq!(engine.parts().unwrap().set_n(state, 2), WriteResult::Refused);
    assert_eq!(engine.n(state).unwrap(), 0);
    assert_eq!(
        engine.parts().unwrap().part_write(state, 5, &[7, 0, 0, 0, 9, 0, 0, 0]),
        WriteResult::Ok
    );
    assert_eq!(engine.parts().unwrap().set_n(state, 2), WriteResult::Ok);
    assert_eq!(engine.n(state).unwrap(), 2);

    let mut out = vec![0u8; kv.len()];
    engine.parts().unwrap().part_read(state, 3, &mut out).unwrap();
    compare_f32_rows(
        "a_cell_part_round_trips_and_sets_n",
        &floats(&kv),
        &floats(&out),
        row_len(3),
    )
    .unwrap();
    let mut out = vec![0u8; keys.len()];
    engine.parts().unwrap().part_read(state, 4, &mut out).unwrap();
    compare_f32_rows(
        "a_cell_part_round_trips_and_sets_n",
        &floats(&keys),
        &floats(&out),
        row_len(4),
    )
    .unwrap();
    let mut history = vec![0u8; 8];
    engine.parts().unwrap().part_read(state, 5, &mut history).unwrap();
    assert_eq!(history, [7, 0, 0, 0, 9, 0, 0, 0]);

    // The write rules refuse what mk stops on.
    let mut short = vec![0u8; 49_152];
    assert_eq!(
        engine.parts().unwrap().part_read(state, 3, &mut short),
        Err(PartError::Refused)
    );
    assert_eq!(
        engine.parts().unwrap().cells_read(state, 3, 0, 1, &mut short),
        Err(PartError::Refused)
    );
    assert_eq!(
        engine.parts().unwrap().part_write(state, 3, &kv[..49_152]),
        WriteResult::Refused
    );
    assert_eq!(engine.parts().unwrap().set_n(state, 3), WriteResult::Refused);
}

#[test]
fn a_ple_history_must_hold_token_ids() {
    let mut engine = open().expect("open mk");
    let state = State::Slot(0);
    let mut before = vec![0u8; 8];
    engine.parts().unwrap().part_read(state, 5, &mut before).unwrap();
    // 248320 is one past the last token id.
    let invalid = 248_320u32.to_le_bytes();
    let mut bytes = [0u8; 8];
    bytes[..4].copy_from_slice(&invalid);
    assert_eq!(
        engine.parts().unwrap().part_write(state, 5, &bytes),
        WriteResult::Refused
    );
    assert!(engine.last_error().unwrap_or_default().contains("not a token id"));
    let mut after = vec![0u8; 8];
    engine.parts().unwrap().part_read(state, 5, &mut after).unwrap();
    assert_eq!(before, after);

    let negative = (-1i32).to_le_bytes();
    let mut bytes = [0u8; 8];
    bytes[..4].copy_from_slice(&negative);
    assert_eq!(
        engine.parts().unwrap().part_write(state, 5, &bytes),
        WriteResult::Refused
    );

    assert_eq!(
        engine.parts().unwrap().part_write(state, 5, &[7, 0, 0, 0, 9, 0, 0, 0]),
        WriteResult::Ok
    );
    let mut after = vec![0u8; 8];
    engine.parts().unwrap().part_read(state, 5, &mut after).unwrap();
    assert_eq!(after, [7, 0, 0, 0, 9, 0, 0, 0]);
}

#[test]
fn a_seal_rounds_the_state_and_refuses_a_non_finite_one() {
    let mut engine = open().expect("open mk");
    let state = State::Slot(0);
    let one = 1f32.to_le_bytes();
    let ones: Vec<u8> = one.iter().copied().cycle().take(113_246_208).collect();
    assert_eq!(engine.parts().unwrap().part_write(state, 0, &ones), WriteResult::Ok);
    Engine::seal(&mut engine)
        .unwrap()
        .seal(state)
        .expect("seal a finite state");
    let mut out = vec![0u8; ones.len()];
    engine.parts().unwrap().part_read(state, 0, &mut out).unwrap();
    for chunk in out.as_chunks::<4>().0 {
        let bits = u32::from_le_bytes(*chunk);
        assert_eq!(bits & 0xffff, 0, "the low 16 bits are not zero");
        assert_eq!(f32::from_bits(bits), 1.0);
    }

    let nan = f32::NAN.to_le_bytes();
    let nans: Vec<u8> = nan.iter().copied().cycle().take(113_246_208).collect();
    assert_eq!(engine.parts().unwrap().part_write(state, 0, &nans), WriteResult::Ok);
    assert_eq!(Engine::seal(&mut engine).unwrap().seal(state), Err(SealError::Refused));
    assert!(engine.last_error().unwrap_or_default().contains("not finite"));
    assert_eq!(engine.parts().unwrap().part_write(state, 0, &ones), WriteResult::Ok);
}

fn segment(state: State, tokens: &[u32], start: u64) -> Segment {
    Segment {
        state,
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (start..start + tokens.len() as u64).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted::default(),
    }
}

#[test]
fn a_pass_runs_and_refuses_a_wrong_position() {
    let mut engine = open().expect("open mk");
    let state = State::Slot(0);
    let mut wanted = segment(state, &[11, 12, 13], 0);
    wanted.outputs.logit_rows = vec![2];
    let out = engine.pass(std::slice::from_ref(&wanted), &[]).expect("a pass");
    assert_eq!(engine.n(state).unwrap(), 3);
    assert_eq!(out.outputs.len(), 1);
    assert_eq!(out.outputs[0].name, "logits");
    assert_eq!(out.outputs[0].bytes.len(), 248_320 * 4);
    assert!(floats(&out.outputs[0].bytes).iter().all(|value| value.is_finite()));

    // The same pass in two more contexts gives the same bits.
    let mut second = open().expect("open mk");
    let mut third = open().expect("open mk");
    let mut fresh = segment(State::Slot(0), &[11, 12, 13], 0);
    fresh.outputs.logit_rows = vec![2];
    let other = second.pass(std::slice::from_ref(&fresh), &[]).expect("second pass");
    let again = third.pass(std::slice::from_ref(&fresh), &[]).expect("third pass");
    compare_f32_rows(
        "a_pass_runs_and_refuses_a_wrong_position",
        &floats(&out.outputs[0].bytes),
        &floats(&other.outputs[0].bytes),
        248_320,
    )
    .unwrap();
    compare_f32_rows(
        "a_pass_runs_and_refuses_a_wrong_position",
        &floats(&other.outputs[0].bytes),
        &floats(&again.outputs[0].bytes),
        248_320,
    )
    .unwrap();
    drop(second);
    drop(third);

    // A position that is not the context's n is refused, and the context stays as it was.
    let empty = State::Slot(1);
    let mut before = vec![0u8; 8];
    engine.parts().unwrap().part_read(empty, 5, &mut before).unwrap();
    let wrong = segment(empty, &[11, 12, 13], 1);
    let error = engine.pass(std::slice::from_ref(&wrong), &[]).unwrap_err();
    assert_eq!(error.kind, PassErrorKind::Refused, "{:?}", error.kind);
    assert!(error.message.contains("appends at 0"), "{}", error.message);
    assert_eq!(engine.n(empty).unwrap(), 0);
    let mut after = vec![0u8; 8];
    engine.parts().unwrap().part_read(empty, 5, &mut after).unwrap();
    assert_eq!(before, after, "a refused pass changed the state");

    // Rows, metadata and taps are not part of this binding.
    let mut rows = segment(empty, &[1], 0);
    rows.inputs = SegmentInputs::Rows(vec![0; 4]);
    assert!(engine.pass(std::slice::from_ref(&rows), &[]).is_err());
    let mut meta = segment(empty, &[1], 0);
    meta.meta = vec![1];
    assert!(engine.pass(std::slice::from_ref(&meta), &[]).is_err());
    let tap = Tap {
        name: "layers".to_owned(),
    };
    assert!(
        engine
            .pass(std::slice::from_ref(&segment(empty, &[1], 0)), &[tap])
            .is_err()
    );

    // Two segments on one context are a segments problem, and the wrapper reports it without a die.
    let same = vec![segment(empty, &[1], 0), segment(empty, &[2], 0)];
    let error = engine.pass(&same, &[]).unwrap_err();
    assert_eq!(error.kind, PassErrorKind::Refused, "{:?}", error.kind);
    assert!(!error.message.is_empty());

    // Seventeen segments are past mk's limit, and the wrapper refuses them before the call.
    let many: Vec<Segment> = (0..17).map(|_| segment(empty, &[1], 0)).collect();
    assert!(engine.pass(&many, &[]).is_err());
}

#[test]
fn a_tile_change_after_open_refuses_the_pass() {
    let mut engine = open().expect("open mk");
    let state = State::Slot(0);
    let tiles = engine.identity().numbers.get("tiles").cloned().unwrap_or_default();
    let other = if tiles == "0" { "1" } else { "0" };
    let original = std::env::var_os("MK_TILES");
    // SAFETY: the hip tests run with --test-threads=1, and this test restores the value before it ends.
    unsafe { std::env::set_var("MK_TILES", other) };
    let result = engine.pass(std::slice::from_ref(&segment(state, &[1], 0)), &[]);
    match original {
        Some(value) => unsafe { std::env::set_var("MK_TILES", value) },
        None => unsafe { std::env::remove_var("MK_TILES") },
    }
    let error = result.expect_err("the pass must refuse a tile change");
    assert!(error.message.contains("MK_TILES changed"), "{}", error.message);
}

#[test]
fn the_conformance_suite_passes_on_mk() {
    let suite = run_all(&mut || open().expect("open mk"));
    println!("{suite}");
    assert!(suite.passed(), "{suite}");
    assert_eq!(suite.failures().count(), 0, "{suite}");
    let not_applicable: Vec<&str> = suite.not_applicable().map(|gate| gate.name).collect();
    assert_eq!(not_applicable, ["quantum_resume_fit"]);
}
