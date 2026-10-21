//! The mk engine on gfx1151: the binding calls, the write rules, one pass, and the conformance suite.
//!
//! Every test calls `open()`. The first call loads the weights; every later call makes a new context pool on the same
//! engine. All mk calls go to one worker thread. Run with `--test-threads=1`.

#![cfg(feature = "hip")]

use nagaya_abi::{
    Engine, OutputsWanted, PartError, PassErrorKind, SealError, Segment, SegmentInputs, State, Tap, WriteResult,
};
use nagaya_conformance::{compare_f32_rows, run_all};
use nagaya_mk_qwen38_gfx1151::{MkEngine, open};

#[path = "support/oracle.rs"]
mod oracle;

/// The first slot of the large class: slots 0-11 hold 4096 tokens, and slots 12-15 hold 32 768 (`spec.rs`).
const LARGE: u32 = 12;

/// The state bytes of one context: the fixed parts and 56 832 bytes for each token (`spec.rs` `context_bytes`).
fn context_bytes_of(capacity: u32) -> u64 {
    118_038_528 + 56_832 * u64::from(capacity)
}

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
    assert_eq!(limits.slots.len(), 2);
    assert_eq!(limits.slots[0].count, 12);
    assert_eq!(limits.slots[1].count, 4);
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
    // The state bytes: 12 * 350 822 400 + 4 * 1 980 309 504 = 4 209 868 800 + 7 921 238 016 = 12 131 106 816.
    // Each of the 16 contexts adds 4 MiB for the allocator (spec.rs `CONTEXT_ALLOCATION_BYTES`).
    assert_eq!(context.bytes, 12_131_106_816 + 16 * 4_194_304);
    // The device cost of each class is at least its state bytes, and the declared bytes hold it. A cost below the
    // state bytes would mean that mk made less than the formula says, and the formula would be wrong.
    for reading in engine.pool_readings() {
        let states = u64::from(reading.class.count) * context_bytes_of(reading.class.capacity);
        assert!(
            reading.device >= states,
            "{} contexts of {} tokens took {} bytes; the state bytes are {states}",
            reading.class.count,
            reading.class.capacity,
            reading.device
        );
        assert!(reading.device <= reading.declared, "{reading:?}");
    }
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
    // The pool holds 12 contexts of 4096 tokens, then 4 of 32 768.
    for slot in 0..LARGE {
        assert_eq!(engine.capacity(State::Slot(slot)).unwrap(), 4096, "slot {slot}");
    }
    for slot in LARGE..16 {
        assert_eq!(engine.capacity(State::Slot(slot)).unwrap(), 32_768, "slot {slot}");
    }
    assert!(engine.capacity(State::Slot(16)).is_err(), "the pool holds 16 contexts");
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

    // The same pass in two more handles gives the same bits. One handle lives at a time besides the first: the
    // weights and three pools do not fit in 125 GiB with the host floor (spec 4.7).
    let mut fresh = segment(State::Slot(0), &[11, 12, 13], 0);
    fresh.outputs.logit_rows = vec![2];
    let first = floats(&out.outputs[0].bytes);
    let mut second = open().expect("open mk");
    let other = second.pass(std::slice::from_ref(&fresh), &[]).expect("second pass");
    compare_f32_rows(
        "a_pass_runs_and_refuses_a_wrong_position",
        &first,
        &floats(&other.outputs[0].bytes),
        248_320,
    )
    .unwrap();
    drop(second);
    let mut third = open().expect("open mk");
    let again = third.pass(std::slice::from_ref(&fresh), &[]).expect("third pass");
    compare_f32_rows(
        "a_pass_runs_and_refuses_a_wrong_position",
        &first,
        &floats(&again.outputs[0].bytes),
        248_320,
    )
    .unwrap();
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

/// The ids of a synthetic prompt: `count` token ids that spread over the vocabulary, the same each time.
fn synthetic_ids(count: usize) -> Vec<u32> {
    (0..count).map(|index| ((index * 7919 + 13) % 248_320) as u32).collect()
}

/// Feed `ids` to the context in passes of at most `chunk` tokens, and return the logits of the last token.
///
/// Each pass starts at the context's token count. Only the last pass asks for logits.
fn feed(engine: &mut MkEngine, slot: u32, ids: &[u32], chunk: usize) -> Vec<f32> {
    let state = State::Slot(slot);
    let mut logits = Vec::new();
    let mut done = 0;
    while done < ids.len() {
        let end = (done + chunk).min(ids.len());
        let mut wanted = segment(state, &ids[done..end], engine.n(state).unwrap());
        if end == ids.len() {
            wanted.outputs.logit_rows = vec![(end - 1) as u64];
        }
        let out = engine
            .pass(std::slice::from_ref(&wanted), &[])
            .unwrap_or_else(|error| panic!("the pass at token {done} failed: {error:?}"));
        if end == ids.len() {
            logits = floats(&out.outputs[0].bytes);
        }
        done = end;
    }
    assert_eq!(engine.n(state).unwrap(), ids.len() as u64);
    logits
}

fn longer_prompt() -> oracle::Prompt {
    let fixture =
        oracle::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/oracle/qwen38.json"));
    fixture
        .prompts
        .into_iter()
        .find(|prompt| prompt.name == "longer")
        .expect("the fixture holds the longer prompt")
}

#[test]
fn a_prompt_past_one_pass_runs_in_several_passes_and_matches_the_oracle() {
    let prompt = longer_prompt();
    assert!(
        prompt.ids.len() > 512,
        "the prompt takes three passes: {} tokens",
        prompt.ids.len()
    );
    let path = oracle::QmmPath::from_env();
    let mut engine = open().expect("open mk");

    // A small context: 4096 tokens. Three passes of 256, 256 and the rest.
    let small = feed(&mut engine, 0, &prompt.ids, 256);
    let gap = oracle::measure(&prompt, &small);
    println!("small slot, {} tokens: {gap:?}", prompt.ids.len());
    assert_eq!(oracle::breaches(&gap, path), Vec::<String>::new(), "{gap:?}");

    // A large context: 32 768 tokens. The capacity does not change the bits.
    let large = feed(&mut engine, LARGE, &prompt.ids, 256);
    assert_eq!(small, large, "the logits depend on the context capacity");
}

#[test]
fn the_pass_size_does_not_change_the_logits() {
    let prompt = longer_prompt();
    let mut engine = open().expect("open mk");
    // mk gives the same bits for any split of the tokens into passes (spec 6.13). 100-token passes share no
    // boundary with 256-token passes except the start.
    let whole = feed(&mut engine, 0, &prompt.ids, 256);
    let split = feed(&mut engine, 1, &prompt.ids, 100);
    assert_eq!(whole, split, "the logits depend on the split into passes");
}

#[test]
fn a_small_context_takes_its_capacity_and_refuses_the_next_token() {
    let mut engine = open().expect("open mk");
    let state = State::Slot(2);
    let ids = synthetic_ids(4096);
    let logits = feed(&mut engine, 2, &ids, 256);
    assert!(logits.iter().all(|value| value.is_finite()));
    assert_eq!(engine.n(state).unwrap(), 4096);
    let next = segment(state, &[1], 4096);
    let error = engine.pass(std::slice::from_ref(&next), &[]).unwrap_err();
    assert_eq!(error.kind, PassErrorKind::Refused, "{:?}", error.kind);
    assert_eq!(engine.n(state).unwrap(), 4096, "a refused pass changed the context");
}

#[test]
fn a_large_context_takes_more_tokens_than_a_small_one() {
    let mut engine = open().expect("open mk");
    let state = State::Slot(LARGE);
    // 5000 tokens are past the small class's 4096 and fill 20 passes.
    let ids = synthetic_ids(5000);
    let logits = feed(&mut engine, LARGE, &ids, 256);
    assert!(logits.iter().all(|value| value.is_finite()));
    assert_eq!(engine.n(state).unwrap(), 5000);
}
