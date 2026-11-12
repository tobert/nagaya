//! The mk engine on gfx1151: the binding calls, the write rules, one pass, and the conformance suite.
//!
//! Every test calls `open()`. The first call loads the weights; every later call makes a new context pool on the same
//! engine. A hold keeps the weights for the whole binary, so a dropped handle does not unload them (engine-abi
//! "close"; `hip_close.rs` tests the unload). All mk calls go to one worker thread. Run with `--test-threads=1`.

#![cfg(feature = "hip")]

use nagaya_abi::{
    CoverageSpan, Engine, OutputsWanted, PartError, PassErrorKind, SealError, Segment, SegmentInputs, State, Tap,
    WriteResult,
};
use nagaya_conformance::{compare_f32_rows, run_all};
use nagaya_mk_qwen38_gfx1151::{Hold, Qwen38Engine, Qwen38Error, hold};

/// The hold of this binary on mk's weights.
static HOLD: std::sync::OnceLock<Hold> = std::sync::OnceLock::new();

/// Open a handle, with the weights held for the binary.
fn open() -> Result<Qwen38Engine, Qwen38Error> {
    HOLD.get_or_init(|| hold().expect("hold mk's weights"));
    nagaya_mk_qwen38_gfx1151::open()
}

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

/// The weight arena is locked in memory (milestone 30 step 5). It is a memfd that the GPU reads through a host
/// registration. Swap could move its pages, and a move remaps them under a running kernel that the driver cannot
/// preempt. On 2026-11-10 that gave spin-timeouts and a "Page not present" fault on gfx1151host.
#[test]
fn the_weight_arena_is_locked_in_memory() {
    let _engine = open().expect("open mk");
    let smaps = std::fs::read_to_string("/proc/self/smaps").expect("read /proc/self/smaps");
    let mut lines = smaps.lines();
    let mut found = 0;
    while let Some(line) = lines.next() {
        if !line.contains("memfd:qwen38-weights") {
            continue;
        }
        let mut size = None;
        let mut locked = None;
        for field in lines.by_ref() {
            let kib = |name: &str| -> Option<u64> {
                field.strip_prefix(name)?.trim().strip_suffix("kB")?.trim().parse().ok()
            };
            if let Some(value) = kib("Size:") {
                size = Some(value);
            }
            if let Some(value) = kib("Locked:") {
                locked = Some(value);
            }
            if let Some(value) = kib("Rss:") {
                println!("Rss: {value} kB");
            }
            if field.starts_with("VmFlags:") {
                break;
            }
        }
        let (size, locked) = (size.expect("a Size line"), locked.expect("a Locked line"));
        println!("{line}: {size} kB, {locked} kB locked");
        assert_eq!(locked, size, "the weight arena is not locked in memory: {line}");
        found += size;
    }
    // The arena holds 65,490,227,200 bytes (deploy/gfx1151host/profile.toml), so the locked maps hold at least that.
    assert!(found * 1024 >= 65_490_227_200, "the locked weight maps hold {found} kB");
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
    // The facts of the static build take the place of the library path and its pin (spec 12.1).
    let build = identity
        .numbers
        .get("engine_build_sha256")
        .expect("the identity names the build");
    assert_eq!(build.len(), 64, "{build}");
    assert!(!identity.numbers.contains_key("engine_lib"), "{:?}", identity.numbers);
    assert!(
        identity.checkpoint_digest.starts_with("spot-sha256:"),
        "{}",
        identity.checkpoint_digest
    );
    // The identity holds the hash of the tokenizer keys in the first shard, as the text library computes it.
    let shards = std::env::var("QWEN38_SHARDS").expect("QWEN38_SHARDS");
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
fn a_part_above_32_bits_is_refused_and_is_not_read_as_its_low_bits() {
    // mk names a part with 32 bits. A cast would read part 2^32 + 5 as part 5 and part 2^32 + 3 as part 3.
    let mut engine = open().expect("open mk");
    let state = State::Slot(0);
    let high = 1usize << 32;
    let mut history = vec![0u8; 8];
    assert_eq!(
        engine.parts().unwrap().part_read(state, high + 5, &mut history),
        Err(PartError::Refused)
    );
    assert!(engine.last_error().unwrap_or_default().contains("no part"));
    assert_eq!(
        engine
            .parts()
            .unwrap()
            .part_write(state, high + 5, &[7, 0, 0, 0, 9, 0, 0, 0]),
        WriteResult::Refused
    );
    assert!(engine.last_error().unwrap_or_default().contains("no part"));
    let kv = finite_bytes(2 * 49_152, 1.0);
    assert_eq!(
        engine.parts().unwrap().cells_write(state, high + 3, 0, 2, &kv),
        WriteResult::Refused
    );
    assert!(engine.last_error().unwrap_or_default().contains("no part"));
    // Nothing changed: the history is the empty context's, and set_n finds no written cells.
    engine.parts().unwrap().part_read(state, 5, &mut history).unwrap();
    assert_ne!(history, [7, 0, 0, 0, 9, 0, 0, 0]);
    assert_eq!(
        engine.parts().unwrap().part_write(state, 5, &[7, 0, 0, 0, 9, 0, 0, 0]),
        WriteResult::Ok
    );
    assert_eq!(engine.parts().unwrap().set_n(state, 2), WriteResult::Refused);
    assert_eq!(engine.n(state).unwrap(), 0);
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
        control: Vec::new(),
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
    let original = std::env::var_os("QWEN38_TILES");
    // SAFETY: the hip tests run with --test-threads=1, and this test restores the value before it ends.
    unsafe { std::env::set_var("QWEN38_TILES", other) };
    let result = engine.pass(std::slice::from_ref(&segment(state, &[1], 0)), &[]);
    match original {
        Some(value) => unsafe { std::env::set_var("QWEN38_TILES", value) },
        None => unsafe { std::env::remove_var("QWEN38_TILES") },
    }
    let error = result.expect_err("the pass must refuse a tile change");
    assert!(error.message.contains("QWEN38_TILES changed"), "{}", error.message);
}

#[test]
fn the_conformance_suite_passes_on_mk() {
    let suite = run_all(&mut || open().expect("open mk"));
    println!("{suite}");
    assert!(suite.passed(), "{suite}");
    assert_eq!(suite.failures().count(), 0, "{suite}");
    // mk declares `coverage`, so every gate applies, also `quantum_resume_fit`. mk has no stages and no feed rule.
    let not_applicable: Vec<&str> = suite.not_applicable().map(|gate| gate.name).collect();
    assert_eq!(not_applicable, vec!["stage_path_bits", "feed_rule_bits"]);
}

/// The ids of a synthetic prompt: `count` token ids that spread over the vocabulary, the same each time.
fn synthetic_ids(count: usize) -> Vec<u32> {
    (0..count).map(|index| ((index * 7919 + 13) % 248_320) as u32).collect()
}

/// Feed `ids` to the context in passes of at most `chunk` tokens, and return the logits of the last token.
///
/// Each pass starts at the context's token count. Only the last pass asks for logits.
fn feed(engine: &mut Qwen38Engine, slot: u32, ids: &[u32], chunk: usize) -> Vec<f32> {
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

#[test]
fn a_slot_counts_its_tokens_and_a_clear_empties_it() {
    let mut engine = open().expect("open mk");
    let state = State::Slot(0);
    let ids = synthetic_ids(300);
    let first = feed(&mut engine, 0, &ids, 256);
    assert_eq!(engine.slots().unwrap().slot_high(0), 300);
    assert_eq!(engine.slots().unwrap().slot_high(1), 0, "slot 1 holds no token");

    // A cells read from 0 of all rows is the whole part.
    let mut whole = vec![0u8; 300 * 49_152];
    engine.parts().unwrap().part_read(state, 3, &mut whole).unwrap();
    let mut rows = vec![0u8; 300 * 49_152];
    engine.parts().unwrap().cells_read(state, 3, 0, 300, &mut rows).unwrap();
    assert!(whole == rows, "the cells read differs from the part read");
    let mut short = vec![0u8; 299 * 49_152];
    assert_eq!(
        engine.parts().unwrap().cells_read(state, 3, 1, 299, &mut short),
        Err(PartError::Refused),
        "mk has no range read"
    );
    assert_eq!(
        engine.parts().unwrap().cells_read(state, 3, 0, 299, &mut short),
        Err(PartError::Refused),
        "a read of fewer rows than the token count is a range read"
    );

    let coverage = engine.coverage().unwrap();
    let span = coverage.coverage(state, 3, &rows);
    assert_eq!(
        span,
        CoverageSpan {
            part: 3,
            start: 0,
            end: 300
        }
    );
    assert_eq!(coverage.fits(&span, 0), Ok(()));

    let capacity = engine.capacity(state).unwrap();
    assert_eq!(engine.slots().unwrap().slot_clear(0), WriteResult::Ok);
    assert_eq!(engine.slots().unwrap().slot_high(0), 0);
    assert_eq!(engine.capacity(state).unwrap(), capacity, "a clear keeps the capacity");
    assert!(!engine.poisoned(state).unwrap());

    // The cleared slot is a new context: the same tokens give the same logits.
    let again = feed(&mut engine, 0, &ids, 256);
    assert_eq!(first, again, "the cleared slot kept state from before the clear");
}

/// The bytes of each stored part of a slot, parts 0 to 5. A cells part gives its rows up to the token count.
fn stored_parts(engine: &mut Qwen38Engine, slot: u32) -> Vec<Vec<u8>> {
    let state = State::Slot(slot);
    let n = engine.n(state).unwrap() as u32;
    (0..6u32)
        .map(|part| {
            let mut out = vec![0u8; engine.part_bytes(part, n).unwrap() as usize];
            engine
                .parts()
                .unwrap()
                .part_read(state, part as usize, &mut out)
                .unwrap();
            out
        })
        .collect()
}

/// Feed `ids` after the rows of the slot, in passes of the given sizes, and return the logits of the last token.
fn feed_sizes(engine: &mut Qwen38Engine, slot: u32, ids: &[u32], sizes: &[usize]) -> Vec<f32> {
    assert_eq!(
        sizes.iter().sum::<usize>(),
        ids.len(),
        "the pass sizes cover the tokens"
    );
    let state = State::Slot(slot);
    let base = engine.n(state).unwrap() as usize;
    let mut logits = Vec::new();
    let mut done = 0;
    for size in sizes {
        let end = done + size;
        let mut wanted = segment(state, &ids[done..end], (base + done) as u64);
        if end == ids.len() {
            wanted.outputs.logit_rows = vec![(base + end - 1) as u64];
        }
        let out = engine
            .pass(std::slice::from_ref(&wanted), &[])
            .unwrap_or_else(|error| panic!("the pass at token {} failed: {error:?}", base + done));
        if end == ids.len() {
            logits = floats(&out.outputs[0].bytes);
        }
        done = end;
    }
    assert_eq!(engine.n(state).unwrap() as usize, base + ids.len());
    logits
}

/// Assert that two slots hold equal bits in every stored part. The message names the first part that differs.
fn assert_parts_equal(what: &str, left: &[Vec<u8>], right: &[Vec<u8>]) {
    let names = ["gdn_state", "gdn_conv", "ple_conv", "kv", "idx_keys", "ple_hist"];
    for (part, name) in names.iter().enumerate() {
        assert!(
            left[part].len() == right[part].len() && left[part] == right[part],
            "{what}: part {part} `{name}` differs ({} and {} bytes)",
            left[part].len(),
            right[part].len()
        );
    }
}

/// The logits as bit patterns, so that a compare finds a NaN and a signed zero.
fn bits(logits: &[f32]) -> Vec<u32> {
    logits.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn an_interval_copy_restores_and_replays_to_the_bits_of_the_live_context() {
    // Spec 6.13: a copy at a multiple of the grain does not seal, so the live bits do not change. A restore of the
    // copy, then a replay of the tokens after it, gives the bits of the live context.
    const GRAIN: usize = 256;
    let ids = synthetic_ids(700);
    let at = 2 * GRAIN;
    let mut engine = open().expect("open mk");
    let live = State::Slot(0);
    feed(&mut engine, 0, &ids[..at], 256);

    let before = stored_parts(&mut engine, 0);
    let rows = [(3usize, 0u64, at as u64), (4, 0, at as u64)];
    let copy = nagaya_store::snapshot_from_engine(&mut engine, live, &rows).expect("the interval copy");
    // The copy goes through the canonical bytes of the store (spec 6.3), as a store put and restore do.
    let file = copy.encode().expect("encode the copy");
    let copy = nagaya_store::Snapshot::decode(&file).expect("decode the copy");
    let after = stored_parts(&mut engine, 0);
    assert_parts_equal("the copy changed the live context", &before, &after);

    // The live context goes on in passes of 100 tokens.
    let tail = &ids[at..];
    let live_logits = feed_sizes(&mut engine, 0, tail, &[100, 88]);
    let live_parts = stored_parts(&mut engine, 0);

    // A context of the large class takes the copy, and replays the tail in another split.
    assert_eq!(engine.slots().unwrap().slot_clear(LARGE), WriteResult::Ok);
    let restored =
        nagaya_store::write_snapshots(&mut engine, State::Slot(LARGE), std::slice::from_ref(&copy)).expect("restore");
    assert_eq!(restored, nagaya_store::Restored::Done);
    assert_eq!(engine.n(State::Slot(LARGE)).unwrap(), at as u64);
    let restored_at_copy = stored_parts(&mut engine, LARGE);
    assert_parts_equal("the restore differs from the copy", &before, &restored_at_copy);
    let replay_logits = feed_sizes(&mut engine, LARGE, tail, &[1, 63, 124]);
    let replay_parts = stored_parts(&mut engine, LARGE);

    assert_parts_equal("the replay differs from the live context", &live_parts, &replay_parts);
    assert!(
        bits(&live_logits) == bits(&replay_logits),
        "the logits of the replay differ from the live context"
    );

    // A context that nobody copied has the bits of the live context too.
    let plain_logits = feed(&mut engine, 1, &ids, 256);
    let plain_parts = stored_parts(&mut engine, 1);
    assert_parts_equal(
        "the copy changed the later bits of the live context",
        &plain_parts,
        &live_parts,
    );
    assert!(
        bits(&plain_logits) == bits(&live_logits),
        "the copy changed the logits of the live context"
    );
}

#[test]
fn a_restore_past_the_dense_reach_pools_the_keys_and_replays_to_the_bits() {
    // A pass whose context reaches past 2051 visible cells scores the pooled block keys (mk's indexer). The pooled
    // keys are derived state: a restore pools them from the written index keys (host.rs pool_context). A restore at
    // 2304 tokens, then a replay, must give the bits of the live context, so the replay reads the pooled keys.
    const AT: usize = 9 * 256;
    let ids = synthetic_ids(AT + 120);
    let mut engine = open().expect("open mk");
    let live = State::Slot(0);
    feed(&mut engine, 0, &ids[..AT], 256);
    let rows = [(3usize, 0u64, AT as u64), (4, 0, AT as u64)];
    let copy = nagaya_store::snapshot_from_engine(&mut engine, live, &rows).expect("the interval copy");
    let live_logits = feed_sizes(&mut engine, 0, &ids[AT..], &[120]);
    let live_parts = stored_parts(&mut engine, 0);

    assert_eq!(engine.slots().unwrap().slot_clear(LARGE), WriteResult::Ok);
    let restored =
        nagaya_store::write_snapshots(&mut engine, State::Slot(LARGE), std::slice::from_ref(&copy)).expect("restore");
    assert_eq!(restored, nagaya_store::Restored::Done);
    let replay_logits = feed_sizes(&mut engine, LARGE, &ids[AT..], &[120]);
    let replay_parts = stored_parts(&mut engine, LARGE);
    assert_parts_equal("the replay past the dense reach differs", &live_parts, &replay_parts);
    assert!(
        bits(&live_logits) == bits(&replay_logits),
        "the logits of the replay past the dense reach differ from the live context"
    );
}

#[test]
fn a_split_across_256_does_not_change_the_bits_in_either_class() {
    // Spec 5.7 and 6.13: the split of the tokens into passes does not change bits. The splits cross the multiples
    // 256 and 512 of the grain at different tokens. One of them crosses 256 one token at a time.
    let ids = synthetic_ids(600);
    let splits: [Vec<usize>; 4] = [
        vec![256, 256, 88],
        vec![100; 6],
        vec![128, 127, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 128, 128, 79],
        vec![255, 2, 254, 89],
    ];
    let mut engine = open().expect("open mk");
    let mut first: Option<(Vec<Vec<u8>>, Vec<u32>)> = None;
    for slot in [0u32, LARGE] {
        for (index, sizes) in splits.iter().enumerate() {
            assert_eq!(engine.slots().unwrap().slot_clear(slot), WriteResult::Ok);
            let logits = bits(&feed_sizes(&mut engine, slot, &ids, sizes));
            let parts = stored_parts(&mut engine, slot);
            match &first {
                None => first = Some((parts, logits)),
                Some((parts0, logits0)) => {
                    let what = format!("slot {slot}, split {index} {sizes:?}");
                    assert_parts_equal(&what, parts0, &parts);
                    assert!(*logits0 == logits, "{what}: the logits differ");
                }
            }
        }
    }
}

#[test]
fn mk_elide_0_runs_the_stages_in_build_order_with_the_same_bits() {
    // QWEN38_ELIDE=0 turns the Rust schedule off (src/schedule.rs `pass_plan`): each stage in build order with a grid
    // barrier after it. The schedule keeps each conflicting pair in order, so the bits and the parts do not change.
    // The passes of 1 and 16 tokens are fused (decode and the fused staging), and the pass of 23 tokens is not.
    let ids = synthetic_ids(40);
    let sizes = [23, 16, 1];
    let mut engine = open().expect("open mk");
    let mut run = |elide: Option<&str>| {
        let original = std::env::var_os("QWEN38_ELIDE");
        // SAFETY: the hip tests run with --test-threads=1, and this closure restores the value before it returns.
        match elide {
            Some(value) => unsafe { std::env::set_var("QWEN38_ELIDE", value) },
            None => unsafe { std::env::remove_var("QWEN38_ELIDE") },
        }
        assert_eq!(engine.slots().unwrap().slot_clear(0), WriteResult::Ok);
        let logits = bits(&feed_sizes(&mut engine, 0, &ids, &sizes));
        let parts = stored_parts(&mut engine, 0);
        match original {
            Some(value) => unsafe { std::env::set_var("QWEN38_ELIDE", value) },
            None => unsafe { std::env::remove_var("QWEN38_ELIDE") },
        }
        (logits, parts)
    };
    let (scheduled, scheduled_parts) = run(None);
    let (plain, plain_parts) = run(Some("0"));
    assert_parts_equal("QWEN38_ELIDE=0", &scheduled_parts, &plain_parts);
    assert!(scheduled == plain, "QWEN38_ELIDE=0 changed the logits");
}

/// The process serves mk on demand from `describe()` before a load, so the description must be the open's facts.
#[test]
fn describe_gives_the_identity_and_the_limits_of_the_open() {
    let (identity, limits) = nagaya_mk_qwen38_gfx1151::describe().expect("describe mk");
    let engine = open().expect("open mk");
    assert_eq!(identity, engine.identity());
    assert_eq!(limits, engine.limits());
}
