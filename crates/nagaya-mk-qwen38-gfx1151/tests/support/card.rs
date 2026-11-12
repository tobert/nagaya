//! Helpers of the card tests of the taps and of the control vectors (milestone 34). Each test file includes this file.
//!
//! Every test calls `open()`. A hold keeps the weights for the whole binary, as in `tests/hip.rs`. All engine calls go
//! to one worker thread. Run with `--test-threads=1`, and never free the engine in a test (spec 3.3).

#![allow(dead_code)]

use std::time::Instant;

use nagaya_abi::{
    ControlItem, ControlOp, Engine, OutputsWanted, PassOut, Segment, SegmentInputs, State, Tap, WriteResult,
};
use nagaya_mk_qwen38_gfx1151::{Hold, Qwen38Engine, Qwen38Error, hold};

/// The hold of this binary on the engine's weights.
static HOLD: std::sync::OnceLock<Hold> = std::sync::OnceLock::new();

/// Open a handle, with the weights held for the binary.
pub fn open() -> Result<Qwen38Engine, Qwen38Error> {
    HOLD.get_or_init(|| hold().expect("hold the engine's weights"));
    nagaya_mk_qwen38_gfx1151::open()
}

/// The values of one residual row: 4 streams of 2,560.
pub const WIDTH: usize = 10_240;

/// The layers of the model.
pub const LAYERS: usize = 48;

/// The values of the fold.
pub const FOLD_WIDTH: usize = 2_560;

/// The vocabulary.
pub const VOCAB: usize = 248_320;

/// Read bytes as f32 values.
pub fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// The values as bit patterns, so that a compare finds a NaN and a signed zero.
pub fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The ids of a synthetic prompt: `count` token ids that spread over the vocabulary. `seed` makes another prompt.
pub fn ids(count: usize, seed: usize) -> Vec<u32> {
    (0..count)
        .map(|index| (((index + seed) * 7919 + 13) % VOCAB) as u32)
        .collect()
}

/// A segment of `tokens` that continues the state at its first position `start`.
pub fn segment(state: State, tokens: &[u32], start: u64, logits: bool, control: Vec<ControlItem>) -> Segment {
    Segment {
        control,
        state,
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (start..start + tokens.len() as u64).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted {
            logit_rows: if logits {
                vec![start + tokens.len() as u64 - 1]
            } else {
                Vec::new()
            },
        },
    }
}

/// The taps with these names.
pub fn taps(names: &[&str]) -> Vec<Tap> {
    names
        .iter()
        .map(|name| Tap {
            name: (*name).to_owned(),
        })
        .collect()
}

/// A control item of one entry, on each layer or on the layers that `layers` names.
pub fn item(index: u32, op: ControlOp, strength: f32, layers: Option<Vec<(u32, f32)>>) -> ControlItem {
    ControlItem {
        index,
        op,
        strength,
        layers,
    }
}

/// The outputs of a pass, by name, and its wall time.
pub struct Run {
    pub logits: Vec<Vec<f32>>,
    pub residual: Vec<Vec<f32>>,
    pub fold: Vec<Vec<f32>>,
    pub millis: f64,
}

/// Split the outputs of a pass by name. The names must be `logits`, `residual` and `fold`, in that order of groups.
pub fn split(out: &PassOut, millis: f64) -> Run {
    let mut run = Run {
        logits: Vec::new(),
        residual: Vec::new(),
        fold: Vec::new(),
        millis,
    };
    let mut group = 0;
    for output in &out.outputs {
        let (list, order, width) = match output.name.as_str() {
            "logits" => (&mut run.logits, 0, VOCAB),
            "residual" => (&mut run.residual, 1, LAYERS * WIDTH),
            "fold" => (&mut run.fold, 2, FOLD_WIDTH),
            other => panic!("the pass gave an output `{other}`"),
        };
        assert!(
            order >= group,
            "the outputs are out of order: {:?}",
            out.outputs.iter().map(|o| &o.name).collect::<Vec<_>>()
        );
        group = order;
        assert_eq!(
            output.bytes.len(),
            width * 4,
            "the size of the output `{}`",
            output.name
        );
        list.push(floats(&output.bytes));
    }
    run
}

/// One pass of `segments` with the taps `tap_names`. A pass that fails stops the test.
pub fn run(engine: &mut Qwen38Engine, segments: &[Segment], tap_names: &[&str]) -> Run {
    let start = Instant::now();
    let out = engine
        .pass(segments, &taps(tap_names))
        .unwrap_or_else(|error| panic!("the pass failed: {error:?}"));
    let millis = start.elapsed().as_secs_f64() * 1e3;
    split(&out, millis)
}

/// The bytes of each stored part of a slot, parts 0 to 5.
pub fn parts(engine: &mut Qwen38Engine, slot: u32) -> Vec<Vec<u8>> {
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

/// Assert that two slots hold equal bits in every stored part. The message names the first part that differs.
pub fn assert_parts_equal(what: &str, left: &[Vec<u8>], right: &[Vec<u8>]) {
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

/// Empty a slot: the engine frees the context and makes a new one of the same capacity.
pub fn clear(engine: &mut Qwen38Engine, slot: u32) {
    assert_eq!(
        engine.slots().unwrap().slot_clear(slot),
        WriteResult::Ok,
        "clear slot {slot}"
    );
}

/// The value of the vector of the oracle at layer `layer`, index `j`: `(((31*j + 17*l) mod 257) - 128) / 256`. It is
/// exact in f32. This is the twin of `tools/oracle_fixture.py` and of the test in `src/control.rs`.
pub fn oracle_vector(layer: usize, j: usize) -> f32 {
    ((((31 * j + 17 * layer) % 257) as i32 - 128) as f32) / 256.0
}

/// The bytes of a vector for `control_write`: 48 layers of the vector of the oracle, shifted by `shift` so that two
/// entries differ.
pub fn vector_bytes(shift: usize) -> Vec<u8> {
    (0..LAYERS)
        .flat_map(|layer| (0..WIDTH).map(move |j| oracle_vector(layer + shift, j + 3 * shift)))
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The one layer of `vector_bytes(shift)`, as f32 values.
pub fn vector_layer(shift: usize, layer: usize) -> Vec<f32> {
    (0..WIDTH)
        .map(|j| oracle_vector(layer + shift, j + 3 * shift))
        .collect()
}

/// The dot product of two rows in f64.
pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum()
}

/// The 2-norm of a row in f64.
pub fn norm(a: &[f32]) -> f64 {
    dot(a, a).sqrt()
}

/// The row of layer `layer` in a residual output.
pub fn layer_row(residual: &[f32], layer: usize) -> &[f32] {
    &residual[layer * WIDTH..(layer + 1) * WIDTH]
}

/// Feed `token_ids` to the slot in passes of at most 256 tokens. Each pass carries the control list. The last pass asks
/// for its logit row and the taps `tap_names`, and its outputs are the result.
pub fn feed(
    engine: &mut Qwen38Engine,
    slot: u32,
    token_ids: &[u32],
    tap_names: &[&str],
    control: &[ControlItem],
) -> Run {
    let state = State::Slot(slot);
    let mut done = 0;
    loop {
        let end = (done + 256).min(token_ids.len());
        let last = end == token_ids.len();
        let start = engine.n(state).unwrap();
        let wanted = segment(state, &token_ids[done..end], start, last, control.to_vec());
        let run = run(
            engine,
            std::slice::from_ref(&wanted),
            if last { tap_names } else { &[] },
        );
        if last {
            return run;
        }
        done = end;
    }
}
