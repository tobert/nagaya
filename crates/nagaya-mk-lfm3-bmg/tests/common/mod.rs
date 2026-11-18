//! The model folders and the oracle files of the tests, and the rule for a machine that has no copy of them.
//!
//! A machine does not hold every model, so this state is normal. An absent file skips the test.
//! `LFM3_MODEL_DIR` and `LFM3_OMNI_MODEL_DIR` name the folders, and the code holds no default path (AGENTS.md).
//! `NAGAYA_REQUIRE_MODELS=1` makes an absent file fail the test. Only absence skips.
#![allow(dead_code)]

pub mod cpu;

use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

pub mod speech;

/// The tolerance of a stage of the card against transformers, or against a CPU form: a relative error. Measured on
/// 2026-11-06 (spec 11.11, "The tolerances of the LFM3 gates"): the worst stage was 6.8e-6, the projector of the vision
/// tower. A wrong rotation, window or gate gave 0.09 or more (`ed55acc`), and a wrong audio op 1.3e-2 or more.
pub const STAGE_TOL: f64 = 1e-4;
/// The tolerance of a logit of the card against transformers, or between two splits of a feed: an absolute gap.
/// Measured on 2026-11-06: the worst was 6.4e-5, between two splits of one feed, and 3.6e-5 against transformers.
pub const LOGIT_TOL: f64 = 1e-3;

pub const TEXT_VARIABLE: &str = "LFM3_MODEL_DIR";
pub const OMNI_VARIABLE: &str = "LFM3_OMNI_MODEL_DIR";
pub const GOLDEN_VARIABLE: &str = "LFM3_GOLDEN_DIR";
pub const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";

/// The two models, with the variable of each folder and the label of its oracle files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Text,
    Omni,
}

impl Which {
    pub const BOTH: [Which; 2] = [Which::Text, Which::Omni];

    pub fn variable(self) -> &'static str {
        match self {
            Which::Text => TEXT_VARIABLE,
            Which::Omni => OMNI_VARIABLE,
        }
    }

    /// The label of the oracle files: `lfm3` or `lfm3-omni`.
    pub fn label(self) -> &'static str {
        match self {
            Which::Text => "lfm3",
            Which::Omni => "lfm3-omni",
        }
    }
}

fn required() -> bool {
    std::env::var(REQUIRE_VARIABLE).is_ok_and(|v| v == "1")
}

/// True when `file` is absent and the test must return. It prints one line to stderr, or panics when
/// `NAGAYA_REQUIRE_MODELS=1`. A path that exists but is not a readable file panics.
fn file_absent(file: Option<PathBuf>, unset: &str, how: &str) -> bool {
    let test = std::thread::current().name().unwrap_or("a test").to_owned();
    let skip = |why: String| {
        if required() {
            panic!("{why}. {REQUIRE_VARIABLE} is set. {how}.");
        }
        eprintln!("skipped: {test}: {why}; {how}");
        true
    };
    let Some(file) = file else {
        return skip(format!("{unset} is not set"));
    };
    match std::fs::metadata(&file) {
        Ok(m) if m.is_file() => false,
        Ok(_) => panic!("{} is not a file. {how}.", file.display()),
        Err(e) if e.kind() == ErrorKind::NotFound => skip(format!("{} does not exist", file.display())),
        Err(e) => panic!("cannot read {}: {e}", file.display()),
    }
}

fn configured(variable: &str) -> Option<PathBuf> {
    std::env::var_os(variable).map(PathBuf::from)
}

/// True when the folder of `which` is absent. Call it first in each test that needs the model.
pub fn absent(which: Which) -> bool {
    let variable = which.variable();
    file_absent(
        configured(variable).map(|dir| dir.join("model.safetensors")),
        variable,
        &format!("set {variable} to the model folder"),
    )
}

/// The folder of `which`. A test that calls it has called `absent` before.
pub fn model_dir(which: Which) -> PathBuf {
    let dir = configured(which.variable()).expect("the test must call `absent` first");
    assert!(
        dir.join("model.safetensors").is_file(),
        "the test must call `absent` first"
    );
    dir
}

/// The folder of the oracle files: `LFM3_GOLDEN_DIR`, or `target/lfm3-golden` of the workspace.
pub fn golden_dir() -> PathBuf {
    configured(GOLDEN_VARIABLE)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/lfm3-golden"))
}

/// True when the oracle file `name` is absent. Call it after `absent`.
pub fn golden_absent(name: &str) -> bool {
    file_absent(
        Some(golden_dir().join(name)),
        GOLDEN_VARIABLE,
        "write it with tools/lfm3_golden.sh, with no GPU job",
    )
}

/// The JSON oracle file `name`.
pub fn golden_json(name: &str) -> serde_json::Value {
    let path = golden_dir().join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{} is not JSON: {e}", path.display()))
}

/// One tensor of an oracle file, as `f64` values with its shape. The oracle holds F32, I64 and U8.
pub struct Tensor {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}

impl Tensor {
    pub fn f32(&self) -> Vec<f32> {
        self.values.iter().map(|v| *v as f32).collect()
    }

    pub fn ids(&self) -> Vec<u32> {
        self.values.iter().map(|v| *v as u32).collect()
    }
}

/// Read every tensor of the safetensors oracle file `name`.
pub fn golden_tensors(name: &str) -> HashMap<String, Tensor> {
    let path = golden_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).expect("the header is JSON");
    let data = &bytes[8 + n..];
    let mut out = HashMap::new();
    for (name, meta) in header.as_object().expect("the header is an object") {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let [start, end] = [0, 1].map(|i| meta["data_offsets"][i].as_u64().unwrap() as usize);
        let raw = &data[start..end];
        let values = match meta["dtype"].as_str().unwrap() {
            "F32" => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b) as f64)
                .collect(),
            "I64" => raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b) as f64)
                .collect(),
            // The PCM samples of the audio oracle.
            "I16" => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b) as f64)
                .collect(),
            "U8" => raw.iter().map(|b| *b as f64).collect(),
            other => panic!("tensor {name} has dtype {other}; the oracle holds F32, I64, I16 and U8"),
        };
        out.insert(name.clone(), Tensor { shape, values });
    }
    out
}

/// The card of the test: the first card of `NAGAYA_TEST_CARDS`.
pub fn cards() -> Vec<String> {
    nagaya_mk_bmg::test_cards::cards(1)
}

/// The relative L2 error of `got` against `want`.
pub fn rel(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "the lengths differ");
    let num: f64 = got.iter().zip(want).map(|(g, w)| (*g as f64 - *w as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|w| (*w as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// The largest absolute difference of two rows.
pub fn max_abs(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "the lengths differ");
    got.iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((*g as f64 - *w as f64).abs()))
}

/// The engine of `which` on card 0, the card of both LFM3 models on the B70 machine (spec 11.11), with `slots` slots
/// and a feed block of `max_block`.
#[cfg(feature = "xe")]
pub fn open(which: Which, slots: usize, max_block: usize) -> nagaya_mk_lfm3_bmg::Lfm3Engine {
    let config = nagaya_mk_lfm3_bmg::Config {
        model_dir: model_dir(which),
        slots: vec![nagaya_mk_lfm3_bmg::spec::SLOT_TOKENS; slots],
        max_block,
        extra_runs: 0,
    };
    nagaya_mk_lfm3_bmg::Lfm3Engine::open(config, &cards()[0]).unwrap_or_else(|e| panic!("open {which:?}: {e}"))
}

/// The engine of `which` on card 0 with slots of the positions `slots`, from short to long (engine-abi "Slot sizes").
#[cfg(feature = "xe")]
pub fn open_sized(which: Which, slots: Vec<usize>, max_block: usize) -> nagaya_mk_lfm3_bmg::Lfm3Engine {
    let config = nagaya_mk_lfm3_bmg::Config {
        model_dir: model_dir(which),
        slots,
        max_block,
        extra_runs: 0,
    };
    nagaya_mk_lfm3_bmg::Lfm3Engine::open(config, &cards()[0]).unwrap_or_else(|e| panic!("open {which:?}: {e}"))
}

/// A segment of `tokens` from position `from` of `slot`, with the logits of the positions `logits`.
pub fn segment(slot: u32, from: u64, tokens: &[u32], logits: &[u64]) -> nagaya_abi::Segment {
    nagaya_abi::Segment {
        control: Vec::new(),
        state: nagaya_abi::State::Slot(slot),
        inputs: nagaya_abi::SegmentInputs::Tokens(tokens.to_vec()),
        positions: (from..from + tokens.len() as u64).collect(),
        meta: Vec::new(),
        outputs: nagaya_abi::OutputsWanted {
            logit_rows: logits.to_vec(),
        },
        layer_loop: None,
    }
}

/// Feed `ids` into `slot` from position `from` in passes of the lengths of `split`, and give the logits of the last
/// position.
pub fn feed_split(engine: &mut dyn nagaya_abi::Engine, slot: u32, from: u64, ids: &[u32], split: &[usize]) -> Vec<f32> {
    assert_eq!(split.iter().sum::<usize>(), ids.len(), "the split covers the ids");
    let mut at = 0usize;
    let mut last = Vec::new();
    for (i, n) in split.iter().enumerate() {
        let pos = from + at as u64;
        let logits: Vec<u64> = if i + 1 == split.len() {
            vec![pos + *n as u64 - 1]
        } else {
            Vec::new()
        };
        let out = engine
            .pass(&[segment(slot, pos, &ids[at..at + n], &logits)], &[])
            .unwrap_or_else(|e| panic!("a pass at {pos} of {n} tokens: {}", e.message));
        if let Some(row) = out.outputs.first() {
            last = row
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
        }
        at += n;
    }
    last
}

/// The split of `n` tokens by the feed rule: passes of `block` from the start, and the rest last (engine-abi D11).
pub fn blocks(n: usize, block: usize) -> Vec<usize> {
    let mut split = vec![block; n / block];
    if !n.is_multiple_of(block) {
        split.push(n % block);
    }
    split
}

/// The index of the first value whose bits differ, or `None`.
pub fn first_other_bits(a: &[f32], b: &[f32]) -> Option<usize> {
    assert_eq!(a.len(), b.len(), "the lengths differ");
    a.iter().zip(b).position(|(x, y)| x.to_bits() != y.to_bits())
}

/// The log of the sum of the exponents of `row`, in f64.
pub fn logsumexp(row: &[f32]) -> f64 {
    let most = row.iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v as f64));
    most + row.iter().map(|v| (*v as f64 - most).exp()).sum::<f64>().ln()
}

/// `n` values in [-1, 1) from a fixed sequence, the same on each run: the range of the patch rows of an image.
pub fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / 8_388_608.0 - 1.0
        })
        .collect()
}

/// `n` token ids from a fixed sequence, below the vocabulary of the family. They are the same on each run.
pub fn ids(n: usize, seed: u64) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 120_000) as u32 + 1000
        })
        .collect()
}
