//! The checkpoint and the oracle files of the tests, and the rule for a machine that has no copy of them.
//!
//! A machine does not hold every model, so this state is normal. An absent checkpoint or oracle file skips
//! the test. `EMBEDDINGGEMMA2_MODEL_DIR` names the checkpoint, and the code holds no default path (AGENTS.md).
//! `NAGAYA_REQUIRE_MODELS=1` makes an absent file fail the test. Only absence skips.
#![allow(dead_code)]

use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

pub const DIR_VARIABLE: &str = "EMBEDDINGGEMMA2_MODEL_DIR";
pub const GOLDEN_VARIABLE: &str = "EMBEDDINGGEMMA2_GOLDEN_DIR";
pub const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";

fn required() -> bool {
    std::env::var(REQUIRE_VARIABLE).is_ok_and(|v| v == "1")
}

/// True when `file` is absent and the test must return. It prints one line to stderr, or panics when
/// `NAGAYA_REQUIRE_MODELS=1`. A path that exists but is not a readable file panics.
fn file_absent(file: Option<PathBuf>, what: &str, how: &str) -> bool {
    let test = std::thread::current().name().unwrap_or("a test").to_owned();
    let skip = |why: String| {
        if required() {
            panic!("{why}. {REQUIRE_VARIABLE} is set. {how}.");
        }
        eprintln!("skipped: {test}: {why}; {how}");
        true
    };
    let Some(file) = file else {
        return skip(format!("{DIR_VARIABLE} is not set"));
    };
    match std::fs::metadata(&file) {
        Ok(m) if m.is_file() => false,
        Ok(_) => panic!("{} is not a file. {how}.", file.display()),
        Err(e) if e.kind() == ErrorKind::NotFound => skip(format!("the {what} {} does not exist", file.display())),
        Err(e) => panic!("cannot read the {what} {}: {e}", file.display()),
    }
}

fn configured() -> Option<PathBuf> {
    std::env::var_os(DIR_VARIABLE).map(PathBuf::from)
}

/// True when the checkpoint is absent. Call it first in each test that needs the checkpoint.
pub fn absent() -> bool {
    file_absent(
        configured().map(|dir| dir.join("model.safetensors")),
        "EmbeddingGemma 2 checkpoint file",
        "set EMBEDDINGGEMMA2_MODEL_DIR to the google/embeddinggemma-2 checkpoint directory",
    )
}

/// The checkpoint directory. A test that calls it has called `absent` before.
pub fn model_dir() -> PathBuf {
    let dir = configured().expect("the test must call `absent` first");
    assert!(
        dir.join("model.safetensors").is_file(),
        "the test must call `absent` first"
    );
    dir
}

/// The directory of the oracle files that `tools/embeddinggemma2_golden.sh` writes.
pub fn golden_dir() -> PathBuf {
    std::env::var_os(GOLDEN_VARIABLE).map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/embeddinggemma2-golden"),
        PathBuf::from,
    )
}

/// True when the oracle file `name` is absent. Call it after `absent`.
pub fn golden_absent(name: &str) -> bool {
    file_absent(
        Some(golden_dir().join(name)),
        "EmbeddingGemma 2 oracle file",
        "write it with tools/embeddinggemma2_golden.sh, with no GPU job",
    )
}

/// One tensor of an oracle file, as `f64` values with its shape. The oracle holds F32 and I64.
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

/// Read every tensor of the oracle file `name`.
pub fn golden(name: &str) -> HashMap<String, Tensor> {
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
            other => panic!("tensor {name} has dtype {other}; the oracle holds F32 and I64"),
        };
        out.insert(name.clone(), Tensor { shape, values });
    }
    out
}

/// The card of the test: the first card of `NAGAYA_TEST_CARDS`.
pub fn cards() -> Vec<String> {
    nagaya_mk_bmg::test_cards::cards(1)
}

/// The values of `v` divided by its L2 norm, as spec 11.10 normalizes: the squares summed in f64 in index order, each
/// value divided in f64, then rounded to f32.
pub fn normalized(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    v.iter().map(|x| (*x as f64 / norm) as f32).collect()
}

/// The largest absolute difference of two vectors.
pub fn max_abs(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "the lengths differ");
    got.iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((*g as f64 - *w as f64).abs()))
}

/// The cosine of two vectors, in f64.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

/// The relative L2 error of `got` against `want`.
pub fn rel(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "the lengths differ");
    let num: f64 = got.iter().zip(want).map(|(g, w)| (*g as f64 - *w as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|w| (*w as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}
