//! The checkpoint and the reference files of the tests, and the rule for a machine that has no copy of them.
//!
//! Amy, 2026-10-21: "adjust the tests so they don't break our tests when we don't have a model downloaded locally.
//! we won't replicate all models to all machines so this state is normal." An absent checkpoint or reference file
//! skips the test. `QWEN3TTS_MODEL_DIR` names the checkpoint, and the code holds no default path (AGENTS.md).
//! `NAGAYA_REQUIRE_MODELS=1` makes an absent file fail the test. Only absence skips.
#![allow(dead_code)]

use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

pub const DIR_VARIABLE: &str = "QWEN3TTS_MODEL_DIR";
pub const GOLDEN_VARIABLE: &str = "QWEN3TTS_GOLDEN_DIR";
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
        "Qwen3-TTS checkpoint file",
        "set QWEN3TTS_MODEL_DIR to the Qwen3-TTS VoiceDesign checkpoint directory",
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

/// The directory of the reference files that `tools/qwen3tts_golden.sh` writes.
pub fn golden_dir() -> PathBuf {
    std::env::var_os(GOLDEN_VARIABLE).map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/qwen3tts-golden"),
        PathBuf::from,
    )
}

/// True when the reference file `name` is absent. Call it after `absent`.
pub fn golden_absent(name: &str) -> bool {
    file_absent(
        Some(golden_dir().join(name)),
        "Qwen3-TTS reference file",
        "write it with tools/qwen3tts_golden.sh, with no GPU job",
    )
}

/// One tensor of a reference file, as `f64` values with its shape. The reference files hold F32 and I64.
pub struct Tensor {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}

impl Tensor {
    pub fn f32(&self) -> Vec<f32> {
        self.values.iter().map(|v| *v as f32).collect()
    }

    /// The row `r` of a tensor whose last dimension is `width`.
    pub fn row(&self, r: usize, width: usize) -> Vec<f32> {
        self.values[r * width..(r + 1) * width]
            .iter()
            .map(|v| *v as f32)
            .collect()
    }
}

/// Read every tensor of the reference file `name`.
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
            other => panic!("tensor {name} has dtype {other}; the reference files hold F32 and I64"),
        };
        out.insert(name.clone(), Tensor { shape, values });
    }
    out
}

/// The relative L2 error of `got` against `want`.
pub fn rel(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "the lengths differ");
    let num: f64 = got.iter().zip(want).map(|(g, w)| (*g as f64 - *w as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|w| (*w as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// The index of the largest value.
pub fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold(
            (0, f32::NEG_INFINITY),
            |best, (i, v)| if *v > best.1 { (i, *v) } else { best },
        )
        .0
}

/// The PCI addresses of the `xe` cards, in PCI order.
pub fn cards() -> Vec<String> {
    let mut found: Vec<String> = std::fs::read_dir("/sys/bus/pci/drivers/xe")
        .expect("the xe driver directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.len() == 12 && name.as_bytes()[4] == b':')
        .collect();
    found.sort();
    found
}
