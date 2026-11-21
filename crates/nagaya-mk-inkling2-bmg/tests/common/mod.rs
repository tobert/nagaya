//! The model folder of the tests, and the rule for a machine that has no copy of it.
//!
//! A machine does not hold every model, so this state is normal. An absent file skips the test.
//! The variable `NAGAYA_REQUIRE_MODELS=1` makes an absent file fail the test. Only absence skips.
#![allow(dead_code)]

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// The variable that names the folder of Inkling 2 Mini. It has no default.
pub const MODEL_VARIABLE: &str = "INKLING2_MODEL_DIR";
pub const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";
/// The one file of the weights in the folder.
pub const WEIGHTS_FILE: &str = "model-00001-of-00001.safetensors";

/// What the check found for the weights file.
#[derive(Debug, PartialEq, Eq)]
pub enum Found {
    Present,
    Skip(String),
}

/// Check the weights file at `path`. An absent file skips, or panics when `require` is set. Any other error panics.
pub fn check(path: &Path, require: bool, test: &str) -> Found {
    match std::fs::metadata(path) {
        Ok(m) if m.is_file() => Found::Present,
        Ok(_) => panic!(
            "the weights path {} is not a file. Set {MODEL_VARIABLE} to the model folder.",
            path.display()
        ),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            let what = format!("the weights file {} does not exist", path.display());
            if require {
                panic!("{what}. {REQUIRE_VARIABLE} is set. Put the file there, or set {MODEL_VARIABLE}.");
            }
            Found::Skip(format!("skipped: {test}: {what}; set {MODEL_VARIABLE}"))
        }
        Err(e) => panic!("cannot read the weights file {}: {e}", path.display()),
    }
}

/// Check the file in the folder that the variable names. An unset variable skips, and it panics when `require` is set.
pub fn check_variable(dir: Option<PathBuf>, require: bool, test: &str) -> Found {
    match dir {
        Some(dir) => check(&dir.join(WEIGHTS_FILE), require, test),
        None if require => {
            panic!(
                "{MODEL_VARIABLE} is not set, and {REQUIRE_VARIABLE} is set. Set it to the folder of Inkling 2 Mini."
            )
        }
        None => Found::Skip(format!("skipped: {test}: {MODEL_VARIABLE} is not set")),
    }
}

fn configured() -> Option<PathBuf> {
    std::env::var_os(MODEL_VARIABLE).map(PathBuf::from)
}

fn required() -> bool {
    std::env::var(REQUIRE_VARIABLE).is_ok_and(|v| v == "1")
}

/// True when the weights file is absent and the test must return. It prints one line to stderr.
/// Call it first in each test that needs the model.
pub fn absent() -> bool {
    let thread = std::thread::current();
    match check_variable(configured(), required(), thread.name().unwrap_or("a test")) {
        Found::Present => false,
        Found::Skip(line) => {
            eprintln!("{line}");
            true
        }
    }
}

/// The path of the weights file. A test that calls it has called `absent` before.
pub fn weights_path() -> PathBuf {
    configured()
        .expect("the test must call `absent` first")
        .join(WEIGHTS_FILE)
}

/// The variable that names the folder of the variable-loop research model. It has no default.
pub const VARLOOP_VARIABLE: &str = "INKLING2_VARLOOP_MODEL_DIR";
/// The variable that names the folder of the oracle files. It defaults to `target/inkling2-golden` of the workspace.
pub const GOLDEN_VARIABLE: &str = "INKLING2_GOLDEN_DIR";

/// The two models, with the variable of each folder and the label of its oracle files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Mini,
    Varloop,
}

impl Which {
    pub const BOTH: [Which; 2] = [Which::Mini, Which::Varloop];

    pub fn variable(self) -> &'static str {
        match self {
            Which::Mini => MODEL_VARIABLE,
            Which::Varloop => VARLOOP_VARIABLE,
        }
    }

    /// The label of the oracle files: `inkling2-mini` or `inkling2-varloop`.
    pub fn label(self) -> &'static str {
        match self {
            Which::Mini => "inkling2-mini",
            Which::Varloop => "inkling2-varloop",
        }
    }
}

/// True when `file` is absent and the test must return. It prints one line to stderr, or panics when
/// `NAGAYA_REQUIRE_MODELS=1`. A path that exists but is not a file panics.
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

/// True when the weights of `which` are absent. Call it first in each test that needs the model.
pub fn model_absent(which: Which) -> bool {
    let variable = which.variable();
    file_absent(
        std::env::var_os(variable).map(|d| PathBuf::from(d).join(WEIGHTS_FILE)),
        variable,
        &format!("set {variable} to the model folder"),
    )
}

/// The weights file of `which`. A test that calls it has called `model_absent` before.
pub fn model_weights(which: Which) -> PathBuf {
    PathBuf::from(std::env::var_os(which.variable()).expect("the test must call `model_absent` first"))
        .join(WEIGHTS_FILE)
}

/// The folder of the oracle files: `INKLING2_GOLDEN_DIR`, or `target/inkling2-golden` of the workspace.
pub fn golden_dir() -> PathBuf {
    std::env::var_os(GOLDEN_VARIABLE)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/inkling2-golden"))
}

/// True when the oracle file `name` is absent. Call it after `model_absent`.
pub fn golden_absent(name: &str) -> bool {
    file_absent(
        Some(golden_dir().join(name)),
        GOLDEN_VARIABLE,
        "write it with tools/inkling2_golden.sh, with no GPU job",
    )
}

/// The JSON oracle file `name`.
pub fn golden_json(name: &str) -> serde_json::Value {
    let path = golden_dir().join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{} is not JSON: {e}", path.display()))
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

/// Read every tensor of the safetensors oracle file `name`.
pub fn golden_tensors(name: &str) -> std::collections::HashMap<String, Tensor> {
    let path = golden_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).expect("the header is JSON");
    let data = &bytes[8 + n..];
    let mut out = std::collections::HashMap::new();
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

/// The worst relative L2 error of the rows of two `rows x width` tensors.
pub fn worst_row(got: &[f32], want: &[f32], width: usize) -> f64 {
    assert_eq!(got.len(), want.len(), "the lengths differ");
    got.chunks_exact(width)
        .zip(want.chunks_exact(width))
        .map(|(g, w)| rel(g, w))
        .fold(0.0, f64::max)
}
