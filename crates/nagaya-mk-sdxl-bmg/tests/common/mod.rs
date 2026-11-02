//! Helpers that the tests share: the oracle files of `tools/sdxl_golden.sh`, and the rule for a machine without them.
//!
//! An absent file skips the test with one line on stderr. `NAGAYA_REQUIRE_MODELS=1` makes absence a failure
//! (AGENTS.md, "Tests that need a model file").
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::PathBuf;

/// The tensors of one safetensors file, as f32 values with their shapes. Other dtypes are refused. An F64 tensor also
/// keeps its exact values, because a strength in f32 can give another step count (spec 11.9b).
pub struct Tensors(BTreeMap<String, (Vec<usize>, Vec<f32>)>, BTreeMap<String, Vec<f64>>);

impl Tensors {
    pub fn get(&self, name: &str) -> &(Vec<usize>, Vec<f32>) {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("the oracle has no tensor {name}"))
    }

    /// True when the file holds the tensor `name`.
    pub fn has(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    pub fn values(&self, name: &str) -> &[f32] {
        &self.get(name).1
    }

    /// The exact values of an F64 tensor.
    pub fn f64s(&self, name: &str) -> &[f64] {
        self.1
            .get(name)
            .unwrap_or_else(|| panic!("the oracle has no F64 tensor {name}"))
    }
}

/// The directory of the oracle: `SDXL_GOLDEN_DIR`, or `target/sdxl-golden` of the workspace.
pub fn golden_dir() -> PathBuf {
    std::env::var_os("SDXL_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/sdxl-golden"))
}

/// The f32 tensors of an oracle file, or `None` when it is absent and absence may skip.
pub fn golden(file: &str) -> Option<Tensors> {
    golden_in(&golden_dir(), file)
}

/// The f32 tensors of a file of the oracle directory `dir`, or `None` when it is absent and absence may skip.
pub fn golden_in(dir: &std::path::Path, file: &str) -> Option<Tensors> {
    let path = dir.join(file);
    if !path.is_file() {
        let require = std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1");
        assert!(
            !require,
            "NAGAYA_REQUIRE_MODELS is set, and {} is absent; run tools/sdxl_golden.sh",
            path.display()
        );
        eprintln!("skipped: {} is absent; run tools/sdxl_golden.sh", path.display());
        return None;
    }
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + len]).unwrap();
    let data = &bytes[8 + len..];
    let mut out = BTreeMap::new();
    let mut wide = BTreeMap::new();
    for (name, entry) in header.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let (start, end) = (
            entry["data_offsets"][0].as_u64().unwrap() as usize,
            entry["data_offsets"][1].as_u64().unwrap() as usize,
        );
        let values = match entry["dtype"].as_str().unwrap() {
            "F32" => data[start..end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            "I64" => data[start..end]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c) as f32)
                .collect(),
            "U8" => data[start..end].iter().map(|b| *b as f32).collect(),
            // The strengths and the schedule cases of tools/sdxl_inpaint_golden.sh.
            "F64" => data[start..end]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c) as f32)
                .collect(),
            other => panic!("tensor {name} has the dtype {other}, which the tests do not read"),
        };
        if entry["dtype"] == "F64" {
            let exact = data[start..end]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect();
            wide.insert(name.clone(), exact);
        }
        out.insert(name.clone(), (shape, values));
    }
    Some(Tensors(out, wide))
}

/// The largest absolute difference, relative to the largest absolute value of `want`.
pub fn rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(1e-30f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs())) / scale
}
