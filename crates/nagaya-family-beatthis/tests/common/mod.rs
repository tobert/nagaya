//! The oracle files of the tests, and the rule for a machine that has no copy of them.
//!
//! `tools/beatthis_golden.sh` writes the files. `BEATTHIS_GOLDEN_DIR` names their folder, and the code holds no
//! default path (AGENTS.md). An absent folder or file skips the test. `NAGAYA_REQUIRE_MODELS=1` makes it fail.
#![allow(dead_code)]

use std::io::ErrorKind;
use std::path::PathBuf;

use safetensors::{Dtype, SafeTensors};

pub const GOLDEN_VARIABLE: &str = "BEATTHIS_GOLDEN_DIR";
pub const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";
const HOW: &str = "run tools/beatthis_golden.sh and set BEATTHIS_GOLDEN_DIR to its output folder";

/// The oracle files: the bytes of `golden.safetensors` and the decoded `cases.json`.
pub struct Golden {
    bytes: Vec<u8>,
    pub cases: serde_json::Value,
}

fn skip(why: String) -> Option<Golden> {
    if std::env::var(REQUIRE_VARIABLE).is_ok_and(|v| v == "1") {
        panic!("{why}. {REQUIRE_VARIABLE} is set. {HOW}.");
    }
    let test = std::thread::current().name().unwrap_or("a test").to_owned();
    eprintln!("skipped: {test}: {why}; {HOW}");
    None
}

/// The oracle files, or `None` when they are absent and the test must return.
pub fn golden() -> Option<Golden> {
    let Some(dir) = std::env::var_os(GOLDEN_VARIABLE).map(PathBuf::from) else {
        return skip(format!("{GOLDEN_VARIABLE} is not set"));
    };
    let file = dir.join("golden.safetensors");
    let bytes = match std::fs::read(&file) {
        Ok(b) => b,
        Err(e) if e.kind() == ErrorKind::NotFound => return skip(format!("{} does not exist", file.display())),
        Err(e) => panic!("cannot read {}: {e}", file.display()),
    };
    let cases_file = dir.join("cases.json");
    let text =
        std::fs::read_to_string(&cases_file).unwrap_or_else(|e| panic!("cannot read {}: {e}", cases_file.display()));
    let cases = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{} is not JSON: {e}", cases_file.display()));
    Some(Golden { bytes, cases })
}

impl Golden {
    /// An f32 tensor by name, with its shape.
    pub fn f32(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let st = SafeTensors::deserialize(&self.bytes).expect("golden.safetensors does not parse");
        let t = st
            .tensor(name)
            .unwrap_or_else(|e| panic!("golden.safetensors has no {name}: {e}"));
        assert_eq!(t.dtype(), Dtype::F32, "{name} is not f32");
        let values = t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        (values, t.shape().to_vec())
    }

    /// An f64 tensor by name, with its shape.
    pub fn f64(&self, name: &str) -> (Vec<f64>, Vec<usize>) {
        let st = SafeTensors::deserialize(&self.bytes).expect("golden.safetensors does not parse");
        let t = st
            .tensor(name)
            .unwrap_or_else(|e| panic!("golden.safetensors has no {name}: {e}"));
        assert_eq!(t.dtype(), Dtype::F64, "{name} is not f64");
        let values = t
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect();
        (values, t.shape().to_vec())
    }

    /// The names of the cases, in the order of `cases.json`.
    pub fn case_names(&self) -> Vec<String> {
        self.cases["cases"]
            .as_object()
            .expect("cases.json has no cases")
            .keys()
            .cloned()
            .collect()
    }
}

/// The largest absolute difference of two slices of the same length.
pub fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "the lengths differ");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}
