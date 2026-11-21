//! The header of the real checkpoint against the formats that this crate expands.

mod common;

use serde_json::Value;
use std::io::{Read, Seek, SeekFrom};

struct Safetensors {
    file: std::fs::File,
    header: Value,
    data_start: u64,
}

fn open() -> Safetensors {
    let path = common::weights_path();
    let mut file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
    let mut len = [0u8; 8];
    file.read_exact(&mut len).unwrap();
    let len = u64::from_le_bytes(len);
    let mut json = vec![0u8; len as usize];
    file.read_exact(&mut json).unwrap();
    Safetensors {
        file,
        header: serde_json::from_slice(&json).unwrap(),
        data_start: 8 + len,
    }
}

impl Safetensors {
    fn tensor(&self, name: &str) -> &Value {
        self.header
            .get(name)
            .unwrap_or_else(|| panic!("the checkpoint holds no tensor {name}"))
    }

    fn check(&self, name: &str, dtype: &str, shape: &[u64]) {
        let t = self.tensor(name);
        assert_eq!(t["dtype"], dtype, "dtype of {name}");
        let got: Vec<u64> = t["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(got, shape, "shape of {name}");
    }

    /// The bytes of `len` elements of one byte each, starting at element `start`. The tensor has one-byte elements.
    fn bytes(&mut self, name: &str, start: u64, len: u64) -> Vec<u8> {
        let begin = self.tensor(name)["data_offsets"][0].as_u64().unwrap();
        self.file
            .seek(SeekFrom::Start(self.data_start + begin + start))
            .unwrap();
        let mut out = vec![0u8; len as usize];
        self.file.read_exact(&mut out).unwrap();
        out
    }
}

fn experts(layer: usize, tensor: &str) -> String {
    format!("model.llm.layers.{layer}.mlp.experts.{tensor}")
}

#[test]
fn the_experts_of_all_24_layers_have_the_dtypes_and_shapes_of_the_formats() {
    if common::absent() {
        return;
    }
    let st = open();
    for layer in 0..24 {
        if layer < 20 {
            st.check(&experts(layer, "w13_weight"), "U8", &[128, 2304, 1152]);
            st.check(&experts(layer, "w13_weight_scale"), "F8_E4M3", &[128, 2304, 144]);
            st.check(&experts(layer, "w2_weight"), "U8", &[128, 2304, 576]);
            st.check(&experts(layer, "w2_weight_scale"), "F8_E4M3", &[128, 2304, 72]);
            for w in ["w13", "w2"] {
                st.check(&experts(layer, &format!("{w}_weight_amax")), "F32", &[128]);
                st.check(
                    &experts(layer, &format!("{w}_weight_outer_scale_denominator")),
                    "F32",
                    &[1],
                );
            }
        } else {
            st.check(&experts(layer, "w13_weight"), "F8_E4M3", &[128, 2304, 2304]);
            st.check(&experts(layer, "w13_weight_scale"), "U8", &[128, 2304, 72]);
            st.check(&experts(layer, "w2_weight"), "F8_E4M3", &[128, 2304, 1152]);
            st.check(&experts(layer, "w2_weight_scale"), "U8", &[128, 2304, 36]);
            for w in ["w13", "w2"] {
                assert!(
                    st.header.get(experts(layer, &format!("{w}_weight_amax"))).is_none(),
                    "layer {layer} has amax"
                );
            }
        }
    }
}

#[test]
fn a_sample_of_the_scales_holds_no_nan_code() {
    if common::absent() {
        return;
    }
    let mut st = open();
    for layer in 0..24 {
        for (w, per_expert) in [("w13", 2304 * 144), ("w2", 2304 * 72)] {
            let per_expert = if layer < 20 { per_expert } else { per_expert / 2 };
            let name = experts(layer, &format!("{w}_weight_scale"));
            for expert in [0u64, 63, 127] {
                for (i, code) in st.bytes(&name, expert * per_expert, per_expert).into_iter().enumerate() {
                    if layer < 20 {
                        assert!(
                            code & 0x7F != 0x7F,
                            "{name} expert {expert} index {i}: E4M3 NaN code {code:#04x}"
                        );
                    } else {
                        assert!(code != 255, "{name} expert {expert} index {i}: UE8M0 NaN code 255");
                    }
                }
            }
        }
    }
}

#[test]
fn a_sample_of_the_mxfp8_values_holds_no_nan_code() {
    if common::absent() {
        return;
    }
    let mut st = open();
    for layer in 20..24 {
        let name = experts(layer, "w13_weight");
        for expert in [0u64, 127] {
            let per_expert = 2304 * 2304;
            for (i, code) in st.bytes(&name, expert * per_expert, per_expert).into_iter().enumerate() {
                assert!(
                    code & 0x7F != 0x7F,
                    "{name} expert {expert} index {i}: E4M3 NaN code {code:#04x}"
                );
            }
        }
    }
}

#[test]
fn the_absent_rule_skips_only_on_absence() {
    use common::{Found, check, check_variable};
    let dir = std::env::temp_dir().join("nagaya-inkling2-absent-rule");
    let missing = dir.join(common::WEIGHTS_FILE);
    assert!(matches!(check(&missing, false, "t"), Found::Skip(_)));
    assert!(matches!(check_variable(None, false, "t"), Found::Skip(_)));
    assert!(
        std::panic::catch_unwind(|| check(&std::env::temp_dir().join("nagaya-inkling2-absent-rule/x"), true, "t"))
            .is_err()
    );
    assert!(std::panic::catch_unwind(|| check_variable(None, true, "t")).is_err());
    // A path that exists but is a folder is not a skip.
    assert!(std::panic::catch_unwind(|| check(&std::env::temp_dir(), false, "t")).is_err());
}
