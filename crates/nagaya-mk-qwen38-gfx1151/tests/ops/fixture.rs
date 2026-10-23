//! Read the fixtures of the op tests, which `tools/op_fixtures.py` writes.
//!
//! A group is a pair of files in `tests/fixtures/ops`: `<group>.json` holds the header, the metadata and an index of
//! arrays, and `<group>.bin` holds the bytes of the arrays. Every array is little-endian and has a type and a shape.

#![allow(dead_code)]

use serde_json::Value;
use std::path::PathBuf;

pub fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ops")
}

pub struct Fixture {
    pub name: String,
    pub json: Value,
    blob: Vec<u8>,
}

impl Fixture {
    pub fn load(group: &str) -> Fixture {
        let dir = directory();
        let text = std::fs::read_to_string(dir.join(format!("{group}.json")))
            .unwrap_or_else(|error| panic!("read the fixture {group}.json: {error}. Run tools/op_fixtures.py."));
        let blob = std::fs::read(dir.join(format!("{group}.bin")))
            .unwrap_or_else(|error| panic!("read the fixture {group}.bin: {error}. Run tools/op_fixtures.py."));
        let json: Value = serde_json::from_str(&text).unwrap_or_else(|error| panic!("parse {group}.json: {error}"));
        Fixture {
            name: group.to_owned(),
            json,
            blob,
        }
    }

    /// A value of the metadata, by name.
    pub fn meta(&self, name: &str) -> &Value {
        self.json["meta"]
            .get(name)
            .unwrap_or_else(|| panic!("the fixture {} has no metadata {name}", self.name))
    }

    fn entry(&self, name: &str, dtype: &str) -> (&[u8], Vec<usize>) {
        let entry = self.json["arrays"]
            .get(name)
            .unwrap_or_else(|| panic!("the fixture {} has no array {name}", self.name));
        assert_eq!(
            entry["dtype"], dtype,
            "the array {name} of {} has another type",
            self.name
        );
        let offset = entry["offset"].as_u64().expect("offset") as usize;
        let len = entry["len"].as_u64().expect("len") as usize;
        let shape = entry["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|n| n.as_u64().expect("dimension") as usize)
            .collect();
        (&self.blob[offset..offset + len], shape)
    }

    pub fn f32s(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let (bytes, shape) = self.entry(name, "f32");
        (
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            shape,
        )
    }

    pub fn f32_flat(&self, name: &str) -> Vec<f32> {
        self.f32s(name).0
    }

    pub fn u32s(&self, name: &str) -> (Vec<u32>, Vec<usize>) {
        let (bytes, shape) = self.entry(name, "u32");
        (
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect(),
            shape,
        )
    }

    pub fn i32s(&self, name: &str) -> (Vec<i32>, Vec<usize>) {
        let (bytes, shape) = self.entry(name, "i32");
        (
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .collect(),
            shape,
        )
    }

    pub fn bytes(&self, name: &str) -> Vec<u8> {
        self.entry(name, "u8").0.to_vec()
    }
}
