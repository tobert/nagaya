//! Compares the host quantizers with the bytes of the vLLM kernels of Inkling 2, bit for bit.
//!
//! The fixture `tests/fixtures/quant_kernels.safetensors` holds the inputs (bf16) and the outputs of the kernels that
//! `tools/inkling2_quant_fixture.py` ran on a GB10. A mismatch names the set, the row, the row class and the block.

use std::collections::BTreeMap;

use nagaya_mk_inkling2_sm121::quant::{MXFP8_BLOCK, NVFP4_BLOCK, quantize_mxfp8_ue8m0, quantize_nvfp4_fouroversix_mae};
use serde_json::Value;

struct Fixture {
    data: Vec<u8>,
    tensors: BTreeMap<String, (Vec<usize>, usize, usize)>,
    classes: Value,
}

impl Fixture {
    fn load() -> Self {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/quant_kernels.safetensors");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read the fixture {path}: {e}"));
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(&bytes[8..8 + header_len]).expect("the header is JSON");
        let classes: Value = serde_json::from_str(header["__metadata__"]["classes"].as_str().expect("classes"))
            .expect("the classes are JSON");
        let mut tensors = BTreeMap::new();
        for (name, entry) in header.as_object().unwrap() {
            if name == "__metadata__" {
                continue;
            }
            let shape = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let offsets = entry["data_offsets"].as_array().unwrap();
            let (a, b) = (
                offsets[0].as_u64().unwrap() as usize,
                offsets[1].as_u64().unwrap() as usize,
            );
            tensors.insert(name.clone(), (shape, a, b));
        }
        Self {
            data: bytes[8 + header_len..].to_vec(),
            tensors,
            classes,
        }
    }

    fn raw(&self, name: &str) -> (&[u8], &[usize]) {
        let (shape, a, b) = self
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("the fixture has no tensor {name}"));
        (&self.data[*a..*b], shape)
    }

    /// The rows of a bf16 tensor, widened to f32 (exact).
    fn rows_f32(&self, name: &str) -> Vec<Vec<f32>> {
        let (raw, shape) = self.raw(name);
        raw.chunks_exact(2 * shape[1])
            .map(|row| {
                row.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                    .collect()
            })
            .collect()
    }

    fn class(&self, set: &str, row: usize) -> &str {
        self.classes[set][row].as_str().expect("a class name")
    }
}

#[track_caller]
fn compare_bytes(f: &Fixture, set: &str, what: &str, row: usize, per_block: usize, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{set} row {row}: {what} length");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "{set} row {row} (class {}): {what} differ at block {} (byte {i}): got 0x{:02x}, kernel 0x{:02x}",
            f.class(set, row),
            i / per_block,
            got[i],
            want[i]
        );
    }
}

fn check_nvfp4(f: &Fixture, set: &str) {
    let xs = f.rows_f32(&format!("{set}.x"));
    let (values, vshape) = f.raw(&format!("{set}.values"));
    let (scales, sshape) = f.raw(&format!("{set}.scales"));
    let (amax, _) = f.raw(&format!("{set}.amax"));
    assert_eq!(xs.len(), vshape[0]);
    for (r, x) in xs.iter().enumerate() {
        let q = quantize_nvfp4_fouroversix_mae(x).unwrap_or_else(|e| panic!("{set} row {r}: {e}"));
        compare_bytes(
            f,
            set,
            "values",
            r,
            NVFP4_BLOCK / 2,
            &q.codes,
            &values[r * vshape[1]..(r + 1) * vshape[1]],
        );
        compare_bytes(
            f,
            set,
            "scales",
            r,
            1,
            &q.scales,
            &scales[r * sshape[1]..(r + 1) * sshape[1]],
        );
        let want = u32::from_le_bytes(amax[4 * r..4 * r + 4].try_into().unwrap());
        assert_eq!(
            q.amax.to_bits(),
            want,
            "{set} row {r} (class {}): amax bits differ",
            f.class(set, r)
        );
    }
}

#[test]
fn nvfp4_k2304_matches_the_kernel() {
    check_nvfp4(&Fixture::load(), "nvfp4_k2304");
}

#[test]
fn nvfp4_k1152_matches_the_kernel() {
    check_nvfp4(&Fixture::load(), "nvfp4_k1152");
}

#[test]
fn mxfp8_k2304_matches_the_kernel() {
    let f = Fixture::load();
    let set = "mxfp8_k2304";
    let xs = f.rows_f32(&format!("{set}.x"));
    let (values, vshape) = f.raw(&format!("{set}.values"));
    let (scales, sshape) = f.raw(&format!("{set}.scales"));
    for (r, x) in xs.iter().enumerate() {
        let q = quantize_mxfp8_ue8m0(x).unwrap_or_else(|e| panic!("{set} row {r}: {e}"));
        compare_bytes(
            &f,
            set,
            "values",
            r,
            MXFP8_BLOCK,
            &q.codes,
            &values[r * vshape[1]..(r + 1) * vshape[1]],
        );
        compare_bytes(
            &f,
            set,
            "scales",
            r,
            1,
            &q.scales,
            &scales[r * sshape[1]..(r + 1) * sshape[1]],
        );
    }
}
