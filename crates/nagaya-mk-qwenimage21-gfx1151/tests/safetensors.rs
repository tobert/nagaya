//! The strict safetensors reader. These tests need no GPU and no model.

use nagaya_mk_qwenimage21_gfx1151::safetensors::{Dtype, Safetensors, write_bytes};

fn file(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.safetensors");
    std::fs::write(&path, bytes).unwrap();
    (dir, path)
}

#[test]
fn a_tensor_reads_back_with_its_metadata() {
    let a: Vec<u8> = (0..12).collect();
    let b: Vec<u8> = (100..108).collect();
    let bytes = write_bytes(
        &[("a", "BF16", &[2, 3], &a), ("b", "F32", &[2], &b)],
        &[("request", "{\"x\": 1}")],
    );
    let (_dir, path) = file(&bytes);
    let st = Safetensors::open(&path).unwrap();
    assert_eq!(st.read("a", Dtype::Bf16, &[2, 3]).unwrap(), a);
    assert_eq!(st.read("b", Dtype::F32, &[2]).unwrap(), b);
    assert_eq!(st.metadata()["request"], "{\"x\": 1}");
    assert_eq!(st.tensors().len(), 2);
}

#[test]
fn a_read_with_another_dtype_or_shape_is_refused() {
    let a = vec![0u8; 12];
    let (_dir, path) = file(&write_bytes(&[("a", "BF16", &[2, 3], &a)], &[]));
    let st = Safetensors::open(&path).unwrap();
    assert!(
        st.read("a", Dtype::F32, &[2, 3])
            .unwrap_err()
            .to_string()
            .contains("needs F32")
    );
    assert!(st.read("a", Dtype::Bf16, &[3, 2]).is_err());
    assert!(
        st.read("missing", Dtype::Bf16, &[3, 2])
            .unwrap_err()
            .to_string()
            .contains("no tensor `missing`")
    );
}

/// A file whose header is `header`, followed by `data` bytes of zeros.
fn raw(header: &str, data: usize) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend(std::iter::repeat_n(0u8, data));
    out
}

#[test]
fn a_bad_header_is_an_error() {
    let cases = [
        (
            r#"{"a": {"dtype": "Q4", "shape": [2], "data_offsets": [0, 4]}}"#,
            4,
            "dtype",
        ),
        (
            r#"{"a": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 4]}}"#,
            4,
            "does not hold",
        ),
        (
            r#"{"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}"#,
            2,
            "does not hold",
        ),
        (
            r#"{"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                "b": {"dtype": "BF16", "shape": [2], "data_offsets": [2, 6]}}"#,
            6,
            "overlap",
        ),
        (r#"[1, 2]"#, 0, "not an object"),
    ];
    for (header, data, what) in cases {
        let (_dir, path) = file(&raw(header, data));
        let error = Safetensors::open(&path)
            .err()
            .unwrap_or_else(|| panic!("{header} opened"))
            .to_string();
        assert!(error.contains(what), "{header}: {error}");
    }
    let (_dir, path) = file(&9_999u64.to_le_bytes());
    assert!(Safetensors::open(&path).is_err(), "a header past the end");
}
