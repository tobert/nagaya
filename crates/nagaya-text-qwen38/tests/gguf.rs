//! The GGUF metadata reader: a built file with every value type, bad files, and the real model file.

mod common;

use common::{JsonExt, fixture, gguf_path};
use nagaya_text_qwen38::Error;
use nagaya_text_qwen38::gguf::{Metadata, Value};

fn string(s: &str) -> Vec<u8> {
    let mut out = (s.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(s.as_bytes());
    out
}

fn header(version: u32, kv_count: u64) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&kv_count.to_le_bytes());
    out
}

fn kv(key: &str, value_type: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = string(key);
    out.extend_from_slice(&value_type.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

fn read(bytes: &[u8]) -> Result<Metadata, Error> {
    Metadata::read(bytes, bytes.len() as u64)
}

fn built_file() -> Vec<u8> {
    let mut f = header(3, 14);
    f.extend(kv("u8", 0, &[200]));
    f.extend(kv("i8", 1, &[0xFE]));
    f.extend(kv("u16", 2, &60000u16.to_le_bytes()));
    f.extend(kv("i16", 3, &(-300i16).to_le_bytes()));
    f.extend(kv("u32", 4, &4_000_000_000u32.to_le_bytes()));
    f.extend(kv("i32", 5, &(-2_000_000_000i32).to_le_bytes()));
    f.extend(kv("f32", 6, &1.5f32.to_le_bytes()));
    f.extend(kv("bool", 7, &[1]));
    f.extend(kv("str", 8, &string("h\u{e9}llo \u{4e16}")));
    let mut arr = 8u32.to_le_bytes().to_vec();
    arr.extend_from_slice(&2u64.to_le_bytes());
    arr.extend(string("a"));
    arr.extend(string("b c"));
    f.extend(kv("strs", 9, &arr));
    f.extend(kv("u64", 10, &u64::MAX.to_le_bytes()));
    f.extend(kv("i64", 11, &i64::MIN.to_le_bytes()));
    f.extend(kv("f64", 12, &(-0.25f64).to_le_bytes()));
    let mut nested = 9u32.to_le_bytes().to_vec();
    nested.extend_from_slice(&1u64.to_le_bytes());
    nested.extend_from_slice(&4u32.to_le_bytes());
    nested.extend_from_slice(&2u64.to_le_bytes());
    nested.extend_from_slice(&7u32.to_le_bytes());
    nested.extend_from_slice(&8u32.to_le_bytes());
    f.extend(kv("nested", 9, &nested));
    f
}

#[test]
fn reads_every_value_type_in_file_order() {
    let m = read(&built_file()).expect("a well-formed file");
    assert_eq!(m.len(), 14);
    assert_eq!(m.get("u8"), Some(&Value::U8(200)));
    assert_eq!(m.get("i8"), Some(&Value::I8(-2)));
    assert_eq!(m.get("u16"), Some(&Value::U16(60000)));
    assert_eq!(m.get("i16"), Some(&Value::I16(-300)));
    assert_eq!(m.get("u32"), Some(&Value::U32(4_000_000_000)));
    assert_eq!(m.get("i32"), Some(&Value::I32(-2_000_000_000)));
    assert_eq!(m.get("f32"), Some(&Value::F32(1.5)));
    assert_eq!(m.get("bool"), Some(&Value::Bool(true)));
    assert_eq!(m.get("str"), Some(&Value::Str("h\u{e9}llo \u{4e16}".to_string())));
    assert_eq!(
        m.get("strs"),
        Some(&Value::Array(vec![Value::Str("a".into()), Value::Str("b c".into())]))
    );
    assert_eq!(m.get("u64"), Some(&Value::U64(u64::MAX)));
    assert_eq!(m.get("i64"), Some(&Value::I64(i64::MIN)));
    assert_eq!(m.get("f64"), Some(&Value::F64(-0.25)));
    assert_eq!(
        m.get("nested"),
        Some(&Value::Array(vec![Value::Array(vec![Value::U32(7), Value::U32(8)])]))
    );
    assert_eq!(m.get("absent"), None);
}

#[test]
fn iterates_the_entries_in_file_order() {
    let m = read(&built_file()).expect("a well-formed file");
    let keys: Vec<&str> = m.iter().map(|(key, _)| key).collect();
    assert_eq!(
        keys,
        [
            "u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "str", "strs", "u64", "i64", "f64", "nested"
        ]
    );
    assert_eq!(m.iter().nth(1).map(|(_, value)| value), Some(&Value::I8(-2)));
}

#[test]
fn stops_after_the_last_entry() {
    // The reference reader reads the tensor records that follow the metadata, from the bytes that this reader leaves.
    let mut file = built_file();
    let metadata_bytes = file.len();
    file.extend_from_slice(b"tensor records");
    let mut rest: &[u8] = &file;
    let m = Metadata::read(&mut rest, file.len() as u64).expect("a well-formed file");
    assert_eq!(m.len(), 14);
    assert_eq!(rest, b"tensor records");
    assert_eq!(file.len() - rest.len(), metadata_bytes);
}

fn assert_gguf_error(bytes: &[u8], needle: &str) {
    match read(bytes) {
        Err(Error::Gguf(m)) => assert!(m.contains(needle), "message {m:?} should name {needle:?}"),
        other => panic!("expected a Gguf error naming {needle:?}, got {other:?}"),
    }
}

#[test]
fn refuses_a_bad_magic() {
    let mut f = built_file();
    f[0] = b'X';
    assert_gguf_error(&f, "magic");
}

#[test]
fn refuses_other_versions() {
    for version in [1u32, 2, 4] {
        let mut f = built_file();
        f[4..8].copy_from_slice(&version.to_le_bytes());
        assert_gguf_error(&f, "version");
    }
}

#[test]
fn refuses_a_file_that_ends_early() {
    let f = built_file();
    for cut in [0, 3, 10, 23, 30, f.len() - 1] {
        assert!(
            matches!(read(&f[..cut]), Err(Error::Gguf(_))),
            "a file cut at {cut} should be refused"
        );
    }
}

#[test]
fn refuses_more_entries_than_the_file_holds() {
    let mut f = built_file();
    f[16..24].copy_from_slice(&15u64.to_le_bytes());
    assert_gguf_error(&f, "ends");
}

#[test]
fn refuses_a_string_longer_than_the_file() {
    let mut f = header(3, 1);
    f.extend_from_slice(&u64::MAX.to_le_bytes());
    assert_gguf_error(&f, "longer");
}

#[test]
fn refuses_an_array_longer_than_the_file() {
    let mut payload = 4u32.to_le_bytes().to_vec();
    payload.extend_from_slice(&(1u64 << 40).to_le_bytes());
    let mut f = header(3, 1);
    f.extend(kv("big", 9, &payload));
    assert_gguf_error(&f, "longer");
}

#[test]
fn refuses_an_unknown_value_type() {
    let mut f = header(3, 1);
    f.extend(kv("odd", 13, &[0; 8]));
    assert_gguf_error(&f, "type");
}

#[test]
fn refuses_a_bool_that_is_not_zero_or_one() {
    let mut f = header(3, 1);
    f.extend(kv("b", 7, &[2]));
    assert_gguf_error(&f, "bool");
}

#[test]
fn refuses_text_that_is_not_utf8() {
    let mut f = header(3, 1);
    let mut payload = 2u64.to_le_bytes().to_vec();
    payload.extend_from_slice(&[0xC3, 0x28]);
    f.extend(kv("s", 8, &payload));
    assert_gguf_error(&f, "UTF-8");
}

#[test]
fn refuses_a_repeated_key() {
    let mut f = header(3, 2);
    f.extend(kv("k", 0, &[1]));
    f.extend(kv("k", 0, &[2]));
    assert_gguf_error(&f, "twice");
}

#[test]
fn refuses_an_array_nested_too_deep() {
    let mut payload = Vec::new();
    for _ in 0..40 {
        payload.extend_from_slice(&9u32.to_le_bytes());
        payload.extend_from_slice(&1u64.to_le_bytes());
    }
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&1u64.to_le_bytes());
    payload.push(0);
    let mut f = header(3, 1);
    f.extend(kv(
        "deep",
        9,
        &[&9u32.to_le_bytes()[..], &1u64.to_le_bytes()[..], &payload[..]].concat(),
    ));
    assert_gguf_error(&f, "deep");
}

#[test]
fn a_missing_file_names_its_path() {
    let err = Metadata::open(std::path::Path::new("/nonexistent/nagaya/model.gguf")).unwrap_err();
    assert!(matches!(err, Error::Io { .. }));
    assert!(err.to_string().contains("/nonexistent/nagaya/model.gguf"), "{err}");
}

#[test]
fn reads_the_model_file() {
    if common::absent() {
        return;
    }
    let m = Metadata::open(&gguf_path()).expect("the model file");
    let header = fixture("tokenizer.json");
    let header = header.get("header");
    assert_eq!(m.len(), 67);
    assert_eq!(m.get("tokenizer.ggml.pre"), Some(&Value::Str("qwen35".into())));
    assert_eq!(m.get("tokenizer.ggml.model"), Some(&Value::Str("gpt2".into())));
    let len = |key: &str| match m.get(key) {
        Some(Value::Array(a)) => a.len() as i128,
        other => panic!("{key} is not an array: {other:?}"),
    };
    assert_eq!(len("tokenizer.ggml.tokens"), header.get("n_tokens").int());
    assert_eq!(len("tokenizer.ggml.token_type"), header.get("n_tokens").int());
    assert_eq!(len("tokenizer.ggml.merges"), header.get("n_merges").int());
    assert!(matches!(m.get("tokenizer.chat_template"), Some(Value::Str(s)) if s.contains("<|im_start|>")));
}
