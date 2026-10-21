//! The tokenizer hash: the identity of the tokenizer that a GGUF file holds (spec 11.3, engine ABI "Identity").
//!
//! The golden values come from an independent Python writer of the same encoding. They are not outputs of the Rust
//! code, so a change to the encoding fails a test.

mod common;

use nagaya_text_qwen38::gguf::Metadata;
use nagaya_text_qwen38::{Error, tokenizer_hash};

/// One metadata entry for the in-memory file.
enum Entry<'a> {
    Str(&'a str),
    U32(u32),
    Bool(bool),
    Strings(&'a [&'a str]),
    I32s(&'a [i32]),
}

fn string(s: &str) -> Vec<u8> {
    let mut out = (s.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(s.as_bytes());
    out
}

/// The bytes of a GGUF file that holds only these entries, in this order.
fn file(entries: &[(&str, Entry)]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (key, entry) in entries {
        out.extend(string(key));
        match entry {
            Entry::Str(s) => {
                out.extend_from_slice(&8u32.to_le_bytes());
                out.extend(string(s));
            }
            Entry::U32(n) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&n.to_le_bytes());
            }
            Entry::Bool(b) => {
                out.extend_from_slice(&7u32.to_le_bytes());
                out.push(u8::from(*b));
            }
            Entry::Strings(items) => {
                out.extend_from_slice(&9u32.to_le_bytes());
                out.extend_from_slice(&8u32.to_le_bytes());
                out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for item in *items {
                    out.extend(string(item));
                }
            }
            Entry::I32s(items) => {
                out.extend_from_slice(&9u32.to_le_bytes());
                out.extend_from_slice(&5u32.to_le_bytes());
                out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for item in *items {
                    out.extend_from_slice(&item.to_le_bytes());
                }
            }
        }
    }
    out
}

fn read(bytes: &[u8]) -> Metadata {
    Metadata::read(bytes, bytes.len() as u64).expect("a built file")
}

/// The entries of the golden fixture. Its hash is `GOLDEN`.
fn base<'a>() -> Vec<(&'a str, Entry<'a>)> {
    vec![
        ("general.name", Entry::Str("a name that the hash ignores")),
        ("tokenizer.ggml.pre", Entry::Str("qwen35")),
        ("tokenizer.ggml.model", Entry::Str("gpt2")),
        ("tokenizer.ggml.tokens", Entry::Strings(&["a", "b", "<|x|>"])),
        ("tokenizer.ggml.token_type", Entry::I32s(&[1, 1, 3])),
        ("tokenizer.ggml.merges", Entry::Strings(&["a b"])),
        ("tokenizer.ggml.eos_token_id", Entry::U32(8)),
        ("tokenizer.ggml.bos_token_id", Entry::U32(7)),
        ("tokenizer.chat_template", Entry::Str("{{ messages }}")),
        ("tokenizer.ggml.add_bos_token", Entry::Bool(false)),
    ]
}

const GOLDEN: &str = "sha256:018a61bfe25a6ed49d01bfd9c5f1415e6f99f365646354005fa617aed73a59a8";

fn hash_of(entries: &[(&str, Entry)]) -> String {
    tokenizer_hash(&read(&file(entries))).expect("a hash")
}

#[test]
fn the_hash_of_the_fixture_is_the_golden_value() {
    if common::absent() {
        return;
    }
    assert_eq!(hash_of(&base()), GOLDEN);
}

#[test]
fn the_hash_of_the_real_file_is_the_golden_value() {
    if common::absent() {
        return;
    }
    let metadata = Metadata::open(&common::gguf_path()).expect("the model file");
    assert_eq!(
        tokenizer_hash(&metadata).expect("a hash"),
        "sha256:9724dd72094b6311776a1f5a8a49d283f676a5f0f56d8e0ba9671adb18df06ad"
    );
}

#[test]
fn the_order_of_the_entries_in_the_file_does_not_matter() {
    if common::absent() {
        return;
    }
    let mut entries = base();
    entries.reverse();
    assert_eq!(hash_of(&entries), GOLDEN);
}

#[test]
fn keys_outside_the_list_do_not_change_the_hash() {
    if common::absent() {
        return;
    }
    let mut entries = base();
    entries.retain(|(key, _)| *key != "tokenizer.chat_template" && *key != "tokenizer.ggml.add_bos_token");
    entries.push(("general.architecture", Entry::Str("other")));
    assert_eq!(hash_of(&entries), GOLDEN);
}

/// Replace the entry for `key`, or add it when the base has none.
fn with<'a>(key: &'a str, entry: Entry<'a>) -> Vec<(&'a str, Entry<'a>)> {
    let mut entries = base();
    entries.retain(|(name, _)| *name != key);
    entries.push((key, entry));
    entries
}

#[test]
fn each_key_in_the_list_changes_the_hash() {
    let cases = [
        with("tokenizer.ggml.model", Entry::Str("llama")),
        with("tokenizer.ggml.pre", Entry::Str("qwen2")),
        with("tokenizer.ggml.tokens", Entry::Strings(&["a", "b", "<|y|>"])),
        with("tokenizer.ggml.token_type", Entry::I32s(&[1, 1, 4])),
        with("tokenizer.ggml.merges", Entry::Strings(&["a c"])),
        with("tokenizer.ggml.bos_token_id", Entry::U32(9)),
        with("tokenizer.ggml.eos_token_id", Entry::U32(9)),
        with("tokenizer.ggml.eot_token_id", Entry::U32(1)),
        with("tokenizer.ggml.padding_token_id", Entry::U32(1)),
        with("tokenizer.ggml.unknown_token_id", Entry::U32(1)),
    ];
    for (index, entries) in cases.iter().enumerate() {
        assert_ne!(hash_of(entries), GOLDEN, "case {index}");
    }
}

#[test]
fn the_order_of_the_tokens_and_the_merges_changes_the_hash() {
    let tokens = with("tokenizer.ggml.tokens", Entry::Strings(&["b", "a", "<|x|>"]));
    assert_ne!(hash_of(&tokens), GOLDEN);
    let two = with("tokenizer.ggml.merges", Entry::Strings(&["a b", "b a"]));
    let swapped = with("tokenizer.ggml.merges", Entry::Strings(&["b a", "a b"]));
    assert_ne!(hash_of(&two), hash_of(&swapped));
}

#[test]
fn the_pieces_of_a_string_array_do_not_run_together() {
    let split = with("tokenizer.ggml.merges", Entry::Strings(&["a", "b"]));
    let joined = with("tokenizer.ggml.merges", Entry::Strings(&["ab"]));
    assert_ne!(hash_of(&split), hash_of(&joined));
}

#[test]
fn a_missing_required_key_is_an_error_that_names_it() {
    if common::absent() {
        return;
    }
    for key in [
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.merges",
    ] {
        let mut entries = base();
        entries.retain(|(name, _)| *name != key);
        match tokenizer_hash(&read(&file(&entries))) {
            Err(Error::Metadata(message)) => assert!(message.contains(key), "{message}"),
            other => panic!("expected a Metadata error for {key}, got {other:?}"),
        }
    }
}
