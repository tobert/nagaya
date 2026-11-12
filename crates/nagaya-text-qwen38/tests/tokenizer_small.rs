//! The tokenizer on small vocabularies built in memory: the order of merges, special tokens and the checks that
//! refuse a vocabulary.

mod common;

use common::{small_metadata, small_metadata_with_types};
use nagaya_text_qwen38::{Error, Tokenizer};

const NORMAL: i32 = 1;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;

fn letters() -> Vec<(&'static str, i32)> {
    vec![
        ("a", NORMAL),
        ("b", NORMAL),
        ("c", NORMAL),
        ("x", NORMAL),
        ("y", NORMAL),
    ]
}

fn build(extra: &[(&str, i32)], merges: &[&str]) -> Tokenizer {
    let mut tokens = letters();
    tokens.extend_from_slice(extra);
    Tokenizer::from_metadata(&small_metadata("qwen35", "gpt2", &tokens, None, merges)).expect("a usable vocabulary")
}

fn refused(metadata: nagaya_text_qwen38::gguf::Metadata, needle: &str) {
    match Tokenizer::from_metadata(&metadata) {
        Err(e) => assert!(e.to_string().contains(needle), "{e} should name {needle:?}"),
        Ok(_) => panic!("the vocabulary should be refused for {needle:?}"),
    }
}

#[test]
fn a_merge_of_lower_rank_goes_first() {
    // Ids: a=0 b=1 c=2 x=3 y=4 ab=5 bc=6 abc=7. "b c" has rank 0, so it goes before "a b" at rank 1.
    let t = build(
        &[("ab", NORMAL), ("bc", NORMAL), ("abc", NORMAL)],
        &["b c", "a b", "ab c"],
    );
    assert_eq!(t.encode("abc").unwrap(), [0, 6]);
}

#[test]
fn the_leftmost_pair_goes_first_when_ranks_are_equal() {
    let t = build(&[("aa", NORMAL)], &["a a"]);
    assert_eq!(t.encode("aaa").unwrap(), [5, 0]);
    assert_eq!(t.encode("aaaa").unwrap(), [5, 5]);
}

#[test]
fn a_repeated_merge_keeps_its_last_rank() {
    // As in the reference's dict: the pair "a b" ends with rank 2, so "b c" (rank 1) goes first.
    let t = build(&[("ab", NORMAL), ("bc", NORMAL)], &["a b", "b c", "a b"]);
    assert_eq!(t.encode("abc").unwrap(), [0, 6]);
}

#[test]
fn a_merge_result_that_is_not_a_token_is_an_error() {
    let t = build(&[], &["a b"]);
    match t.encode("ab") {
        Err(Error::Vocab(m)) => assert!(m.contains("ab"), "{m}"),
        other => panic!("expected a Vocab error, got {other:?}"),
    }
}

#[test]
fn the_longest_special_token_matches_first() {
    let t = build(&[("<|a|>", CONTROL), ("<|a|>b", USER_DEFINED)], &[]);
    assert_eq!(t.encode("<|a|>b").unwrap(), [6]);
    assert_eq!(t.encode("<|a|>c<|a|>b<|a|>").unwrap(), [5, 2, 6, 5]);
    assert_eq!(t.special_ids(), [5, 6]);
}

#[test]
fn a_normal_token_that_looks_special_is_plain_text() {
    // Ids: a=0 b=1 c=2 x=3 y=4 "<"=5 "|"=6 ">"=7 "<|b|>"=8. The token "<|b|>" is normal, so no merge builds it.
    let t = build(&[("<", NORMAL), ("|", NORMAL), (">", NORMAL), ("<|b|>", NORMAL)], &[]);
    assert_eq!(t.encode("<|b|>").unwrap(), [5, 6, 1, 6, 7]);
    assert!(t.special_ids().is_empty());
}

#[test]
fn escape_swaps_special_text_for_a_noncharacter_and_back() {
    let t = build(&[("<|a|>", CONTROL), ("<", NORMAL), ("|", NORMAL), (">", NORMAL)], &[]);
    assert_eq!(t.escape("x<|a|>y").unwrap(), "x\u{fdd0}y");
    assert_eq!(t.unescape("x\u{fdd0}y").unwrap(), "x<|a|>y");
    // Escaped text encodes the token text as plain characters: "<|" is one word, "a" another, "|>" a third.
    assert_eq!(t.encode_escaped("x\u{fdd0}y").unwrap(), [3, 6, 7, 0, 7, 8, 4]);
    // Raw text with the special text is the one special token.
    assert_eq!(t.encode("x<|a|>y").unwrap(), [3, 5, 4]);
}

#[test]
fn the_second_special_token_has_the_second_noncharacter() {
    let t = build(&[("<|a|>", CONTROL), ("<|b|>", USER_DEFINED)], &[]);
    assert_eq!(t.escape("<|b|><|a|>").unwrap(), "\u{fdd1}\u{fdd0}");
    assert!(matches!(t.unescape("\u{fdd2}"), Err(Error::Reserved { .. })));
}

#[test]
fn the_decoder_maps_bytes_through_the_gpt2_table() {
    let t = build(&[], &[]);
    assert_eq!(t.decode(&[0, 1, 2]).unwrap(), "abc");
    assert_eq!(t.decode(&[]).unwrap(), "");
}

#[test]
fn refuses_another_pre_tokenizer() {
    refused(small_metadata("llama-bpe", "gpt2", &letters(), None, &[]), "qwen35");
}

#[test]
fn refuses_another_model_kind() {
    refused(small_metadata("qwen35", "llama", &letters(), None, &[]), "gpt2");
}

#[test]
fn refuses_tokens_and_types_of_different_lengths() {
    refused(
        small_metadata("qwen35", "gpt2", &letters(), Some(&[1, 1]), &[]),
        "length",
    );
}

#[test]
fn refuses_a_token_text_that_is_in_the_vocabulary_twice() {
    let mut tokens = letters();
    tokens.push(("a", NORMAL));
    refused(small_metadata("qwen35", "gpt2", &tokens, None, &[]), "duplicate");
}

#[test]
fn refuses_a_merge_that_is_not_two_pieces() {
    refused(
        small_metadata("qwen35", "gpt2", &letters(), None, &["a b c"]),
        "merge 0",
    );
    refused(
        small_metadata("qwen35", "gpt2", &letters(), None, &["a b", "ab"]),
        "merge 1",
    );
}

#[test]
fn refuses_more_special_tokens_than_noncharacters() {
    let names: Vec<String> = (0..67).map(|i| format!("<|s{i}|>")).collect();
    let tokens: Vec<(&str, i32)> = names.iter().map(|n| (n.as_str(), CONTROL)).collect();
    refused(small_metadata("qwen35", "gpt2", &tokens, None, &[]), "special tokens");
    let ok: Vec<(&str, i32)> = names[..66].iter().map(|n| (n.as_str(), CONTROL)).collect();
    Tokenizer::from_metadata(&small_metadata("qwen35", "gpt2", &ok, None, &[])).expect("66 special tokens fit");
}

#[test]
fn refuses_metadata_without_a_key() {
    let empty = nagaya_text_qwen38::gguf::Metadata::default();
    match Tokenizer::from_metadata(&empty) {
        Err(Error::Metadata(m)) => assert!(m.contains("tokenizer.ggml.pre"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn sixty_six_special_tokens_use_the_noncharacters_in_order() {
    let names: Vec<String> = (0..66).map(|i| format!("<|s{i}|>")).collect();
    let tokens: Vec<(&str, i32)> = names.iter().map(|n| (n.as_str(), CONTROL)).collect();
    let t = Tokenizer::from_metadata(&small_metadata("qwen35", "gpt2", &tokens, None, &[])).unwrap();
    // The order of Unicode noncharacters in the reference: U+FDD0 to U+FDEF, then U+FFFE and U+FFFF of planes 0 to 16.
    let mut expected: Vec<char> = (0xFDD0u32..=0xFDEF).map(|c| char::from_u32(c).unwrap()).collect();
    for plane in 0..17u32 {
        expected.push(char::from_u32(plane << 16 | 0xFFFE).unwrap());
        expected.push(char::from_u32(plane << 16 | 0xFFFF).unwrap());
    }
    assert_eq!(expected.len(), 66);
    for (i, name) in names.iter().enumerate() {
        let escaped = t.escape(name).unwrap();
        assert_eq!(escaped, expected[i].to_string(), "escape of special token {i}");
        assert_eq!(t.unescape(&escaped).unwrap(), *name);
    }
    let all: String = expected.iter().collect();
    assert_eq!(t.unescape(&all).unwrap(), names.concat());
    assert_eq!(t.encode(&names.concat()).unwrap(), (0..66).collect::<Vec<u32>>());
}

/// The element types of GGUF that hold integers, each with a function that writes a value as its bytes.
type Writer = fn(u8) -> Vec<u8>;

fn integer_types() -> Vec<(&'static str, u32, Writer)> {
    vec![
        ("U8", 0, |v| vec![v]),
        ("I8", 1, |v| vec![v]),
        ("U16", 2, |v| (v as u16).to_le_bytes().to_vec()),
        ("I16", 3, |v| (v as i16).to_le_bytes().to_vec()),
        ("U32", 4, |v| (v as u32).to_le_bytes().to_vec()),
        ("I32", 5, |v| (v as i32).to_le_bytes().to_vec()),
        ("U64", 10, |v| (v as u64).to_le_bytes().to_vec()),
        ("I64", 11, |v| (v as i64).to_le_bytes().to_vec()),
    ]
}

#[test]
fn reads_token_types_of_each_integer_type() {
    let texts = ["a", "b", "<|s|>"];
    for (name, kind, write) in integer_types() {
        let values = [write(1), write(1), write(CONTROL as u8)];
        let m = small_metadata_with_types("qwen35", "gpt2", &texts, kind, &values, &[]);
        let t = Tokenizer::from_metadata(&m).unwrap_or_else(|e| panic!("token_type of {name}: {e}"));
        assert_eq!(t.special_ids(), [2], "token_type of {name}");
    }
}

#[test]
fn refuses_token_types_that_are_floats_and_names_the_type() {
    for (name, kind, bytes) in [
        ("F32", 6u32, 1.0f32.to_le_bytes().to_vec()),
        ("F64", 12, 1.0f64.to_le_bytes().to_vec()),
    ] {
        let values = [bytes.clone(), bytes];
        let m = small_metadata_with_types("qwen35", "gpt2", &["a", "b"], kind, &values, &[]);
        match Tokenizer::from_metadata(&m) {
            Err(Error::Metadata(msg)) => {
                assert!(msg.contains("tokenizer.ggml.token_type") && msg.contains(name), "{msg}");
                assert!(msg.contains("float"), "{msg}");
            }
            other => panic!("token_type of {name}: {other:?}"),
        }
    }
}

#[test]
fn refuses_a_token_type_that_does_not_fit_a_signed_64_bit_integer() {
    let values = [u64::MAX.to_le_bytes().to_vec()];
    let m = small_metadata_with_types("qwen35", "gpt2", &["a"], 10, &values, &[]);
    match Tokenizer::from_metadata(&m) {
        Err(Error::Metadata(msg)) => assert!(msg.contains("does not fit"), "{msg}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn refuses_a_special_token_with_an_empty_text_and_names_its_id() {
    // The reference builds an empty alternative here and then splits the text at every position. This reader refuses it.
    for kind in [CONTROL, USER_DEFINED] {
        let tokens = [("a", NORMAL), ("<|s|>", CONTROL), ("", kind)];
        match Tokenizer::from_metadata(&small_metadata("qwen35", "gpt2", &tokens, None, &[])) {
            Err(Error::Vocab(m)) => assert!(m.contains("token 2") && m.contains("empty"), "{m}"),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn an_empty_text_is_fine_for_a_token_that_is_not_special() {
    let tokens = [("a", NORMAL), ("", NORMAL)];
    Tokenizer::from_metadata(&small_metadata("qwen35", "gpt2", &tokens, None, &[])).expect("a normal empty token");
}
