//! The detokenizer against the outputs of the reference's Python (`service/chatfmt.py` `Utf8Stream`,
//! `service/common.py` `piece_text`).

mod common;

use common::{JsonExt, fixture, tokenizer};
use nagaya_text_qwen38::detok::{Utf8Stream, piece_text};
use nagaya_text_qwen38::json::Json;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn strings(j: &Json) -> Vec<&str> {
    j.items().iter().map(|s| s.str()).collect()
}

fn run(chunks: &[Vec<u8>]) -> (Vec<String>, String) {
    let mut stream = Utf8Stream::new();
    let outputs = chunks.iter().map(|c| stream.push(c)).collect();
    (outputs, stream.flush())
}

#[test]
fn a_stream_of_bytes_gives_the_text_of_the_reference() {
    let f = fixture("utf8.json");
    let cases = f.get("streams").items();
    assert!(cases.len() >= 36);
    for case in cases {
        let name = case.get("name").str();
        let chunks: Vec<Vec<u8>> = case.get("chunks").items().iter().map(|c| unhex(c.str())).collect();
        let (outputs, flush) = run(&chunks);
        assert_eq!(outputs, strings(case.get("outputs")), "outputs of {name}");
        assert_eq!(flush, case.get("flush").str(), "flush of {name}");
    }
}

#[test]
fn a_stream_of_token_pieces_gives_the_text_of_the_reference() {
    if common::absent() {
        return;
    }
    let f = fixture("utf8.json");
    let cases = f.get("token_streams").items();
    assert!(cases.len() >= 9);
    for case in cases {
        let name = case.get("name").str();
        let special = case.has("special") && matches!(case.get("special"), Json::Bool(true));
        let chunks: Vec<Vec<u8>> = case
            .get("ids")
            .ids()
            .iter()
            .map(|&id| tokenizer().token_piece(id, special).unwrap())
            .collect();
        let (outputs, flush) = run(&chunks);
        assert_eq!(outputs, strings(case.get("outputs")), "outputs of {name}");
        assert_eq!(flush, case.get("flush").str(), "flush of {name}");
    }
}

#[test]
fn a_cut_anywhere_gives_the_same_text() {
    let f = fixture("utf8.json");
    for case in f.get("streams").items() {
        let name = case.get("name").str();
        let all: Vec<u8> = case.get("chunks").items().iter().flat_map(|c| unhex(c.str())).collect();
        let whole = String::from_utf8_lossy(&all).into_owned();
        for cut in 0..=all.len() {
            let (outputs, flush) = run(&[all[..cut].to_vec(), all[cut..].to_vec()]);
            assert_eq!(outputs.concat() + &flush, whole, "{name} cut at {cut}");
        }
    }
}

#[test]
fn a_split_character_comes_out_with_the_byte_that_completes_it() {
    let mut s = Utf8Stream::new();
    assert_eq!(s.push(&[0xE4, 0xB8]), "");
    assert_eq!(s.push(&[0x96, b'x']), "\u{4e16}x");
    assert_eq!(s.flush(), "");
}

#[test]
fn flush_reports_an_unfinished_character_once() {
    let mut s = Utf8Stream::new();
    assert_eq!(s.push(&[b'a', 0xF0, 0x9F]), "a");
    assert_eq!(s.flush(), "\u{fffd}");
    assert_eq!(s.flush(), "");
}

#[test]
fn piece_text_gives_the_text_of_the_reference() {
    let f = fixture("utf8.json");
    let cases = f.get("piece_text").items();
    assert!(cases.len() >= 18);
    for case in cases {
        let name = case.get("name").str();
        assert_eq!(
            piece_text(&unhex(case.get("hex").str())),
            case.get("text").str(),
            "{name}"
        );
    }
}
