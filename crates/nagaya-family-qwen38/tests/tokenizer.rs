//! The tokenizer against the outputs of the reference's Python (`tools/qwen38_reference/tokenizer.py`), id for id.

mod common;

use common::{JsonExt, fixture, tokenizer};
use nagaya_family_qwen38::Error;
use nagaya_family_qwen38::json::Json;
use nagaya_family_qwen38::tokenizer::pretokenize;
use std::sync::OnceLock;

fn tokenizer_fixture() -> &'static Json {
    static F: OnceLock<Json> = OnceLock::new();
    F.get_or_init(|| fixture("tokenizer.json"))
}

fn first_difference(got: &[u32], want: &[u32]) -> String {
    let at = got
        .iter()
        .zip(want)
        .position(|(a, b)| a != b)
        .unwrap_or(got.len().min(want.len()));
    format!(
        "first difference at index {at}: got {:?}, want {:?}",
        got.get(at),
        want.get(at)
    )
}

#[test]
fn the_vocabulary_matches_the_fixture() {
    if common::absent() {
        return;
    }
    let tok = tokenizer();
    let f = tokenizer_fixture();
    assert_eq!(tok.len() as i128, f.get("header").get("n_tokens").int());
    let info = f.get("token_info");
    assert_eq!(tok.special_ids(), info.get("special_ids").ids().as_slice());
    for (id, text) in tok.special_ids().iter().zip(info.get("special_texts").items()) {
        assert_eq!(tok.token_text(*id).unwrap(), text.str());
    }
    for (kind, count) in match info.get("type_counts") {
        Json::Object(o) => o,
        _ => panic!("type_counts"),
    } {
        let kind: i64 = kind.parse().unwrap();
        let n = (0..tok.len() as u32)
            .filter(|&i| tok.token_type(i).unwrap() == kind)
            .count();
        assert_eq!(n as i128, count.int(), "tokens of type {kind}");
    }
}

#[test]
fn pretokenize_splits_like_the_pattern_of_the_reference() {
    let cases = tokenizer_fixture().get("pretok").items();
    assert!(cases.len() > 300);
    for case in cases {
        let text = case.get("text").str();
        let want: Vec<&str> = case.get("words").items().iter().map(|w| w.str()).collect();
        assert_eq!(pretokenize(text), want, "case {}", case.get("name").str());
    }
}

#[test]
fn encode_gives_the_ids_of_the_reference() {
    if common::absent() {
        return;
    }
    let cases = tokenizer_fixture().get("encode").items();
    assert!(cases.len() > 400);
    for case in cases {
        let name = case.get("name").str();
        let want = case.get("ids").ids();
        let got = tokenizer()
            .encode(case.get("text").str())
            .unwrap_or_else(|e| panic!("case {name}: {e}"));
        assert!(got == want, "case {name}: {}", first_difference(&got, &want));
    }
}

/// The error is `Reserved` with the character and the offset that the reference names, and its message says them.
fn assert_reserved(name: &str, error: &Error, case: &Json) {
    let want_char = char::from_u32(u32::from_str_radix(case.get("char").str(), 16).unwrap()).unwrap();
    let want_offset = case.get("offset").int() as usize;
    match error {
        Error::Reserved { ch, offset } => {
            assert_eq!(
                (*ch, *offset),
                (want_char, want_offset),
                "case {name}: the reference says {:?}",
                case.get("error").str()
            );
        }
        other => panic!("case {name}: expected Reserved, got {other:?}"),
    }
    let text = error.to_string();
    assert!(
        text.contains(&format!(
            "U+{} at offset {want_offset} is a Unicode noncharacter",
            case.get("char").str()
        )),
        "case {name}: {text}"
    );
}

#[test]
fn encode_refuses_each_noncharacter() {
    if common::absent() {
        return;
    }
    let cases = tokenizer_fixture().get("encode_refused").items();
    assert_eq!(cases.len(), 69);
    for case in cases {
        let name = case.get("name").str();
        match tokenizer().encode(case.get("text").str()) {
            Err(e) => assert_reserved(name, &e, case),
            other => panic!("case {name}: expected a Reserved error, got {other:?}"),
        }
    }
}

#[test]
fn a_noncharacter_error_names_the_character_and_its_offset() {
    if common::absent() {
        return;
    }
    let err = tokenizer().encode("ab\u{fdd0}").unwrap_err();
    assert!(
        matches!(
            err,
            Error::Reserved {
                ch: '\u{fdd0}',
                offset: 2
            }
        ),
        "{err:?}"
    );
    assert!(err.to_string().contains("U+FDD0"), "{err}");
}

#[test]
fn escape_and_unescape_give_the_text_of_the_reference() {
    if common::absent() {
        return;
    }
    let cases = tokenizer_fixture().get("escape").items();
    assert!(cases.len() > 40);
    for case in cases {
        let text = case.get("text").str();
        let escaped = tokenizer().escape(text).unwrap();
        assert_eq!(escaped, case.get("escaped").str(), "escape of {text:?}");
        assert_eq!(
            tokenizer().unescape(&escaped).unwrap(),
            case.get("unescaped").str(),
            "unescape of {text:?}"
        );
        let want = case.get("ids").ids();
        let got = tokenizer().encode_escaped(&escaped).unwrap();
        assert!(
            got == want,
            "encode_escaped of {escaped:?}: {}",
            first_difference(&got, &want)
        );
    }
}

#[test]
fn each_escape_character_encodes_as_the_text_of_its_special_token() {
    if common::absent() {
        return;
    }
    let cases = tokenizer_fixture().get("escape_chars").items();
    assert_eq!(cases.len(), 33);
    for case in cases {
        let ch = case.get("char").str();
        let got = tokenizer().encode_escaped(ch).unwrap();
        assert_eq!(got, case.get("ids").ids(), "escape character {ch:?}");
        let special = case.get("special_id").int() as u32;
        assert_eq!(
            tokenizer().escape(tokenizer().token_text(special).unwrap()).unwrap(),
            ch
        );
    }
}

#[test]
fn escape_and_unescape_refuse_what_the_reference_refuses() {
    if common::absent() {
        return;
    }
    let cases = tokenizer_fixture().get("escape_refused").items();
    assert_eq!(cases.len(), 6);
    for case in cases {
        let text = case.get("text").str();
        let result = match case.get("op").str() {
            "escape" => tokenizer().escape(text).map(|_| ()),
            "unescape" => tokenizer().unescape(text).map(|_| ()),
            "encode_escaped" => tokenizer().encode_escaped(text).map(|_| ()),
            op => panic!("operation {op}"),
        };
        match result {
            Err(e) => assert_reserved(case.get("op").str(), &e, case),
            other => panic!("{case:?}: expected a Reserved error, got {other:?}"),
        }
    }
}

#[test]
fn encode_never_reads_an_escape_character_as_text() {
    if common::absent() {
        return;
    }
    // The reference refuses an escape character in raw text. `encode_escaped` is the only call that reads it.
    assert!(matches!(tokenizer().encode("\u{fdd0}"), Err(Error::Reserved { .. })));
    assert!(tokenizer().encode_escaped("\u{fdd0}").is_ok());
}

#[test]
fn decode_gives_the_text_of_the_reference() {
    if common::absent() {
        return;
    }
    let cases = tokenizer_fixture().get("decode").items();
    assert!(cases.len() > 100);
    for case in cases {
        let name = case.get("name").str();
        let got = tokenizer()
            .decode(&case.get("ids").ids())
            .unwrap_or_else(|e| panic!("case {name}: {e}"));
        assert_eq!(got, case.get("text").str(), "case {name}");
    }
}

#[test]
fn decode_refuses_what_the_reference_refuses() {
    if common::absent() {
        return;
    }
    let cases = tokenizer_fixture().get("decode_errors").items();
    assert_eq!(cases.len(), 6);
    for case in cases {
        let name = case.get("name").str();
        let error = match tokenizer().decode(&case.get("ids").ids()) {
            Err(e) => e,
            Ok(text) => panic!(
                "case {name} gave {text:?}, and the reference says {:?}",
                case.get("error").str()
            ),
        };
        let said = case.get("error").str();
        match case.get("exception").str() {
            // An id outside the vocabulary: the error names the id and the size of the vocabulary.
            "IndexError" => match error {
                Error::TokenId { id, count } => {
                    assert_eq!(
                        id as i128,
                        case.get("bad_id").int(),
                        "case {name}: the reference says {said:?}"
                    );
                    assert_eq!(count, tokenizer().len(), "case {name}");
                }
                other => panic!("case {name}: the reference says {said:?}; got {other:?}"),
            },
            // Bytes that are not UTF-8: the message gives the index where the reference's error starts.
            "UnicodeDecodeError" => match &error {
                Error::Bytes(m) => {
                    let from = format!("from index {}", case.get("start").int());
                    assert!(m.contains(&from), "case {name}: the reference says {said:?}; got {m:?}");
                }
                other => panic!("case {name}: the reference says {said:?}; got {other:?}"),
            },
            // A token with a character outside the byte table: the message names the token.
            "RuntimeError" => match &error {
                Error::Bytes(m) => {
                    let token = format!("token {} (", case.get("token").int());
                    assert!(
                        m.contains(&token),
                        "case {name}: the reference says {said:?}; got {m:?}"
                    );
                }
                other => panic!("case {name}: the reference says {said:?}; got {other:?}"),
            },
            other => panic!("case {name}: the exception {other} is not known"),
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn token_piece_gives_the_bytes_of_the_reference() {
    if common::absent() {
        return;
    }
    let cases = tokenizer_fixture().get("token_pieces").items();
    assert!(cases.len() > 300);
    for case in cases {
        let id = case.get("id").int() as u32;
        assert_eq!(
            tokenizer().token_type(id).unwrap() as i128,
            case.get("type").int(),
            "type of {id}"
        );
        assert_eq!(
            hex(&tokenizer().token_piece(id, false).unwrap()),
            case.get("plain").str(),
            "plain piece {id}"
        );
        assert_eq!(
            hex(&tokenizer().token_piece(id, true).unwrap()),
            case.get("special").str(),
            "special piece {id}"
        );
    }
}

#[test]
fn token_piece_refuses_an_id_outside_the_vocabulary() {
    if common::absent() {
        return;
    }
    let n = tokenizer().len() as u32;
    assert!(matches!(tokenizer().token_piece(n, false), Err(Error::TokenId { .. })));
    assert!(matches!(
        tokenizer().token_piece(u32::MAX, true),
        Err(Error::TokenId { .. })
    ));
}

#[test]
fn every_piece_of_the_vocabulary_matches_the_digest_of_the_reference() {
    if common::absent() {
        return;
    }
    let digest = tokenizer_fixture().get("piece_digest");
    for (special, key) in [(false, "plain"), (true, "special")] {
        let mut h = Digest::new();
        for id in 0..tokenizer().len() as u32 {
            let piece = tokenizer().token_piece(id, special).unwrap();
            h.update(&id.to_le_bytes());
            h.update(&(piece.len() as u32).to_le_bytes());
            h.update(&piece);
        }
        assert_eq!(h.finish_hex(), digest.get(key).str(), "pieces with special={special}");
    }
}

#[test]
fn encoding_every_vocabulary_text_matches_the_digest_of_the_reference() {
    if common::absent() {
        return;
    }
    let sweep = tokenizer_fixture().get("vocab_sweep");
    let tok = tokenizer();
    let mut h = Digest::new();
    let mut encoded = 0;
    for id in 0..tok.len() as u32 {
        if tok.token_type(id).unwrap() != 1 {
            continue;
        }
        let Ok(text) = String::from_utf8(tok.token_piece(id, false).unwrap()) else {
            continue;
        };
        if text
            .chars()
            .any(|c| matches!(c as u32, 0xFDD0..=0xFDEF) || c as u32 & 0xFFFE == 0xFFFE)
        {
            continue;
        }
        let ids = tok.encode(&text).unwrap();
        h.update(&id.to_le_bytes());
        h.update(&(ids.len() as u32).to_le_bytes());
        for i in ids {
            h.update(&i.to_le_bytes());
        }
        encoded += 1;
    }
    assert_eq!(encoded as i128, sweep.get("encoded").int());
    assert_eq!(h.finish_hex(), sweep.get("sha256").str());
}

/// SHA-256, in the form that the digests of the fixtures use.
struct Digest(sha2::Sha256);

impl Digest {
    fn new() -> Digest {
        Digest(sha2::Sha256::default())
    }

    fn update(&mut self, data: &[u8]) {
        sha2::Digest::update(&mut self.0, data);
    }

    fn finish_hex(self) -> String {
        sha2::Digest::finalize(self.0)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

#[test]
fn control_hits_list_the_special_token_text_in_order() {
    if common::absent() {
        return;
    }
    let t = tokenizer();
    assert_eq!(
        t.control_hits("a <|im_start|>user<|im_end|> b <think> c"),
        ["<|im_start|>", "<|im_end|>", "<think>"]
    );
    assert!(t.control_hits("plain text, <not a token> and |im_start|").is_empty());
    // A token that is not whole is not a hit.
    assert!(t.control_hits("<|im_start").is_empty());
    // A hit at the start of the text, and two hits side by side.
    assert_eq!(t.control_hits("<|im_end|>x"), ["<|im_end|>"]);
    assert_eq!(t.control_hits("<|im_start|><|im_end|>"), ["<|im_start|>", "<|im_end|>"]);
    // Each hit is one token when the text is encoded with special tokens on.
    for hit in t.control_hits("<|im_start|><|im_end|>") {
        assert_eq!(t.encode(&hit).unwrap().len(), 1, "{hit}");
    }
}
