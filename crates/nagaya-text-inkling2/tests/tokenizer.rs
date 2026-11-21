mod common;

use common::model::{absent, absent_varloop, mini_dir, varloop_dir};
use nagaya_text_inkling2::Tokenizer;
use nagaya_text_inkling2::tokenizer::check_text;
use serde_json::Value;

#[test]
fn tokenizer_matches_ids_fixture_and_escapes_markers() {
    if absent() {
        return;
    }
    let dir = mini_dir();
    let tok_bytes = std::fs::read(dir.join("tokenizer.json")).unwrap();
    let tok = Tokenizer::from_bytes(&tok_bytes).unwrap();

    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(common::fixture("ids.json")).unwrap()).unwrap();
    let expected_hash = fixture["header"]["models"]["mini"]["files"]["tokenizer.json"]
        .as_str()
        .unwrap();
    assert_eq!(tok.hash(), format!("sha256:{expected_hash}"));

    let count = fixture["cases"]["count"].as_u64().unwrap() as usize;
    assert_eq!(tok.len(), count);
    assert_eq!(count, 200_058);

    for row in fixture["cases"]["strings"].as_array().unwrap() {
        let text = row["text"].as_str().unwrap();
        let want_ids: Vec<u32> = row["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let want_content: Vec<u32> = row["content_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let want_decoded = row["decoded"].as_str().unwrap();

        assert_eq!(tok.encode(text).unwrap(), want_ids, "encode {text:?}");
        assert_eq!(tok.encode_content(text).unwrap(), want_content, "content {text:?}");
        assert_eq!(tok.decode_whole(&want_ids).unwrap(), want_decoded, "decode {text:?}");

        let escaped = tok.escape(text).unwrap();
        let escaped_ids = tok.encode_escaped(&escaped).unwrap();
        assert!(
            escaped_ids.iter().all(|&id| id < 199_998),
            "escaped {text:?} produced added token in {escaped_ids:?}"
        );
        assert_eq!(tok.decode_whole(&escaped_ids).unwrap(), text, "decode escaped {text:?}");
        assert_eq!(tok.unescape(&escaped).unwrap(), text, "unescape {text:?}");
    }
}

#[test]
fn varloop_tokenizer_and_template_hashes_match_mini() {
    if absent() || absent_varloop() {
        return;
    }
    let mini = mini_dir();
    let varloop = varloop_dir();
    for file in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    ] {
        let a = nagaya_text_inkling2::tokenizer::sha256_of(&std::fs::read(mini.join(file)).unwrap());
        let b = nagaya_text_inkling2::tokenizer::sha256_of(&std::fs::read(varloop.join(file)).unwrap());
        assert_eq!(a, b, "hash mismatch for {file}");
    }
}

#[test]
fn refuses_all_66_unicode_noncharacters_with_char_and_offset() {
    let mut nonchars = Vec::new();
    for cp in 0xFDD0u32..=0xFDEF {
        nonchars.push(char::from_u32(cp).unwrap());
    }
    for plane in 0u32..=16 {
        let base = plane << 16;
        nonchars.push(char::from_u32(base | 0xFFFE).unwrap());
        nonchars.push(char::from_u32(base | 0xFFFF).unwrap());
    }
    assert_eq!(nonchars.len(), 66);

    for ch in nonchars {
        let text = format!("ab{ch}cd");
        let err = check_text(&text).expect_err("noncharacter must be refused").to_string();
        let hex = format!("U+{:04X}", ch as u32);
        assert!(err.contains(&hex), "error {err:?} missing {hex}");
        assert!(err.contains("offset 2"), "error {err:?} missing offset 2");
    }
}
