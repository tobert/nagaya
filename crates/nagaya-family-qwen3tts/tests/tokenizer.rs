//! The tokenizer against the reference processor (`tests/fixtures/ids.json`, from `tools/qwen3tts_family_fixture.py`).

mod common;

use common::{absent, family, fixture, ids};

#[test]
fn ids_match_the_reference_processor() {
    if absent() {
        return;
    }
    let fixture = fixture("ids.json");
    let strings = fixture["strings"].as_array().expect("strings");
    assert!(strings.len() >= 20, "the fixture has too few strings");
    let mut wrong = Vec::new();
    for entry in strings {
        let text = entry["text"].as_str().expect("text");
        let want = ids(&entry["ids"]);
        let got = family().tokenize(text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
        if got != want {
            wrong.push(format!("{text:?}: got {got:?}, the reference gives {want:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} strings differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The reference replaces the Qwen2 split pattern (transformers 4.57.3, `fix_mistral_regex=True`). The fixture marks
/// the strings where that changes the ids. At least one such string keeps the test of the pattern honest.
#[test]
fn the_fixture_holds_a_string_that_the_qwen2_pattern_splits_otherwise() {
    let fixture = fixture("ids.json");
    let differs = fixture["strings"]
        .as_array()
        .expect("strings")
        .iter()
        .filter(|e| e["qwen2_pattern_differs"].as_bool().expect("a flag"))
        .count();
    assert!(
        differs >= 1,
        "no string of the fixture tells the two split patterns apart"
    );
}

#[test]
fn special_tokens_are_whole_tokens() {
    if absent() {
        return;
    }
    let f = family();
    let cases = [
        ("<|im_start|>", 151644),
        ("<|im_end|>", 151645),
        ("<tts_pad>", 151671),
        ("<tts_text_bos>", 151672),
        ("<tts_text_eod>", 151673),
        ("<think>", 151667),
    ];
    for (text, id) in cases {
        assert_eq!(f.tokenize(text).expect(text), vec![id], "{text}");
    }
}

/// A copy of the checkpoint's small files in a temporary directory, with one file changed.
fn changed_copy(name: &str, file: &str, edit: impl FnOnce(&mut serde_json::Value)) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nagaya-qwen3tts-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temporary directory");
    for f in ["config.json", "vocab.json", "merges.txt", "tokenizer_config.json"] {
        std::fs::copy(common::model_dir().join(f), dir.join(f)).expect("a copy");
    }
    let text = std::fs::read_to_string(dir.join(file)).expect("the copy");
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    edit(&mut v);
    std::fs::write(dir.join(file), v.to_string()).expect("the changed file");
    dir
}

#[test]
fn a_config_id_that_is_not_its_token_is_refused() {
    if absent() {
        return;
    }
    let dir = changed_copy("config", "config.json", |v| v["tts_pad_token_id"] = 151670.into());
    match nagaya_family_qwen3tts::Qwen3TtsFamily::from_dir(&dir) {
        Err(e) => assert!(e.to_string().contains("tts_pad_token_id"), "{e}"),
        Ok(_) => panic!("a wrong tts_pad_token_id built a family"),
    }
}

#[test]
fn an_added_token_with_another_id_is_refused() {
    if absent() {
        return;
    }
    let dir = changed_copy("added", "tokenizer_config.json", |v| {
        let decoder = v["added_tokens_decoder"].as_object_mut().expect("added tokens");
        let entry = decoder.remove("151675").expect("the last added token");
        decoder.insert("151680".into(), entry);
    });
    match nagaya_family_qwen3tts::Qwen3TtsFamily::from_dir(&dir) {
        Err(e) => assert!(e.to_string().contains("151680"), "{e}"),
        Ok(_) => panic!("an added token with a wrong id built a family"),
    }
}

#[test]
fn the_text_is_composed_before_the_split() {
    if absent() {
        return;
    }
    let f = family();
    assert_eq!(
        f.tokenize("cafe\u{301}").expect("decomposed"),
        f.tokenize("caf\u{e9}").expect("composed")
    );
}
