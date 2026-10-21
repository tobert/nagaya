//! The tokenizer gives the ids of Hugging Face's Python `tokenizers` for the checkpoint's file (spec 11.3, 12.3).

mod common;

use common::model::{absent, family, model_dir};
use common::{JsonExt, fixture};
use nagaya_text_gemma4::Tokenizer;
use nagaya_text_gemma4::tokenizer::sha256_of;

#[test]
fn ids_match_python_with_markers_and_as_text() {
    if absent() {
        return;
    }
    let tok = family().tokenizer();
    let cases = fixture("ids.json");
    for case in cases.get("cases").items() {
        let text = case.get("text").str();
        assert_eq!(tok.encode(text).unwrap(), case.get("ids").ids(), "ids of {text:?}");
        assert_eq!(
            tok.encode_content(text).unwrap(),
            case.get("content_ids").ids(),
            "content ids of {text:?}"
        );
        assert_eq!(
            tok.decode(&case.get("ids").ids()).unwrap(),
            case.get("decoded").str(),
            "{text:?}"
        );
    }
}

#[test]
fn the_hash_is_of_the_whole_file_of_the_checkpoint() {
    if absent() {
        return;
    }
    let path = model_dir().join("tokenizer.json");
    let hash = family().tokenizer_hash();
    assert_eq!(hash, sha256_of(&std::fs::read(&path).unwrap()));
    let want = fixture("ids.json")
        .get("header")
        .get("tokenizer_sha256")
        .str()
        .to_string();
    assert_eq!(
        hash,
        format!("sha256:{want}"),
        "the fixture was made from another tokenizer file"
    );
    assert_eq!(hash.len(), "sha256:".len() + 64);
}

#[test]
fn the_vocabulary_has_the_markers_of_the_template() {
    if absent() {
        return;
    }
    let tok = family().tokenizer();
    assert_eq!(tok.len(), 262_144);
    for (marker, id) in [
        ("<bos>", 2),
        ("<eos>", 1),
        ("<|turn>", 105),
        ("<turn|>", 106),
        ("<|channel>", 100),
    ] {
        assert_eq!(tok.token_id(marker), Some(id), "{marker}");
    }
    assert_eq!(
        tok.specials()[0].len(),
        tok.specials().iter().map(String::len).max().unwrap()
    );
}

#[test]
fn an_id_outside_the_vocabulary_is_an_error() {
    if absent() {
        return;
    }
    let e = family().tokenizer().decode(&[1, 262_144]).unwrap_err();
    assert!(e.to_string().contains("262144"), "{e}");
}

#[test]
fn a_file_that_is_not_a_tokenizer_is_an_error_that_names_it() {
    let dir = std::env::temp_dir().join(format!("nagaya-gemma4-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tokenizer.json");
    std::fs::write(&path, b"{}").unwrap();
    let e = Tokenizer::from_file(&path).unwrap_err().to_string();
    assert!(e.contains("tokenizer.json") && e.contains("Gemma 4"), "{e}");
}
