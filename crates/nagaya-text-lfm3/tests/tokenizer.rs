//! The tokenizer of both models against the ids of Hugging Face `tokenizers` (fixture `ids.json`).

mod common;

use common::ids;
use common::model::{absent_any, family, fixture};
use nagaya_council::family::Family;

const MODELS: [&str; 2] = ["omni", "text"];

#[test]
fn the_ids_are_the_ids_of_the_reference() {
    if absent_any() {
        return;
    }
    let f = fixture("ids.json");
    for model in MODELS {
        let tok = family(model).tokenizer();
        let cases = &f["cases"][model];
        assert_eq!(tok.len() as u64, cases["count"].as_u64().unwrap(), "{model}");
        for case in cases["strings"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            assert_eq!(tok.encode(text).unwrap(), ids(&case["ids"]), "{model} {text:?}");
            assert_eq!(
                tok.encode_content(text).unwrap(),
                ids(&case["content_ids"]),
                "{model} content {text:?}"
            );
            assert_eq!(
                tok.decode_whole(&ids(&case["ids"])).unwrap(),
                case["decoded"].as_str().unwrap(),
                "{model} decode {text:?}"
            );
        }
    }
}

#[test]
fn the_hash_is_of_the_whole_file() {
    if absent_any() {
        return;
    }
    let f = fixture("ids.json");
    for model in MODELS {
        let want = f["header"]["models"][model]["files"]["tokenizer.json"]
            .as_str()
            .unwrap();
        assert_eq!(family(model).tokenizer_hash(), format!("sha256:{want}"), "{model}");
    }
    // The two files hold the same tokens, and the hashes differ (AGENTS.md milestone 24).
    assert_ne!(family("omni").tokenizer_hash(), family("text").tokenizer_hash());
}

#[test]
fn client_text_that_spells_a_marker_stays_text() {
    if absent_any() {
        return;
    }
    for model in MODELS {
        let fam = family(model);
        let tok = fam.tokenizer();
        for marker in [
            "<think>",
            "</think>",
            "<tool_call>",
            "<|im_end|>",
            "<|image|>",
            "<|audio_start|>",
        ] {
            let id = tok.token_id(marker).unwrap();
            let text = format!("a {marker} b");
            let content = fam.tokenize(&text).unwrap();
            assert!(!content.contains(&id), "{model}: {marker} became its token");
            assert_eq!(fam.detokenize(&content).unwrap(), text, "{model}");
            assert!(fam.tokenize_special(&text).unwrap().contains(&id), "{model}: {marker}");
        }
    }
}

#[test]
fn each_of_the_66_noncharacters_is_refused_with_its_offset() {
    if absent_omni_only() {
        return;
    }
    let fam = family("omni");
    let mut seen = 0;
    for cp in (0xFDD0..=0xFDEF).chain((0..17).flat_map(|p| [p * 0x10000 + 0xFFFE, p * 0x10000 + 0xFFFF])) {
        let c = char::from_u32(cp).unwrap();
        let e = fam.tokenize(&format!("ab{c}")).expect_err("a noncharacter is refused");
        assert!(e.is_client(), "{e}");
        let message = e.to_string();
        assert!(
            message.contains(&format!("U+{cp:04X}")) && message.contains("offset 2"),
            "{message}"
        );
        seen += 1;
    }
    assert_eq!(seen, 66);
    assert!(fam.tokenize("ab\u{FDCF}\u{FFFD}").is_ok());
}

fn absent_omni_only() -> bool {
    common::model::absent_omni()
}

#[test]
fn detokenize_refuses_a_cut_character_and_pieces_keep_it() {
    if absent_any() {
        return;
    }
    let fam = family("text");
    let crab = fam.tokenize("🦀").unwrap();
    assert!(crab.len() > 1, "the crab is more than one token: {crab:?}");
    let e = fam.detokenize(&crab[..1]).expect_err("a cut character is refused");
    assert!(e.to_string().contains("inside a character"), "{e}");
    assert!(fam.pieces(&crab[..1]).unwrap()[0].contains('\u{FFFD}'));
    let e = fam
        .detokenize(&[fam.vocab_size() as u32])
        .expect_err("an id outside the vocabulary");
    assert!(e.is_client());
}

#[test]
fn a_decode_and_a_new_encode_can_give_other_ids() {
    if absent_any() {
        return;
    }
    // BPE is not one to one: the ids of a reply do not always come back from its text. The service keeps the ids
    // of a reply in its held head, so a replay feeds them (spec 11.3c). This test pins that the family cannot do it.
    let fam = family("text");
    let hello = fam.tokenize("hello").unwrap();
    let split: Vec<u32> = ["he", "llo"].iter().flat_map(|t| fam.tokenize(t).unwrap()).collect();
    assert_eq!(fam.detokenize(&split).unwrap(), "hello");
    assert_ne!(fam.tokenize(&fam.detokenize(&split).unwrap()).unwrap(), split);
    assert_eq!(fam.tokenize("hello").unwrap(), hello);
}
