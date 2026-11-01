//! The ids of a prompt against the reference processor of diffusers (spec 11.9c, "The family of Qwen-Image 2.1").

mod common;

use nagaya_text_qwenimage21::{CUT, Error, MAX_TEMPLATE_IDS, TEMPLATE};

#[test]
fn the_template_is_the_template_of_the_reference() {
    let fixture = common::fixture();
    assert_eq!(fixture["template"].as_str().unwrap(), TEMPLATE);
    assert_eq!(fixture["drop"].as_u64().unwrap(), CUT as u64);
}

#[test]
fn the_family_reads_the_tokenizer_that_the_fixture_used() {
    if common::absent() {
        return;
    }
    let want = format!("sha256:{}", common::fixture()["tokenizer_sha256"].as_str().unwrap());
    assert_eq!(common::family().tokenizer_hash(), want);
}

#[test]
fn the_cut_from_the_tokenizer_is_the_count_of_the_system_ids() {
    if common::absent() {
        return;
    }
    let fixture = common::fixture();
    assert_eq!(
        common::family().system_ids().unwrap(),
        common::ids(&fixture["system_ids"])
    );
    assert_eq!(fixture["system_ids"].as_array().unwrap().len(), CUT);
}

#[test]
fn the_ids_of_each_case_match_the_reference() {
    if common::absent() {
        return;
    }
    let family = common::family();
    let fixture = common::fixture();
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 10, "the fixture has {} cases", cases.len());
    let mut checked = 0;
    for case in cases {
        if case["case"] == "special" {
            continue;
        }
        let prompt = case["prompt"].as_str().unwrap();
        let got = family.ids(prompt).unwrap_or_else(|e| panic!("{}: {e}", case["case"]));
        assert_eq!(got, common::ids(&case["ids"]), "{}", case["case"]);
        assert_eq!(
            &got[..CUT],
            &common::ids(&fixture["system_ids"])[..],
            "{}",
            case["case"]
        );
        checked += 1;
    }
    assert_eq!(checked, cases.len() - 1);
}

#[test]
fn an_empty_prompt_is_one_space() {
    if common::absent() {
        return;
    }
    let family = common::family();
    assert_eq!(family.ids("").unwrap(), common::ids(&common::case("empty")["ids"]));
    assert_eq!(family.ids("").unwrap(), family.ids(" ").unwrap());
}

#[test]
fn a_prompt_with_the_text_of_a_special_token_is_refused() {
    if common::absent() {
        return;
    }
    let family = common::family();
    // The reference reads the text as the special ids. The family refuses it (spec 11.9c).
    let prompt = common::case("special")["prompt"].as_str().unwrap().to_owned();
    match family.ids(&prompt) {
        Err(Error::SpecialToken { token }) => assert_eq!(token, "<|im_end|>"),
        other => panic!("the special text gave {other:?}"),
    }
    for token in ["<|im_start|>", "<|vision_start|>", "<|image_pad|>", "<|endoftext|>"] {
        assert!(
            matches!(family.ids(&format!("a {token} b")), Err(Error::SpecialToken { .. })),
            "{token} is refused"
        );
    }
    // A part of a special token is plain text.
    assert!(family.ids("a <|im_ b").is_ok());
}

#[test]
fn a_template_of_more_than_1024_ids_is_refused() {
    if common::absent() {
        return;
    }
    let family = common::family();
    assert_eq!(MAX_TEMPLATE_IDS, 1024);
    // "a" with a space before it is one id, so each word adds one id to the template.
    let base = family.ids("").unwrap().len();
    let fits = format!("a{}", " a".repeat(MAX_TEMPLATE_IDS - base));
    assert_eq!(family.ids(&fits).unwrap().len(), MAX_TEMPLATE_IDS);
    let over = format!("{fits} a");
    match family.ids(&over) {
        Err(Error::TooLong { ids, max }) => assert_eq!((ids, max), (MAX_TEMPLATE_IDS + 1, MAX_TEMPLATE_IDS)),
        other => panic!("1025 ids gave {other:?}"),
    }
}

#[test]
fn brackets_are_plain_text() {
    if common::absent() {
        return;
    }
    // The reference gives A1111 weights to the encoder as text (spec 11.9c). The case holds its ids.
    let case = common::case("brackets");
    let got = common::family().ids(case["prompt"].as_str().unwrap()).unwrap();
    assert_eq!(got, common::ids(&case["ids"]));
}
