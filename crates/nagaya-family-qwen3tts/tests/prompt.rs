//! The prompt of a speech request against the reference `generate_voice_design` (`tests/fixtures/prompts.json`).
//!
//! The fixture tool checked in Python that each pair gives the reference row of `talker_input_embeds`. These tests
//! check that the library gives the same pairs.

mod common;

use common::{SMALL_CONFIG, absent, family, fixture, id, ids};
use nagaya_family_qwen3tts::{Error, PromptPosition, SpeechIds};

fn pos(text: Option<u32>, codec: Option<u32>) -> PromptPosition {
    PromptPosition { text, codec }
}

#[test]
fn prompts_match_the_reference() {
    if absent() {
        return;
    }
    let fixture = fixture("prompts.json");
    let requests = fixture["requests"].as_array().expect("requests");
    assert!(requests.len() >= 3, "the fixture has too few requests");
    for r in requests {
        let name = r["name"].as_str().expect("name");
        let want: Vec<PromptPosition> = r["positions"]
            .as_array()
            .expect("positions")
            .iter()
            .map(|p| pos(id(&p["text"]), id(&p["codec"])))
            .collect();
        let got = family()
            .prompt(
                r["text"].as_str().expect("text"),
                r["instructions"].as_str().expect("instructions"),
                r["language"].as_str().expect("language"),
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got, want, "{name}");
    }
}

#[test]
fn the_decode_constants_match_the_reference() {
    if absent() {
        return;
    }
    let fixture = fixture("prompts.json");
    let c = &fixture["constants"];
    let speech = family().ids();
    assert_eq!(Some(speech.codec_end()), id(&c["codec_eos"]));
    assert_eq!(Some(speech.trailing_text()), id(&c["trailing_text"]));
    assert_eq!(speech.suppressed_codecs(), ids(&c["suppress"]));
    let languages: Vec<&str> = c["languages"]
        .as_array()
        .expect("languages")
        .iter()
        .map(|l| l.as_str().expect("a name"))
        .collect();
    assert_eq!(speech.languages(), languages);
}

#[test]
fn a_prompt_in_an_unknown_language_is_refused() {
    if absent() {
        return;
    }
    match family().prompt("Hello.", "A voice.", "klingon") {
        Err(e @ Error::Language { .. }) => assert!(e.to_string().contains("english"), "{e}"),
        other => panic!("{other:?}"),
    }
}

// The tests below need no checkpoint. They use the ids of the small config and made-up token ids.

const IM_START: u32 = 151644;
const IM_END: u32 = 151645;
const ASSISTANT: u32 = 77091;
const NL: u32 = 198;
const PAD: u32 = 151671;
const BOS: u32 = 151672;
const EOS: u32 = 151673;

fn speech() -> SpeechIds {
    SpeechIds::from_config(SMALL_CONFIG).expect("the small config")
}

/// The ids of `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n` for text tokens `text`.
fn assistant(text: &[u32]) -> Vec<u32> {
    let mut v = vec![IM_START, ASSISTANT, NL];
    v.extend_from_slice(text);
    v.extend_from_slice(&[IM_END, NL, IM_START, ASSISTANT, NL]);
    v
}

#[test]
fn the_layout_of_a_prompt_with_auto() {
    let instructions = [IM_START, 872, NL, 7, 8, IM_END, NL];
    let got = speech()
        .positions(Some(&instructions), &assistant(&[10, 11]), "auto")
        .expect("a prompt");
    let mut want: Vec<_> = instructions.iter().map(|&t| pos(Some(t), None)).collect();
    want.extend([
        pos(Some(IM_START), None),
        pos(Some(ASSISTANT), None),
        pos(Some(NL), None),
        pos(Some(PAD), Some(2155)),
        pos(Some(PAD), Some(2156)),
        pos(Some(PAD), Some(2157)),
        pos(Some(BOS), Some(2148)),
        pos(Some(10), Some(2148)),
        pos(Some(11), Some(2148)),
        pos(Some(EOS), Some(2148)),
        pos(Some(PAD), Some(2149)),
    ]);
    assert_eq!(got, want);
}

#[test]
fn the_layout_of_a_prompt_with_a_language() {
    let got = speech()
        .positions(None, &assistant(&[10]), "English")
        .expect("a prompt");
    let want = vec![
        pos(Some(IM_START), None),
        pos(Some(ASSISTANT), None),
        pos(Some(NL), None),
        pos(Some(PAD), Some(2154)),
        pos(Some(PAD), Some(2156)),
        pos(Some(PAD), Some(2050)),
        pos(Some(PAD), Some(2157)),
        pos(Some(BOS), Some(2148)),
        pos(Some(10), Some(2148)),
        pos(Some(EOS), Some(2148)),
        pos(Some(PAD), Some(2149)),
    ];
    assert_eq!(got, want);
}

/// The reference takes no special case for empty text: the prompt has no text positions.
#[test]
fn the_layout_of_a_prompt_with_no_text() {
    let got = speech().positions(None, &assistant(&[]), "auto").expect("a prompt");
    assert_eq!(got.len(), 3 + 4 + 2);
    assert_eq!(got[7], pos(Some(EOS), Some(2148)));
}

#[test]
fn assistant_ids_that_are_not_the_template_are_refused() {
    let s = speech();
    let good = assistant(&[10]);
    let mut cases = vec![good[..7].to_vec()];
    for k in [0, 1, good.len() - 5, good.len() - 3, good.len() - 2] {
        let mut bad = good.clone();
        bad[k] = 5;
        cases.push(bad);
    }
    for bad in cases {
        match s.positions(None, &bad, "auto") {
            Err(e @ Error::Prompt(_)) => assert!(e.to_string().contains("template"), "{e}"),
            other => panic!("{bad:?}: {other:?}"),
        }
    }
}

#[test]
fn an_unknown_language_is_refused_before_the_layout() {
    assert!(matches!(
        speech().positions(None, &assistant(&[10]), "klingon"),
        Err(Error::Language { .. })
    ));
}
