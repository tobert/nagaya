//! The ids of `config.json` that the prompt and the decode need. These tests need no checkpoint.

mod common;

use common::SMALL_CONFIG;
use nagaya_family_qwen3tts::{Error, SpeechIds};

fn changed(edit: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(SMALL_CONFIG).expect("the small config");
    edit(&mut v);
    v.to_string()
}

#[test]
fn the_small_config_gives_the_ids() {
    let ids = SpeechIds::from_config(SMALL_CONFIG).expect("the small config");
    assert_eq!(ids.text.pad, 151671);
    assert_eq!(ids.text.bos, 151672);
    assert_eq!(ids.text.eos, 151673);
    assert_eq!(ids.text.im_start, 151644);
    assert_eq!(ids.text.im_end, 151645);
    assert_eq!(ids.text.assistant, 77091);
    assert_eq!(ids.codec.pad, 2148);
    assert_eq!(ids.codec.bos, 2149);
    assert_eq!(ids.codec.eos, 2150);
    assert_eq!(ids.codec.think, 2154);
    assert_eq!(ids.codec.nothink, 2155);
    assert_eq!(ids.codec.think_bos, 2156);
    assert_eq!(ids.codec.think_eos, 2157);
    assert_eq!(ids.codec.vocab_size, 3072);
}

/// The reference lists `auto` and each language whose name does not hold `dialect` (`get_supported_languages`).
#[test]
fn the_languages_are_auto_and_the_names_without_dialect() {
    let ids = SpeechIds::from_config(SMALL_CONFIG).expect("the small config");
    assert_eq!(ids.languages(), vec!["auto", "chinese", "english"]);
    assert_eq!(ids.language_id("auto").expect("auto"), None);
    assert_eq!(ids.language_id("english").expect("english"), Some(2050));
    // The reference compares the lower case of the name.
    assert_eq!(ids.language_id("Chinese").expect("Chinese"), Some(2055));
    assert_eq!(ids.language_id("AUTO").expect("AUTO"), None);
}

#[test]
fn an_unknown_language_is_an_error_that_names_the_languages() {
    let ids = SpeechIds::from_config(SMALL_CONFIG).expect("the small config");
    for given in ["klingon", "sichuan_dialect", ""] {
        match ids.language_id(given) {
            Err(e @ Error::Language { .. }) => {
                let text = e.to_string();
                assert!(text.contains("auto, chinese, english"), "{text}");
                assert!(text.contains(&format!("{given:?}")), "{text}");
            }
            other => panic!("{given:?}: {other:?}"),
        }
    }
}

#[test]
fn the_decode_constants_follow_the_reference() {
    let ids = SpeechIds::from_config(SMALL_CONFIG).expect("the small config");
    assert_eq!(ids.codec_end(), 2150);
    assert_eq!(ids.trailing_text(), 151671);
    let suppressed = ids.suppressed_codecs();
    assert_eq!(suppressed.len(), 1023);
    assert_eq!(suppressed.first(), Some(&2048));
    assert_eq!(suppressed.last(), Some(&3071));
    assert!(!suppressed.contains(&2150));
    assert!(suppressed.contains(&2149));
}

#[test]
fn a_missing_key_is_an_error_that_names_it() {
    let text = changed(|v| {
        v["talker_config"]
            .as_object_mut()
            .expect("talker")
            .remove("codec_think_bos_id");
    });
    match SpeechIds::from_config(&text) {
        Err(e @ Error::Checkpoint(_)) => {
            assert!(e.to_string().contains("talker_config.codec_think_bos_id"), "{e}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn another_model_type_is_refused() {
    let text = changed(|v| v["tts_model_type"] = "custom_voice".into());
    match SpeechIds::from_config(&text) {
        Err(e @ Error::Checkpoint(_)) => assert!(e.to_string().contains("custom_voice"), "{e}"),
        other => panic!("{other:?}"),
    }
    let text = changed(|v| v["model_type"] = "qwen3".into());
    assert!(matches!(SpeechIds::from_config(&text), Err(Error::Checkpoint(_))));
}

#[test]
fn an_id_that_is_not_a_32_bit_integer_is_refused() {
    for bad in [
        serde_json::json!(-1),
        serde_json::json!(2.5),
        serde_json::json!("2148"),
        serde_json::json!(1u64 << 33),
    ] {
        let text = changed(|v| v["talker_config"]["codec_pad_id"] = bad.clone());
        match SpeechIds::from_config(&text) {
            Err(e @ Error::Checkpoint(_)) => assert!(e.to_string().contains("codec_pad_id"), "{bad}: {e}"),
            other => panic!("{bad}: {other:?}"),
        }
    }
}

/// The suppress range is the last 1,024 codec ids, so a smaller codec vocabulary has no such range.
#[test]
fn a_codec_vocabulary_below_1024_is_refused() {
    let text = changed(|v| v["talker_config"]["vocab_size"] = 1000.into());
    assert!(matches!(SpeechIds::from_config(&text), Err(Error::Checkpoint(_))));
}
