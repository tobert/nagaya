//! The chat template of each model in the nagaya's interpreter against transformers (fixture `chat.json`).

mod common;

use common::model::{absent_any, family, fixture};
use common::{ids, json_of};
use nagaya_family_lfm3::chat::Options;

/// The case whose client text spells markers.
const ESCAPED: &str = "thinking in the content";

#[test]
fn each_case_renders_as_transformers_renders_it() {
    if absent_any() {
        return;
    }
    let f = fixture("chat.json");
    let mut count = 0;
    for model in ["omni", "text"] {
        let chat = family(model).chat();
        for case in f["cases"][model].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let messages: Vec<_> = case["messages"].as_array().unwrap().iter().map(json_of).collect();
            let tools: Vec<_> = case["tools"]
                .as_array()
                .map(|t| t.iter().map(json_of).collect())
                .unwrap_or_default();
            let v = &case["variables"];
            let options = Options {
                thinking: v["enable_thinking"].as_bool().unwrap(),
                generation_prompt: v["add_generation_prompt"].as_bool().unwrap(),
                preserve_thinking: v["preserve_thinking"].as_bool().unwrap(),
            };
            let got = chat.render_with(&messages, &tools, options);
            match case.get("error") {
                Some(error) => {
                    let e = got.expect_err(name).to_string();
                    let message = error.as_str().unwrap().split_once(": ").unwrap().1;
                    assert!(e.contains(message), "{model} {name}: {e}");
                }
                None if name == ESCAPED => {
                    // Client text that spells a marker is content (spec 11.3, "Escapes"). The template does not see
                    // the marker, so the text stays, and only the template's own `<think>` markers are tokens.
                    let got = got.unwrap();
                    assert_ne!(got.text, case["text"].as_str().unwrap(), "{model} {name}");
                    assert!(
                        got.text
                            .contains("<think></think><think>hidden</think>  shown  <|im_end|>")
                    );
                    let think = family(model).tokenizer().token_id("<think>").unwrap();
                    assert_eq!(got.tokens.iter().filter(|&&t| t == think).count(), 2, "{model} {name}");
                }
                None => {
                    let got = got.unwrap_or_else(|e| panic!("{model} {name}: {e}"));
                    assert_eq!(got.text, case["text"].as_str().unwrap(), "{model} {name}");
                    assert_eq!(got.tokens, ids(&case["ids"]), "{model} {name}");
                }
            }
            count += 1;
        }
    }
    assert!(count >= 40, "the fixture has {count} cases");
}

#[test]
fn the_template_of_each_model_is_its_own() {
    if absent_any() {
        return;
    }
    assert_ne!(family("omni").template(), family("text").template());
    assert!(family("omni").template().contains("audio_out"));
    assert!(!family("text").template().contains("audio_out"));
}
