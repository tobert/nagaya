//! The chat rendering of the library: Google's template with the variables that the library sets, and the escapes
//! (spec 11.3). The fixture holds the text and the ids of transformers for each case.

mod common;

use common::model::{absent, family};
use common::{JsonExt, fixture};
use nagaya_jinja::Json;
use nagaya_text_gemma4::chat::{Chat, THOUGHT_OPEN};
use nagaya_text_gemma4::{Error, Tokenizer};
use std::sync::Arc;

fn chat() -> Chat {
    let tok: Arc<Tokenizer> =
        Arc::new(Tokenizer::from_file(&common::model::model_dir().join("tokenizer.json")).unwrap());
    Chat::new(tok).unwrap()
}

fn spells_a_marker(j: &Json) -> bool {
    match j {
        Json::Str(s) => family().tokenizer().specials().iter().any(|m| s.contains(m.as_str())),
        Json::Array(a) => a.iter().any(spells_a_marker),
        Json::Object(o) => o
            .iter()
            .any(|(k, v)| spells_a_marker(&Json::Str(k.clone())) || spells_a_marker(v)),
        _ => false,
    }
}

fn flag(case: &Json, name: &str) -> bool {
    match case.get("variables").get(name) {
        Json::Bool(b) => *b,
        other => panic!("{name} is not a bool: {other:?}"),
    }
}

/// Each case with the library's variables renders to the text and the ids of transformers. With thinking on, the
/// generation prompt also opens the thought channel, an addition of the first Gemma 4 engine.
#[test]
fn the_library_renders_what_transformers_renders() {
    if absent() {
        return;
    }
    let chat = chat();
    let mut compared = 0;
    for case in fixture("chat.json").get("cases").items() {
        let name = case.get("name").str();
        // The library escapes the marker text of content, and Python does not. Such a case renders otherwise here.
        if !flag(case, "preserve_thinking") || spells_a_marker(case.get("messages")) {
            continue;
        }
        let tools = match case.get("tools") {
            Json::Array(t) => t.clone(),
            _ => Vec::new(),
        };
        let (thinking, prompt) = (flag(case, "enable_thinking"), flag(case, "add_generation_prompt"));
        let got = chat.render(case.get("messages").items(), &tools, thinking, prompt);
        if !case.has("text") {
            assert!(matches!(got, Err(Error::Template(_))), "case {name:?}: {got:?}");
            compared += 1;
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("case {name:?}: {e}"));
        let python = case.get("text").str();
        if thinking && prompt && !python.ends_with(THOUGHT_OPEN) {
            assert_eq!(got.text, format!("{python}{THOUGHT_OPEN}"), "case {name:?}");
            let mut ids = case.get("ids").ids();
            ids.extend(family().tokenizer().encode(THOUGHT_OPEN).unwrap());
            assert_eq!(got.tokens, ids, "case {name:?}");
        } else {
            assert_eq!(got.text, python, "case {name:?}");
            assert_eq!(got.tokens, case.get("ids").ids(), "case {name:?}");
        }
        compared += 1;
    }
    assert_eq!(
        compared, 24,
        "the cases with preserve_thinking on and no marker text in the messages"
    );
}

fn user(content: &str) -> Json {
    Json::Object(vec![
        ("role".to_string(), Json::Str("user".to_string())),
        ("content".to_string(), Json::Str(content.to_string())),
    ])
}

/// The user's `<|turn>` is text: its special token appears only where the template put one.
#[test]
fn content_that_spells_a_marker_stays_text() {
    if absent() {
        return;
    }
    let chat = chat();
    let r = chat
        .render(&[user("say <|turn>done and <|tool_call>")], &[], false, true)
        .unwrap();
    assert!(r.text.contains("say <|turn>done and <|tool_call>"), "{}", r.text);
    let turn = chat.tokenizer().token_id("<|turn>").unwrap();
    let call = chat.tokenizer().token_id("<|tool_call>").unwrap();
    assert_eq!(
        r.tokens.iter().filter(|&&t| t == turn).count(),
        2,
        "the user turn and the model turn"
    );
    assert!(!r.tokens.contains(&call));
    let mut want = chat.tokenizer().encode("<bos><|turn>user\nsay ").unwrap();
    want.extend(chat.tokenizer().encode_content("<|turn>").unwrap());
    assert!(r.tokens.starts_with(&want), "{:?}", r.tokens);
}

#[test]
fn a_noncharacter_in_content_is_refused_with_its_offset() {
    if absent() {
        return;
    }
    for (text, ch, offset) in [
        ("x\u{fdd0}", '\u{fdd0}', 1),
        ("ab\u{10ffff}", '\u{10ffff}', 2),
        ("\u{fffe}", '\u{fffe}', 0),
    ] {
        match chat().render(&[user(text)], &[], false, true) {
            Err(Error::Reserved { ch: c, offset: o }) => assert_eq!((c, o), (ch, offset)),
            other => panic!("{text:?}: {other:?}"),
        }
    }
}

/// An assistant message with text and a call renders the text before the call, where the model wrote it.
#[test]
fn text_before_a_call_renders_before_it() {
    if absent() {
        return;
    }
    let messages = Json::parse(
        r#"[{"role": "system", "content": "S"}, {"role": "user", "content": "List the files."},
            {"role": "assistant", "content": "Let me look.", "reasoning_content": "I should list them.",
             "tool_calls": [{"id": "c0", "function": {"name": "shell", "arguments": {"command": "ls"}}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "a.txt"}]"#,
    )
    .unwrap();
    let r = chat().render(messages.items(), &[], true, true).unwrap();
    let turn = &r.text[r.text.rfind("<|turn>model\n").unwrap()..];
    assert!(
        turn.starts_with(
            "<|turn>model\n<|channel>thought\nI should list them.\n<channel|>Let me look.<|tool_call>call:shell"
        ),
        "{turn}"
    );
    assert!(turn.ends_with("<tool_response|><|channel>thought\n"), "{turn}");
    assert_eq!(turn.matches("<turn|>").count(), 0, "one open model turn: {turn}");
}
