//! Continuations in the Gemma 4 family (gogi "Continuations", spec 11.3c): the rendering of an open message, its seam
//! check, and the reply parser after it.
//!
//! The generation prompt is `<|turn>model\n<|channel>thought\n` with thinking on, and
//! `<|turn>model\n<|channel>thought\n<channel|>` with thinking off. An open message renders as that prompt, then the
//! open text as the model writes it (spec 11.3c, "The Gemma 4 rendering").

mod common;

use common::model::{absent, family};
use gogi_proto::context::Context;
use gogi_proto::generation::Phase;
use gogi_proto::message::Message;
use nagaya_council::family::{Family, FamilyError, ReplyStart};
use serde_json::{Value, json};

fn text(ids: &[u32]) -> String {
    family().tokenizer().decode(ids).expect("the tokens make text")
}

fn tpl(s: &str) -> Vec<u32> {
    family().tokenizer().encode(s).unwrap()
}

/// A context of one marked user message and `last`.
fn context(thinking: bool, last: Value) -> Context {
    let value = json!({"thinking": thinking, "messages": [
        {"role": "user", "content": "Write the config.", "snap": true}, last]});
    serde_json::from_value(value).expect("the test context is valid")
}

fn open(reasoning: Option<&str>, content: Option<&str>) -> Value {
    let mut m = json!({"role": "assistant", "continue": true});
    if let Some(r) = reasoning {
        m["reasoning"] = json!(r);
    }
    if let Some(c) = content {
        m["content"] = json!(c);
    }
    m
}

/// The context without its last message.
fn before(c: &Context) -> Context {
    let mut c = c.clone();
    c.messages.pop();
    c
}

fn refusal(e: FamilyError) -> gogi_proto::Invalid {
    match e {
        FamilyError::Client(invalid) => invalid,
        other => panic!("want a client error, got {other:?}"),
    }
}

#[test]
fn an_open_message_renders_as_the_generation_prompt_and_the_open_text() {
    if absent() {
        return;
    }
    for (thinking, last, tail) in [
        (
            true,
            open(Some("I should list"), None),
            "<|channel>thought\nI should list",
        ),
        (true, open(Some("plan"), Some("")), "<|channel>thought\nplan<channel|>"),
        (
            true,
            open(Some("plan"), Some("fn main() {")),
            "<|channel>thought\nplan<channel|>fn main() {",
        ),
        (
            true,
            open(None, Some("fn main() {")),
            "<|channel>thought\n<channel|>fn main() {",
        ),
        (false, open(None, Some("Plan: ")), "<|channel>thought\n<channel|>Plan: "),
        (false, open(None, Some("")), "<|channel>thought\n<channel|>"),
    ] {
        let c = context(thinking, last.clone());
        let user = family().render(&before(&c), false).unwrap();
        for prompt in [false, true] {
            let r = family().render(&c, prompt).unwrap();
            assert_eq!(
                text(&r.tokens),
                format!("{}<|turn>model\n{tail}", text(&user.tokens)),
                "{last}"
            );
            assert_eq!(
                r.end,
                r.tokens.len(),
                "{last}: no generation prompt follows an open message"
            );
            assert_eq!(r.boundaries.len(), 2, "{last}");
            assert_eq!(r.boundaries[0].tokens, user.end, "{last}");
            assert_eq!((r.boundaries[1].message, r.boundaries[1].tokens), (1, r.end), "{last}");
        }
    }
}

/// The open text is as sent. The template trims the content of a closed message, so the trailing space of a cut
/// reply would go, and the model would see other text than `done` holds.
#[test]
fn the_open_text_keeps_its_white_space() {
    if absent() {
        return;
    }
    let r = family()
        .render(&context(false, open(None, Some("fn main() "))), true)
        .unwrap();
    assert!(
        text(&r.tokens).ends_with("<channel|>fn main() "),
        "{:?}",
        text(&r.tokens)
    );
    let r = family()
        .render(&context(true, open(Some("I list "), None)), true)
        .unwrap();
    assert!(text(&r.tokens).ends_with("thought\nI list "), "{:?}", text(&r.tokens));
}

/// The open rendering continues the generation prompt: a reply that the client starts is the same prompt with its
/// text after it.
#[test]
fn the_open_rendering_begins_with_the_generation_prompt_of_the_context_before_it() {
    if absent() {
        return;
    }
    for (thinking, last) in [
        (true, open(Some("I should list the files"), None)),
        (true, open(None, Some("Plan: list the files"))),
        (false, open(None, Some("Plan: list the files"))),
    ] {
        let c = context(thinking, last.clone());
        let prompt = family().render(&before(&c), true).unwrap().tokens;
        let r = family().render(&c, true).unwrap();
        assert!(r.tokens.starts_with(&prompt), "{last}");
        assert!(r.tokens.len() > prompt.len(), "{last}");
    }
}

/// After a tool result the template opens the thought channel itself. The open text follows that channel once.
#[test]
fn an_open_message_after_a_tool_result_follows_the_prompt_of_the_template() {
    if absent() {
        return;
    }
    let value = json!({"thinking": true, "tools": [{"name": "shell", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": "List the files."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "name": "shell", "input": {}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "a.txt"},
            {"role": "assistant", "reasoning": "One file.", "continue": true}]});
    let c: Context = serde_json::from_value(value).unwrap();
    let prompt = family().render(&before(&c), true).unwrap().tokens;
    assert!(text(&prompt).ends_with("<|channel>thought\n"), "{:?}", text(&prompt));
    let r = family().render(&c, true).unwrap();
    assert_eq!(text(&r.tokens), format!("{}One file.", text(&prompt)));
}

/// The check of the reference's `prefill`: a field whose first characters merge with the text before it is a 400 that
/// names `continue`. The thought channel ends with a newline, and a newline after it merges.
#[test]
fn an_open_field_that_merges_with_the_text_before_it_is_refused_naming_continue() {
    if absent() {
        return;
    }
    let e = refusal(
        family()
            .render(&context(true, open(Some("\nplan"), None)), true)
            .unwrap_err(),
    );
    assert_eq!(e.param.as_deref(), Some("continue"), "{}", e.message);
    assert!(e.message.contains("the open reasoning"), "{}", e.message);
    // A marker comes before the content, so its first character merges with nothing.
    family()
        .render(&context(false, open(None, Some("\n```toml"))), true)
        .unwrap();
    family()
        .render(&context(true, open(Some("plan\n"), None)), true)
        .unwrap();
}

/// gogi rule 2: the open text is content, never control. `<channel|>` in it is text.
#[test]
fn template_text_in_an_open_field_stays_text() {
    if absent() {
        return;
    }
    let close = family().tokenizer().token_id("<channel|>").unwrap();
    let c = context(true, open(Some("I write <channel|> as text"), None));
    let r = family().render(&c, true).unwrap();
    assert!(!r.tokens.contains(&close), "{:?}", text(&r.tokens));
    assert!(
        r.control_text
            .iter()
            .any(|hit| hit.place == "message:1" && hit.token == "<channel|>")
    );
}

#[test]
fn snap_on_the_open_message_changes_nothing() {
    if absent() {
        return;
    }
    let mut marked = open(None, Some("Plan:"));
    marked["snap"] = json!(true);
    let a = family()
        .render(&context(false, open(None, Some("Plan:"))), true)
        .unwrap();
    let b = family().render(&context(false, marked), true).unwrap();
    assert_eq!(a, b);
}

/// A reply cut by `max_tokens` and sent back open renders as the prompt and the sampled tokens. So the model goes on
/// from the tokens that it saw (spec 11.3c, "The Gemma 4 rendering").
#[test]
fn a_cut_reply_sent_back_open_renders_as_its_prompt_and_its_tokens() {
    if absent() {
        return;
    }
    for (thinking, reply) in [
        (true, "I should list the files "),
        (true, "I list them.<channel|>Two files: a"),
        (true, "I list them.<channel|>"),
        (false, "Two files: a.txt and "),
    ] {
        let mut c = context(thinking, json!({"role": "user", "content": "Go."}));
        c.messages.pop();
        let prompt = family().render(&c, true).unwrap().tokens;
        let sampled = tpl(reply);
        let mut parser = family().reply_parser(ReplyStart::of(&c), &[]).unwrap();
        for &id in &sampled {
            parser.push(id).unwrap();
        }
        let phase = parser.phase();
        let parsed = parser.finish(true);
        // gogi "Continuations": `phase` content sends the reply back with a `content` member, and reasoning without.
        let mut message = json!({"role": "assistant", "continue": true});
        if !parsed.reasoning.is_empty() {
            message["reasoning"] = json!(parsed.reasoning);
        }
        if phase == Phase::Content {
            message["content"] = json!(parsed.content);
        }
        c.messages.push(serde_json::from_value::<Message>(message).unwrap());
        let r = family().render(&c, true).unwrap();
        assert_eq!(
            text(&r.tokens),
            text(&[prompt.clone(), sampled.clone()].concat()),
            "{reply:?}"
        );
        assert_eq!(r.tokens, [prompt, sampled].concat(), "{reply:?}");
    }
}

/// gogi "Continuations": `done` joins the open text and the new text, and the server trims nothing at the seam. A
/// reply that ends with `max_tokens` keeps its trailing white space. The Gemma 4 parser trims no reply.
#[test]
fn a_continuation_and_a_cut_reply_keep_their_white_space() {
    if absent() {
        return;
    }
    let parse = |start: ReplyStart, reply: &str| {
        let mut parser = family().reply_parser(start, &[]).unwrap();
        for id in tpl(reply) {
            parser.push(id).unwrap();
        }
        let parsed = parser.finish(true);
        (parsed.reasoning, parsed.content)
    };
    let open = ReplyStart {
        thinking: false,
        open: true,
    };
    assert_eq!(parse(open, " {}\n"), (String::new(), " {}\n".to_owned()));
    let thought = ReplyStart {
        thinking: true,
        open: true,
    };
    assert_eq!(
        parse(thought, " more <channel|> ok "),
        (" more ".to_owned(), " ok ".to_owned())
    );
}
