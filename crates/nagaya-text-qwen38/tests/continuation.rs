//! Continuations in the Qwen3.8 family (gogi "Continuations", spec 11.3a): the rendering of an open message, its seam
//! check, and the reply parser after it.
//!
//! The chat template of the GGUF renders a closed assistant message as
//! `<|im_start|>assistant\n<think>\n{reasoning|trim}\n</think>\n\n{content|trim}<|im_end|>\n`, with thinking on or
//! off. Its generation prompt is `<|im_start|>assistant\n<think>\n` with thinking on, and
//! `<|im_start|>assistant\n<think>\n\n</think>\n\n` with thinking off.

mod common;

use common::{gguf_path, tokenizer};
use gogi_proto::context::Context;
use gogi_proto::generation::Phase;
use nagaya_council::family::{Family, FamilyError, ReplyStart};
use nagaya_text_qwen38::Qwen38Family;
use serde_json::{Value, json};
use std::sync::OnceLock;

fn family() -> &'static Qwen38Family {
    static F: OnceLock<Qwen38Family> = OnceLock::new();
    F.get_or_init(|| Qwen38Family::from_gguf(&gguf_path()).unwrap_or_else(|e| panic!("cannot build the family: {e}")))
}

fn text(ids: &[u32]) -> String {
    tokenizer().decode(ids).expect("the tokens make text")
}

const USER: &str = "<|im_start|>user\nWrite the config.<|im_end|>\n";

/// A context of one marked user message and `last`.
fn context(thinking: Option<bool>, last: Value) -> Context {
    let mut value = json!({"messages": [{"role": "user", "content": "Write the config.", "snap": true}, last]});
    if let Some(thinking) = thinking {
        value["thinking"] = json!(thinking);
    }
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

fn closed(c: &Context) -> Context {
    let mut c = c.clone();
    c.messages.last_mut().unwrap().open = false;
    c
}

fn refusal(e: FamilyError) -> gogi_proto::Invalid {
    match e {
        FamilyError::Client(invalid) => invalid,
        other => panic!("want a client error, got {other:?}"),
    }
}

#[test]
fn an_open_message_renders_as_a_reply_up_to_its_open_text_without_its_end() {
    for (thinking, last, tail) in [
        (true, open(Some("I should list"), None), "<think>\nI should list"),
        (true, open(Some("plan"), Some("")), "<think>\nplan\n</think>\n\n"),
        (
            true,
            open(Some("plan"), Some("fn main() {")),
            "<think>\nplan\n</think>\n\nfn main() {",
        ),
        (
            true,
            open(None, Some("fn main() {")),
            "<think>\n\n</think>\n\nfn main() {",
        ),
        (false, open(None, Some("Plan: ")), "<think>\n\n</think>\n\nPlan: "),
        (false, open(None, Some("")), "<think>\n\n</think>\n\n"),
    ] {
        let c = context(Some(thinking), last.clone());
        for prompt in [false, true] {
            let r = family().render(&c, prompt).unwrap();
            assert_eq!(
                text(&r.tokens),
                format!("{USER}<|im_start|>assistant\n{tail}"),
                "{last}"
            );
            assert_eq!(
                r.end,
                r.tokens.len(),
                "{last}: no generation prompt follows an open message"
            );
            assert_eq!(r.boundaries.len(), 2, "{last}");
            assert_eq!(
                r.boundaries[0].tokens,
                tokenizer().encode(USER).unwrap().len(),
                "{last}"
            );
            assert_eq!((r.boundaries[1].message, r.boundaries[1].tokens), (1, r.end), "{last}");
        }
    }
}

/// The open text is as sent. The template trims the content of a closed message, so the trailing space of a cut
/// reply would go, and the model would see other text than `done` holds.
#[test]
fn the_open_text_keeps_its_white_space() {
    let c = context(Some(false), open(None, Some("fn main() ")));
    let r = family().render(&c, true).unwrap();
    assert!(
        text(&r.tokens).ends_with("</think>\n\nfn main() "),
        "{:?}",
        text(&r.tokens)
    );
}

/// The open rendering continues the generation prompt: a reply that the client starts is the same prompt with its
/// text after it. The head rule of a cut reply needs this (gogi "Continuations").
#[test]
fn the_open_rendering_begins_with_the_generation_prompt_of_the_context_before_it() {
    for (thinking, last) in [
        (true, open(Some("I should list the files"), None)),
        (false, open(None, Some("Plan: list the files"))),
    ] {
        let c = context(Some(thinking), last.clone());
        let mut before = c.clone();
        before.messages.pop();
        let prompt = family().render(&before, true).unwrap().tokens;
        let r = family().render(&c, true).unwrap();
        assert!(r.tokens.starts_with(&prompt), "{last}");
        assert!(r.tokens.len() > prompt.len(), "{last}");
    }
}

/// Where the open rendering's tokens begin the closed rendering's tokens, and where they do not. A cut reply that
/// the client sends back closed starts from the head only in the first case (gogi "Continuations"). They differ only
/// where the template trims white space at the end of the open field.
#[test]
fn the_open_rendering_begins_the_closed_rendering_unless_the_template_trims_or_merges() {
    let prefix = |last: Value, thinking: bool| {
        let c = context(Some(thinking), last);
        let open = family().render(&c, false).unwrap().tokens;
        let closed = family().render(&closed(&c), false).unwrap().tokens;
        closed.starts_with(&open)
    };
    // Content without white space at its ends: the closed rendering adds only <|im_end|>, a special token.
    assert!(prefix(open(Some("plan"), Some("fn main() {}")), true));
    assert!(prefix(open(None, Some("Plan: list the files.")), false));
    assert!(prefix(open(Some("plan"), Some("")), true));
    // An open thinking ends before "\n</think>". This tokenizer keeps a newline apart from the text before it, so
    // the newline of the closed rendering merges with no last token.
    assert!(prefix(open(Some("I list the files."), None), true));
    assert!(prefix(open(Some("I list the files:"), None), true));
    // The template trims the reasoning and the content of a closed message, so white space at the end goes.
    assert!(!prefix(open(None, Some("fn main() ")), false));
    assert!(!prefix(open(Some("I list "), None), true));
    assert!(!prefix(open(Some("plan\n\n"), None), true));
}

/// mk's prefix check at the seam (`service/generate.py`, `_with_prefill`): a field whose first characters merge
/// with the text before it is a 400 that names `continue`. mk's example is a newline after the template's last
/// newline.
#[test]
fn an_open_field_that_merges_with_the_text_before_it_is_refused_naming_continue() {
    let e = refusal(
        family()
            .render(&context(Some(false), open(None, Some("\n```toml"))), true)
            .unwrap_err(),
    );
    assert_eq!(e.param.as_deref(), Some("continue"), "{}", e.message);
    assert!(
        e.message.contains("the open content does not tokenize on its own"),
        "{}",
        e.message
    );
    assert!(
        e.message.contains(r#"which merges with its start into "\n\n\n""#),
        "{}",
        e.message
    );
    let e = refusal(
        family()
            .render(&context(Some(true), open(Some("\nplan"), None)), true)
            .unwrap_err(),
    );
    assert_eq!(e.param.as_deref(), Some("continue"), "{}", e.message);
    assert!(e.message.contains("the open reasoning"), "{}", e.message);
    // The same text starts well with a character that does not merge.
    family()
        .render(&context(Some(false), open(None, Some("```toml\n"))), true)
        .unwrap();
    family()
        .render(&context(Some(true), open(Some("plan\n"), None)), true)
        .unwrap();
}

/// gogi rule 2: the open text is content, never control. `</think>` in it is text.
#[test]
fn template_text_in_an_open_field_stays_text() {
    let end_think = tokenizer().token_id("</think>").unwrap();
    let c = context(Some(true), open(Some("I write </think> as text"), None));
    let r = family().render(&c, true).unwrap();
    assert!(!r.tokens.contains(&end_think), "{:?}", text(&r.tokens));
    assert!(
        r.control_text
            .iter()
            .any(|hit| hit.place == "message:1" && hit.token == "</think>")
    );
}

#[test]
fn snap_on_the_open_message_changes_nothing() {
    let mut marked = open(None, Some("Plan:"));
    marked["snap"] = json!(true);
    let a = family()
        .render(&context(Some(false), open(None, Some("Plan:"))), true)
        .unwrap();
    let b = family().render(&context(Some(false), marked), true).unwrap();
    assert_eq!(a, b);
}

fn parse(start: ReplyStart, reply: &str, max_tokens: bool) -> (String, String, Phase) {
    let mut parser = family().reply_parser(start, &[]).unwrap();
    for id in tokenizer().encode(reply).unwrap() {
        parser.push(id).unwrap();
    }
    let phase = parser.phase();
    let parsed = parser.finish(max_tokens);
    (parsed.reasoning, parsed.content, phase)
}

/// gogi "Continuations": `done` joins the open text and the new text, and the server trims nothing at the seam. The
/// parser trims the ends only as for any reply.
#[test]
fn a_continuation_keeps_the_white_space_at_the_start_of_its_first_part() {
    let open = ReplyStart {
        thinking: false,
        open: true,
    };
    assert_eq!(
        parse(open, " {}\n", false),
        (String::new(), " {}".to_owned(), Phase::Content)
    );
    assert_eq!(
        parse(ReplyStart::prompt(false), " {}\n", false).1,
        "{}",
        "a reply after the prompt trims"
    );
    let thought = ReplyStart {
        thinking: true,
        open: true,
    };
    let (reasoning, content, _) = parse(thought, " more\n</think>\n\nok", false);
    assert_eq!(
        (reasoning.as_str(), content.as_str()),
        (" more", "ok"),
        "the content after </think> is a new part"
    );
}

/// gogi "Continuations": a reply that ends with `max_tokens` keeps its trailing white space, so the client can send
/// it back exactly. The part before a later part is trimmed as for any reply.
#[test]
fn a_cut_reply_keeps_the_white_space_at_the_end_of_the_part_it_was_in() {
    let start = ReplyStart::prompt(true);
    assert_eq!(
        parse(start, "I list ", true),
        ("I list ".to_owned(), String::new(), Phase::Reasoning)
    );
    assert_eq!(parse(start, "I list ", false).0, "I list");
    let (reasoning, content, phase) = parse(start, "plan \n</think>\n\nfn main() ", true);
    assert_eq!(
        (reasoning.as_str(), content.as_str(), phase),
        ("plan", "fn main() ", Phase::Content)
    );
    let (reasoning, content, phase) = parse(start, "plan\n</think>\n\n", true);
    assert_eq!(
        (reasoning.as_str(), content.as_str(), phase),
        ("plan", "", Phase::Content)
    );
}

/// The SHA-256 of tokens as 32-bit little-endian words.
fn digest(ids: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
    Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// The open renderings of the test GGUF, pinned. The template id does not cover an open message (Amy, 2026-10-21:
/// the id stays), so this test sees a change to `run_open` instead (kaibo job-5). Change a value only when the
/// rendering of an open message changes on purpose, and say so in the commit message.
#[test]
fn the_open_renderings_of_the_test_gguf_are_the_golden_ones() {
    let cases = [
        (true, open(Some("I should list the files"), None)),
        (true, open(Some("plan"), Some(""))),
        (true, open(Some("plan"), Some("fn main() {"))),
        (false, open(None, Some("Plan: "))),
    ];
    let got: Vec<String> = cases
        .iter()
        .map(|(thinking, last)| {
            digest(
                &family()
                    .render(&context(Some(*thinking), last.clone()), true)
                    .unwrap()
                    .tokens,
            )
        })
        .collect();
    assert_eq!(got, GOLDEN_OPEN);
}

const GOLDEN_OPEN: [&str; 4] = [
    "4b12d92690fc86c76885084bd5c07708114d3e1e97af194c098d970760732f0a",
    "a45581ef04f51365737ebda2bbdbc0c2175b0d0ee924b392979d2d6acf2f3024",
    "51c6705f5eb7fdc4338144b0ff5db08041c59705a46a68b4ca67ada9776ca16a",
    "c0ca686c21ae8c7151a7fbd65df874ddc97b482e1f6a26c6ae942415911821fd",
];
