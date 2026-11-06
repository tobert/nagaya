//! The reply parser against `parse_response` of transformers with the `response_template` of LFM3-1.7B (fixture
//! `reply.json`), and the differences that spec 11.11 names ("Differences of the parser").

mod common;

use common::ids;
use common::model::{absent_any, family, fixture};
use gogi_proto::context::{Context, Tool};
use gogi_proto::generation::Phase;
use nagaya_council::family::{Family, ParsedReply, ReplyCall, ReplyStart, ReplyStep};
use serde_json::{Value, json};

fn tools_of(v: &Value) -> Vec<Tool> {
    v.as_array()
        .map(|list| {
            list.iter()
                .map(|t| {
                    let f = &t["function"];
                    let mut tool = Tool::new(f["name"].as_str().unwrap(), f["parameters"].clone());
                    tool.description = f["description"].as_str().map(str::to_string);
                    tool
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Push the ids and finish. Check that the pieces of each part join to its text, up to the white space at its ends.
fn run(model: &str, thinking: bool, tools: &[Tool], tokens: &[u32], max_tokens: bool) -> (Vec<ReplyStep>, ParsedReply) {
    let fam = family(model);
    let mut parser = fam.reply_parser(ReplyStart::prompt(thinking), tools).unwrap();
    let steps: Vec<ReplyStep> = tokens.iter().map(|&t| parser.push(t).unwrap()).collect();
    let reply = parser.finish(max_tokens);
    let joined = |phase: Phase| -> String {
        steps
            .iter()
            .chain(&reply.tail)
            .filter(|s| s.phase == phase)
            .map(|s| s.piece.as_str())
            .collect()
    };
    assert_eq!(joined(Phase::Reasoning).trim(), reply.reasoning, "the reasoning pieces");
    assert_eq!(joined(Phase::Content).trim(), reply.content, "the content pieces");
    let call_text: String = joined(Phase::Call);
    for call in &reply.calls {
        if let ReplyCall::Malformed { text, .. } = call {
            assert!(call_text.contains(text.as_str()), "the call pieces hold {text:?}");
        }
    }
    (steps, reply)
}

fn calls_json(reply: &ParsedReply) -> Value {
    Value::Array(
        reply
            .calls
            .iter()
            .map(|c| match c {
                ReplyCall::Parsed { name, input } => json!({"name": name, "arguments": input}),
                ReplyCall::Malformed { text, error } => json!({"malformed": text, "error": error}),
            })
            .collect(),
    )
}

#[test]
fn each_reply_parses_as_the_reference_parses_it() {
    if absent_any() {
        return;
    }
    let f = fixture("reply.json");
    let mut same = 0;
    for case in f["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let thinking = case["thinking"].as_bool().unwrap();
        let tools = tools_of(&case["tools"]);
        let tokens = ids(&case["ids"]);
        let length = case["ending"] == "length";
        // Omni does not reason, so a reply with thinking runs on the text model only.
        let models: &[&str] = if thinking { &["text"] } else { &["omni", "text"] };
        for &model in models {
            let (_, reply) = run(model, thinking, &tools, &tokens, length);
            assert_eq!(reply.ended, !length, "{model} {name}");
            let want = &case["parsed"];
            match case["difference"].as_str() {
                None => {
                    assert_eq!(reply.content, want["content"], "{model} {name}");
                    assert_eq!(reply.reasoning, want["reasoning"], "{model} {name}");
                    assert_eq!(calls_json(&reply), want["tool_calls"], "{model} {name}");
                    same += 1;
                }
                Some("unclosed") => {
                    assert!(
                        matches!(&reply.calls[..], [ReplyCall::Malformed { text, error }]
                            if text.starts_with("<tool_call>") && error.contains("did not close")),
                        "{model} {name}: {:?}",
                        reply.calls
                    );
                    assert_eq!(
                        want["tool_calls"].as_array().unwrap().len(),
                        1,
                        "the reference reads it"
                    );
                }
                Some("no function") => {
                    assert!(
                        matches!(&reply.calls[..], [ReplyCall::Malformed { error, .. }] if error.contains("<function=NAME>")),
                        "{model} {name}: {:?}",
                        reply.calls
                    );
                    assert_eq!(reply.content, "");
                    assert_eq!(
                        want["content"], "<tool_call>\nhello\n</tool_call>",
                        "the reference keeps it"
                    );
                }
                Some("content runs") => {
                    assert_eq!(reply.content, "First.\n\nThen this.", "{model} {name}");
                    assert_eq!(want["content"], "Then this.", "the reference keeps the last run");
                    assert_eq!(calls_json(&reply), want["tool_calls"], "{model} {name}");
                }
                Some(other) => panic!("the fixture names the difference {other:?}, which this test does not know"),
            }
        }
    }
    assert!(same >= 25, "{same} replies agree with the reference");
}

#[test]
fn an_integer_past_64_bits_makes_the_call_malformed() {
    if absent_any() {
        return;
    }
    let fam = family("text");
    let tools = vec![Tool::new(
        "f",
        json!({"type": "object", "properties": {"n": {"type": "integer"}}}),
    )];
    let text =
        "<tool_call>\n<function=f>\n<parameter=n>\n123456789012345678901234\n</parameter>\n</function>\n</tool_call>";
    let mut tokens = fam.tokenize_special(text).unwrap();
    tokens.push(fam.stop_tokens()[0]);
    let (_, reply) = run("text", false, &tools, &tokens, false);
    assert!(
        matches!(&reply.calls[..], [ReplyCall::Malformed { error, .. }] if error.contains("64 bits")),
        "{:?}",
        reply.calls
    );
}

#[test]
fn the_stop_tokens_come_from_the_generation_config() {
    if absent_any() {
        return;
    }
    let id = |model: &str, t: &str| family(model).tokenizer().token_id(t).unwrap();
    assert_eq!(family("text").stop_tokens(), [id("text", "<|im_end|>")]);
    assert_eq!(
        family("omni").stop_tokens(),
        [id("omni", "<|im_end|>"), id("omni", "<|text_end|>")]
    );
    let mut parser = family("omni").reply_parser(ReplyStart::prompt(false), &[]).unwrap();
    parser.push(id("omni", "<|text_end|>")).unwrap();
    assert!(parser.ended());
    assert!(
        parser.push(id("omni", "A")).is_err(),
        "a token after the stop is a defect"
    );
}

#[test]
fn an_omni_reply_never_starts_in_the_reasoning() {
    if absent_any() {
        return;
    }
    // gogi takes an absent `thinking` as true, and Omni does not reason (spec 11.11).
    let context: Context = serde_json::from_value(json!({"messages": [{"role": "user", "content": "hi"}]})).unwrap();
    let start = ReplyStart::of(&context);
    assert!(start.thinking);
    for (model, phase) in [("omni", Phase::Content), ("text", Phase::Reasoning)] {
        let parser = family(model).reply_parser(start, &[]).unwrap();
        assert_eq!(parser.phase(), phase, "{model}");
    }
}

#[test]
fn a_marker_gives_an_empty_piece_and_a_cut_character_waits() {
    if absent_any() {
        return;
    }
    let fam = family("text");
    let tok = fam.tokenizer();
    let crab = fam.tokenize("🦀").unwrap();
    let think_close = tok.token_id("</think>").unwrap();
    let mut tokens = fam.tokenize("hm").unwrap();
    tokens.push(think_close);
    tokens.extend(&crab);
    let (steps, reply) = run("text", true, &[], &tokens, true);
    assert_eq!(steps[1].piece, "");
    assert_eq!(steps[1].phase, Phase::Reasoning);
    assert_eq!(steps[2].piece, "", "the first byte of the crab waits");
    assert_eq!(steps.last().unwrap().piece, "🦀");
    assert_eq!((reply.reasoning.as_str(), reply.content.as_str()), ("hm", "🦀"));
    // A cut inside the crab gives U+FFFD in the tail.
    let (_, reply) = run("text", false, &[], &crab[..1], true);
    assert_eq!(reply.tail.len(), 1);
    assert_eq!(reply.content, "\u{FFFD}");
}

#[test]
fn max_tokens_keeps_the_end_and_an_open_reply_keeps_the_start() {
    if absent_any() {
        return;
    }
    let fam = family("text");
    let tokens = fam.tokenize("  a b  ").unwrap();
    let parse = |start: ReplyStart, max: bool| {
        let mut p = fam.reply_parser(start, &[]).unwrap();
        for &t in &tokens {
            p.push(t).unwrap();
        }
        p.finish(max).content
    };
    assert_eq!(parse(ReplyStart::prompt(false), false), "a b");
    assert_eq!(parse(ReplyStart::prompt(false), true), "a b  ");
    let open = ReplyStart {
        thinking: false,
        open: true,
    };
    assert_eq!(parse(open, false), "  a b");
}

#[test]
fn think_before_any_content_opens_the_reasoning_and_after_it_is_text() {
    if absent_any() {
        return;
    }
    let fam = family("text");
    let tokens = fam.tokenize_special("<think>hm</think>ok").unwrap();
    let (_, reply) = run("text", false, &[], &tokens, true);
    assert_eq!((reply.reasoning.as_str(), reply.content.as_str()), ("hm", "ok"));
    let tokens = fam.tokenize_special("ok <think>hm").unwrap();
    let (_, reply) = run("text", false, &[], &tokens, true);
    assert_eq!((reply.reasoning.as_str(), reply.content.as_str()), ("", "ok <think>hm"));
}
