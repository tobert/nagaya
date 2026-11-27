//! The reply parser of the council family (spec 11.3), through the `ReplyParser` trait of `nagaya-council`.
//!
//! The cases are `tests/fixtures/reply.json`, the outputs of the reference's Python. The raw parser has its own tests
//! in `tests/reply.rs`. These tests check the adapter: the stop token, the phase and piece of each step, the tail, the
//! calls and the tool list.

mod common;

use common::{absent, gguf_path, tokenizer};
use gogi_proto::context::Tool;
use gogi_proto::generation::Phase;
use nagaya_council::family::{Family, FamilyError, ParsedReply, ReplyCall, ReplyParser, ReplyStart, ReplyStep};
use nagaya_family_qwen38::Qwen38Family;
use serde_json::{Value, json};
use std::sync::OnceLock;

/// `<|im_end|>`, the first stop id of the reference (`STOP_IDS` in `service/generate.py`).
const IM_END: u32 = 248046;
/// `<|endoftext|>`, the second.
const END_OF_TEXT: u32 = 248044;

fn family() -> &'static Qwen38Family {
    static F: OnceLock<Qwen38Family> = OnceLock::new();
    F.get_or_init(|| Qwen38Family::from_gguf(&gguf_path()).unwrap_or_else(|e| panic!("cannot build the family: {e}")))
}

fn fixture() -> &'static Value {
    static F: OnceLock<Value> = OnceLock::new();
    F.get_or_init(|| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reply.json");
        serde_json::from_str(&std::fs::read_to_string(&path).expect("the fixture exists")).expect("the fixture is JSON")
    })
}

/// The tools of a named tool set, as the tools of a context. A tool set may hold a bare tool or a `function` tool.
fn tools_of(name: &str) -> Vec<Tool> {
    let set = &fixture()["toolsets"][name];
    set.as_array().map_or_else(Vec::new, |tools| {
        tools
            .iter()
            .map(|t| {
                let function = t.get("function").unwrap_or(t);
                let schema = function.get("parameters").cloned().unwrap_or_else(|| json!({}));
                let mut tool = Tool::new(function["name"].as_str().expect("a tool name"), schema);
                tool.description = function.get("description").and_then(Value::as_str).map(str::to_string);
                tool
            })
            .collect()
    })
}

fn phase_of(name: &str) -> Phase {
    match name {
        "think" => Phase::Reasoning,
        "answer" => Phase::Content,
        "tool" => Phase::Call,
        other => panic!("The reference has no phase {other:?}"),
    }
}

/// The ids of a case, then a stop token when the reference stopped. The reference's loop does not feed the stop token
/// to its tracker.
fn ids_of(case: &Value) -> Vec<u32> {
    let mut ids: Vec<u32> = case["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|i| u32::try_from(i.as_u64().expect("an id")).expect("a u32"))
        .collect();
    if case["finish"] == "stop" {
        ids.push(IM_END);
    }
    ids
}

fn parser_of(case: &Value) -> Box<dyn ReplyParser> {
    let tools = tools_of(case["tools"].as_str().expect("a tool set name"));
    family()
        .reply_parser(
            ReplyStart::prompt(case["thinking"].as_bool().expect("thinking")),
            &tools,
        )
        .unwrap_or_else(|e| panic!("{}: {e}", case["name"]))
}

fn run(case: &Value) -> (Vec<ReplyStep>, ParsedReply) {
    let mut parser = parser_of(case);
    let steps = ids_of(case)
        .into_iter()
        .map(|id| parser.push(id).unwrap_or_else(|e| panic!("{}: {e}", case["name"])))
        .collect();
    (steps, parser.finish(false))
}

fn cases() -> &'static Vec<Value> {
    fixture()["cases"].as_array().expect("cases")
}

/// The steps that the reference's released text gives for one token: the phase of the token, and the text released in
/// that phase. The end of thinking is not part of the reasoning.
fn want_step(event: &Value) -> ReplyStep {
    let phase = phase_of(event[0].as_str().expect("a phase"));
    let piece: String = event[2]
        .as_array()
        .expect("released")
        .iter()
        .filter(|s| phase_of(s[0].as_str().expect("a phase")) == phase)
        .map(|s| s[1].as_str().expect("text"))
        .map(|text| match phase {
            Phase::Reasoning => text.strip_suffix("</think>").unwrap_or(text),
            _ => text,
        })
        .collect();
    ReplyStep { phase, piece }
}

#[test]
fn each_token_has_the_phase_and_the_released_text_of_the_reference() {
    if common::absent() {
        return;
    }
    for case in cases() {
        let name = &case["name"];
        let (steps, _) = run(case);
        let events = case["events"].as_array().expect("events");
        // The last step is the stop token, which the reference does not feed. It releases nothing.
        let stop = usize::from(case["finish"] == "stop");
        assert_eq!(steps.len(), events.len() + stop, "token count of {name}");
        for (n, (got, event)) in steps.iter().zip(events).enumerate() {
            assert_eq!(got, &want_step(event), "step {n} of {name}");
        }
    }
}

#[test]
fn the_stop_token_adds_no_text_and_keeps_the_phase() {
    if common::absent() {
        return;
    }
    let mut parser = family().reply_parser(ReplyStart::prompt(false), &[]).unwrap();
    for id in tokenizer().encode("Hello").unwrap() {
        parser.push(id).unwrap();
    }
    assert_eq!(
        parser.push(IM_END).unwrap(),
        ReplyStep {
            phase: Phase::Content,
            piece: String::new()
        }
    );
    let reply = parser.finish(false);
    assert_eq!(reply.content, "Hello");
    assert!(reply.ended);
}

/// gogi "Continuations": `done.phase` of a cut reply is the phase that its next token would have.
#[test]
fn a_reply_parser_names_the_phase_of_the_next_token() {
    if absent() {
        return;
    }
    let tools = vec![serde_json::from_value(json!({"name": "sh", "input_schema": {"type": "object"}})).unwrap()];
    let mut parser = family().reply_parser(ReplyStart::prompt(true), &tools).unwrap();
    assert_eq!(parser.phase(), Phase::Reasoning);
    for id in tokenizer().encode("plan").unwrap() {
        parser.push(id).unwrap();
    }
    assert_eq!(parser.phase(), Phase::Reasoning);
    parser.push(tokenizer().token_id("</think>").unwrap()).unwrap();
    assert_eq!(
        parser.phase(),
        Phase::Content,
        "a reply cut just after </think> resumes in its content"
    );
    parser.push(tokenizer().token_id("<tool_call>").unwrap()).unwrap();
    assert_eq!(parser.phase(), Phase::Call);
    let open = family()
        .reply_parser(
            ReplyStart {
                thinking: false,
                open: true,
            },
            &[],
        )
        .unwrap();
    assert_eq!(open.phase(), Phase::Content);
}

fn same_call(got: &ReplyCall, want: &Value, name: &Value) {
    match got {
        ReplyCall::Parsed { name: n, input } => {
            assert!(
                want.get("error").is_none(),
                "{name}: parsed a call that the reference refused: {want}"
            );
            assert_eq!(n, want["name"].as_str().unwrap(), "name in {name}");
            // The input keeps the order of the text, so compare the text of the object.
            assert_eq!(
                serde_json::to_string(input).unwrap(),
                serde_json::to_string(&want["arguments"]).unwrap(),
                "input in {name}"
            );
        }
        ReplyCall::Malformed { text, error } => {
            let want_error = want["error"]
                .as_str()
                .unwrap_or_else(|| panic!("{name}: refused a call: {want}"));
            assert_eq!(text, want["raw"].as_str().unwrap(), "text in {name}");
            // The text of a JSON error of Python is not ported, so compare the text around it.
            match want_error.find("not JSON (") {
                Some(at) => {
                    let head = &want_error[..at + "not JSON (".len()];
                    assert!(
                        error.starts_with(head) && error.ends_with(')'),
                        "{error:?} against {want_error:?}"
                    );
                }
                None => assert_eq!(error, want_error, "error in {name}"),
            }
        }
    }
}

#[test]
fn the_reply_is_the_reply_of_the_reference() {
    if common::absent() {
        return;
    }
    for case in cases() {
        let name = &case["name"];
        let (_, reply) = run(case);
        let want = &case["result"];
        assert_eq!(
            reply.reasoning,
            want["reasoning_content"].as_str().unwrap(),
            "reasoning of {name}"
        );
        assert_eq!(reply.content, want["content"].as_str().unwrap(), "content of {name}");
        let calls = want["tool_calls"].as_array().unwrap();
        assert_eq!(reply.calls.len(), calls.len(), "call count of {name}");
        for (got, want) in reply.calls.iter().zip(calls) {
            same_call(got, want, name);
        }
        // The stop token ends the turn. A cut by length does not.
        assert_eq!(reply.ended, case["finish"] == "stop", "ended of {name}");
    }
}

#[test]
fn the_tail_is_the_text_the_reference_held_back() {
    if common::absent() {
        return;
    }
    let mut with_tail = 0;
    for case in cases() {
        let name = &case["name"];
        let (_, reply) = run(case);
        let want: Vec<ReplyStep> = case["tail"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| ReplyStep {
                phase: phase_of(s[0].as_str().unwrap()),
                piece: s[1].as_str().unwrap().to_string(),
            })
            .collect();
        with_tail += usize::from(!want.is_empty());
        assert_eq!(reply.tail, want, "tail of {name}");
    }
    assert!(with_tail > 0, "no case has a tail");
}

/// gogi's rule: the pieces of a part, joined, equal the part up to the white space that `strip()` removes. The
/// tail pieces belong to the parts too.
#[test]
fn the_pieces_of_a_part_join_to_the_part() {
    if common::absent() {
        return;
    }
    use nagaya_family_qwen38::pystr::strip;
    for case in cases() {
        let name = &case["name"];
        let (steps, reply) = run(case);
        let join = |phase: Phase| -> String {
            steps
                .iter()
                .chain(&reply.tail)
                .filter(|s| s.phase == phase)
                .map(|s| s.piece.as_str())
                .collect()
        };
        assert_eq!(
            strip(&join(Phase::Reasoning), None, true, true),
            reply.reasoning,
            "reasoning of {name}"
        );
        assert_eq!(
            strip(&join(Phase::Content), None, true, true),
            reply.content,
            "content of {name}"
        );
    }
}

/// The reply of a model that thinks and then calls a tool: the case the first kaijutsu run has.
#[test]
fn a_thinking_block_then_a_tool_call() {
    if common::absent() {
        return;
    }
    let tools = vec![Tool::new(
        "read_file",
        json!({"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}}),
    )];
    let mut parser = family().reply_parser(ReplyStart::prompt(true), &tools).unwrap();
    let text = "I should read it.</think>\n\nOpening the file.\n\n<tool_call>\n<function=read_file>\n\
                <parameter=path>\n/etc/hosts\n</parameter>\n<parameter=limit>\n20\n</parameter>\n\
                </function>\n</tool_call>";
    let mut steps = Vec::new();
    for id in tokenizer().encode(text).unwrap() {
        steps.push(parser.push(id).unwrap());
    }
    steps.push(parser.push(IM_END).unwrap());
    let reply = parser.finish(false);
    assert_eq!(reply.reasoning, "I should read it.");
    assert_eq!(reply.content, "Opening the file.");
    assert_eq!(
        reply.calls,
        vec![ReplyCall::Parsed {
            name: "read_file".into(),
            input: match json!({"path": "/etc/hosts", "limit": 20}) {
                Value::Object(m) => m,
                _ => unreachable!(),
            },
        }]
    );
    assert!(reply.ended);
    // The steps run through the three phases in order.
    let phases: Vec<Phase> = steps.iter().map(|s| s.phase).collect();
    assert_eq!(phases.first(), Some(&Phase::Reasoning));
    assert!(phases.contains(&Phase::Content));
    assert_eq!(phases.iter().rev().nth(1), Some(&Phase::Call));
    // No piece holds the end-of-thinking marker or the text of a stop token.
    assert!(
        steps
            .iter()
            .all(|s| !s.piece.contains("</think>") && !s.piece.contains("<|im_end|>"))
    );
}

#[test]
fn the_other_stop_token_ends_the_turn_too() {
    if common::absent() {
        return;
    }
    let mut parser = family().reply_parser(ReplyStart::prompt(false), &[]).unwrap();
    parser.push(tokenizer().encode("Hi").unwrap()[0]).unwrap();
    parser.push(END_OF_TEXT).unwrap();
    assert!(parser.finish(false).ended);
}

#[test]
fn the_parser_says_that_the_reply_ended_after_a_stop_token() {
    if common::absent() {
        return;
    }
    for stop in [IM_END, END_OF_TEXT] {
        let mut parser = family().reply_parser(ReplyStart::prompt(false), &[]).unwrap();
        parser.push(tokenizer().encode("Hi").unwrap()[0]).unwrap();
        assert!(!parser.ended(), "text does not end the reply");
        parser.push(stop).unwrap();
        assert!(parser.ended(), "stop token {stop} ends the reply");
    }
}

#[test]
fn a_cut_reply_has_not_ended() {
    if common::absent() {
        return;
    }
    let mut parser = family().reply_parser(ReplyStart::prompt(false), &[]).unwrap();
    for id in tokenizer().encode("Hi th").unwrap() {
        parser.push(id).unwrap();
    }
    let reply = parser.finish(false);
    assert_eq!((reply.content.as_str(), reply.ended), ("Hi th", false));
}

#[test]
fn a_token_after_the_stop_token_is_an_internal_error() {
    if common::absent() {
        return;
    }
    let mut parser = family().reply_parser(ReplyStart::prompt(false), &[]).unwrap();
    parser.push(IM_END).unwrap();
    let e = parser.push(tokenizer().encode("x").unwrap()[0]).unwrap_err();
    assert!(matches!(e, FamilyError::Internal(_)), "{e:?}");
}

#[test]
fn an_id_outside_the_vocabulary_is_an_internal_error() {
    if common::absent() {
        return;
    }
    let mut parser = family().reply_parser(ReplyStart::prompt(false), &[]).unwrap();
    let id = u32::try_from(family().vocab_size()).unwrap();
    let e = parser.push(id).unwrap_err();
    assert!(
        matches!(&e, FamilyError::Internal(m) if m.contains(&id.to_string())),
        "{e:?}"
    );
}

#[test]
fn a_tool_list_the_parser_cannot_read_is_a_client_error() {
    if common::absent() {
        return;
    }
    let tools = vec![Tool::new("f", json!({"properties": {"p": {"type": "text"}}}))];
    match family().reply_parser(ReplyStart::prompt(false), &tools) {
        Err(FamilyError::Client(invalid)) => assert!(invalid.message.contains("unsupported"), "{}", invalid.message),
        Err(other) => panic!("want a client error, got {other:?}"),
        Ok(_) => panic!("accepted a tool with the type \"text\""),
    }
}

#[test]
fn an_integer_that_json_numbers_cannot_hold_makes_a_malformed_call() {
    if common::absent() {
        return;
    }
    let tools = vec![Tool::new("f", json!({"properties": {"n": {"type": "integer"}}}))];
    let mut parser = family().reply_parser(ReplyStart::prompt(false), &tools).unwrap();
    let text = "<tool_call>\n<function=f>\n<parameter=n>\n123456789012345678901234567890\n</parameter>\n</function>\n</tool_call>";
    for id in tokenizer().encode(text).unwrap() {
        parser.push(id).unwrap();
    }
    parser.push(IM_END).unwrap();
    let reply = parser.finish(false);
    match reply.calls.as_slice() {
        [ReplyCall::Malformed { text, error }] => {
            assert!(text.contains("123456789012345678901234567890"), "{text}");
            assert!(error.contains("64"), "{error}");
        }
        other => panic!("want one malformed call, got {other:?}"),
    }
}
