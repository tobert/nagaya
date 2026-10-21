//! The reply parser against the outputs of mk's Python (`service/chatfmt.py` `ReplyTracker`, and the token loop of
//! `service/generate.py`). The fixture is `tests/fixtures/reply.json`, made by `tools/reply_fixture.py`.

mod common;

use common::{JsonExt, fixture, tokenizer};
use nagaya_text_qwen38::json::Json;
use nagaya_text_qwen38::reply::{Ending, Finish, Phase, Reply, ReplyParser, TokenEvent, ToolCall};

fn phase_of(mk: &str) -> Phase {
    match mk {
        "think" => Phase::Reasoning,
        "answer" => Phase::Content,
        "tool" => Phase::Call,
        other => panic!("mk has no phase {other:?}"),
    }
}

fn segments(j: &Json) -> Vec<(Phase, String)> {
    j.items()
        .iter()
        .map(|s| (phase_of(s.items()[0].str()), s.items()[1].str().to_string()))
        .collect()
}

fn toolset(fixture: &Json, name: &Json) -> Option<Json> {
    match name {
        Json::Null => None,
        Json::Str(n) => match fixture.get("toolsets").get(n) {
            Json::Null => None,
            tools => Some(tools.clone()),
        },
        other => panic!("not a tool set name: {other:?}"),
    }
}

/// Run one case the way mk's loop does. The stop token is not pushed.
fn run(f: &Json, case: &Json) -> (Vec<TokenEvent>, Reply) {
    let tools = toolset(f, case.get("tools"));
    let thinking = matches!(case.get("thinking"), Json::Bool(true));
    let mut parser = ReplyParser::new(tokenizer(), thinking, tools.as_ref()).unwrap();
    let events: Vec<TokenEvent> = case
        .get("ids")
        .ids()
        .iter()
        .map(|&id| parser.push(id).unwrap())
        .collect();
    let ending = match case.get("finish").str() {
        "stop" => Ending::StopToken,
        "length" => Ending::MaxTokens,
        other => panic!("mk has no finish {other:?}"),
    };
    (events, parser.finish(ending))
}

fn cases() -> (Json, Vec<Json>) {
    let f = fixture("reply.json");
    let cases = f.get("cases").items().to_vec();
    (f, cases)
}

/// The text of a JSON error of Python is not ported, so compare the text around it.
fn same_error(got: &str, want: &str) {
    match want.find("not JSON (") {
        Some(at) => {
            let head = &want[..at + "not JSON (".len()];
            assert!(
                got.starts_with(head) && got.ends_with(')'),
                "error {got:?} against {want:?}"
            );
        }
        None => assert_eq!(got, want),
    }
}

fn same_call(got: &ToolCall, want: &Json, name: &str) {
    match got {
        ToolCall::Parsed { name: n, input, .. } => {
            assert!(!want.has("error"), "{name}: parsed a call that mk refused: {want:?}");
            assert_eq!(n, want.get("name").str(), "name in {name}");
            assert_eq!(input, want.get("arguments"), "input in {name}");
        }
        ToolCall::Malformed { name: n, raw, error } => {
            assert!(want.has("error"), "{name}: refused a call that mk parsed: {want:?}");
            match want.get("name") {
                Json::Null => assert_eq!(n, &None, "name in {name}"),
                s => assert_eq!(n.as_deref(), Some(s.str()), "name in {name}"),
            }
            assert_eq!(raw, want.get("raw").str(), "raw in {name}");
            same_error(error, want.get("error").str());
        }
    }
}

#[test]
fn the_fixture_holds_many_kinds_of_reply() {
    let (_, cases) = cases();
    assert!(cases.len() >= 200, "{} cases", cases.len());
    let names: Vec<&str> = cases.iter().map(|c| c.get("name").str()).collect();
    for kind in [
        "plain:",
        "think_then_content:",
        "think_empty:",
        "call_one:",
        "call_several:",
        "bad_missing_required:",
        "think_cut:",
        "cut_in_function:",
        "utf8_four_byte:",
        "utf8_cut_four:half",
        "utf8_bad_surrogate:",
    ] {
        assert!(names.iter().any(|n| n.starts_with(kind)), "no case {kind}");
    }
    // At least one case has a token whose piece is empty while a character is not whole.
    assert!(
        cases
            .iter()
            .any(|c| c.get("events").items().iter().any(|e| e.items()[1].str().is_empty()))
    );
}

#[test]
fn each_token_has_the_phase_and_the_piece_of_mk() {
    let (f, cases) = cases();
    for case in &cases {
        let name = case.get("name").str();
        let (events, _) = run(&f, case);
        let want = case.get("events").items();
        assert_eq!(events.len(), want.len(), "token count of {name}");
        for (n, (got, want)) in events.iter().zip(want).enumerate() {
            // An event is [phase, piece, released].
            let want = want.items();
            assert_eq!(got.piece, want[1].str(), "piece {n} of {name}");
            assert_eq!(got.phase, phase_of(want[0].str()), "phase {n} of {name}");
            assert_eq!(got.released, segments(&want[2]), "released {n} of {name}");
        }
    }
}

#[test]
fn the_reply_is_the_reply_of_mk() {
    let (f, cases) = cases();
    for case in &cases {
        let name = case.get("name").str();
        let (_, reply) = run(&f, case);
        let want = case.get("result");
        assert_eq!(
            reply.reasoning,
            want.get("reasoning_content").str(),
            "reasoning of {name}"
        );
        assert_eq!(reply.content, want.get("content").str(), "content of {name}");
        let calls = want.get("tool_calls").items();
        assert_eq!(reply.tool_calls.len(), calls.len(), "call count of {name}");
        for (got, want) in reply.tool_calls.iter().zip(calls) {
            same_call(got, want, name);
        }
        assert_eq!(reply.tail, segments(case.get("tail")), "tail of {name}");
    }
}

#[test]
fn the_finish_follows_how_the_reply_ended() {
    let (f, cases) = cases();
    let mut tool_use = 0;
    let mut end_turn = 0;
    for case in &cases {
        let name = case.get("name").str();
        let (_, reply) = run(&f, case);
        let stopped = case.get("finish").str() == "stop";
        let want = match (stopped, reply.tool_calls.is_empty()) {
            (false, _) => Finish::MaxTokens,
            (true, true) => Finish::EndTurn,
            (true, false) => Finish::ToolUse,
        };
        assert_eq!(reply.finish, want, "finish of {name}");
        match reply.finish {
            Finish::ToolUse => tool_use += 1,
            Finish::EndTurn => end_turn += 1,
            _ => {}
        }
    }
    assert!(
        tool_use > 20 && end_turn > 20,
        "tool_use {tool_use}, end_turn {end_turn}"
    );
}

#[test]
fn a_full_context_is_its_own_finish() {
    let (f, cases) = cases();
    let case = &cases[0];
    let tools = toolset(&f, case.get("tools"));
    let mut parser = ReplyParser::new(tokenizer(), false, tools.as_ref()).unwrap();
    for &id in &case.get("ids").ids() {
        parser.push(id).unwrap();
    }
    assert_eq!(parser.finish(Ending::ContextFull).finish, Finish::ContextFull);
}

/// gogi's rule: the reasoning pieces, joined, equal `reasoning`, and the content pieces equal `content`, each up to
/// the white space that `str.strip()` removes. The text that the tracker releases (with the end of thinking left out)
/// keeps the rule in every case. The `piece` of a token does not: it holds the marker text (see the report).
#[test]
fn the_released_text_follows_the_piece_rule_of_gogi() {
    use nagaya_text_qwen38::pystr::strip;
    let (f, cases) = cases();
    for case in &cases {
        let name = case.get("name").str();
        let (events, reply) = run(&f, case);
        let all: Vec<&(Phase, String)> = events.iter().flat_map(|e| &e.released).chain(&reply.tail).collect();
        let join = |phase: Phase| -> String { all.iter().filter(|s| s.0 == phase).map(|s| s.1.as_str()).collect() };
        let reasoning = join(Phase::Reasoning);
        let reasoning = reasoning.strip_suffix("</think>").unwrap_or(&reasoning);
        assert_eq!(
            strip(reasoning, None, true, true),
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

#[test]
fn the_phase_now_is_the_phase_after_everything_pushed() {
    let t = tokenizer();
    let tools = Json::parse(r#"[{"type":"function","function":{"name":"f"}}]"#).unwrap();
    let mut p = ReplyParser::new(t, true, Some(&tools)).unwrap();
    assert_eq!(p.phase(), Phase::Reasoning);
    for id in t.encode("plan</thi").unwrap() {
        p.push(id).unwrap();
    }
    assert_eq!(p.phase(), Phase::Reasoning);
    for id in t.encode("nk>hello").unwrap() {
        p.push(id).unwrap();
    }
    assert_eq!(p.phase(), Phase::Content);
    p.push(t.token_id("<tool_call>").unwrap()).unwrap();
    assert_eq!(p.phase(), Phase::Call);
    assert_eq!(ReplyParser::new(t, false, None).unwrap().phase(), Phase::Content);
}

#[test]
fn an_id_outside_the_vocabulary_is_an_error() {
    let mut p = ReplyParser::new(tokenizer(), false, None).unwrap();
    let id = u32::try_from(tokenizer().len()).unwrap();
    let e = p.push(id).unwrap_err();
    assert!(e.to_string().contains(&id.to_string()), "{e}");
}

fn tools_error(tools: &str) -> String {
    let tools = Json::parse(tools).unwrap();
    match ReplyParser::new(tokenizer(), false, Some(&tools)) {
        Ok(_) => panic!("accepted {tools:?}"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn a_tool_list_that_mk_refuses_is_an_error() {
    // Each text is a ValueError in mk's `_specs` or `_types_of`.
    assert!(tools_error(r#"[1]"#).contains("needs a function with a name"));
    assert!(tools_error(r#"[{"function":{"name":3}}]"#).contains("needs a function with a name"));
    assert!(tools_error(r#"[{"function":{}}]"#).contains("needs a function with a name"));
    assert!(tools_error(r#"[{"name":"a","parameters":{"properties":[1]}}]"#).contains("must be an object"));
    assert!(tools_error(r#"[{"name":"a","parameters":{"properties":{"p":{"type":"text"}}}}]"#).contains("unsupported"));
    assert!(
        tools_error(r#"[{"name":"a","parameters":{"properties":{"p":{"anyOf":[{"type":"float"}]}}}}]"#)
            .contains("float")
    );
    // The nagaya is stricter than mk where mk would crash or iterate by accident.
    assert!(tools_error(r#"{"name":"a"}"#).contains("array"));
    assert!(tools_error(r#"[{"name":"a","parameters":5}]"#).contains("parameters"));
    assert!(tools_error(r#"[{"name":"a","parameters":{"required":"p"}}]"#).contains("required"));
    assert!(tools_error(r#"[{"name":"a","parameters":{"properties":{"p":{"type":[1]}}}}]"#).contains("type"));
}

#[test]
fn an_empty_tool_list_means_no_tools() {
    let t = tokenizer();
    let empty = Json::parse("[]").unwrap();
    let mut p = ReplyParser::new(t, false, Some(&empty)).unwrap();
    for id in t.encode("a <tool_call> b").unwrap() {
        p.push(id).unwrap();
    }
    let r = p.finish(Ending::StopToken);
    assert_eq!(r.content, "a <tool_call> b");
    assert!(r.tool_calls.is_empty());
}
