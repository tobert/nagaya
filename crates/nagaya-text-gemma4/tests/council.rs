//! The council family library of Gemma 4 (spec 11.3, 11.3a): contexts, boundaries, reads, replies and the template id.

mod common;

use common::model::{absent, family};
use gogi_proto::context::Context;
use gogi_proto::generation::Phase;
use gogi_proto::spec::{Spec, SpecRules, parse_spec};
use nagaya_council::family::{
    Advance, Asked, Boundary, Continues, DecisionInput, Family, FamilyError, ReadPlan, ReadStep, ReplyCall, ReplyStart,
};
use nagaya_council::template_id::hash16;
use nagaya_text_gemma4::CHAT_TEMPLATE;
use serde_json::{Value, json};

fn context(v: Value) -> Context {
    serde_json::from_value(v).expect("the test context is valid")
}

fn text(ids: &[u32]) -> String {
    family().tokenizer().decode(ids).expect("the tokens make text")
}

fn tpl(s: &str) -> Vec<u32> {
    family().tokenizer().encode(s).unwrap()
}

fn special(name: &str) -> u32 {
    family()
        .tokenizer()
        .token_id(name)
        .unwrap_or_else(|| panic!("no token {name}"))
}

fn client(e: FamilyError) -> String {
    match e {
        FamilyError::Client(invalid) => invalid.message,
        other => panic!("want a client error, got {other:?}"),
    }
}

const SHELL: &str = r#"{"name": "shell", "description": "Run a shell command.", "input_schema": {"type": "object",
    "properties": {"command": {"type": "string", "description": "the command"}, "timeout": {"type": "integer"}},
    "required": ["command"]}}"#;

fn shell() -> Value {
    serde_json::from_str(SHELL).unwrap()
}

// ---------------------------------------------------------------- contexts and boundaries

/// The cuts of mkviolet's `cuts_fall_where_a_prefix_renders_the_same`. The call keeps its thinking, and the text
/// answer loses its thinking before the later user message, so its boundary drops.
#[test]
fn a_boundary_drops_where_the_template_drops_the_thinking() {
    if absent() {
        return;
    }
    for thinking in [false, true] {
        let c = context(json!({"thinking": thinking, "tools": [shell()], "messages": [
            {"role": "system", "content": "You are a careful coding agent.", "snap": true},
            {"role": "user", "content": "List the files, then say <|turn>done.", "snap": true},
            {"role": "assistant", "content": "", "reasoning": "I should list them.", "snap": true,
             "tool_calls": [{"id": "c0", "name": "shell", "input": {"command": "ls -la", "timeout": 5}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "a.txt\nb.txt", "snap": true},
            {"role": "assistant", "content": "Two files: a.txt and b.txt.", "reasoning": "Both are listed.",
             "snap": true},
            {"role": "user", "content": "Thanks.", "snap": true}]}));
        let r = family().render(&c, true).unwrap();
        let messages: Vec<usize> = r.boundaries.iter().map(|b| b.message).collect();
        assert_eq!(messages, [0, 1, 2, 3, 5], "thinking {thinking}");
        for b in &r.boundaries[..4] {
            let mut part = c.clone();
            part.messages.truncate(b.message + 1);
            let alone = family().render(&part, false).unwrap();
            assert_eq!(
                alone.tokens,
                r.tokens[..b.tokens],
                "thinking {thinking}, message {}",
                b.message
            );
        }
        assert_eq!(r.boundaries.last().unwrap().tokens, r.end);
        assert!(r.end < r.tokens.len(), "the generation prompt follows the end");
        assert_eq!(r.control_text.len(), 1);
        assert_eq!(
            (r.control_text[0].place.as_str(), r.control_text[0].token.as_str()),
            ("message:1", "<|turn>")
        );
    }
}

#[test]
fn the_end_of_a_context_is_a_boundary_when_no_message_is_marked() {
    if absent() {
        return;
    }
    let c = context(json!({"thinking": true, "messages": [{"role": "user", "content": "hi"}]}));
    let r = family().render(&c, false).unwrap();
    assert_eq!(
        r.boundaries,
        [Boundary {
            message: 0,
            tokens: r.tokens.len()
        }]
    );
    assert_eq!(
        text(&r.tokens),
        "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhi<turn|>\n"
    );
}

#[test]
fn the_generation_prompt_opens_the_thinking_or_closes_it() {
    if absent() {
        return;
    }
    for (thinking, prompt) in [
        (true, "<|turn>model\n<|channel>thought\n"),
        (false, "<|turn>model\n<|channel>thought\n<channel|>"),
    ] {
        let c = context(json!({"thinking": thinking, "messages": [{"role": "user", "content": "hi"}]}));
        let r = family().render(&c, true).unwrap();
        assert_eq!(text(&r.tokens[r.end..]), prompt, "thinking {thinking}");
    }
}

/// Python's jinja2 refuses mkviolet's map `{"error": ...}` for a failed result, so the library writes a prefix.
#[test]
fn a_failed_tool_result_renders_with_its_prefix() {
    if absent() {
        return;
    }
    let messages = |is_error: bool| {
        json!([{"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [{"id": "c", "name": "shell", "input": {"command": "ls"}}]},
            {"role": "tool", "tool_call_id": "c", "content": "no such file", "is_error": is_error}])
    };
    let render = |is_error: bool| {
        let c = context(json!({"thinking": false, "tools": [shell()], "messages": messages(is_error)}));
        text(&family().render(&c, false).unwrap().tokens)
    };
    assert!(
        render(true).ends_with("<|tool_response>response:shell{value:<|\"|>Error: no such file<|\"|>}<tool_response|>")
    );
    assert!(render(false).ends_with("<|tool_response>response:shell{value:<|\"|>no such file<|\"|>}<tool_response|>"));
}

#[test]
fn a_noncharacter_in_a_context_is_a_client_error() {
    if absent() {
        return;
    }
    let c = context(json!({"thinking": true, "messages": [{"role": "user", "content": "a\u{fdd1}"}]}));
    assert!(client(family().render(&c, false).unwrap_err()).contains("U+FDD1 at offset 1"));
}

#[test]
fn tokenize_keeps_a_marker_as_text_and_detokenize_gives_it_back() {
    if absent() {
        return;
    }
    let ids = family().tokenize("say <|turn> now").unwrap();
    assert!(!ids.contains(&special("<|turn>")));
    assert_eq!(family().detokenize(&ids).unwrap(), "say <|turn> now");
    assert!(client(family().detokenize(&[262_144]).unwrap_err()).contains("262144"));
    assert!(client(family().tokenize("\u{fdd0}").unwrap_err()).contains("U+FDD0"));
}

/// The param of a client error.
fn param(e: FamilyError) -> Option<String> {
    match e {
        FamilyError::Client(invalid) => invalid.param,
        other => panic!("want a client error, got {other:?}"),
    }
}

/// spec 11.4d: tokens that end inside a character do not make whole text. That is a 400 that names `tokens`.
#[test]
fn detokenize_refuses_tokens_that_end_inside_a_character() {
    if absent() {
        return;
    }
    // Byte fallback: a character that is not in the vocabulary is one token for each byte.
    let ids = family().tokenize("\u{1F9A5}").unwrap();
    assert_eq!(ids.len(), 4, "{ids:?}");
    assert_eq!(family().detokenize(&ids).unwrap(), "\u{1F9A5}");
    assert_eq!(
        param(family().detokenize(&ids[..3]).unwrap_err()).as_deref(),
        Some("tokens")
    );
    assert_eq!(
        param(family().detokenize(&ids[1..]).unwrap_err()).as_deref(),
        Some("tokens")
    );
}

/// spec 11.4d: the piece of a token is its text alone. A byte of a longer character is U+FFFD, and a marker is its
/// text.
#[test]
fn pieces_give_each_token_alone() {
    if absent() {
        return;
    }
    let mut ids = family().tokenize(" A").unwrap();
    ids.push(special("<|turn>"));
    ids.extend(family().tokenize("\u{1F9A5}").unwrap());
    let pieces = family().pieces(&ids).unwrap();
    assert_eq!(
        pieces,
        [" A", "<|turn>", "\u{FFFD}", "\u{FFFD}", "\u{FFFD}", "\u{FFFD}"]
    );
    assert_eq!(
        param(family().pieces(&[262_144]).unwrap_err()).as_deref(),
        Some("tokens")
    );
}

/// gogi "Raw reads": in a suffix, the text of a marker is the marker token. Client text keeps it as text.
#[test]
fn tokenize_special_gives_the_marker_token() {
    if absent() {
        return;
    }
    let ids = family().tokenize_special("<|turn>model\n").unwrap();
    assert_eq!(ids[0], special("<|turn>"));
    assert_eq!(ids, tpl("<|turn>model\n"));
}

// ---------------------------------------------------------------- the raw read

#[test]
fn a_raw_read_takes_letters_as_content_and_suffixes_with_their_markers() {
    if absent() {
        return;
    }
    let own = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let raw = family()
        .raw_read(&own(&["A", " B"]), &own(&["plain", "<turn|>\n<|turn>model\n"]))
        .unwrap();
    assert_eq!(raw.letters, [tpl("A")[0], tpl(" B")[0]]);
    assert_eq!(raw.suffixes[1], tpl("<turn|>\n<|turn>model\n"));
    assert_eq!(raw.suffixes[1][0], special("<turn|>"));
    assert_eq!(raw.control_text.len(), 2);
    assert!(client(family().raw_read(&own(&["AB C"]), &[]).unwrap_err()).contains("exactly one token"));
    assert!(client(family().raw_read(&own(&["A", "A"]), &[]).unwrap_err()).contains("distinct"));
    let e = family().raw_read(&[], &own(&["\u{fdd0}"])).unwrap_err();
    assert!(
        matches!(&e, FamilyError::Client(i) if i.param.as_deref() == Some("suffixes")),
        "{e:?}"
    );
}

#[test]
fn the_26_letters_are_one_token_each() {
    if absent() {
        return;
    }
    assert_eq!(family().letter_count(), 26);
}

// ---------------------------------------------------------------- decisions

fn spec() -> Spec {
    parse_spec(
        &json!({"name": "g", "instructions": "Judge the command.", "input_label": "command", "questions": [
            {"id": "verdict", "type": "choice", "instructions": "What to do. Pick one.", "criteria": [
                {"option": "allow", "means": "run it"}, {"option": "ask", "means": "ask first"}]},
            {"id": "undo", "type": "score", "instructions": "How hard to undo", "criteria": ["easy", "hard"]},
            {"id": "safe", "type": "noul", "instructions": "It is safe.", "criteria": "No writes."}]}),
        &SpecRules::default(),
    )
    .expect("the test spec is valid")
}

fn asked(question: usize, options: &[usize]) -> Asked {
    Asked {
        question,
        options: options.to_vec(),
    }
}

/// Run a plan to its end. The first letter is the likeliest, unless `likeliest` names another for a step.
fn drive(plan: &mut dyn ReadPlan, likeliest: &[usize]) -> (Vec<ReadStep>, String) {
    let mut steps = Vec::new();
    let mut last: Option<Vec<f64>> = None;
    loop {
        match plan.advance(last.as_deref()).expect("the plan runs") {
            Advance::Done(done) => return (steps, done.rendered_hash),
            Advance::Step(step) => {
                let best = likeliest.get(steps.len()).copied().unwrap_or(0);
                last = Some(
                    (0..step.letters.len())
                        .map(|i| if i == best { -0.1 } else { -2.0 - i as f64 })
                        .collect(),
                );
                steps.push(step);
            }
        }
    }
}

const SPEC_LAYER: &str = "<|turn>user\nJudge the command.\n\nAnswer each question below about the command in the next \
message. Answer question N with the letter of one option, as \"Answer to question N: X\".\n\nQuestion 1. What to do. \
Pick one.\nA. allow: run it\nB. ask: ask first\n\nQuestion 2. How hard to undo The options run from lowest to \
highest.\nA. easy\nB. hard\n\nQuestion 3. Is this true? It is safe.\nNo writes.\nA. yes\nB. no\n<turn|>\n";

const CASE_LAYER: &str =
    "<|turn>user\ncommand:\nrm -rf /tmp/x<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Answer to question";

/// A plain read with no context has the council layout: `<bos>`, the spec turn, the case turn and the questions.
#[test]
fn a_plain_read_has_the_council_layout() {
    if absent() {
        return;
    }
    let spec = spec();
    let asks = [asked(0, &[0, 1]), asked(2, &[0, 1])];
    let input = DecisionInput {
        context: None,
        spec: &spec,
        case: "rm -rf /tmp/x",
        asked: &asks,
    };
    let mut plan = family().decision_read(&input).unwrap();
    let prefix = plan.prefix().to_vec();
    assert_eq!(text(&prefix), format!("<bos>{SPEC_LAYER}{CASE_LAYER}"));
    // Template text with markers on, client text with them off, piece by piece, as the first Gemma 4 engine tokenized.
    assert_eq!(prefix[..2], [special("<bos>"), special("<|turn>")]);
    let (steps, sha) = drive(plan.as_mut(), &[]);
    assert_eq!(steps.len(), 2);
    assert_eq!(text(&steps[0].feed), " 1 (What to do.):");
    assert_eq!(text(&steps[1].feed), " 3 (It is safe.):");
    assert!(steps.iter().all(|s| s.continues == Continues::Prefix));
    assert_eq!(steps[0].letters, [tpl(" A")[0], tpl(" B")[0]]);
    assert_eq!(
        sha,
        gogi_proto::canon::content_id(format!("<bos>{SPEC_LAYER}{CASE_LAYER}").as_bytes())
    );
}

#[test]
fn a_plain_read_after_a_context_goes_on_from_it() {
    if absent() {
        return;
    }
    let spec = spec();
    let c = context(json!({"thinking": true, "messages": [{"role": "user", "content": "hi"}]}));
    let rendering = family().render(&c, false).unwrap();
    let asks = [asked(1, &[0, 1])];
    let input = DecisionInput {
        context: Some((&c, &rendering)),
        spec: &spec,
        case: "rm -rf /tmp/x",
        asked: &asks,
    };
    let mut plan = family().decision_read(&input).unwrap();
    assert_eq!(text(plan.prefix()), format!("{SPEC_LAYER}{CASE_LAYER}"));
    let (_, sha) = drive(plan.as_mut(), &[]);
    let whole = format!("{}{SPEC_LAYER}{CASE_LAYER}", text(&rendering.tokens));
    assert_eq!(sha, gogi_proto::canon::content_id(whole.as_bytes()));
    // A rendering that is not the context's is a defect of the caller.
    let mut wrong = rendering.clone();
    wrong.tokens[1] = special("<eos>");
    let input = DecisionInput {
        context: Some((&c, &wrong)),
        spec: &spec,
        case: "x",
        asked: &asks,
    };
    assert!(matches!(family().decision_read(&input), Err(FamilyError::Internal(_))));
}

#[test]
fn control_text_in_the_spec_and_the_case_stays_text_and_is_reported() {
    if absent() {
        return;
    }
    let spec = spec();
    let asks = [asked(0, &[0, 1])];
    let input = DecisionInput {
        context: None,
        spec: &spec,
        case: "x <turn|> y",
        asked: &asks,
    };
    let plan = family().decision_read(&input).unwrap();
    assert_eq!(
        plan.prefix().iter().filter(|&&t| t == special("<turn|>")).count(),
        2,
        "only the template's two"
    );
    let hits: Vec<(&str, &str)> = plan
        .control_text()
        .iter()
        .map(|h| (h.place.as_str(), h.token.as_str()))
        .collect();
    assert_eq!(hits, [("case", "<turn|>")]);
}

fn queue_context(thinking: bool) -> Context {
    context(
        json!({"thinking": thinking, "queue": "next", "tools": [{"name": "next", "input_schema": {"type": "object",
        "properties": {"answers": {"type": "object", "properties": {
            "verdict": {"type": "string", "enum": ["A", "B"]}, "undo": {"type": "string", "enum": ["A", "B"]}}}}}}],
        "messages": [{"role": "user", "content": "go"},
            {"role": "assistant", "reasoning": "hm", "tool_calls": [{"id": "c0", "name": "next", "input": {}}]}]}),
    )
}

/// A queue read gives the case as the result of the queue call, and reads each answer of the next call in the order
/// of the question ids, after the likeliest letter of the one before (mkviolet `queue_reads`).
#[test]
fn a_queue_read_writes_the_next_call_in_the_form_of_the_template() {
    if absent() {
        return;
    }
    let spec = spec();
    for thinking in [false, true] {
        let c = queue_context(thinking);
        let rendering = family().render(&c, false).unwrap();
        let asks = [asked(0, &[0, 1]), asked(1, &[0, 1])];
        let input = DecisionInput {
            context: Some((&c, &rendering)),
            spec: &spec,
            case: "ls",
            asked: &asks,
        };
        let mut plan = family().decision_read(&input).unwrap();
        let open = if thinking { "<|channel>thought\n<channel|>" } else { "" };
        // The template ends a call with no result with `<|tool_response>`, so the context holds that marker.
        assert!(text(&rendering.tokens).ends_with("<tool_call|><|tool_response>"));
        assert_eq!(
            text(plan.prefix()),
            format!("response:next{{value:<|\"|>ls<|\"|>}}<tool_response|>{open}<|tool_call>call:next{{answers:{{"),
            "thinking {thinking}"
        );
        let (steps, sha) = drive(plan.as_mut(), &[1, 0]);
        // `undo` sorts before `verdict`, so question 1 of `asked` comes first.
        assert_eq!(steps.iter().map(|s| s.question).collect::<Vec<_>>(), [1, 0]);
        assert_eq!(text(&steps[0].feed), "undo:<|\"|>");
        assert_eq!(text(&steps[1].feed), "B<|\"|>,verdict:<|\"|>");
        assert_eq!(steps[1].feed[0], tpl("B")[0], "the bare letter that was read");
        assert!(steps.iter().all(|s| s.continues == Continues::PreviousStep));
        assert_eq!(steps[0].letters, [tpl("A")[0], tpl("B")[0]]);
        let whole = format!(
            "{}{}{}{}A<|\"|>",
            text(&rendering.tokens),
            text(plan.prefix()),
            text(&steps[0].feed),
            text(&steps[1].feed)
        );
        assert_eq!(
            sha,
            gogi_proto::canon::content_id(whole.as_bytes()),
            "thinking {thinking}"
        );
    }
}

#[test]
fn a_spec_past_the_letters_and_a_question_with_no_option_are_client_errors() {
    if absent() {
        return;
    }
    let spec = spec();
    let none = [asked(0, &[])];
    let input = DecisionInput {
        context: None,
        spec: &spec,
        case: "x",
        asked: &none,
    };
    assert!(client(family().decision_read(&input).err().unwrap()).contains("no option"));
}

// ---------------------------------------------------------------- replies

fn push_all(thinking: bool, ids: &[u32]) -> (Vec<(Phase, String)>, nagaya_council::family::ParsedReply) {
    let mut parser = family().reply_parser(ReplyStart::prompt(thinking), &[]).unwrap();
    let steps = ids
        .iter()
        .map(|&t| parser.push(t).unwrap())
        .map(|s| (s.phase, s.piece))
        .collect();
    (steps, parser.finish(false))
}

fn joined(steps: &[(Phase, String)], phase: Phase) -> String {
    steps
        .iter()
        .filter(|(p, _)| *p == phase)
        .map(|(_, s)| s.as_str())
        .collect()
}

/// A reply with thinking, a text, a call and the stop token: each part, and pieces that join to each part.
#[test]
fn a_reply_splits_into_reasoning_content_and_a_call() {
    if absent() {
        return;
    }
    let mut ids = tpl(
        "I should list them.\n<channel|>Let me look.<|tool_call>call:shell{command:<|\"|>ls -la<|\"|>,n:3}<tool_call|>",
    );
    ids.push(special("<turn|>"));
    let (steps, reply) = push_all(true, &ids);
    assert_eq!(reply.reasoning, "I should list them.\n");
    assert_eq!(reply.content, "Let me look.");
    let mut input = serde_json::Map::new();
    input.insert("command".into(), json!("ls -la"));
    input.insert("n".into(), json!(3));
    assert_eq!(
        reply.calls,
        [ReplyCall::Parsed {
            name: "shell".into(),
            input
        }]
    );
    assert!(reply.ended);
    assert_eq!(joined(&steps, Phase::Reasoning), reply.reasoning);
    assert_eq!(joined(&steps, Phase::Content), reply.content);
}

/// After a tool result with thinking off, Gemma 4 opens a thought channel itself. Its text is reasoning without the
/// channel's name, and its markers show nothing.
#[test]
fn a_thought_channel_that_the_model_opens_is_reasoning() {
    if absent() {
        return;
    }
    let ids = tpl("<|channel>thought\nPlan it.<channel|>Here.");
    let (steps, reply) = push_all(false, &ids);
    assert_eq!(reply.reasoning, "\nPlan it.");
    assert_eq!(reply.content, "Here.");
    assert_eq!(joined(&steps, Phase::Reasoning), reply.reasoning);
    assert!(!reply.ended);
}

#[test]
fn a_call_that_does_not_close_or_parse_is_malformed() {
    if absent() {
        return;
    }
    let (_, reply) = push_all(false, &tpl("<|tool_call>call:shell{command:<|\"|>ls"));
    assert!(matches!(&reply.calls[..], [ReplyCall::Malformed { text, error }]
        if text == "call:shell{command:<|\"|>ls" && error.contains("did not close")));
    let (_, reply) = push_all(false, &tpl("<|tool_call>shell{}<tool_call|>"));
    assert!(matches!(&reply.calls[..], [ReplyCall::Malformed { error, .. }] if error.contains("call:")));
}

#[test]
fn every_stop_token_ends_a_reply_and_a_later_token_is_a_defect() {
    if absent() {
        return;
    }
    for stop in ["<turn|>", "<eos>", "<|tool_response>", "<|turn>"] {
        let mut parser = family().reply_parser(ReplyStart::prompt(false), &[]).unwrap();
        parser.push(tpl("Hi")[0]).unwrap();
        assert_eq!(parser.push(special(stop)).unwrap().piece, "");
        assert!(parser.ended(), "{stop}");
        assert!(matches!(parser.push(tpl("x")[0]), Err(FamilyError::Internal(_))));
    }
    let mut parser = family().reply_parser(ReplyStart::prompt(false), &[]).unwrap();
    assert!(matches!(parser.push(262_144), Err(FamilyError::Internal(_))));
}

#[test]
fn a_character_split_across_tokens_comes_out_whole() {
    if absent() {
        return;
    }
    // Byte fallback: a character that is not in the vocabulary is several byte tokens.
    let ids = tpl("\u{1F9A5}");
    assert!(ids.len() > 1, "{ids:?}");
    let (steps, reply) = push_all(false, &ids);
    assert_eq!(reply.content, "\u{1F9A5}");
    assert!(steps[..steps.len() - 1].iter().all(|(_, s)| s.is_empty()));
}

// ---------------------------------------------------------------- the template id

#[test]
fn the_template_id_has_the_council_hash_and_the_hash_of_the_template() {
    if absent() {
        return;
    }
    let id = family().template_id();
    let (council, chat) = id.split_once("+gemma4-chat:").expect("the family part");
    assert_eq!(chat, hash16(CHAT_TEMPLATE.as_bytes()));
    assert!(
        council.starts_with("nagaya-council:") && council.len() == "nagaya-council:".len() + 16,
        "{id}"
    );
    let again = nagaya_text_gemma4::Gemma4Family::from_dir(&common::model::model_dir()).unwrap();
    assert_eq!(again.template_id(), id, "the id is the same for each build");
    assert_eq!(family().vocab_size(), 262_144);
}

/// gogi "Continuations": the phase of the next token. A reply that ends with `max_tokens` reports it as `done.phase`.
#[test]
fn the_parser_tells_the_phase_of_the_next_token() {
    if absent() {
        return;
    }
    let mut parser = family().reply_parser(ReplyStart::prompt(true), &[]).unwrap();
    assert_eq!(parser.phase(), Phase::Reasoning);
    parser.push(tpl("plan")[0]).unwrap();
    parser.push(special("<channel|>")).unwrap();
    assert_eq!(parser.phase(), Phase::Content);
    parser.push(special("<|tool_call>")).unwrap();
    assert_eq!(parser.phase(), Phase::Call);
    let parser = family().reply_parser(ReplyStart::prompt(false), &[]).unwrap();
    assert_eq!(parser.phase(), Phase::Content);
}
