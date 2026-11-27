//! The toy family gives the tokens and the numbers of gogi-toy and gogi-fake (spec 11.3b).

mod support;

use gogi_fake::{Config, Server};
use gogi_proto::generation::Phase;
use gogi_toy::{self as toy, State, log_softmax};
use nagaya_council::family::{Asked, Boundary, ControlHit, DecisionInput, Family, FamilyError, ReplyCall, ReplyStart};
use nagaya_family_toy::ToyFamily;
use serde_json::{Value, json};
use support::{SEED, context, drive, post, spec};

fn family() -> Box<dyn Family> {
    Box::new(ToyFamily::new())
}

fn contexts() -> Vec<Value> {
    let tool = json!({"name": "sh", "input_schema": {"type": "object"}});
    vec![
        json!({"thinking": true, "messages": [{"role": "user", "content": "hi"}]}),
        json!({"thinking": false, "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u", "snap": true}]}),
        json!({"thinking": true, "tools": [tool], "messages": [
            {"role": "user", "content": "u", "snap": true},
            {"role": "assistant", "content": "a", "reasoning": "r", "snap": true, "tool_calls": [{"id": "c", "name": "sh", "input": {"k": 1}}]},
            {"role": "tool", "tool_call_id": "c", "content": "out", "is_error": true, "snap": true},
            {"role": "assistant", "content": "done", "reasoning": "r2", "snap": true}]}),
        json!({"thinking": true, "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "reasoning": "dropped", "snap": true},
            {"role": "user", "content": "v", "snap": true}]}),
    ]
}

#[test]
fn a_context_renders_to_the_tokens_and_boundaries_of_gogi_toy() {
    let f = family();
    for (i, v) in contexts().into_iter().enumerate() {
        let c = context(v);
        for prompt in [false, true] {
            let got = f.render(&c, prompt).unwrap();
            let want = toy::render(&c.tools, c.thinking, &c.messages, prompt);
            assert_eq!(got.tokens, want.tokens, "context {i}, prompt {prompt}");
            assert_eq!(got.end, want.end, "context {i}, prompt {prompt}");
            let cuts: Vec<Boundary> = want
                .cuts
                .iter()
                .map(|c| Boundary {
                    message: c.message,
                    tokens: c.tokens,
                })
                .collect();
            assert_eq!(got.boundaries, cuts, "context {i}");
            assert_eq!(got.boundaries.last().unwrap().tokens, got.end);
        }
        let with = f.render(&c, true).unwrap();
        let without = f.render(&c, false).unwrap();
        assert!(with.tokens.len() > with.end, "the generation prompt adds tokens");
        assert_eq!(without.tokens.len(), without.end);
    }
}

/// gogi "Continuations": an open last message renders as gogi-toy renders it, without its end and in place of the
/// generation prompt. So nagaya and gogi-fake feed the same prompt.
#[test]
fn an_open_message_renders_to_the_tokens_of_gogi_toy_without_its_end() {
    let f = family();
    let u = json!({"role": "user", "content": "u", "snap": true});
    for (thinking, open) in [
        (
            true,
            json!({"role": "assistant", "reasoning": "plan", "continue": true}),
        ),
        (
            true,
            json!({"role": "assistant", "reasoning": "plan", "content": "", "continue": true}),
        ),
        (
            true,
            json!({"role": "assistant", "reasoning": "plan", "content": "fn ", "continue": true, "snap": true}),
        ),
        (false, json!({"role": "assistant", "content": "fn ", "continue": true})),
        (false, json!({"role": "assistant", "content": "", "continue": true})),
    ] {
        let c = context(json!({"thinking": thinking, "messages": [u, open]}));
        let want = toy::render_tokens(&c.tools, thinking, &c.messages, true);
        let user = f
            .render(&context(json!({"thinking": thinking, "messages": [u]})), false)
            .unwrap()
            .end;
        for prompt in [false, true] {
            let got = f.render(&c, prompt).unwrap();
            assert_eq!(got.tokens, want, "{open}, prompt {prompt}");
            assert_eq!(
                got.end,
                got.tokens.len(),
                "{open}: an open message has no generation prompt after it"
            );
            assert_eq!(
                got.boundaries,
                [
                    Boundary {
                        message: 0,
                        tokens: user
                    },
                    Boundary {
                        message: 1,
                        tokens: got.end
                    }
                ],
                "{open}"
            );
        }
        let mut ended = want.clone();
        ended.extend(toy::end_open(thinking, &c.messages[1]));
        let mut closed = c.clone();
        closed.messages[1].open = false;
        assert_eq!(
            f.render(&closed, false).unwrap().tokens,
            ended,
            "{open}: the open rendering begins the closed one"
        );
    }
}

#[test]
fn a_boundary_goes_when_a_later_user_message_changes_the_rendering() {
    let f = family();
    let r = f.render(&context(contexts()[3].clone()), false).unwrap();
    assert!(r.boundaries.iter().all(|b| b.message != 1), "{:?}", r.boundaries);
    assert!(r.boundaries.iter().any(|b| b.message == 2));
}

#[test]
fn text_that_spells_a_marker_in_a_message_is_a_control_hit_and_stays_content() {
    let f = family();
    let c = context(
        json!({"thinking": true, "messages": [{"role": "user", "content": "a <end> b"}, {"role": "assistant", "content": "<asst>", "reasoning": "<think>"}]}),
    );
    let r = f.render(&c, false).unwrap();
    let hit = |place: &str, token: &str| ControlHit {
        place: place.into(),
        token: token.into(),
    };
    assert_eq!(
        r.control_text,
        [
            hit("message:0", "<end>"),
            hit("message:1", "<asst>"),
            hit("message:1", "<think>")
        ]
    );
    assert_eq!(
        r.tokens.iter().filter(|t| **t == toy::END).count(),
        2,
        "only the template's <end> tokens"
    );
}

#[test]
fn tokenize_special_makes_one_token_of_the_text_of_a_marker() {
    let f = family();
    assert_eq!(
        f.tokenize_special("a<end>b").unwrap(),
        [u32::from(b'a'), toy::END, u32::from(b'b')]
    );
    assert_eq!(f.tokenize_special("a<end>b").unwrap(), toy::tokenize_suffix("a<end>b"));
    // Without `special` the marker text is bytes.
    assert_eq!(f.tokenize("<end>").unwrap().len(), 5);
    assert_eq!(f.tokenize_special("").unwrap(), Vec::<u32>::new());
}

#[test]
fn pieces_are_the_text_of_each_token() {
    let f = family();
    let tokens = f.tokenize_special("hi<end>").unwrap();
    assert_eq!(f.pieces(&tokens).unwrap(), ["h", "i", "<end>"]);
    assert_eq!(f.pieces(&[]).unwrap(), Vec::<String>::new());
}

#[test]
fn a_byte_that_is_not_whole_text_is_a_replacement_character_in_its_piece() {
    let f = family();
    // The first byte of a two-byte character is not text alone.
    let bytes = "é".as_bytes().iter().map(|&b| u32::from(b)).collect::<Vec<_>>();
    assert_eq!(f.pieces(&bytes).unwrap(), ["\u{fffd}", "\u{fffd}"]);
    assert!(f.detokenize(&bytes[..1]).is_err());
    assert_eq!(f.detokenize(&bytes).unwrap(), "é");
}

#[test]
fn pieces_of_a_token_outside_the_vocabulary_is_a_client_error_that_names_tokens() {
    let f = family();
    let FamilyError::Client(invalid) = f.pieces(&[0, f.vocab_size() as u32]).unwrap_err() else {
        panic!("want a client error")
    };
    assert_eq!(invalid.param.as_deref(), Some("tokens"));
}

#[test]
fn a_context_with_no_message_is_a_client_error() {
    let err = family()
        .render(&context(json!({"thinking": true, "messages": []})), false)
        .unwrap_err();
    assert!(err.is_client(), "{err:?}");
}

fn queue_tool(undo: &[&str]) -> Value {
    json!({"name": "next", "input_schema": {"type": "object", "properties": {"answers": {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": ["A", "B", "C"]}, "undo": {"type": "string", "enum": undo}}}}}})
}

fn queue_context(undo: &[&str]) -> Value {
    json!({"thinking": true, "tools": [queue_tool(undo)], "queue": "next", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "go", "snap": true},
        {"role": "assistant", "reasoning": "hm", "tool_calls": [{"id": "c0", "name": "next", "input": {}}]}]})
}

fn spec_json() -> Value {
    json!({"name": "g", "instructions": "i <end>", "input_label": "x", "questions": [
        {"id": "verdict", "type": "choice", "instructions": "what", "criteria": [{"option": "allow", "means": "a"}, {"option": "ask", "means": "b"}, {"option": "no", "means": "c"}]},
        {"id": "undo", "type": "score", "instructions": "how", "criteria": ["easy", "hard"]}]})
}

/// What gogi-fake answers for a decision: for each question id, its letter log probabilities, and the hash.
fn fake_answers(
    s: &Server,
    ask: Option<Value>,
    options: Option<Value>,
    ctx: Option<Value>,
    case: &str,
) -> (Value, String) {
    let (_, held) = post(s, "/council/v2/specs", spec_json());
    let mut req = json!({"model": "toy", "spec_id": held["spec_id"], "case": case});
    if let Some(a) = ask {
        req["ask"] = a;
    }
    if let Some(o) = options {
        req["options"] = o;
    }
    if let Some(c) = ctx {
        req["contexts"] = json!([c]);
    }
    let (code, d) = post(s, "/council/v2/decisions", req);
    assert_eq!(code, 200, "{d}");
    (
        d["reads"][0]["answers"].clone(),
        d["reads"][0]["rendered_hash"].as_str().unwrap().to_string(),
    )
}

/// The numbers of a plan, as gogi-fake writes them: for each question id, a map of answer key to log probability.
fn run_plan(spec_value: &Value, ctx: Option<&Value>, case: &str, asked: &[Asked]) -> (Value, String) {
    let f = family();
    let spec = spec(spec_value);
    let c = ctx.map(|v| context(v.clone()));
    let rendering = c.as_ref().map(|c| f.render(c, false).unwrap());
    let input = DecisionInput {
        context: c.as_ref().zip(rendering.as_ref()),
        spec: &spec,
        case,
        asked,
    };
    let mut plan = f.decision_read(&input).unwrap();
    let tokens = rendering.as_ref().map_or(Vec::new(), |r| r.tokens[..r.end].to_vec());
    let done = drive(plan.as_mut(), &tokens);
    let mut out = serde_json::Map::new();
    for (slot, lps) in &done.steps {
        let a = &asked[*slot];
        let q = &spec.questions[a.question];
        let keys = q.answer_keys();
        let map: serde_json::Map<String, Value> = a
            .options
            .iter()
            .zip(lps)
            .map(|(&j, lp)| (keys[j].clone(), json!(lp)))
            .collect();
        out.insert(q.id.clone(), Value::Object(map));
    }
    (Value::Object(out), done.rendered_hash)
}

fn logprobs_of(answers: &Value) -> Value {
    Value::Object(
        answers
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v["logprobs"].clone()))
            .collect(),
    )
}

fn both() -> Vec<Asked> {
    vec![
        Asked {
            question: 0,
            options: vec![0, 1, 2],
        },
        Asked {
            question: 1,
            options: vec![0, 1],
        },
    ]
}

#[test]
fn a_decision_with_no_context_gives_the_numbers_and_the_hash_of_gogi_fake() {
    let s = Server::start(Config::default()).unwrap();
    let (want, hash) = fake_answers(&s, None, None, None, "rm -rf build");
    let (got, got_hash) = run_plan(&spec_json(), None, "rm -rf build", &both());
    assert_eq!(got, logprobs_of(&want));
    assert_eq!(got_hash, hash);
}

#[test]
fn a_decision_over_a_context_with_narrowed_options_gives_the_numbers_of_gogi_fake() {
    let s = Server::start(Config::default()).unwrap();
    let ctx = contexts()[1].clone();
    let (want, hash) = fake_answers(
        &s,
        Some(json!(["verdict"])),
        Some(json!({"verdict": ["allow", "no"]})),
        Some(ctx.clone()),
        "x",
    );
    let asked = [Asked {
        question: 0,
        options: vec![0, 2],
    }];
    let (got, got_hash) = run_plan(&spec_json(), Some(&ctx), "x", &asked);
    assert_eq!(got, logprobs_of(&want));
    assert_eq!(got_hash, hash);
}

#[test]
fn a_plain_read_reads_each_question_after_the_same_prefix() {
    let f = family();
    let spec = spec(&spec_json());
    let asked = both();
    let mut plan = f
        .decision_read(&DecisionInput {
            context: None,
            spec: &spec,
            case: "c",
            asked: &asked,
        })
        .unwrap();
    let text = |t: &[u32]| String::from_utf8(t.iter().map(|&x| x as u8).collect()).unwrap();
    assert_eq!(plan.prefix()[0], toy::BOS, "no context: the prefix starts with <bos>");
    assert_eq!(plan.prefix()[1], toy::USER);
    assert_eq!(&plan.prefix()[plan.prefix().len() - 2..], [toy::END, toy::ASST]);
    let nagaya_council::family::Advance::Step(first) = plan.advance(None).unwrap() else {
        panic!()
    };
    assert_eq!(
        (first.question, first.continues, text(&first.feed).as_str()),
        (0, nagaya_council::family::Continues::Prefix, "verdict what:")
    );
    assert_eq!(first.letters, [65, 66, 67]);
    let nagaya_council::family::Advance::Step(second) = plan.advance(Some(&[-1.0, -2.0, -3.0])).unwrap() else {
        panic!()
    };
    assert_eq!((second.question, text(&second.feed).as_str()), (1, "undo how:"));
    assert_eq!(second.letters, [65, 66]);
    assert!(matches!(
        plan.advance(Some(&[-1.0, -2.0])).unwrap(),
        nagaya_council::family::Advance::Done(_)
    ));
}

#[test]
fn the_control_text_of_a_spec_and_a_case_is_reported_by_place() {
    let f = family();
    let spec = spec(&spec_json());
    let asked = both();
    let plan = f
        .decision_read(&DecisionInput {
            context: None,
            spec: &spec,
            case: "a <user> b",
            asked: &asked,
        })
        .unwrap();
    let hit = |place: &str, token: &str| ControlHit {
        place: place.into(),
        token: token.into(),
    };
    assert_eq!(plan.control_text(), [hit("case", "<user>"), hit("spec", "<end>")]);
}

#[test]
fn a_queue_read_follows_the_most_likely_letter_and_matches_gogi_fake() {
    let s = Server::start(Config::default()).unwrap();
    let ctx = queue_context(&["A", "B"]);
    let (want, hash) = fake_answers(&s, None, None, Some(ctx.clone()), "git push");
    let (got, got_hash) = run_plan(&spec_json(), Some(&ctx), "git push", &both());
    assert_eq!(got, logprobs_of(&want));
    assert_eq!(got_hash, hash);
}

#[test]
fn a_queue_read_asked_in_another_order_still_reads_by_question_id() {
    let s = Server::start(Config::default()).unwrap();
    let ctx = queue_context(&["A", "B"]);
    let (want, hash) = fake_answers(&s, Some(json!(["verdict", "undo"])), None, Some(ctx.clone()), "c");
    let (got, got_hash) = run_plan(&spec_json(), Some(&ctx), "c", &both());
    assert_eq!((got, got_hash.clone()), (logprobs_of(&want), hash));
    // Steps run in the order of the question ids: undo before verdict.
    let f = family();
    let sp = spec(&spec_json());
    let c = context(ctx);
    let r = f.render(&c, false).unwrap();
    let asked = both();
    let mut plan = f
        .decision_read(&DecisionInput {
            context: Some((&c, &r)),
            spec: &sp,
            case: "c",
            asked: &asked,
        })
        .unwrap();
    let nagaya_council::family::Advance::Step(first) = plan.advance(None).unwrap() else {
        panic!()
    };
    assert_eq!(first.question, 1);
    assert_eq!(first.continues, nagaya_council::family::Continues::PreviousStep);
    assert_eq!(first.feed, "undo:\"".bytes().map(u32::from).collect::<Vec<_>>());
    let nagaya_council::family::Advance::Step(second) = plan.advance(Some(&[-0.5, -0.1])).unwrap() else {
        panic!()
    };
    assert_eq!(second.question, 0);
    assert_eq!(
        second.feed,
        "B\",verdict:\"".bytes().map(u32::from).collect::<Vec<_>>(),
        "the likelier letter, B, goes first"
    );
    let _ = got_hash;
}

#[test]
fn a_queue_schema_that_does_not_match_is_a_client_error_that_names_the_queue() {
    let f = family();
    let sp = spec(&spec_json());
    let c = context(queue_context(&["A", "B", "C"]));
    let r = f.render(&c, false).unwrap();
    let asked = both();
    let err = f
        .decision_read(&DecisionInput {
            context: Some((&c, &r)),
            spec: &sp,
            case: "c",
            asked: &asked,
        })
        .err()
        .unwrap();
    let FamilyError::Client(invalid) = err else {
        panic!("{err:?}")
    };
    assert_eq!(invalid.param.as_deref(), Some("queue"));
}

#[test]
fn a_question_with_no_option_to_read_is_a_client_error() {
    let f = family();
    let sp = spec(&spec_json());
    let asked = [Asked {
        question: 0,
        options: vec![],
    }];
    let err = f
        .decision_read(&DecisionInput {
            context: None,
            spec: &sp,
            case: "c",
            asked: &asked,
        })
        .err()
        .unwrap();
    assert!(err.is_client(), "{err:?}");
}

#[test]
fn a_caller_that_breaks_the_rules_of_a_plan_gets_an_internal_error() {
    let f = family();
    let sp = spec(&spec_json());
    let asked = both();
    let new_plan = || {
        f.decision_read(&DecisionInput {
            context: None,
            spec: &sp,
            case: "c",
            asked: &asked,
        })
        .unwrap()
    };
    assert!(
        matches!(new_plan().advance(Some(&[0.0])), Err(FamilyError::Internal(_))),
        "a first call with letters"
    );
    let mut p = new_plan();
    p.advance(None).unwrap();
    assert!(
        matches!(p.advance(None), Err(FamilyError::Internal(_))),
        "a later call without letters"
    );
    assert!(
        matches!(p.advance(Some(&[0.0])), Err(FamilyError::Internal(_))),
        "letters of the wrong length"
    );
    let mut p = new_plan();
    p.advance(None).unwrap();
    p.advance(Some(&[-1.0, -2.0, -3.0])).unwrap();
    p.advance(Some(&[-1.0, -2.0])).unwrap();
    assert!(
        matches!(p.advance(Some(&[-1.0, -2.0])), Err(FamilyError::Internal(_))),
        "a call after Done"
    );
    assert!(
        matches!(p.advance(None), Err(FamilyError::Internal(_))),
        "a call after Done, without letters"
    );
    let mut p = new_plan();
    p.advance(None).unwrap();
    assert!(
        matches!(p.advance(Some(&[f64::NAN, 0.0, 0.0])), Err(FamilyError::Internal(_))),
        "a NaN"
    );
}

#[test]
fn a_raw_read_gives_the_letters_and_suffix_tokens_of_gogi_fake() {
    let s = Server::start(Config::default()).unwrap();
    let ctx = contexts()[0].clone();
    let body = json!({"model": "toy", "contexts": [ctx.clone()], "suffixes": ["<asst>A", "plain"], "letters": ["A", "B", "C"], "top": 3});
    let (code, d) = post(&s, "/council/v2/reads", body);
    assert_eq!(code, 200, "{d}");
    let f = family();
    let letters: Vec<String> = ["A", "B", "C"].map(String::from).to_vec();
    let suffixes: Vec<String> = ["<asst>A", "plain"].map(String::from).to_vec();
    let raw = f.raw_read(&letters, &suffixes).unwrap();
    assert_eq!(raw.letters, [65, 66, 67]);
    assert_eq!(raw.suffixes[0], [toy::ASST, 65]);
    assert_eq!(raw.suffixes[1], "plain".bytes().map(u32::from).collect::<Vec<_>>());
    assert_eq!(
        raw.control_text,
        [ControlHit {
            place: "suffix:0".into(),
            token: "<asst>".into()
        }]
    );
    assert_eq!(
        d["signals"]["control_text"],
        json!([{"where": "suffix:0", "token": "<asst>"}])
    );
    let c = context(ctx);
    let rendering = f.render(&c, false).unwrap();
    for (i, suffix) in raw.suffixes.iter().enumerate() {
        let st = suffix
            .iter()
            .fold(State::after(SEED, &rendering.tokens), |s, &t| s.step(t));
        let lp = log_softmax(&st.logits());
        for (l, id) in letters.iter().zip(&raw.letters) {
            assert_eq!(
                d["reads"][i]["logprobs"][l].as_f64().unwrap(),
                lp[*id as usize],
                "suffix {i}, letter {l}"
            );
        }
    }
}

#[test]
fn a_letter_that_is_not_one_token_of_its_own_is_a_client_error() {
    let f = family();
    let one = |l: &[&str]| f.raw_read(&l.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[]);
    for bad in [&["AB", "C"][..], &["A", "A"], &[""]] {
        let FamilyError::Client(invalid) = one(bad).unwrap_err() else {
            panic!()
        };
        assert_eq!(invalid.param.as_deref(), Some("letters"), "{bad:?}");
    }
}

#[test]
fn a_reply_parser_labels_tokens_and_parses_the_reply() {
    let f = family();
    let mut parser = f.reply_parser(ReplyStart::prompt(true), &[]).unwrap();
    let mut tokens: Vec<u32> = "hmm".bytes().map(u32::from).collect();
    tokens.push(toy::ENDTHINK);
    tokens.extend("ok".bytes().map(u32::from));
    tokens.push(toy::CALL);
    tokens.extend(r#"sh({"b":1})"#.bytes().map(u32::from));
    tokens.push(toy::ENDCALL);
    tokens.push(toy::CALL);
    tokens.extend("broken(".bytes().map(u32::from));
    tokens.push(toy::ENDCALL);
    tokens.push(toy::END);
    let steps: Vec<_> = tokens.iter().map(|&t| parser.push(t).unwrap()).collect();
    assert_eq!(steps[0].phase, Phase::Reasoning);
    assert_eq!(steps[0].piece, "h");
    assert_eq!(steps[3].phase, Phase::Reasoning, "<endthink> closes the reasoning");
    assert_eq!(steps[3].piece, "");
    assert_eq!(steps[4].phase, Phase::Content);
    assert_eq!(steps[6].phase, Phase::Call);
    let reply = parser.finish(false);
    assert_eq!(
        (reply.reasoning.as_str(), reply.content.as_str(), reply.ended),
        ("hmm", "ok", true)
    );
    assert_eq!(reply.calls.len(), 2);
    let ReplyCall::Parsed { name, input } = &reply.calls[0] else {
        panic!()
    };
    assert_eq!((name.as_str(), input["b"].clone()), ("sh", json!(1)));
    let ReplyCall::Malformed { text, .. } = &reply.calls[1] else {
        panic!()
    };
    assert_eq!(text, "broken(");
}

#[test]
fn the_tools_of_the_context_do_not_change_a_toy_reply() {
    let f = family();
    let tools = [gogi_proto::context::Tool::new("sh", json!({"type": "object"}))];
    let parse = |tools: &[gogi_proto::context::Tool]| {
        let mut parser = f.reply_parser(ReplyStart::prompt(false), tools).unwrap();
        for t in [toy::CALL]
            .into_iter()
            .chain(r#"sh({"b":1})"#.bytes().map(u32::from))
            .chain([toy::ENDCALL, toy::END])
        {
            parser.push(t).unwrap();
        }
        parser.finish(false)
    };
    assert_eq!(parse(&tools), parse(&[]));
}

#[test]
fn a_reply_without_thinking_is_content_and_a_reply_with_no_end_has_not_ended() {
    let f = family();
    let mut parser = f.reply_parser(ReplyStart::prompt(false), &[]).unwrap();
    assert_eq!(parser.push(u32::from(b'x')).unwrap().phase, Phase::Content);
    let reply = parser.finish(false);
    assert_eq!((reply.content.as_str(), reply.ended), ("x", false));
    assert!(reply.tail.is_empty(), "the toy parser holds nothing back");
}

/// gogi "Continuations": `done.phase` of a cut reply is the phase that its next token would have.
#[test]
fn a_reply_parser_names_the_phase_of_the_next_token() {
    let f = family();
    let mut parser = f.reply_parser(ReplyStart::prompt(true), &[]).unwrap();
    assert_eq!(parser.phase(), Phase::Reasoning);
    parser.push(u32::from(b'a')).unwrap();
    assert_eq!(parser.phase(), Phase::Reasoning);
    parser.push(gogi_toy::ENDTHINK).unwrap();
    assert_eq!(
        parser.phase(),
        Phase::Content,
        "a reply cut just after <endthink> resumes in its content"
    );
    parser.push(gogi_toy::CALL).unwrap();
    assert_eq!(parser.phase(), Phase::Call);
    let open = f
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

#[test]
fn a_reply_parser_refuses_a_token_outside_the_vocabulary_and_a_token_after_the_end() {
    let f = family();
    let mut parser = f.reply_parser(ReplyStart::prompt(false), &[]).unwrap();
    assert!(matches!(
        parser.push(f.vocab_size() as u32),
        Err(FamilyError::Internal(_))
    ));
    parser.push(toy::END).unwrap();
    assert!(matches!(parser.push(u32::from(b'x')), Err(FamilyError::Internal(_))));
}

#[test]
fn tokenize_makes_bytes_and_never_a_marker_and_detokenize_undoes_it() {
    let f = family();
    let t = f.tokenize("a<end>é").unwrap();
    assert_eq!(t, "a<end>é".bytes().map(u32::from).collect::<Vec<_>>());
    assert_eq!(f.detokenize(&t).unwrap(), "a<end>é");
    assert_eq!(f.detokenize(&[toy::USER, 104, toy::END]).unwrap(), "<user>h<end>");
}

#[test]
fn detokenize_refuses_a_token_outside_the_vocabulary_and_bytes_that_are_not_text() {
    let f = family();
    assert!(f.detokenize(&[toy::VOCAB as u32]).unwrap_err().is_client());
    assert!(f.detokenize(&[0xff]).unwrap_err().is_client());
}

#[test]
fn the_vocabulary_is_the_toy_vocabulary() {
    assert_eq!(family().vocab_size(), toy::VOCAB);
}

#[test]
fn the_template_id_is_stable_and_has_the_form_of_spec_11_3a() {
    let id = ToyFamily::new().template_id().to_owned();
    assert_eq!(id, ToyFamily::new().template_id());
    let re = |s: &str| s.len() == 16 && s.bytes().all(|b| b.is_ascii_hexdigit());
    let (council, chat) = id
        .strip_prefix("nagaya-council:")
        .unwrap()
        .split_once("+toy-chat:")
        .unwrap();
    assert!(re(council) && re(chat), "{id}");
    // A fixed value, so that a change in the toy rendering shows in review.
    assert_eq!(id, "nagaya-council:b4d0ab381879513e+toy-chat:8cd37a15493d0e2e");
}

#[test]
fn a_rendering_that_is_not_the_contexts_is_an_internal_error_in_a_queue_read() {
    let f = family();
    let sp = spec(&spec_json());
    let c = context(queue_context(&["A", "B"]));
    let other = f.render(&context(contexts()[0].clone()), false).unwrap();
    let asked = both();
    let err = f
        .decision_read(&DecisionInput {
            context: Some((&c, &other)),
            spec: &sp,
            case: "c",
            asked: &asked,
        })
        .err()
        .unwrap();
    assert!(matches!(err, FamilyError::Internal(_)), "{err:?}");
}

#[test]
fn asked_questions_that_break_the_rules_of_the_caller_are_internal_errors() {
    let f = family();
    let sp = spec(&spec_json());
    let plan = |asked: &[Asked]| {
        f.decision_read(&DecisionInput {
            context: None,
            spec: &sp,
            case: "c",
            asked,
        })
        .err()
    };
    let internal = |e: Option<FamilyError>| matches!(e, Some(FamilyError::Internal(_)));
    assert!(internal(plan(&[])), "no question");
    assert!(
        internal(plan(&[Asked {
            question: 2,
            options: vec![0, 1]
        }])),
        "a question outside the spec"
    );
    let twice = [
        Asked {
            question: 0,
            options: vec![0, 1],
        },
        Asked {
            question: 0,
            options: vec![0, 1],
        },
    ];
    assert!(internal(plan(&twice)), "a question twice");
    assert!(
        internal(plan(&[Asked {
            question: 0,
            options: vec![1, 0]
        }])),
        "options out of order"
    );
    assert!(
        internal(plan(&[Asked {
            question: 0,
            options: vec![0, 3]
        }])),
        "an option the question lacks"
    );
    assert!(
        plan(&[Asked {
            question: 0,
            options: vec![0, 2]
        }])
        .is_none()
    );
}

#[test]
fn the_fake_engine_gives_logit_rows_as_wide_as_the_vocabulary_of_the_family() {
    use nagaya_abi::fake::FakeAlways;
    use nagaya_abi::{Engine, OutputsWanted, Segment, SegmentInputs, State as Slot};
    let f = family();
    let rendering = f.render(&context(contexts()[0].clone()), true).unwrap();
    let mut engine = FakeAlways::new().with_vocab(f.vocab_size());
    let last = rendering.tokens.len() as u64 - 1;
    let segment = Segment {
        control: Vec::new(),
        state: Slot::Slot(0),
        inputs: SegmentInputs::Tokens(rendering.tokens.clone()),
        positions: (0..=last).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted { logit_rows: vec![last] },
        layer_loop: None,
        cycles: None,
    };
    let out = engine.pass(&[segment], &[]).unwrap();
    let rows: Vec<_> = out.outputs.iter().filter(|o| o.name == "logits").collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].bytes.len(), f.vocab_size() * 4);
}

#[test]
fn a_reply_parser_says_when_the_reply_ended() {
    let f = family();
    let mut parser = f.reply_parser(ReplyStart::prompt(true), &[]).unwrap();
    for token in [u32::from(b'h'), toy::ENDTHINK, u32::from(b'o'), toy::CALL, toy::ENDCALL] {
        parser.push(token).unwrap();
        assert!(!parser.ended(), "token {token} does not end the reply");
    }
    parser.push(toy::END).unwrap();
    assert!(parser.ended(), "<end> ends the reply");
}

#[test]
fn the_speaking_bias_keeps_only_tokens_that_a_reply_renders_back() {
    let bias = nagaya_family_toy::speaking_bias();
    assert_eq!(bias.len(), toy::VOCAB);
    // The fake's values lie between -6 and 6. A kept token beats every other token by more than 12.
    let kept: Vec<u32> = (0..toy::VOCAB as u32).filter(|t| bias[*t as usize] > -12.0).collect();
    let mut want: Vec<u32> = (0..0x80).collect();
    want.push(toy::END);
    assert_eq!(kept, want);
    assert!(bias.iter().all(|b| *b >= -40.0 || *b == 0.0));

    // A reply of kept tokens, in each phase, renders back as the prompt, the reply but its stop token, and at most
    // two markers.
    let f = family();
    for thinking in [true, false] {
        let value = json!({"thinking": thinking, "messages": [{"role": "user", "content": "go", "snap": true}]});
        let c = context(value.clone());
        let prompt = f.render(&c, true).unwrap().tokens;
        let mut reply: Vec<u32> = (0..0x80).collect();
        reply.push(toy::END);
        let mut parser = f.reply_parser(ReplyStart::prompt(thinking), &[]).unwrap();
        for token in &reply {
            parser.push(*token).unwrap();
        }
        let parsed = parser.finish(false);
        let mut message = json!({"role": "assistant"});
        if !parsed.content.is_empty() {
            message["content"] = json!(parsed.content);
        }
        if !parsed.reasoning.is_empty() {
            message["reasoning"] = json!(parsed.reasoning);
        }
        let mut with_reply = value.clone();
        with_reply["messages"].as_array_mut().unwrap().push(message);
        let rendered = f.render(&context(with_reply), false).unwrap().tokens;
        let fed: Vec<u32> = prompt.iter().chain(&reply[..reply.len() - 1]).copied().collect();
        assert!(rendered.starts_with(&fed), "thinking {thinking}");
        assert!(
            rendered.len() - fed.len() <= 2,
            "thinking {thinking}: {:?}",
            &rendered[fed.len()..]
        );
    }
}
