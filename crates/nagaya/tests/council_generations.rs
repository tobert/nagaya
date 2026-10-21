//! Generations: a reply sampled after a context (gogi "Generations", spec 11.2, 11.3c). The engine is the fake
//! engine over the toy vocabulary, and the family is the toy family. A scripted engine favors the tokens of a known
//! reply, so a test knows what the sampler picks.

mod council_support;

use std::sync::mpsc;

use council_support::{Direct, Fixture, Script, TermFactory};
use gogi_proto::context::Context;
use gogi_proto::generation::{
    Finish, GenerationDone, GenerationRequest, Phase, ReplyToolCall, TokenEvent, check_pieces,
};
use nagaya::council::ServiceConfig;
use nagaya::council::error::ServiceError;
use nagaya_council::family::Family;
use nagaya_council::sampler::{Settings, pick};
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};

fn ctx(thinking: bool, messages: Vec<Value>) -> Value {
    json!({"thinking": thinking, "tools": [{"name": "sh", "input_schema": {"type": "object"}}], "messages": messages})
}

fn opening() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "You are a coding agent. Use the shell tool."}),
        json!({"role": "user", "content": "Delete the build directory, then rebuild.", "snap": true}),
    ]
}

fn context(value: &Value) -> Context {
    serde_json::from_value(value.clone()).unwrap()
}

fn request(value: &Value, max_tokens: u32, sample: Value) -> GenerationRequest {
    serde_json::from_value(json!({"model": "toy", "context": value, "max_tokens": max_tokens, "sample": sample}))
        .unwrap()
}

fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

/// A fixture whose engine favors a script, and the script.
fn scripted() -> (Fixture, Script) {
    let script = Script::default();
    let factory = TermFactory {
        script: script.clone(),
        ..TermFactory::toy()
    };
    let config = ServiceConfig::new("toy", council_support::ENGINE);
    (Fixture::with_factory(factory, 1 << 20, None, config), script)
}

/// The tokens of the prompt of a context.
fn prompt(value: &Value) -> usize {
    ToyFamily::new().render(&context(value), true).unwrap().tokens.len()
}

fn think_then_answer() -> Vec<u32> {
    let mut reply = bytes("hi");
    reply.push(gogi_toy::ENDTHINK);
    reply.extend(bytes("ok"));
    reply.push(gogi_toy::END);
    reply
}

fn generate(f: &Fixture, value: &Value, max_tokens: u32, sample: Value) -> GenerationDone {
    f.service.generate(&request(value, max_tokens, sample), None).unwrap()
}

#[test]
fn a_reply_ends_its_turn_and_reports_its_parts_usage_and_seed() {
    let (f, script) = scripted();
    let value = ctx(true, opening());
    script.set(prompt(&value), think_then_answer());
    let built = f.service.build(&context(&value), false, Default::default()).unwrap();
    let done = generate(&f, &value, 50, json!({"temperature": 1.0, "seed": 7}));
    assert_eq!((done.reasoning.as_str(), done.content.as_str()), ("hi", "ok"));
    assert_eq!(done.finish, Finish::EndTurn);
    assert!(done.tool_calls.is_empty());
    assert_eq!(done.seed, Some(7));
    let u = done.usage;
    assert_eq!(
        (u.completion, u.reasoning),
        (6, 3),
        "<endthink> belongs to the reasoning"
    );
    assert_eq!(u.prompt, prompt(&value) as u64);
    assert!(
        u.kept >= built.usage.prompt,
        "the generation starts from the build: {u:?}"
    );
    assert_eq!(u.kept + u.fed, u.prompt);
    let stats = done.stats.unwrap();
    assert_eq!(stats.reasoning.unwrap().tokens, 3);
    assert_eq!(stats.content.unwrap().tokens, 3, "<end> belongs to the content");
    assert_eq!(done.model, "toy");
}

#[test]
fn the_head_after_the_reply_is_the_head_of_the_context_with_the_reply_appended() {
    let (f, script) = scripted();
    let value = ctx(true, opening());
    script.set(prompt(&value), think_then_answer());
    let done = generate(&f, &value, 50, json!({"temperature": 0}));
    let head = done.head.clone().expect("the generation holds its head");
    assert_eq!(done.snapshots.last().unwrap().id, head);
    let mut messages = opening();
    messages.push(serde_json::to_value(done.as_message().unwrap()).unwrap());
    let after = ctx(true, messages);
    let built = f.service.build(&context(&after), false, Default::default()).unwrap();
    assert_eq!(built.head, Some(head));
    assert_eq!(built.usage.fed, 0, "the build keeps everything");
    // The head holds the same state as a straight build of the context with the reply.
    let fresh = Fixture::new();
    let straight = fresh
        .service
        .build(&context(&after), false, Default::default())
        .unwrap();
    assert_eq!(straight.head, built.head);
}

#[test]
fn a_reply_with_a_call_names_the_call_and_ends_with_tool_use() {
    let (f, script) = scripted();
    let value = ctx(false, opening());
    let mut reply = vec![gogi_toy::CALL];
    reply.extend(bytes(r#"sh({"cmd":"ls"})"#));
    reply.extend([gogi_toy::ENDCALL, gogi_toy::END]);
    script.set(prompt(&value), reply);
    let done = generate(&f, &value, 50, json!({"temperature": 0}));
    assert_eq!(done.finish, Finish::ToolUse);
    let [ReplyToolCall::Parsed { id, name, input, .. }] = &done.tool_calls[..] else {
        panic!("{:?}", done.tool_calls)
    };
    assert!(id.starts_with("call_"), "{id}");
    assert_eq!((name.as_str(), input["cmd"].clone()), ("sh", json!("ls")));
    assert!(done.head.is_some(), "a reply with a call holds its head");
    let again = generate(&f, &value, 50, json!({"temperature": 0}));
    assert_eq!(
        again.tool_calls, done.tool_calls,
        "a replay names the calls the same way"
    );
}

#[test]
fn a_reply_stops_at_max_tokens_and_at_the_end_of_the_context() {
    let (f, script) = scripted();
    let value = ctx(false, opening());
    script.set(prompt(&value), bytes("abcdefgh"));
    let done = generate(&f, &value, 3, json!({"temperature": 0}));
    assert_eq!(
        (done.finish, done.content.as_str(), done.usage.completion),
        (Finish::MaxTokens, "abc", 3)
    );
    // A served model that holds two tokens after the prompt.
    let mut config = ServiceConfig::new("toy", council_support::ENGINE);
    config.context_tokens = prompt(&value) + 2;
    let factory = TermFactory {
        script,
        ..TermFactory::toy()
    };
    let small = Fixture::with_factory(factory, 1 << 20, None, config);
    let done = generate(&small, &value, 5, json!({"temperature": 0}));
    assert_eq!(
        (done.finish, done.content.as_str(), done.usage.completion),
        (Finish::ContextFull, "ab", 2)
    );
    let done = generate(&small, &value, 2, json!({"temperature": 0}));
    assert_eq!(done.finish, Finish::MaxTokens, "max_tokens goes first when both hold");
}

#[test]
fn the_sampled_tokens_are_the_picks_of_a_direct_run() {
    let f = Fixture::new();
    let value = ctx(true, opening());
    let (tx, rx) = mpsc::channel();
    let done = f
        .service
        .generate(&request(&value, 12, json!({"temperature": 1.0, "seed": 42})), Some(tx))
        .unwrap();
    let events: Vec<TokenEvent> = rx.try_iter().collect();
    let tokens: Vec<u32> = events.iter().map(|e| e.id).collect();
    assert_eq!(tokens.len() as u64, done.usage.completion);
    // The oracle: the context built and sealed, the generation prompt fed, and each pick fed back.
    let rendering = ToyFamily::new().render(&context(&value), true).unwrap();
    let mut run = Direct::built(&context(&value));
    let mut row = run.feed_all(&rendering.tokens[rendering.end..]);
    let settings = Settings {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: 42,
    };
    let mut want = Vec::new();
    let mut parser = ToyFamily::new().reply_parser(true);
    for (k, event) in events.iter().enumerate() {
        let chosen = pick(&row, &settings, (rendering.tokens.len() + k) as u64).unwrap();
        want.push(chosen.token);
        assert_eq!((event.p, event.h), (chosen.p, chosen.h), "token {k}");
        parser.push(chosen.token).unwrap();
        if k + 1 < tokens.len() {
            row = run.feed(&[chosen.token], true).unwrap();
        }
    }
    assert_eq!(tokens, want);
    assert_eq!(parser.ended(), done.finish != Finish::MaxTokens);
}

#[test]
fn a_stream_sends_one_event_for_each_token_and_its_pieces_join_to_the_reply() {
    let (f, script) = scripted();
    let value = ctx(true, opening());
    script.set(prompt(&value), think_then_answer());
    let (tx, rx) = mpsc::channel();
    let done = f
        .service
        .generate(&request(&value, 50, json!({"temperature": 0})), Some(tx))
        .unwrap();
    let events: Vec<TokenEvent> = rx.try_iter().collect();
    assert_eq!(events.len() as u64, done.usage.completion);
    assert!(events.iter().enumerate().all(|(i, e)| e.n == i as u64 + 1));
    let thought = events.iter().filter(|e| e.phase == Phase::Reasoning).count() as u64;
    assert_eq!(thought, done.usage.reasoning);
    check_pieces(&events, &done).unwrap();
}

#[test]
fn a_closed_stream_cancels_the_reply_and_holds_no_head() {
    let (f, script) = scripted();
    let value = ctx(true, opening());
    script.set(prompt(&value), think_then_answer());
    let built = f.service.build(&context(&value), false, Default::default()).unwrap();
    let entries = f.entries();
    let (tx, rx) = mpsc::channel();
    drop(rx);
    let err = f
        .service
        .generate(&request(&value, 50, json!({"temperature": 0})), Some(tx))
        .unwrap_err();
    assert_eq!(err, ServiceError::Cancelled);
    assert_eq!(f.entries(), entries, "a cancelled reply holds nothing");
    assert_eq!(f.service.free_slots(), 3);
    assert!(built.head.is_some());
}

#[test]
fn a_seed_replays_and_a_request_with_no_seed_reports_the_one_it_chose() {
    let f = Fixture::new();
    let value = ctx(false, opening());
    let one = generate(&f, &value, 10, json!({"seed": 9}));
    let two = generate(&f, &value, 10, json!({"seed": 9}));
    assert_eq!(
        (one.content.clone(), one.usage.completion),
        (two.content.clone(), two.usage.completion)
    );
    let chosen = generate(&f, &value, 10, json!({}));
    let seed = chosen.seed.expect("the response names the seed");
    let replay = generate(&f, &value, 10, json!({"seed": seed}));
    assert_eq!(replay.content, chosen.content);
}

fn invalid_param(result: Result<GenerationDone, ServiceError>) -> String {
    match result {
        Err(ServiceError::Invalid(invalid)) => invalid.param.unwrap_or_default(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn each_refused_generation_names_its_field_and_runs_nothing() {
    let f = Fixture::new();
    let value = ctx(false, opening());
    let generate = |r: GenerationRequest| f.service.generate(&r, None);
    assert_eq!(invalid_param(generate(request(&value, 0, json!({})))), "max_tokens");
    assert_eq!(invalid_param(generate(request(&value, 5000, json!({})))), "max_tokens");
    let mut riding = request(&value, 5, json!({}));
    riding.decisions = serde_json::from_value(json!([{"spec_id": "sha256:00", "when": "after"}])).unwrap();
    assert_eq!(invalid_param(generate(riding)), "decisions");
    assert!(matches!(
        generate(request(&value, 5, json!({"top_p": 0.0}))),
        Err(ServiceError::Invalid(_))
    ));
    let mut config = ServiceConfig::new("toy", council_support::ENGINE);
    config.context_tokens = prompt(&value);
    let full = Fixture::with(0, 1 << 20, None, config);
    let err = full.service.generate(&request(&value, 1, json!({})), None).unwrap_err();
    assert_eq!(
        err,
        ServiceError::ContextFull {
            prompt_tokens: prompt(&value) as u64,
            context_tokens: prompt(&value) as u64
        }
    );
    assert_eq!(f.entries() + full.entries(), 0, "nothing was built");
}
