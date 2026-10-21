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
fn the_next_request_starts_from_the_head_of_the_reply_and_keeps_its_boundary() {
    // Thinking off: the template keeps the reply as it is when a user message follows.
    let (f, script) = scripted();
    let value = ctx(false, opening());
    let mut reply = bytes("ok");
    reply.push(gogi_toy::END);
    script.set(prompt(&value), reply);
    let done = generate(&f, &value, 50, json!({"temperature": 0}));
    let head = done.head.clone().expect("the generation holds its head");
    let head_tokens = done.snapshots.last().unwrap().tokens;
    // The client sends the reply unmarked, then a new user message (gogi "Generations": head).
    let mut messages = opening();
    messages.push(serde_json::to_value(done.as_message().unwrap()).unwrap());
    messages.push(json!({"role": "user", "content": "Now list what is left.", "snap": true}));
    let next = ctx(false, messages);
    let built = f.service.build(&context(&next), false, Default::default()).unwrap();
    assert_eq!(built.from, Some(head), "the build starts from the held head");
    assert!(built.usage.kept >= head_tokens, "{:?}", built.usage);
    // The reply's end is not a boundary of the context, so it is not in the list (gogi "Terms": boundary).
    assert!(
        built.snapshots.iter().all(|snapshot| snapshot.message != 2),
        "{:?}",
        built.snapshots
    );
    // The state has the seal at the end of the reply, as the head was made. Thus it equals a straight build with the
    // reply marked, and it differs from a straight build without the mark (gogi "Boundaries are content").
    let mut marked = next.clone();
    marked["messages"][2]["snap"] = json!(true);
    let fresh = Fixture::new();
    let with_mark = fresh
        .service
        .build(&context(&marked), false, Default::default())
        .unwrap();
    assert_eq!(with_mark.head, built.head);
    let fresh = Fixture::new();
    let without = fresh.service.build(&context(&next), false, Default::default()).unwrap();
    assert_ne!(without.head, built.head);
}

#[test]
fn a_generation_that_starts_from_a_cut_holds_its_head_and_a_dry_run_reports_the_cut() {
    let (f, script) = scripted();
    let value = ctx(false, opening());
    let mut reply = bytes("ok");
    reply.push(gogi_toy::END);
    script.set(prompt(&value), reply.clone());
    let first = generate(&f, &value, 50, json!({"temperature": 0}));
    let mut messages = opening();
    messages.push(serde_json::to_value(first.as_message().unwrap()).unwrap());
    messages.push(json!({"role": "user", "content": "And then?", "snap": true}));
    let next = ctx(false, messages.clone());
    let dry = f.service.build(&context(&next), true, Default::default()).unwrap();
    assert_eq!(dry.from, first.head, "a dry run starts from the cut, as a build does");
    script.set(prompt(&next), reply);
    let second = generate(&f, &next, 50, json!({"temperature": 0}));
    assert_eq!(second.usage.kept, first.snapshots.last().unwrap().tokens);
    let head = second.head.clone().expect("a generation from a cut holds its head");
    messages.push(serde_json::to_value(second.as_message().unwrap()).unwrap());
    // The store holds the head under the recipe of its own boundaries, the cut too. Thus the context with the first
    // reply marked finds it at its end with nothing to feed.
    let mut marked = ctx(false, messages.clone());
    marked["messages"][2]["snap"] = json!(true);
    let dry = f.service.build(&context(&marked), true, Default::default()).unwrap();
    assert_eq!((dry.head, dry.usage.fed), (Some(head.clone()), 0));
    let built = f
        .service
        .build(&context(&ctx(false, messages)), false, Default::default())
        .unwrap();
    assert_eq!((built.head, built.usage.fed), (Some(head), 0));
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
    let (tx, rx) = mpsc::sync_channel(1 << 16);
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
    let mut parser = ToyFamily::new().reply_parser(true, &[]).unwrap();
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
    let (tx, rx) = mpsc::sync_channel(1 << 16);
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
    let (tx, rx) = mpsc::sync_channel(1 << 16);
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

/// A family that holds back the last character of a part, as a parser holds text that may begin a marker. A marker
/// token releases the held text in its own step, except the stop token: after it, the held text is in the tail.
/// With `flip`, the parser gives its tail the other phase, which is a defect of a family.
struct Holding {
    toy: ToyFamily,
    flip: bool,
}

struct HoldingParser {
    inner: Box<dyn nagaya_council::family::ReplyParser>,
    held: String,
    flip: bool,
}

impl nagaya_council::family::ReplyParser for HoldingParser {
    fn push(&mut self, token: u32) -> Result<nagaya_council::family::ReplyStep, nagaya_council::family::FamilyError> {
        let mut step = self.inner.push(token)?;
        if step.piece.is_empty() {
            if token != gogi_toy::END {
                step.piece = std::mem::take(&mut self.held);
            }
        } else {
            self.held.push_str(&step.piece);
            let last = self.held.chars().last().unwrap();
            let cut = self.held.len() - last.len_utf8();
            step.piece = self.held[..cut].to_owned();
            self.held = last.to_string();
        }
        Ok(step)
    }

    fn ended(&self) -> bool {
        self.inner.ended()
    }

    fn finish(self: Box<Self>) -> nagaya_council::family::ParsedReply {
        let this = *self;
        let mut parsed = this.inner.finish();
        let phase = if parsed.content.is_empty() != this.flip {
            Phase::Reasoning
        } else {
            Phase::Content
        };
        parsed.tail = vec![nagaya_council::family::ReplyStep {
            phase,
            piece: this.held,
        }];
        parsed
    }
}

impl Family for Holding {
    fn render(
        &self,
        context: &Context,
        generation_prompt: bool,
    ) -> Result<nagaya_council::family::Rendering, nagaya_council::family::FamilyError> {
        self.toy.render(context, generation_prompt)
    }

    fn decision_read(
        &self,
        input: &nagaya_council::family::DecisionInput<'_>,
    ) -> Result<Box<dyn nagaya_council::family::ReadPlan>, nagaya_council::family::FamilyError> {
        self.toy.decision_read(input)
    }

    fn raw_read(
        &self,
        letters: &[String],
        suffixes: &[String],
    ) -> Result<nagaya_council::family::RawRead, nagaya_council::family::FamilyError> {
        self.toy.raw_read(letters, suffixes)
    }

    fn reply_parser(
        &self,
        thinking: bool,
        tools: &[gogi_proto::context::Tool],
    ) -> Result<Box<dyn nagaya_council::family::ReplyParser>, nagaya_council::family::FamilyError> {
        Ok(Box::new(HoldingParser {
            inner: self.toy.reply_parser(thinking, tools)?,
            held: String::new(),
            flip: self.flip,
        }))
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, nagaya_council::family::FamilyError> {
        self.toy.tokenize(text)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, nagaya_council::family::FamilyError> {
        self.toy.detokenize(tokens)
    }

    fn template_id(&self) -> &str {
        self.toy.template_id()
    }

    fn vocab_size(&self) -> usize {
        self.toy.vocab_size()
    }
}

/// A fixture with the holding family, and the script of its engine.
fn holding(flip: bool) -> (Fixture, Script) {
    let script = Script::default();
    let factory = TermFactory {
        script: script.clone(),
        ..TermFactory::toy()
    };
    let config = ServiceConfig::new("toy", council_support::ENGINE);
    let family = std::sync::Arc::new(Holding {
        toy: ToyFamily::new(),
        flip,
    });
    (Fixture::with_family(factory, family, 1 << 20, None, config), script)
}

/// Stream a generation, and give its token events and its reply.
fn stream(f: &Fixture, value: &Value, max_tokens: u32) -> (Vec<TokenEvent>, GenerationDone) {
    let (tx, rx) = mpsc::sync_channel(1 << 16);
    let done = f
        .service
        .generate(&request(value, max_tokens, json!({"temperature": 0})), Some(tx))
        .unwrap();
    (rx.try_iter().collect(), done)
}

#[test]
fn the_text_that_a_parser_releases_after_the_last_token_joins_the_last_piece() {
    // A reply that ends its turn: the held `k` comes after the stop token.
    let (f, script) = holding(false);
    let value = ctx(true, opening());
    script.set(prompt(&value), think_then_answer());
    let (events, done) = stream(&f, &value, 50);
    assert_eq!((done.reasoning.as_str(), done.content.as_str()), ("hi", "ok"));
    assert_eq!(events.len() as u64, done.usage.completion);
    check_pieces(&events, &done).unwrap();
    assert_eq!(
        events.last().unwrap().piece,
        "k",
        "the stop token carries the held text"
    );

    // A reply that max_tokens cuts in its reasoning: the held `l` comes after the last token.
    let (f, script) = holding(false);
    script.set(prompt(&value), bytes("hello"));
    let (events, done) = stream(&f, &value, 3);
    assert_eq!((done.reasoning.as_str(), done.finish), ("hel", Finish::MaxTokens));
    assert_eq!(events.len(), 3);
    check_pieces(&events, &done).unwrap();
}

#[test]
fn text_released_after_the_last_token_in_another_phase_is_an_internal_error() {
    let (f, script) = holding(true);
    let value = ctx(true, opening());
    script.set(prompt(&value), think_then_answer());
    let (tx, _rx) = mpsc::sync_channel(1 << 16);
    let err = f
        .service
        .generate(&request(&value, 50, json!({"temperature": 0})), Some(tx))
        .unwrap_err();
    assert!(
        matches!(&err, ServiceError::Internal(message) if message.contains("a token event has one phase")),
        "{err:?}"
    );
}
