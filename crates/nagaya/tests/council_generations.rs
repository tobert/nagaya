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
use nagaya_council::family::{Family, ReplyStart};
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

/// The user message of a chat client has no mark. The prompt still seals at the end of its last message, so the
/// state holds that seal, and the head after the reply must keep it (spec 6.14). Measured live on the Qwen3.8 engine:
/// the second turn kept 34 of 180 prompt tokens and fed 146.
fn unmarked() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "You are a coding agent. Use the shell tool."}),
        json!({"role": "user", "content": "Delete the build directory, then rebuild."}),
    ]
}

#[test]
fn a_reply_to_an_unmarked_message_holds_a_head_and_the_next_turn_feeds_only_its_message() {
    let (f, script) = scripted();
    let value = ctx(false, unmarked());
    let mut reply = bytes("ok");
    reply.push(gogi_toy::END);
    script.set(prompt(&value), reply);
    let done = generate(&f, &value, 50, json!({"temperature": 0}));
    let head = done
        .head
        .clone()
        .expect("a reply to an unmarked message holds its head");
    assert_eq!(done.snapshots.last().unwrap().id, head);
    let head_tokens = done.snapshots.last().unwrap().tokens;
    let mut messages = unmarked();
    messages.push(serde_json::to_value(done.as_message().unwrap()).unwrap());
    messages.push(json!({"role": "user", "content": "Now list what is left."}));
    let next = ctx(false, messages);
    let built = f.service.build(&context(&next), false, Default::default()).unwrap();
    assert_eq!(built.from, Some(head), "the build starts from the held head");
    assert_eq!(
        built.usage.kept, head_tokens as u64,
        "the next turn keeps the whole reply: {:?}",
        built.usage
    );
    assert_eq!(built.usage.fed, built.usage.prompt - head_tokens as u64);
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
    assert_eq!(
        (done.finish, done.phase),
        (Finish::ContextFull, None),
        "context_full goes before max_tokens, and a context_full reply has no phase (gogi a07d85e)"
    );
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
    let mut parser = ToyFamily::new().reply_parser(ReplyStart::prompt(true), &[]).unwrap();
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

    fn phase(&self) -> Phase {
        self.inner.phase()
    }

    fn finish(self: Box<Self>, max_tokens: bool) -> nagaya_council::family::ParsedReply {
        let this = *self;
        let mut parsed = this.inner.finish(max_tokens);
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
        start: nagaya_council::family::ReplyStart,
        tools: &[gogi_proto::context::Tool],
    ) -> Result<Box<dyn nagaya_council::family::ReplyParser>, nagaya_council::family::FamilyError> {
        Ok(Box::new(HoldingParser {
            inner: self.toy.reply_parser(start, tools)?,
            held: String::new(),
            flip: self.flip,
        }))
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, nagaya_council::family::FamilyError> {
        self.toy.tokenize(text)
    }

    fn tokenize_special(&self, text: &str) -> Result<Vec<u32>, nagaya_council::family::FamilyError> {
        self.toy.tokenize_special(text)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, nagaya_council::family::FamilyError> {
        self.toy.detokenize(tokens)
    }

    fn pieces(&self, tokens: &[u32]) -> Result<Vec<String>, nagaya_council::family::FamilyError> {
        self.toy.pieces(tokens)
    }

    fn template_id(&self) -> &str {
        self.toy.template_id()
    }

    fn vocab_size(&self) -> usize {
        self.toy.vocab_size()
    }

    fn thinks(&self) -> bool {
        self.toy.thinks()
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

// ---------------------------------------------------------------- continuations (gogi "Continuations")

/// The context `value` with its last message replaced by `open`.
fn with_open(value: &Value, open: Value) -> Value {
    let mut value = value.clone();
    value["messages"].as_array_mut().unwrap().push(open);
    value
}

fn think_then_answer_long() -> Vec<u32> {
    let mut reply = bytes("hi");
    reply.push(gogi_toy::ENDTHINK);
    reply.extend(bytes("okay"));
    reply.push(gogi_toy::END);
    reply
}

#[test]
fn a_continuation_goes_on_from_the_open_text_and_counts_only_the_sampled_tokens() {
    let (f, script) = scripted();
    f.service
        .build(&context(&ctx(false, opening())), false, Default::default())
        .unwrap();
    let value = with_open(
        &ctx(false, opening()),
        json!({"role": "assistant", "content": "Plan:", "continue": true}),
    );
    let mut reply = bytes(" ls");
    reply.push(gogi_toy::END);
    script.set(prompt(&value), reply);
    let (tx, rx) = mpsc::sync_channel(1 << 16);
    let done = f
        .service
        .generate(&request(&value, 50, json!({"temperature": 0})), Some(tx))
        .unwrap();
    assert_eq!((done.content.as_str(), done.finish), ("Plan: ls", Finish::EndTurn));
    assert_eq!(done.phase, None, "only a reply cut by max_tokens names its phase");
    let u = done.usage;
    assert_eq!((u.prompt, u.completion, u.reasoning), (prompt(&value) as u64, 4, 0));
    assert_eq!(u.kept + u.fed, u.prompt);
    assert!(
        u.fed >= 1,
        "the last token of the open text gives the first logit row: {u:?}"
    );
    let events: Vec<TokenEvent> = rx.try_iter().collect();
    assert_eq!(events.iter().map(|e| e.n).collect::<Vec<_>>(), [1, 2, 3, 4]);
    let open = context(&value).messages.last().unwrap().clone();
    gogi_proto::generation::check_pieces_after(&events, &done, &open).unwrap();
    // The build sealed only the boundaries before the open message, so the reply goes back closed and the next
    // request starts from the head after it.
    let user = ToyFamily::new()
        .render(&context(&ctx(false, opening())), false)
        .unwrap()
        .end as u64;
    assert_eq!(
        u.kept, user,
        "the continuation builds the context before its open message: {u:?}"
    );
    let head = done.head.clone().expect("a continuation holds the head of its reply");
    let mut messages = opening();
    messages.push(serde_json::to_value(done.as_message().unwrap()).unwrap());
    messages[2]["snap"] = json!(true);
    let built = f
        .service
        .build(&context(&ctx(false, messages)), true, Default::default())
        .unwrap();
    assert_eq!((built.head, built.usage.fed), (Some(head), 0));
}

/// Decided by Amy 2026-10-21 (spec 11.3c): a cut reply holds a sealed head. A snapshot holds no logit row, so the head
/// is the state before the last sampled token, and a continuation from it feeds that token.
#[test]
fn a_cut_reply_holds_a_sealed_head_and_a_continuation_starts_from_it() {
    let (f, script) = scripted();
    let value = ctx(true, opening());
    script.set(prompt(&value), think_then_answer_long());
    let cut = generate(&f, &value, 4, json!({"temperature": 0}));
    assert_eq!(
        (cut.reasoning.as_str(), cut.content.as_str(), cut.finish, cut.phase),
        ("hi", "o", Finish::MaxTokens, Some(Phase::Content))
    );
    let head = cut.head.clone().expect("a cut reply holds a sealed head");
    let last = cut.snapshots.last().unwrap().clone();
    assert_eq!(last.id, head, "the head is listed last");
    assert_eq!(
        (last.message, last.tokens),
        (2, (prompt(&value) + 3) as u64),
        "the head holds the state before the last sampled token"
    );
    let open = cut.as_open_message().unwrap();
    let resumed = with_open(&value, serde_json::to_value(&open).unwrap());
    assert_eq!(
        prompt(&resumed),
        prompt(&value) + 4,
        "the open rendering is the prompt and the sampled tokens"
    );
    let dry = f.service.build(&context(&resumed), true, Default::default()).unwrap();
    assert_eq!(
        (dry.from.clone(), dry.usage.kept, dry.usage.fed),
        (Some(head.clone()), last.tokens, 1)
    );
    let done = generate(&f, &resumed, 46, json!({"temperature": 0}));
    assert_eq!(
        (done.usage.kept, done.usage.fed),
        (last.tokens, 1),
        "the continuation feeds the last sampled token"
    );
    assert_eq!(
        (
            done.reasoning.as_str(),
            done.content.as_str(),
            done.finish,
            done.usage.completion
        ),
        ("hi", "okay", Finish::EndTurn, 4)
    );
}

/// A cut reply sent back closed starts from its head when the open rendering's tokens begin the closed rendering's.
/// The toy template drops the reasoning of a reply when a user message follows, so the reply has none.
#[test]
fn a_cut_reply_sent_back_closed_starts_from_its_head() {
    let (f, script) = scripted();
    let value = ctx(false, opening());
    let mut reply = bytes("okay");
    reply.push(gogi_toy::END);
    script.set(prompt(&value), reply);
    let cut = generate(&f, &value, 3, json!({"temperature": 0}));
    assert_eq!((cut.content.as_str(), cut.finish), ("oka", Finish::MaxTokens));
    let head = cut.head.clone().expect("a cut reply holds a sealed head");
    let tokens = cut.snapshots.last().unwrap().tokens;
    let mut messages = opening();
    messages.push(serde_json::to_value(cut.as_message().unwrap()).unwrap());
    messages.push(json!({"role": "user", "content": "Go on.", "snap": true}));
    let next = ctx(false, messages);
    let dry = f.service.build(&context(&next), true, Default::default()).unwrap();
    assert_eq!((dry.from, dry.usage.kept), (Some(head), tokens), "{:?}", dry.usage);
    let built = f.service.build(&context(&next), false, Default::default()).unwrap();
    assert_eq!(built.usage.kept, tokens);
    // The state keeps the seal of the head, so the build equals a straight build that seals there too. It differs
    // from a straight build without that seal (gogi "Boundaries are content").
    let fresh = Fixture::new();
    let straight = fresh.service.build(&context(&next), false, Default::default()).unwrap();
    assert_ne!(straight.head, built.head);
}

/// A continuation that does not start from a head seals nothing inside the open text, so it writes what the whole
/// reply wrote, bit for bit. The fake engine's logits differ after a seal. The engine speaks the toy language, so the
/// reply renders back as its tokens.
#[test]
fn a_continuation_from_before_the_open_message_writes_what_the_whole_reply_wrote() {
    let speaking = || {
        let factory = TermFactory {
            speaking: true,
            ..TermFactory::toy()
        };
        Fixture::with_factory(
            factory,
            1 << 20,
            None,
            ServiceConfig::new("toy", council_support::ENGINE),
        )
    };
    let f = speaking();
    let value = ctx(true, opening());
    let whole = generate(&f, &value, 30, json!({"temperature": 0}));
    let cut = generate(&f, &value, 3, json!({"temperature": 0}));
    let Some(open) = cut.as_open_message() else {
        panic!("the cut reply goes back open: {cut:?}")
    };
    let resumed = with_open(&value, serde_json::to_value(&open).unwrap());
    // Another store holds no head, so the continuation builds the context before the open message.
    let other = speaking();
    let done = generate(&other, &resumed, 27, json!({"temperature": 0}));
    assert_eq!(done.usage.kept, 0);
    assert_eq!(
        (
            done.reasoning.as_str(),
            done.content.as_str(),
            done.finish,
            done.usage.completion + 3
        ),
        (
            whole.reasoning.as_str(),
            whole.content.as_str(),
            whole.finish,
            whole.usage.completion
        )
    );
    // The store with the head starts from it.
    let from_head = generate(&f, &resumed, 27, json!({"temperature": 0}));
    assert_eq!(
        (from_head.usage.kept, from_head.usage.fed),
        (cut.snapshots.last().unwrap().tokens, 1)
    );
}

#[test]
fn a_reply_cut_inside_its_thinking_resumes_inside_it() {
    let (f, script) = scripted();
    let value = ctx(true, opening());
    script.set(prompt(&value), think_then_answer_long());
    let cut = generate(&f, &value, 1, json!({"temperature": 0}));
    assert_eq!((cut.reasoning.as_str(), cut.phase), ("h", Some(Phase::Reasoning)));
    let open = cut.as_open_message().unwrap();
    assert_eq!(open.content, None);
    let done = generate(
        &f,
        &with_open(&value, serde_json::to_value(&open).unwrap()),
        49,
        json!({"temperature": 0}),
    );
    assert_eq!((done.reasoning.as_str(), done.content.as_str()), ("hi", "okay"));
    assert_eq!(done.usage.reasoning, 2, "the rest of the thought and <endthink>");
    assert_eq!(
        done.usage.fed, 1,
        "the continuation starts from the head of the cut reply"
    );
}

#[test]
fn a_reply_cut_inside_a_call_names_the_call_phase_and_holds_no_head() {
    let (f, script) = scripted();
    let value = ctx(false, opening());
    let mut reply = vec![gogi_toy::CALL];
    reply.extend(bytes(r#"sh({"cmd":"ls"})"#));
    reply.extend([gogi_toy::ENDCALL, gogi_toy::END]);
    script.set(prompt(&value), reply);
    let cut = generate(&f, &value, 3, json!({"temperature": 0}));
    assert_eq!((cut.finish, cut.phase), (Finish::MaxTokens, Some(Phase::Call)));
    assert_eq!(
        cut.head, None,
        "an open message holds no calls, so the reply has no open rendering"
    );
    assert_eq!(cut.as_open_message(), None);
}

#[test]
fn an_open_message_builds_only_as_a_dry_run() {
    let f = Fixture::new();
    let value = with_open(
        &ctx(false, opening()),
        json!({"role": "assistant", "content": "Plan:", "continue": true}),
    );
    let Err(ServiceError::Invalid(invalid)) = f.service.build(&context(&value), false, Default::default()) else {
        panic!("a build of an open message is refused")
    };
    assert_eq!(invalid.param.as_deref(), Some("continue"));
    assert_eq!(f.entries(), 0);
    let dry = f.service.build(&context(&value), true, Default::default()).unwrap();
    assert_eq!(
        (dry.usage.prompt, dry.usage.kept, dry.usage.fed),
        (prompt(&value) as u64, 0, prompt(&value) as u64)
    );
}

/// The head after a continuation's reply chains through the seam of the cut reply. A context that sends that reply
/// back closed follows the store's index through the seam to that head, and starts there (spec 11.3c).
#[test]
fn a_continued_reply_sent_back_closed_starts_from_its_head() {
    let (f, script) = scripted();
    let value = ctx(false, opening());
    let mut reply = bytes("okay");
    reply.push(gogi_toy::END);
    script.set(prompt(&value), reply);
    let cut = generate(&f, &value, 2, json!({"temperature": 0}));
    let seam = cut.snapshots.last().unwrap().tokens;
    let resumed = with_open(&value, serde_json::to_value(cut.as_open_message().unwrap()).unwrap());
    let done = generate(&f, &resumed, 10, json!({"temperature": 0}));
    assert_eq!((done.content.as_str(), done.usage.kept), ("okay", seam));
    let mut messages = opening();
    messages.push(serde_json::to_value(done.as_message().unwrap()).unwrap());
    messages.push(json!({"role": "user", "content": "Go on.", "snap": true}));
    let dry = f
        .service
        .build(&context(&ctx(false, messages)), true, Default::default())
        .unwrap();
    let head = done.snapshots.last().unwrap();
    assert_eq!(done.head.as_ref(), Some(&head.id));
    assert_eq!(
        (dry.from, dry.usage.kept),
        (done.head.clone(), head.tokens),
        "{:?}",
        dry.usage
    );
}

/// kaibo job-5 item 3: the generation of a cut reply started from a cut, the unmarked end of an earlier reply. The
/// head chains through that cut and through the end of the context before the reply. A context that sends the cut
/// reply back closed, with neither end marked, still starts from the head.
#[test]
fn a_cut_reply_whose_generation_started_from_a_cut_goes_back_closed_and_starts_from_its_head() {
    let (f, script) = scripted();
    let value = ctx(false, opening());
    let mut first = bytes("ok");
    first.push(gogi_toy::END);
    script.set(prompt(&value), first);
    let reply = generate(&f, &value, 10, json!({"temperature": 0}));
    let mut messages = opening();
    messages.push(serde_json::to_value(reply.as_message().unwrap()).unwrap());
    messages.push(json!({"role": "user", "content": "Again."}));
    let again = ctx(false, messages.clone());
    let mut second = bytes("okay");
    second.push(gogi_toy::END);
    script.set(prompt(&again), second);
    let cut = generate(&f, &again, 3, json!({"temperature": 0}));
    assert_eq!(
        cut.usage.kept,
        reply.snapshots.last().unwrap().tokens,
        "the generation started from the cut"
    );
    let head = cut.head.clone().expect("a cut reply holds a sealed head");
    let tokens = cut.snapshots.last().unwrap().tokens;
    messages.push(serde_json::to_value(cut.as_message().unwrap()).unwrap());
    messages.push(json!({"role": "user", "content": "Go on."}));
    let dry = f
        .service
        .build(&context(&ctx(false, messages.clone())), true, Default::default())
        .unwrap();
    assert_eq!(
        (dry.from, dry.usage.kept),
        (Some(head.clone()), tokens),
        "{:?}",
        dry.usage
    );
    // A build from the head lists no snapshot at the cuts that it chains through.
    let built = f
        .service
        .build(&context(&ctx(false, messages)), false, Default::default())
        .unwrap();
    assert_eq!(built.usage.kept, tokens);
    assert!(
        built.snapshots.iter().all(|s| s.message == 1 || s.message == 5),
        "{:?}",
        built.snapshots
    );
}

/// kaibo job-5 item 4: a cut chain must not hide a seam held on another chain. The cut reply's generation started
/// from the context's own boundary. Later the store holds a snapshot at the unmarked end of the earlier reply. A
/// continuation still starts from the head.
#[test]
fn a_held_cut_does_not_hide_the_head_of_a_cut_reply() {
    let (f, script) = scripted();
    let value = ctx(false, opening());
    let mut messages = opening();
    messages.push(json!({"role": "assistant", "content": "ok"}));
    messages.push(json!({"role": "user", "content": "Again.", "snap": true}));
    let again = ctx(false, messages);
    let mut second = bytes("okay");
    second.push(gogi_toy::END);
    script.set(prompt(&again), second);
    let cut = generate(&f, &again, 3, json!({"temperature": 0}));
    let tokens = cut.snapshots.last().unwrap().tokens;
    // A generation of the opening writes the earlier reply, so the store holds a snapshot at its unmarked end.
    let mut first = bytes("ok");
    first.push(gogi_toy::END);
    script.set(prompt(&value), first);
    let reply = generate(&f, &value, 10, json!({"temperature": 0}));
    assert_eq!(reply.content, "ok");
    script.set(prompt(&again), {
        let mut second = bytes("okay");
        second.push(gogi_toy::END);
        second
    });
    let resumed = with_open(&again, serde_json::to_value(cut.as_open_message().unwrap()).unwrap());
    let dry = f.service.build(&context(&resumed), true, Default::default()).unwrap();
    assert_eq!(
        (dry.from, dry.usage.kept),
        (cut.head.clone(), tokens),
        "{:?}",
        dry.usage
    );
    let done = generate(&f, &resumed, 10, json!({"temperature": 0}));
    assert_eq!((done.usage.kept, done.usage.fed), (tokens, 1));
}

/// kaibo job-5 item 6: a dry run of an open context needs a slot for the whole open rendering, as its generation does.
/// When no slot holds it, both are a 413 that names context_tokens.
#[test]
fn a_dry_run_of_an_open_context_that_no_slot_holds_is_too_large() {
    let f = Fixture::new();
    let open = json!({"role": "assistant", "content": "x".repeat(3000), "continue": true});
    let value = with_open(&ctx(false, opening()), open);
    let generation = f
        .service
        .generate(&request(&value, 4, json!({"temperature": 0})), None)
        .unwrap_err();
    assert!(
        matches!(&generation, ServiceError::TooLarge { param, .. } if param == "context_tokens"),
        "{generation:?}"
    );
    let dry = f.service.build(&context(&value), true, Default::default()).unwrap_err();
    assert!(
        matches!(&dry, ServiceError::TooLarge { param, .. } if param == "context_tokens"),
        "{dry:?}"
    );
}
