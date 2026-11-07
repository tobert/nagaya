//! The head of a cut reply (spec 11.3c, "Sealed head") with the Qwen3.8 family: its real tokenizer and chat template
//! from the GGUF, on the fake engine. The engine is scripted, so a test knows the reply. No GPU runs.

mod council_support;
#[path = "../../nagaya-text-qwen38/tests/common/model.rs"]
mod qwen_model;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use council_support::{Fixture, Script, TermFactory};
use gogi_proto::context::{Context, Tool};
use gogi_proto::generation::{Finish, GenerationDone, GenerationRequest, Phase};
use nagaya::council::ServiceConfig;
use nagaya_abi::SlotLimit;
use nagaya_council::family::{
    DecisionInput, Family, FamilyError, RawRead, ReadPlan, Rendering, ReplyParser, ReplyStart,
};
use nagaya_text_qwen38::Qwen38Family;
use serde_json::{Value, json};

/// The Qwen3.8 family. A test that calls it has called `qwen_model::absent` before (AGENTS.md, "Tests that need a
/// model file").
fn qwen() -> Arc<Qwen38Family> {
    Arc::new(Qwen38Family::from_gguf(&qwen_model::gguf_path()).expect("the family builds from the GGUF"))
}

/// A family that counts the renders of the family inside it.
struct Counting {
    inner: Arc<dyn Family>,
    renders: Arc<AtomicUsize>,
}

impl Family for Counting {
    fn render(&self, context: &Context, generation_prompt: bool) -> Result<Rendering, FamilyError> {
        self.renders.fetch_add(1, Ordering::SeqCst);
        self.inner.render(context, generation_prompt)
    }

    fn decision_read(&self, input: &DecisionInput<'_>) -> Result<Box<dyn ReadPlan>, FamilyError> {
        self.inner.decision_read(input)
    }

    fn raw_read(&self, letters: &[String], suffixes: &[String]) -> Result<RawRead, FamilyError> {
        self.inner.raw_read(letters, suffixes)
    }

    fn reply_parser(&self, start: ReplyStart, tools: &[Tool]) -> Result<Box<dyn ReplyParser>, FamilyError> {
        self.inner.reply_parser(start, tools)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        self.inner.tokenize(text)
    }

    fn tokenize_special(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        self.inner.tokenize_special(text)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, FamilyError> {
        self.inner.detokenize(tokens)
    }

    fn pieces(&self, tokens: &[u32]) -> Result<Vec<String>, FamilyError> {
        self.inner.pieces(tokens)
    }

    fn template_id(&self) -> &str {
        self.inner.template_id()
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn thinks(&self) -> bool {
        self.inner.thinks()
    }
}

/// A fixture with `family` on a scripted fake engine over its vocabulary, with slots of 32768 tokens.
fn fixture(family: Arc<dyn Family>) -> (Fixture, Script) {
    let script = Script::default();
    let factory = TermFactory {
        script: script.clone(),
        vocab: Some(family.vocab_size()),
        slots: vec![SlotLimit {
            bytes: 8 * 32768,
            count: 2,
            tokens: 32768,
        }],
        ..TermFactory::toy()
    };
    let mut config = ServiceConfig::new("qwen", council_support::ENGINE);
    config.context_tokens = 32768;
    (Fixture::with_family(factory, family, 1 << 28, None, config), script)
}

fn context(value: &Value) -> Context {
    serde_json::from_value(value.clone()).unwrap()
}

fn generate(f: &Fixture, value: &Value, max_tokens: u32) -> GenerationDone {
    let request: GenerationRequest = serde_json::from_value(
        json!({"model": "qwen", "context": value, "max_tokens": max_tokens, "sample": {"temperature": 0}}),
    )
    .unwrap();
    f.service.generate(&request, None).unwrap()
}

const QUESTION: &str = "In two sentences, why do snapshots of model state help an agent?";

/// senpai's live check on 8090: a reply cut inside its thinking, sent back closed with `content: ""` and then a user
/// message, fed all of the reply again (kept 19 of 76).
#[test]
fn a_reply_cut_inside_its_thinking_and_sent_back_closed_starts_from_its_head() {
    if qwen_model::absent() {
        return;
    }
    let family = qwen();
    let (f, script) = fixture(family.clone());
    let user = json!({"role": "user", "content": QUESTION, "snap": true});
    let value = json!({"thinking": true, "messages": [user]});
    let prompt = family.render(&context(&value), true).unwrap().tokens.len();
    let thought = "The user asks why snapshots help an agent. A snapshot keeps the model state, so the agent";
    let tokens = family.tokenize(thought).unwrap();
    script.set(prompt, tokens.clone());
    let cut = generate(&f, &value, tokens.len() as u32 - 2);
    assert_eq!((cut.finish, cut.phase), (Finish::MaxTokens, Some(Phase::Reasoning)));
    assert_eq!(cut.content, "");
    let head = cut.head.clone().expect("a cut reply holds a sealed head");
    let head_tokens = cut.snapshots.last().unwrap().tokens;
    let closed = json!({"role": "assistant", "reasoning": cut.reasoning, "content": ""});
    let next = json!({"thinking": true, "messages": [user, closed, {"role": "user", "content": "Thanks."}]});
    let dry = f.service.build(&context(&next), true, Default::default()).unwrap();
    assert_eq!(
        (dry.from, dry.usage.kept),
        (Some(head.clone()), head_tokens),
        "{:?}",
        dry.usage
    );
    // The same reply sent back with no content member starts there too.
    let bare = json!({"role": "assistant", "reasoning": cut.reasoning});
    let next = json!({"thinking": true, "messages": [user, bare, {"role": "user", "content": "Thanks."}]});
    let dry = f.service.build(&context(&next), true, Default::default()).unwrap();
    assert_eq!((dry.from, dry.usage.kept), (Some(head), head_tokens), "{:?}", dry.usage);
}

/// kaibo job-5 (cost): a request renders its context once. The cuts come from the store's index, not from a second
/// render with every message marked. Each generation, read and decision renders once more for its own work.
#[test]
fn a_dry_run_renders_its_context_once() {
    let renders = Arc::new(AtomicUsize::new(0));
    let family = Arc::new(Counting {
        inner: Arc::new(nagaya_text_toy::ToyFamily::new()),
        renders: Arc::clone(&renders),
    });
    let (f, _) = fixture(family);
    let value = json!({"thinking": false, "messages": [
        {"role": "user", "content": "Read this.", "snap": true},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Summarize."}]});
    f.service.build(&context(&value), false, Default::default()).unwrap();
    renders.store(0, Ordering::SeqCst);
    f.service.build(&context(&value), true, Default::default()).unwrap();
    assert_eq!(renders.load(Ordering::SeqCst), 1);
}

/// senpai's timing on 8090: an 18k-token context with an assistant message took 132 ms for a dry run, and 42 ms
/// without it. The test prints the times of a host dry run, best of three, with the real tokenizer.
#[test]
fn a_long_context_with_an_assistant_message_costs_what_one_without_it_costs() {
    if qwen_model::absent() {
        return;
    }
    let family = qwen();
    let (f, _) = fixture(family.clone());
    let long = "The store keeps snapshots of model state by recipe. ".repeat(1800);
    let with = json!({"thinking": false, "messages": [
        {"role": "user", "content": long, "snap": true},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Summarize."}]});
    let without = json!({"thinking": false, "messages": [
        {"role": "user", "content": long, "snap": true},
        {"role": "user", "content": "Summarize."}]});
    let best = |value: &Value| {
        (0..3)
            .map(|_| {
                let started = Instant::now();
                f.service.build(&context(value), true, Default::default()).unwrap();
                started.elapsed()
            })
            .min()
            .unwrap()
    };
    let tokens = family.render(&context(&with), false).unwrap().end;
    let (a, b) = (best(&with), best(&without));
    eprintln!("dry run of {tokens} tokens: {a:?} with the assistant message, {b:?} without it");
    // The work is the same: two renders each. Allow for noise, but not for the three extra renders of a seam.
    assert!(a < b * 2 + Duration::from_millis(20), "{a:?} against {b:?}");
}

/// The lookup of held cuts reads only the entries that chain from the boundaries of the context. A store with 10,000
/// unrelated entries of the same engine, at tokens inside the context, costs a build no more than an empty store.
#[test]
fn ten_thousand_unrelated_entries_do_not_slow_a_build() {
    let (f, _) = fixture(Arc::new(nagaya_text_toy::ToyFamily::new()));
    let long = "Keep the build directory clean. ".repeat(250);
    let value = json!({"thinking": false, "messages": [
        {"role": "user", "content": long, "snap": true},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Summarize."}]});
    f.service.build(&context(&value), false, Default::default()).unwrap();
    let best = || {
        (0..3)
            .map(|_| {
                let started = Instant::now();
                f.service.build(&context(&value), true, Default::default()).unwrap();
                started.elapsed()
            })
            .min()
            .unwrap()
    };
    let empty = best();
    {
        let ports = f.process.ports();
        let mut store = ports.store().unwrap().lock().unwrap();
        let identity = store
            .entries()
            .values()
            .next()
            .expect("the build held a snapshot")
            .identity;
        store.fill_unrelated(10_000, council_support::ENGINE, identity, 100..7900);
    }
    let full = best();
    eprintln!("dry run: {empty:?} with an empty store, {full:?} with 10,000 unrelated entries");
    assert!(
        full < empty * 2 + Duration::from_millis(5),
        "{full:?} against {empty:?}"
    );
}

/// Build a cut reply with the Qwen3.8 family: the engine writes `reply` after the prompt, and the generation stops
/// `short` tokens before its end. Give the fixture, the context, the cut reply and its head's tokens.
fn cut_reply(reply: &str, short: usize) -> (Fixture, Value, GenerationDone, u64) {
    let family = qwen();
    let (f, script) = fixture(family.clone());
    let user = json!({"role": "user", "content": QUESTION, "snap": true});
    let value = json!({"thinking": true, "messages": [user]});
    let prompt = family.render(&context(&value), true).unwrap().tokens.len();
    let mut tokens = family.tokenize_special(reply).unwrap();
    tokens.push(family.tokenize_special("<|im_end|>").unwrap()[0]);
    script.set(prompt, tokens.clone());
    let cut = generate(&f, &value, (tokens.len() - short) as u32);
    assert_eq!(cut.finish, Finish::MaxTokens, "{cut:?}");
    let head = cut.snapshots.last().unwrap().tokens;
    assert_eq!(cut.head.as_ref(), Some(&cut.snapshots.last().unwrap().id));
    (f, value, cut, head)
}

/// The kept tokens of a dry run of the context, the cut reply sent back as `closed`, and one more user message.
fn kept_after(f: &Fixture, value: &Value, closed: Value) -> u64 {
    let mut next = value.clone();
    let messages = next["messages"].as_array_mut().unwrap();
    messages.push(closed);
    messages.push(json!({"role": "user", "content": "Thanks."}));
    f.service
        .build(&context(&next), true, Default::default())
        .unwrap()
        .usage
        .kept
}

/// kaibo job-5 item 2: the open field is rendered as sent, and the template trims the closed reasoning. A reasoning
/// cut that ends with white space still starts from its head, because the head is the seam before the last token.
#[test]
fn a_reasoning_cut_that_ends_with_white_space_goes_back_closed_in_both_forms() {
    if qwen_model::absent() {
        return;
    }
    let (f, value, cut, head) = cut_reply("First I list the files.\n\nThen", 2);
    assert_eq!(cut.phase, Some(Phase::Reasoning));
    assert!(cut.reasoning.ends_with(char::is_whitespace), "{:?}", cut.reasoning);
    let reasoning = cut.reasoning.clone();
    assert_eq!(
        kept_after(
            &f,
            &value,
            json!({"role": "assistant", "reasoning": reasoning, "content": ""})
        ),
        head
    );
    assert_eq!(
        kept_after(&f, &value, json!({"role": "assistant", "reasoning": reasoning})),
        head
    );
}

/// A content cut goes back closed with its reasoning, and starts from its head.
#[test]
fn a_content_cut_goes_back_closed_and_starts_from_its_head() {
    if qwen_model::absent() {
        return;
    }
    let (f, value, cut, head) = cut_reply(
        "I list the files.\n</think>\n\nSnapshots keep the state of the model.",
        4,
    );
    assert_eq!(cut.phase, Some(Phase::Content));
    let closed = json!({"role": "assistant", "reasoning": cut.reasoning, "content": cut.content});
    assert_eq!(kept_after(&f, &value, closed), head);
}

/// kaibo job-5 item 9: the Qwen3.8 template renders an empty system message as nothing, so its end is at token 0.
/// The cuts came from the ends of the messages, and a cut at token 0 was a 500. The search of the store's index gives
/// no cut there.
#[test]
fn a_message_that_renders_no_token_gives_no_cut_at_token_zero() {
    if qwen_model::absent() {
        return;
    }
    let (f, _) = fixture(qwen());
    let value =
        json!({"thinking": true, "messages": [{"role": "system", "content": ""}, {"role": "user", "content": "Hi."}]});
    f.service.build(&context(&value), false, Default::default()).unwrap();
    let mut next = value.clone();
    next["messages"]
        .as_array_mut()
        .unwrap()
        .push(json!({"role": "assistant", "content": "Hello."}));
    next["messages"]
        .as_array_mut()
        .unwrap()
        .push(json!({"role": "user", "content": "Bye."}));
    let dry = f.service.build(&context(&next), true, Default::default()).unwrap();
    assert!(dry.usage.kept > 0, "{:?}", dry.usage);
}

/// A marked message that renders no token is no boundary: a boundary at token 0 would hold an empty state.
#[test]
fn a_marked_message_that_renders_no_token_is_no_boundary() {
    if qwen_model::absent() {
        return;
    }
    let (f, _) = fixture(qwen());
    let value = json!({"thinking": true, "messages": [{"role": "system", "content": "", "snap": true}, {"role": "user", "content": "Hi."}]});
    let built = f.service.build(&context(&value), false, Default::default()).unwrap();
    assert!(built.snapshots.iter().all(|s| s.tokens > 0), "{:?}", built.snapshots);
}
