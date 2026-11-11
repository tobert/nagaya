//! The feed rule on the job path (spec 6.14, engine-abi D11). The engine is the fake with a feed rule: its bits
//! depend on the token count of a pass. The oracle feeds each span by the rule, with no process.
//!
//! A generation holds a decoded head after its reply (spec 6.14, "The decoded head"). The oracle of a decoded head
//! feeds the prompt by the rule, and then each later token in a pass alone.

mod feed_support;

use std::sync::Arc;

use feed_support::{BLOCK, Direct, FeedFactory, FeedFixture, rendering};
use gogi_proto::ContextId;
use gogi_proto::context::{Context, Tool};
use gogi_proto::generation::{Finish, GenerationDone, GenerationRequest, Phase};
use gogi_proto::ids::SnapshotId;
use gogi_proto::message::Role;
use nagaya::council::snapshot_id;
use nagaya::park::identity_digest;
use nagaya_abi::Engine;
use nagaya_council::family::{
    Boundary, DecisionInput, Family, FamilyError, ParsedReply, RawRead, ReadPlan, Rendering, ReplyParser, ReplyStart,
    ReplyStep,
};
use nagaya_store::{ContentId, EngineFacts, Put, Reason, RecipeId, Tier};
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};

fn context(messages: &[Value]) -> Context {
    serde_json::from_value(json!({"thinking": false, "tools": [], "messages": messages})).unwrap()
}

fn system() -> Value {
    json!({"role": "system", "content": "You are a careful agent. Read the files first."})
}

fn first() -> Value {
    json!({"role": "user", "content": "List the crates of the workspace, with one line each.", "snap": true})
}

fn second() -> Value {
    json!({"role": "user", "content": "Now name the crate that owns the ledger.", "snap": true})
}

/// The text of a user message that is not a boundary.
const THIRD: &str = "Thanks.";

fn third() -> Value {
    json!({"role": "user", "content": THIRD})
}

#[test]
fn a_recipe_built_in_two_request_shapes_has_one_content_id() {
    let whole = context(&[system(), first(), second()]);
    let one = FeedFixture::new(1);
    let at_once = one.service.build(&whole, false, Default::default()).unwrap();

    let two = FeedFixture::new(1);
    let prefix = two
        .service
        .build(&context(&[system(), first()]), false, Default::default())
        .unwrap();
    let continued = two.service.build(&whole, false, Default::default()).unwrap();
    assert_eq!(
        continued.from, prefix.head,
        "the second shape starts from the held prefix"
    );
    assert!(continued.usage.kept > 0);

    assert_eq!(
        continued.head, at_once.head,
        "both shapes hold one content id at the head"
    );
    let (tokens, boundaries) = rendering(&whole);
    assert!(boundaries.len() >= 2 && tokens.len() as u64 > 2 * feed_support::BLOCK);
    let expected = Direct::by_rule(&tokens, &boundaries).content_id();
    assert_eq!(
        at_once.head,
        Some(snapshot_id(&expected)),
        "the job path feeds each span by the rule"
    );
}

/// The token that [`LikeLfm3`] puts after the end of each closed `assistant` message: a newline, as LFM3's template
/// does.
const NEWLINE: u32 = 10;

/// A habit of [`LikeLfm3`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Habit {
    /// The template puts a newline after the end token of each closed `assistant` message, as LFM3's does.
    Newline,
    /// The newline, and the reply parser removes the last character of the content, as LFM3's removes white space.
    Trim,
    /// The template puts the newline only after an `assistant` message that ends the context. Thus the rendering of
    /// a longer context does not start with the rendering of a context that ends with a reply.
    NewlineAtTheEnd,
}

/// The toy family with a habit of LFM3's family (spec 6.14, "The closing tokens", "No head when the tokens differ").
struct LikeLfm3 {
    toy: ToyFamily,
    habit: Habit,
}

impl LikeLfm3 {
    fn new(habit: Habit) -> Arc<Self> {
        Arc::new(Self {
            toy: ToyFamily::new(),
            habit,
        })
    }
}

/// A reply parser that removes the last character of the content when the reply ends.
struct Trim(Box<dyn ReplyParser>);

impl ReplyParser for Trim {
    fn push(&mut self, token: u32) -> Result<ReplyStep, FamilyError> {
        self.0.push(token)
    }

    fn ended(&self) -> bool {
        self.0.ended()
    }

    fn phase(&self) -> Phase {
        self.0.phase()
    }

    fn finish(self: Box<Self>, max_tokens: bool) -> ParsedReply {
        let mut parsed = self.0.finish(max_tokens);
        parsed.content.pop();
        parsed
    }
}

impl Family for LikeLfm3 {
    fn render(&self, context: &Context, generation_prompt: bool) -> Result<Rendering, FamilyError> {
        let mut rendering = self.toy.render(context, generation_prompt)?;
        let last = context.messages.len() - 1;
        // The end of each closed `assistant` message in the toy rendering.
        let ends: Vec<usize> = context
            .messages
            .iter()
            .enumerate()
            .filter(|(index, message)| message.role == Role::Assistant && !(message.open && *index == last))
            .filter(|(index, _)| self.habit != Habit::NewlineAtTheEnd || *index == last)
            .map(|(index, _)| {
                let prefix =
                    gogi_toy::render_tokens(&context.tools, context.thinking, &context.messages[..=index], false);
                assert!(
                    rendering.tokens.starts_with(&prefix),
                    "the toy template renders a message the same way in a longer context when thinking is false"
                );
                prefix.len()
            })
            .collect();
        let moved = |position: usize| position + ends.iter().filter(|end| **end <= position).count();
        rendering.end = moved(rendering.end);
        for boundary in &mut rendering.boundaries {
            boundary.tokens = moved(boundary.tokens);
        }
        for end in ends.iter().rev() {
            rendering.tokens.insert(*end, NEWLINE);
        }
        Ok(rendering)
    }

    fn decision_read(&self, input: &DecisionInput<'_>) -> Result<Box<dyn ReadPlan>, FamilyError> {
        self.toy.decision_read(input)
    }

    fn raw_read(&self, letters: &[String], suffixes: &[String]) -> Result<RawRead, FamilyError> {
        self.toy.raw_read(letters, suffixes)
    }

    fn reply_parser(&self, start: ReplyStart, tools: &[Tool]) -> Result<Box<dyn ReplyParser>, FamilyError> {
        let parser = self.toy.reply_parser(start, tools)?;
        Ok(match self.habit {
            Habit::Trim => Box::new(Trim(parser)),
            _ => parser,
        })
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        self.toy.tokenize(text)
    }

    fn tokenize_special(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        self.toy.tokenize_special(text)
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, FamilyError> {
        self.toy.detokenize(tokens)
    }

    fn pieces(&self, tokens: &[u32]) -> Result<Vec<String>, FamilyError> {
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

/// A greedy generation request of `max_tokens` tokens at most after `context`. A greedy reply of the fake ends its
/// turn after about ten tokens (`speaking_bias`).
fn request(context: &Context, max_tokens: u32) -> GenerationRequest {
    sampled(context, max_tokens, 0.0, 0)
}

/// A generation request that samples at `temperature` with the seed `seed`.
fn sampled(context: &Context, max_tokens: u32, temperature: f64, seed: u64) -> GenerationRequest {
    serde_json::from_value(json!({
        "model": "toy",
        "context": context,
        "max_tokens": max_tokens,
        "sample": {"temperature": temperature, "seed": seed},
    }))
    .unwrap()
}

/// The context of the record `parent` with `messages` after it, as the route makes it whole (spec 11.13).
fn after(fixture: &FeedFixture, parent: &ContextId, messages: &[Value]) -> Context {
    let mut body = json!({"context": {"parent": parent.as_str(), "messages": messages}});
    fixture.service.resolve(&mut body, false).unwrap();
    serde_json::from_value(body["context"].clone()).unwrap()
}

/// The first turn: a reply that ends its turn after the system message and the first user message.
fn first_turn(fixture: &FeedFixture) -> (Context, GenerationDone) {
    let opening = context(&[system(), first()]);
    let done = fixture.service.generate(&request(&opening, 48), None).unwrap();
    assert_eq!(
        done.finish,
        Finish::EndTurn,
        "the reply of the test ends its turn: {done:?}"
    );
    (opening, done)
}

/// The decoded head of a reply to `prompt`, fed with no process: the prompt by the rule, with its last span from its
/// last boundary to its end, and then each token of `decoded` in a pass alone.
fn decoded(prompt: &Rendering, decoded: &[u32]) -> Direct {
    let mut seals = prompt.boundaries.clone();
    seals.push(Boundary {
        message: 0,
        tokens: prompt.tokens.len(),
    });
    let mut direct = Direct::by_rule(&prompt.tokens, &seals);
    for token in decoded {
        direct.pass(&[*token]);
    }
    direct
}

/// Feed `tokens` after the rows of `direct` as one span, by the rule.
fn feed_span(direct: &mut Direct, tokens: &[u32]) {
    for chunk in tokens.chunks(BLOCK as usize) {
        direct.pass(chunk);
    }
}

fn content_id(id: &SnapshotId) -> ContentId {
    ContentId::from_hex(id.as_str().strip_prefix("blake3:").unwrap()).unwrap()
}

/// Put the state of `direct` in the store of `fixture` under `recipe`, as a build does. A recipe that holds other
/// bytes fires the tripwire (spec 6.7), and the put is an error.
fn put_by_rule(fixture: &FeedFixture, direct: &mut Direct, recipe: RecipeId) -> ContentId {
    let snapshot = direct.snapshot();
    let identity = identity_digest(&direct.engine.identity());
    let ports = fixture.process.ports();
    ports
        .store()
        .unwrap()
        .lock()
        .unwrap()
        .put(Put {
            snapshot: &snapshot,
            recipe,
            previous_recipe: None,
            engine: EngineFacts::of(feed_support::ENGINE, &direct.engine.limits()),
            identity,
            owner: 1,
            tier: Tier::Host,
            reason: Reason::PrefillDone,
            sealed_recipe: None,
            chains_from: None,
            provenance: None,
            now_ms: 10_000_000,
        })
        .expect("the recipe of the rule holds no other bytes")
}

/// Decided 2026-11-11 (spec 6.14, "The decoded head"): a generation on an engine with a feed rule holds a head after
/// its reply. This replaces the test of the rule of 2026-10-21, which checked a null head.
///
/// Falsified by a head under the recipe of the rule: the put of the rule's state fires the tripwire. Also falsified
/// by a feed of the two closing tokens in one pass: the head has other bits than the oracle.
#[test]
fn a_generation_holds_a_decoded_head_under_a_recipe_of_its_own() {
    let family = LikeLfm3::new(Habit::Newline);
    let fixture = FeedFixture::with_family(family.clone());
    let (opening, done) = first_turn(&fixture);
    let head = done
        .head
        .clone()
        .expect("a generation holds a decoded head (spec 6.14)");

    let prompt = family.render(&opening, true).unwrap();
    let mut extended = opening.clone();
    extended
        .messages
        .push(done.as_message().expect("the reply renders as a message"));
    let closed = family.render(&extended, false).unwrap();
    let start = prompt.tokens.len();
    assert_eq!(closed.tokens[..start], prompt.tokens[..]);
    assert_eq!(
        closed.end - start,
        done.usage.completion as usize + 1,
        "the head has the sampled tokens and the closing token of the template"
    );
    assert_eq!(closed.tokens[closed.end - 1], NEWLINE);
    let mut oracle = decoded(&prompt, &closed.tokens[start..closed.end]);
    assert_eq!(
        head,
        snapshot_id(&oracle.content_id()),
        "the head keeps the decode rows, and each token after the prompt had a pass alone"
    );

    // A build of the context with its reply starts from the head, and it feeds nothing.
    let built = fixture.service.build(&extended, false, Default::default()).unwrap();
    assert_eq!(built.from, Some(head.clone()));
    assert_eq!(built.head, Some(head.clone()));
    assert_eq!((built.usage.kept, built.usage.fed), (closed.end as u64, 0));

    // The recipe of the same tokens fed by the rule is another recipe. The state that the rule makes goes under it
    // with no tripwire (spec 6.7), and it has other bits.
    let mut by_rule = Direct::by_rule(&closed.tokens[..closed.end], &closed.boundaries);
    let id = put_by_rule(&fixture, &mut by_rule, built.head_recipe);
    assert_ne!(snapshot_id(&id), head);
}

/// Falsified by a build that does not look for the head of the record: the turn feeds the reply again.
#[test]
fn the_second_turn_by_parent_feeds_only_its_new_message() {
    let fixture = FeedFixture::new(1);
    let toy = ToyFamily::new();
    let (_, one) = first_turn(&fixture);
    let parent = one.context.clone().expect("a reply names its record");
    let head_tokens = toy.render(&after(&fixture, &parent, &[]), false).unwrap().end as u64;

    let turn = after(&fixture, &parent, &[second()]);
    let two = fixture.service.generate(&request(&turn, 48), None).unwrap();
    let prompt = toy.render(&turn, true).unwrap().tokens.len() as u64;
    assert_eq!(two.usage.prompt, prompt);
    assert_eq!(two.usage.kept, head_tokens, "the turn starts from the decoded head");
    // The user token, the text and the end token of the new message, then the one token of the generation prompt.
    let new_message = second()["content"].as_str().unwrap().len() as u64 + 2;
    assert_eq!(two.usage.fed, new_message + 1);
    assert_eq!(two.finish, Finish::EndTurn, "{two:?}");
    let second_head = two.head.clone().expect("the second reply holds a decoded head too");

    // The same turn sent whole, with the reply as a client appends it, finds the same head. The turn before it held
    // the boundary at the end of the new message, and that boundary chains from the head. So this turn starts there.
    let mut whole = context(&[system(), first()]);
    whole.messages.push(one.as_message().unwrap());
    whole.messages.push(serde_json::from_value(second()).unwrap());
    assert!(!whole.messages[2].snap);
    let again = fixture.service.generate(&request(&whole, 48), None).unwrap();
    assert_eq!(again.usage.kept, head_tokens + new_message);
    assert_eq!(again.content, two.content);
    assert_eq!(again.head, two.head);

    // The third turn starts from the head of the second reply. Its message is not a boundary, so the span from the
    // head holds the message and the generation prompt.
    let parent = two.context.clone().unwrap();
    let second_tokens = toy.render(&after(&fixture, &parent, &[]), false).unwrap().end as u64;
    let turn = after(&fixture, &parent, &[third()]);
    let three = fixture.service.generate(&request(&turn, 48), None).unwrap();
    assert_eq!(three.usage.kept, second_tokens);
    assert_eq!(three.usage.fed, THIRD.len() as u64 + 2 + 1);

    // The head of the second reply chains from the head of the first: the prompt of the second reply by the rule from
    // the first head, with a boundary at the end of its user message, then each token of the reply alone.
    let first_prompt = toy.render(&context(&[system(), first()]), true).unwrap();
    let closed = toy.render(&after(&fixture, &parent, &[]), false).unwrap();
    let tokens = &closed.tokens[..closed.end];
    let first_end = head_tokens as usize;
    let mut oracle = decoded(&first_prompt, &tokens[first_prompt.tokens.len()..first_end]);
    let message_end = first_end + new_message as usize;
    feed_span(&mut oracle, &tokens[first_end..message_end]);
    feed_span(&mut oracle, &tokens[message_end..message_end + 1]);
    for token in &tokens[message_end + 1..] {
        oracle.pass(&[*token]);
    }
    assert_eq!(second_head, snapshot_id(&oracle.content_id()));
}

/// Falsified by an engine whose rows depend on a count of all its passes: the second run has other bits.
#[test]
fn the_same_request_from_the_same_head_gives_the_same_bits_twice() {
    let fixture = FeedFixture::new(1);
    let (_, one) = first_turn(&fixture);
    let turn = after(&fixture, one.context.as_ref().unwrap(), &[second()]);
    // The reply is sampled with a seed. It must end its turn to hold a head.
    let run = || fixture.service.generate(&sampled(&turn, 400, 0.5, 4), None).unwrap();
    let (a, b) = (run(), run());
    assert_eq!(a.finish, Finish::EndTurn, "the reply of the test ends its turn: {a:?}");
    assert!(a.usage.kept > 0 && a.head.is_some());
    // The first run starts from the head of the first reply. It holds the boundary at the end of the new message,
    // which chains from that head, and the second run starts there.
    let new_message = second()["content"].as_str().unwrap().len() as u64 + 2;
    assert_eq!(b.usage.kept, a.usage.kept + new_message);
    assert_eq!((&a.content, &a.reasoning), (&b.content, &b.reasoning));
    // The id of a head is the hash of its bytes, so equal ids are equal bits of the whole state.
    assert_eq!(a.head, b.head);
    assert_eq!(a.context, b.context);
}

/// A turn from a decoded head and the same context built from its text are two readings of one context (spec 6.14,
/// "The bits differ a little"). Both run. On the fake, their snapshots differ.
///
/// Falsified by a generation that holds no decoded head: the two builds hold one snapshot.
#[test]
fn a_build_from_a_decoded_head_and_a_build_from_the_text_both_run_and_hold_other_snapshots() {
    let toy = ToyFamily::new();
    let held = FeedFixture::new(1);
    let (opening, one) = first_turn(&held);
    let turn = after(&held, one.context.as_ref().unwrap(), &[second()]);
    let from_head = held.service.build(&turn, false, Default::default()).unwrap();
    assert_eq!(from_head.from, one.head);

    let fresh = FeedFixture::new(1);
    let from_text = fresh.service.build(&turn, false, Default::default()).unwrap();
    assert_eq!((from_text.from.clone(), from_text.usage.kept), (None, 0));
    let (tokens, boundaries) = rendering(&turn);
    assert_eq!(
        from_text.head,
        Some(snapshot_id(&Direct::by_rule(&tokens, &boundaries).content_id())),
        "the build from the text feeds each span by the rule"
    );

    // The build from the head feeds the rest by the rule, as a new span that starts at the head.
    let prompt = toy.render(&opening, true).unwrap();
    let head_end = from_head.usage.kept as usize;
    let mut oracle = decoded(&prompt, &tokens[prompt.tokens.len()..head_end]);
    feed_span(&mut oracle, &tokens[head_end..]);
    assert_eq!(from_head.head, Some(snapshot_id(&oracle.content_id())));
    assert_ne!(
        from_head.head, from_text.head,
        "the rows of the reply differ: decode quanta made them in one build, and the rule in the other"
    );

    // A generation runs on both.
    for fixture in [&held, &fresh] {
        let done = fixture.service.generate(&request(&turn, 48), None).unwrap();
        assert!(done.usage.completion > 0);
    }
}

/// Falsified by a head that the job holds when the slot does not hold the tokens of the closed rendering.
#[test]
fn a_trimmed_reply_holds_no_decoded_head_and_the_next_turn_feeds_the_reply_by_the_rule() {
    let family = LikeLfm3::new(Habit::Trim);
    let fixture = FeedFixture::with_family(family.clone());
    let (opening, one) = first_turn(&fixture);
    assert!(!one.content.is_empty(), "the test needs a reply with content: {one:?}");
    assert_eq!(one.head, None, "the record has other tokens than the model sampled");

    let turn = after(&fixture, one.context.as_ref().unwrap(), &[second()]);
    let two = fixture.service.generate(&request(&turn, 48), None).unwrap();
    let boundary = family.render(&opening, false).unwrap().end as u64;
    assert_eq!(
        two.usage.kept, boundary,
        "the turn starts from the boundary before the reply"
    );

    // The reply is fed by the rule: the state at the end of the turn's context is the state of the oracle.
    let built = fixture.service.build(&turn, false, Default::default()).unwrap();
    let closed = family.render(&turn, false).unwrap();
    let expected = Direct::by_rule(&closed.tokens[..closed.end], &closed.boundaries).content_id();
    assert_eq!(built.head, Some(snapshot_id(&expected)));
}

/// Falsified by a build that takes the cut of a head that the store does not hold: it puts the state of the rule
/// under the recipe of the decoded head.
#[test]
fn an_evicted_head_is_a_miss_and_the_build_feeds_the_reply_by_the_rule() {
    let fixture = FeedFixture::new(1);
    let toy = ToyFamily::new();
    let (_, one) = first_turn(&fixture);
    let head = content_id(one.head.as_ref().unwrap());
    let store = Arc::clone(fixture.process.ports().store().unwrap());
    // Eviction takes the oldest entry first, so the test evicts each entry: the boundary of the prompt and the head.
    {
        let mut store = store.lock().unwrap();
        let bytes: u64 = store.entries().values().map(|entry| entry.size).sum();
        let gone = store.evict(1, Tier::Host, bytes, 20_000_000).unwrap();
        assert!(gone.contains(&head), "{gone:?}");
    }

    let parent = one.context.clone().unwrap();
    let turn = after(&fixture, &parent, &[second()]);
    let two = fixture.service.generate(&request(&turn, 48), None).unwrap();
    assert_eq!(
        two.usage.kept, 0,
        "the store holds no prefix, so the turn feeds its whole context"
    );
    assert_eq!(two.usage.fed, toy.render(&turn, true).unwrap().tokens.len() as u64);
    assert_eq!(two.finish, Finish::EndTurn);

    // The record marks the reply as a boundary. The build held the state of the rule there, under the recipe of the
    // rule: the recipe that a dry run of the context with the reply names.
    let replied = after(&fixture, &parent, &[]);
    let at_reply = two
        .snapshots
        .iter()
        .find(|snapshot| snapshot.message == 2)
        .expect("the build holds a snapshot at the end of the reply");
    let (tokens, boundaries) = rendering(&replied);
    assert_eq!(
        at_reply.id,
        snapshot_id(&Direct::by_rule(&tokens, &boundaries).content_id())
    );
    let plan = fixture.service.build(&replied, true, Default::default()).unwrap();
    let recipe = store.lock().unwrap().entries()[&content_id(&at_reply.id)].recipe;
    assert_eq!(recipe, plan.head_recipe);
}

/// A template can render a reply in one way at the end of a context, and in another way before a later message. Then
/// the rendering of the next turn does not start with the tokens of the head, and the head is a miss.
///
/// Falsified by a build that takes the note of a head with no check of its recipe.
#[test]
fn a_head_whose_tokens_do_not_start_the_next_rendering_is_a_miss() {
    let family = LikeLfm3::new(Habit::NewlineAtTheEnd);
    let fixture = FeedFixture::with_family(family.clone());
    let (opening, one) = first_turn(&fixture);
    assert!(one.head.is_some(), "the reply holds a head with the newline after it");

    let parent = one.context.clone().unwrap();
    let replied = family.render(&after(&fixture, &parent, &[]), false).unwrap();
    let turn = after(&fixture, &parent, &[second()]);
    let next = family.render(&turn, true).unwrap();
    assert_ne!(next.tokens[..replied.end], replied.tokens[..replied.end]);
    let two = fixture.service.generate(&request(&turn, 48), None).unwrap();
    let boundary = family.render(&opening, false).unwrap().end as u64;
    assert_eq!(
        two.usage.kept, boundary,
        "the turn starts from the boundary before the reply"
    );
}

/// A reply whose decode rows shared a pass with the rows of another job holds no decoded head (spec 6.14, "No head
/// from a shared pass"). The fake's bits depend on the token count of a pass, so the rows of a shared pass do not
/// follow from the recipe.
///
/// Falsified by a driver that does not tell a job of a shared pass: both replies hold a head.
#[test]
fn a_reply_whose_decode_rows_shared_a_pass_holds_no_decoded_head() {
    // Each pass takes 3 ms, so the two replies are in flight together.
    let factory = FeedFactory::new(1);
    factory.delay_ms.store(3, std::sync::atomic::Ordering::SeqCst);
    let fixture = FeedFixture::with(factory, std::time::Duration::from_secs(10));
    let contexts = [context(&[system(), first()]), context(&[system(), second()])];
    let replies: Vec<GenerationDone> = std::thread::scope(|scope| {
        let runs: Vec<_> = contexts
            .iter()
            .map(|context| scope.spawn(|| fixture.service.generate(&request(context, 48), None).unwrap()))
            .collect();
        runs.into_iter().map(|run| run.join().unwrap()).collect()
    });
    let sizes = fixture.factory.sizes.lock().unwrap().clone();
    assert!(sizes.contains(&2), "no pass had the rows of both replies: {sizes:?}");
    for reply in &replies {
        assert_eq!(reply.head, None, "{reply:?}");
    }
    // A reply alone on the same nagaya holds its head.
    let alone = fixture
        .service
        .generate(&request(&context(&[system(), third()]), 48), None)
        .unwrap();
    assert!(alone.head.is_some(), "{alone:?}");
}

/// A reply that ends with `max_tokens` holds a decoded head at the seam of its open rendering, the position before
/// its last token (spec 6.14, "A cut reply", and 11.3c). The record of the open context names it.
///
/// Falsified by a head of a cut reply at the end of its open rendering: the continuation finds no head before its
/// last token.
#[test]
fn a_cut_reply_holds_a_decoded_head_and_its_continuation_starts_from_it() {
    let fixture = FeedFixture::new(1);
    let toy = ToyFamily::new();
    let opening = context(&[system(), first()]);
    let cut = fixture.service.generate(&sampled(&opening, 6, 1.0, 3), None).unwrap();
    assert_eq!(cut.finish, Finish::MaxTokens, "{cut:?}");
    let head = cut.head.clone().expect("a cut reply holds a decoded head");

    let open = after(&fixture, cut.context.as_ref().unwrap(), &[]);
    assert!(
        open.open().is_some(),
        "the record of a cut reply ends with an open message"
    );
    let prompt = toy.render(&opening, true).unwrap();
    let rendering = toy.render(&open, true).unwrap();
    let seam = rendering.tokens.len() - 1;
    assert_eq!(
        seam,
        prompt.tokens.len() + 5,
        "the slot holds each sampled token but the last"
    );
    let mut oracle = decoded(&prompt, &rendering.tokens[prompt.tokens.len()..seam]);
    assert_eq!(head, snapshot_id(&oracle.content_id()));

    // The continuation starts from the head, and it feeds the last token of the open text for its first logit row.
    let more = fixture.service.generate(&request(&open, 48), None).unwrap();
    assert_eq!((more.usage.kept, more.usage.fed), (seam as u64, 1));
    assert!(more.content.starts_with(&cut.content) && more.content.len() > cut.content.len());
    assert_eq!(more.finish, Finish::EndTurn, "{more:?}");

    // The head of the whole reply chains from the head of the cut reply. The span after that head is one token, and
    // each later token had a pass alone.
    let closed = toy
        .render(&after(&fixture, more.context.as_ref().unwrap(), &[]), false)
        .unwrap();
    assert_eq!(closed.tokens[..seam], rendering.tokens[..seam]);
    for token in &closed.tokens[seam..closed.end] {
        oracle.pass(&[*token]);
    }
    assert_eq!(more.head, Some(snapshot_id(&oracle.content_id())));
}
