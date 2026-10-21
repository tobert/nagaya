//! The feed rule on the job path (spec 6.14, engine-abi D11). The engine is the fake with a feed rule: its bits
//! depend on the token count of a pass. The oracle feeds each span by the rule, with no process.

mod feed_support;

use feed_support::{Direct, FeedFixture, rendering};
use gogi_proto::context::Context;
use gogi_proto::generation::GenerationRequest;
use nagaya::council::snapshot_id;
use nagaya::park::identity_digest;
use nagaya_abi::Engine;
use nagaya_store::{EngineFacts, Put, Reason, Tier};
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

#[test]
fn a_generation_holds_no_head_so_the_next_build_feeds_the_reply_by_the_rule() {
    let fixture = FeedFixture::new(1);
    let opening = vec![system(), first()];
    let request: GenerationRequest = serde_json::from_value(json!({
        "model": "toy",
        "context": {"thinking": false, "tools": [], "messages": opening},
        "max_tokens": 12,
        "sample": {"temperature": 1.0, "seed": 3},
    }))
    .unwrap();
    let done = fixture.service.generate(&request, None).unwrap();
    let reply = done.as_message().expect("the reply renders as a message");
    let mut messages: Vec<Value> = opening.clone();
    messages.push(serde_json::to_value(&reply).unwrap());
    let extended = context(&messages);
    let built = fixture.service.build(&extended, false, Default::default()).unwrap();

    // The store keeps one content id for the recipe of the head. A snapshot of the state that the rule makes must
    // join it. A head with the rows of decode quanta would fire the tripwire here (spec 6.7).
    let (tokens, boundaries) = rendering(&extended);
    let mut direct = Direct::by_rule(&tokens, &boundaries);
    let snapshot = direct.snapshot();
    let identity = identity_digest(&direct.engine.identity());
    let ports = fixture.process.ports();
    let id = ports
        .store()
        .unwrap()
        .lock()
        .unwrap()
        .put(Put {
            snapshot: &snapshot,
            recipe: built.head_recipe,
            previous_recipe: None,
            engine: EngineFacts::of(feed_support::ENGINE, &direct.engine.limits()),
            identity,
            owner: 1,
            tier: Tier::Host,
            reason: Reason::PrefillDone,
            sealed_recipe: None,
            provenance: None,
            now_ms: 10_000_000,
        })
        .unwrap();
    assert_eq!(built.head, Some(snapshot_id(&id)));
    assert_eq!(
        done.head, None,
        "a generation on an engine with a feed rule holds no head (spec 6.14)"
    );
}
