//! Several replies from one generation, through the router (gogi "Several replies", spec 11.15). The engine is the
//! fake engine, and the family is the toy family. gogi's suite checks the shape of the response
//! (`http_conformance.rs`). These tests check what the suite cannot see.

mod council_support;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::{ControlLog, ENGINE, Fixture, TermFactory};
use nagaya::council::ServiceConfig;
use nagaya::http::{Council, Served, router};
use serde_json::{Value, json};
use tower::ServiceExt;

/// A nagaya whose engine takes 20 ms for each pass, with `slots` slots. Thus a reply of 40 tokens takes 0.8 s or
/// more, and the time of a request shows if its replies ran together. The engine joins 4 segments in a pass, so the
/// decode quanta of jobs in flight join (spec 5.7). `log` gets the segment count of each pass.
fn slow(slots: u32, log: &ControlLog) -> Fixture {
    let factory = TermFactory {
        delay: Duration::from_millis(20),
        slots: council_support::roomy_slots(slots),
        segments: 4,
        log: log.clone(),
        ..TermFactory::toy()
    };
    Fixture::with_factory(factory, 1 << 20, None, ServiceConfig::new("toy", ENGINE))
}

fn council(f: &Fixture) -> Arc<Council> {
    let served = Served {
        service: f.attach_again(),
        aliases: Vec::new(),
        device: None,
    };
    Arc::new(Council::new(vec![served], None).unwrap())
}

/// Send one generation through the router. Give the status and the JSON body.
fn generate(council: &Arc<Council>, body: &Value) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method("POST")
            .uri("/council/v2/generations")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

fn context(text: &str) -> Value {
    json!({"thinking": false, "messages": [
        {"role": "user", "content": "We pack items into bins. Each bin holds 100. Use few bins."},
        {"role": "user", "content": text},
    ]})
}

/// A generation of 40 tokens at most after `context`, with the seed `seed`.
fn request(context: &Value, seed: u64) -> Value {
    json!({"model": "toy", "context": context, "max_tokens": 40, "sample": {"temperature": 1.0, "seed": seed}})
}

fn with(mut body: Value, member: &str, value: Value) -> Value {
    body[member] = value;
    body
}

/// The members of a reply that the same request gives alone.
fn reply_of(value: &Value) -> Vec<&Value> {
    ["content", "reasoning", "finish", "context", "seed"]
        .iter()
        .map(|member| &value[*member])
        .collect()
}

/// Falsified by a route that runs the replies in a row: the request takes the time of four generations.
#[test]
fn the_forks_of_one_request_are_in_flight_together_and_feed_the_prompt_once() {
    let log = ControlLog::default();
    let f = slow(4, &log);
    let council = council(&f);
    let plan = context("Write a plan.");
    let start = Instant::now();
    let (status, several) = generate(&council, &with(request(&plan, 3), "forks", json!(4)));
    let together = start.elapsed();
    assert_eq!(status, StatusCode::OK, "{several}");
    let replies = several["replies"].as_array().unwrap();
    assert_eq!(replies.len(), 4, "{several}");
    // The decode quanta of the four replies joined in one pass.
    let sizes = log.take().sizes;
    assert!(sizes.contains(&4), "no pass had the four replies: {sizes:?}");

    // The same four generations in a row, on a new nagaya with the same engine.
    let g = slow(4, &ControlLog::default());
    let alone = self::council(&g);
    let start = Instant::now();
    let singles: Vec<Value> = (0..4)
        .map(|i| {
            let (status, single) = generate(&alone, &request(&plan, 3 + i));
            assert_eq!(status, StatusCode::OK, "{single}");
            single
        })
        .collect();
    let in_a_row = start.elapsed();
    for (reply, single) in replies.iter().zip(&singles) {
        assert_eq!(reply_of(reply), reply_of(single));
    }

    // One fork fed the prompt, and the request counts it once.
    let prompt = singles[0]["usage"]["prompt"].as_u64().unwrap();
    assert_eq!(
        (
            &several["usage"]["prompt"],
            &several["usage"]["fed"],
            &several["usage"]["kept"]
        ),
        (&json!(prompt), &json!(prompt), &json!(0)),
        "{several}"
    );
    assert!(
        in_a_row >= Duration::from_millis(1600),
        "the replies were not long: {in_a_row:?}"
    );
    assert!(
        together < in_a_row / 2,
        "the forks ran in a row: {together:?} together, {in_a_row:?} in a row"
    );
}

/// Each reply of `contexts` reports its own `fed`, so the replies show the feed of the whole request. A reply that
/// keeps the prompt feeds its last token again for the logits, as a generation from a held head does.
///
/// Falsified by a generation that takes no claim in the book of 11.14: each reply feeds the whole prompt.
#[test]
fn equal_contexts_of_one_request_feed_their_tokens_once() {
    let f = slow(4, &ControlLog::default());
    let council = council(&f);
    let plan = context("Write a plan.");
    let mut body = request(&plan, 3);
    body.as_object_mut().unwrap().remove("context");
    let (status, several) = generate(&council, &with(body, "contexts", json!([plan, plan, plan, plan])));
    assert_eq!(status, StatusCode::OK, "{several}");
    let replies = several["replies"].as_array().unwrap();
    let prompt = replies[0]["usage"]["prompt"].as_u64().unwrap();
    let mut fed: Vec<u64> = replies
        .iter()
        .map(|reply| reply["usage"]["fed"].as_u64().unwrap())
        .collect();
    fed.sort();
    assert_eq!(
        fed,
        [1, 1, 1, prompt],
        "one reply feeds the context, and the others keep it"
    );
    assert_eq!(
        (&several["usage"]["prompt"], &several["usage"]["fed"]),
        (&json!(4 * prompt), &json!(prompt + 3)),
        "{}",
        several["usage"]
    );
}

/// A reply takes a slot as any job does: with no free slot it is `busy` (spec 11.3c). The error of that reply is the
/// error of the request.
///
/// Falsified by a route that leaves out a failed reply: the request is a 200 with two replies.
#[test]
fn a_request_with_more_forks_than_free_slots_is_busy() {
    let f = slow(2, &ControlLog::default());
    let council = council(&f);
    let (status, e) = generate(
        &council,
        &with(request(&context("Write a plan."), 3), "forks", json!(4)),
    );
    assert_eq!(
        (status, &e["error"]["type"]),
        (StatusCode::TOO_MANY_REQUESTS, &json!("busy")),
        "{e}"
    );
    assert!(e.get("replies").is_none(), "{e}");
}

/// Falsified by a route that gives the error of the last failed reply: the 400 names `contexts[2]`.
#[test]
fn the_error_of_the_failed_reply_with_the_lowest_index_is_the_error_of_the_request() {
    let f = Fixture::new();
    let council = council(&f);
    let good = context("Write a plan.");
    let bad = |text: &str| json!({"thinking": false, "messages": [{"role": "user", "content": text}, {"role": "user", "content": []}]});
    let mut body = request(&good, 3);
    body.as_object_mut().unwrap().remove("context");
    let contexts = json!([good, bad("one"), bad("two"), good]);
    let (status, e) = generate(&council, &with(body, "contexts", contexts));
    assert_eq!(
        (status, &e["error"]["param"]),
        (StatusCode::BAD_REQUEST, &json!("contexts[1].messages[1].content")),
        "{e}"
    );
}

/// A request with no seed gets the seed of reply 0 from the server, once, and reply `i` has that seed plus `i`.
///
/// Falsified by a first seed of 0 for each request: two requests have the same seeds.
#[test]
fn a_request_with_no_seed_gets_a_new_first_seed_and_the_next_seeds_follow_it() {
    let f = Fixture::new();
    let council = council(&f);
    let mut body = with(request(&context("Write a plan."), 3), "forks", json!(3));
    body["sample"] = json!({"temperature": 1.0});
    body["max_tokens"] = json!(2);
    let seeds = || {
        let (status, several) = generate(&council, &body);
        assert_eq!(status, StatusCode::OK, "{several}");
        let seeds: Vec<u64> = several["replies"]
            .as_array()
            .unwrap()
            .iter()
            .map(|reply| reply["seed"].as_u64().unwrap())
            .collect();
        assert_eq!(seeds, [seeds[0], seeds[0].wrapping_add(1), seeds[0].wrapping_add(2)]);
        seeds[0]
    };
    assert_ne!(seeds(), seeds(), "each request gets a new first seed");
}
