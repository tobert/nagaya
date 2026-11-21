//! Requests in flight share a prefix (gogi "Contexts", spec 11.13). Two requests can be in flight together with
//! contexts that begin with the same tokens, while the store holds no snapshot of them. Then the nagaya feeds those
//! tokens once. The engine is the fake engine, and the family is the toy family.

mod council_support;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use council_support::{Fixture, TermFactory};
use gogi_proto::context::Context;
use nagaya::council::{Built, ServiceConfig};
use serde_json::{Value, json};

fn context(messages: Vec<Value>) -> Context {
    serde_json::from_value(json!({"thinking": false, "messages": messages})).unwrap()
}

/// A nagaya whose engine takes 20 ms for each pass, with a slot for each request of a test. With a pass that takes
/// no time, the first request ends before the next one starts, and no two are in flight together.
fn slow() -> Fixture {
    slow_counting(&Arc::default())
}

/// [`slow`], with `starts` counting the passes that began.
fn slow_counting(starts: &Arc<AtomicU64>) -> Fixture {
    let factory = TermFactory {
        delay: Duration::from_millis(20),
        slots: council_support::roomy_slots(4),
        starts: Arc::clone(starts),
        ..TermFactory::toy()
    };
    Fixture::with_factory(
        factory,
        1 << 20,
        None,
        ServiceConfig::new("toy", council_support::ENGINE),
    )
}

/// A head that each context of a test begins with. It ends at a boundary.
fn head() -> Value {
    json!({"role": "user", "content": "We pack items into bins. Each bin holds 100. Use few bins.", "snap": true})
}

/// Build each context at once, each on its own thread, and give the responses in the order of the contexts.
fn build_together(f: &Fixture, contexts: &[Context]) -> Vec<Built> {
    let service = &f.service;
    let start = std::sync::Barrier::new(contexts.len());
    std::thread::scope(|scope| {
        let jobs: Vec<_> = contexts
            .iter()
            .map(|context| {
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    service.build(context, false, Default::default()).unwrap()
                })
            })
            .collect();
        jobs.into_iter().map(|job| job.join().unwrap()).collect()
    })
}

#[test]
fn equal_contexts_in_flight_feed_their_tokens_once() {
    let f = slow();
    let one = context(vec![head(), json!({"role": "user", "content": "Write a plan."})]);
    let built = build_together(&f, &vec![one.clone(); 4]);
    let prompt = built[0].usage.prompt;
    let fed: u64 = built.iter().map(|b| b.usage.fed).sum();
    assert_eq!(fed, prompt, "one request feeds the context, and the others keep it");
    for b in &built {
        assert_eq!(
            (b.usage.prompt, b.usage.kept + b.usage.fed, &b.head),
            (prompt, prompt, &built[0].head)
        );
    }
    // Each request gives the numbers that it gives alone: a build of the same context on a new nagaya has the head.
    let alone = Fixture::new().service.build(&one, false, Default::default()).unwrap();
    assert_eq!(alone.head, built[0].head);
}

#[test]
fn contexts_in_flight_that_share_a_marked_head_feed_the_head_once() {
    let f = slow();
    let contexts: Vec<Context> = ["the fit", "the gaps", "the large items", "the ties"]
        .iter()
        .map(|focus| {
            context(vec![
                head(),
                json!({"role": "user", "content": format!("Focus on {focus}.")}),
            ])
        })
        .collect();
    let built = build_together(&f, &contexts);
    // The tokens of the head, from a build of the head alone on a new nagaya.
    let shared = Fixture::new()
        .service
        .build(&context(vec![head()]), false, Default::default())
        .unwrap()
        .usage
        .prompt;
    let prompts: u64 = built.iter().map(|b| b.usage.prompt).sum();
    let fed: u64 = built.iter().map(|b| b.usage.fed).sum();
    assert_eq!(fed, prompts - 3 * shared, "three of the four requests keep the head");
    // Each head is the head of its context built alone.
    for (context, b) in contexts.iter().zip(&built) {
        let alone = Fixture::new()
            .service
            .build(context, false, Default::default())
            .unwrap();
        assert_eq!(alone.head, b.head);
    }
}

#[test]
fn a_request_alone_does_not_wait() {
    let f = slow();
    let one = context(vec![head(), json!({"role": "user", "content": "Write a plan."})]);
    let first = f.service.build(&one, false, Default::default()).unwrap();
    let again = f.service.build(&one, false, Default::default()).unwrap();
    assert_eq!((first.usage.kept, again.usage.fed, &again.head), (0, 0, &first.head));
}

/// A generation of `context` that decodes 40 tokens: at 20 ms for each pass, it stays in flight for 0.8 s or more.
fn long_generation(context: &Context) -> gogi_proto::generation::GenerationRequest {
    serde_json::from_value(
        json!({"model": "toy", "context": context, "max_tokens": 40, "sample": {"temperature": 1.0, "seed": 3}}),
    )
    .unwrap()
}

/// Start a long generation of `first`. When its first pass begins, build `second`: the generation holds its claim
/// then. The wait is for the pass itself, so a slow machine cannot start the build first. `starts` is the count of
/// passes of the engine of `f`, and a pass of an earlier request does not count. Give the build, the time of the build, and the time of the generation.
fn build_beside_a_generation(
    f: &Fixture,
    starts: &AtomicU64,
    first: &Context,
    second: &Context,
) -> (Built, Duration, Duration) {
    let service = &f.service;
    let before = starts.load(Ordering::SeqCst);
    std::thread::scope(|scope| {
        let generation = scope.spawn(move || {
            let start = std::time::Instant::now();
            service.generate(&long_generation(first), None).unwrap();
            start.elapsed()
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        while starts.load(Ordering::SeqCst) == before {
            assert!(Instant::now() < deadline, "the generation began no pass in 10 s");
            std::thread::yield_now();
        }
        let start = Instant::now();
        let built = service.build(second, false, Default::default()).unwrap();
        let build = start.elapsed();
        (built, build, generation.join().unwrap())
    })
}

#[test]
fn a_build_starts_from_the_prompt_of_a_generation_in_flight_and_does_not_wait_for_its_reply() {
    let starts = Arc::<AtomicU64>::default();
    let f = slow_counting(&starts);
    let one = context(vec![
        head(),
        json!({"role": "user", "content": "Write a plan.", "snap": true}),
    ]);
    let (built, build, generation) = build_beside_a_generation(&f, &starts, &one, &one);
    // The generation claims the boundaries of its prompt, so the build feeds nothing.
    assert_eq!(built.usage.fed, 0, "the build keeps the prompt of the generation");
    // The boundary is held when the prompt is fed. The build goes on then, long before the reply ends.
    assert!(
        generation >= Duration::from_millis(600),
        "the generation was not long: {generation:?}"
    );
    assert!(
        build < generation / 2,
        "the build waited for the reply: {build:?} of {generation:?}"
    );
}

#[test]
fn a_request_does_not_wait_for_a_boundary_that_the_store_holds() {
    let starts = Arc::<AtomicU64>::default();
    let f = slow_counting(&starts);
    // The store holds the head before the two requests start.
    f.service
        .build(&context(vec![head()]), false, Default::default())
        .unwrap();
    let first = context(vec![head(), json!({"role": "user", "content": "Focus on the fit."})]);
    let second = context(vec![head(), json!({"role": "user", "content": "Focus on the gaps."})]);
    let (built, build, generation) = build_beside_a_generation(&f, &starts, &first, &second);
    assert!(built.usage.kept > 0, "the build keeps the head");
    assert!(
        generation >= Duration::from_millis(600),
        "the generation was not long: {generation:?}"
    );
    assert!(
        build < generation / 2,
        "the build waited for a boundary that was held: {build:?} of {generation:?}"
    );
}
