//! Interval copies on the job path (spec 6.13): a context that crosses a multiple of the grain gets a copy in the
//! store between quanta, and a later build restores from it. The engine is the fake engine with a declared grain.

mod council_support;

use council_support::{Fixture, Script, TermFactory};
use gogi_proto::context::Context;
use gogi_proto::generation::GenerationRequest;
use nagaya::council::{Built, Schedule, ServiceConfig};
use nagaya_council::build::{Step, plan_steps};
use nagaya_council::family::Family;
use nagaya_store::{Entry, Reason};
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};

/// The grain of the fake engine in these tests, in tokens. A pass of the fake holds at most 64 tokens.
const GRAIN: u64 = 16;

fn fixture(grain: Option<u64>, copies: u32) -> Fixture {
    let factory = TermFactory {
        grain,
        interval_copies: copies,
        ..TermFactory::toy()
    };
    Fixture::with_factory(
        factory,
        1 << 22,
        None,
        ServiceConfig::new("toy", council_support::ENGINE),
    )
}

fn context(value: &Value) -> Context {
    serde_json::from_value(value.clone()).unwrap()
}

fn build(f: &Fixture, value: &Value) -> Built {
    f.service.build(&context(value), false, Schedule::default()).unwrap()
}

/// One long message: the toy tokenizer gives one token for each byte, so the message takes several passes.
fn long(words: &str) -> Value {
    json!({"thinking": false, "messages": [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": words},
    ]})
}

const WORDS: &str = "Read the build log from the top, find the first error, explain it, and then propose a fix that keeps \
                     the cache. Do not delete the build directory, and do not touch the lock file.";

/// The interval copies in the store.
fn copies(f: &Fixture) -> Vec<Entry> {
    let ports = f.process.ports();
    let store = ports.store().unwrap().lock().unwrap();
    store
        .entries()
        .values()
        .filter(|entry| entry.reason == Reason::Interval)
        .cloned()
        .collect()
}

/// The position of a copy: the end of its cells coverage.
fn position(entry: &Entry) -> u64 {
    entry
        .coverage
        .iter()
        .map(|span| span.end)
        .max()
        .expect("a copy has rows")
}

#[test]
fn a_build_that_crosses_the_grain_leaves_one_interval_copy() {
    let f = fixture(Some(GRAIN), 4);
    let built = build(&f, &long(WORDS));
    assert!(
        built.usage.prompt > 64,
        "the context takes two passes: {:?}",
        built.usage
    );
    let held = copies(&f);
    // The store keeps the latest copy of the context only.
    assert_eq!(held.len(), 1, "{held:?}");
    let copy = &held[0];
    assert!(copy.parent.is_none(), "a copy holds whole parts");
    assert_eq!(copy.tier, nagaya_store::Tier::Host, "a copy goes to host memory");
    // The latest copy is at the end of the last pass that did not end on a boundary. A pass of the fake holds 64
    // tokens, so each pass crosses a multiple of the grain.
    let rendering = ToyFamily::new().render(&context(&long(WORDS)), false).unwrap();
    let ends: Vec<usize> = plan_steps(0, &rendering.boundaries, 64)
        .into_iter()
        .filter_map(|step| match step {
            Step::Feed { to, .. } => Some(to),
            Step::Snapshot { .. } => None,
        })
        .filter(|to| rendering.boundaries.iter().all(|boundary| boundary.tokens != *to))
        .collect();
    assert!(
        ends.len() > 1,
        "the test needs two copies, so that the first one goes: {ends:?}"
    );
    assert_eq!(position(copy), *ends.last().unwrap() as u64, "{copy:?}");
}

#[test]
fn a_cap_of_zero_makes_no_copies() {
    // The gfx1151host profile has `interval_copies = 0`: the engine declares a grain, and the process makes no copy.
    let f = fixture(Some(GRAIN), 0);
    build(&f, &long(WORDS));
    assert_eq!(copies(&f), Vec::<Entry>::new());
}

#[test]
fn an_engine_with_no_grain_gets_no_copies() {
    let f = fixture(None, 4);
    build(&f, &long(WORDS));
    assert_eq!(copies(&f), Vec::<Entry>::new());
}

#[test]
fn a_later_build_restores_from_the_copy_and_gets_the_bits_of_a_full_feed() {
    let f = fixture(Some(GRAIN), 4);
    build(&f, &long(WORDS));
    let at = position(&copies(&f)[0]);

    // The same message with more words: the tokens up to the copy are the same, and the boundary is later.
    let longer = long(&format!("{WORDS} Then run the tests twice."));
    let built = build(&f, &longer);
    assert_eq!(
        built.usage.kept, at,
        "the build starts from the copy: {:?}",
        built.usage
    );

    // A store with no copy feeds every token. The head has the same bits.
    let plain = fixture(Some(GRAIN), 0);
    let fed = build(&plain, &longer);
    assert_eq!(fed.usage.kept, 0);
    assert_eq!(
        built.head, fed.head,
        "the head after a restore from the copy has other bits"
    );
}

#[test]
fn a_build_with_other_tokens_before_the_copy_does_not_use_it() {
    let f = fixture(Some(GRAIN), 4);
    build(&f, &long(WORDS));
    let other = long(&format!("Please {WORDS}"));
    let built = build(&f, &other);
    assert_eq!(built.usage.kept, 0, "{:?}", built.usage);
}

#[test]
fn a_generation_that_crosses_the_grain_leaves_an_interval_copy() {
    let script = Script::default();
    let factory = TermFactory {
        grain: Some(GRAIN),
        interval_copies: 4,
        script: script.clone(),
        ..TermFactory::toy()
    };
    let f = Fixture::with_factory(
        factory,
        1 << 22,
        None,
        ServiceConfig::new("toy", council_support::ENGINE),
    );
    let value = json!({"thinking": false, "messages": [{"role": "user", "content": "Say a long thing."}]});
    let rendered = ToyFamily::new().render(&context(&value), true).unwrap().tokens.len();
    // The engine favors `a` after the prompt, so the reply runs to max_tokens.
    script.set(rendered, vec![u32::from(b'a'); 64]);
    let request: GenerationRequest = serde_json::from_value(
        json!({"model": "toy", "context": value, "max_tokens": 40, "sample": {"temperature": 0}}),
    )
    .unwrap();
    let done = f.service.generate(&request, None).unwrap();
    let prompt = done.usage.prompt;
    assert!(prompt < GRAIN * 2, "the prompt stays below 32 tokens: {prompt}");
    assert_eq!(done.usage.completion, 40, "{:?}", done.usage);
    let held = copies(&f);
    assert_eq!(held.len(), 1, "{held:?}");
    let at = position(&held[0]);
    let end = prompt + done.usage.completion;
    assert!(
        at > prompt && at <= end,
        "the copy is in the reply: {at}, prompt {prompt}, end {end}"
    );
    // A decode quantum feeds one token, so the copy is on the last multiple of the grain that the reply crossed.
    assert_eq!(at % GRAIN, 0, "{at}");
}

#[test]
fn a_read_that_rewinds_across_the_grain_gives_the_numbers_of_an_engine_with_no_grain() {
    // A read with two suffixes marks the state, feeds a suffix, and rewinds to the mark. The rewind moves the rows
    // back, and the copies go on from there. The first suffix is longer, so the rows after the second one are fewer. A
    // grain of 4 tokens makes each suffix cross a multiple.
    let body = json!({"model": "toy", "contexts": [long(WORDS)], "suffixes": ["\nAnswer at some length: ", "\nV: "],
        "letters": ["A", "B", "C"], "top": 4});
    let request: gogi_proto::reads::ReadsRequest = serde_json::from_value(body).unwrap();
    let with_grain = fixture(Some(4), 4);
    let got = with_grain.service.read(&request).unwrap();
    assert!(!copies(&with_grain).is_empty(), "the read crossed the grain");
    let plain = fixture(None, 4);
    let want = plain.service.read(&request).unwrap();
    assert_eq!(
        serde_json::to_value(&got.reads).unwrap(),
        serde_json::to_value(&want.reads).unwrap()
    );
}

#[test]
fn a_client_cannot_pin_a_copy_so_a_stop_parks_a_slot_that_holds_the_bytes_of_a_copy() {
    // A read with one suffix ends on its last pass. With a grain of 4, that pass crosses a multiple, so the copy after
    // it holds the bytes of the slot. A pin on that copy would stop the park from removing it (spec 10.3), and the
    // stop would exit 1.
    let f = fixture(Some(4), 4);
    let body = json!({"model": "toy", "contexts": [long(WORDS)], "suffixes": ["\nAnswer at some length: "],
        "letters": ["A", "B", "C"], "top": 4});
    let request: gogi_proto::reads::ReadsRequest = serde_json::from_value(body).unwrap();
    f.service.read(&request).unwrap();
    let held = copies(&f);
    assert_eq!(held.len(), 1, "{held:?}");
    let id = nagaya::council::snapshot_id(&held[0].content_id);

    let pins = gogi_proto::pins::PinsPut {
        snapshots: vec![id.as_str().to_owned()],
        ttl_s: 3600,
    };
    // The park reads the system clock, so the pin must use it too.
    let system = nagaya::council::Service::attach(
        f.process.ports(),
        std::sync::Arc::new(ToyFamily::new()),
        f.directory.path(),
        ServiceConfig::new("toy", council_support::ENGINE),
    )
    .unwrap();
    let result = system.put_pins("kernel-1", &pins).unwrap();
    drop(system);
    assert!(
        !result.snapshots[0].held,
        "a client pinned an interval copy: {result:?}"
    );

    let Fixture {
        service, mut process, ..
    } = f;
    drop(service);
    let report = process.stop(std::time::Duration::from_secs(5), &nagaya::signal::SignalFlag::new());
    assert!(report.parked.failures.is_empty(), "{:?}", report.parked.failures);
    assert_eq!(report.exit_code(), nagaya::exit::ExitCode::Clean, "{report:?}");
    assert_eq!(report.parked.states, 1, "{report:?}");
}

/// kaibo job-5 item 6: a dry run reports what a build would keep, an interval copy too.
#[test]
fn a_dry_run_starts_from_the_copy_as_a_build_does() {
    let f = fixture(Some(GRAIN), 4);
    build(&f, &long(WORDS));
    let at = position(&copies(&f)[0]);
    let longer = long(&format!("{WORDS} Then run the tests twice."));
    let dry = f.service.build(&context(&longer), true, Schedule::default()).unwrap();
    assert_eq!(dry.usage.kept, at, "{:?}", dry.usage);
    assert_eq!(build(&f, &longer).usage.kept, at);
}
