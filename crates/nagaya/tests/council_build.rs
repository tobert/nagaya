//! A build renders a context, restores the longest held prefix, feeds the rest and holds a snapshot at each boundary
//! (spec 11.1, 11.3c). The engine is the fake engine over the toy vocabulary, and the family is the toy family.

mod council_support;

use std::io::{Read as _, Write as _};
use std::net::TcpStream;

use council_support::ENGINE;
use gogi_fake::{Config, Server};
use gogi_proto::SnapshotId;
use gogi_proto::context::{Context, Usage};
use nagaya::council::error::ServiceError;
use nagaya::council::{Built, Schedule, ServiceConfig};
use nagaya_store::{ContentId, Reason};
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};

type Fixture = council_support::Fixture;

fn fixture() -> Fixture {
    Fixture::new()
}

fn fixture_with(term: u64, host_budget: u64, host_store: Option<u64>) -> Fixture {
    Fixture::with(term, host_budget, host_store, ServiceConfig::new("toy", ENGINE))
}

fn fixture_config(term: u64, host_budget: u64, host_store: Option<u64>, config: ServiceConfig) -> Fixture {
    Fixture::with(term, host_budget, host_store, config)
}

fn context(value: &Value) -> Context {
    serde_json::from_value(value.clone()).unwrap()
}

fn build(f: &Fixture, value: &Value) -> Built {
    f.service.build(&context(value), false, Schedule::default()).unwrap()
}

fn dry(f: &Fixture, value: &Value) -> Built {
    f.service.build(&context(value), true, Schedule::default()).unwrap()
}

fn opening() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "You are a coding agent. Use the shell tool."}),
        json!({"role": "user", "content": "Delete the build directory, then rebuild.", "snap": true}),
    ]
}

fn ctx(messages: Vec<Value>) -> Value {
    json!({"thinking": true, "tools": [{"name": "sh", "input_schema": {"type": "object"}}], "messages": messages})
}

fn longer() -> Vec<Value> {
    let mut messages = opening();
    messages.push(json!({"role": "assistant", "content": "I will look first."}));
    messages.push(json!({"role": "user", "content": "Go ahead.", "snap": true}));
    messages
}

fn entries(f: &Fixture) -> usize {
    f.entries()
}

fn id_text(id: &SnapshotId) -> &str {
    id.as_str()
}

#[test]
fn a_first_build_feeds_everything_and_holds_a_snapshot_at_each_boundary() {
    let f = fixture();
    let built = build(&f, &ctx(longer()));
    let u = built.usage;
    assert!(
        u.prompt > 64,
        "the context needs more than one pass of the fake engine (64 tokens): {u:?}"
    );
    assert_eq!((u.kept, u.fed), (0, u.prompt), "{u:?}");
    assert_eq!(built.from, None);
    let messages: Vec<u64> = built.snapshots.iter().map(|s| s.message).collect();
    assert_eq!(
        messages,
        [1, 3],
        "a marked user message is a boundary, and so is the end"
    );
    assert_eq!(built.snapshots.last().unwrap().id, built.head.clone().unwrap());
    assert_eq!(built.snapshots.last().unwrap().tokens, u.prompt);
    assert_eq!(entries(&f), 2);
    assert_eq!(f.service.free_slots(), 3, "a build gives its slot back");
    // A marked boundary is an anchor, and the end of the context is a finished prefill (spec 6.4).
    let ports = f.process.ports();
    let store = ports.store().unwrap().lock().unwrap();
    let reasons: Vec<Reason> = built
        .snapshots
        .iter()
        .map(|s| {
            let id = ContentId::from_hex(s.id.as_str().trim_start_matches("blake3:")).unwrap();
            store.entries()[&id].reason
        })
        .collect();
    assert_eq!(reasons, [Reason::Anchor, Reason::PrefillDone]);
}

#[test]
fn a_snapshot_id_is_blake3_and_64_lowercase_hex_digits() {
    let f = fixture();
    let built = build(&f, &ctx(longer()));
    for snapshot in &built.snapshots {
        let text = id_text(&snapshot.id);
        let hex = text.strip_prefix("blake3:").expect("the scheme is blake3");
        assert_eq!(hex.len(), 64, "{text}");
        assert!(
            hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "{text}"
        );
    }
}

#[test]
fn a_second_build_of_the_same_context_keeps_everything_and_reports_the_same_head() {
    let f = fixture();
    let first = build(&f, &ctx(longer()));
    let second = build(&f, &ctx(longer()));
    assert_eq!(second.usage.kept, second.usage.prompt);
    assert_eq!(second.usage.fed, 0);
    assert_eq!(second.head, first.head);
    assert_eq!(second.from, first.head, "the build starts from the head");
    assert_eq!(second.snapshots, first.snapshots);
    assert_eq!(entries(&f), 2, "a second build holds nothing new");
    assert_eq!(f.service.free_slots(), 3);
}

#[test]
fn a_build_that_shares_a_prefix_starts_from_that_snapshot() {
    let f = fixture();
    let short = build(&f, &ctx(opening()));
    let long = build(&f, &ctx(longer()));
    assert_eq!(
        long.from, short.head,
        "the long context starts from the head of the short one"
    );
    assert_eq!(long.usage.kept, short.usage.prompt);
    assert_eq!(long.usage.fed, long.usage.prompt - short.usage.prompt);
    assert_eq!(
        long.snapshots[0], short.snapshots[0],
        "the shared boundary keeps its id"
    );
    assert_eq!(long.snapshots.len(), 2);
}

#[test]
fn a_changed_earlier_message_starts_from_the_last_shared_boundary() {
    let f = fixture();
    let mut base = longer();
    base.push(json!({"role": "assistant", "content": "Looking."}));
    base.push(json!({"role": "user", "content": "Now report.", "snap": true}));
    let first = build(&f, &ctx(base.clone()));
    assert_eq!(first.snapshots.iter().map(|s| s.message).collect::<Vec<_>>(), [1, 3, 5]);
    // Edit the assistant message after the second boundary: the first two boundaries are shared.
    let mut edited = base.clone();
    edited[4] = json!({"role": "assistant", "content": "Looking again."});
    let second = build(&f, &ctx(edited));
    assert_eq!(second.from, Some(first.snapshots[1].id.clone()));
    assert_eq!(second.usage.kept, first.snapshots[1].tokens);
    assert_eq!(second.snapshots[0], first.snapshots[0]);
    assert_eq!(second.snapshots[1], first.snapshots[1]);
    assert_ne!(second.head, first.head);
    // Edit the first user message: nothing is shared.
    let mut early = base;
    early[1] = json!({"role": "user", "content": "Only delete it.", "snap": true});
    let third = build(&f, &ctx(early));
    assert_eq!(third.from, None);
    assert_eq!(third.usage.kept, 0);
    assert_eq!(third.usage.fed, third.usage.prompt);
}

#[test]
fn a_dry_run_feeds_nothing_and_holds_nothing() {
    let f = fixture();
    let before = entries(&f);
    let fresh = dry(&f, &ctx(longer()));
    assert_eq!(entries(&f), before);
    assert!(fresh.snapshots.is_empty());
    assert_eq!(fresh.head, None, "the store holds no head to name (spec 11.3c)");
    assert_eq!((fresh.usage.kept, fresh.usage.fed), (0, fresh.usage.prompt));

    let built = build(&f, &ctx(opening()));
    let held = entries(&f);
    let partly = dry(&f, &ctx(longer()));
    assert_eq!(partly.from, built.head);
    assert_eq!(partly.usage.kept, built.usage.prompt);
    assert_eq!(partly.usage.fed, partly.usage.prompt - built.usage.prompt);
    assert_eq!(entries(&f), held, "a dry run holds nothing");

    let full = dry(&f, &ctx(opening()));
    assert_eq!(full.head, built.head, "the held head is named");
    assert_eq!(full.usage.fed, 0);
    assert_eq!(f.service.free_slots(), 3);
}

#[test]
fn a_refused_context_charge_changes_nothing() {
    use nagaya_abi::Tier;
    use nagaya_ledger::LedgerError;

    // Each context takes 2048 host bytes. A build also needs host bytes for the copy of a snapshot (spec 4.2), so the
    // budget is roomy. The test takes the rest of the unassigned bytes but 2047, so that one context charge is
    // short and the copy is not the refusal.
    let f = fixture_with(2048, 1 << 20, Some(1 << 20));
    let ports = f.process.ports();
    let initial = ports.unassigned();
    // Hold the only charge, as a build in flight does.
    ports.admit_context(ENGINE).unwrap();
    let free = ports
        .unassigned()
        .into_iter()
        .find(|(tiers, _)| tiers.contains(&Tier::Host))
        .map(|(_, bytes)| bytes)
        .unwrap();
    ports.admit_staging(ENGINE, Tier::Host, free - 2047).unwrap();
    let held = ports.unassigned();
    let error = f
        .service
        .build(&context(&ctx(longer())), false, Schedule::default())
        .unwrap_err();
    assert!(
        matches!(error, ServiceError::ContextRefused(LedgerError::ContextRefused(_))),
        "{error:?}"
    );
    assert_eq!(ports.unassigned(), held, "the ledger is as it was");
    assert_eq!(entries(&f), 0, "no snapshot");
    assert_eq!(f.service.free_slots(), 3, "no slot");
    // The charge of a build is released: after the holder lets go, one build works, and so does the next.
    ports.release_staging(Tier::Host, free - 2047).unwrap();
    ports.release_context(ENGINE).unwrap();
    build(&f, &ctx(longer()));
    build(&f, &ctx(opening()));
    assert_eq!(ports.unassigned(), initial, "each build released its charge");
}

#[test]
fn a_context_that_fits_no_slot_is_too_large() {
    let f = fixture();
    // 16384 bytes of cells in a slot hold 2048 rows.
    let long_text = "x".repeat(3000);
    let error = f
        .service
        .build(
            &context(&ctx(vec![json!({"role": "user", "content": long_text, "snap": true})])),
            false,
            Schedule::default(),
        )
        .unwrap_err();
    assert!(
        matches!(&error, ServiceError::TooLarge { param, .. } if param == "context_tokens"),
        "{error:?}"
    );
    assert_eq!(f.service.free_slots(), 3);
    assert_eq!(entries(&f), 0);
    assert_eq!(f.process.ports().unassigned().len(), 2);
}

#[test]
fn a_context_at_the_token_limit_is_context_full() {
    let mut config = ServiceConfig::new("toy", ENGINE);
    config.context_tokens = 30;
    let f = fixture_config(0, 1 << 20, None, config);
    let error = f
        .service
        .build(&context(&ctx(longer())), false, Schedule::default())
        .unwrap_err();
    assert!(
        matches!(&error, ServiceError::ContextFull { context_tokens: 30, .. }),
        "{error:?}"
    );
    assert_eq!(entries(&f), 0);
}

#[test]
fn a_context_with_no_message_is_a_client_error() {
    let f = fixture();
    let error = f
        .service
        .build(&context(&json!({"messages": []})), false, Schedule::default())
        .unwrap_err();
    assert!(matches!(error, ServiceError::Invalid(_)), "{error:?}");
}

#[test]
fn a_pin_keeps_a_snapshot_from_eviction() {
    // The host tier of the store holds two snapshots of one boundary each, not three.
    let probe = fixture();
    let probe_built = build(
        &probe,
        &ctx(vec![json!({"role": "user", "content": "probe 0", "snap": true})]),
    );
    let size = probe
        .service
        .snapshot(&probe_built.head.unwrap())
        .unwrap()
        .bytes
        .unwrap();
    let budget = size * 5 / 2;
    for pinned in [true, false] {
        let f = fixture_with(0, 1 << 20, Some(budget));
        let a = build(
            &f,
            &ctx(vec![json!({"role": "user", "content": "probe a", "snap": true})]),
        );
        let head = a.head.clone().unwrap();
        if pinned {
            let put = gogi_proto::pins::PinsPut {
                snapshots: vec![head.to_string()],
                ttl_s: 3600,
            };
            let result = f.service.put_pins("owner", &put).unwrap();
            assert!(result.snapshots[0].held);
        }
        build(
            &f,
            &ctx(vec![json!({"role": "user", "content": "probe b", "snap": true})]),
        );
        build(
            &f,
            &ctx(vec![json!({"role": "user", "content": "probe c", "snap": true})]),
        );
        let held = f.service.snapshot(&head).is_ok();
        assert_eq!(held, pinned, "pinned: {pinned}");
    }
}

fn post(server: &Server, path: &str, body: Value) -> Value {
    let addr = server.url().trim_start_matches("http://").to_string();
    let mut c = TcpStream::connect(&addr).unwrap();
    let b = body.to_string();
    write!(
        c,
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
        b.len()
    )
    .unwrap();
    let mut raw = String::new();
    c.read_to_string(&mut raw).unwrap();
    let (head, body) = raw.split_once("\r\n\r\n").unwrap();
    assert!(head.contains(" 200 "), "{head}\n{body}");
    serde_json::from_str(body).unwrap()
}

fn usage_of(value: &Value) -> Usage {
    serde_json::from_value(value["usage"].clone()).unwrap()
}

/// The pairs (message, tokens) of the snapshots of a gogi-fake response.
fn cuts_of(value: &Value) -> Vec<(u64, u64)> {
    value["snapshots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| (s["message"].as_u64().unwrap(), s["tokens"].as_u64().unwrap()))
        .collect()
}

#[test]
fn usage_and_message_indices_match_gogi_fake_for_the_same_contexts() {
    let fake = Server::start(Config::default()).unwrap();
    let f = fixture();
    let mut longest = longer();
    longest.push(json!({"role": "assistant", "content": "Looking."}));
    longest.push(json!({"role": "user", "content": "Now report.", "snap": true}));
    let mut edited_late = longest.clone();
    edited_late[4] = json!({"role": "assistant", "content": "Looking again."});
    let mut edited_early = longest.clone();
    edited_early[0] = json!({"role": "system", "content": "You are another agent."});
    let sequence = [
        ctx(opening()),
        ctx(opening()),
        ctx(longer()),
        ctx(longest.clone()),
        ctx(edited_late),
        ctx(edited_early),
        json!({"thinking": false, "tools": [], "messages": opening()}),
    ];
    for (i, value) in sequence.iter().enumerate() {
        for dry_run in [true, false] {
            let want = post(
                &fake,
                "/council/v2/contexts",
                json!({"model": "toy", "context": value, "dry_run": dry_run}),
            );
            let got = f.service.build(&context(value), dry_run, Schedule::default()).unwrap();
            assert_eq!(got.usage, usage_of(&want), "context {i}, dry_run {dry_run}");
            if !dry_run {
                let got_cuts: Vec<(u64, u64)> = got.snapshots.iter().map(|s| (s.message, s.tokens)).collect();
                assert_eq!(got_cuts, cuts_of(&want), "context {i}");
            }
        }
    }
}

/// What a direct run of the fake engine gives for a context: the content id of the sealed state at each boundary.
fn direct_ids(value: &Value) -> Vec<String> {
    use nagaya_abi::fake::FakeAlways;
    use nagaya_abi::{Engine, OutputsWanted, Segment, SegmentInputs, SlotLimit, State};
    use nagaya_council::family::Family;
    let rendering = ToyFamily::new().render(&context(value), false).unwrap();
    let mut engine = FakeAlways::with_slots(vec![SlotLimit { bytes: 16384, count: 1 }]).with_vocab(gogi_toy::VOCAB);
    let state = State::Slot(0);
    let mut position = 0;
    let mut ids = Vec::new();
    for boundary in &rendering.boundaries {
        for from in (position..boundary.tokens).step_by(64) {
            let to = boundary.tokens.min(from + 64);
            engine
                .pass(
                    &[Segment {
                        state,
                        inputs: SegmentInputs::Tokens(rendering.tokens[from..to].to_vec()),
                        positions: (from as u64..to as u64).collect(),
                        meta: Vec::new(),
                        outputs: OutputsWanted::default(),
                    }],
                    &[],
                )
                .unwrap();
        }
        position = boundary.tokens;
        engine.seal().unwrap().seal(state).unwrap();
        let n = position as u64;
        let window = n.min(4);
        let snapshot =
            nagaya_store::snapshot_from_engine(&mut engine, state, &[(1, 0, n), (2, n - window, window)]).unwrap();
        ids.push(format!("blake3:{}", snapshot.content_id().unwrap().hex()));
    }
    ids
}

#[test]
fn a_snapshot_id_is_the_content_id_of_the_sealed_state_that_a_direct_run_gives() {
    let f = fixture();
    let mut messages = longer();
    messages.push(json!({"role": "assistant", "content": "Looking."}));
    messages.push(json!({"role": "user", "content": "Now report.", "snap": true}));
    let value = ctx(messages);
    let built = build(&f, &value);
    let ids: Vec<String> = built.snapshots.iter().map(|s| s.id.to_string()).collect();
    assert_eq!(ids, direct_ids(&value));
}

#[test]
fn a_build_that_restores_a_prefix_gives_the_snapshots_of_a_straight_build() {
    let value = ctx(longer());
    let straight = build(&fixture(), &value);
    let f = fixture();
    build(&f, &ctx(opening()));
    let restored = build(&f, &value);
    assert!(restored.usage.kept > 0, "the second build restores the first");
    assert_eq!(restored.snapshots, straight.snapshots);
    assert_eq!(restored.head, straight.head);
}

#[test]
fn a_build_starts_from_an_empty_slot_even_when_an_earlier_context_used_it() {
    let value = ctx(longer());
    let alone = build(&fixture(), &value);
    let f = fixture();
    build(
        &f,
        &ctx(vec![
            json!({"role": "user", "content": "Something else entirely.", "snap": true}),
        ]),
    );
    let after = build(&f, &value);
    assert_eq!(after.usage.kept, 0);
    assert_eq!(after.snapshots, alone.snapshots);
}

/// Run every quantum of a job body in order, as the group thread does.
fn run_body(body: &mut dyn nagaya::jobs::JobBody, engines: &mut [Box<dyn nagaya_abi::Engine>]) {
    use nagaya::jobs::Ran;
    let mut queue = std::collections::VecDeque::from([nagaya_abi::Kind::Copy]);
    while let Some(kind) = queue.pop_front() {
        match body.run(kind, engines) {
            Ran::More(kinds) => queue.extend(kinds),
            Ran::Done => {}
            other => panic!("the job ended with {other:?}"),
        }
    }
}

#[test]
fn a_job_frees_its_slot_before_its_caller_hears_the_result() {
    use std::sync::{Arc, Mutex, mpsc};

    use nagaya::council::build::{BuildBody, Shared};
    use nagaya::council::slots::SlotPool;
    use nagaya_abi::Engine;
    use nagaya_council::build::{boundary_recipes, root_recipe};
    use nagaya_council::family::Family;

    let directory = tempfile::tempdir().unwrap();
    let engine = nagaya_abi::fake::FakeAlways::with_slots(council_support::roomy_slots(3)).with_vocab(gogi_toy::VOCAB);
    let limits = engine.limits();
    let pool = Arc::new(Mutex::new(SlotPool::new(&limits)));
    let process = fixture();
    let shared = Arc::new(Shared {
        ports: process.process.ports(),
        store: Arc::new(Mutex::new(nagaya_store::Store::open(directory.path()).unwrap())),
        engine_index: 0,
        engine_name: ENGINE.to_owned(),
        identity: [7; 32],
        limits: Arc::new(limits),
        chunk: 64,
        vocab: gogi_toy::VOCAB,
        pool: Arc::clone(&pool),
        clock: Arc::new(|| 1_000_000),
    });
    let rendering = ToyFamily::new().render(&context(&ctx(opening())), false).unwrap();
    let recipes = boundary_recipes(root_recipe(&[7; 32], b""), &rendering).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut body = BuildBody::new(shared, rendering.tokens[..rendering.end].into(), recipes, tx);
    let mut engines: Vec<Box<dyn Engine>> = vec![Box::new(engine)];
    run_body(&mut body, &mut engines);
    assert!(rx.try_recv().unwrap().is_ok());
    // The body still exists, as it does on the group thread until the scheduler hears that the quantum ended.
    assert_eq!(
        pool.lock().unwrap().free(),
        3,
        "the slot is free when the caller hears the result"
    );
    drop(body);
}
