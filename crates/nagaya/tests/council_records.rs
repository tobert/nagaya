//! Context records of a service (gogi "Context records", spec 6.17, 11.13): the limit and the order of the drops,
//! pins, a restart, the check at a read, and what holds a record.

mod council_support;

use std::collections::BTreeSet;
use std::sync::mpsc;

use council_support::{ENGINE, Fixture};
use gogi_proto::context::Context;
use gogi_proto::generation::{GenerationRequest, TokenEvent};
use gogi_proto::pins::PinsPut;
use gogi_proto::{ContextId, context_id, record_text};
use nagaya::council::error::ServiceError;
use nagaya::council::records::{RECORD_DIR, Records};
use nagaya::council::{Schedule, ServiceConfig};
use serde_json::{Value, json};

fn context(text: &str) -> Context {
    let value: Value = json!({"thinking": true, "messages": [{"role": "user", "content": text, "snap": true}]});
    serde_json::from_value(value).unwrap()
}

/// A fixture whose server holds at most `limit` records that no pin keeps.
fn fixture(limit: usize) -> Fixture {
    let config = ServiceConfig {
        context_records: limit,
        ..ServiceConfig::new("toy", ENGINE)
    };
    Fixture::with(0, 1 << 20, None, config)
}

fn held(f: &Fixture, text: &str) -> bool {
    match f.service.record(&context_id(&context(text))) {
        Ok(found) => {
            assert_eq!(found, context(text));
            true
        }
        Err(ServiceError::NotFound(_)) => false,
        Err(other) => panic!("{other:?}"),
    }
}

fn pin(records: &[&ContextId], ttl_s: u64) -> PinsPut {
    PinsPut {
        snapshots: Vec::new(),
        contexts: records.iter().map(|id| id.to_string()).collect(),
        ttl_s,
    }
}

fn none() -> BTreeSet<String> {
    BTreeSet::new()
}

#[test]
fn past_the_limit_the_least_recently_named_record_goes_first() {
    let f = fixture(3);
    for text in ["a", "b", "c"] {
        f.service.hold_record(&context(text)).unwrap();
    }
    assert_eq!(f.service.context_records(), 3);
    // A read names `a`, and a hold of a held record names `b`. Thus `c` is the least recently named.
    assert!(held(&f, "a"));
    assert_eq!(f.service.hold_record(&context("b")).unwrap(), context_id(&context("b")));
    f.service.hold_record(&context("d")).unwrap();
    let file = |text: &str| {
        f.directory
            .path()
            .join(RECORD_DIR)
            .join(format!("{}.json", context_id(&context(text))))
    };
    assert!(!file("c").exists(), "the drop of a record removes its file");
    assert!(file("d").exists());
    assert!(!held(&f, "c"));
    for text in ["a", "b", "d"] {
        assert!(held(&f, text), "{text}");
    }
}

#[test]
fn a_parent_names_its_record() {
    let f = fixture(2);
    let a = f.service.hold_record(&context("a")).unwrap();
    f.service.hold_record(&context("b")).unwrap();
    // A request that names `a` as its parent makes `b` the least recently named.
    let mut body = json!({"model": "toy", "context": {"parent": a.as_str()}});
    f.service.resolve(&mut body, true).unwrap();
    assert_eq!(body["context"], serde_json::to_value(context("a")).unwrap());
    f.service.hold_record(&context("c")).unwrap();
    assert!(held(&f, "a"));
    assert!(!held(&f, "b"));
}

#[test]
fn a_pinned_record_is_never_dropped_and_a_pin_that_ended_keeps_nothing() {
    let f = fixture(2);
    let a = f.service.hold_record(&context("a")).unwrap();
    let gone = ContextId::parse(format!("blake3:{}", "0".repeat(64))).unwrap();
    let result = f.service.put_pins("kernel-1", &pin(&[&a, &gone], 3600)).unwrap();
    let report: Vec<(String, bool)> = result.contexts.iter().map(|r| (r.id.to_string(), r.held)).collect();
    assert_eq!(report, [(a.to_string(), true), (gone.to_string(), false)]);
    assert_eq!(
        result.expires_in_s,
        Some(3600),
        "a pin of a record alone keeps its owner"
    );
    let listing = f.service.get_pins("kernel-1").unwrap();
    assert_eq!((listing.snapshots.len(), listing.contexts), (0, vec![a.clone()]));
    // `a` is the least recently named, and the pin keeps it. A pinned record counts toward the limit of 2, so one
    // other record stays.
    for text in ["b", "c", "d"] {
        f.service.hold_record(&context(text)).unwrap();
    }
    let file = |text: &str| {
        f.directory
            .path()
            .join(RECORD_DIR)
            .join(format!("{}.json", context_id(&context(text))))
            .exists()
    };
    assert_eq!((file("a"), file("b"), file("c"), file("d")), (true, false, false, true));
    // After the pin ends, `a` goes as any record does.
    f.advance(3600 * 1000);
    f.service.hold_record(&context("e")).unwrap();
    assert_eq!((file("a"), file("c"), file("d"), file("e")), (false, false, true, true));
}

#[test]
fn a_delete_of_the_pins_frees_the_record() {
    let f = fixture(1);
    let a = f.service.hold_record(&context("a")).unwrap();
    f.service.put_pins("kernel-1", &pin(&[&a], 3600)).unwrap();
    f.service.hold_record(&context("b")).unwrap();
    assert!(held(&f, "a"));
    f.service.delete_pins("kernel-1").unwrap();
    f.service.hold_record(&context("c")).unwrap();
    assert!(!held(&f, "a"));
}

#[test]
fn records_and_their_pins_survive_a_restart() {
    let f = fixture(8);
    let a = f.service.hold_record(&context("a")).unwrap();
    f.service.put_pins("kernel-1", &pin(&[&a], 3600)).unwrap();
    // A second service opens the files of the store directory again, as a restart does.
    let again = f.attach_again();
    assert_eq!(again.record(&a).unwrap(), context("a"));
    assert_eq!(again.get_pins("kernel-1").unwrap().contexts, vec![a.clone()]);
    // A pinned record whose file is gone leaves the pin book at the next start.
    std::fs::remove_file(f.directory.path().join(RECORD_DIR).join(format!("{a}.json"))).unwrap();
    let again = f.attach_again();
    assert!(matches!(again.record(&a), Err(ServiceError::NotFound(_))));
    assert!(matches!(again.get_pins("kernel-1"), Err(ServiceError::NotFound(_))));
}

#[test]
fn after_a_restart_the_records_have_the_order_of_their_file_times() {
    let directory = tempfile::tempdir().unwrap();
    let mut book = Records::open_with(directory.path(), 3).unwrap();
    for text in ["a", "b", "c"] {
        book.hold(&context(text), &none()).unwrap();
        // The file times must differ.
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    // This read names `a` in memory only: no read writes a file (spec 6.17).
    assert!(book.get(context_id(&context("a")).as_str()).unwrap().is_some());
    drop(book);
    let mut book = Records::open_with(directory.path(), 3).unwrap();
    assert_eq!(book.len(), 3);
    book.hold(&context("d"), &none()).unwrap();
    let holds = |book: &Records, text: &str| book.holds(context_id(&context(text)).as_str());
    assert_eq!(
        (
            holds(&book, "a"),
            holds(&book, "b"),
            holds(&book, "c"),
            holds(&book, "d")
        ),
        (false, true, true, true),
        "the oldest file goes first"
    );
}

#[test]
fn a_record_file_whose_text_is_not_its_id_is_dropped_at_the_read() {
    let f = fixture(8);
    let a = f.service.hold_record(&context("a")).unwrap();
    let b = f.service.hold_record(&context("b")).unwrap();
    let file = |id: &ContextId| f.directory.path().join(RECORD_DIR).join(format!("{id}.json"));
    // The file of `a` holds the text of another context, and the file of `b` holds no context.
    std::fs::write(file(&a), record_text(&context("other"))).unwrap();
    std::fs::write(file(&b), b"{").unwrap();
    for id in [&a, &b] {
        assert!(matches!(f.service.record(id), Err(ServiceError::NotFound(_))));
        assert!(!file(id).exists(), "the read removes the bad file");
    }
    // A parent that names the dropped record is a miss, and the whole context holds the record again.
    let mut body = json!({"context": {"parent": a.as_str()}});
    match f.service.resolve(&mut body, true) {
        Err(ServiceError::NotHeld { param, message }) => {
            assert_eq!(param, "context.parent");
            assert!(message.contains(a.as_str()), "{message}");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(f.service.hold_record(&context("a")).unwrap(), a);
    assert!(held(&f, "a"));
}

#[test]
fn the_open_removes_a_temporary_file_and_refuses_a_file_with_another_name() {
    let directory = tempfile::tempdir().unwrap();
    let mut book = Records::open_with(directory.path(), 8).unwrap();
    let a = book.hold(&context("a"), &none()).unwrap();
    drop(book);
    let records = directory.path().join(RECORD_DIR);
    let temporary = records.join(format!("{a}.json.tmp"));
    std::fs::write(&temporary, b"half").unwrap();
    let book = Records::open_with(directory.path(), 8).unwrap();
    assert!(!temporary.exists());
    assert!(book.holds(a.as_str()));
    assert_eq!(book.len(), 1);
    drop(book);
    std::fs::write(records.join("notes.txt"), b"x").unwrap();
    match Records::open_with(directory.path(), 8) {
        Err(ServiceError::Internal(message)) => {
            assert!(
                message.contains("notes.txt") && message.contains("Remove the file"),
                "{message}"
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_build_names_its_record_and_a_dry_run_or_a_refused_build_holds_none() {
    let f = fixture(8);
    let built = f.service.build(&context("a"), false, Schedule::default()).unwrap();
    assert_eq!(built.context, Some(context_id(&context("a"))));
    assert!(held(&f, "a"));
    let dry = f.service.build(&context("b"), true, Schedule::default()).unwrap();
    assert_eq!(dry.context, None);
    assert!(!held(&f, "b"));
    // A build of an open message fails its checks.
    let open: Context = serde_json::from_value(json!({"thinking": false, "messages": [
        {"role": "user", "content": "u"}, {"role": "assistant", "content": "so far", "continue": true}]}))
    .unwrap();
    assert!(f.service.build(&open, false, Schedule::default()).is_err());
    assert!(matches!(
        f.service.record(&context_id(&open)),
        Err(ServiceError::NotFound(_))
    ));
}

#[test]
fn a_generation_names_the_record_with_its_reply_and_a_failed_one_keeps_the_record_of_its_context() {
    let f = fixture(8);
    let request = |text: &str| -> GenerationRequest {
        serde_json::from_value(json!({"model": "toy", "context": context(text), "max_tokens": 8,
                                      "sample": {"temperature": 0.0}}))
        .unwrap()
    };
    let done = f.service.generate(&request("a"), None).unwrap();
    assert!(held(&f, "a"));
    let with_reply = gogi_proto::with_reply(&context("a"), &done);
    assert_eq!(done.context, Some(context_id(&with_reply)));
    assert_eq!(f.service.record(done.context.as_ref().unwrap()).unwrap(), with_reply);
    assert_eq!(with_reply.messages.len(), 2);
    // A closed stream cancels the reply after its checks passed. The record of the context stays, and no record has
    // a reply.
    let (events, receiver) = mpsc::sync_channel::<TokenEvent>(1);
    drop(receiver);
    assert_eq!(
        f.service.generate(&request("b"), Some(events)).unwrap_err(),
        ServiceError::Cancelled
    );
    assert!(held(&f, "b"));
    let records = std::fs::read_dir(f.directory.path().join(RECORD_DIR)).unwrap().count();
    assert_eq!(records, 3, "a, a with its reply, and b");
    // A generation that fails its checks holds no record.
    let bad: GenerationRequest = serde_json::from_value(
        json!({"model": "toy", "context": context("c"), "max_tokens": 8, "sample": {"temperature": -1.0}}),
    )
    .unwrap();
    assert!(f.service.generate(&bad, None).is_err());
    assert!(!held(&f, "c"));
}
