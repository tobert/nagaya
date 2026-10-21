//! The park of a state whose bytes the store already holds (spec 10.3 step 3, 6.4, 6.11).
//!
//! A build puts a snapshot at the head of its context in the host tier (11.3c). The slot keeps those bytes, so a
//! stop finds the same content id in the store, under the build's recipe. On gfx1151 on 2026-10-21 at 00:44, that
//! park failed with "the content id already has another recipe", and the process exited 1.

mod council_support;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use council_support::{ENGINE, TICK_MS, TermFactory};
use gogi_proto::context::Context;
use nagaya::council::{Clock, Schedule, Service, ServiceConfig};
use nagaya::exit::ExitCode;
use nagaya::signal::SignalFlag;
use nagaya_store::{ContentId, Store, Tier};
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};

fn context() -> Context {
    let value: Value = json!({
        "thinking": true,
        "tools": [{"name": "sh", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "system", "content": "You are a coding agent. Use the shell tool."},
            {"role": "user", "content": "Delete the build directory, then rebuild.", "snap": true},
        ],
    });
    serde_json::from_value(value).unwrap()
}

fn attach(process: &nagaya::process::Process, directory: &std::path::Path) -> Service {
    let clock = Arc::new(AtomicU64::new(1_000_000));
    let read: Clock = Arc::new(move || clock.fetch_add(TICK_MS, Ordering::SeqCst));
    Service::attach_with_clock(
        process.ports(),
        Arc::new(ToyFamily::new()),
        directory,
        ServiceConfig::new("toy", ENGINE),
        read,
    )
    .unwrap()
}

fn head_id(built: &nagaya::council::Built) -> ContentId {
    let text = built.head.as_ref().expect("the build holds its head").as_str();
    ContentId::from_hex(text.strip_prefix("blake3:").expect("a snapshot id is blake3")).unwrap()
}

#[test]
fn a_park_of_a_state_that_the_store_holds_keeps_that_entry_and_a_restart_restores_it() {
    let directory = tempfile::tempdir().unwrap();
    let (mut process, _) = council_support::start(directory.path(), TermFactory::toy(), 1 << 20, None);
    let service = attach(&process, directory.path());
    let built = service.build(&context(), false, Schedule::default()).unwrap();
    let head = head_id(&built);
    let recipe = {
        let ports = process.ports();
        let store = ports.store().unwrap().lock().unwrap();
        let entry = &store.entries()[&head];
        assert_eq!(
            entry.tier,
            Tier::Host,
            "a build puts its snapshots in the host tier (11.3c)"
        );
        entry.recipe
    };
    drop(service);

    let report = process.stop(Duration::from_secs(5), &SignalFlag::new());
    assert!(report.parked.failures.is_empty(), "{:?}", report.parked.failures);
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
    assert_eq!(report.parked.states, 1, "one slot held tokens: {report:?}");
    drop(process);

    // Spec 6.11: the host tier does not survive a restart. The parked head is on NVMe, under the build's recipe.
    {
        let store = Store::open(directory.path()).unwrap();
        let entry = store.entries().get(&head).expect("the restart keeps the parked head");
        assert_eq!(entry.tier, Tier::Nvme, "{entry:?}");
        assert_eq!(entry.recipe, recipe, "the park keeps the recipe of the build");
    }

    // A build of the same context after the restart restores the parked head and feeds nothing.
    let (_process, _) = council_support::start(directory.path(), TermFactory::toy(), 1 << 20, None);
    let service = attach(&_process, directory.path());
    let again = service.build(&context(), false, Schedule::default()).unwrap();
    assert_eq!(head_id(&again), head, "the same context gives the same head");
    assert_eq!(
        (again.usage.kept, again.usage.fed),
        (again.usage.prompt, 0),
        "{:?}",
        again.usage
    );
}
