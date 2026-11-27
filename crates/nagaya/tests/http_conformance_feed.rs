//! gogi's suite on an engine with a feed rule (spec 6.14, engine-abi D11). A generation on such an engine holds a
//! decoded head, so the suite runs its head checks here: the head after a reply, the head of a cut reply, and the
//! record of a reply. The engine is the fake with a feed rule, and the family is the toy family.

mod feed_support;

use std::sync::Arc;
use std::time::Duration;

use feed_support::{FeedFactory, FeedFixture};
use gogi_conformance::{Opts, Status, run};
use nagaya::council::{Service, ServiceConfig};
use nagaya::http::{Council, Served, Server};
use nagaya_family_toy::ToyFamily;

/// The checks of the suite that read the decoded head of a reply. Each must run, not skip.
const HEAD_CHECKS: [&str; 4] = [
    "generation holds its head",
    "the head is the snapshot after the reply as it renders",
    "the record of a reply builds to the head of the generation",
    "a continuation resumed from the head feeds at most the last sampled token",
];

#[test]
fn gogi_conformance_passes_on_an_engine_with_a_feed_rule() {
    let FeedFixture {
        service,
        process,
        directory,
        ..
    } = FeedFixture::with(
        FeedFactory {
            // The Gemma 4 engine takes one segment in a pass. The bits of the fake depend on the token count of a
            // pass, so a joined decode row would give a fork other bits than the same request gives alone. That
            // depends on the timing of the jobs, and the suite would fail on a loaded machine.
            max_segments: Some(1),
            ..FeedFactory::new(1)
        },
        Duration::from_secs(10),
    );
    // The clock of the fixture steps one minute at each read, so a pin of the suite would end at once. The suite runs
    // on a service with the clock of the system.
    drop(service);
    let service = Service::attach(
        process.ports(),
        Arc::new(ToyFamily::new()),
        directory.path(),
        ServiceConfig::new("toy", feed_support::ENGINE),
    )
    .unwrap();
    let served = Served {
        service,
        aliases: Vec::new(),
        device: None,
    };
    let council = Arc::new(Council::new(vec![served], None).unwrap());
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let server = Server::start(runtime.handle(), "127.0.0.1:0".parse().unwrap(), council)
        .expect("the HTTP server binds a loopback port");
    let report = run(
        Opts {
            url: format!("http://{}", server.local_addr()),
            restart: None,
            model: None,
        },
        false,
    );
    let count = |status: Status| report.checks.iter().filter(|check| check.status == status).count();
    println!(
        "gogi-conformance on the fake with a feed rule: {} ok, {} FAIL, {} skip",
        count(Status::Ok),
        count(Status::Fail),
        count(Status::Skip)
    );
    let failed: Vec<String> = report
        .checks
        .iter()
        .filter(|check| check.status == Status::Fail)
        .map(|check| format!("{}: {}", check.name, check.detail))
        .collect();
    assert!(failed.is_empty(), "gogi checks failed: {failed:#?}");
    for name in HEAD_CHECKS {
        let status: Vec<Status> = report
            .checks
            .iter()
            .filter(|check| check.name == name)
            .map(|check| check.status)
            .collect();
        assert_eq!(status, [Status::Ok], "the check `{name}` runs and passes");
    }
}
