//! A device group whose engine has two stages (engine-abi D1, spec 3.2, 7.4). The engine is the fake with a feed rule
//! in two stages. Its bits equal those of the same fake in one stage, so the one-stage process is the oracle.

mod feed_support;
mod task_support;

use std::sync::atomic::Ordering;
use std::time::Duration;

use feed_support::{FeedFactory, FeedFixture};
use gogi_proto::context::Context;
use gogi_proto::generation::GenerationRequest;
use gogi_proto::reads::ReadsRequest;
use nagaya::exit::ExitCode;
use nagaya::process::Event;
use nagaya::signal::SignalFlag;
use serde_json::{Value, json};

fn messages() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "You are a careful agent. Read the files first."}),
        json!({"role": "user", "content": "List the crates of the workspace, with one line each.", "snap": true}),
    ]
}

fn context() -> Context {
    serde_json::from_value(json!({"thinking": false, "tools": [], "messages": messages()})).unwrap()
}

fn generation() -> GenerationRequest {
    serde_json::from_value(json!({
        "model": "toy",
        "context": {"thinking": false, "tools": [], "messages": messages()},
        "max_tokens": 10,
        "sample": {"temperature": 1.0, "seed": 11},
    }))
    .unwrap()
}

fn reads() -> ReadsRequest {
    serde_json::from_value(json!({
        "model": "toy",
        "contexts": [{"thinking": false, "tools": [], "messages": messages()}],
        "suffixes": ["\nVerdict: "],
        "letters": ["A", "B", "C"],
    }))
    .unwrap()
}

#[test]
fn a_build_a_generation_and_a_read_on_two_stages_give_the_bits_of_one_stage() {
    let one = FeedFixture::new(1);
    let two = FeedFixture::new(2);
    let devices: Vec<Vec<u32>> = two
        .factory
        .opened_on
        .lock()
        .unwrap()
        .iter()
        .map(|devices| devices.iter().map(|device| device.id).collect())
        .collect();
    assert_eq!(devices, vec![vec![0, 1]], "the factory gets both devices of the group");

    let built = two.service.build(&context(), false, Default::default()).unwrap();
    assert_eq!(
        built.head,
        one.service.build(&context(), false, Default::default()).unwrap().head
    );
    assert!(built.head.is_some());

    let reply = two.service.generate(&generation(), None).unwrap();
    let expected = one.service.generate(&generation(), None).unwrap();
    assert_eq!(
        (&reply.content, &reply.reasoning),
        (&expected.content, &expected.reasoning)
    );
    assert_eq!(reply.usage.completion, expected.usage.completion);
    // A generation on an engine with a feed rule holds a decoded head (spec 6.14). The id of a
    // head is the hash of its bytes, so two stages make the state of one stage.
    assert!(reply.head.is_some(), "{reply:?}");
    assert_eq!(reply.head, expected.head);

    let read = two.service.read(&reads()).unwrap();
    assert_eq!(read.reads, one.service.read(&reads()).unwrap().reads);

    // The passes ran stage by stage on the stage threads.
    let log = two.factory.log.lock().unwrap();
    for stage in 0..2 {
        assert!(
            log.iter().any(|(ran, _, _)| *ran == stage),
            "stage {stage} ran a quantum"
        );
    }
    assert!(
        one.factory.log.lock().unwrap().is_empty(),
        "one stage has no stage path"
    );
}

/// The profile's `devices` list is in group order (spec 2.4). With `devices = [1, 0]`, stage 0 runs on the device
/// thread of machine device 1, and stage 1 on the thread of machine device 0. The device threads go in machine order,
/// so `nagaya-dev-0-1` is the thread of machine device 1. The bits stay those of one stage.
#[test]
fn an_engine_on_devices_one_then_zero_runs_each_stage_on_the_thread_of_its_device() {
    let one = FeedFixture::new(1);
    let mut factory = FeedFactory::new(2);
    factory.devices = Some(vec![1, 0]);
    let two = FeedFixture::with(factory.clone(), Duration::from_secs(10));
    let opened: Vec<Vec<u32>> = factory
        .opened_on
        .lock()
        .unwrap()
        .iter()
        .map(|devices| devices.iter().map(|device| device.id).collect())
        .collect();
    assert_eq!(opened, vec![vec![1, 0]], "the open gets the devices in group order");

    let reply = two.service.generate(&generation(), None).unwrap();
    let expected = one.service.generate(&generation(), None).unwrap();
    assert_eq!(
        (&reply.content, &reply.reasoning),
        (&expected.content, &expected.reasoning)
    );
    let read = two.service.read(&reads()).unwrap();
    assert_eq!(read.reads, one.service.read(&reads()).unwrap().reads);

    let threads = factory.threads.lock().unwrap();
    for (stage, thread) in [(0, "nagaya-dev-0-1"), (1, "nagaya-dev-0-0")] {
        assert!(threads.iter().any(|(ran, _)| *ran == stage), "stage {stage} has a call");
        let wrong: Vec<&String> = threads
            .iter()
            .filter(|(ran, name)| *ran == stage && name != thread)
            .map(|(_, name)| name)
            .collect();
        assert!(
            wrong.is_empty(),
            "stage {stage} runs only on {thread}, not on {wrong:?}"
        );
    }
}

#[test]
fn the_group_has_a_stage_thread_for_each_device() {
    let two = FeedFixture::new(2);
    let mut names: Vec<String> = task_support::thread_names()
        .into_iter()
        .filter(|name| name.starts_with("nagaya-dev-"))
        .collect();
    names.sort();
    // Other tests of this binary run their own processes, so count the threads of this group shape only once.
    names.dedup();
    assert_eq!(names, vec!["nagaya-dev-0-0", "nagaya-dev-0-1"]);
    drop(two);
}

#[test]
fn two_blocks_of_a_build_run_on_both_devices_at_once() {
    let factory = FeedFactory::new(2);
    let fixture = FeedFixture::with(factory.clone(), Duration::from_secs(10));
    factory.delay_ms.store(15, Ordering::SeqCst);
    fixture.service.build(&context(), false, Default::default()).unwrap();
    let log = factory.log.lock().unwrap();
    let blocks = log.iter().filter(|(stage, _, _)| *stage == 0).count();
    assert!(
        blocks >= 3,
        "the context takes several blocks of {} tokens",
        feed_support::BLOCK
    );
    let overlap = log.iter().filter(|(stage, _, _)| *stage == 0).any(|(_, start, end)| {
        log.iter()
            .filter(|(stage, _, _)| *stage == 1)
            .any(|(_, other_start, other_end)| start < other_end && other_start < end)
    });
    assert!(overlap, "a stage 0 quantum ran while a stage 1 quantum ran");
}

#[test]
fn a_stalled_stage_thread_stops_the_process() {
    let factory = FeedFactory::new(2);
    let fixture = FeedFixture::with(factory.clone(), Duration::from_millis(400));
    factory.delay_ms.store(3000, Ordering::SeqCst);
    let service = std::sync::Arc::new(fixture.service);
    let builder = std::sync::Arc::clone(&service);
    std::thread::spawn(move || {
        let _ = builder.build(&context(), false, Default::default());
    });
    let event = fixture
        .process
        .recv(Duration::from_secs(2))
        .expect("the watchdog reports the stall");
    assert_eq!(
        event,
        Event::Stalled {
            group: 0,
            devices: vec![0]
        }
    );
    assert_eq!(event.exit_code(), Some(ExitCode::Crash));
    factory.delay_ms.store(0, Ordering::SeqCst);
}

#[test]
fn a_stop_parks_the_states_of_both_stages() {
    let mut fixture = FeedFixture::new(2);
    fixture.service.build(&context(), false, Default::default()).unwrap();
    let report = fixture.process.stop(Duration::from_secs(5), &SignalFlag::new());
    assert_eq!(report.exit_code(), ExitCode::Clean, "{report:?}");
    assert!(report.parked.failures.is_empty(), "{report:?}");
    assert_eq!(
        report.parked.parts, 2,
        "one part of each stage in the slot that holds the context"
    );
}

/// The child of the next test: a stage panics with no plan, and the build waits for a reply that never comes.
#[test]
#[ignore = "the next test runs it in a child process"]
fn child_a_stage_panics_with_no_plan() {
    let factory = FeedFactory::new(2);
    let fixture = FeedFixture::with(factory.clone(), Duration::from_secs(10));
    factory.surprise_stage.store(1, Ordering::SeqCst);
    let _ = fixture.service.build(&context(), false, Default::default());
}

#[test]
fn a_stage_panic_with_no_plan_aborts_the_test_binary_instead_of_a_wait() {
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "child_a_stage_panics_with_no_plan",
            "--test-threads=1",
            "--nocapture",
        ])
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let started = std::time::Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if started.elapsed() > Duration::from_secs(30) {
            child.kill().unwrap();
            panic!("the child still waits after 30 s, so a panic with no plan does not fail the test");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut stderr = String::new();
    std::io::Read::read_to_string(&mut child.stderr.take().unwrap(), &mut stderr).unwrap();
    assert!(!status.success(), "{stderr}");
    assert!(stderr.contains("panicked with no plan"), "{stderr}");
}

#[test]
fn a_build_and_a_generation_on_two_stages_record_wait_and_run_durations_and_quantum_spans() {
    let capture = nagaya_telemetry::testing::Capture::start(Vec::new());
    let two = FeedFixture::new(2);
    two.service.build(&context(), false, Default::default()).unwrap();
    two.service.generate(&generation(), None).unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let wait = capture.metric("nagaya.quantum.wait.duration");
        let run = capture.metric("nagaya.quantum.run.duration");
        let wait_count: u64 = wait.iter().map(|point| point.count).sum();
        let run_count: u64 = run.iter().map(|point| point.count).sum();
        let jobs: Vec<_> = capture
            .spans()
            .into_iter()
            .filter(|span| span.name == "nagaya.job")
            .collect();
        if wait_count == run_count && wait_count > 0 && jobs.len() >= 2 {
            let quanta: Vec<_> = capture
                .spans()
                .into_iter()
                .filter(|span| span.name == "nagaya.quantum")
                .collect();
            assert!(!quanta.is_empty(), "the staged group made no nagaya.quantum span");
            for span in &quanta {
                let kind = span
                    .attributes
                    .iter()
                    .find(|item| item.key.as_str() == "kind")
                    .map(|item| item.value.to_string());
                assert_ne!(kind.as_deref(), Some("decode"));
                let has_parent = jobs.iter().any(|job| job.span_context.span_id() == span.parent_span_id);
                assert!(has_parent, "quantum span lacks a job parent: {span:?}");
            }
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the staged quanta never matched: wait={wait_count}, run={run_count}, jobs={}",
            jobs.len()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_failed_stage_quantum_logs_the_failure_and_the_machine_evidence() {
    let root = tempfile::tempdir().unwrap();
    let write = |path: &str, text: &str| {
        let path = root.path().join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    write("proc/meminfo", "MemAvailable:   1048576 kB\n");
    write(
        "proc/vmstat",
        "compact_stall 10\npgmigrate_success 0\npswpout 0\npgscan_direct 0\n",
    );
    let factory = FeedFactory::new(2);
    factory.fail_stage.store(1, Ordering::SeqCst);
    let lines = nagaya_telemetry::testing::log_lines(|| {
        let mut fixture = FeedFixture::with_machine(
            factory.clone(),
            Duration::from_secs(10),
            nagaya::evidence::Machine::at(root.path(), 4242),
        );
        let result = fixture.service.build(&context(), false, Default::default());
        assert!(result.is_err(), "stage 1 failed the build");
        fixture.process.stop(Duration::from_secs(5), &SignalFlag::new());
    });
    let failures: Vec<_> = lines
        .into_iter()
        .filter(|line| line.starts_with("quantum failed:"))
        .collect();
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("prefill quantum"), "{}", failures[0]);
    assert!(failures[0].contains("engine feed"), "{}", failures[0]);
    assert!(failures[0].contains("compact_stall +0"), "{}", failures[0]);
}
