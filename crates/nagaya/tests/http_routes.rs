//! The council routes in the process, with the fake engine and the toy family (spec 9.2, 9.4, 11, 11.5).

mod council_support;

use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::{ENGINE, Fixture, TermFactory};
use gogi_proto::decision::{ControlTextHit, Signals};
use gogi_proto::identity::{Capability, ModelStatus, ServerIdentity};
use nagaya::council::{Service, ServiceConfig};
use nagaya::http::{Council, Served, Server, keep_gogi_places, router};
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};
use tower::ServiceExt;

/// A council on a fixture's process. The service is a second one on the process, with the same store directory.
fn council(f: &Fixture) -> Arc<Council> {
    let service = f.attach_again();
    Arc::new(
        Council::new(
            vec![Served {
                service,
                aliases: vec!["tiny".to_owned()],
                device: Some("fake-0".to_owned()),
            }],
            None,
        )
        .unwrap(),
    )
}

/// Send one request through the router on this thread. Give the status, the request-id header and the JSON body.
fn send(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, String, Value) {
    send_with(council, method, path, body, &[])
}

fn send_with(
    council: &Arc<Council>,
    method: &str,
    path: &str,
    body: Option<Value>,
    headers: &[(&str, &str)],
) -> (StatusCode, String, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut request = Request::builder().method(method).uri(path);
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        let body = body.map_or_else(Body::empty, |b| Body::from(b.to_string()));
        let response = router(Arc::clone(council))
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let id = response.headers()["request-id"].to_str().unwrap().to_owned();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
        let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, id, value)
    })
}

fn context() -> Value {
    json!({"messages": [{"role": "user", "content": "Delete the build directory.", "snap": true}]})
}

#[test]
fn identity_lists_the_served_model_with_its_limits_capabilities_and_status() {
    let f = Fixture::new();
    let council = council(&f);
    let (status, id, body) = send(&council, "GET", "/council/v2/identity", None);
    assert_eq!(status, StatusCode::OK);
    assert!(gogi_proto::ids::is_request_id(&id), "{id}");
    let identity: ServerIdentity = serde_json::from_value(body).unwrap();
    let model = &identity.models[0];
    assert_eq!((model.model.as_str(), model.status), ("toy", ModelStatus::Ready));
    assert_eq!(model.aliases, ["tiny"]);
    assert!(model.engine.starts_with("nagaya:"), "{}", model.engine);
    assert_eq!(model.limits.context_tokens, 4096);
    assert!(model.limits.snapshot_bytes > 0);
    for capability in [
        Capability::Reads,
        Capability::Generate,
        Capability::DryRun,
        Capability::LeaveOneOut,
        Capability::Continuation,
    ] {
        assert!(model.has(capability), "{capability:?}");
    }
    assert!(!model.has(Capability::Ride), "riding decisions are not built");
    assert!(identity.has(Capability::Pins) && identity.has(Capability::Classes));
    assert!(identity.limits.pin_bytes.is_some());
}

/// A refused request: method, path, body, status, error type and param.
type Case<'a> = (&'a str, &'a str, Option<Value>, StatusCode, &'a str, Option<&'a str>);

#[test]
fn every_error_has_gogis_body_and_repeats_the_request_id() {
    let f = Fixture::new();
    let council = council(&f);
    let unknown_snapshot = format!("/council/v2/snapshots/snap:{}", "0".repeat(64));
    let unknown_spec = format!("/council/v2/specs/sha256:{}", "0".repeat(64));
    let cases: Vec<Case> = vec![
        (
            "GET",
            "/council/v2/nothing",
            None,
            StatusCode::NOT_FOUND,
            "not_found",
            None,
        ),
        (
            "DELETE",
            "/council/v2/identity",
            None,
            StatusCode::NOT_FOUND,
            "not_found",
            None,
        ),
        (
            "POST",
            "/council/v2/contexts",
            Some(json!({"context": context()})),
            StatusCode::BAD_REQUEST,
            "invalid_request",
            Some("model"),
        ),
        (
            "POST",
            "/council/v2/contexts",
            Some(json!({"model": "gpt-2", "context": context()})),
            StatusCode::BAD_REQUEST,
            "invalid_request",
            Some("model"),
        ),
        (
            "POST",
            "/council/v2/contexts",
            Some(json!({"model": "toy", "context": context(), "extra": 1})),
            StatusCode::BAD_REQUEST,
            "invalid_request",
            Some("extra"),
        ),
        (
            "POST",
            "/council/v2/contexts",
            Some(json!("x")),
            StatusCode::BAD_REQUEST,
            "invalid_request",
            None,
        ),
        ("GET", &unknown_snapshot, None, StatusCode::NOT_FOUND, "not_found", None),
        (
            "GET",
            "/council/v2/snapshots/nope",
            None,
            StatusCode::NOT_FOUND,
            "not_found",
            None,
        ),
        ("GET", &unknown_spec, None, StatusCode::NOT_FOUND, "not_found", None),
        (
            "GET",
            "/council/v2/pins/nobody",
            None,
            StatusCode::NOT_FOUND,
            "not_found",
            None,
        ),
        (
            "PUT",
            "/council/v2/pins/someone",
            Some(json!({"snapshots": [], "ttl_s": 0})),
            StatusCode::BAD_REQUEST,
            "invalid_request",
            Some("ttl_s"),
        ),
        (
            "POST",
            "/council/v2/generations",
            Some(json!({"model": "toy", "context": context(), "max_tokens": 5000})),
            StatusCode::BAD_REQUEST,
            "invalid_request",
            Some("max_tokens"),
        ),
    ];
    for (method, path, body, want, kind, param) in cases {
        let (status, id, value) = send(&council, method, path, body);
        assert_eq!(status, want, "{method} {path}: {value}");
        assert_eq!(value["error"]["type"], kind, "{method} {path}: {value}");
        assert!(value["error"]["message"].is_string(), "{method} {path}: {value}");
        assert_eq!(value["request_id"], json!(id), "{method} {path}");
        if let Some(param) = param {
            assert_eq!(value["error"]["param"], param, "{method} {path}: {value}");
        }
    }
}

#[test]
fn a_build_routes_by_alias_and_a_dry_run_with_no_held_head_names_its_recipe() {
    let f = Fixture::new();
    let council = council(&f);
    let (status, _, dry) = send(
        &council,
        "POST",
        "/council/v2/contexts",
        Some(json!({"model": "tiny", "context": context(), "dry_run": true})),
    );
    assert_eq!(status, StatusCode::OK, "{dry}");
    assert_eq!(dry["model"], "toy");
    assert!(dry["head"].as_str().unwrap().starts_with("recipe:"), "{dry}");
    let (_, _, built) = send(
        &council,
        "POST",
        "/council/v2/contexts",
        Some(json!({"model": "toy", "context": context()})),
    );
    assert!(built["head"].as_str().unwrap().starts_with("blake3:"), "{built}");
    let (_, _, again) = send(
        &council,
        "POST",
        "/council/v2/contexts",
        Some(json!({"model": "toy", "context": context(), "dry_run": true})),
    );
    assert_eq!(again["head"], built["head"], "a dry run reports a held head");
    let path = format!("/council/v2/snapshots/{}", built["head"].as_str().unwrap());
    let (status, _, snapshot) = send(&council, "GET", &path, None);
    assert_eq!(status, StatusCode::OK);
    assert_eq!(snapshot["model"], "toy");
}

#[test]
fn a_draining_council_refuses_a_request_for_its_model_with_a_503() {
    let f = Fixture::new();
    let council = council(&f);
    council.drain();
    let (status, _, value) = send(
        &council,
        "POST",
        "/council/v2/contexts",
        Some(json!({"model": "toy", "context": context()})),
    );
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(value["error"]["type"], "model_unavailable");
    assert_eq!(
        (value["error"]["model"].as_str(), value["error"]["status"].as_str()),
        (Some("toy"), Some("draining"))
    );
    let (_, _, identity) = send(&council, "GET", "/council/v2/identity", None);
    assert_eq!(identity["models"][0]["status"], "draining");
}

#[test]
fn a_council_serves_at_most_one_model() {
    let f = Fixture::new();
    let served = |service: Service| Served {
        service,
        aliases: Vec::new(),
        device: None,
    };
    let two = vec![served(f.attach_again()), served(f.attach_again())];
    assert!(Council::new(two, None).is_err());
}

#[test]
fn a_council_with_no_model_has_an_identity_and_refuses_specs_with_a_503() {
    let council = Arc::new(Council::new(Vec::new(), None).unwrap());
    let (status, _, identity) = send(&council, "GET", "/council/v2/identity", None);
    assert_eq!(status, StatusCode::OK);
    assert_eq!(identity["models"], json!([]));
    assert!(identity["limits"]["pin_bytes"].is_null());
    let (status, _, value) = send(&council, "GET", "/council/v2/specs", None);
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(value["error"]["type"], "unavailable");
}

#[test]
fn only_gogis_places_of_control_text_stay_in_a_response() {
    let hit = |place: &str| ControlTextHit {
        r#where: place.to_owned(),
        token: "<end>".to_owned(),
    };
    let mut signals = Signals {
        control_text: [
            "case",
            "message:3",
            "spec",
            "suffix:0",
            "suffix:",
            "suffix:x",
            "suffix:12",
        ]
        .iter()
        .map(|place| hit(place))
        .collect(),
    };
    keep_gogi_places(&mut signals);
    let kept: Vec<&str> = signals.control_text.iter().map(|hit| hit.r#where.as_str()).collect();
    assert_eq!(kept, ["case", "spec", "suffix:0", "suffix:12"]);
}

#[test]
fn a_closed_stream_cancels_the_generation() {
    // Each pass takes 20 ms, so a reply of 400 tokens would take 8 s.
    let factory = TermFactory {
        delay: Duration::from_millis(20),
        ..TermFactory::toy()
    };
    let f = Fixture::with_factory(factory, 1 << 20, None, ServiceConfig::new("toy", ENGINE));
    let service = Service::attach(
        f.process.ports(),
        Arc::new(ToyFamily::new()),
        f.directory.path(),
        ServiceConfig::new("toy", ENGINE),
    )
    .unwrap();
    let slots = service.free_slots();
    let council = Arc::new(
        Council::new(
            vec![Served {
                service,
                aliases: Vec::new(),
                device: None,
            }],
            None,
        )
        .unwrap(),
    );
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let server = Server::start(runtime.handle(), "127.0.0.1:0".parse().unwrap(), Arc::clone(&council)).unwrap();
    // A context with thinking off, and a large bias toward text: the reply is long.
    let body =
        json!({"model": "toy", "context": context(), "max_tokens": 400, "sample": {"temperature": 0}, "stream": true})
            .to_string();
    let mut connection = std::net::TcpStream::connect(server.local_addr()).unwrap();
    write!(
        connection,
        "POST /council/v2/generations HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut reader = BufReader::new(connection.try_clone().unwrap());
    let mut head = String::new();
    reader.read_line(&mut head).unwrap();
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    let started = Instant::now();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line.starts_with("event: token") {
            break;
        }
        assert!(
            !line.starts_with("event: done"),
            "the reply ended before the test closed the stream"
        );
    }
    connection.shutdown(std::net::Shutdown::Both).unwrap();
    drop(reader);
    // The generation ends at its next token, and its slot is free again.
    let deadline = Instant::now() + Duration::from_secs(3);
    let service = &council;
    while council_free(service) != slots {
        assert!(
            Instant::now() < deadline,
            "the generation did not stop after the stream closed"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(started.elapsed() < Duration::from_secs(4));
    server.stop(Duration::from_secs(1));
}

/// A stop cuts the requests that are still open after its wait. No connection is served after the budget (spec 11.7).
#[test]
fn a_stop_cuts_a_stream_that_is_open_after_its_wait() {
    // Each pass takes 20 ms, so a reply of 400 tokens would take 8 s.
    let factory = TermFactory {
        delay: Duration::from_millis(20),
        ..TermFactory::toy()
    };
    let f = Fixture::with_factory(factory, 1 << 20, None, ServiceConfig::new("toy", ENGINE));
    let service = Service::attach(
        f.process.ports(),
        Arc::new(ToyFamily::new()),
        f.directory.path(),
        ServiceConfig::new("toy", ENGINE),
    )
    .unwrap();
    let council = Arc::new(
        Council::new(
            vec![Served {
                service,
                aliases: Vec::new(),
                device: None,
            }],
            None,
        )
        .unwrap(),
    );
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let server = Server::start(runtime.handle(), "127.0.0.1:0".parse().unwrap(), council).unwrap();
    let address = server.local_addr();
    let body =
        json!({"model": "toy", "context": context(), "max_tokens": 400, "sample": {"temperature": 0}, "stream": true})
            .to_string();
    let mut connection = std::net::TcpStream::connect(address).unwrap();
    connection.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    write!(
        connection,
        "POST /council/v2/generations HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut reader = BufReader::new(connection.try_clone().unwrap());
    let mut line = String::new();
    while !line.starts_with("event: token") {
        line.clear();
        assert_ne!(
            reader.read_line(&mut line).unwrap(),
            0,
            "the stream ended before a token"
        );
    }
    let stopped = Instant::now();
    server.stop(Duration::from_millis(100));
    // The reply would run for 8 s. The connection must end soon after the stop.
    let mut ended = false;
    let deadline = stopped + Duration::from_secs(3);
    while Instant::now() < deadline {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => {
                ended = true;
                break;
            }
            Ok(_) => {}
        }
    }
    assert!(
        ended,
        "the stream was still served {:?} after the stop",
        stopped.elapsed()
    );
    assert!(
        std::net::TcpStream::connect(address).is_err(),
        "the server accepts a connection after the stop"
    );
}

/// The free slots of the served model of a council, read through identity's served model.
fn council_free(council: &Arc<Council>) -> usize {
    council.free_slots()
}

/// A call thread that cannot pin to its CPUs panics, and the route turns the panic into a 500 `internal`. The council
/// records the failure, so the process stops with exit 1 (spec 3.2a, 8.5). It does not abort.
#[test]
fn a_call_thread_that_cannot_pin_is_a_500_that_the_council_records() {
    let f = Fixture::new();
    // CPU 1023 is the last bit of a mask. No machine has it online, so `sched_setaffinity` refuses the mask.
    let council = Arc::new(
        Council::new(
            vec![Served {
                service: f.attach_again(),
                aliases: Vec::new(),
                device: None,
            }],
            Some(vec![1023]),
        )
        .unwrap(),
    );
    assert!(council.failure().is_none());
    let (status, _, value) = send(
        &council,
        "POST",
        "/council/v2/contexts",
        Some(json!({"model": "toy", "context": context()})),
    );
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{value}");
    assert_eq!(value["error"]["type"], "internal", "{value}");
    assert!(council.failure().is_some(), "the council did not record the failure");
    let streamed = json!({"model": "toy", "context": context(), "max_tokens": 5, "stream": true});
    let (status, _, value) = send(&council, "POST", "/council/v2/generations", Some(streamed));
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{value}");
}

/// The council reports the failure of a service: a failed release of a ledger charge stops the process (spec 4.2, 8.5).
#[test]
fn the_council_reports_a_failed_release_of_a_service() {
    let f = Fixture::new();
    let council = council(&f);
    assert_eq!(council.failure(), None);
    let ports = f.process.ports();
    let staging = nagaya::council::staging::Staging::take(&ports, ENGINE, 10).unwrap();
    ports.release_staging(nagaya_abi::Tier::Host, 10).unwrap();
    drop(staging);
    assert!(council.failure().is_some(), "the main loop would not see the failure");
}

/// Send one request with a body that the test builds, and give the status and the error body.
fn send_body(council: &Arc<Council>, body: Body) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method("POST")
            .uri("/council/v2/contexts")
            .body(body)
            .unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

#[test]
fn a_body_past_the_limit_is_a_413_that_names_the_body() {
    let f = Fixture::new();
    let council = council(&f);
    let (status, value) = send_body(&council, Body::from(vec![b' '; nagaya::http::MAX_BODY_BYTES + 1]));
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{value}");
    assert_eq!(value["error"]["type"], "too_large", "{value}");
    assert_eq!(value["error"]["param"], "body", "{value}");
}

#[test]
fn a_body_that_does_not_arrive_whole_is_a_400_invalid_request() {
    let f = Fixture::new();
    let council = council(&f);
    let chunks: Vec<Result<&'static [u8], std::io::Error>> = vec![
        Ok(&b"{\"model\": \"to"[..]),
        Err(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "the client went away",
        )),
    ];
    let (status, value) = send_body(&council, Body::from_stream(futures_util::stream::iter(chunks)));
    assert_eq!(status, StatusCode::BAD_REQUEST, "{value}");
    assert_eq!(value["error"]["type"], "invalid_request", "{value}");
    assert_eq!(value["error"]["param"], "body", "{value}");
}

/// Kaibo review 2026-10-20, unverified items. `Service::read` takes the longest suffix with `.expect`, and the Qwen3.8
/// queue read takes the queue name with `.expect`. gogi's `validate` and `Context::check` must refuse the bad request
/// first. These requests reach no `.expect`: each is a 400 that names the parameter.
#[test]
fn a_read_with_no_suffix_is_a_400_before_any_family_code_runs() {
    let f = Fixture::new();
    let council = council(&f);
    for (suffixes, param) in [(json!([]), "suffixes"), (json!([""]), "suffixes")] {
        let body = json!({"model": "toy", "contexts": [context()], "suffixes": suffixes, "letters": ["A", "B"]});
        let (status, _, value) = send(&council, "POST", "/council/v2/reads", Some(body));
        assert_eq!(status, StatusCode::BAD_REQUEST, "{value}");
        assert_eq!(value["error"]["param"], param, "{value}");
    }
}

/// gogi "Continuations": `continue` is a boolean, and any other value is a 400 that names `continue`. The body does
/// not decode, so the error comes before any route code runs.
#[test]
fn a_continue_that_is_not_a_boolean_is_a_400_that_names_continue() {
    let f = Fixture::new();
    let council = council(&f);
    let mut open = context();
    open["messages"]
        .as_array_mut()
        .unwrap()
        .push(json!({"role": "assistant", "content": "Plan:", "continue": "yes"}));
    let spec = format!("sha256:{}", "0".repeat(64));
    for (path, body) in [
        (
            "/council/v2/generations",
            json!({"model": "toy", "context": open, "max_tokens": 4}),
        ),
        (
            "/council/v2/contexts",
            json!({"model": "toy", "context": open, "dry_run": true}),
        ),
        (
            "/council/v2/decisions",
            json!({"model": "toy", "spec_id": spec, "case": "x", "contexts": [open]}),
        ),
        (
            "/council/v2/reads",
            json!({"model": "toy", "contexts": [open], "suffixes": ["x"], "letters": ["A", "B"]}),
        ),
    ] {
        let (status, _, value) = send(&council, "POST", path, Some(body));
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {value}");
        assert_eq!(value["error"]["param"], "continue", "{path}: {value}");
    }
}

#[test]
fn a_decision_context_whose_queue_names_no_tool_is_a_400() {
    let f = Fixture::new();
    let council = council(&f);
    let mut queued = context();
    queued["queue"] = json!("next_command");
    let body = json!({"model": "toy", "spec_id": format!("sha256:{}", "0".repeat(64)), "case": "x",
        "contexts": [queued]});
    let (status, _, value) = send(&council, "POST", "/council/v2/decisions", Some(body));
    assert_eq!(status, StatusCode::BAD_REQUEST, "{value}");
    assert_eq!(value["error"]["param"], "contexts", "{value}");
}

#[test]
fn a_build_whose_snapshot_copy_the_ledger_refuses_is_a_429_busy_that_names_the_shortfall() {
    use nagaya_abi::Tier;

    let f = Fixture::new();
    let council = council(&f);
    let ports = f.process.ports();
    let free = ports
        .unassigned()
        .into_iter()
        .find(|(tiers, _)| tiers.contains(&Tier::Host))
        .map(|(_, bytes)| bytes)
        .unwrap();
    ports.admit_staging("toy", Tier::Host, free - 100).unwrap();
    let body = json!({"model": "toy", "context": context()});
    let (status, _, value) = send(&council, "POST", "/council/v2/contexts", Some(body));
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{value}");
    assert_eq!(value["error"]["type"], "busy", "{value}");
    assert!(
        value["error"]["message"].as_str().unwrap().contains("short by"),
        "{value}"
    );
}
