//! The routes of a profile for another host (spec 2.4): a `PUT` publishes `profile:desired:<host>`, a `GET` reads the
//! `profile:active` of that node from gossip, and the target handles the key with the actions of 3.3.

mod cluster_support;

use std::time::{Duration, Instant};

use cluster_support::{Member, WAIT, send, toy_profile};
use nagaya::profile::PlacementProfile;
use serde_json::{Value, json};

fn profile_json(model: Option<&str>) -> Value {
    serde_json::to_value(PlacementProfile::parse(&toy_profile(model)).unwrap()).unwrap()
}

fn pair() -> (Member, Member) {
    let a = Member::start("host-a", &[], &toy_profile(Some("toy-a")));
    let b = Member::start("host-b", &[a.gossip], &toy_profile(Some("toy-b")));
    let want = [("toy-a", "ready"), ("toy-b", "ready")];
    a.wait_models(&want);
    b.wait_models(&want);
    (a, b)
}

/// Wait until `ready` accepts the value that `read` gives. Panic with the last value.
fn wait_until(what: &str, read: impl Fn() -> Value, ready: impl Fn(&Value) -> bool) -> Value {
    let deadline = Instant::now() + WAIT;
    loop {
        let value = read();
        if ready(&value) {
            return value;
        }
        assert!(
            Instant::now() < deadline,
            "no {what} in {WAIT:?}; the last value:\n{value:#}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// The node `name` in the cluster route of `member`.
fn cluster_node(member: &Member, name: &str) -> Value {
    let body = send(member.http, "GET", "/nagaya/v1/cluster", None, &[]).json();
    body["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["name"] == name)
        .cloned()
        .unwrap_or(Value::Null)
}

#[test]
fn a_put_for_another_host_changes_it_in_place_and_a_get_reads_its_active_profile() {
    let (mut a, mut b) = pair();
    let mut wanted = profile_json(Some("toy-b"));
    wanted["engines"][0]["share"] = json!(2);
    let reply = send(
        a.http,
        "PUT",
        "/nagaya/v1/profile?host=host-b&version=1",
        Some(&wanted),
        &[],
    );
    assert_eq!(reply.status, 202, "{}", reply.text());
    assert_eq!(reply.json(), json!({"host": "host-b", "version": 1}));
    // The target keeps its engine and takes the share in place, so its active profile in gossip changes.
    let active = wait_until(
        "the new share of host-b in gossip",
        || send(a.http, "GET", "/nagaya/v1/profile?host=host-b", None, &[]).json(),
        |profile| profile["engines"][0]["share"] == 2,
    );
    assert_eq!(active["engines"][0]["config"]["model"], "toy-b");
    b.lines
        .wait_for("desired profile version 1 is accepted; the change runs in place", WAIT);
    assert_eq!(
        std::fs::read_to_string(b.store.join("profile.version")).unwrap().trim(),
        "1"
    );
    // The same version again is a conflict.
    let again = send(
        a.http,
        "PUT",
        "/nagaya/v1/profile?host=host-b&version=1",
        Some(&wanted),
        &[],
    );
    assert_eq!(again.status, 409, "{}", again.text());
    assert_eq!(again.json()["error"]["type"], "conflict");
    assert_eq!(b.terminate().code(), Some(0), "the target did not restart");
    assert_eq!(a.terminate().code(), Some(0));
}

#[test]
fn the_checks_of_the_query_and_the_body() {
    let (a, _b) = pair();
    let body = profile_json(Some("toy-b"));
    let cases: [(&str, u16, &str); 5] = [
        ("/nagaya/v1/profile?host=host-b", 400, "version"),
        ("/nagaya/v1/profile?host=host-b&version=0", 400, "version"),
        ("/nagaya/v1/profile?host=host-b&version=x", 400, "version"),
        ("/nagaya/v1/profile?host=host-d&version=1", 404, "host"),
        ("/nagaya/v1/profile?host=host-a&version=1", 400, "version"),
    ];
    for (path, status, param) in cases {
        let reply = send(a.http, "PUT", path, Some(&body), &[]);
        assert_eq!(reply.status, status, "{path}: {}", reply.text());
        assert_eq!(reply.json()["error"]["param"], param, "{path}");
    }
    let list = send(
        a.http,
        "PUT",
        "/nagaya/v1/profile?host=host-b&version=1",
        Some(&json!([1, 2])),
        &[],
    );
    assert_eq!(list.status, 400, "{}", list.text());
    assert_eq!(list.json()["error"]["param"], "profile");
    let unknown = send(a.http, "GET", "/nagaya/v1/profile?host=host-d", None, &[]);
    assert_eq!(unknown.status, 404, "{}", unknown.text());
    assert_eq!(unknown.json()["error"]["param"], "host");
    // The own name is the local profile.
    let own = send(a.http, "GET", "/nagaya/v1/profile?host=host-a", None, &[]);
    assert_eq!(own.json()["engines"][0]["config"]["model"], "toy-a");
}

#[test]
fn a_desired_profile_with_a_name_of_a_live_peer_is_refused_and_the_cluster_route_shows_it() {
    let (a, b) = pair();
    // host-b would serve toy-a, which host-a serves (spec 2.3).
    let reply = send(
        a.http,
        "PUT",
        "/nagaya/v1/profile?host=host-b&version=3",
        Some(&profile_json(Some("toy-a"))),
        &[],
    );
    assert_eq!(reply.status, 202, "{}", reply.text());
    let node = wait_until(
        "the refusal of host-b in the cluster route",
        || cluster_node(&a, "host-b"),
        |node| !node["refused"].is_null(),
    );
    assert_eq!(node["refused"]["version"], 3);
    let reason = node["refused"]["reason"].as_str().unwrap();
    assert!(reason.contains("toy-a") && reason.contains("host-a"), "{reason}");
    assert_eq!(
        send(b.http, "GET", "/nagaya/v1/profile", None, &[]).json()["engines"][0]["config"]["model"],
        "toy-b",
        "the active profile stays"
    );
}
