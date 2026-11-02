//! A served model name is unique in the cluster (spec 2.3, "A served model name in the cluster", and "The check of a
//! name"): a profile that adds a name that a live peer serves is refused, at a `PUT` and at the start.

mod cluster_support;

use cluster_support::{Member, WAIT, send, toy_profile};
use nagaya::profile::PlacementProfile;
use serde_json::Value;

/// The profile of `toy_profile` as the JSON of a `PUT`.
fn profile_json(model: Option<&str>) -> Value {
    serde_json::to_value(PlacementProfile::parse(&toy_profile(model)).unwrap()).unwrap()
}

/// Two nodes that serve `toy-a` and `toy-b`, each listing both.
fn pair() -> (Member, Member) {
    let a = Member::start("host-a", &[], &toy_profile(Some("toy-a")));
    let b = Member::start("host-b", &[a.gossip], &toy_profile(Some("toy-b")));
    let want = [("toy-a", "ready"), ("toy-b", "ready")];
    a.wait_models(&want);
    b.wait_models(&want);
    (a, b)
}

#[test]
fn a_put_that_adds_a_name_of_a_live_peer_is_a_400_that_names_the_model_and_the_peer() {
    let (_a, b) = pair();
    let reply = send(
        b.http,
        "PUT",
        "/nagaya/v1/profile",
        Some(&profile_json(Some("toy-a"))),
        &[],
    );
    assert_eq!(reply.status, 400, "{}", reply.text());
    let error = &reply.json()["error"];
    assert_eq!(error["param"], "profile");
    let message = error["message"].as_str().unwrap();
    assert!(
        message.contains("toy-a") && message.contains("host-a"),
        "the refusal names the model and the peer: {message}"
    );
    // Nothing changed.
    let active = send(b.http, "GET", "/nagaya/v1/profile", None, &[]);
    assert_eq!(active.json()["engines"][0]["config"]["model"], "toy-b");
}

#[test]
fn a_put_of_a_name_that_no_peer_serves_is_not_refused_for_its_name() {
    let (_a, b) = pair();
    let reply = send(
        b.http,
        "PUT",
        "/nagaya/v1/profile",
        Some(&profile_json(Some("toy-c"))),
        &[],
    );
    assert_eq!(reply.status, 202, "{}", reply.text());
}

#[test]
fn a_start_with_a_name_of_a_live_peer_exits_three() {
    let a = Member::start("host-a", &[], &toy_profile(Some("toy-a")));
    // The open waits, so the gossip of host-a reaches host-b before the check.
    let slow = toy_profile(Some("toy-a")).replace(
        "[engines.config]\nmodel = \"toy-a\"\n",
        "[engines.config]\nmodel = \"toy-a\"\nopen_ms = 4000\n",
    );
    assert!(slow.contains("open_ms"), "{slow}");
    let mut b = Member::spawn("host-b", &[a.gossip], &slow);
    let status = b.wait_exit(WAIT);
    let lines = b.lines.drain().join("\n");
    assert_eq!(status.code(), Some(3), "{lines}");
    assert!(
        lines.contains("toy-a") && lines.contains("host-a"),
        "the refusal names the model and the peer:\n{lines}"
    );
}
