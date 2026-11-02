//! The forward of spec 11.1a between nagaya processes on loopback: a forwarded reply has the bytes of a direct one.

mod cluster_support;

use cluster_support::{Member, generation, send, toy_profile};
use serde_json::json;

/// Two nodes that each serve one toy model and list both.
fn pair() -> (Member, Member) {
    let a = Member::start("host-a", &[], &toy_profile(Some("toy-a")));
    let b = Member::start("host-b", &[a.gossip], &toy_profile(Some("toy-b")));
    let want = [("toy-a", "ready"), ("toy-b", "ready")];
    a.wait_models(&want);
    b.wait_models(&want);
    (a, b)
}

/// A generation of `model` with its own prompt, so that no earlier request of a test warmed the store for it.
fn prompt(model: &str, text: &str, stream: bool) -> serde_json::Value {
    let mut body = generation(model, stream);
    body["context"]["messages"][0]["content"] = json!(text);
    body
}

/// The text with each `"tps":<number>` of a token event cut out. The rate is a clock reading of the target, so two
/// runs differ in it (gogi "Streams"). A body with no stream has no rate, and it stays as it is.
fn without_tps(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find("\"tps\":") {
        out.push_str(&rest[..at]);
        let after = &rest[at + 6..];
        let end = after.find(['}', ',']).unwrap_or(after.len());
        out.push_str("\"tps\":_");
        rest = &after[end..];
    }
    out.push_str(rest);
    out
}

#[test]
fn a_forwarded_generation_has_the_bytes_of_a_direct_one() {
    // The usage of a reply counts the tokens that the store kept, so each reply runs on a cold store: the direct ones
    // on one pair of nodes, and the forwarded ones on another.
    let (mut a1, mut b1) = pair();
    let (mut a2, mut b2) = pair();
    for (stream, text) in [(false, "hello there"), (true, "good morning")] {
        let direct = send(
            b1.http,
            "POST",
            "/council/v2/generations",
            Some(&prompt("toy-b", text, stream)),
            &[],
        );
        let forwarded = send(
            a2.http,
            "POST",
            "/council/v2/generations",
            Some(&prompt("toy-b", text, stream)),
            &[],
        );
        assert_eq!(direct.status, 200, "{}", direct.text());
        assert_eq!(forwarded.status, direct.status);
        assert_eq!(forwarded.header("content-type"), direct.header("content-type"));
        assert_eq!(
            without_tps(&forwarded.text()),
            without_tps(&direct.text()),
            "stream {stream}"
        );
        if stream {
            assert!(forwarded.text().contains("event: done"), "{}", forwarded.text());
        }
        // The other direction forwards too.
        let direct = send(
            a1.http,
            "POST",
            "/council/v2/generations",
            Some(&prompt("toy-a", text, stream)),
            &[],
        );
        let forwarded = send(
            b2.http,
            "POST",
            "/council/v2/generations",
            Some(&prompt("toy-a", text, stream)),
            &[],
        );
        assert_eq!(
            without_tps(&forwarded.text()),
            without_tps(&direct.text()),
            "stream {stream}"
        );
    }
    for member in [&mut a1, &mut b1, &mut a2, &mut b2] {
        assert_eq!(member.terminate().code(), Some(0));
    }
}

#[test]
fn a_forwarded_request_is_not_forwarded_again() {
    let (mut a, mut b) = pair();
    let body = generation("toy-b", false);
    let reply = send(
        a.http,
        "POST",
        "/council/v2/generations",
        Some(&body),
        &[("nagaya-forwarded", "host-c"), ("nagaya-request-id", "req_c_1")],
    );
    assert_eq!(reply.status, 503, "{}", reply.text());
    let error = reply.json();
    assert_eq!(error["error"]["type"], "model_unavailable");
    assert_eq!(error["error"]["status"], "unloaded");
    assert_eq!(reply.header("request-id"), Some("req_c_1"));
    assert_eq!(a.terminate().code(), Some(0));
    assert_eq!(b.terminate().code(), Some(0));
}

#[test]
fn a_peer_that_is_gone_gives_a_503_unloaded() {
    let (mut a, mut b) = pair();
    b.kill();
    let reply = send(
        a.http,
        "POST",
        "/council/v2/generations",
        Some(&generation("toy-b", false)),
        &[],
    );
    assert_eq!(reply.status, 503, "{}", reply.text());
    let error = reply.json();
    assert_eq!(error["error"]["type"], "model_unavailable");
    assert_eq!(error["error"]["status"], "unloaded");
    assert_eq!(error["error"]["model"], "toy-b");
    assert_eq!(a.terminate().code(), Some(0));
}

#[test]
fn a_node_with_no_family_forwards_the_text_routes() {
    let mut b = Member::start("host-b", &[], &toy_profile(Some("toy-b")));
    let mut c = Member::start("host-c", &[b.gossip], &toy_profile(None));
    c.wait_models(&[("toy-b", "ready")]);
    let body = json!({"model": "toy-b", "text": "hello there"});
    let direct = send(b.http, "POST", "/nagaya/v1/tokenize", Some(&body), &[]);
    let forwarded = send(c.http, "POST", "/nagaya/v1/tokenize", Some(&body), &[]);
    assert_eq!(direct.status, 200, "{}", direct.text());
    assert_eq!(forwarded.text(), direct.text());
    assert_eq!(c.terminate().code(), Some(0));
    assert_eq!(b.terminate().code(), Some(0));
}
