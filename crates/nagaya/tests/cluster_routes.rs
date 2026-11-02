//! The routes with no model in the cluster (spec 11.1a, "The routes with no model"): reads by id ask the peers on a
//! miss, and pins, keeps and the delete of a spec go to each live node.

mod peer_support;

use gogi_proto::SnapshotId;
use gogi_proto::context::Context;
use nagaya::council::Schedule;
use peer_support::{Has, Node, link, object};
use serde_json::{Value, json};

fn spec(name: &str) -> Value {
    json!({"name": name, "instructions": "Judge it.", "input_label": "Statement", "questions": [
        {"id": "verdict", "type": "choice", "instructions": "What happens.",
         "criteria": [{"option": "allow", "means": "fine"}, {"option": "ask", "means": "not fine"}]}]})
}

fn send(node: &Node, method: &str, path: &str, body: Option<&Value>) -> peer_support::Reply {
    let text = body.map(Value::to_string).unwrap_or_default();
    node.send(method, path, text.as_bytes(), &[("content-type", "application/json")])
}

/// A held snapshot on `node`, built from `text`.
fn built(node: &Node, text: &str) -> SnapshotId {
    let context: Context =
        serde_json::from_value(json!({"messages": [{"role": "user", "content": text, "snap": true}]})).unwrap();
    let service = &node.fixture.as_ref().unwrap().service;
    service
        .build(&context, false, Schedule::default())
        .unwrap()
        .head
        .unwrap()
}

fn pair() -> (Node, Node) {
    let a = Node::start("host-a", Has::Both);
    let b = Node::start("host-b", Has::Both);
    link(&[&a, &b]);
    (a, b)
}

#[test]
fn a_spec_of_a_peer_reads_by_id_and_the_list_stays_local() {
    let (a, b) = pair();
    let held = send(&b, "POST", "/council/v2/specs", Some(&spec("on-b")));
    assert_eq!(held.status, 200, "{}", held.text());
    let spec_id = held.json()["spec_id"].as_str().unwrap().to_owned();
    let read = send(&a, "GET", &format!("/council/v2/specs/{spec_id}"), None);
    assert_eq!(read.status, 200, "{}", read.text());
    assert_eq!(read.json()["spec_id"], json!(spec_id));
    let list = send(&a, "GET", "/council/v2/specs", None);
    assert_eq!(list.json()["specs"], json!([]), "the list names the local specs only");
    let missing = format!("/council/v2/specs/blake3:{}", "0".repeat(64));
    assert_eq!(send(&a, "GET", &missing, None).status, 404);
}

#[test]
fn a_delete_of_a_spec_goes_to_each_node() {
    let (a, b) = pair();
    let held = send(&b, "POST", "/council/v2/specs", Some(&spec("on-b")));
    let spec_id = held.json()["spec_id"].as_str().unwrap().to_owned();
    let path = format!("/council/v2/specs/{spec_id}");
    let deleted = send(&a, "DELETE", &path, None);
    assert_eq!(deleted.status, 204, "{}", deleted.text());
    assert_eq!(send(&b, "GET", &path, None).status, 404, "the peer dropped the spec");
    assert_eq!(send(&a, "DELETE", &path, None).status, 404, "no node holds it now");
}

#[test]
fn a_snapshot_of_a_peer_reads_by_id() {
    let (a, b) = pair();
    let id = built(&b, "a context that only host-b built");
    let read = send(&a, "GET", &format!("/council/v2/snapshots/{id}"), None);
    assert_eq!(read.status, 200, "{}", read.text());
    assert_eq!(read.json()["id"], json!(id.to_string()));
    let missing = format!("/council/v2/snapshots/blake3:{}", "0".repeat(64));
    assert_eq!(send(&a, "GET", &missing, None).status, 404);
}

#[test]
fn pins_go_to_each_node_and_their_answers_join() {
    let (a, b) = pair();
    let on_a = built(&a, "a context of host-a");
    let on_b = built(&b, "a context of host-b");
    let nowhere = format!("blake3:{}", "0".repeat(64));
    let body = json!({"snapshots": [on_a.to_string(), on_b.to_string(), nowhere], "ttl_s": 600});
    let put = send(&a, "PUT", "/council/v2/pins/kernel-1", Some(&body));
    assert_eq!(put.status, 200, "{}", put.text());
    let held: Vec<bool> = put.json()["snapshots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["held"].as_bool().unwrap())
        .collect();
    assert_eq!(held, [true, true, false], "{}", put.text());
    assert_eq!(put.json()["expires_in_s"], json!(600));
    let got = send(&a, "GET", "/council/v2/pins/kernel-1", None);
    assert_eq!(got.status, 200, "{}", got.text());
    let mut ids: Vec<String> = serde_json::from_value(got.json()["snapshots"].clone()).unwrap();
    ids.sort();
    let mut want = vec![on_a.to_string(), on_b.to_string()];
    want.sort();
    assert_eq!(ids, want);
    // Each node pins only the snapshots that it holds.
    let own: Vec<String> = serde_json::from_value(
        b.send("GET", "/council/v2/pins/kernel-1", &[], &[("nagaya-forwarded", "test")])
            .json()["snapshots"]
            .clone(),
    )
    .unwrap();
    assert_eq!(own, [on_b.to_string()]);
    assert_eq!(send(&a, "DELETE", "/council/v2/pins/kernel-1", None).status, 204);
    assert_eq!(send(&a, "GET", "/council/v2/pins/kernel-1", None).status, 404);
    assert_eq!(send(&b, "GET", "/council/v2/pins/kernel-1", None).status, 404);
    assert_eq!(send(&a, "DELETE", "/council/v2/pins/kernel-1", None).status, 404);
}

#[test]
fn the_smallest_expiry_of_the_nodes_is_the_answer() {
    let (a, b) = pair();
    let on_a = built(&a, "a context of host-a");
    let on_b = built(&b, "a context of host-b");
    let body = json!({"snapshots": [on_a.to_string(), on_b.to_string()], "ttl_s": 600});
    assert_eq!(send(&a, "PUT", "/council/v2/pins/kernel-1", Some(&body)).status, 200);
    // Each reading of the clock of a fixture is a minute later, so host-b's pins now end sooner.
    b.fixture.as_ref().unwrap().advance(300_000);
    let local = a.send("GET", "/council/v2/pins/kernel-1", &[], &[("nagaya-forwarded", "test")]);
    let peer = b.send("GET", "/council/v2/pins/kernel-1", &[], &[("nagaya-forwarded", "test")]);
    let smallest = local.json()["expires_in_s"]
        .as_u64()
        .unwrap()
        .min(peer.json()["expires_in_s"].as_u64().unwrap());
    let joined = send(&a, "GET", "/council/v2/pins/kernel-1", None).json()["expires_in_s"]
        .as_u64()
        .unwrap();
    assert!(joined <= smallest, "{joined} > {smallest}");
    assert!(
        joined < 600 - 300,
        "the answer of host-b, which is later, sets it: {joined}"
    );
}

#[test]
fn keeps_go_to_each_node_that_holds_objects() {
    let a = Node::start("host-a", Has::Objects);
    let b = Node::start("host-b", Has::Objects);
    let m = Node::start("host-m", Has::Model);
    link(&[&a, &b, &m]);
    let (path, id) = object(b"a sound that host-b holds");
    assert_eq!(
        b.send(
            "PUT",
            &path,
            b"a sound that host-b holds",
            &[("content-type", "audio/wav")]
        )
        .status,
        201
    );
    let body = json!({"objects": [id], "ttl_s": 3600});
    // host-m has no store, so it declares no `objects`, and it takes no part.
    let put = send(&m, "PUT", "/council/v2/keeps/kaijutsu", Some(&body));
    assert_eq!(put.status, 200, "{}", put.text());
    assert_eq!(put.json()["objects"][0]["held"], json!(true), "{}", put.text());
    assert_eq!(put.json()["expires_in_s"], json!(3600));
    let got = send(&a, "GET", "/council/v2/keeps/kaijutsu", None);
    assert_eq!(got.status, 200, "{}", got.text());
    assert_eq!(got.json()["objects"], json!([id]));
    assert_eq!(send(&a, "DELETE", "/council/v2/keeps/kaijutsu", None).status, 204);
    assert_eq!(send(&b, "GET", "/council/v2/keeps/kaijutsu", None).status, 404);
}

#[test]
fn a_413_of_one_node_is_the_response() {
    let a = Node::start("host-a", Has::Objects);
    let b = Node::start("host-b", Has::Objects);
    link(&[&a, &b]);
    // Two objects of 300 KiB each pass the object limit of 512 KiB, and together they pass keep_bytes on host-b.
    let one = vec![1u8; 300 << 10];
    let two = vec![2u8; 300 << 10];
    let (one_path, one_id) = object(&one);
    let (two_path, two_id) = object(&two);
    for (path, bytes) in [(&one_path, &one), (&two_path, &two)] {
        let put = b.send("PUT", path, bytes, &[("content-type", "audio/wav")]);
        assert_eq!(put.status, 201, "{}", put.text());
    }
    let body = json!({"objects": [one_id, two_id], "ttl_s": 3600});
    let put = send(&a, "PUT", "/council/v2/keeps/kaijutsu", Some(&body));
    assert_eq!(put.status, 413, "{}", put.text());
    assert!(put.text().contains("keep_bytes"), "{}", put.text());
}
