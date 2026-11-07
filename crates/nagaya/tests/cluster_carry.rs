//! What a forward takes to its target first (spec 6.15, "The objects of a forward", and 11.1a, "The routes with no
//! model"): the objects that the body names, and the spec of a decision.
//!
//! The target `host-b` is a real council that gossip shows with the model `toy-b`. It serves only `toy`, so the
//! forward itself gets its 503 of a model that it does not serve. The tests read what it holds after the forward.

mod peer_support;

use peer_support::{Has, Node, object};
use serde_json::{Value, json};

const IMAGE: &[u8] = b"an image that the request names";

/// `host-a` takes the requests, `host-b` is the target, and `host-c` is a third node.
fn nodes(target: Has) -> (Node, Node, Node) {
    let a = Node::start("host-a", Has::Both);
    let b = Node::start("host-b", target);
    let c = Node::start("host-c", Has::Objects);
    a.see(vec![b.as_peer_serving("toy-b"), c.as_peer()]);
    b.see(vec![a.as_peer(), c.as_peer()]);
    c.see(vec![a.as_peer(), b.as_peer()]);
    (a, b, c)
}

fn generation(ids: &[&str]) -> Value {
    let mut content = vec![json!({"type": "text", "text": "look"})];
    for id in ids {
        content.push(json!({"type": "object", "id": id, "mime": "image/png"}));
    }
    json!({"model": "toy-b", "context": {"thinking": true, "messages": [{"role": "user", "content": content}]}, "max_tokens": 4})
}

fn post(node: &Node, path: &str, body: &Value) -> peer_support::Reply {
    node.send(
        "POST",
        path,
        body.to_string().as_bytes(),
        &[("content-type", "application/json")],
    )
}

/// True when `node` holds the object of `path` itself.
fn holds(node: &Node, path: &str) -> bool {
    let reply = node.send("HEAD", path, &[], &[("nagaya-forwarded", "test")]);
    match reply.status {
        200 => true,
        404 => false,
        other => panic!("a HEAD on {} gave {other}", node.name),
    }
}

/// The 503 of the target: the forward ran.
fn assert_forwarded(reply: &peer_support::Reply) {
    assert_eq!(reply.status, 503, "{}", reply.text());
    assert!(
        reply.text().contains("does not forward again"),
        "the target did not answer: {}",
        reply.text()
    );
}

#[test]
fn an_object_of_the_coordinator_or_of_a_third_node_goes_to_the_target_first() {
    let (a, b, c) = nodes(Has::Both);
    let (on_a, id_a) = object(IMAGE);
    let (on_c, id_c) = object(b"an image that only host-c holds");
    assert_eq!(
        a.send("PUT", &on_a, IMAGE, &[("content-type", "image/png")]).status,
        201
    );
    assert_eq!(
        c.send(
            "PUT",
            &on_c,
            b"an image that only host-c holds",
            &[("content-type", "image/png")]
        )
        .status,
        201
    );
    assert!(!holds(&b, &on_a) && !holds(&b, &on_c));
    let reply = post(&a, "/council/v2/generations", &generation(&[&id_a, &id_c]));
    assert_forwarded(&reply);
    assert!(holds(&b, &on_a), "the object of the coordinator is on the target");
    assert!(holds(&b, &on_c), "the object of a third node is on the target");
    let read = b.send("GET", &on_c, &[], &[("nagaya-forwarded", "test")]);
    assert_eq!(read.body, b"an image that only host-c holds");
    assert_eq!(read.header("content-type"), Some("image/png"));
}

#[test]
fn an_object_that_no_node_holds_does_not_stop_the_forward() {
    let (a, b, _c) = nodes(Has::Both);
    let (path, id) = object(b"bytes that nobody put");
    let reply = post(&a, "/council/v2/generations", &generation(&[&id]));
    assert_forwarded(&reply);
    assert!(!holds(&b, &path));
}

#[test]
fn a_put_that_the_target_refuses_is_the_response_and_nothing_forwards() {
    // The target has no object store, so its HEAD and its put are 404s.
    let (a, _b, _c) = nodes(Has::Model);
    let (path, id) = object(IMAGE);
    assert_eq!(
        a.send("PUT", &path, IMAGE, &[("content-type", "image/png")]).status,
        201
    );
    let reply = post(&a, "/council/v2/generations", &generation(&[&id]));
    assert_eq!(reply.status, 404, "{}", reply.text());
    assert!(
        reply.text().contains("objects"),
        "the put's 404 of the target: {}",
        reply.text()
    );
}

#[test]
fn a_forwarded_decision_takes_its_spec_to_the_target_first() {
    let (a, b, _c) = nodes(Has::Both);
    let spec = json!({"name": "gate", "instructions": "Judge it.", "input_label": "Statement", "questions": [
        {"id": "verdict", "type": "choice", "instructions": "What happens.",
         "criteria": [{"option": "allow", "means": "fine"}, {"option": "ask", "means": "not fine"}]}]});
    let held = post(&a, "/council/v2/specs", &spec);
    assert_eq!(held.status, 200, "{}", held.text());
    let spec_id = held.json()["spec_id"].as_str().unwrap().to_owned();
    let on_b = format!("/council/v2/specs/{spec_id}");
    assert_eq!(b.send("GET", &on_b, &[], &[("nagaya-forwarded", "test")]).status, 404);
    let decision = json!({"model": "toy-b", "case": "ls", "spec_id": spec_id,
                          "contexts": [{"thinking": true, "messages": [{"role": "user", "content": "hi"}]}]});
    let reply = post(&a, "/council/v2/decisions", &decision);
    assert_forwarded(&reply);
    let read = b.send("GET", &on_b, &[], &[("nagaya-forwarded", "test")]);
    assert_eq!(read.status, 200, "{}", read.text());
    assert_eq!(read.json()["spec_id"], json!(spec_id));
}

#[test]
fn a_spec_that_the_coordinator_lacks_does_not_stop_the_forward() {
    let (a, _b, _c) = nodes(Has::Both);
    let missing = format!("blake3:{}", "0".repeat(64));
    let decision = json!({"model": "toy-b", "case": "ls", "spec_id": missing,
                          "contexts": [{"thinking": true, "messages": [{"role": "user", "content": "hi"}]}]});
    assert_forwarded(&post(&a, "/council/v2/decisions", &decision));
}
