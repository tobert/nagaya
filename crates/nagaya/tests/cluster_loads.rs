//! The node of a load route (spec 3.3b, "The node of a load route"): a load or an unload goes to the node that holds
//! the engine.

mod peer_support;

use peer_support::{ENGINE, Has, Node, bare_peer, refused_addr, with_engine};

fn load(node: &Node, query: &str) -> peer_support::Reply {
    node.send("POST", &format!("/nagaya/v1/engines/{ENGINE}/load{query}"), &[], &[])
}

#[test]
fn a_load_of_an_engine_of_one_peer_goes_to_that_peer() {
    let a = Node::start("host-a", Has::Objects);
    let b = Node::start("host-b", Has::Model);
    a.see(vec![with_engine(b.as_peer(), ENGINE)]);
    for query in ["", "?host=host-b"] {
        let reply = load(&a, query);
        assert_eq!(reply.status, 200, "{query}: {}", reply.text());
        assert_eq!(reply.json()["engine"], ENGINE);
        assert_eq!(reply.json()["status"], "ready");
    }
    let unload = a.send(
        "POST",
        &format!("/nagaya/v1/engines/{ENGINE}/unload?host=host-b"),
        &[],
        &[],
    );
    assert_ne!(unload.status, 404, "the unload reached host-b: {}", unload.text());
}

#[test]
fn a_host_that_no_live_node_has_is_a_404_that_names_host() {
    let a = Node::start("host-a", Has::Objects);
    let reply = load(&a, "?host=host-d");
    assert_eq!(reply.status, 404, "{}", reply.text());
    assert_eq!(reply.json()["error"]["param"], "host");
}

#[test]
fn the_own_host_is_a_local_load() {
    let a = Node::start("host-a", Has::Model);
    let reply = load(&a, "?host=host-a");
    assert_eq!(reply.status, 200, "{}", reply.text());
}

#[test]
fn two_peers_with_the_engine_are_a_400_that_names_host_and_lists_them() {
    let a = Node::start("host-a", Has::Objects);
    let b = Node::start("host-b", Has::Model);
    let c = Node::start("host-c", Has::Model);
    a.see(vec![with_engine(b.as_peer(), ENGINE), with_engine(c.as_peer(), ENGINE)]);
    let reply = load(&a, "");
    assert_eq!(reply.status, 400, "{}", reply.text());
    assert_eq!(reply.json()["error"]["param"], "host");
    let message = reply.json()["error"]["message"].as_str().unwrap().to_owned();
    assert!(message.contains("host-b") && message.contains("host-c"), "{message}");
    assert_eq!(load(&a, "?host=host-c").status, 200);
}

#[test]
fn the_local_engine_comes_first() {
    // host-c publishes the engine too, at an address with no listener: a forward to it would be a 503.
    let b = Node::start("host-b", Has::Model);
    b.see(vec![with_engine(bare_peer("host-c", refused_addr()), ENGINE)]);
    assert_eq!(load(&b, "").status, 200);
}

#[test]
fn a_peer_that_does_not_connect_is_a_503_that_names_it() {
    let a = Node::start("host-a", Has::Objects);
    a.see(vec![with_engine(bare_peer("host-c", refused_addr()), ENGINE)]);
    let reply = load(&a, "");
    assert_eq!(reply.status, 503, "{}", reply.text());
    assert_eq!(reply.json()["error"]["type"], "unavailable");
    assert!(reply.text().contains("host-c"), "{}", reply.text());
}

#[test]
fn an_engine_that_no_node_has_is_the_404_of_an_unknown_name() {
    let a = Node::start("host-a", Has::Model);
    let reply = a.send("POST", "/nagaya/v1/engines/nowhere/load", &[], &[]);
    assert_eq!(reply.status, 404, "{}", reply.text());
    assert_eq!(reply.json()["error"]["param"], "name");
}
