mod support;

use std::time::{Duration, Instant};

use nagaya::gossip::{ClusterView, Gossip, GossipConfig, Key, NodeView};
use nagaya::node::{Node, duplicate_names};
use serde_json::json;
use support::{WAIT, wait_for_view};

fn config(name: &str) -> GossipConfig {
    let mut config = GossipConfig::new(name, "127.0.0.1:0".parse().unwrap());
    config.interval = Duration::from_millis(50);
    config
}

fn node(name: &str, seed: Option<&Gossip>) -> Node {
    let mut config = config(name);
    config.seeds = seed.map(|gossip| vec![gossip.local_addr()]).unwrap_or_default();
    Node::new(Gossip::start(config).unwrap(), "127.0.0.1:1".parse().unwrap())
}

/// The value of `name:conflict` of the node itself.
fn own_conflict(node: &Node) -> Option<serde_json::Value> {
    node.gossip()
        .view()
        .live
        .iter()
        .find(|peer| peer.is_self)
        .and_then(|peer| peer.value(&Key::NameConflict))
}

/// Call `check_name` as `main` does, until `ready` accepts the key. Panic with the view when the wait ends.
fn poll(node: &mut Node, what: &str, ready: impl Fn(Option<serde_json::Value>) -> bool) -> Option<serde_json::Value> {
    let deadline = Instant::now() + WAIT;
    loop {
        node.check_name();
        let value = own_conflict(node);
        if ready(value.clone()) {
            return value;
        }
        if Instant::now() > deadline {
            panic!(
                "no {what} in {WAIT:?}; the last view of {}:\n{}",
                node.gossip().node_name(),
                node.gossip().view()
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn two_nodes_with_one_name_see_the_conflict() {
    let mut one = node("twin", None);
    let mut two = node("twin", Some(one.gossip()));
    let (id_one, id_two) = (one.gossip().id(), two.gossip().id());
    assert_ne!(id_one, id_two);

    let on_one = poll(&mut one, "conflict key on one", |value| value.is_some());
    assert_eq!(on_one, Some(json!({"own": id_one, "peers": [id_two]})));
    let on_two = poll(&mut two, "conflict key on two", |value| value.is_some());
    assert_eq!(on_two, Some(json!({"own": id_two, "peers": [id_one]})));
    one.shutdown();
    two.shutdown();
}

#[test]
fn nodes_with_other_names_have_no_conflict() {
    let mut one = node("host-one", None);
    let mut two = node("host-two", Some(one.gossip()));
    wait_for_view(one.gossip(), "host-two live", |view| view.live.len() == 2);
    // A conflict would show within a few rounds of 50 ms. Check for 1 s.
    for _ in 0..40 {
        one.check_name();
        two.check_name();
        std::thread::sleep(Duration::from_millis(25));
    }
    for node in [&one, &two] {
        let view = node.gossip().view();
        assert!(
            view.live.iter().all(|peer| !peer.keys.contains_key("name:conflict")),
            "{view}"
        );
    }
    one.shutdown();
    two.shutdown();
}

#[test]
fn the_key_goes_when_the_other_node_stops() {
    let mut one = node("twin", None);
    let two = node("twin", Some(one.gossip()));
    poll(&mut one, "conflict key on one", |value| value.is_some());
    two.shutdown();
    poll(&mut one, "conflict key removal after the peer stops", |value| {
        value.is_none()
    });
    one.shutdown();
}

fn peer(name: &str, generation: u64, addr: &str, is_self: bool) -> NodeView {
    NodeView {
        name: name.to_owned(),
        id: format!("{name}:{generation}:{addr}"),
        generation,
        addr: addr.parse().unwrap(),
        is_self,
        keys: Default::default(),
    }
}

fn view(nodes: Vec<NodeView>) -> ClusterView {
    ClusterView {
        live: nodes,
        dead: Vec::new(),
    }
}

#[test]
fn only_a_peer_that_is_not_an_earlier_run_is_a_duplicate() {
    let addr: std::net::SocketAddr = "127.0.0.1:9000".parse().unwrap();
    let cases = [
        (
            "an earlier run on this address",
            peer("twin", 5, "127.0.0.1:9000", false),
            false,
        ),
        (
            "another address, lower generation",
            peer("twin", 5, "127.0.0.1:9001", false),
            true,
        ),
        (
            "another address, higher generation",
            peer("twin", 20, "127.0.0.1:9001", false),
            true,
        ),
        (
            "this address, higher generation",
            peer("twin", 20, "127.0.0.1:9000", false),
            true,
        ),
        (
            "this address, equal generation",
            peer("twin", 10, "127.0.0.1:9000", false),
            true,
        ),
        ("another name", peer("other", 5, "127.0.0.1:9000", false), false),
    ];
    for (what, other, duplicate) in cases {
        let nodes = view(vec![peer("twin", 10, "127.0.0.1:9000", true), other.clone()]);
        let found = duplicate_names(&nodes, "twin", addr, 10);
        let expected = if duplicate { vec![other.id.clone()] } else { Vec::new() };
        assert_eq!(found, expected, "{what}");
    }
}

#[test]
fn a_restart_on_the_same_address_has_no_conflict_in_a_running_cluster() {
    let mut watcher = node("watcher", None);
    let mut first_config = config("twin");
    first_config.seeds = vec![watcher.gossip().local_addr()];
    let first = Gossip::start(first_config).unwrap();
    let (address, old_id) = (first.local_addr(), first.id());
    wait_for_view(watcher.gossip(), "the first run live", |view| {
        view.live.iter().any(|node| node.id == old_id)
    });
    first.shutdown();

    std::thread::sleep(Duration::from_millis(5));
    let mut second_config = config("twin");
    second_config.bind = address;
    second_config.seeds = vec![watcher.gossip().local_addr()];
    let mut second = Node::new(Gossip::start(second_config).unwrap(), "127.0.0.1:1".parse().unwrap());
    let new_id = second.gossip().id();
    assert_ne!(old_id, new_id);
    // The failure detector may hold the first run as live or as dead. Neither is a conflict.
    wait_for_view(second.gossip(), "the watcher live", |view| view.live.len() == 2);
    for _ in 0..40 {
        second.check_name();
        watcher.check_name();
        std::thread::sleep(Duration::from_millis(25));
    }
    assert_eq!(own_conflict(&second), None, "{}", second.gossip().view());
    second.shutdown();
    watcher.shutdown();
}
