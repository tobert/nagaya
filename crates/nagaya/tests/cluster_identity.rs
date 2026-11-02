//! The identity of the cluster (spec 11.1a, "The identity of the cluster") and the keeper of a clashing name (spec
//! 2.3, "A served model name in the cluster"), with nagaya processes on loopback.

mod cluster_support;

use std::time::Duration;

use cluster_support::{Member, WAIT, toy_profile};
use nagaya::gossip::{Gossip, GossipConfig, Key};
use serde_json::{Value, json};

/// The entry of `model` in an identity.
fn entry<'a>(identity: &'a Value, model: &str) -> &'a Value {
    identity["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["model"] == model)
        .unwrap_or_else(|| panic!("no entry {model} in {identity:#}"))
}

#[test]
fn each_node_lists_the_served_models_of_both_nodes_in_byte_order() {
    // The names sort in the other order than the nodes, so the order is the order of `model`.
    let mut a = Member::start("host-a", &[], &toy_profile(Some("zeta")));
    let mut b = Member::start("host-b", &[a.gossip], &toy_profile(Some("alpha")));
    let want = [("alpha", "ready"), ("zeta", "ready")];
    let on_a = a.wait_models(&want);
    let on_b = b.wait_models(&want);
    // A peer's entry is the entry that the peer lists for itself.
    assert_eq!(entry(&on_a, "alpha"), entry(&on_b, "alpha"));
    assert_eq!(entry(&on_a, "zeta"), entry(&on_b, "zeta"));
    // Both nodes declare pins, with one limit, and both list the extensions.
    assert_eq!(on_a["limits"], on_b["limits"]);
    assert!(on_a["capabilities"].as_array().unwrap().contains(&json!("pins")));
    assert_eq!(on_a["extensions"], json!(["/nagaya/v1"]));
    assert_eq!(a.terminate().code(), Some(0));
    assert_eq!(b.terminate().code(), Some(0));
}

#[test]
fn a_node_with_no_engine_lists_the_models_and_the_pins_of_its_peer() {
    let mut peer = Member::start("host-a", &[], &toy_profile(Some("toy-a")));
    let mut coordinator = Member::start("host-c", &[peer.gossip], &toy_profile(None));
    let identity = coordinator.wait_models(&[("toy-a", "ready")]);
    let own = peer.identity();
    assert_eq!(identity["limits"]["pin_bytes"], own["limits"]["pin_bytes"]);
    assert!(identity["capabilities"].as_array().unwrap().contains(&json!("pins")));
    assert_eq!(identity["extensions"], json!(["/nagaya/v1"]));
    assert_eq!(coordinator.terminate().code(), Some(0));
    assert_eq!(peer.terminate().code(), Some(0));
}

#[test]
fn a_model_whose_node_leaves_stays_listed_as_unloaded() {
    let mut a = Member::start("host-a", &[], &toy_profile(Some("toy-a")));
    let mut b = Member::start("host-b", &[a.gossip], &toy_profile(Some("toy-b")));
    a.wait_models(&[("toy-a", "ready"), ("toy-b", "ready")]);
    // A kill writes no state, so only the failure detector takes the node out of the live set.
    b.kill();
    a.lines.wait_for_line("gossip: peer host-b:", " is dead", WAIT);
    a.wait_models(&[("toy-a", "ready"), ("toy-b", "unloaded")]);
    assert_eq!(a.terminate().code(), Some(0));
}

#[test]
fn the_node_with_the_lower_name_keeps_a_clashing_name() {
    let mut a = Member::start("host-a", &[], &toy_profile(Some("same")));
    let mut b = Member::start("host-b", &[a.gossip], &toy_profile(Some("same")));
    let mut config = GossipConfig::new("observer", "127.0.0.1:0".parse().unwrap());
    config.interval = Duration::from_millis(50);
    config.seeds = vec![a.gossip];
    let watcher = Gossip::start(config).unwrap();
    // Each node logs the clash, with the model and both nodes.
    let line = "error: served model `same` has one name on node host-a and node host-b; node host-a keeps the name";
    a.lines.wait_for(line, WAIT);
    b.lines.wait_for(line, WAIT);
    // Each node lists the name once.
    a.wait_models(&[("same", "ready")]);
    b.wait_models(&[("same", "ready")]);
    // The node that lost the name writes `model:conflict`, and the keeper writes none.
    let deadline = std::time::Instant::now() + WAIT;
    let view = loop {
        let view = watcher.view();
        if view
            .live
            .iter()
            .any(|node| node.name == "host-b" && node.keys.contains_key("model:conflict"))
        {
            break view;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "no model:conflict of host-b:\n{view}"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    let node_b = view.live.iter().find(|node| node.name == "host-b").unwrap();
    assert_eq!(
        node_b.value(&Key::ModelConflict).unwrap(),
        json!({"models": [{"model": "same", "kept_by": "host-a"}]})
    );
    let node_a = view.live.iter().find(|node| node.name == "host-a").unwrap();
    assert!(!node_a.keys.contains_key("model:conflict"), "{view}");
    // The engine of the node that lost the name stays, with its key.
    assert!(node_b.keys.contains_key("engine:toy"));
    // When the keeper leaves, the clash ends: the other node serves the name and removes the key.
    a.kill();
    b.wait_models(&[("same", "ready")]);
    let deadline = std::time::Instant::now() + WAIT;
    loop {
        let view = watcher.view();
        let gone = view
            .live
            .iter()
            .any(|node| node.name == "host-b" && !node.keys.contains_key("model:conflict"));
        if gone {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "host-b keeps model:conflict:\n{view}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(b.terminate().code(), Some(0));
    watcher.shutdown();
}
