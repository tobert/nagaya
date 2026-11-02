//! The join of the cluster: what a node publishes for its peers (spec 2.3, "What a node publishes for the cluster").

mod cluster_support;

use std::net::SocketAddr;
use std::time::Duration;

use cluster_support::{Member, WAIT, toy_profile};
use nagaya::gossip::{ClusterView, Gossip, GossipConfig, Key};

fn observer(seed: SocketAddr) -> Gossip {
    let mut config = GossipConfig::new("observer", "127.0.0.1:0".parse().unwrap());
    config.interval = Duration::from_millis(50);
    config.seeds = vec![seed];
    Gossip::start(config).unwrap()
}

/// Wait until `ready` accepts the view. Panic with the last view.
fn wait_for(gossip: &Gossip, what: &str, ready: impl Fn(&ClusterView) -> bool) -> ClusterView {
    let deadline = std::time::Instant::now() + WAIT;
    loop {
        let view = gossip.view();
        if ready(&view) {
            return view;
        }
        assert!(std::time::Instant::now() < deadline, "no view with {what}:\n{view}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_node_publishes_its_served_models_and_its_server() {
    let mut node = Member::start("host-a", &[], &toy_profile(Some("toy-a")));
    let watcher = observer(node.gossip);
    let view = wait_for(&watcher, "the server key and a served model of host-a", |view| {
        view.live.iter().any(|peer| {
            peer.name == "host-a"
                && peer.keys.contains_key("server")
                && peer
                    .keys
                    .get("engine:toy")
                    .is_some_and(|value| value.contains("\"served\":[{"))
        })
    });
    let peer = view.live.iter().find(|peer| peer.name == "host-a").unwrap();
    let engine = peer.value(&Key::Engine("toy".to_owned())).unwrap();
    let served = engine["served"].as_array().expect("`served` is a list");
    assert_eq!(served.len(), 1, "{engine}");
    let entry = &served[0];
    assert_eq!(entry["model"], "toy-a");
    assert_eq!(entry["tokenizer_hash"], "fake:0");
    assert!(
        entry.get("status").is_none(),
        "the state of the engine gives the status: {entry}"
    );
    assert!(
        entry["capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c == "generate")
    );
    let server = peer.value(&Key::Server).unwrap();
    assert_eq!(server["http"], node.http.to_string());
    let identity = node.identity();
    assert_eq!(server["limits"], identity["limits"]);
    assert_eq!(server["capabilities"], identity["capabilities"]);
    assert_eq!(node.terminate().code(), Some(0));
    watcher.shutdown();
}

#[test]
fn a_node_with_no_engine_publishes_its_server() {
    let mut node = Member::start("host-empty", &[], &toy_profile(None));
    let watcher = observer(node.gossip);
    let view = wait_for(&watcher, "the server key of host-empty", |view| {
        view.live
            .iter()
            .any(|peer| peer.name == "host-empty" && peer.keys.contains_key("server"))
    });
    let peer = view.live.iter().find(|peer| peer.name == "host-empty").unwrap();
    assert_eq!(peer.value(&Key::Server).unwrap()["http"], node.http.to_string());
    assert_eq!(node.terminate().code(), Some(0));
    watcher.shutdown();
}

#[test]
fn an_engine_value_past_8192_bytes_refuses_the_start() {
    let model = "m".repeat(9000);
    let mut node = Member::spawn("host-big", &[], &toy_profile(Some(&model)));
    let status = node.wait_exit(WAIT);
    let lines = node.lines.drain().join("\n");
    assert_eq!(status.code(), Some(3), "{lines}");
    assert!(
        lines.contains("the value of gossip key `engine:toy` has") && lines.contains("the bound is 8192 bytes"),
        "the refusal names the key and the size:\n{lines}"
    );
}
