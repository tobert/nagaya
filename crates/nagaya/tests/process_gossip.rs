mod support;

use std::net::SocketAddr;
use std::time::Duration;

use nagaya::gossip::{Gossip, GossipConfig, Key};
use serde_json::json;
use support::{Proc, WAIT, wait_for_view};

fn observer(seed: SocketAddr) -> Gossip {
    let mut config = GossipConfig::new("observer", "127.0.0.1:0".parse().unwrap());
    config.interval = Duration::from_millis(50);
    config.seeds = vec![seed];
    Gossip::start(config).unwrap()
}

const KEYS: &str = "engine:fake-one,inventory,profile:active";

#[test]
fn two_processes_see_the_keys_of_each_other() {
    let mut one = Proc::spawn("host-one", &[]);
    let mut two = Proc::spawn("host-two", &[one.gossip_addr]);
    let watcher = observer(one.gossip_addr);

    // The observer reads the values that each process wrote.
    let view = wait_for_view(&watcher, "both processes with a ready engine", |view| {
        ["host-one", "host-two"].iter().all(|name| {
            view.live
                .iter()
                .any(|node| node.name == *name && node.keys.get("engine:fake-one").is_some_and(|v| v.contains("ready")))
        })
    });
    for (name, http) in [("host-one", one.http.as_str()), ("host-two", two.http.as_str())] {
        let node = view.live.iter().find(|node| node.name == name).unwrap();
        let names: Vec<&str> = node.keys.keys().map(String::as_str).collect();
        assert_eq!(names.join(","), KEYS, "{view}");
        let engine = node.value(&Key::Engine("fake-one".to_owned())).unwrap();
        assert_eq!(engine["state"], "ready");
        assert_eq!(engine["http"], http);
        assert_eq!(engine["identity"].as_str().unwrap().len(), 64, "a BLAKE3 digest in hex");
        assert_eq!(node.value(&Key::Inventory).unwrap()["devices"][0]["name"], "fake-0");
        assert_eq!(
            node.value(&Key::ProfileActive).unwrap()["engines"][0]["name"],
            "fake-one"
        );
    }

    // Each process sees the other (its stderr says so).
    one.lines
        .wait_for_line("gossip: peer host-two:", &format!(" keys {KEYS}"), WAIT);
    two.lines
        .wait_for_line("gossip: peer host-one:", &format!(" keys {KEYS}"), WAIT);
    assert_eq!(json!(one.terminate().code()), json!(0));
    assert_eq!(json!(two.terminate().code()), json!(0));
    watcher.shutdown();
}

#[test]
fn a_stopped_process_is_marked_dead_by_the_other() {
    let mut one = Proc::spawn("host-one", &[]);
    let mut two = Proc::spawn("host-two", &[one.gossip_addr]);
    let watcher = observer(one.gossip_addr);
    wait_for_view(&watcher, "host-two live", |view| {
        view.live.iter().any(|node| node.name == "host-two")
    });
    one.lines.wait_for("gossip: peer host-two:", WAIT);

    assert_eq!(two.terminate().code(), Some(0));
    // The failure detector (phi-accrual) marks the peer dead; the live view loses it.
    let view = wait_for_view(&watcher, "host-two dead and not live", |view| {
        !view.live.iter().any(|node| node.name == "host-two") && view.dead.iter().any(|name| name == "host-two")
    });
    assert!(view.live.iter().any(|node| node.name == "host-one"), "{view}");
    one.lines.wait_for_line("gossip: peer host-two:", " is dead", WAIT);
    assert_eq!(one.terminate().code(), Some(0));
    watcher.shutdown();
}
