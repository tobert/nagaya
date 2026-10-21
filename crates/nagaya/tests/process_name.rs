mod support;

use nagaya::gossip::{Gossip, GossipConfig, Key};
use serde_json::{Value, json};
use support::{Proc, WAIT, wait_for_view};

fn observer(name: &str, seed: std::net::SocketAddr) -> Gossip {
    let mut config = GossipConfig::new(name, "127.0.0.1:0".parse().unwrap());
    config.interval = std::time::Duration::from_millis(50);
    config.seeds = vec![seed];
    Gossip::start(config).unwrap()
}

/// The value of `name:conflict` of the node at `addr`, in the view of `gossip`.
fn conflict(gossip: &Gossip, addr: std::net::SocketAddr) -> Option<Value> {
    gossip
        .view()
        .live
        .iter()
        .find(|node| node.addr == addr)?
        .value(&Key::NameConflict)
}

#[test]
fn two_processes_with_one_name_log_and_publish_the_conflict() {
    let mut one = Proc::spawn("twin", &[]);
    let mut two = Proc::spawn("twin", &[one.gossip_addr]);
    let watcher = observer("observer", one.gossip_addr);

    // Each process logs one error that names its own id and the id of the other.
    let line_one = one.lines.wait_for("error: duplicate node name `twin`: ", WAIT);
    let line_two = two.lines.wait_for("error: duplicate node name `twin`: ", WAIT);
    let view = wait_for_view(&watcher, "the conflict key of both processes", |_| {
        conflict(&watcher, one.gossip_addr).is_some() && conflict(&watcher, two.gossip_addr).is_some()
    });
    let on_one = conflict(&watcher, one.gossip_addr).unwrap();
    let on_two = conflict(&watcher, two.gossip_addr).unwrap();
    let (id_one, id_two) = (on_one["own"].as_str().unwrap(), on_two["own"].as_str().unwrap());
    assert_ne!(id_one, id_two, "{view}");
    assert_eq!(on_one, json!({"own": id_one, "peers": [id_two]}));
    assert_eq!(on_two, json!({"own": id_two, "peers": [id_one]}));
    assert!(line_one.contains(id_one) && line_one.contains(id_two), "{line_one}");
    assert!(line_two.contains(id_one) && line_two.contains(id_two), "{line_two}");

    // The process logs the conflict once for each peer, and it runs on.
    std::thread::sleep(std::time::Duration::from_millis(500));
    let count = |lines: &[String]| lines.iter().filter(|line| line.contains("duplicate node name")).count();
    assert_eq!(count(one.lines.drain()), 1);
    assert_eq!(count(two.lines.drain()), 1);
    assert_eq!(one.terminate().code(), Some(0));
    assert_eq!(two.terminate().code(), Some(0));
    watcher.shutdown();
}
