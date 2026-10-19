mod support;

use std::time::Duration;

use nagaya::gossip::{Gossip, GossipConfig};
use support::wait_for_view;

fn config(name: &str) -> GossipConfig {
    let mut config = GossipConfig::new(name, "127.0.0.1:0".parse().unwrap());
    config.interval = Duration::from_millis(50);
    config
}

#[test]
fn a_peer_that_stops_leaves_the_live_view_and_is_marked_dead() {
    let one = Gossip::start(config("host-one")).unwrap();
    let mut seeded = config("host-two");
    seeded.seeds = vec![one.local_addr()];
    let two = Gossip::start(seeded).unwrap();
    wait_for_view(&one, "host-two live", |view| {
        view.live.iter().any(|node| node.name == "host-two")
    });
    assert_eq!(one.view().dead, Vec::<String>::new(), "no node is dead while both run");

    two.shutdown();
    let view = wait_for_view(&one, "host-two dead and not live", |view| {
        !view.live.iter().any(|node| node.name == "host-two") && view.dead.iter().any(|name| name == "host-two")
    });
    assert_eq!(view.live.len(), 1, "{view}");
    assert!(view.live[0].is_self);
    one.shutdown();
}
