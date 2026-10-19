mod support;

use std::net::UdpSocket;

use nagaya::gossip::{Gossip, GossipConfig};
use support::wait_for_view;

fn config(name: &str) -> GossipConfig {
    GossipConfig::new(name, "127.0.0.1:0".parse().unwrap())
}

#[test]
fn a_node_with_no_seeds_binds_its_gossip_port() {
    let gossip = Gossip::start(config("solo")).unwrap();
    let addr = gossip.local_addr();
    assert_ne!(addr.port(), 0, "the node must report the port that the system gave it");
    let error = UdpSocket::bind(addr).expect_err("the gossip socket holds the port");
    assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
    gossip.shutdown();
}

#[test]
fn a_node_with_no_seeds_has_itself_in_its_view() {
    let gossip = Gossip::start(config("solo")).unwrap();
    let view = wait_for_view(&gossip, "this node alone", |view| view.live.len() == 1);
    assert_eq!(view.live[0].name, "solo");
    assert!(view.live[0].is_self);
    assert_eq!(view.live[0].addr, gossip.local_addr());
    gossip.shutdown();
}

#[test]
fn a_stopped_node_frees_its_port() {
    let gossip = Gossip::start(config("solo")).unwrap();
    let addr = gossip.local_addr();
    gossip.shutdown();
    UdpSocket::bind(addr).expect("the port is free after the shutdown");
}
