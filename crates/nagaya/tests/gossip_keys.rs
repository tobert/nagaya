mod support;

use nagaya::gossip::{Gossip, GossipConfig, Key, MAX_KEY_BYTES, MAX_VALUE_BYTES, PublishError};
use nagaya::inventory::{Device, Inventory};
use nagaya::node::Node;
use nagaya::notify::EngineState;
use nagaya::process::OpenedEngine;
use nagaya::profile::PlacementProfile;
use nagaya_abi::Engine;
use nagaya_abi::fake::FakeAlways;
use serde_json::json;
use support::wait_for_view;

fn config(name: &str) -> GossipConfig {
    let mut config = GossipConfig::new(name, "127.0.0.1:0".parse().unwrap());
    config.interval = std::time::Duration::from_millis(50);
    config
}

fn inventory() -> Inventory {
    Inventory {
        devices: vec![Device {
            id: 0,
            name: "fake-0".to_owned(),
            pci_address: "0000:03:00.0".to_owned(),
            memory_bytes: 1 << 30,
            unified: false,
        }],
        host_bytes: 1 << 30,
    }
}

fn profile() -> PlacementProfile {
    PlacementProfile::parse(
        "[[engines]]\nname = \"fake-one\"\nkind = \"fake\"\ndevices = [0]\nshare = 1\n\
         [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 4096\n\
         [[budgets]]\ntiers = [\"host\"]\nbytes = 256\n\
         [store]\nnvme = 65536\n",
    )
    .unwrap()
}

fn opened() -> OpenedEngine {
    OpenedEngine {
        name: "fake-one".to_owned(),
        thread: std::thread::current().id(),
        identity: FakeAlways::new().identity(),
        limits: FakeAlways::new().limits(),
    }
}

#[test]
fn a_key_at_the_bound_is_published_and_one_byte_more_is_refused() {
    let gossip = Gossip::start(config("solo")).unwrap();
    let prefix = "engine:".len();
    let at_the_bound = Key::Engine("k".repeat(MAX_KEY_BYTES - prefix));
    gossip.publish(&at_the_bound, &json!(1)).unwrap();
    wait_for_view(&gossip, "the key at the bound", |view| {
        view.live[0].keys.contains_key(&at_the_bound.name())
    });

    let over = Key::Engine("k".repeat(MAX_KEY_BYTES - prefix + 1));
    let error = gossip.publish(&over, &json!(1)).unwrap_err();
    assert_eq!(
        error,
        PublishError::KeyTooLong {
            key: over.name(),
            bytes: MAX_KEY_BYTES + 1
        }
    );
    assert!(
        error.to_string().contains(&over.name()),
        "the message names the key: {error}"
    );
    assert!(
        !gossip.view().live[0].keys.contains_key(&over.name()),
        "a refused key is not written"
    );
    gossip.shutdown();
}

#[test]
fn a_value_at_the_bound_is_published_and_one_byte_more_is_refused() {
    let gossip = Gossip::start(config("solo")).unwrap();
    let key = Key::Engine("big".to_owned());
    // A JSON string of n characters has n + 2 bytes.
    let at_the_bound = json!("a".repeat(MAX_VALUE_BYTES - 2));
    gossip.publish(&key, &at_the_bound).unwrap();
    let view = wait_for_view(&gossip, "the key at the bound", |view| {
        view.live[0].keys.contains_key("engine:big")
    });
    assert_eq!(view.live[0].keys["engine:big"].len(), MAX_VALUE_BYTES);

    let over = json!("a".repeat(MAX_VALUE_BYTES - 1));
    let error = gossip.publish(&Key::Engine("over".to_owned()), &over).unwrap_err();
    assert_eq!(
        error,
        PublishError::TooLarge {
            key: "engine:over".to_owned(),
            bytes: MAX_VALUE_BYTES + 1
        }
    );
    assert!(
        error.to_string().contains("engine:over"),
        "the message names the key: {error}"
    );
    assert!(
        !gossip.view().live[0].keys.contains_key("engine:over"),
        "a refused value is not written"
    );
    gossip.shutdown();
}

#[test]
fn a_pin_or_a_store_entry_has_no_gossip_key() {
    for name in [
        "pin:ctx-1",
        "store:abc",
        "entry:abc",
        "engine",
        "engine:",
        "inventory:x",
        "profile",
        "",
    ] {
        assert_eq!(Key::parse(name), None, "`{name}` is not a key of spec 2.3");
    }
    for key in [
        Key::Inventory,
        Key::ProfileActive,
        Key::ProfileRefused,
        Key::ProfileDesired("gfx1151host".to_owned()),
        Key::Engine("mk".to_owned()),
    ] {
        assert_eq!(Key::parse(&key.name()), Some(key.clone()), "{key:?} reads back");
    }
    assert_eq!(Key::Engine("mk".to_owned()).name(), "engine:mk");
    assert_eq!(Key::ProfileDesired("gfx1151host".to_owned()).name(), "profile:desired:gfx1151host");
}

#[test]
fn a_peer_sees_the_keys_of_a_node() {
    let mut a = Node::new(
        Gossip::start(config("host-a")).unwrap(),
        "127.0.0.1:8090".parse().unwrap(),
    );
    a.publish_machine(&inventory(), &profile()).unwrap();
    a.engines_loading(&["fake-one".to_owned()]).unwrap();
    let mut seeded = config("host-b");
    seeded.seeds = vec![a.gossip().local_addr()];
    let b = Gossip::start(seeded).unwrap();

    let view = wait_for_view(&b, "the loading engine of host-a", |view| {
        view.live.iter().any(|node| {
            node.name == "host-a" && node.keys.get("engine:fake-one").is_some_and(|v| v.contains("loading"))
        })
    });
    let node = view.live.iter().find(|node| node.name == "host-a").unwrap();
    assert!(node.keys.contains_key("inventory"), "{view}");
    assert!(node.keys.contains_key("profile:active"), "{view}");
    assert_eq!(
        node.value(&Key::Inventory).unwrap(),
        json!({"devices": [{"id": 0, "name": "fake-0", "pci_address": "0000:03:00.0", "memory_bytes": 1u64 << 30, "unified": false}], "host_bytes": 1u64 << 30})
    );
    let active = node.value(&Key::ProfileActive).unwrap();
    assert_eq!(active["engines"][0]["name"], "fake-one");
    assert_eq!(active["store"]["nvme"], 65536);

    a.engines_ready(&[opened()]).unwrap();
    let view = wait_for_view(&b, "the ready engine of host-a", |view| {
        view.live
            .iter()
            .any(|node| node.name == "host-a" && node.keys.get("engine:fake-one").is_some_and(|v| v.contains("ready")))
    });
    let node = view.live.iter().find(|node| node.name == "host-a").unwrap();
    let engine = node.value(&Key::Engine("fake-one".to_owned())).unwrap();
    let identity = FakeAlways::new().identity();
    let digest = nagaya::park::identity_digest(&identity);
    assert_eq!(engine["identity"], blake3::Hash::from_bytes(digest).to_hex().as_str());
    assert_eq!(engine["models"], json!([identity.model]));
    assert_eq!(engine["state"], "ready");
    assert_eq!(engine["http"], "127.0.0.1:8090");

    a.engines_state(&["fake-one".to_owned()], EngineState::Draining)
        .unwrap();
    let view = wait_for_view(&b, "the draining engine of host-a", |view| {
        view.live.iter().any(|node| {
            node.name == "host-a" && node.keys.get("engine:fake-one").is_some_and(|v| v.contains("draining"))
        })
    });
    let node = view.live.iter().find(|node| node.name == "host-a").unwrap();
    let engine = node.value(&Key::Engine("fake-one".to_owned())).unwrap();
    assert_eq!(
        engine["identity"],
        blake3::Hash::from_bytes(digest).to_hex().as_str(),
        "a new state keeps the identity"
    );
    a.shutdown();
    b.shutdown();
}

#[test]
fn a_node_writes_only_the_keys_of_spec_2_3() {
    let mut a = Node::new(
        Gossip::start(config("host-a")).unwrap(),
        "127.0.0.1:8090".parse().unwrap(),
    );
    a.publish_machine(&inventory(), &profile()).unwrap();
    a.engines_loading(&["fake-one".to_owned()]).unwrap();
    a.engines_ready(&[opened()]).unwrap();
    let mut seeded = config("host-b");
    seeded.seeds = vec![a.gossip().local_addr()];
    let b = Gossip::start(seeded).unwrap();
    let view = wait_for_view(&b, "the ready engine of host-a", |view| {
        view.live
            .iter()
            .any(|node| node.name == "host-a" && node.keys.get("engine:fake-one").is_some_and(|v| v.contains("ready")))
    });
    let node = view.live.iter().find(|node| node.name == "host-a").unwrap();
    let names: Vec<&str> = node.keys.keys().map(String::as_str).collect();
    assert_eq!(names, ["engine:fake-one", "inventory", "profile:active"], "{view}");
    for name in names {
        assert!(Key::parse(name).is_some(), "`{name}` is a key of spec 2.3");
    }
    a.shutdown();
    b.shutdown();
}

#[test]
fn the_inventory_key_holds_the_pci_address_of_each_device() {
    let gossip = Gossip::start(config("solo")).unwrap();
    let node = Node::new(gossip, "127.0.0.1:8090".parse().unwrap());
    node.publish_machine(&inventory(), &profile()).unwrap();
    wait_for_view(node.gossip(), "the inventory key", |view| {
        view.live[0].keys.contains_key(&Key::Inventory.name())
    });
    let value = node.gossip().view().live[0].keys[&Key::Inventory.name()].clone();
    let value: serde_json::Value = serde_json::from_str(&value).unwrap();
    assert_eq!(value["devices"][0]["pci_address"], "0000:03:00.0");
}
