mod support;

use std::net::SocketAddr;
use std::time::Duration;

use nagaya::gossip::{ClusterView, Gossip, GossipConfig, Key};
use nagaya::profile::PlacementProfile;
use nagaya::swap::desired_value;
use serde_json::Value;
use support::{ACTIVE_PROFILE, Proc, WAIT, wait_for_view};

/// A gossip node that posts desired profiles, as a client machine does (spec 2.3).
fn poster(name: &str, seed: SocketAddr) -> Gossip {
    let mut config = GossipConfig::new(name, "127.0.0.1:0".parse().unwrap());
    config.interval = Duration::from_millis(50);
    config.seeds = vec![seed];
    Gossip::start(config).unwrap()
}

/// A profile of fake engines. Each engine needs 256 bytes of host memory. `device` is the device budget.
fn profile(engines: &[&str], device: u64) -> PlacementProfile {
    let mut text = String::new();
    for name in engines {
        text.push_str(&format!(
            "[[engines]]\nname = \"{name}\"\nkind = \"fake\"\ndevices = [0]\nshare = 1\n"
        ));
    }
    text.push_str(&format!(
        "[[budgets]]\ntiers = [\"device(0)\"]\nbytes = {device}\n[[budgets]]\ntiers = [\"host\"]\nbytes = {}\n\
         [store]\nnvme = 65536\n",
        256 * engines.len()
    ));
    PlacementProfile::parse(&text).unwrap()
}

/// The value of a key of the node `name`, when the view has it.
fn value(view: &ClusterView, name: &str, key: &Key) -> Option<Value> {
    view.live.iter().find(|node| node.name == name)?.value(key)
}

/// Wait until the process is ready and the poster sees it.
fn ready(poster: &Gossip) {
    wait_for_view(poster, "host-x with a ready engine", |view| {
        value(view, "host-x", &Key::Engine("fake-one".to_owned())).is_some_and(|engine| engine["state"] == "ready")
    });
}

#[test]
fn a_desired_profile_that_changes_the_engines_stops_the_process_with_exit_four() {
    let mut process = Proc::spawn("host-x", "127.0.0.1:18090", &[]);
    let poster = poster("poster", process.gossip_addr);
    ready(&poster);
    let wanted = profile(&["fake-one", "fake-two"], 8192);
    poster
        .publish(&Key::ProfileDesired("host-x".to_owned()), &desired_value(1, &wanted))
        .unwrap();

    let status = process.wait_exit(WAIT);
    assert_eq!(status.code(), Some(4), "stderr:\n{}", process.lines.drain().join("\n"));
    assert_eq!(PlacementProfile::read_active(&process.store).unwrap(), wanted);
    assert_eq!(
        std::fs::read_to_string(process.store.join("profile.version"))
            .unwrap()
            .trim(),
        "1"
    );
    // The exit follows the drain and the park of spec 10.3.
    process.lines.wait_for("the stop ", Duration::from_secs(1));
    poster.shutdown();
}

#[test]
fn a_desired_profile_that_does_not_fit_is_refused_and_the_process_runs_on() {
    let mut process = Proc::spawn("host-x", "127.0.0.1:18091", &[]);
    let poster = poster("poster", process.gossip_addr);
    ready(&poster);
    let key = Key::ProfileDesired("host-x".to_owned());
    let too_small = profile(&["fake-one", "fake-two"], 1);
    poster.publish(&key, &desired_value(1, &too_small)).unwrap();

    let view = wait_for_view(&poster, "the refusal of host-x", |view| {
        value(view, "host-x", &Key::ProfileRefused).is_some()
    });
    let refusal = value(&view, "host-x", &Key::ProfileRefused).unwrap();
    assert_eq!(refusal["version"], 1);
    assert_eq!(refusal["shortfall"][0]["tiers"], serde_json::json!(["device(0)"]));
    assert!(refusal["shortfall"][0]["bytes"].as_u64().unwrap() > 0, "{refusal}");
    // The refusal names the shortfall in the log too.
    process
        .lines
        .wait_for("refused desired profile version 1", Duration::from_secs(5));
    // The active profile stays, on the wire and in the file.
    let active: PlacementProfile =
        serde_json::from_value(value(&view, "host-x", &Key::ProfileActive).unwrap()).unwrap();
    assert_eq!(active, PlacementProfile::parse(ACTIVE_PROFILE).unwrap());
    assert_eq!(
        std::fs::read_to_string(process.store.join("profile.toml")).unwrap(),
        ACTIVE_PROFILE
    );
    assert!(!process.store.join("profile.version").exists());
    assert_eq!(process.try_exit(), None, "the process runs on");

    // A newer profile that fits and keeps the engines clears the refusal and becomes the active profile.
    let fits = profile(&["fake-one"], 8192);
    poster.publish(&key, &desired_value(2, &fits)).unwrap();
    let view = wait_for_view(&poster, "the new active profile of host-x", |view| {
        value(view, "host-x", &Key::ProfileActive).is_some_and(|active| active["budgets"][0]["bytes"] == 8192)
    });
    assert_eq!(value(&view, "host-x", &Key::ProfileRefused), None, "{view}");
    assert_eq!(PlacementProfile::read_active(&process.store).unwrap(), fits);
    assert_eq!(
        process.try_exit(),
        None,
        "a profile that keeps the engines does not restart"
    );
    assert_eq!(process.terminate().code(), Some(0));
    poster.shutdown();
}

#[test]
fn an_older_version_from_a_second_peer_does_not_replace_a_newer_one() {
    let mut process = Proc::spawn("host-x", "127.0.0.1:18092", &[]);
    let newer = poster("poster-a", process.gossip_addr);
    let older = poster("poster-b", process.gossip_addr);
    ready(&newer);
    let key = Key::ProfileDesired("host-x".to_owned());
    // The older version sorts last and would restart the process.
    older
        .publish(&key, &desired_value(3, &profile(&["fake-one", "fake-two"], 8192)))
        .unwrap();
    newer
        .publish(&key, &desired_value(5, &profile(&["fake-one"], 8192)))
        .unwrap();

    wait_for_view(&newer, "the version 5 profile of host-x", |view| {
        value(view, "host-x", &Key::ProfileActive).is_some_and(|active| active["budgets"][0]["bytes"] == 8192)
    });
    assert_eq!(process.try_exit(), None, "version 3 does not stop the process");
    assert_eq!(
        std::fs::read_to_string(process.store.join("profile.version"))
            .unwrap()
            .trim(),
        "5"
    );
    assert_eq!(process.terminate().code(), Some(0));
    newer.shutdown();
    older.shutdown();
}
