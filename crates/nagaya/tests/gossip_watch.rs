mod support;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nagaya::gossip::{Gossip, GossipConfig};
use support::{WAIT, wait_for_view};

fn config(name: &str, seed: Option<&Gossip>) -> GossipConfig {
    let mut config = GossipConfig::new(name, "127.0.0.1:0".parse().unwrap());
    config.interval = Duration::from_millis(50);
    config.seeds = seed.map(|gossip| vec![gossip.local_addr()]).unwrap_or_default();
    config
}

/// Wait until `lines` holds a line for which `ready` is true. Panic with the lines.
fn wait_line(lines: &Arc<Mutex<Vec<String>>>, what: &str, ready: impl Fn(&str) -> bool) {
    let deadline = Instant::now() + WAIT;
    loop {
        if lines.lock().unwrap().iter().any(|line| ready(line)) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "no line for {what}; the lines:\n{}",
            lines.lock().unwrap().join("\n")
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn two_peers_with_one_name_are_two_peers_in_the_log() {
    let watcher = Gossip::start(config("watcher", None)).unwrap();
    let lines = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = Arc::clone(&lines);
    watcher.watch_peers(move |line| sink.lock().unwrap().push(line));
    let first = Gossip::start(config("twin", Some(&watcher))).unwrap();
    let second = Gossip::start(config("twin", Some(&watcher))).unwrap();
    let (id_first, id_second) = (first.id(), second.id());
    assert_ne!(id_first, id_second);
    wait_for_view(&watcher, "both twins live", |view| view.live.len() == 3);

    let live = |id: &str| format!("gossip: peer {id} is live");
    wait_line(&lines, "the first twin live", |line| line == live(&id_first));
    wait_line(&lines, "the second twin live", |line| line == live(&id_second));

    // One twin stops. The log calls only that twin dead.
    first.shutdown();
    wait_line(&lines, "the first twin dead", |line| {
        line == format!("gossip: peer {id_first} is dead")
    });
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        !lines
            .lock()
            .unwrap()
            .iter()
            .any(|line| line == &format!("gossip: peer {id_second} is dead")),
        "the second twin runs: {:?}",
        lines.lock().unwrap()
    );
    second.shutdown();
    watcher.shutdown();
}
