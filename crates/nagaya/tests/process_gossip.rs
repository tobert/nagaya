mod support;

use std::net::SocketAddr;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use nagaya::gossip::{Gossip, GossipConfig, Key};
use serde_json::json;
use support::{Lines, WAIT, wait_for_view};

const SIGTERM: i32 = 15;

unsafe extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

/// A nagaya process with its own directory. The process is killed when the value drops.
struct Proc {
    _directory: tempfile::TempDir,
    child: Option<Child>,
    lines: Lines,
    gossip_addr: SocketAddr,
}

impl Proc {
    /// Start a process and wait for its gossip address.
    fn spawn(name: &str, http: &str, seeds: &[SocketAddr]) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let store = directory.path().join("store");
        std::fs::create_dir_all(&store).unwrap();
        let seeds: Vec<String> = seeds.iter().map(|seed| format!("\"{seed}\"")).collect();
        let config = directory.path().join("nagaya.toml");
        std::fs::write(
            &config,
            format!(
                "http_bind = \"{http}\"\n\
                 gossip_bind = \"127.0.0.1:0\"\n\
                 store_dir = \"{}\"\n\
                 node_name = \"{name}\"\n\
                 seed_peers = [{}]\n\
                 park_budget_secs = 1\n",
                store.display(),
                seeds.join(", ")
            ),
        )
        .unwrap();
        std::fs::write(
            store.join("profile.toml"),
            "[[engines]]\nname = \"fake-one\"\nkind = \"fake\"\ndevices = [0]\nshare = 1\n\
             [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 4096\n\
             [[budgets]]\ntiers = [\"host\"]\nbytes = 256\n\
             [store]\nnvme = 65536\n",
        )
        .unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_nagaya"))
            .env("NAGAYA_CONFIG", &config)
            .env_remove("NOTIFY_SOCKET")
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut lines = Lines::new(child.stderr.take().unwrap());
        let gossip_addr = lines.wait_for("gossip listening on ", WAIT).parse().unwrap();
        Self {
            _directory: directory,
            child: Some(child),
            lines,
            gossip_addr,
        }
    }

    /// Send SIGTERM and wait for the exit.
    fn terminate(&mut self) -> ExitStatus {
        let mut child = self.child.take().unwrap();
        // SAFETY: the child pid is live, and SIGTERM is the stop signal of spec 10.3.
        assert_eq!(unsafe { kill(child.id() as i32, SIGTERM) }, 0);
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "the process did not exit after SIGTERM");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn observer(seed: SocketAddr) -> Gossip {
    let mut config = GossipConfig::new("observer", "127.0.0.1:0".parse().unwrap());
    config.interval = Duration::from_millis(50);
    config.seeds = vec![seed];
    Gossip::start(config).unwrap()
}

const KEYS: &str = "engine:fake-one,inventory,profile:active";

#[test]
fn two_processes_see_the_keys_of_each_other() {
    let mut one = Proc::spawn("host-one", "127.0.0.1:18081", &[]);
    let mut two = Proc::spawn("host-two", "127.0.0.1:18082", &[one.gossip_addr]);
    let watcher = observer(one.gossip_addr);

    // The observer reads the values that each process wrote.
    let view = wait_for_view(&watcher, "both processes with a ready engine", |view| {
        ["host-one", "host-two"].iter().all(|name| {
            view.live
                .iter()
                .any(|node| node.name == *name && node.keys.get("engine:fake-one").is_some_and(|v| v.contains("ready")))
        })
    });
    for (name, http) in [("host-one", "127.0.0.1:18081"), ("host-two", "127.0.0.1:18082")] {
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
    one.lines.wait_for(&format!("gossip: peer host-two keys {KEYS}"), WAIT);
    two.lines.wait_for(&format!("gossip: peer host-one keys {KEYS}"), WAIT);
    assert_eq!(json!(one.terminate().code()), json!(0));
    assert_eq!(json!(two.terminate().code()), json!(0));
    watcher.shutdown();
}
