//! Helpers for the tests of the cluster (spec 2.3, 11.1a): nagaya processes with the fake-toy engine on loopback, and
//! a small HTTP client. A wait is bounded and fails with what it saw.

#![allow(dead_code)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

#[path = "../support/mod.rs"]
mod support;
pub use support::{Lines, WAIT};

const SIGTERM: i32 = 15;

unsafe extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

/// The profile of a node with one fake-toy engine `toy` that serves `model`, or with no engine for `None`.
pub fn toy_profile(model: Option<&str>) -> String {
    let budgets = "[[budgets]]\ntiers = [\"device(0)\"]\nbytes = 4096\n\
                   [[budgets]]\ntiers = [\"host\"]\nbytes = 33554432\n\
                   [store]\nnvme = 67108864\nhost = 16777216\n";
    match model {
        Some(model) => format!(
            "[[engines]]\nname = \"toy\"\nkind = \"fake-toy\"\ndevices = [0]\nshare = 1\n\
             [engines.config]\nmodel = \"{model}\"\n{budgets}"
        ),
        None => format!("engines = []\n{budgets}"),
    }
}

/// The profile of a node with one fake engine `fake-one`, which has no family library, so it serves no model.
pub fn plain_profile() -> String {
    "[[engines]]\nname = \"fake-one\"\nkind = \"fake\"\ndevices = [0]\nshare = 1\n\
     [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 4096\n\
     [[budgets]]\ntiers = [\"host\"]\nbytes = 256\n\
     [store]\nnvme = 65536\n"
        .to_owned()
}

/// A free TCP port on loopback.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A nagaya process with its own directory. The process is killed when the value drops.
pub struct Member {
    _directory: tempfile::TempDir,
    child: Option<Child>,
    /// The node name.
    pub name: String,
    /// The lines that the process writes to stderr.
    pub lines: Lines,
    /// The HTTP address of the process.
    pub http: SocketAddr,
    /// The gossip address of the process.
    pub gossip: SocketAddr,
    /// The store directory of the process.
    pub store: PathBuf,
}

impl Member {
    /// Start a process with `profile`, seeded with `seeds`, and wait until it is ready.
    pub fn start(name: &str, seeds: &[SocketAddr], profile: &str) -> Self {
        let mut member = Self::spawn(name, seeds, profile);
        member.lines.wait_for("ready with ", WAIT);
        member
    }

    /// Start a process and wait only for its gossip address.
    pub fn spawn(name: &str, seeds: &[SocketAddr], profile: &str) -> Self {
        let http: SocketAddr = format!("127.0.0.1:{}", free_port()).parse().unwrap();
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
        std::fs::write(store.join("profile.toml"), profile).unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_nagaya"))
            .env("NAGAYA_CONFIG", &config)
            .env_remove("NOTIFY_SOCKET")
            .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut lines = Lines::new(child.stderr.take().unwrap());
        let gossip = lines.wait_for("gossip listening on ", WAIT).parse().unwrap();
        Self {
            _directory: directory,
            child: Some(child),
            name: name.to_owned(),
            lines,
            http,
            gossip,
            store,
        }
    }

    /// Send SIGTERM and wait for the exit.
    pub fn terminate(&mut self) -> ExitStatus {
        let mut child = self.child.take().expect("the process runs");
        // SAFETY: the child pid is live, and SIGTERM is the stop signal of spec 10.3.
        assert_eq!(unsafe { kill(child.id() as i32, SIGTERM) }, 0);
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "the process did not exit after SIGTERM");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Kill the process at once, with no stop.
    pub fn kill(&mut self) {
        let mut child = self.child.take().expect("the process runs");
        child.kill().unwrap();
        child.wait().unwrap();
    }

    /// Wait for the process to end on its own. Panic with the stderr lines when the wait ends.
    pub fn wait_exit(&mut self, timeout: Duration) -> ExitStatus {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.child.as_mut().and_then(|child| child.try_wait().unwrap()) {
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "the process did not exit in {timeout:?}; its stderr:\n{}",
                self.lines.drain().join("\n")
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// The identity of the process.
    pub fn identity(&self) -> Value {
        let reply = send(self.http, "GET", "/council/v2/identity", None, &[]);
        assert_eq!(reply.status, 200, "{}", reply.text());
        reply.json()
    }

    /// The `model` names of the identity, in its order.
    pub fn models(&self) -> Vec<(String, String)> {
        self.identity()["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| {
                (
                    entry["model"].as_str().unwrap().to_owned(),
                    entry["status"].as_str().unwrap().to_owned(),
                )
            })
            .collect()
    }

    /// Wait until the identity satisfies `ready`. Panic with the last identity when the wait ends.
    pub fn wait_identity(&self, what: &str, ready: impl Fn(&Value) -> bool) -> Value {
        let deadline = Instant::now() + WAIT;
        loop {
            let identity = self.identity();
            if ready(&identity) {
                return identity;
            }
            assert!(
                Instant::now() < deadline,
                "no identity of {} with {what} in {WAIT:?}; the last one:\n{identity:#}",
                self.name
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Wait until the identity lists exactly these `(model, status)` pairs.
    pub fn wait_models(&self, want: &[(&str, &str)]) -> Value {
        let want: Vec<(String, String)> = want
            .iter()
            .map(|(model, status)| ((*model).to_owned(), (*status).to_owned()))
            .collect();
        let deadline = Instant::now() + WAIT;
        loop {
            let models = self.models();
            if models == want {
                return self.identity();
            }
            assert!(
                Instant::now() < deadline,
                "the identity of {} lists {models:?}, not {want:?}, after {WAIT:?}",
                self.name
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for Member {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// One HTTP response, with its body decoded from chunks.
#[derive(Clone, Debug)]
pub struct Reply {
    /// The status code.
    pub status: u16,
    /// The headers, with lowercase names, in their order.
    pub headers: Vec<(String, String)>,
    /// The body bytes.
    pub body: Vec<u8>,
}

impl Reply {
    /// The value of a header.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    /// The body as text.
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// The body as JSON.
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|error| panic!("the body is not JSON ({error}): {}", self.text()))
    }
}

/// Send one request with `connection: close`, and read the whole response.
pub fn send(addr: SocketAddr, method: &str, path: &str, body: Option<&Value>, headers: &[(&str, &str)]) -> Reply {
    let body = body.map(Value::to_string).unwrap_or_default();
    let mut all = vec![("content-type", "application/json")];
    all.extend_from_slice(headers);
    send_bytes(addr, method, path, body.as_bytes(), &all)
}

/// Send one request of raw body bytes with `connection: close`, and read the whole response. A `HEAD` reads no body.
pub fn send_bytes(addr: SocketAddr, method: &str, path: &str, body: &[u8], headers: &[(&str, &str)]) -> Reply {
    let mut stream = TcpStream::connect(addr).unwrap_or_else(|error| panic!("cannot connect to {addr}: {error}"));
    stream.set_read_timeout(Some(WAIT)).unwrap();
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nhost: {addr}\r\nconnection: close\r\ncontent-length: {}\r\n",
        body.len()
    );
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).unwrap();
    stream.write_all(body).unwrap();
    read_reply(stream, method == "HEAD")
}

/// Read one response from a stream that the server closes after it.
fn read_reply(stream: TcpStream, head: bool) -> Reply {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let status: u16 = line
        .split(' ')
        .nth(1)
        .unwrap_or_else(|| panic!("no status line: {line:?}"))
        .parse()
        .unwrap();
    let mut headers = Vec::new();
    loop {
        line.clear();
        reader.read_line(&mut line).unwrap();
        let text = line.trim_end();
        if text.is_empty() {
            break;
        }
        let (name, value) = text.split_once(':').unwrap();
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
    }
    let chunked = headers
        .iter()
        .any(|(name, value)| name == "transfer-encoding" && value.contains("chunked"));
    let mut body = Vec::new();
    if head {
        return Reply { status, headers, body };
    }
    if chunked {
        loop {
            line.clear();
            reader.read_line(&mut line).unwrap();
            let size = usize::from_str_radix(line.trim(), 16).unwrap_or_else(|_| panic!("bad chunk size {line:?}"));
            if size == 0 {
                break;
            }
            let mut chunk = vec![0u8; size];
            reader.read_exact(&mut chunk).unwrap();
            body.extend_from_slice(&chunk);
            line.clear();
            reader.read_line(&mut line).unwrap();
        }
    } else {
        reader.read_to_end(&mut body).unwrap();
    }
    Reply { status, headers, body }
}

/// The gogi toy prompt of a generation of `model`.
pub fn generation(model: &str, stream: bool) -> Value {
    serde_json::json!({
        "model": model,
        "context": {"thinking": true, "messages": [{"role": "user", "content": "hello there"}]},
        "max_tokens": 8,
        "sample": {"seed": 7},
        "stream": stream,
    })
}
