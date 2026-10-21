//! The process logs to stderr always, and exports over OTLP when the endpoint is set (spec 9.1).

mod support;

use std::io::Read;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

const SIGTERM: i32 = 15;

unsafe extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

/// The HTTP/2 preface that every gRPC client sends first.
const PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

struct Process {
    child: Option<Child>,
    lines: support::Lines,
    _directory: tempfile::TempDir,
}

impl Process {
    fn spawn(env: &[(&str, &str)]) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let store: PathBuf = directory.path().join("store");
        std::fs::create_dir_all(&store).unwrap();
        let config = directory.path().join("nagaya.toml");
        std::fs::write(
            &config,
            format!(
                "http_bind = \"127.0.0.1:0\"\n\
                 gossip_bind = \"127.0.0.1:0\"\n\
                 store_dir = \"{}\"\n\
                 park_budget_secs = 1\n",
                store.display()
            ),
        )
        .unwrap();
        std::fs::write(
            store.join("profile.toml"),
            "[[engines]]\n\
             name = \"fake-one\"\n\
             kind = \"fake\"\n\
             devices = [0]\n\
             share = 1\n\
             \n\
             [[budgets]]\n\
             tiers = [\"device(0)\"]\n\
             bytes = 4096\n\
             \n\
             [[budgets]]\n\
             tiers = [\"host\"]\n\
             bytes = 256\n\
             \n\
             [store]\n\
             nvme = 65536\n",
        )
        .unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_nagaya"));
        command
            .env("NAGAYA_CONFIG", &config)
            .env_remove("NOTIFY_SOCKET")
            .env_remove("RUST_LOG")
            .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
            .env_remove("OTEL_SDK_DISABLED")
            .env_remove("NAGAYA_OTEL_FILE_DIR")
            .env_remove("KAIJUTSU_OTEL_FILE_DIR")
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        for (key, value) in env {
            command.env(key, value);
        }
        let mut child = command.spawn().unwrap();
        let lines = support::Lines::new(child.stderr.take().unwrap());
        Self {
            child: Some(child),
            lines,
            _directory: directory,
        }
    }

    /// Send SIGTERM and wait for the exit.
    fn terminate(&mut self) -> ExitStatus {
        let child = self.child.as_mut().unwrap();
        // SAFETY: the child pid is live, and SIGTERM is the stop signal of spec 10.3.
        assert_eq!(unsafe { kill(child.id() as i32, SIGTERM) }, 0);
        self.wait(Duration::from_secs(20))
    }

    fn wait(&mut self, timeout: Duration) -> ExitStatus {
        let child = self.child.as_mut().unwrap();
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "the process did not exit in {timeout:?}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Accept connections on `listener` on a new thread. Send the first bytes of each connection to the channel.
fn accept_all(listener: TcpListener) -> Receiver<Vec<u8>> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let sender = sender.clone();
            std::thread::spawn(move || {
                let mut buffer = vec![0u8; PREFACE.len()];
                if stream.read_exact(&mut buffer).is_ok() {
                    let _ = sender.send(buffer);
                }
                let mut rest = Vec::new();
                let _ = stream.read_to_end(&mut rest);
            });
        }
    });
    receiver
}

fn collector() -> (String, Receiver<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    (endpoint, accept_all(listener))
}

#[test]
fn the_process_logs_to_stderr_with_no_telemetry_variable() {
    let mut process = Process::spawn(&[]);
    process.lines.wait_for("ready with 1 engines", Duration::from_secs(10));
    assert_eq!(process.terminate().code(), Some(0));
    let line = process.lines.wait_for("the process exits ", Duration::from_secs(5));
    assert_eq!(line, "exit_code=0");
}

#[test]
fn an_endpoint_makes_the_process_export_spans_and_metrics_to_it() {
    let (endpoint, connections) = collector();
    let mut process = Process::spawn(&[("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint.as_str())]);
    process.lines.wait_for("ready with 1 engines", Duration::from_secs(10));
    assert_eq!(process.terminate().code(), Some(0));
    // The spans and the metrics leave at the stop. Each signal has its own connection.
    for signal in ["spans", "metrics"] {
        let bytes = connections
            .recv_timeout(Duration::from_secs(10))
            .unwrap_or_else(|_| panic!("the {signal} exporter did not connect"));
        assert_eq!(bytes, PREFACE);
    }
}

#[test]
fn sdk_disabled_stops_the_export_but_not_the_log() {
    let (endpoint, connections) = collector();
    let mut process = Process::spawn(&[
        ("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint.as_str()),
        ("OTEL_SDK_DISABLED", "true"),
    ]);
    process.lines.wait_for("ready with 1 engines", Duration::from_secs(10));
    assert_eq!(process.terminate().code(), Some(0));
    assert!(
        connections.recv_timeout(Duration::from_millis(500)).is_err(),
        "OTEL_SDK_DISABLED=true must stop the export"
    );
}

#[test]
fn with_no_endpoint_the_process_connects_nowhere() {
    let (_endpoint, connections) = collector();
    let mut process = Process::spawn(&[]);
    process.lines.wait_for("ready with 1 engines", Duration::from_secs(10));
    assert_eq!(process.terminate().code(), Some(0));
    assert!(connections.recv_timeout(Duration::from_millis(500)).is_err());
}

#[test]
fn a_collector_that_is_gone_does_not_stop_the_process() {
    // Bind a port, then close it, so that nothing listens.
    let endpoint = {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", listener.local_addr().unwrap())
    };
    let mut process = Process::spawn(&[("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint.as_str())]);
    process.lines.wait_for("ready with 1 engines", Duration::from_secs(10));
    assert_eq!(process.terminate().code(), Some(0));
}

#[test]
fn a_file_export_directory_gets_a_warning_because_the_export_is_not_built() {
    let mut process = Process::spawn(&[("NAGAYA_OTEL_FILE_DIR", "/nonexistent/otel")]);
    let line = process
        .lines
        .wait_for("NAGAYA_OTEL_FILE_DIR is set", Duration::from_secs(10));
    assert!(line.ends_with("level=warn"), "{line}");
    process.lines.wait_for("ready with 1 engines", Duration::from_secs(10));
    assert_eq!(process.terminate().code(), Some(0));
}

#[test]
fn a_log_filter_that_does_not_parse_refuses_to_start() {
    let mut process = Process::spawn(&[("RUST_LOG", "nagaya=[bad")]);
    let status = process.wait(Duration::from_secs(10));
    assert_eq!(status.code(), Some(3));
    let seen = process.lines.drain().join("\n");
    assert!(seen.contains("RUST_LOG"), "the message must name the variable: {seen}");
}

#[test]
fn a_log_filter_sets_the_level_of_the_stderr_log() {
    let mut process = Process::spawn(&[("RUST_LOG", "error")]);
    // An info line is gone. The exit line is info, so it is gone too.
    std::thread::sleep(Duration::from_millis(1500));
    assert_eq!(process.terminate().code(), Some(0));
    let seen = process.lines.drain().join("\n");
    assert!(!seen.contains("ready with"), "{seen}");
}
