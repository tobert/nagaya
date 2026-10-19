mod support;

use std::io;
use std::net::UdpSocket;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const SIGTERM: i32 = 15;

unsafe extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

/// A child process that is killed when the test ends, unless it exited first.
struct Guard(Option<Child>);

impl Guard {
    fn id(&self) -> u32 {
        self.0.as_ref().unwrap().id()
    }

    fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.0.as_mut().unwrap().try_wait()
    }

    fn take(&mut self) -> Child {
        self.0.take().unwrap()
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn wait_timeout(guard: &mut Guard, timeout: Duration) -> ExitStatus {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = guard.try_wait().unwrap() {
            return status;
        }
        if Instant::now() > deadline {
            panic!("the process did not exit in {timeout:?}");
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn output(mut guard: Guard) -> Output {
    guard.take().wait_with_output().unwrap()
}

struct Fixture {
    _directory: tempfile::TempDir,
    config: PathBuf,
    store: PathBuf,
    notify: PathBuf,
}

impl Fixture {
    fn new(kind: &str) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let store = directory.path().join("store");
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
            format!(
                "[[engines]]\n\
                 name = \"fake-one\"\n\
                 kind = \"{kind}\"\n\
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
                  nvme = 65536\n"
            ),
        )
        .unwrap();
        let notify = directory.path().join("notify");
        Self {
            _directory: directory,
            config,
            store,
            notify,
        }
    }

    fn spawn(&self, notify: Option<&Path>) -> Guard {
        let mut command = Command::new(env!("CARGO_BIN_EXE_nagaya"));
        command
            .env("NAGAYA_CONFIG", &self.config)
            .env_remove("NOTIFY_SOCKET")
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        if let Some(notify) = notify {
            command.env("NOTIFY_SOCKET", notify);
        }
        Guard(Some(command.spawn().unwrap()))
    }
}

#[test]
fn ready_and_a_sigterm_exit_zero() {
    let fixture = Fixture::new("fake");
    let socket = UnixDatagram::bind(&fixture.notify).unwrap();
    socket.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut child = fixture.spawn(Some(&fixture.notify));
    let mut buffer = [0u8; 512];
    let count = socket.recv(&mut buffer).expect("the process sends its loading status");
    let text = std::str::from_utf8(&buffer[..count]).unwrap();
    assert_eq!(text, "STATUS=engine fake-one loading");
    let count = socket.recv(&mut buffer).expect("the process sends READY");
    let text = std::str::from_utf8(&buffer[..count]).unwrap();
    assert_eq!(text, "READY=1\nSTATUS=engine fake-one ready");
    // SAFETY: the child pid is live, and SIGTERM is the stop signal of spec 10.3.
    assert_eq!(unsafe { kill(child.id() as i32, SIGTERM) }, 0);
    let status = wait_timeout(&mut child, Duration::from_secs(10));
    assert_eq!(status.code(), Some(0));
}

#[test]
fn ready_comes_with_gossip_up() {
    let fixture = Fixture::new("fake");
    let socket = UnixDatagram::bind(&fixture.notify).unwrap();
    socket.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut child = fixture.spawn(Some(&fixture.notify));
    let mut lines = support::Lines::new(child.0.as_mut().unwrap().stderr.take().unwrap());
    let mut buffer = [0u8; 512];
    socket.recv(&mut buffer).expect("the process sends its loading status");
    let count = socket.recv(&mut buffer).expect("the process sends READY");
    assert!(std::str::from_utf8(&buffer[..count]).unwrap().starts_with("READY=1"));
    let addr = lines.wait_for("gossip listening on ", Duration::from_secs(10));
    let addr: std::net::SocketAddr = addr.parse().unwrap();
    assert_ne!(addr.port(), 0);
    let error = UdpSocket::bind(addr).expect_err("the gossip socket holds the port while the process is ready");
    assert_eq!(error.kind(), io::ErrorKind::AddrInUse);
    // SAFETY: the child pid is live, and SIGTERM is the stop signal of spec 10.3.
    assert_eq!(unsafe { kill(child.id() as i32, SIGTERM) }, 0);
    let status = wait_timeout(&mut child, Duration::from_secs(10));
    assert_eq!(status.code(), Some(0));
    UdpSocket::bind(addr).expect("the port is free after the process exits");
}

#[test]
fn a_panic_in_a_group_thread_exits_one() {
    let fixture = Fixture::new("fake-panic");
    let output = output(fixture.spawn(None));
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("the fake engine cannot open"),
        "the panic must reach the report: {stderr}"
    );
}

#[test]
fn a_config_that_does_not_parse_exits_three() {
    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("nagaya.toml");
    std::fs::write(&config, "this is not TOML =\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_nagaya"))
        .env("NAGAYA_CONFIG", &config)
        .env_remove("NOTIFY_SOCKET")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(config.to_str().unwrap()),
        "the message must name the path: {stderr}"
    );
}

#[test]
fn a_missing_config_exits_three() {
    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("missing.toml");
    let output = Command::new(env!("CARGO_BIN_EXE_nagaya"))
        .env("NAGAYA_CONFIG", &config)
        .env_remove("NOTIFY_SOCKET")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(config.to_str().unwrap()),
        "the message must name the path: {stderr}"
    );
}

#[test]
fn a_profile_that_does_not_admit_exits_three() {
    let fixture = Fixture::new("ghost");
    let output = output(fixture.spawn(None));
    assert_eq!(output.status.code(), Some(3));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("ghost"), "the message names the kind: {stderr}");
}

#[test]
fn a_profile_without_a_store_budget_exits_three() {
    let fixture = Fixture::new("fake");
    std::fs::write(
        fixture.store.join("profile.toml"),
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
         bytes = 256\n",
    )
    .unwrap();
    let output = output(fixture.spawn(None));
    assert_eq!(output.status.code(), Some(3));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("store.nvme"), "the message must name the key: {stderr}");
}

#[test]
fn a_gossip_address_that_is_in_use_exits_three() {
    let fixture = Fixture::new("fake");
    let held = UdpSocket::bind("127.0.0.1:0").unwrap();
    let text = std::fs::read_to_string(&fixture.config).unwrap().replace(
        "gossip_bind = \"127.0.0.1:0\"",
        &format!("gossip_bind = \"{}\"", held.local_addr().unwrap()),
    );
    std::fs::write(&fixture.config, text).unwrap();
    let output = output(fixture.spawn(None));
    assert_eq!(output.status.code(), Some(3));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("gossip"), "the message must name gossip: {stderr}");
}

#[test]
fn an_unspecified_gossip_address_exits_three() {
    for address in ["0.0.0.0:0", "[::]:0"] {
        let fixture = Fixture::new("fake");
        let text = std::fs::read_to_string(&fixture.config)
            .unwrap()
            .replace("gossip_bind = \"127.0.0.1:0\"", &format!("gossip_bind = \"{address}\""));
        std::fs::write(&fixture.config, text).unwrap();
        let output = output(fixture.spawn(None));
        assert_eq!(output.status.code(), Some(3), "{address}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("gossip_bind"),
            "the message must name the key for {address}: {stderr}"
        );
    }
}
