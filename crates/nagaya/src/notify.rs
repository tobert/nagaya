//! The systemd readiness and status messages (spec 10.1).

use std::ffi::OsString;
use std::io;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};

/// The environment variable that names the systemd notify socket.
pub const NOTIFY_SOCKET: &str = "NOTIFY_SOCKET";

/// The state of one engine, for systemd's `STATUS=` (spec 10.1, 2.3). The telemetry crate owns the type, so that
/// `nagaya.engine.state` and `STATUS=` use one word for one state.
pub use nagaya_telemetry::EngineState;

/// The `STATUS=` text for the engines and their states. An empty list gives the empty-profile text.
fn status_text(engines: &[String], state: EngineState) -> String {
    if engines.is_empty() {
        return "STATUS=no engines in the profile".to_owned();
    }
    let states: Vec<String> = engines
        .iter()
        .map(|engine| format!("engine {engine} {}", state.as_str()))
        .collect();
    format!("STATUS={}", states.join("; "))
}

/// The systemd notify socket. A process without the variable has no notifier.
pub struct Notifier {
    socket: UnixDatagram,
    path: PathBuf,
}

impl Notifier {
    /// Read the socket path from `NOTIFY_SOCKET`. An unset or empty variable gives `None`.
    pub fn from_env() -> io::Result<Option<Self>> {
        Self::from_value(std::env::var_os(NOTIFY_SOCKET))
    }

    /// Make a notifier from a socket path value. An unset or empty value gives `None`.
    pub fn from_value(value: Option<OsString>) -> io::Result<Option<Self>> {
        let path = match value {
            Some(path) if !path.is_empty() => PathBuf::from(path),
            _ => return Ok(None),
        };
        Ok(Some(Self {
            socket: UnixDatagram::unbound()?,
            path,
        }))
    }

    /// Make a notifier for a socket at a fixed path. The tests use it.
    pub fn at(path: &Path) -> io::Result<Self> {
        Ok(Self {
            socket: UnixDatagram::unbound()?,
            path: path.to_owned(),
        })
    }

    /// Write one datagram. systemd reads one message with an optional newline.
    pub fn notify(&self, text: &str) -> io::Result<()> {
        self.socket.send_to(text.as_bytes(), &self.path).map(|_| ())
    }

    /// Report one state for every engine (spec 10.1). An empty list sends nothing.
    pub fn states(&self, engines: &[String], state: EngineState) -> io::Result<()> {
        if engines.is_empty() {
            return Ok(());
        }
        self.notify(&status_text(engines, state))
    }

    /// Report that every engine opened (spec 10.1). One datagram holds `READY=1` and `STATUS=`.
    pub fn ready(&self, engines: &[String]) -> io::Result<()> {
        let status = if engines.is_empty() {
            "no engines in the profile".to_owned()
        } else {
            status_text(engines, EngineState::Ready)
                .trim_start_matches("STATUS=")
                .to_owned()
        };
        self.notify(&format!("READY=1\nSTATUS={status}"))
    }

    /// Report an engine that did not open.
    pub fn failed(&self, engine: &str, message: &str) -> io::Result<()> {
        self.notify(&format!("STATUS=engine {engine} fatal: {message}"))
    }
}
