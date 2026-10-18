//! The systemd readiness and status messages (spec 10.1).

use std::ffi::OsString;
use std::io;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};

/// The environment variable that names the systemd notify socket.
pub const NOTIFY_SOCKET: &str = "NOTIFY_SOCKET";

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

    /// Report that every engine opened (spec 10.1). One datagram holds `READY=1` and `STATUS=`.
    pub fn ready(&self, engines: &[String]) -> io::Result<()> {
        let status = if engines.is_empty() {
            "no engines in the profile".to_owned()
        } else {
            format!("engines opened: {}", engines.join(", "))
        };
        self.notify(&format!("READY=1\nSTATUS={status}"))
    }

    /// Report an engine that did not open. The status names the engine and the reason.
    pub fn failed(&self, engine: &str, message: &str) -> io::Result<()> {
        self.notify(&format!("STATUS=engine {engine} failed to open: {message}"))
    }
}
