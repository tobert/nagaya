use std::ffi::OsString;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};
use std::time::Duration;

use nagaya::notify::{EngineState, Notifier};

fn bound(directory: &Path) -> (UnixDatagram, PathBuf) {
    let path = directory.join("notify");
    let socket = UnixDatagram::bind(&path).unwrap();
    socket.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    (socket, path)
}

fn receive(socket: &UnixDatagram) -> String {
    let mut buffer = [0u8; 512];
    let count = socket.recv(&mut buffer).unwrap();
    std::str::from_utf8(&buffer[..count]).unwrap().to_owned()
}

#[test]
fn the_ready_message_reaches_the_socket() {
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    let notifier = Notifier::at(&path).unwrap();
    notifier.ready(&["a".to_owned(), "b".to_owned()]).unwrap();
    let text = receive(&socket);
    assert_eq!(
        text, "READY=1\nSTATUS=engine a ready; engine b ready",
        "systemd reads READY and STATUS in one datagram"
    );
}

#[test]
fn an_empty_profile_reports_ready() {
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    Notifier::at(&path).unwrap().ready(&[]).unwrap();
    let text = receive(&socket);
    assert_eq!(text, "READY=1\nSTATUS=no engines in the profile");
}

#[test]
fn a_failed_engine_is_named_in_the_status() {
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    Notifier::at(&path)
        .unwrap()
        .failed("a", "the weights are absent")
        .unwrap();
    let text = receive(&socket);
    assert_eq!(text, "STATUS=engine a fatal: the weights are absent");
    assert!(!text.contains("READY=1"), "a failed start is not ready: {text}");
}

#[test]
fn an_unset_socket_gives_no_notifier() {
    assert!(Notifier::from_value(None).unwrap().is_none());
    assert!(Notifier::from_value(Some(OsString::new())).unwrap().is_none());
    assert!(
        Notifier::from_value(Some(OsString::from("/tmp/nagaya-notify")))
            .unwrap()
            .is_some()
    );
}

#[test]
fn a_state_message_names_every_engine() {
    let directory = tempfile::tempdir().unwrap();
    let (socket, path) = bound(directory.path());
    let notifier = Notifier::at(&path).unwrap();
    notifier
        .states(&["a".to_owned(), "b".to_owned()], EngineState::Draining)
        .unwrap();
    assert_eq!(receive(&socket), "STATUS=engine a draining; engine b draining");
}

#[test]
fn each_engine_state_has_its_word() {
    let states = [
        (EngineState::Loading, "loading"),
        (EngineState::Ready, "ready"),
        (EngineState::Draining, "draining"),
        (EngineState::Unloaded, "unloaded"),
        (EngineState::Fatal, "fatal"),
    ];
    for (state, word) in states {
        assert_eq!(state.as_str(), word);
    }
}
