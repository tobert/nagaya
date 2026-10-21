//! The names of stored fds, the fds that systemd passes back, and the messages to its fd store (spec 10.2).

mod fd_support;

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::net::UnixDatagram;
use std::time::Duration;

use fd_support::{bytes_of, dup_at, inode, memfd, recv_with_fds};
use nagaya::arena::{ArenaSet, FdName, FdStore, Listen};

fn listen(pid: &str, fds: &str, names: &str) -> Listen {
    Listen {
        pid: Some(pid.to_owned()),
        fds: Some(fds.to_owned()),
        names: Some(names.to_owned()),
        pidfd_id: None,
    }
}

/// Two memfds at two numbers in a row, at or above 600, as systemd puts them at 3 and 4.
fn two_in_a_row(first: &[u8], second: &[u8]) -> (i32, u64, u64) {
    let one = memfd(first);
    let two = memfd(second);
    let inodes = (inode(&one), inode(&two));
    let mut at = 600;
    loop {
        let a = dup_at(&one, at);
        let b = dup_at(&two, a + 1);
        if b == a + 1 {
            return (a, inodes.0, inodes.1);
        }
        // SAFETY: the test made both fds and closes them here.
        unsafe {
            libc::close(a);
            libc::close(b);
        }
        at = b + 1;
    }
}

#[test]
fn a_name_has_the_engine_the_device_and_the_part() {
    let name = FdName::parse("gemma/device1/w15").unwrap();
    assert_eq!(name.engine, "gemma");
    assert_eq!(name.device, 1);
    assert_eq!(name.part, "w15");
    assert_eq!(name.to_string(), "gemma/device1/w15");
    assert_eq!(
        FdName::new("gemma", 0, "head").unwrap().to_string(),
        "gemma/device0/head"
    );
}

#[test]
fn a_name_that_breaks_the_rule_is_refused() {
    for bad in [
        "gemma/w15",
        "gemma/card1/w15",
        "gemma/device/w15",
        "gemma/device1/",
        "/device1/w15",
        "gemma/device1/w:15",
        "gemma/device1/w\n15",
        "gemma/device1/a/b",
    ] {
        assert!(FdName::parse(bad).is_err(), "`{bad}` is accepted");
    }
    let long = "w".repeat(250);
    assert!(
        FdName::new("gemma", 0, &long).is_err(),
        "a name of more than 255 characters is accepted"
    );
    assert!(
        FdName::new("gemma", 0, "wé").is_err(),
        "a name that is not ASCII is accepted"
    );
}

#[test]
fn the_fds_of_our_pid_come_back_by_name_and_close_on_exec() {
    let (first, one, two) = two_in_a_row(b"one", b"two");
    // SAFETY: the test owns the two fds at `first` and `first + 1`.
    let set = unsafe { ArenaSet::from_listen_at(&listen("77", "2", "e/device0/a:e/device0/b"), 77, first, || Ok(0)) }
        .unwrap();
    assert_eq!(set.names(), vec!["e/device0/a", "e/device0/b"]);
    let fds = set.into_fds();
    assert_eq!(inode(&fds[0].1), one);
    assert_eq!(inode(&fds[1].1), two);
    for (_, fd) in &fds {
        // SAFETY: the fd is open.
        let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFD) };
        assert_ne!(flags & libc::FD_CLOEXEC, 0, "a passed fd stays open across exec");
    }
}

#[test]
fn the_fds_of_another_pid_are_not_ours() {
    let fd = memfd(b"x");
    let raw = dup_at(&fd, 700);
    // SAFETY: the set takes no fd when the pid is not ours.
    let set = unsafe { ArenaSet::from_listen_at(&listen("78", "1", "e/device0/a"), 77, raw, || Ok(0)) }.unwrap();
    assert!(set.is_empty());
    // The fd is still open: the call did not close it.
    // SAFETY: the fd is open.
    assert!(unsafe { libc::fcntl(raw, libc::F_GETFD) } >= 0);
    // SAFETY: the test owns it.
    drop(unsafe { OwnedFd::from_raw_fd(raw) });
}

#[test]
fn no_listen_pid_gives_an_empty_set() {
    let none = Listen {
        pid: None,
        fds: Some("3".to_owned()),
        names: None,
        pidfd_id: None,
    };
    // SAFETY: with no pid the set takes no fd.
    let set = unsafe { ArenaSet::from_listen_at(&none, 77, 3, || Ok(0)) }.unwrap();
    assert!(set.is_empty());
}

#[test]
fn another_pidfd_gives_an_empty_set() {
    let mut other = listen("77", "1", "e/device0/a");
    other.pidfd_id = Some("12".to_owned());
    // SAFETY: with a pidfd of another process the set takes no fd.
    let set = unsafe { ArenaSet::from_listen_at(&other, 77, 3, || Ok(13)) }.unwrap();
    assert!(set.is_empty());
}

#[test]
fn a_count_that_differs_from_the_names_or_does_not_parse_is_an_error() {
    let (first, _, _) = two_in_a_row(b"one", b"two");
    // SAFETY: the test owns the two fds; an error takes them and closes them.
    let error = unsafe { ArenaSet::from_listen_at(&listen("77", "2", "e/device0/a"), 77, first, || Ok(0)) }
        .err()
        .unwrap();
    assert!(error.to_string().contains("LISTEN_FDNAMES"), "{error}");
    // SAFETY: as above, with no fd taken.
    let error = unsafe { ArenaSet::from_listen_at(&listen("77", "two", "a:b"), 77, first, || Ok(0)) }
        .err()
        .unwrap();
    assert!(error.to_string().contains("LISTEN_FDS"), "{error}");
}

#[test]
fn the_set_splits_by_engine_and_keeps_the_names_of_no_engine_apart() {
    let set = ArenaSet::from_fds(vec![
        ("gemma/device0/w00".to_owned(), memfd(b"0")),
        ("gemma/device1/head".to_owned(), memfd(b"h")),
        ("old/device0/w".to_owned(), memfd(b"o")),
        ("stored".to_owned(), memfd(b"s")),
        ("toy/device0/w".to_owned(), memfd(b"t")),
    ]);
    let split = set.split(&["gemma".to_owned(), "toy".to_owned()]);
    let gemma: Vec<String> = split.engines["gemma"]
        .iter()
        .map(|(name, _)| name.to_string())
        .collect();
    assert_eq!(gemma, ["gemma/device0/w00", "gemma/device1/head"]);
    assert_eq!(bytes_of(&split.engines["toy"][0].1), b"t");
    let orphans: Vec<&str> = split.orphans.iter().map(|orphan| orphan.name.as_str()).collect();
    assert_eq!(orphans, ["old/device0/w", "stored"]);
}

#[test]
fn the_ceiling_comes_from_fdstore_and_zero_or_unset_keeps_no_arena() {
    let socket = Some("/run/notify".into());
    assert_eq!(
        FdStore::from_values(socket.clone(), Some("512".to_owned()))
            .unwrap()
            .unwrap()
            .ceiling(),
        512
    );
    assert!(
        FdStore::from_values(socket.clone(), Some("0".to_owned()))
            .unwrap()
            .is_none()
    );
    assert!(FdStore::from_values(socket.clone(), None).unwrap().is_none());
    assert!(FdStore::from_values(None, Some("512".to_owned())).unwrap().is_none());
    let error = FdStore::from_values(socket, Some("many".to_owned())).err().unwrap();
    assert!(error.contains("FDSTORE"), "{error}");
}

#[test]
fn a_store_sends_the_name_and_the_fd_and_a_remove_sends_the_name() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("notify");
    let socket = UnixDatagram::bind(&path).unwrap();
    socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let store = FdStore::at(&path, 8);
    let fd = memfd(b"slab");
    let name = FdName::new("gemma", 1, "w15").unwrap();
    store.store(&name, &fd).unwrap();
    let (text, fds) = recv_with_fds(&socket).unwrap();
    assert_eq!(text, "FDSTORE=1\nFDNAME=gemma/device1/w15");
    assert_eq!(fds.len(), 1);
    assert_eq!(inode(&fds[0]), inode(&fd));
    store.remove("gemma/device1/w15").unwrap();
    let (text, fds) = recv_with_fds(&socket).unwrap();
    assert_eq!(text, "FDSTOREREMOVE=1\nFDNAME=gemma/device1/w15");
    assert!(fds.is_empty());
}

#[test]
fn a_socket_path_that_starts_with_at_is_an_abstract_socket() {
    use std::os::linux::net::SocketAddrExt;
    let name = format!("nagaya-test-{}", std::process::id());
    let address = std::os::unix::net::SocketAddr::from_abstract_name(name.as_bytes()).unwrap();
    let socket = UnixDatagram::bind_addr(&address).unwrap();
    socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let store = FdStore::at(std::path::Path::new(&format!("@{name}")), 8);
    store.remove("e/device0/w").unwrap();
    let (text, _) = recv_with_fds(&socket).unwrap();
    assert_eq!(text, "FDSTOREREMOVE=1\nFDNAME=e/device0/w");
}
