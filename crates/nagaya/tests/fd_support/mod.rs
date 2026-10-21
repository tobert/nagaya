//! Helpers for the tests of the fd store (spec 10.2): memfds, inodes, and a receiver of `SCM_RIGHTS` datagrams.

#![allow(dead_code)]

use std::ffi::CString;
use std::fs::File;
use std::io::Write;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::os::unix::net::UnixDatagram;

/// A memfd that holds `bytes`.
pub fn memfd(bytes: &[u8]) -> OwnedFd {
    let name = CString::new("nagaya-test").unwrap();
    // SAFETY: `name` is a valid C string.
    let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0, "memfd_create failed");
    // SAFETY: the fd is new, and nothing else owns it.
    let mut file = File::from(unsafe { OwnedFd::from_raw_fd(fd) });
    file.write_all(bytes).unwrap();
    file.into()
}

/// The inode of the object of an fd.
pub fn inode(fd: &OwnedFd) -> u64 {
    File::from(fd.try_clone().unwrap()).metadata().unwrap().ino()
}

/// The bytes of a memfd, from offset 0.
pub fn bytes_of(fd: &OwnedFd) -> Vec<u8> {
    let file = File::from(fd.try_clone().unwrap());
    let mut out = vec![0u8; file.metadata().unwrap().len() as usize];
    file.read_exact_at(&mut out, 0).unwrap();
    out
}

/// Duplicate `fd` to the lowest free number at or above `at`. The new fd has no `FD_CLOEXEC`, as an fd that systemd
/// passes.
pub fn dup_at(fd: &OwnedFd, at: RawFd) -> RawFd {
    // SAFETY: `fd` is open; the call gives a new fd or -1.
    let new = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_DUPFD, at) };
    assert!(new >= at, "F_DUPFD failed");
    new
}

/// One datagram and the fds that came with it.
pub fn recv_with_fds(socket: &UnixDatagram) -> std::io::Result<(String, Vec<OwnedFd>)> {
    let mut data = vec![0u8; 4096];
    let mut control = vec![0u8; 1024];
    let mut iov = libc::iovec {
        iov_base: data.as_mut_ptr().cast(),
        iov_len: data.len(),
    };
    // SAFETY: an all-zero msghdr is valid; the fields are set below.
    let mut header: libc::msghdr = unsafe { std::mem::zeroed() };
    header.msg_iov = &mut iov;
    header.msg_iovlen = 1;
    header.msg_control = control.as_mut_ptr().cast();
    header.msg_controllen = control.len();
    // SAFETY: the header points to live buffers.
    let count = unsafe { libc::recvmsg(socket.as_raw_fd(), &mut header, libc::MSG_CMSG_CLOEXEC) };
    if count < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut fds = Vec::new();
    // SAFETY: the kernel filled the control buffer, and the CMSG macros walk it.
    unsafe {
        let mut cmsg = libc::CMSG_FIRSTHDR(&header);
        while !cmsg.is_null() {
            if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                let payload = (*cmsg).cmsg_len as usize - libc::CMSG_LEN(0) as usize;
                let first = libc::CMSG_DATA(cmsg) as *const RawFd;
                for index in 0..payload / std::mem::size_of::<RawFd>() {
                    fds.push(OwnedFd::from_raw_fd(first.add(index).read_unaligned()));
                }
            }
            cmsg = libc::CMSG_NXTHDR(&header, cmsg);
        }
    }
    Ok((String::from_utf8(data[..count as usize].to_vec()).unwrap(), fds))
}

/// Make `command` start `program` as systemd starts a unit with stored fds: the fds at 3 and up, and `$LISTEN_PID`,
/// `$LISTEN_FDS` and `$LISTEN_FDNAMES` set. A shell sets `$LISTEN_PID` to its own pid, which its `exec` keeps.
///
/// The fds wait at 200 and up with `FD_CLOEXEC`, so no other child of the test gets them. After the fork, the child
/// moves them to 3 and up with dup2, which is safe after fork. The caller drops the result after the spawn.
pub fn systemd_command(program: &std::path::Path, fds: &[(&str, &OwnedFd)]) -> (std::process::Command, Vec<OwnedFd>) {
    use std::os::unix::process::CommandExt;
    let high: Vec<OwnedFd> = fds
        .iter()
        .map(|(_, fd)| {
            // SAFETY: `fd` is open; the call gives a new fd or -1.
            let new = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 200) };
            assert!(new >= 200, "F_DUPFD_CLOEXEC failed");
            // SAFETY: the fd is new, and nothing else owns it.
            unsafe { OwnedFd::from_raw_fd(new) }
        })
        .collect();
    let names: Vec<&str> = fds.iter().map(|(name, _)| *name).collect();
    let script = if fds.is_empty() {
        "exec \"$0\" \"$@\"".to_owned()
    } else {
        format!(
            "export LISTEN_PID=$$ LISTEN_FDS={} LISTEN_FDNAMES={}; exec \"$0\" \"$@\"",
            fds.len(),
            names.join(":")
        )
    };
    let mut command = std::process::Command::new("/bin/sh");
    command
        .arg("-c")
        .arg(script)
        .arg(program)
        .env_remove("LISTEN_PID")
        .env_remove("LISTEN_FDS")
        .env_remove("LISTEN_FDNAMES")
        .env_remove("LISTEN_PIDFDID");
    let raw: Vec<RawFd> = high.iter().map(AsRawFd::as_raw_fd).collect();
    // SAFETY: the closure calls only dup2, which is safe after fork.
    unsafe {
        command.pre_exec(move || {
            for (index, fd) in raw.iter().enumerate() {
                if libc::dup2(*fd, 3 + index as RawFd) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    (command, high)
}
