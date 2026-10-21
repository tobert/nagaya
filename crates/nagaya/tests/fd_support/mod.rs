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
