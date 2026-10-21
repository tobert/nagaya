//! Arenas through a restart: the fds that systemd passes back, their names, and the messages to the fd store
//! (spec 10.2).
//!
//! systemd keeps the stored fds while the unit restarts. At start it passes them at fd 3 and up, with
//! `$LISTEN_FDS` and `$LISTEN_FDNAMES` (sd_listen_fds(3)). The process sends an fd to the store with `FDSTORE=1` and
//! `FDNAME=` on `$NOTIFY_SOCKET` (sd_notify(3)).

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io;
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use nagaya_abi::DeviceId;

use crate::notify::NOTIFY_SOCKET;

/// The first fd that systemd passes (`SD_LISTEN_FDS_START`).
pub const LISTEN_FDS_START: RawFd = 3;

/// The longest name that `FDNAME=` takes (sd_notify(3)).
pub const MAX_FD_NAME: usize = 255;

/// The variable that gives the ceiling of the fd store (systemd.exec(5)).
pub const FDSTORE: &str = "FDSTORE";

/// The name of a stored fd: `<engine>/device<id>/<part>` (spec 10.2).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FdName {
    /// The engine name in the profile.
    pub engine: String,
    /// The machine device id (spec 2.4).
    pub device: DeviceId,
    /// The name that the engine gives the part in `arena_export`.
    pub part: String,
}

impl FdName {
    /// Make a name. The name must be legal for `FDNAME=`: ASCII, no control character, no `:`, at most 255
    /// characters. The engine and the part have no `/`.
    pub fn new(engine: &str, device: DeviceId, part: &str) -> Result<Self, String> {
        let name = Self {
            engine: engine.to_owned(),
            device,
            part: part.to_owned(),
        };
        for (what, text) in [("engine", engine), ("part", part)] {
            if text.is_empty() || text.contains('/') {
                return Err(format!(
                    "the {what} `{text}` of a stored fd name is empty or holds `/`; give a name with no `/`"
                ));
            }
        }
        let text = name.to_string();
        if text.len() > MAX_FD_NAME {
            return Err(format!(
                "the stored fd name `{text}` has {} characters; systemd takes at most {MAX_FD_NAME}",
                text.len()
            ));
        }
        if !text
            .bytes()
            .all(|byte| byte.is_ascii() && !byte.is_ascii_control() && byte != b':')
        {
            return Err(format!(
                "the stored fd name `{text}` holds a character that systemd refuses: use ASCII with no control \
                 character and no `:`"
            ));
        }
        Ok(name)
    }

    /// Read a name of the form `<engine>/device<id>/<part>`.
    pub fn parse(text: &str) -> Result<Self, String> {
        let fields: Vec<&str> = text.split('/').collect();
        let [engine, device, part] = fields[..] else {
            return Err(format!(
                "the stored fd name `{text}` does not have the form `<engine>/device<id>/<part>`"
            ));
        };
        let device = device
            .strip_prefix("device")
            .and_then(|id| id.parse::<DeviceId>().ok())
            .ok_or_else(|| format!("the stored fd name `{text}` does not name a device as `device<id>`"))?;
        Self::new(engine, device, part)
    }
}

impl std::fmt::Display for FdName {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}/device{}/{}", self.engine, self.device, self.part)
    }
}

/// Why the process cannot read the fds that systemd passed. The start is refused (spec 10.4, exit 3).
#[derive(Debug)]
pub struct ArenaError {
    /// What is wrong.
    pub message: String,
}

impl std::fmt::Display for ArenaError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for ArenaError {}

/// The values of the variables of sd_listen_fds(3).
#[derive(Clone, Debug, Default)]
pub struct Listen {
    /// `$LISTEN_PID`.
    pub pid: Option<String>,
    /// `$LISTEN_FDS`.
    pub fds: Option<String>,
    /// `$LISTEN_FDNAMES`.
    pub names: Option<String>,
    /// `$LISTEN_PIDFDID`.
    pub pidfd_id: Option<String>,
}

/// The variables of sd_listen_fds(3), in the order of [`Listen`].
const LISTEN_VARIABLES: [&str; 4] = ["LISTEN_PID", "LISTEN_FDS", "LISTEN_FDNAMES", "LISTEN_PIDFDID"];

/// The named fds that systemd passed to this process, in fd order.
#[derive(Debug, Default)]
pub struct ArenaSet {
    fds: Vec<(String, OwnedFd)>,
}

/// A stored fd that no engine can use. The process drops it after admission (spec 10.2).
#[derive(Debug)]
pub struct Orphan {
    /// The name that systemd gave.
    pub name: String,
    /// The fd.
    pub fd: OwnedFd,
    /// Why no engine can use it.
    pub why: String,
}

/// The stored fds by engine, and the fds of no engine in the profile.
#[derive(Debug, Default)]
pub struct Split {
    /// The fds of each engine of the profile, in name order.
    pub engines: BTreeMap<String, Vec<(FdName, OwnedFd)>>,
    /// The fds whose name breaks the rule or names no engine of the profile.
    pub orphans: Vec<Orphan>,
}

impl ArenaSet {
    /// A set with no fd.
    pub fn empty() -> Self {
        Self::default()
    }

    /// A set of fds that the caller gives. The tests use it.
    pub fn from_fds(fds: Vec<(String, OwnedFd)>) -> Self {
        Self { fds }
    }

    /// Read the fds that systemd passed, and remove the variables of sd_listen_fds(3).
    ///
    /// # Safety
    ///
    /// Call it once, before the process starts a thread: it changes the environment. When `$LISTEN_PID` is the pid
    /// of this process, the set takes the fds from 3 up, and nothing else may own them.
    pub unsafe fn from_env() -> Result<Self, ArenaError> {
        let value = |name: &str| std::env::var_os(name).map(|value| value.to_string_lossy().into_owned());
        let listen = Listen {
            pid: value(LISTEN_VARIABLES[0]),
            fds: value(LISTEN_VARIABLES[1]),
            names: value(LISTEN_VARIABLES[2]),
            pidfd_id: value(LISTEN_VARIABLES[3]),
        };
        for name in LISTEN_VARIABLES {
            // SAFETY: the caller starts no thread before this call.
            unsafe { std::env::remove_var(name) };
        }
        // SAFETY: the caller gives the fds from 3 up to this call.
        unsafe { Self::from_listen_at(&listen, std::process::id(), LISTEN_FDS_START, pidfd_inode) }
    }

    /// Read the fds of `listen` from `first` up, for the process `own_pid`. `pidfd` gives the inode of the pidfd of
    /// this process, for `$LISTEN_PIDFDID`.
    ///
    /// The rules are those of sd_listen_fds(3). With no `$LISTEN_PID`, or the pid of another process, or the pidfd
    /// of another process, the set is empty and takes no fd. Each fd that the set takes gets `FD_CLOEXEC`.
    ///
    /// # Safety
    ///
    /// When the pid is `own_pid`, the set takes the fds `first` to `first + $LISTEN_FDS - 1`. Nothing else may own
    /// them. After an error the call closes each of them.
    pub unsafe fn from_listen_at(
        listen: &Listen,
        own_pid: u32,
        first: RawFd,
        pidfd: impl FnOnce() -> io::Result<u64>,
    ) -> Result<Self, ArenaError> {
        let fail = |message: String| ArenaError { message };
        let Some(pid) = &listen.pid else {
            return Ok(Self::empty());
        };
        let pid: u32 = pid
            .parse()
            .map_err(|_| fail(format!("LISTEN_PID is `{pid}`, which is not a pid")))?;
        if pid != own_pid {
            return Ok(Self::empty());
        }
        if let Some(id) = &listen.pidfd_id {
            let id: u64 = id
                .parse()
                .map_err(|_| fail(format!("LISTEN_PIDFDID is `{id}`, which is not an inode number")))?;
            let own = pidfd().map_err(|error| fail(format!("cannot read the pidfd of this process: {error}")))?;
            if id != own {
                return Ok(Self::empty());
            }
        }
        let Some(count) = &listen.fds else {
            return Ok(Self::empty());
        };
        let count: RawFd = count
            .parse()
            .ok()
            .filter(|count| *count >= 0)
            .ok_or_else(|| fail(format!("LISTEN_FDS is `{count}`, which is not a count of fds")))?;
        // SAFETY: the caller gives these fds to this call.
        let fds: Vec<OwnedFd> = (first..first + count)
            .map(|fd| unsafe { OwnedFd::from_raw_fd(fd) })
            .collect();
        for fd in &fds {
            // SAFETY: the fd is open, and F_SETFD changes only its flags.
            if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
                return Err(fail(format!(
                    "cannot set FD_CLOEXEC on the passed fd {}: {}",
                    fd.as_raw_fd(),
                    io::Error::last_os_error()
                )));
            }
        }
        let names: Vec<String> = match &listen.names {
            Some(names) if count > 0 => names.split(':').map(str::to_owned).collect(),
            Some(_) => Vec::new(),
            // sd_listen_fds_with_names(3) calls an fd with no name `unknown`.
            None => vec!["unknown".to_owned(); count as usize],
        };
        if names.len() != fds.len() {
            return Err(fail(format!(
                "LISTEN_FDNAMES has {} names, and LISTEN_FDS gives {} fds; systemd gives one name for each fd",
                names.len(),
                fds.len()
            )));
        }
        Ok(Self {
            fds: names.into_iter().zip(fds).collect(),
        })
    }

    /// True when the set has no fd.
    pub fn is_empty(&self) -> bool {
        self.fds.is_empty()
    }

    /// The count of fds.
    pub fn len(&self) -> usize {
        self.fds.len()
    }

    /// The names, in fd order.
    pub fn names(&self) -> Vec<&str> {
        self.fds.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// The named fds, in fd order.
    pub fn into_fds(self) -> Vec<(String, OwnedFd)> {
        self.fds
    }

    /// Split the fds by the engine in their name. A name that breaks the rule, or names an engine that is not in
    /// `engines`, is an orphan.
    pub fn split(self, engines: &[String]) -> Split {
        let mut split = Split::default();
        for (text, fd) in self.fds {
            match FdName::parse(&text) {
                Ok(name) if engines.contains(&name.engine) => {
                    split.engines.entry(name.engine.clone()).or_default().push((name, fd));
                }
                Ok(name) => split.orphans.push(Orphan {
                    name: text,
                    fd,
                    why: format!("the profile has no engine `{}`", name.engine),
                }),
                Err(why) => split.orphans.push(Orphan { name: text, fd, why }),
            }
        }
        for fds in split.engines.values_mut() {
            fds.sort_by(|a, b| a.0.cmp(&b.0));
        }
        split
    }
}

/// The inode of a new pidfd of this process. `$LISTEN_PIDFDID` names it (sd_listen_fds(3)).
fn pidfd_inode() -> io::Result<u64> {
    // SAFETY: pidfd_open takes a pid and flags, and gives a new fd or -1.
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, libc::getpid(), 0) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the fd is new, and nothing else owns it.
    let fd = unsafe { OwnedFd::from_raw_fd(fd as RawFd) };
    let file = std::fs::File::from(fd);
    Ok(std::os::unix::fs::MetadataExt::ino(&file.metadata()?))
}

/// The fd store of systemd for this process (spec 10.2). A process with no store keeps no arena.
#[derive(Clone, Debug)]
pub struct FdStore {
    socket: PathBuf,
    ceiling: usize,
}

impl FdStore {
    /// Read `$NOTIFY_SOCKET` and `$FDSTORE`. With either one unset, or a ceiling of 0, there is no store.
    pub fn from_env() -> Result<Option<Self>, String> {
        Self::from_values(
            std::env::var_os(NOTIFY_SOCKET),
            std::env::var_os(FDSTORE).map(|value| value.to_string_lossy().into_owned()),
        )
    }

    /// Make a store from the values of `$NOTIFY_SOCKET` and `$FDSTORE`.
    pub fn from_values(socket: Option<OsString>, ceiling: Option<String>) -> Result<Option<Self>, String> {
        let Some(ceiling) = ceiling else {
            return Ok(None);
        };
        let ceiling: usize = ceiling.trim().parse().map_err(|_| {
            format!("{FDSTORE} is `{ceiling}`, which is not a count of fds; check FileDescriptorStoreMax in the unit")
        })?;
        match socket {
            Some(socket) if !socket.is_empty() && ceiling > 0 => Ok(Some(Self {
                socket: PathBuf::from(socket),
                ceiling,
            })),
            _ => Ok(None),
        }
    }

    /// Make a store on a socket at a fixed path. The tests use it.
    pub fn at(socket: &Path, ceiling: usize) -> Self {
        Self {
            socket: socket.to_owned(),
            ceiling,
        }
    }

    /// The most fds that systemd keeps for the unit (`FileDescriptorStoreMax`).
    pub fn ceiling(&self) -> usize {
        self.ceiling
    }

    /// Send `fd` to the store under `name`. systemd keeps its own copy, so the caller can close `fd` after.
    pub fn store(&self, name: &FdName, fd: &impl AsFd) -> io::Result<()> {
        self.send(&format!("FDSTORE=1\nFDNAME={name}"), Some(fd.as_fd().as_raw_fd()))
    }

    /// Remove each fd that the store holds under `name`.
    pub fn remove(&self, name: &str) -> io::Result<()> {
        self.send(&format!("FDSTOREREMOVE=1\nFDNAME={name}"), None)
    }

    /// Send one datagram to the notify socket, with one fd in `SCM_RIGHTS` when `fd` is given.
    fn send(&self, text: &str, fd: Option<RawFd>) -> io::Result<()> {
        // SAFETY: an all-zero sockaddr_un is valid; the fields are set below.
        let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
        address.sun_family = libc::AF_UNIX as libc::sa_family_t;
        let path = self.socket.as_os_str().as_bytes();
        if path.is_empty() || path.len() >= address.sun_path.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("the notify socket `{}` is not a socket path", self.socket.display()),
            ));
        }
        for (slot, byte) in address.sun_path.iter_mut().zip(path) {
            *slot = *byte as libc::c_char;
        }
        // A path that starts with `@` names an abstract socket (sd_notify(3)).
        let abstract_name = path[0] == b'@';
        if abstract_name {
            address.sun_path[0] = 0;
        }
        let length = std::mem::size_of::<libc::sa_family_t>() + path.len() + usize::from(!abstract_name);
        let socket = std::os::unix::net::UnixDatagram::unbound()?;
        let mut iov = libc::iovec {
            iov_base: text.as_ptr() as *mut libc::c_void,
            iov_len: text.len(),
        };
        // SAFETY: CMSG_SPACE only computes a size.
        let space = unsafe { libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as u32) } as usize;
        let mut control = vec![0u8; space];
        // SAFETY: an all-zero msghdr is valid; the fields are set below.
        let mut header: libc::msghdr = unsafe { std::mem::zeroed() };
        header.msg_name = (&mut address as *mut libc::sockaddr_un).cast();
        header.msg_namelen = length as libc::socklen_t;
        header.msg_iov = &mut iov;
        header.msg_iovlen = 1;
        if let Some(fd) = fd {
            header.msg_control = control.as_mut_ptr().cast();
            header.msg_controllen = space;
            // SAFETY: the control buffer holds one header and one fd, as CMSG_SPACE computed.
            unsafe {
                let cmsg = libc::CMSG_FIRSTHDR(&header);
                (*cmsg).cmsg_level = libc::SOL_SOCKET;
                (*cmsg).cmsg_type = libc::SCM_RIGHTS;
                (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as u32) as usize;
                (libc::CMSG_DATA(cmsg) as *mut RawFd).write_unaligned(fd);
            }
        }
        // SAFETY: the header points to live buffers.
        let sent = unsafe { libc::sendmsg(socket.as_raw_fd(), &header, libc::MSG_NOSIGNAL) };
        if sent < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}
