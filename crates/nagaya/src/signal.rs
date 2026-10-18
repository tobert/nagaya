//! The SIGTERM stop signal (spec 10.3).
//!
//! std has no signal API. The process declares `signal(2)` here and sets one atomic flag. A Proposed note in
//! `docs/spec.md` 10.3 gives the reason.

use std::io;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// The SIGTERM number on Linux (`signal(7)`).
pub const SIGTERM: i32 = 15;

/// The value that `signal(2)` gives on failure.
const SIG_ERR: usize = usize::MAX;

/// The flag that the signal handler raises. A worker reads it to learn that a stop was requested.
pub struct SignalFlag {
    count: AtomicUsize,
}

impl SignalFlag {
    /// Make a flag at zero. It can be a `static`.
    pub const fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
        }
    }

    /// The number of signals that arrived.
    pub fn count(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }

    /// Add one signal. The handler calls this, and a test may call it.
    pub fn raise(&self) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }
}

impl Default for SignalFlag {
    fn default() -> Self {
        Self::new()
    }
}

/// The flag that the handler raises. It lives for the process.
static TERM_FLAG: AtomicPtr<SignalFlag> = AtomicPtr::new(std::ptr::null_mut());

/// The C handler. It does one atomic load and one atomic increment (async-signal-safe).
extern "C" fn handle_terminate(_signal: i32) {
    let flag = TERM_FLAG.load(Ordering::SeqCst);
    if !flag.is_null() {
        // SAFETY: `install_terminate` stores a pointer to a flag that lives for the process.
        unsafe { (*flag).raise() };
    }
}

unsafe extern "C" {
    /// Install a handler for a signal. The return value is the previous handler, or `SIG_ERR`.
    fn signal(signal: i32, handler: extern "C" fn(i32)) -> usize;
}

/// Send SIGTERM to `flag`. A second install replaces the first. The flag must live for the process.
pub fn install_terminate(flag: &'static SignalFlag) -> io::Result<()> {
    TERM_FLAG.store(flag as *const SignalFlag as *mut SignalFlag, Ordering::SeqCst);
    // SAFETY: the handler is `extern "C"` with the C signature, and it touches only one atomic.
    let previous = unsafe { signal(SIGTERM, handle_terminate) };
    if previous == SIG_ERR {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
