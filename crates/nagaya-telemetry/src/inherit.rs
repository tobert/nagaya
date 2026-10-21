//! The telemetry that a thread gives to the thread that it starts (spec 9.5).
//!
//! The instruments and the span dispatcher belong to one thread each, so that a test on its own thread reads only its
//! own points. A thread that the process starts would record nowhere in a test. Thus the starting thread captures its
//! telemetry, and the new thread enters it.

use tracing::Dispatch;
use tracing::dispatcher::{DefaultGuard, set_default};

use crate::instruments::{Instruments, ScopeGuard, instruments, scope};
use std::sync::Arc;

/// The instruments and the span dispatcher of one thread.
///
/// In the process, every thread uses the instruments and the dispatcher of the process, so this changes nothing there.
pub struct Inherit {
    instruments: Arc<Instruments>,
    dispatch: Dispatch,
}

impl Inherit {
    /// Take the telemetry of the calling thread.
    pub fn capture() -> Self {
        Self {
            instruments: instruments(),
            dispatch: tracing::dispatcher::get_default(Dispatch::clone),
        }
    }

    /// Make the calling thread record on this telemetry until the guard drops.
    pub fn enter(self) -> InheritGuard {
        InheritGuard {
            // The fields drop in the order of the declaration: the dispatcher first, then the instruments.
            _dispatch: set_default(&self.dispatch),
            _scope: scope(self.instruments),
        }
    }
}

/// The guard of [`Inherit::enter`]. The thread has its earlier telemetry again when the guard drops.
pub struct InheritGuard {
    _dispatch: DefaultGuard,
    _scope: ScopeGuard,
}
