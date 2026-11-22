//! The long pass of an engine: a pass with a bound that the engine declares before it runs (engine-abi D8, spec 7.4).
//!
//! Each engine of a group runs inside a [`Watched`] engine. Before each pass, it asks the engine for the bound of the
//! pass. A pass with no bound runs as before, under the heartbeat of its thread. A long pass arms a deadline in the
//! watchdog for that pass alone, and the table [`LongPasses`] shows it to the route `busy`.

use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nagaya_abi::{
    Arena, Control, Coverage, DeviceId, Embedding, Engine, Identity, Image, Limits, Parts, PassError, PassErrorKind,
    PassOut, PassStatus, Seal, Segment, Slots, Speech, SpeechFrames, Stages, Tap,
};
use serde::Serialize;

use super::WatchMessage;

/// The key of a watch: the group, and the stage of a device thread (spec 7.4).
pub(super) type WatchKey = (usize, Option<usize>);

/// A long pass that runs now.
#[derive(Clone, Debug)]
struct Running {
    group: usize,
    devices: Vec<DeviceId>,
    started: Instant,
    bound: NonZeroU32,
}

/// What the route `GET /nagaya/v1/devices/{id}/busy` shows of a long pass (spec 7.4).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LongPass {
    /// The group of the pass.
    pub group: usize,
    /// The machine devices of the engine of the pass.
    pub devices: Vec<DeviceId>,
    /// The bound that the engine declared.
    pub bound_ms: u32,
    /// The time since the pass started.
    pub elapsed_ms: u64,
    /// True when the pass runs past its bound.
    pub overdue: bool,
}

/// The long passes that run now, one at most for each watch. The group and device threads write it, and the HTTP
/// routes read it.
#[derive(Clone, Debug, Default)]
pub struct LongPasses(Arc<Mutex<BTreeMap<WatchKey, Running>>>);

impl LongPasses {
    /// The long pass that runs on machine device `device`, or `None`.
    pub fn on_device(&self, device: DeviceId) -> Option<LongPass> {
        let now = Instant::now();
        let passes = self.0.lock().expect("the table of long passes is poisoned");
        passes.values().find(|pass| pass.devices.contains(&device)).map(|pass| {
            let elapsed_ms = now.saturating_duration_since(pass.started).as_millis() as u64;
            LongPass {
                group: pass.group,
                devices: pass.devices.clone(),
                bound_ms: pass.bound.get(),
                elapsed_ms,
                overdue: elapsed_ms > u64::from(pass.bound.get()),
            }
        })
    }

    fn insert(&self, key: WatchKey, pass: Running) {
        let old = self
            .0
            .lock()
            .expect("the table of long passes is poisoned")
            .insert(key, pass);
        assert!(old.is_none(), "watch {key:?} starts a long pass while one runs");
    }

    fn remove(&self, key: WatchKey) {
        self.0
            .lock()
            .expect("the table of long passes is poisoned")
            .remove(&key);
    }
}

/// What a group gives each engine that it opens, so that the engine's long passes reach the watchdog.
#[derive(Clone)]
pub(super) struct LongLinks {
    pub heartbeats: Sender<WatchMessage>,
    pub passes: LongPasses,
    /// The watchdog timeout (spec 7.4).
    pub timeout: Duration,
    /// The heartbeat interval.
    pub interval: Duration,
}

/// The deadline of a long pass of bound `bound` that starts at `started`: the start plus the larger of the timeout and
/// the bound plus the heartbeat interval (engine-abi D8).
pub(super) fn deadline(started: Instant, bound: NonZeroU32, timeout: Duration, interval: Duration) -> Instant {
    started + timeout.max(Duration::from_millis(u64::from(bound.get())) + interval)
}

/// An engine of a group, with the watch of its long passes. Every other call goes to the engine as it is.
pub(super) struct Watched {
    inner: Box<dyn Engine>,
    /// The limits of the open, which the open checked against the admitted limits.
    limits: Limits,
    key: WatchKey,
    devices: Vec<DeviceId>,
    links: LongLinks,
    /// The id of the next long pass.
    next: u64,
    /// True after a long pass returned past its deadline. The process stops, and the engine runs no more passes.
    late: bool,
}

impl Watched {
    /// Watch the long passes of `inner`, which runs on the thread of `key` and holds the machine devices `devices`.
    pub(super) fn boxed(
        inner: Box<dyn Engine>,
        limits: Limits,
        key: WatchKey,
        devices: Vec<DeviceId>,
        links: LongLinks,
    ) -> Box<dyn Engine> {
        Box::new(Self {
            inner,
            limits,
            key,
            devices,
            links,
            next: 0,
            late: false,
        })
    }

    fn refused(message: String) -> PassError {
        PassError {
            kind: PassErrorKind::Refused,
            message,
        }
    }
}

impl Engine for Watched {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        if self.late {
            return Err(Self::refused(
                "nothing ran: a long pass of this engine ran past its deadline, and the process stops (spec 7.4)"
                    .to_owned(),
            ));
        }
        let bound = match self.inner.pass_bound_ms(segments, taps) {
            Ok(None) => return self.inner.pass(segments, taps),
            Ok(Some(bound)) => bound,
            Err(message) => {
                return Err(Self::refused(format!(
                    "nothing ran: the engine refused the pass: {message}"
                )));
            }
        };
        if let Err(message) = self.limits.check_pass_bound(segments, bound) {
            return Err(Self::refused(format!("nothing ran: {message}")));
        }
        let dispatch = self.next;
        self.next += 1;
        let started = Instant::now();
        let deadline = deadline(started, bound, self.links.timeout, self.links.interval);
        self.links.passes.insert(
            self.key,
            Running {
                group: self.key.0,
                devices: self.devices.clone(),
                started,
                bound,
            },
        );
        // The watchdog has the deadline before the pass starts, so no beat is due while the pass runs.
        let _ = self.links.heartbeats.send(WatchMessage::LongStart {
            key: self.key,
            dispatch,
            started,
            bound,
            deadline,
        });
        let result = self.inner.pass(segments, taps);
        let at = Instant::now();
        // The watchdog reads the same instant, so it stops the process when this check fails.
        let _ = self.links.heartbeats.send(WatchMessage::LongEnd {
            key: self.key,
            dispatch,
            at,
        });
        if at > deadline {
            self.late = true;
            return Err(PassError {
                kind: PassErrorKind::Ran(PassStatus::Timeout),
                message: format!(
                    "the pass ran {} ms, past its deadline of {} ms; its result is dropped, and the process stops \
                     (spec 7.4)",
                    at.duration_since(started).as_millis(),
                    deadline.duration_since(started).as_millis()
                ),
            });
        }
        self.links.passes.remove(self.key);
        result
    }

    fn pass_bound_ms(&self, segments: &[Segment], taps: &[Tap]) -> Result<Option<NonZeroU32>, String> {
        self.inner.pass_bound_ms(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn Seal> {
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }

    fn stages(&mut self) -> Option<&mut dyn Stages> {
        self.inner.stages()
    }

    fn arena(&mut self) -> Option<&mut dyn Arena> {
        self.inner.arena()
    }

    fn speech(&mut self) -> Option<&mut dyn Speech> {
        self.inner.speech()
    }

    fn frames(&mut self) -> Option<&mut dyn SpeechFrames> {
        self.inner.frames()
    }

    fn image(&mut self) -> Option<&mut dyn Image> {
        self.inner.image()
    }

    fn embedding(&mut self) -> Option<&mut dyn Embedding> {
        self.inner.embedding()
    }

    fn control(&mut self) -> Option<&mut dyn Control> {
        self.inner.control()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{self, Receiver};
    use std::thread;

    use super::super::{Event, watchdog};
    use super::*;

    const KEY: WatchKey = (0, None);

    /// A watchdog with a timeout of `timeout` that watches group 0, and the receiver of its events.
    fn watched(timeout: Duration) -> (Sender<WatchMessage>, Receiver<Event>) {
        let (messages, received) = mpsc::channel();
        let (events, heard) = mpsc::channel();
        thread::spawn(move || watchdog(received, events, timeout));
        messages
            .send(WatchMessage::Start {
                group: 0,
                devices: vec![0],
            })
            .unwrap();
        (messages, heard)
    }

    fn start(messages: &Sender<WatchMessage>, dispatch: u64, deadline: Duration) -> Instant {
        let started = Instant::now();
        messages
            .send(WatchMessage::LongStart {
                key: KEY,
                dispatch,
                started,
                bound: NonZeroU32::new(10).unwrap(),
                deadline: started + deadline,
            })
            .unwrap();
        started
    }

    fn stalled() -> Event {
        Event::Stalled {
            group: 0,
            devices: vec![0],
        }
    }

    /// Falsified by a beat that clears or extends the deadline of a long pass.
    #[test]
    fn a_beat_does_not_extend_the_deadline() {
        let (messages, events) = watched(Duration::from_secs(5));
        let started = start(&messages, 0, Duration::from_millis(60));
        for _ in 0..40 {
            let _ = messages.send(WatchMessage::Beat {
                group: 0,
                at: Instant::now(),
            });
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(events.recv_timeout(Duration::from_secs(1)), Ok(stalled()));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    /// Falsified by an end that clears the pass of another dispatch.
    #[test]
    fn the_end_of_another_dispatch_does_not_clear_the_deadline() {
        let (messages, events) = watched(Duration::from_secs(5));
        start(&messages, 1, Duration::from_millis(50));
        messages
            .send(WatchMessage::LongEnd {
                key: KEY,
                dispatch: 2,
                at: Instant::now(),
            })
            .unwrap();
        assert_eq!(events.recv_timeout(Duration::from_secs(1)), Ok(stalled()));
    }

    /// Falsified by an end that clears the pass when it returned past its deadline.
    #[test]
    fn an_end_past_the_deadline_stops_the_process() {
        let (messages, events) = watched(Duration::from_secs(5));
        let started = start(&messages, 0, Duration::from_millis(30));
        messages
            .send(WatchMessage::LongEnd {
                key: KEY,
                dispatch: 0,
                at: started + Duration::from_millis(40),
            })
            .unwrap();
        assert_eq!(events.recv_timeout(Duration::from_secs(1)), Ok(stalled()));
    }

    /// An end before the deadline gives the watch back to the timeout: the beats keep it, and a stop of the beats
    /// stalls it. Falsified by an end that does not clear the pass.
    #[test]
    fn an_end_in_time_gives_the_watch_back_to_the_timeout() {
        let (messages, events) = watched(Duration::from_millis(100));
        start(&messages, 0, Duration::from_millis(150));
        messages
            .send(WatchMessage::LongEnd {
                key: KEY,
                dispatch: 0,
                at: Instant::now(),
            })
            .unwrap();
        for _ in 0..30 {
            let _ = messages.send(WatchMessage::Beat {
                group: 0,
                at: Instant::now(),
            });
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(events.try_recv(), Err(mpsc::TryRecvError::Empty));
        assert_eq!(events.recv_timeout(Duration::from_secs(1)), Ok(stalled()));
    }

    #[test]
    fn the_deadline_is_the_larger_of_the_timeout_and_the_bound_and_a_beat() {
        let at = Instant::now();
        let (timeout, beat) = (Duration::from_secs(30), Duration::from_secs(5));
        let ms = |value: u32| NonZeroU32::new(value).unwrap();
        assert_eq!(deadline(at, ms(10_000), timeout, beat), at + timeout);
        assert_eq!(
            deadline(at, ms(88_200), timeout, beat),
            at + Duration::from_millis(93_200)
        );
    }
}
