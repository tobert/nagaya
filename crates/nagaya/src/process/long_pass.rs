//! The long pass of an engine: a pass with a bound that the engine declares before it runs (engine-abi D8, spec 7.4).
//!
//! Each engine of a group runs inside a [`Watched`] engine. Before each pass, it asks the engine for the bound of the
//! pass. A pass with no bound runs as before, under the heartbeat of its thread. A long pass puts its deadline in the
//! table [`LongPasses`], which the watchdog and the route `busy` read.
//!
//! Whether a long pass ended in time is one decision, made under the lock of the table: by the end of the pass, or by
//! the watchdog when it finds the deadline past. The one that takes the lock first decides, and the other reads it.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use nagaya_abi::{
    Arena, Control, Coverage, DeviceId, Embedding, Engine, Identity, Image, Limits, Parts, PassError, PassErrorKind,
    PassOut, PassStatus, Seal, Segment, Slots, Speech, SpeechFrames, Stages, Tap,
};
use serde::Serialize;

/// The key of a watch: the group, and the stage of a device thread (spec 7.4).
pub(super) type WatchKey = (usize, Option<usize>);

/// The thread of a watch, in words, for a message.
fn thread_of(key: WatchKey) -> String {
    match key.1 {
        None => format!("the thread of group {}", key.0),
        Some(place) => format!("device thread {place} of group {}", key.0),
    }
}

/// A long pass that runs now.
#[derive(Clone, Debug)]
pub(super) struct Running {
    pub group: usize,
    pub devices: Vec<DeviceId>,
    pub started: Instant,
    pub bound: NonZeroU32,
    pub deadline: Instant,
    /// True after the report of a pass past its bound.
    pub reported: bool,
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

/// The state of the long passes of the process.
#[derive(Debug, Default)]
struct Table {
    /// The long pass that runs on each watch, one at most.
    running: BTreeMap<WatchKey, Running>,
    /// The end of the last long pass that ended in time on each watch. The watch counts it as a beat.
    ended: BTreeMap<WatchKey, Instant>,
    /// The watches whose long pass returned past its deadline. The watchdog stops the process for each one.
    late: BTreeSet<WatchKey>,
}

/// How a long pass ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Finish {
    /// Before its deadline: its result goes to the job.
    InTime,
    /// Past its deadline: its result is dropped, and the process stops.
    Late,
}

/// What the watchdog reads of the long passes at one instant.
#[derive(Debug, Default)]
pub(super) struct Judged {
    /// The watches with a long pass that is running and in time. The watchdog does not read their beats.
    pub running: BTreeSet<WatchKey>,
    /// The watches whose long pass went past its deadline. The process stops.
    pub late: Vec<WatchKey>,
    /// The watches whose long pass first ran past its bound now, with the bound in milliseconds.
    pub over_bound: Vec<(WatchKey, u32)>,
    /// The end of the last long pass of each watch that ended in time.
    pub ended: BTreeMap<WatchKey, Instant>,
}

/// The long passes of the process. The group and device threads write it, and the watchdog and the HTTP routes read
/// it.
#[derive(Clone, Debug, Default)]
pub struct LongPasses(Arc<Mutex<Table>>);

impl LongPasses {
    fn lock(&self) -> MutexGuard<'_, Table> {
        self.0.lock().expect("the table of long passes is poisoned")
    }

    /// The long pass that runs on machine device `device`, or `None`.
    pub fn on_device(&self, device: DeviceId) -> Option<LongPass> {
        let now = Instant::now();
        let table = self.lock();
        table
            .running
            .values()
            .find(|pass| pass.devices.contains(&device))
            .map(|pass| {
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

    /// Start the long pass of the thread of `key`.
    pub(super) fn arm(&self, key: WatchKey, pass: Running) {
        let mut table = self.lock();
        let devices = pass.devices.clone();
        let old = table.running.insert(key, pass);
        assert!(
            old.is_none(),
            "{} starts a long pass on devices {devices:?} while another long pass runs there; a thread runs one pass \
             at a time, so this is a defect of the group",
            thread_of(key)
        );
    }

    /// End the long pass of the thread of `key`, at the time of this call. The pass is late when the watchdog found
    /// its deadline past first, or when this call is past it. The entry goes in either case.
    pub(super) fn finish(&self, key: WatchKey) -> Finish {
        let mut table = self.lock();
        let now = Instant::now();
        match table.running.remove(&key) {
            Some(pass) if now <= pass.deadline => {
                table.ended.insert(key, now);
                Finish::InTime
            }
            Some(_) => {
                table.late.insert(key);
                Finish::Late
            }
            // The watchdog found the deadline past, took the entry, and stops the process.
            None => Finish::Late,
        }
    }

    /// Judge each long pass at `now`. A pass past its deadline leaves the table here, so its end reads it as late.
    pub(super) fn judge(&self, now: Instant) -> Judged {
        let mut table = self.lock();
        let mut judged = Judged {
            late: std::mem::take(&mut table.late).into_iter().collect(),
            ended: table.ended.clone(),
            ..Judged::default()
        };
        let past: Vec<WatchKey> = table
            .running
            .iter()
            .filter(|(_, pass)| now > pass.deadline)
            .map(|(key, _)| *key)
            .collect();
        for key in past {
            table.running.remove(&key);
            judged.late.push(key);
        }
        for (key, pass) in &mut table.running {
            judged.running.insert(*key);
            if !pass.reported && now.duration_since(pass.started) > Duration::from_millis(u64::from(pass.bound.get())) {
                pass.reported = true;
                judged.over_bound.push((*key, pass.bound.get()));
            }
        }
        judged
    }
}

/// What a group gives each engine that it opens, so that the engine's long passes reach the watchdog.
#[derive(Clone)]
pub(super) struct LongLinks {
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
    /// The name of the engine in the profile.
    name: String,
    /// The limits of the open, which the open checked against the admitted limits.
    limits: Limits,
    key: WatchKey,
    devices: Vec<DeviceId>,
    links: LongLinks,
    /// True after a long pass returned past its deadline. The process stops, and the engine runs no more passes.
    late: bool,
}

impl Watched {
    /// Watch the long passes of engine `name`, which runs on the thread of `key` and holds the machine devices
    /// `devices`.
    pub(super) fn boxed(
        inner: Box<dyn Engine>,
        name: String,
        limits: Limits,
        key: WatchKey,
        devices: Vec<DeviceId>,
        links: LongLinks,
    ) -> Box<dyn Engine> {
        Box::new(Self {
            inner,
            name,
            limits,
            key,
            devices,
            links,
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
            return Err(Self::refused(format!(
                "nothing ran: a long pass of engine `{}` ran past its deadline, and the process stops (spec 7.4); send \
                 the request again after the restart",
                self.name
            )));
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
        let started = Instant::now();
        let deadline = deadline(started, bound, self.links.timeout, self.links.interval);
        // The watchdog reads the deadline before the pass starts, so no beat is due while the pass runs.
        self.links.passes.arm(
            self.key,
            Running {
                group: self.key.0,
                devices: self.devices.clone(),
                started,
                bound,
                deadline,
                reported: false,
            },
        );
        let result = self.inner.pass(segments, taps);
        match self.links.passes.finish(self.key) {
            Finish::InTime => result,
            Finish::Late => {
                self.late = true;
                Err(PassError {
                    kind: PassErrorKind::Ran(PassStatus::Timeout),
                    message: format!(
                        "the pass of engine `{}` ran {} ms, past its deadline of {} ms; its result is dropped, and \
                         the process stops (spec 7.4); send the request again after the restart",
                        self.name,
                        started.elapsed().as_millis(),
                        deadline.duration_since(started).as_millis()
                    ),
                })
            }
        }
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
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::thread;

    use super::super::{Event, WatchMessage, watchdog};
    use super::*;

    const KEY: WatchKey = (0, None);

    /// A watchdog with a timeout of `timeout` that watches group 0, its table of long passes, and the receiver of its
    /// events.
    fn watched(timeout: Duration) -> (Sender<WatchMessage>, LongPasses, Receiver<Event>) {
        let (messages, received) = mpsc::channel();
        let (events, heard) = mpsc::channel();
        let passes = LongPasses::default();
        let read = passes.clone();
        thread::spawn(move || watchdog(received, events, timeout, read));
        messages
            .send(WatchMessage::Start {
                group: 0,
                devices: vec![0],
            })
            .unwrap();
        (messages, passes, heard)
    }

    fn arm(passes: &LongPasses, deadline: Duration) -> Instant {
        let started = Instant::now();
        passes.arm(
            KEY,
            Running {
                group: 0,
                devices: vec![0],
                started,
                bound: NonZeroU32::new(10).unwrap(),
                deadline: started + deadline,
                reported: false,
            },
        );
        started
    }

    fn stalled() -> Event {
        Event::Stalled {
            group: 0,
            devices: vec![0],
        }
    }

    fn beat(messages: &Sender<WatchMessage>) {
        let _ = messages.send(WatchMessage::Beat {
            group: 0,
            at: Instant::now(),
        });
    }

    /// Falsified by a beat that clears or extends the deadline of a long pass.
    #[test]
    fn a_beat_does_not_extend_the_deadline() {
        let (messages, passes, events) = watched(Duration::from_secs(5));
        let started = arm(&passes, Duration::from_millis(60));
        for _ in 0..40 {
            beat(&messages);
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(events.recv_timeout(Duration::from_secs(1)), Ok(stalled()));
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(
            passes.finish(KEY),
            Finish::Late,
            "the watchdog took the pass, so its end is late"
        );
    }

    /// An end in time is one decision: after it, the watchdog never stops the process for that pass, also when the
    /// deadline passes before it reads a message. Falsified by a watchdog that judges a pass that ended.
    #[test]
    fn an_end_in_time_is_never_a_stall() {
        let (messages, passes, events) = watched(Duration::from_millis(100));
        arm(&passes, Duration::from_millis(20));
        assert_eq!(passes.finish(KEY), Finish::InTime);
        for _ in 0..20 {
            beat(&messages);
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(events.try_recv(), Err(mpsc::TryRecvError::Empty));
    }

    /// The end of a long pass counts as a beat, so a watch whose last beat is older than the timeout does not stall
    /// at once. A stop of the beats then stalls it at the timeout. Falsified by an end that does not count as a beat,
    /// and by an end that leaves the pass in the table.
    #[test]
    fn an_end_in_time_gives_the_watch_back_to_the_timeout() {
        let (_messages, passes, events) = watched(Duration::from_millis(100));
        arm(&passes, Duration::from_millis(400));
        // No beat for 200 ms, which is more than the timeout.
        thread::sleep(Duration::from_millis(200));
        assert_eq!(passes.finish(KEY), Finish::InTime);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(events.try_recv(), Err(mpsc::TryRecvError::Empty), "the end is a beat");
        assert_eq!(events.recv_timeout(Duration::from_secs(1)), Ok(stalled()));
    }

    /// An end past the deadline is late, and it leaves the table. The watchdog reads it as late at its next judge, and
    /// a next long pass on the same thread arms with no panic. No watchdog runs here, so the end decides. Falsified by
    /// an end that is in time past the deadline, by an end that keeps the pass, and by a late end that the judge does
    /// not read.
    #[test]
    fn an_end_past_the_deadline_is_late_and_leaves_the_table() {
        let passes = LongPasses::default();
        arm(&passes, Duration::ZERO);
        thread::sleep(Duration::from_millis(2));
        assert_eq!(passes.finish(KEY), Finish::Late);
        assert_eq!(passes.on_device(0), None);
        let judged = passes.judge(Instant::now());
        assert_eq!(judged.late, [KEY]);
        assert!(judged.running.is_empty() && judged.ended.is_empty(), "{judged:?}");
        arm(&passes, Duration::from_secs(5));
        assert!(
            passes.judge(Instant::now()).late.is_empty(),
            "the judge reads a late end once"
        );
    }

    /// A late end reaches the watchdog, which stops the process.
    #[test]
    fn a_late_end_stops_the_process() {
        let (_messages, passes, events) = watched(Duration::from_secs(5));
        arm(&passes, Duration::ZERO);
        thread::sleep(Duration::from_millis(2));
        assert_eq!(passes.finish(KEY), Finish::Late);
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
