//! The state of each engine in gossip (spec 2.3, "The state of an engine in gossip").
//!
//! The main loop asks the feed which states to write. The feed writes the key of one engine at most once each second,
//! and the newest state wins. It counts the writes of each engine, so a later rule for flaps can start from the count.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::{Duration, Instant};

use nagaya_telemetry::EngineState;

/// The shortest time between two writes of the state of one engine.
pub const MIN_INTERVAL: Duration = Duration::from_secs(1);

/// The window in which the feed counts the writes of one engine.
pub const FLAP_WINDOW: Duration = Duration::from_secs(60);

/// The most writes of one engine in the window before the feed logs a flap.
pub const FLAP_LIMIT: usize = 10;

/// The writes of engine states to gossip.
#[derive(Debug, Default)]
pub struct StateFeed {
    /// The times of the writes of each engine in the window, oldest first.
    writes: BTreeMap<String, VecDeque<Instant>>,
    /// The engines with more than `FLAP_LIMIT` writes in the window. The feed logs each one once.
    flapping: BTreeSet<String>,
}

impl StateFeed {
    /// A feed with no write.
    pub fn new() -> Self {
        Self::default()
    }

    /// The states to write now. Each entry gives an engine, its state, and its state in gossip.
    ///
    /// A state that equals its state in gossip needs no write. A changed state waits until the last write of its
    /// engine is `MIN_INTERVAL` old. The caller writes each state that this gives.
    pub fn due(
        &mut self,
        engines: &[(String, EngineState, Option<EngineState>)],
        now: Instant,
    ) -> Vec<(String, EngineState)> {
        let mut due = Vec::new();
        for (name, state, published) in engines {
            let writes = self.writes.entry(name.clone()).or_default();
            while writes.front().is_some_and(|at| now.duration_since(*at) > FLAP_WINDOW) {
                writes.pop_front();
            }
            if writes.len() <= FLAP_LIMIT && self.flapping.remove(name) {
                tracing::info!("engine `{name}` changes its state less often now");
            }
            if *published == Some(*state) {
                continue;
            }
            if writes.back().is_some_and(|at| now.duration_since(*at) < MIN_INTERVAL) {
                continue;
            }
            writes.push_back(now);
            if writes.len() > FLAP_LIMIT && self.flapping.insert(name.clone()) {
                tracing::warn!(
                    "engine `{name}` changed its state {} times in {} s; check the client that loads and unloads it",
                    writes.len(),
                    FLAP_WINDOW.as_secs()
                );
            }
            due.push((name.clone(), *state));
        }
        due
    }

    /// The writes of an engine in the window that ends at `now`.
    pub fn writes(&self, engine: &str, now: Instant) -> usize {
        self.writes.get(engine).map_or(0, |writes| {
            writes
                .iter()
                .filter(|at| now.duration_since(**at) <= FLAP_WINDOW)
                .count()
        })
    }

    /// True while an engine has more than `FLAP_LIMIT` writes in the window.
    pub fn flapping(&self, engine: &str) -> bool {
        self.flapping.contains(engine)
    }
}
