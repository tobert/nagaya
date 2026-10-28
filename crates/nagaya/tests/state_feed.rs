//! The state of each engine in gossip (spec 2.3, "The state of an engine in gossip"): at most one write each second
//! for each engine, the newest state wins, and a count of writes for a later rule for flaps.

use std::time::{Duration, Instant};

use nagaya::notify::EngineState;
use nagaya::state_feed::{FLAP_LIMIT, StateFeed};

const SECOND: Duration = Duration::from_secs(1);

/// One engine `voice` in this state, with this state in gossip.
fn voice(state: EngineState, published: EngineState) -> Vec<(String, EngineState, Option<EngineState>)> {
    vec![("voice".to_owned(), state, Some(published))]
}

fn write(state: EngineState) -> Vec<(String, EngineState)> {
    vec![("voice".to_owned(), state)]
}

#[test]
fn a_changed_state_is_due_and_a_published_state_is_not() {
    let mut feed = StateFeed::new();
    let start = Instant::now();
    assert!(
        feed.due(&voice(EngineState::Ready, EngineState::Ready), start)
            .is_empty()
    );
    assert_eq!(
        feed.due(&voice(EngineState::Loading, EngineState::Ready), start),
        write(EngineState::Loading)
    );
    assert!(
        feed.due(&voice(EngineState::Loading, EngineState::Loading), start + 2 * SECOND)
            .is_empty()
    );
}

#[test]
fn a_second_change_in_the_same_second_waits_and_the_newest_state_wins() {
    let mut feed = StateFeed::new();
    let start = Instant::now();
    feed.due(&voice(EngineState::Draining, EngineState::Ready), start);
    let soon = start + Duration::from_millis(300);
    assert!(
        feed.due(&voice(EngineState::Unloaded, EngineState::Draining), soon)
            .is_empty(),
        "the second write waits"
    );
    let later = start + Duration::from_millis(600);
    assert!(
        feed.due(&voice(EngineState::Loading, EngineState::Draining), later)
            .is_empty()
    );
    assert_eq!(
        feed.due(&voice(EngineState::Loading, EngineState::Draining), start + SECOND),
        write(EngineState::Loading),
        "the newest state, after one second"
    );
}

#[test]
fn a_state_that_changes_back_within_the_second_makes_no_write() {
    let mut feed = StateFeed::new();
    let start = Instant::now();
    feed.due(&voice(EngineState::Draining, EngineState::Ready), start);
    let soon = start + Duration::from_millis(200);
    assert!(
        feed.due(&voice(EngineState::Unloaded, EngineState::Draining), soon)
            .is_empty()
    );
    assert!(
        feed.due(&voice(EngineState::Draining, EngineState::Draining), start + 2 * SECOND)
            .is_empty()
    );
}

#[test]
fn each_engine_has_its_own_second() {
    let mut feed = StateFeed::new();
    let start = Instant::now();
    feed.due(&voice(EngineState::Draining, EngineState::Ready), start);
    let both = vec![
        ("voice".to_owned(), EngineState::Unloaded, Some(EngineState::Draining)),
        ("chat".to_owned(), EngineState::Loading, Some(EngineState::Unloaded)),
    ];
    assert_eq!(
        feed.due(&both, start + Duration::from_millis(100)),
        vec![("chat".to_owned(), EngineState::Loading)]
    );
}

#[test]
fn an_engine_with_no_state_in_gossip_is_due() {
    let mut feed = StateFeed::new();
    let fresh = vec![("voice".to_owned(), EngineState::Unloaded, None)];
    assert_eq!(feed.due(&fresh, Instant::now()), write(EngineState::Unloaded));
}

#[test]
fn more_than_the_limit_of_writes_in_a_minute_is_a_flap_until_the_writes_age_out() {
    let mut feed = StateFeed::new();
    let start = Instant::now();
    let states = [EngineState::Ready, EngineState::Unloaded];
    for step in 0..=FLAP_LIMIT {
        let now = start + SECOND * step as u32;
        let state = states[step % 2];
        let published = states[(step + 1) % 2];
        assert_eq!(feed.due(&voice(state, published), now), write(state), "write {step}");
    }
    let last = start + SECOND * FLAP_LIMIT as u32;
    assert_eq!(feed.writes("voice", last), FLAP_LIMIT + 1);
    assert!(feed.flapping("voice"));
    // A minute with no write ages the writes out.
    let calm = last + Duration::from_secs(61);
    assert!(
        feed.due(&voice(EngineState::Ready, EngineState::Ready), calm)
            .is_empty()
    );
    assert_eq!(feed.writes("voice", calm), 0);
    assert!(!feed.flapping("voice"));
}
