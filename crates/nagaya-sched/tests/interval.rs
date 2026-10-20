//! The scheduler's part of interval copies (spec 6.13), on the limits of the fake engine.

use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Engine, Limits};
use nagaya_sched::{CopyDue, IntervalCopies, IntervalError, IntervalPolicy};

fn limits(grain: Option<u64>, segments_exact: bool) -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.checkpoint_grain = grain;
    limits.segments_exact = segments_exact;
    limits
}

fn tracker(grain: u64) -> IntervalCopies<&'static str> {
    let policy = IntervalPolicy::from_limits(&limits(Some(grain), true), 4)
        .unwrap()
        .expect("a grain and a cap make a policy");
    IntervalCopies::new(policy)
}

fn due(context: u64, position: u64) -> Option<CopyDue> {
    Some(CopyDue { context, position })
}

#[test]
fn an_engine_without_a_grain_makes_no_policy() {
    assert_eq!(IntervalPolicy::from_limits(&limits(None, true), 4), Ok(None));
}

#[test]
fn a_grain_and_a_cap_make_a_policy() {
    let policy = IntervalPolicy::from_limits(&limits(Some(256), true), 4)
        .unwrap()
        .unwrap();
    assert_eq!(policy.grain(), 256);
}

#[test]
fn a_cap_of_zero_makes_no_copies() {
    assert_eq!(IntervalPolicy::from_limits(&limits(Some(256), true), 0), Ok(None));
}

#[test]
fn a_grain_of_zero_is_an_error() {
    assert_eq!(
        IntervalPolicy::from_limits(&limits(Some(0), true), 4),
        Err(IntervalError::ZeroGrain)
    );
}

#[test]
fn a_grain_without_segments_exact_is_an_error() {
    assert_eq!(
        IntervalPolicy::from_limits(&limits(Some(256), false), 4),
        Err(IntervalError::NeedsSegmentsExact)
    );
}

#[test]
fn a_copy_is_due_when_a_quantum_reaches_a_multiple() {
    let mut copies = tracker(256);
    copies.track(1, 0).unwrap();
    assert_eq!(copies.advance(1, 255).unwrap(), None);
    assert_eq!(copies.advance(1, 256).unwrap(), due(1, 256));
    assert_eq!(copies.advance(1, 257).unwrap(), None);
}

#[test]
fn a_copy_is_due_after_a_quantum_that_crosses_a_multiple() {
    let mut copies = tracker(256);
    copies.track(1, 0).unwrap();
    assert_eq!(copies.advance(1, 300).unwrap(), due(1, 300));
}

#[test]
fn a_jump_over_several_multiples_makes_one_copy() {
    let mut copies = tracker(256);
    copies.track(1, 0).unwrap();
    assert_eq!(copies.advance(1, 1030).unwrap(), due(1, 1030));
    assert_eq!(copies.advance(1, 1100).unwrap(), None);
    assert_eq!(copies.advance(1, 1280).unwrap(), due(1, 1280));
}

#[test]
fn a_context_that_starts_late_waits_for_the_next_multiple() {
    let mut copies = tracker(256);
    copies.track(1, 300).unwrap();
    assert_eq!(copies.advance(1, 511).unwrap(), None);
    assert_eq!(copies.advance(1, 512).unwrap(), due(1, 512));
}

#[test]
fn a_failed_put_is_not_tried_again_before_the_next_multiple() {
    let mut copies = tracker(256);
    copies.track(1, 0).unwrap();
    assert_eq!(copies.advance(1, 256).unwrap(), due(1, 256));
    // The put failed: the caller never calls `stored`.
    assert_eq!(copies.advance(1, 257).unwrap(), None);
    assert_eq!(copies.advance(1, 511).unwrap(), None);
    assert_eq!(copies.advance(1, 512).unwrap(), due(1, 512));
    assert_eq!(copies.copy(1).unwrap(), None);
}

#[test]
fn a_new_copy_names_the_previous_copy_of_its_context_for_removal() {
    let mut copies = tracker(256);
    copies.track(1, 0).unwrap();
    copies.track(2, 0).unwrap();
    assert_eq!(copies.stored(1, "first").unwrap(), None);
    assert_eq!(copies.stored(2, "other").unwrap(), None);
    assert_eq!(copies.stored(1, "second").unwrap(), Some("first"));
    assert_eq!(copies.copy(1).unwrap(), Some(&"second"));
    assert_eq!(copies.copy(2).unwrap(), Some(&"other"));
}

#[test]
fn contexts_count_their_multiples_on_their_own() {
    let mut copies = tracker(256);
    copies.track(1, 0).unwrap();
    copies.track(2, 0).unwrap();
    assert_eq!(copies.advance(1, 256).unwrap(), due(1, 256));
    assert_eq!(copies.advance(2, 100).unwrap(), None);
    assert_eq!(copies.advance(2, 300).unwrap(), due(2, 300));
}

#[test]
fn a_position_that_goes_back_is_an_error() {
    let mut copies = tracker(256);
    copies.track(1, 0).unwrap();
    copies.advance(1, 300).unwrap();
    assert_eq!(
        copies.advance(1, 299),
        Err(IntervalError::PositionWentBack {
            context: 1,
            from: 300,
            to: 299
        })
    );
}

#[test]
fn an_unknown_or_a_duplicate_context_is_an_error() {
    let mut copies = tracker(256);
    assert_eq!(copies.advance(9, 10), Err(IntervalError::UnknownContext(9)));
    assert_eq!(copies.stored(9, "x"), Err(IntervalError::UnknownContext(9)));
    assert_eq!(copies.copy(9), Err(IntervalError::UnknownContext(9)));
    copies.track(1, 0).unwrap();
    assert_eq!(copies.track(1, 0), Err(IntervalError::DuplicateContext(1)));
}

#[test]
fn an_ended_context_is_forgotten_and_its_copy_stays_with_the_store() {
    let mut copies = tracker(256);
    copies.track(1, 0).unwrap();
    copies.stored(1, "kept").unwrap();
    assert_eq!(copies.untrack(1).unwrap(), Some("kept"));
    assert_eq!(copies.advance(1, 10), Err(IntervalError::UnknownContext(1)));
}
