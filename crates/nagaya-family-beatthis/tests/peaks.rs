//! The minimal postprocess of upstream: peaks of the logits, then the downbeats moved onto beats.

mod common;

use nagaya_family_beatthis::peaks::{self, Beats};

#[test]
fn a_peak_is_a_maximum_within_three_frames_and_above_zero() {
    let mut beat = vec![-5.0f32; 40];
    beat[10] = 2.0; // a peak
    beat[12] = 1.0; // within three frames of a larger one: not a peak
    beat[20] = -0.5; // a maximum, but not above zero
    beat[30] = 0.5; // a peak
    let got = peaks::frames(&beat);
    assert_eq!(got, [10.0, 30.0]);
}

#[test]
fn a_maximum_of_exactly_zero_is_not_a_peak() {
    let mut beat = vec![-5.0f32; 20];
    beat[10] = 0.0;
    assert!(peaks::frames(&beat).is_empty());
}

/// Upstream's `deduplicate_peaks` keeps a running mean and compares the next frame with it, not with the last frame.
#[test]
fn adjacent_peaks_join_by_the_running_mean() {
    assert_eq!(peaks::deduplicate(&[3, 4, 5]), [3.5, 5.0]);
    assert_eq!(peaks::deduplicate(&[3, 4]), [3.5]);
    assert_eq!(peaks::deduplicate(&[3, 5, 6, 7]), [3.0, 5.5, 7.0]);
    assert_eq!(peaks::deduplicate(&[]), Vec::<f64>::new());
}

#[test]
fn two_equal_frames_side_by_side_are_one_peak_at_their_mean() {
    let mut beat = vec![-5.0f32; 20];
    beat[8] = 1.0;
    beat[9] = 1.0;
    assert_eq!(peaks::frames(&beat), [8.5]);
}

#[test]
fn a_downbeat_moves_to_the_nearest_beat_and_the_first_on_a_tie() {
    let got = peaks::snap(&[1.0, 2.0, 3.0], &[1.4, 2.5, 2.6]);
    assert_eq!(got, [1.0, 2.0, 3.0]);
    // Two downbeats onto one beat keep one.
    assert_eq!(peaks::snap(&[1.0, 2.0], &[1.1, 0.9]), [1.0]);
    // No beats: the downbeats stay where they are, as upstream skips the step.
    assert_eq!(peaks::snap(&[], &[1.25, 0.5]), [0.5, 1.25]);
}

#[test]
fn the_beats_match_upstream_in_each_case_and_model() {
    let Some(g) = common::golden() else { return };
    let mut checked = 0;
    for case in g.case_names() {
        for model in ["final0", "small0"] {
            let (beat, _) = g.f32(&format!("{model}.{case}.beat"));
            let (down, _) = g.f32(&format!("{model}.{case}.downbeat"));
            let (want_beats, _) = g.f64(&format!("{model}.{case}.beats_s"));
            let (want_downs, _) = g.f64(&format!("{model}.{case}.downbeats_s"));
            let Beats { beats, downbeats } = peaks::beats(&beat, &down);
            assert_eq!(beats, want_beats, "{model} {case}: beats");
            assert_eq!(downbeats, want_downs, "{model} {case}: downbeats");
            checked += 1;
        }
    }
    assert_eq!(checked, 10);
}
