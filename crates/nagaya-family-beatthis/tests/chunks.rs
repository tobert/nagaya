//! The split of a long spectrum into chunks and the keep_first join of their logits, against upstream's own split and
//! join (`join.<T>.*` in the reference files).

mod common;

use nagaya_family_beatthis::chunks::{self, CHUNK, Chunk};
use nagaya_family_beatthis::mel::MELS;

/// The fake logits of the oracle: at chunk position i of chunk c, `10000 * c + (start + i)`.
fn tagged(plan: &[Chunk]) -> Vec<(Vec<f32>, Vec<f32>)> {
    plan.iter()
        .enumerate()
        .map(|(c, ch)| {
            let beat: Vec<f32> = (0..ch.len())
                .map(|i| (10_000 * c as isize + ch.start + i as isize) as f32)
                .collect();
            let down = beat.iter().map(|v| -v).collect();
            (beat, down)
        })
        .collect()
}

#[test]
fn a_short_piece_is_one_chunk_with_both_borders() {
    let plan = chunks::plan(26);
    assert_eq!(
        plan,
        [Chunk {
            start: -6,
            left: 6,
            rows: 26,
            right: 6
        }]
    );
    assert_eq!(plan[0].len(), 38);
}

#[test]
fn the_last_chunk_moves_left_to_end_at_the_piece() {
    let plan = chunks::plan(3501);
    let starts: Vec<isize> = plan.iter().map(|c| c.start).collect();
    assert_eq!(starts, [-6, 1482, 2007]);
    assert!(plan.iter().all(|c| c.len() == CHUNK));
}

#[test]
fn a_chunk_input_has_zero_rows_for_its_borders() {
    let t = 26;
    let spect: Vec<f32> = (0..t * MELS).map(|i| i as f32 + 1.0).collect();
    let plan = chunks::plan(t);
    let x = chunks::input(&spect, t, &plan[0]);
    assert_eq!(x.len(), 38 * MELS);
    assert!(x[..6 * MELS].iter().all(|&v| v == 0.0), "left border");
    assert_eq!(&x[6 * MELS..32 * MELS], &spect[..], "the rows");
    assert!(x[32 * MELS..].iter().all(|&v| v == 0.0), "right border");
}

#[test]
#[should_panic(expected = "logits")]
fn a_join_refuses_logits_of_the_wrong_length() {
    let plan = chunks::plan(26);
    chunks::join(26, &plan, &[(vec![0.0; 37], vec![0.0; 37])]);
}

#[test]
#[should_panic(expected = "does not cover frame")]
fn a_join_refuses_a_plan_with_a_gap() {
    let mut plan = chunks::plan(3000);
    plan.remove(1);
    let logits = tagged(&plan);
    chunks::join(3000, &plan, &logits);
}

#[test]
fn the_plan_and_the_join_match_upstream_at_each_length() {
    let Some(g) = common::golden() else { return };
    let join = g.cases["join"].as_object().expect("cases.json has no join");
    assert!(join.len() >= 16, "the oracle has {} lengths", join.len());
    for (t, want) in join {
        let t: usize = t.parse().unwrap();
        let plan = chunks::plan(t);
        let starts: Vec<i64> = plan.iter().map(|c| c.start as i64).collect();
        let sizes: Vec<i64> = plan.iter().map(|c| c.len() as i64).collect();
        let want_starts: Vec<i64> = want["starts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let want_sizes: Vec<i64> = want["sizes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(starts, want_starts, "{t} frames: starts");
        assert_eq!(sizes, want_sizes, "{t} frames: sizes");
        let (beat, down) = chunks::join(t, &plan, &tagged(&plan));
        let (want_beat, _) = g.f32(&format!("join.{t}.beat"));
        let (want_down, _) = g.f32(&format!("join.{t}.downbeat"));
        assert_eq!(beat, want_beat, "{t} frames: beat");
        assert_eq!(down, want_down, "{t} frames: downbeat");
    }
}
