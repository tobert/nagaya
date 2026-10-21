//! The pure parts of a build: the recipe of each boundary and the steps that feed the rest (spec 6.2, 11.3c).

// A plan of one pass is a list of one range, not the list of the tokens in it.
#![allow(clippy::single_range_in_vec_init)]

use nagaya_abi::Kind;
use nagaya_council::build::{Step, boundary_recipes, plan_chunks, plan_steps, root_recipe};
use nagaya_council::family::{Boundary, FamilyError, Rendering};

fn rendering(tokens: Vec<u32>, cuts: &[(usize, usize)]) -> Rendering {
    let end = tokens.len();
    Rendering {
        tokens,
        end,
        boundaries: cuts
            .iter()
            .map(|&(message, tokens)| Boundary { message, tokens })
            .collect(),
        control_text: Vec::new(),
    }
}

fn root() -> nagaya_store::RecipeId {
    root_recipe(&[7; 32], b"")
}

#[test]
fn a_shared_prefix_gives_the_same_recipe_at_the_shared_boundary_and_a_later_edit_does_not() {
    let a = rendering(vec![1, 2, 3, 4, 5, 6], &[(0, 3), (1, 6)]);
    let b = rendering(vec![1, 2, 3, 4, 5, 9, 9], &[(0, 3), (2, 7)]);
    let ra = boundary_recipes(root(), &a).unwrap();
    let rb = boundary_recipes(root(), &b).unwrap();
    assert_eq!(ra[0], rb[0], "the first boundary is the same tokens and position");
    assert_ne!(ra[1], rb[1]);
}

#[test]
fn an_earlier_edit_changes_every_later_recipe() {
    let a = rendering(vec![1, 2, 3, 4, 5, 6], &[(0, 3), (1, 6)]);
    let b = rendering(vec![1, 2, 8, 4, 5, 6], &[(0, 3), (1, 6)]);
    let ra = boundary_recipes(root(), &a).unwrap();
    let rb = boundary_recipes(root(), &b).unwrap();
    assert_ne!(ra[0], rb[0]);
    assert_ne!(ra[1], rb[1], "the later recipe covers the earlier one");
}

#[test]
fn boundaries_are_content_the_same_tokens_with_other_boundaries_give_another_end_recipe() {
    let a = rendering(vec![1, 2, 3, 4, 5, 6], &[(0, 3), (1, 6)]);
    let b = rendering(vec![1, 2, 3, 4, 5, 6], &[(1, 6)]);
    let ra = boundary_recipes(root(), &a).unwrap();
    let rb = boundary_recipes(root(), &b).unwrap();
    assert_ne!(ra[1], rb[0]);
}

#[test]
fn the_identity_is_part_of_every_recipe() {
    let a = rendering(vec![1, 2, 3], &[(0, 3)]);
    let one = boundary_recipes(root_recipe(&[1; 32], b""), &a).unwrap();
    let two = boundary_recipes(root_recipe(&[2; 32], b""), &a).unwrap();
    assert_ne!(one, two);
    let other_geometry = boundary_recipes(root_recipe(&[1; 32], b"g"), &a).unwrap();
    assert_ne!(one, other_geometry);
}

#[test]
fn each_recipe_carries_its_boundary() {
    let a = rendering(vec![1, 2, 3, 4], &[(0, 2), (3, 4)]);
    let recipes = boundary_recipes(root(), &a).unwrap();
    assert_eq!(recipes.len(), 2);
    assert_eq!(recipes[0].boundary, Boundary { message: 0, tokens: 2 });
    assert_eq!(recipes[1].boundary, Boundary { message: 3, tokens: 4 });
}

#[test]
fn boundaries_that_do_not_increase_are_an_internal_error() {
    for cuts in [
        &[(0usize, 2usize), (1, 4), (2, 4)][..],
        &[(0, 3), (1, 2), (2, 4)][..],
        &[(0, 0), (1, 4)][..],
    ] {
        let a = rendering(vec![1, 2, 3, 4], cuts);
        let error = boundary_recipes(root(), &a).unwrap_err();
        assert!(matches!(error, FamilyError::Internal(_)), "{cuts:?}: {error:?}");
    }
}

#[test]
fn a_rendering_whose_last_boundary_is_not_its_end_is_an_internal_error() {
    let a = rendering(vec![1, 2, 3, 4], &[(0, 2)]);
    assert!(matches!(boundary_recipes(root(), &a), Err(FamilyError::Internal(_))));
    let none = rendering(vec![1, 2, 3, 4], &[]);
    assert!(matches!(boundary_recipes(root(), &none), Err(FamilyError::Internal(_))));
}

fn bounds(tokens: &[usize]) -> Vec<Boundary> {
    tokens
        .iter()
        .enumerate()
        .map(|(message, &tokens)| Boundary { message, tokens })
        .collect()
}

#[test]
fn the_plan_feeds_up_to_each_boundary_and_holds_a_snapshot_there() {
    let steps = plan_steps(0, &bounds(&[5, 8]), 100);
    assert_eq!(
        steps,
        [
            Step::Feed { from: 0, to: 5 },
            Step::Snapshot { boundary: 0 },
            Step::Feed { from: 5, to: 8 },
            Step::Snapshot { boundary: 1 },
        ]
    );
}

#[test]
fn a_feed_has_at_most_one_chunk_of_tokens_and_never_crosses_a_boundary() {
    let steps = plan_steps(0, &bounds(&[5, 8]), 2);
    let feeds: Vec<(usize, usize)> = steps
        .iter()
        .filter_map(|step| match step {
            Step::Feed { from, to } => Some((*from, *to)),
            Step::Snapshot { .. } => None,
        })
        .collect();
    assert_eq!(feeds, [(0, 2), (2, 4), (4, 5), (5, 7), (7, 8)]);
    assert!(feeds.iter().all(|(from, to)| to - from <= 2));
}

#[test]
fn the_plan_starts_after_the_kept_tokens_and_skips_the_boundaries_it_holds() {
    let steps = plan_steps(5, &bounds(&[3, 5, 9]), 100);
    assert_eq!(steps, [Step::Feed { from: 5, to: 9 }, Step::Snapshot { boundary: 2 }]);
}

#[test]
fn a_context_that_is_all_kept_has_an_empty_plan() {
    assert_eq!(plan_steps(9, &bounds(&[3, 9]), 4), []);
}

#[test]
fn a_feed_is_a_prefill_and_a_snapshot_is_a_copy() {
    assert_eq!(Step::Feed { from: 0, to: 1 }.kind(), Kind::Prefill);
    assert_eq!(Step::Snapshot { boundary: 0 }.kind(), Kind::Copy);
}

#[test]
#[should_panic(expected = "chunk")]
fn a_chunk_of_zero_tokens_is_a_defect() {
    plan_steps(0, &bounds(&[3]), 0);
}

/// The ranges of a plan, with a bound. Each range holds a token, so a plan with more than `len` ranges has an empty
/// one and never ends.
fn chunks(len: usize, rows: &[usize], chunk: usize, max_rows: usize) -> Vec<std::ops::Range<usize>> {
    let ranges: Vec<_> = plan_chunks(len, rows, chunk, max_rows).take(len + 1).collect();
    assert!(
        ranges.len() <= len,
        "the plan of {len} tokens, rows {rows:?}, chunk {chunk}, max_rows {max_rows} has more than {len} ranges: \
         {:?}",
        &ranges[..ranges.len().min(6)]
    );
    ranges
}

#[test]
fn chunks_cover_the_tokens_in_order_and_hold_at_most_one_chunk() {
    assert_eq!(chunks(10, &[], 4, 8), [0..4, 4..8, 8..10]);
    assert_eq!(chunks(4, &[], 4, 8), [0..4]);
    assert!(chunks(0, &[], 4, 8).is_empty());
}

#[test]
fn a_chunk_holds_at_most_the_rows_that_a_pass_gives() {
    let rows: Vec<usize> = (0..10).collect();
    assert_eq!(chunks(10, &rows, 100, 3), [0..3, 3..6, 6..9, 9..10]);
    // Rows far apart share a chunk.
    assert_eq!(chunks(10, &[1, 8], 100, 2), [0..10]);
}

#[test]
fn a_later_chunk_counts_its_rows_from_its_own_start() {
    // The second range starts at row 8. Its rows are [8], so it holds the rest of the tokens.
    assert_eq!(chunks(10, &[1, 5, 8], 100, 2), [0..8, 8..10]);
    assert_eq!(chunks(12, &[0, 1, 2, 3, 4, 5], 100, 2), [0..2, 2..4, 4..12]);
}

#[test]
fn every_row_falls_in_exactly_one_chunk_and_no_chunk_is_empty() {
    for len in 0..30usize {
        for step in 1..5usize {
            let rows: Vec<usize> = (0..len).step_by(step).collect();
            for chunk in 1..9usize {
                for max_rows in 1..5usize {
                    let ranges = chunks(len, &rows, chunk, max_rows);
                    let mut next = 0;
                    for range in &ranges {
                        assert_eq!(range.start, next);
                        assert!(range.end > range.start && range.len() <= chunk);
                        assert!(rows.iter().filter(|r| range.contains(r)).count() <= max_rows);
                        next = range.end;
                    }
                    assert_eq!(next, len);
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "rows")]
fn rows_that_do_not_increase_are_a_defect() {
    let _ = plan_chunks(10, &[3, 3], 4, 4);
}

#[test]
#[should_panic(expected = "rows")]
fn a_row_past_the_last_token_is_a_defect() {
    let _ = plan_chunks(10, &[10], 4, 4);
}
