//! The pure parts of a build: the recipe of each boundary and the steps that feed the rest (spec 6.2, 11.3c).

// A plan of one pass is a list of one range, not the list of the tokens in it.
#![allow(clippy::single_range_in_vec_init)]

use nagaya_abi::Kind;
use nagaya_council::build::{Step, boundary_recipes, plan_chunks, plan_steps, range_recipe, root_recipe};
use nagaya_council::family::{Boundary, FamilyError, MediaSpan, Rendering};
use nagaya_store::{ContentId, RecipeMedia, recipe_boundary, recipe_boundary_media};

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
        media: Vec::new(),
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

/// A continuation that starts from the head of a cut reply keeps the seal of that head, inside its reply. A build may
/// also start from a cut. The state of the reply has both among its boundaries (spec 11.3c).
#[test]
fn the_recipes_with_cuts_have_each_cut_among_the_boundaries() {
    use nagaya_council::build::recipes_with_cuts;
    let plain = rendering(vec![1, 2, 3, 4, 5, 6], &[(0, 3), (2, 6)]);
    let cuts = [Boundary { message: 1, tokens: 4 }, Boundary { message: 1, tokens: 2 }];
    let want = boundary_recipes(
        root(),
        &rendering(vec![1, 2, 3, 4, 5, 6], &[(1, 2), (0, 3), (1, 4), (2, 6)]),
    )
    .unwrap();
    assert_eq!(recipes_with_cuts(root(), &plain, &cuts).unwrap(), want);
    assert_eq!(
        recipes_with_cuts(root(), &plain, &[]).unwrap(),
        boundary_recipes(root(), &plain).unwrap()
    );
    for tokens in [0, 3, 6, 7] {
        let bad = [Boundary { message: 1, tokens }];
        assert!(
            matches!(recipes_with_cuts(root(), &plain, &bad), Err(FamilyError::Internal(_))),
            "a cut at token {tokens}"
        );
    }
}

/// A media span of an image, with the content id of 64 copies of the hex digit `digit`.
fn image(first: usize, count: usize, digit: char) -> MediaSpan {
    MediaSpan {
        first,
        count,
        mime: "image/png".to_owned(),
        id: format!("blake3:{}", digit.to_string().repeat(64)),
    }
}

/// The tokens 1 to 3, an image of 3 positions (the placeholder 9), then 4 to 6, with boundaries after 3 and at the end.
fn with_image(digit: char) -> Rendering {
    let mut r = rendering(vec![1, 2, 3, 9, 9, 9, 4, 5, 6], &[(0, 3), (1, 9)]);
    r.media = vec![image(3, 3, digit)];
    r
}

/// A media span enters the recipe of the range that holds its first position, as tag 9 (spec 11.11, "Media through
/// the council"). The range before it keeps the recipe of its tokens.
#[test]
fn a_media_span_enters_the_recipe_of_the_range_that_holds_its_first_position() {
    let r = with_image('a');
    let recipes = boundary_recipes(root(), &r).unwrap();
    assert_eq!(recipes[0].recipe, recipe_boundary(&root(), &[1, 2, 3], 3));
    let id = ContentId::from_hex(&"a".repeat(64)).unwrap();
    let media = [RecipeMedia {
        mime: "image/png",
        id: &id,
        positions: 3,
    }];
    assert_eq!(
        recipes[1].recipe,
        recipe_boundary_media(&recipes[0].recipe, &[9, 9, 9, 4, 5, 6], 9, &media)
    );
    // Two images of one size give two recipes.
    let other = boundary_recipes(root(), &with_image('b')).unwrap();
    assert_eq!(other[0], recipes[0]);
    assert_ne!(other[1], recipes[1]);
    // The range recipe gives the same recipe over the same range.
    assert_eq!(
        range_recipe(&recipes[0].recipe, &r.tokens, &r.media, 3, 9).unwrap(),
        recipes[1].recipe
    );
    // A range that starts inside the span does not hold its first position, so the span is not in it.
    assert_eq!(
        range_recipe(&recipes[0].recipe, &r.tokens, &r.media, 4, 9).unwrap(),
        recipe_boundary(&recipes[0].recipe, &[9, 9, 4, 5, 6], 9)
    );
    // A range that ends inside the span holds its first position, so the span is in it.
    assert_eq!(
        range_recipe(&recipes[0].recipe, &r.tokens, &r.media, 3, 4).unwrap(),
        recipe_boundary_media(&recipes[0].recipe, &[9], 4, &media)
    );
}

/// A span that is not inside the tokens, spans out of order, and a span with a bad content id are defects of the
/// family library.
#[test]
fn a_bad_media_span_is_an_internal_error() {
    let cases = [
        vec![image(7, 3, 'a')],
        vec![image(3, 0, 'a')],
        vec![image(3, 2, 'a'), image(4, 1, 'b')],
        vec![MediaSpan {
            id: "blake3:short".to_owned(),
            ..image(3, 3, 'a')
        }],
    ];
    for media in cases {
        let mut r = with_image('a');
        r.media = media.clone();
        let error = boundary_recipes(root(), &r).unwrap_err();
        assert!(matches!(error, FamilyError::Internal(_)), "{media:?}: {error:?}");
    }
}

/// The seals of a state give its recipes (spec 6.14, "The decoded head"). A boundary has the recipe of its span. A
/// decoded head has the recipe of its prompt, then its decoded tokens with the item "decoded". A later seal chains
/// from the head.
#[test]
fn the_seals_of_a_state_give_the_recipe_of_a_decoded_head_and_of_each_seal_after_it() {
    use nagaya_council::build::{Seal, seal_recipes};
    use nagaya_store::recipe_decoded;
    let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let rule = |tokens| Seal { tokens, prompt: None };
    let seals = [
        rule(3),
        Seal {
            tokens: 7,
            prompt: Some(5),
        },
        rule(10),
    ];
    let got = seal_recipes(root(), &tokens, &[], &seals).unwrap().unwrap();
    let boundary = recipe_boundary(&root(), &[1, 2, 3], 3);
    let prompt = recipe_boundary(&boundary, &[4, 5], 5);
    let head = recipe_decoded(&prompt, &[6, 7], 7);
    assert_eq!(got, vec![boundary, head, recipe_boundary(&head, &[8, 9, 10], 10)]);
    // The same tokens with no decoded head have the recipes of the boundaries.
    let plain = seal_recipes(root(), &tokens, &[], &[rule(3), rule(7), rule(10)])
        .unwrap()
        .unwrap();
    let by_rule = boundary_recipes(root(), &rendering(tokens.to_vec(), &[(0, 3), (1, 7), (2, 10)])).unwrap();
    assert_eq!(plain, by_rule.iter().map(|recipe| recipe.recipe).collect::<Vec<_>>());
    assert_ne!(got[1], plain[1], "a decoded head is not the boundary at its position");
    // Another split between the prompt and the decoded tokens is another recipe.
    let other = [
        rule(3),
        Seal {
            tokens: 7,
            prompt: Some(6),
        },
    ];
    assert_ne!(seal_recipes(root(), &tokens, &[], &other).unwrap().unwrap()[1], head);
    // A head with no decoded token is the head of a cut reply of one token. It is not the recipe of its prompt.
    let empty = [Seal {
        tokens: 5,
        prompt: Some(5),
    }];
    let prompt = recipe_boundary(&root(), &[1, 2, 3, 4, 5], 5);
    assert_eq!(
        seal_recipes(root(), &tokens, &[], &empty).unwrap().unwrap(),
        vec![recipe_decoded(&prompt, &[], 5)]
    );
}

/// A media span that starts in the prompt of a decoded head enters the recipe of the prompt (spec 11.11).
#[test]
fn a_media_span_in_the_prompt_of_a_decoded_head_enters_its_recipe() {
    use nagaya_council::build::{Seal, seal_recipes};
    let tokens = [1u32, 2, 3, 4, 5, 6];
    let seals = [Seal {
        tokens: 6,
        prompt: Some(4),
    }];
    let plain = seal_recipes(root(), &tokens, &[], &seals).unwrap().unwrap();
    let with_image = seal_recipes(root(), &tokens, &[image(1, 2, 'a')], &seals)
        .unwrap()
        .unwrap();
    assert_ne!(plain, with_image);
    let other_image = seal_recipes(root(), &tokens, &[image(1, 2, 'b')], &seals)
        .unwrap()
        .unwrap();
    assert_ne!(with_image, other_image);
}

/// Seals that do not fit the tokens give no recipe: the note of a head came from another rendering.
#[test]
fn seals_that_do_not_fit_the_tokens_give_no_recipe() {
    use nagaya_council::build::{Seal, seal_recipes};
    let tokens = [1u32, 2, 3, 4, 5, 6];
    let rule = |tokens| Seal { tokens, prompt: None };
    let decoded = |prompt, tokens| Seal {
        tokens,
        prompt: Some(prompt),
    };
    for (why, seals) in [
        ("a seal past the tokens", vec![rule(7)]),
        ("a seal at 0", vec![rule(0)]),
        ("seals that do not increase", vec![rule(3), rule(3)]),
        ("a prompt at the seal before it", vec![rule(3), decoded(3, 5)]),
        ("a prompt after its head", vec![decoded(5, 4)]),
    ] {
        assert_eq!(seal_recipes(root(), &tokens, &[], &seals).unwrap(), None, "{why}");
    }
    assert_eq!(seal_recipes(root(), &tokens, &[], &[]).unwrap(), Some(Vec::new()));
}
