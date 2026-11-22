use super::*;
use nagaya_text_inkling2::media::ImageRows;

const P: u32 = IMAGE_TOKEN;

/// The record that the family writes for an image of `grid` at `first`.
fn record(grid: Grid, first: usize) -> Vec<u8> {
    ImageRows {
        grid,
        values: Vec::new(),
    }
    .record(first)
}

fn span(rows: usize, columns: usize, first: u64) -> Span {
    Span::image(Grid { rows, columns }, first)
}

/// The family's record of an image is the span of the engine: one reader serves both (engine-abi "The meta of a
/// span").
#[test]
fn the_record_of_the_family_is_the_span_of_the_engine() {
    let mut meta = record(Grid { rows: 3, columns: 6 }, 12);
    meta.extend(record(Grid { rows: 12, columns: 16 }, 33));
    let spans = decode(&meta).unwrap();
    assert_eq!(spans, [span(3, 6, 12), span(12, 16, 33)]);
    assert_eq!((spans[0].count(), spans[0].positions()), (18, 12..30));
    // 4 patches of 1,200 f32 values for each token.
    assert_eq!(spans[1].bytes(), 192 * 4 * 1200 * 4);
    // The largest image of the size rule, 103 by 103 tokens.
    let largest = record(
        Grid {
            rows: 103,
            columns: 103,
        },
        0,
    );
    assert_eq!(decode(&largest).unwrap()[0].count(), 10_609);
}

#[test]
fn a_record_that_breaks_the_rule_of_an_image_is_refused_by_name() {
    let good = record(Grid { rows: 2, columns: 3 }, 5);
    let bad = |edit: &dyn Fn(&mut Vec<u8>), want: &str| {
        let mut meta = good.clone();
        meta.extend(good.clone());
        edit(&mut meta);
        let e = decode(&meta).unwrap_err();
        assert!(e.contains(want), "{want}: {e}");
    };
    // The second record is record 1.
    bad(&|m| m.truncate(40), "40 bytes, not whole records");
    bad(&|m| m[32] = 2, "record 1 has the kind 2");
    bad(&|m| m[48] = 2, "grid_t 2");
    bad(&|m| m[60] = 1, "fourth word of 1");
    bad(&|m| m[52] = 5, "5 by 6 patches");
    bad(&|m| m[56] = 0, "4 by 0 patches");
    bad(
        &|m| m[52..56].copy_from_slice(&208u32.to_le_bytes()),
        "208 by 6 patches",
    );
    bad(&|m| m[36] = 7, "with 7 positions; it has 6");
    bad(
        &|m| m[40..48].copy_from_slice(&u64::MAX.to_le_bytes()),
        "past the last position",
    );
    // 206 patches is the side of 4,096 px.
    let mut side = good.clone();
    side[20..24].copy_from_slice(&206u32.to_le_bytes());
    side[4..8].copy_from_slice(&309u32.to_le_bytes());
    assert_eq!(
        decode(&side).unwrap()[0].grid().unwrap(),
        Grid { rows: 103, columns: 3 }
    );
}

/// A span that crosses a block boundary: each block takes its own positions, and each gets the rows of the whole span.
#[test]
fn each_block_of_a_span_takes_its_positions_and_the_rows_of_the_whole_span() {
    let image = span(2, 3, 2);
    let rows = image.bytes();
    // Block 0..4: two text tokens, then the first two positions of the span.
    let uses = check_spans(&[image], &[7, 7, P, P], 0, rows).unwrap();
    assert_eq!(
        uses,
        [SpanUse {
            span: image,
            rows_at: 0,
            take: 0..2,
            block_row: 2
        }]
    );
    // Block 4..10: the last four positions of the span, then text.
    let uses = check_spans(&[image], &[P, P, P, P, 7, 7], 4, rows).unwrap();
    assert_eq!((uses[0].take.clone(), uses[0].block_row), (2..6, 0));
    // Two spans in one block: the rows of the second start after the rows of the first.
    let second = span(1, 1, 9);
    let uses = check_spans(&[image, second], &[P, P, P, P, 7, P], 4, rows + second.bytes()).unwrap();
    assert_eq!(
        (uses[1].rows_at, uses[1].take.clone(), uses[1].block_row),
        (rows, 0..1, 5)
    );
}

#[test]
fn a_block_whose_spans_do_not_fit_its_tokens_and_rows_is_refused() {
    let image = span(1, 2, 1);
    let rows = image.bytes();
    let refused = |spans: &[Span], tokens: &[u32], pos0: u64, rows: usize, want: &str| {
        let e = check_spans(spans, tokens, pos0, rows).unwrap_err();
        assert!(e.contains(want), "{want}: {e}");
    };
    refused(&[image], &[7, P, 7], 0, rows, "position 2 is in media record 0");
    refused(&[image], &[7, P, P], 0, rows - 4, "need 38400");
    refused(&[image], &[7, P, P], 0, rows + 4, "has 38404 bytes");
    refused(&[image], &[7, 7], 3, rows, "a block names only the spans that it reads");
    refused(
        &[image, span(1, 1, 2)],
        &[7, P, P],
        0,
        2 * rows,
        "starts at 2 inside the span before it",
    );
    // A placeholder that no record names, with a span and with none.
    refused(
        &[image],
        &[7, P, P, P],
        0,
        rows,
        "position 3 holds the image placeholder",
    );
    refused(&[], &[7, P], 10, 0, "position 11 holds the image placeholder 200054");
    // A text block has no span and no rows.
    assert_eq!(check_spans(&[], &[7, 8, 9], 0, 0), Ok(Vec::new()));
}

#[test]
fn the_held_rows_run_the_tower_once_for_the_blocks_of_one_image() {
    let mut held = HeldRows::default();
    let g = Grid { rows: 2, columns: 2 };
    let (a, b) = (vec![1u8; 64], vec![2u8; 64]);
    assert!(!held.holds(g, blake3::hash(&a)), "nothing is held at the start");
    assert!(held.holds(g, blake3::hash(&a)), "the second block of the image");
    assert_eq!(held.runs(), 1);
    assert!(!held.holds(g, blake3::hash(&b)), "other patch rows are another image");
    assert!(
        !held.holds(Grid { rows: 1, columns: 4 }, blake3::hash(&b)),
        "another grid is another image"
    );
    assert!(!held.holds(g, blake3::hash(&a)), "the first image is held no more");
    assert_eq!(held.runs(), 4);
    held.forget();
    assert!(!held.holds(g, blake3::hash(&a)), "after a forget the tower runs again");
    assert_eq!(held.runs(), 5);
}

/// The towers of a pass follow the held rows as `holds` does, and they change nothing. Falsified by `towers` that
/// skip the image that the pass itself held before.
#[test]
fn the_towers_of_a_pass_are_the_images_that_the_held_rows_lack() {
    let mut held = HeldRows::default();
    let g = Grid { rows: 2, columns: 2 };
    let (a, b) = (vec![1u8; 64], vec![2u8; 64]);
    assert_eq!(
        held.towers([(g, blake3::hash(&a))]),
        [g],
        "nothing is held at the start"
    );
    assert_eq!(held.towers([(g, blake3::hash(&a))]), [g], "towers changed nothing");
    assert!(!held.holds(g, blake3::hash(&a)));
    assert_eq!(
        held.towers([(g, blake3::hash(&a))]),
        Vec::<Grid>::new(),
        "a later block of the image"
    );
    assert_eq!(held.towers([(g, blake3::hash(&b))]), [g], "other rows of the same grid");
    let other = Grid { rows: 1, columns: 4 };
    assert_eq!(held.towers([(other, blake3::hash(&a))]), [other], "another grid");
    // Image b replaces a in the held rows, so a runs again after it.
    assert_eq!(
        held.towers([(g, blake3::hash(&a)), (g, blake3::hash(&b)), (g, blake3::hash(&a))]),
        [g, g]
    );
    assert_eq!(held.runs(), 1);
}

/// The bound and the pass of one segment hash each image once: `prepare` hashes, and the next `take` of the same rows
/// reuses the hashes. A `take` with no `prepare`, or of other rows, hashes again, and a `take` keeps nothing.
/// Falsified by a `take` that ignores the prepared hashes, and by a `take` that keeps them.
#[test]
fn the_bound_and_the_pass_hash_each_image_once() {
    let hashes = RowHashes::default();
    let (a, b) = (vec![1u8; 64], vec![2u8; 64]);
    let want = vec![blake3::hash(&a), blake3::hash(&b)];
    assert_eq!(hashes.prepare(&[&a, &b]), want);
    assert_eq!(hashes.hashed(), 2);
    assert_eq!(hashes.take(&[&a, &b]), want);
    assert_eq!(hashes.hashed(), 2, "the pass reuses the hashes of the bound");
    assert_eq!(hashes.take(&[&a, &b]), want);
    assert_eq!(
        hashes.hashed(),
        4,
        "a take keeps nothing, so a second pass hashes again"
    );
    hashes.prepare(&[&a]);
    assert_eq!(hashes.take(&[&b]), vec![blake3::hash(&b)], "other rows hash again");
    assert_eq!(hashes.hashed(), 6);
    assert_eq!(
        hashes.take(&[&a]),
        vec![blake3::hash(&a)],
        "the take of b dropped the prepared a"
    );
    assert_eq!(hashes.hashed(), 7);
}

/// The record of `count` latent rows of `width` values from `first` (engine-abi "The lab of Inkling 2", "Latent
/// rows").
fn latent_record(first: u64, count: u32, width: u32) -> Vec<u8> {
    let mut r = Vec::with_capacity(RECORD_BYTES);
    r.extend(LATENT_KIND.to_le_bytes());
    r.extend(count.to_le_bytes());
    r.extend(first.to_le_bytes());
    for word in [width, 0, 0, 0] {
        r.extend(word.to_le_bytes());
    }
    r
}

/// A latent record names the width of its rows and its positions, and a span takes one row of 2,304 `f32` values for
/// each position. Another width, no position, or another word is refused with the record.
#[test]
fn a_latent_record_is_a_span_of_rows_of_the_width_of_the_residual() {
    let mut meta = latent_record(30, 2, 2304);
    meta.extend(record(Grid { rows: 1, columns: 2 }, 40));
    let spans = decode(&meta).unwrap();
    assert_eq!(spans, [Span::latent(2, 30), span(1, 2, 40)]);
    assert_eq!(
        (spans[0].count(), spans[0].bytes(), spans[0].grid()),
        (2, 2 * 2304 * 4, None)
    );
    assert_eq!(spans[0].placeholder(), 199_999);
    for (bad, want) in [
        (latent_record(0, 1, 2048), "latent span with the words [2048, 0, 0, 0]"),
        (latent_record(0, 0, 2304), "latent span of 0 positions"),
        (latent_record(u64::MAX, 1, 2304), "before the last position"),
        (
            {
                let mut m = latent_record(0, 1, 2304);
                m[24] = 1;
                m
            },
            "words [2304, 0, 1, 0]",
        ),
    ] {
        let e = decode(&bad).unwrap_err();
        assert!(e.contains(want), "{want}: {e}");
    }
}

/// Each position of a latent span holds `<|endoftext|>`, and the rows of the spans follow the order of the records.
/// The placeholder is a token of the vocabulary, so it can be outside a span.
#[test]
fn a_latent_span_takes_its_rows_at_positions_of_its_placeholder() {
    const L: u32 = LATENT_TOKEN;
    let image = span(1, 1, 1);
    let latent = Span::latent(2, 3);
    let rows = image.bytes() + latent.bytes();
    let uses = check_spans(&[image, latent], &[7, P, L, L, L], 0, rows).unwrap();
    assert_eq!(
        (uses[1].rows_at, uses[1].take.clone(), uses[1].block_row),
        (image.bytes(), 0..2, 3)
    );
    // A later block takes the second row of the span, and the rows are the whole span.
    let uses = check_spans(&[latent], &[L, 7], 4, latent.bytes()).unwrap();
    assert_eq!((uses[0].take.clone(), uses[0].block_row), (1..2, 0));
    let e = check_spans(&[latent], &[7, 7, 7, P, L], 0, latent.bytes()).unwrap_err();
    assert!(
        e.contains("position 3 is in media record 0, and it holds the token 200054, not the placeholder 199999"),
        "{e}"
    );
    let e = check_spans(&[latent], &[7, 7, 7, L, L], 0, latent.bytes() - 4).unwrap_err();
    assert!(e.contains("need 18432"), "{e}");
}
