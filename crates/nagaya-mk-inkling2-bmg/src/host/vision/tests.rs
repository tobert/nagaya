//! The rules of the processor and the small parts of the tower, with no model file.

use super::*;

#[test]
fn the_grid_counts_tokens_of_40_px_and_refuses_sizes_outside_the_rules() {
    assert_eq!(grid(213, 118), Ok(Grid { rows: 3, columns: 6 }));
    assert_eq!(grid(637, 479), Ok(Grid { rows: 12, columns: 16 }));
    assert_eq!(grid(40, 40), Ok(Grid { rows: 1, columns: 1 }));
    assert_eq!(grid(41, 1), Ok(Grid { rows: 1, columns: 2 }));
    assert_eq!(grid(1, 1), Ok(Grid { rows: 1, columns: 1 }));
    assert_eq!(grid(920, 880).map(Grid::tokens), Ok(506));
    assert_eq!(grid(4096, 40).map(Grid::tokens), Ok(103));
    for (w, h) in [(0, 10), (10, 0), (4097, 40), (40, 4097)] {
        let e = grid(w, h).unwrap_err();
        assert!(e.contains("1 to 4096 px"), "{e}");
    }
    let e = grid(1000, 1000).unwrap_err();
    assert!(e.contains("625 image tokens") && e.contains("512"), "{e}");
    let g = Grid { rows: 3, columns: 6 };
    assert_eq!((g.tokens(), g.patches(), g.meta_grid()), (18, 72, [1, 6, 12]));
}

#[test]
fn the_pixel_values_normalize_each_byte_and_pad_on_the_right_and_at_the_bottom() {
    // A 41 x 1 image: two tokens. Pixel 40 of line 0 is the first pixel of token 1.
    let mut rgb = vec![0u8; 41 * 3];
    rgb[..3].copy_from_slice(&[255, 0, 127]);
    rgb[40 * 3..].copy_from_slice(&[10, 20, 30]);
    let (g, px) = pixel_values(&rgb, 41, 1).unwrap();
    assert_eq!(g, Grid { rows: 1, columns: 2 });
    let square = TOKEN_PX * TOKEN_PX * 3;
    assert_eq!(px.len(), 2 * square);
    assert_eq!(&px[..3], &[1.0, -1.0, (127.0 - 127.5) / 127.5]);
    assert_eq!(&px[3..6], &[-1.0, -1.0, -1.0]);
    let pad = (-1.0f32 - 127.5) / 127.5;
    // Token 0, line 1 is pad, and token 1 holds pixel 40 and then pad.
    assert_eq!(px[3 * TOKEN_PX], pad);
    assert_eq!(
        &px[square..square + 3],
        &[(10.0 - 127.5) / 127.5, (20.0 - 127.5) / 127.5, (30.0 - 127.5) / 127.5]
    );
    assert!(px[square + 3..].iter().all(|v| *v == pad));
    assert!(pixel_values(&rgb, 0, 1).is_err());
}

#[test]
fn the_patch_rows_follow_the_merge_order() {
    let g = Grid { rows: 1, columns: 2 };
    let px: Vec<f32> = (0..g.tokens() * TOKEN_PX * TOKEN_PX * 3).map(|i| i as f32).collect();
    let rows = patch_rows(&px, g);
    assert_eq!(rows.len(), g.patches() * PATCH_VALUES);
    // The value of pixel (y, x), channel c, of token t in the processor output.
    let at = |t: usize, y: usize, x: usize, c: usize| (t * TOKEN_PX * TOKEN_PX * 3 + 3 * (y * TOKEN_PX + x) + c) as f32;
    let row = |n: usize| &rows[n * PATCH_VALUES..(n + 1) * PATCH_VALUES];
    // Patch 1 is the right half of the top of token 0, and patch 2 the left half of its bottom.
    assert_eq!(
        row(0)[..4],
        [at(0, 0, 0, 0), at(0, 0, 0, 1), at(0, 0, 0, 2), at(0, 0, 1, 0)]
    );
    assert_eq!(row(0)[3 * PATCH_PX], at(0, 1, 0, 0));
    assert_eq!(row(1)[0], at(0, 0, 20, 0));
    assert_eq!(row(2)[0], at(0, 20, 0, 0));
    assert_eq!(row(3)[PATCH_VALUES - 1], at(0, 39, 39, 2));
    assert_eq!(row(4)[0], at(1, 0, 0, 0));
}

#[test]
fn a_patch_position_reads_the_merge_blocks_line_by_line() {
    let g = Grid { rows: 2, columns: 3 };
    let got: Vec<(usize, usize)> = [0, 1, 2, 3, 4, 7, 12, 23]
        .iter()
        .map(|n| patch_position(*n, g))
        .collect();
    assert_eq!(got, [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 3), (2, 0), (3, 5)]);
}

#[test]
fn the_axis_taps_are_the_identity_on_a_grid_of_32_and_their_weights_sum_to_one() {
    for i in 0..GRID {
        let [(t0, w0), (_t1, w1)] = axis_taps(i, GRID);
        assert_eq!((t0, w0, w1), (i, 1.0, 0.0));
    }
    for size in [1, 2, 6, 24, 33, 206] {
        for i in 0..size {
            let [(t0, w0), (t1, w1)] = axis_taps(i, size);
            assert!(t0 < GRID && t1 < GRID && t1 >= t0);
            assert!((w0 + w1 - 1.0).abs() < 1e-6, "size {size} index {i}: {w0} + {w1}");
        }
    }
    // Half-pixel centres: on an axis of 64 patches, patch 0 sits a quarter before entry 0, so the clamp gives 0 twice.
    assert_eq!(axis_taps(0, 64), [(0, 0.25), (0, 0.75)]);
}

#[test]
fn the_rope_turns_the_first_half_by_the_column_and_the_second_by_the_line() {
    assert_eq!(rope_angle(0, 5, 7), 7.0);
    assert_eq!(rope_angle(16, 5, 7), 5.0);
    assert_eq!(rope_angle(1, 5, 7), 7.0 * (1.0 / ROPE_BASE.powf(2.0 / 32.0)));
    assert_eq!(rope_angle(31, 5, 0), 5.0 * (1.0 / ROPE_BASE.powf(30.0 / 32.0)));
}

#[test]
fn an_alpha_channel_goes_over_the_checkerboard() {
    let mut rgba = vec![0u8; 9 * 4];
    rgba[..4].copy_from_slice(&[10, 20, 30, 255]);
    let rgb = rgba_to_rgb(&rgba, 9, 1);
    assert_eq!(&rgb[..3], &[10, 20, 30]);
    assert_eq!(&rgb[3..6], &[204, 204, 204]);
    assert_eq!(&rgb[8 * 3..], &[153, 153, 153]);
    // Half alpha rounds to the nearest byte: (100 * 128 + 204 * 127 + 127) / 255 = 152.
    let half = rgba_to_rgb(&[100, 100, 100, 128], 1, 1);
    assert_eq!(half, [152, 152, 152]);
}

#[test]
fn the_rows_replace_the_placeholders_in_order_and_a_count_that_differs_is_refused() {
    let ids = [1, IMAGE_TOKEN, 2, IMAGE_TOKEN];
    let mut x = vec![0f32; 4 * HIDDEN];
    let rows: Vec<f32> = (0..2 * HIDDEN).map(|i| (i / HIDDEN + 1) as f32).collect();
    place_rows(&ids, &mut x, &rows).unwrap();
    assert!(x[..HIDDEN].iter().all(|v| *v == 0.0));
    assert!(x[HIDDEN..2 * HIDDEN].iter().all(|v| *v == 1.0));
    assert!(x[3 * HIDDEN..].iter().all(|v| *v == 2.0));
    let e = place_rows(&ids, &mut x, &rows[..HIDDEN]).unwrap_err();
    assert!(e.contains("2 image placeholders and 1 image rows"), "{e}");
}

#[test]
fn the_gelu_is_the_gelu_of_erf() {
    assert_eq!(gelu(0.0), 0.0);
    assert!((gelu(1.0) - 0.841_344_8).abs() < 1e-7);
    assert!((gelu(-3.5) - -3.5 * 2.326_291e-4).abs() < 1e-9);
}
