//! The repack of the expert planes for the card: each word goes to the lane and the step that read it.

use nagaya_mk_inkling2_bmg::repack::*;

/// A row-major matrix of `rows` rows of `row_bytes` bytes, where each byte names its row and column.
fn source(rows: usize, row_bytes: usize) -> Vec<u8> {
    (0..rows * row_bytes)
        .map(|i| ((i / row_bytes) * 7 + (i % row_bytes) * 13) as u8)
        .collect()
}

#[test]
fn a_lane_reads_the_words_of_its_own_row_in_each_step() {
    let (rows, row_bytes, block) = (32, 24, 8);
    let src = source(rows, row_bytes);
    let planes = repack_codes(&src, rows, row_bytes, block).unwrap();
    let words = block / 4;
    assert_eq!(planes.len(), rows * row_bytes / 4);
    for row in 0..rows {
        let (tile, lane) = (row / 16, row % 16);
        for step in 0..row_bytes / block {
            for w in 0..words {
                let at = ((tile * (row_bytes / block) + step) * words + w) * 16 + lane;
                let start = row * row_bytes + step * block + 4 * w;
                let want = u32::from_le_bytes(src[start..start + 4].try_into().unwrap());
                assert_eq!(planes[at], want, "row {row}, step {step}, word {w}");
            }
        }
    }
}

#[test]
fn a_lane_reads_the_scales_of_its_own_row() {
    let (rows, scales) = (16, 9);
    let src = source(rows, scales);
    let planes = repack_scales(&src, rows, scales).unwrap();
    for row in 0..rows {
        for s in 0..scales {
            assert_eq!(planes[s * 16 + row], src[row * scales + s], "row {row}, scale {s}");
        }
    }
}

/// The 16 lanes of tile `2t` hold the gate rows `16t..16t+16`, and tile `2t + 1` holds the up rows of the same `i`.
#[test]
fn the_tiles_of_w13_pair_the_gate_and_the_up_row_of_each_lane() {
    let order = gate_up_tile_order(64);
    assert_eq!(order.len(), 64);
    for i in 0..32 {
        let (t, lane) = (i / 16, i % 16);
        assert_eq!(order[(2 * t) * 16 + lane], 2 * i, "gate row {i}");
        assert_eq!(order[(2 * t + 1) * 16 + lane], 2 * i + 1, "up row {i}");
    }
}

#[test]
fn a_shape_that_is_not_whole_tiles_is_an_error() {
    assert!(repack_codes(&[0; 15 * 8], 15, 8, 8).is_err(), "15 rows are not a tile");
    assert!(
        repack_codes(&[0; 16 * 12], 16, 12, 8).is_err(),
        "12 bytes are not whole steps of 8"
    );
    assert!(
        repack_codes(&[0; 16 * 8], 16, 8, 6).is_err(),
        "a step of 6 bytes is not whole words"
    );
    assert!(repack_codes(&[0; 16 * 8 - 1], 16, 8, 8).is_err(), "the source is short");
    assert!(repack_scales(&[0; 16 * 3 - 1], 16, 3).is_err(), "the scales are short");
}
