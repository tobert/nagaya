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

/// The slab of a layer holds the codes of each expert, then the scales of each expert, then the factors.
#[test]
fn a_slab_holds_the_planes_of_each_expert_then_the_factors() {
    let (experts, rows, cols) = (3, 32, 64);
    let (row_bytes, row_scales) = (cols / 2, cols / 16);
    let values = source(experts * rows, row_bytes);
    let scales = source(experts * rows, row_scales);
    let factors = [0.5f32, 0.25, 2.0];
    let slab = expert_slab(&values, &scales, Some(&factors), experts, rows, cols, false, true).unwrap();
    let words = rows * row_bytes / 4;
    let scale_bytes = rows * row_scales;
    assert_eq!(slab.len(), experts * (words * 4 + scale_bytes) + experts * 4);
    for e in 0..experts {
        let order = gate_up_tile_order(rows);
        let one = |src: &[u8], width: usize| -> Vec<u8> {
            let rows_of_e = &src[e * rows * width..(e + 1) * rows * width];
            order
                .iter()
                .flat_map(|&r| rows_of_e[r * width..(r + 1) * width].to_vec())
                .collect()
        };
        let codes: Vec<u8> = repack_codes(&one(&values, row_bytes), rows, row_bytes, 8)
            .unwrap()
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        assert_eq!(
            &slab[e * words * 4..(e + 1) * words * 4],
            &codes[..],
            "codes of expert {e}"
        );
        let sc = repack_scales(&one(&scales, row_scales), rows, row_scales).unwrap();
        let at = experts * words * 4 + e * scale_bytes;
        assert_eq!(&slab[at..at + scale_bytes], &sc[..], "scales of expert {e}");
        let f = experts * (words * 4 + scale_bytes) + 4 * e;
        assert_eq!(
            f32::from_le_bytes(slab[f..f + 4].try_into().unwrap()),
            factors[e],
            "factor of expert {e}"
        );
    }
}

#[test]
fn an_mxfp8_slab_has_no_factors_and_one_scale_for_32_columns() {
    let (experts, rows, cols) = (2, 16, 64);
    let values = source(experts * rows, cols);
    let scales = source(experts * rows, cols / 32);
    let slab = expert_slab(&values, &scales, None, experts, rows, cols, true, false).unwrap();
    assert_eq!(slab.len(), experts * (rows * cols + rows * cols / 32));
}
