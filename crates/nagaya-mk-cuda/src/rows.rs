//! The weights of the message gates in the order that `OP_GATE_DOWN` and `OP_GATE_UP` read them (milestone 37, wave B
//! lane R).
//!
//! The down matrix of a gate is `[64, 4608]`. It is a dense matmul, and `repack::repack_bf16` lays it out with
//! `MatmulShape::new(64, 4608)`: 4 tiles of 16 columns and 6 K slices of 768. [`gate_down_weights`] joins the
//! weights of several gates, one after the other, and then item `gate * 24 + tile * 6 + slice` is one span.
//!
//! The forget and gain matrices are `[2304, 64]` each. K is 64, so an item has no K split: it is 8 warps of 16 rows,
//! and each warp runs the 4 steps of K. [`repack_gate_up`] puts the forget row of the channel `c` and its gain row in
//! one MMA tile: the row `g` of the tile of a warp is the forget row of the channel `8 w + g`, and the row `g + 8` is
//! the gain row of the same channel. A lane then holds the forget and the gain of one channel in one MMA result,
//! and the blend needs no shuffle.

use crate::repack::{MatmulShape, repack_bf16};

/// The width of a row of the residual stream.
pub const HIDDEN: usize = 2304;

/// The rank of a message gate.
pub const GATE_RANK: usize = 64;

/// The channels of one item of `OP_GATE_UP`: 8 warps of 8 channels.
pub const GATE_UP_CHANNELS: usize = 64;

/// The items of one gate in `OP_GATE_UP`.
pub const GATE_UP_TILES: usize = HIDDEN / GATE_UP_CHANNELS;

/// The bytes of one item of `OP_GATE_UP`: 8 warps, 4 steps, 32 lanes of 16 bytes.
pub const GATE_UP_ITEM_BYTES: usize = 8 * 4 * 512;

/// The items of one gate in `OP_GATE_DOWN`: 4 tiles of 16 columns and 6 K slices.
pub const GATE_DOWN_ITEMS: usize = 24;

/// The shape of the down matmul of a gate.
pub fn gate_down_shape() -> MatmulShape {
    MatmulShape::new(GATE_RANK, 2 * HIDDEN).expect("the shape of the down matrix is in KT_TABLE")
}

/// The repacked down weights of several gates, one after the other. Each `down` is the bytes of `[64, 4608]`.
pub fn gate_down_weights(down: &[&[u8]]) -> Result<Vec<u8>, String> {
    let shape = gate_down_shape();
    let mut out = Vec::with_capacity(down.len() * shape.packed_bytes());
    for weight in down {
        out.extend_from_slice(&repack_bf16(weight, &shape)?);
    }
    Ok(out)
}

/// The byte offset in the packed weight of one gate of the 32-bit word that holds `forget[c][kk]` (`gain` false) or
/// `gain[c][kk]` (`gain` true), with `kk` even.
pub fn gate_up_word_offset(channel: usize, kk: usize, gain: bool) -> usize {
    debug_assert!(channel < HIDDEN && kk < GATE_RANK && kk.is_multiple_of(2));
    let (tile, in_tile) = (channel / GATE_UP_CHANNELS, channel % GATE_UP_CHANNELS);
    let (warp, g) = (in_tile / 8, in_tile % 8);
    let (step, in_step) = (kk / 16, kk % 16);
    let (t, part) = ((in_step / 2) % 4, in_step / 8);
    let lane = g * 4 + t;
    tile * GATE_UP_ITEM_BYTES + ((warp * 4 + step) * 32 + lane) * 16 + (part * 2 + usize::from(gain)) * 4
}

/// Repack the forget and gain matrices of one gate (`bf16` bytes of `[2304, 64]` each) for `OP_GATE_UP`. The result has
/// `GATE_UP_TILES * GATE_UP_ITEM_BYTES` bytes, and the weights of several gates go one after the other.
pub fn repack_gate_up(forget: &[u8], gain: &[u8]) -> Result<Vec<u8>, String> {
    let want = HIDDEN * GATE_RANK * 2;
    for (name, bytes) in [("forget", forget), ("gain", gain)] {
        if bytes.len() != want {
            return Err(format!(
                "the {name} matrix has {} bytes, and a bf16 matrix of {HIDDEN} x {GATE_RANK} needs {want}. \
                 Check the shape of the tensor",
                bytes.len()
            ));
        }
    }
    let mut out = vec![0u8; GATE_UP_TILES * GATE_UP_ITEM_BYTES];
    for channel in 0..HIDDEN {
        for kk in (0..GATE_RANK).step_by(2) {
            let src = (channel * GATE_RANK + kk) * 2;
            for (gain_flag, bytes) in [(false, forget), (true, gain)] {
                let dst = gate_up_word_offset(channel, kk, gain_flag);
                out[dst..dst + 4].copy_from_slice(&bytes[src..src + 4]);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(seed: u32) -> Vec<u8> {
        (0..HIDDEN * GATE_RANK)
            .flat_map(|i| {
                let h = (i as u32 ^ seed).wrapping_mul(0x9E37_79B1);
                ((h >> 13) as u16).to_le_bytes()
            })
            .collect()
    }

    #[test]
    fn every_word_of_forget_and_gain_lands_once_and_a_lane_holds_the_a_fragment() {
        let (forget, gain) = (matrix(1), matrix(2));
        let packed = repack_gate_up(&forget, &gain).unwrap();
        assert_eq!(packed.len(), 36 * 16384);
        assert_eq!(packed.len(), 2 * forget.len(), "a repack moves values and adds none");
        // The inverse: each offset of the packed bytes is the word of exactly one source word.
        let mut seen = vec![false; packed.len() / 4];
        for channel in 0..HIDDEN {
            for kk in (0..GATE_RANK).step_by(2) {
                for gain_flag in [false, true] {
                    let at = gate_up_word_offset(channel, kk, gain_flag);
                    assert!(!std::mem::replace(&mut seen[at / 4], true), "offset {at} is used twice");
                    let src = (channel * GATE_RANK + kk) * 2;
                    let from = if gain_flag { &gain } else { &forget };
                    assert_eq!(&packed[at..at + 4], &from[src..src + 4]);
                }
            }
        }
        assert!(seen.iter().all(|s| *s));
        // The fragment: lane (g, t) of warp w at step s holds a0 = forget[8w+g], a1 = gain[8w+g] at K = 16 s + 2 t,
        // and a2, a3 the same rows at K + 8. Check one lane against the words of the matrices.
        let (tile, warp, step, g, t) = (5usize, 3usize, 2usize, 6usize, 1usize);
        let lane = g * 4 + t;
        let at = tile * GATE_UP_ITEM_BYTES + ((warp * 4 + step) * 32 + lane) * 16;
        let channel = tile * 64 + warp * 8 + g;
        let word = |m: &[u8], k: usize| m[(channel * GATE_RANK + k) * 2..(channel * GATE_RANK + k) * 2 + 4].to_vec();
        let k0 = step * 16 + 2 * t;
        assert_eq!(packed[at..at + 4], word(&forget, k0)[..]);
        assert_eq!(packed[at + 4..at + 8], word(&gain, k0)[..]);
        assert_eq!(packed[at + 8..at + 12], word(&forget, k0 + 8)[..]);
        assert_eq!(packed[at + 12..at + 16], word(&gain, k0 + 8)[..]);
    }

    #[test]
    fn a_matrix_of_the_wrong_size_is_an_error_with_the_size() {
        let error = repack_gate_up(&[0; 10], &matrix(2)).unwrap_err();
        assert!(error.contains("forget") && error.contains("294912"), "{error}");
    }

    #[test]
    fn the_down_weights_of_gates_follow_one_another() {
        let down = vec![0u8; 64 * 4608 * 2];
        let joined = gate_down_weights(&[&down, &down, &down]).unwrap();
        assert_eq!(joined.len(), 3 * gate_down_shape().packed_bytes());
        assert_eq!(gate_down_shape().items(), GATE_DOWN_ITEMS);
    }
}
