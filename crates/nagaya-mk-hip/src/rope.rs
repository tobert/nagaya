//! The host tables of the rotary positions and the timestep embedding of Qwen-Image 2.1 (spec 11.9c).
//!
//! The kernel `head_norm_rope` reads one (cos, sin) pair in f32 for each row and each pair of a head. These tables
//! compute the angles as the reference does: the inverse frequencies and the products in f32. The cos and the sin of
//! each f32 angle are exact to f32.

/// The inverse frequencies of `dim` dimensions: `1 / theta^(2i / dim)`, in f32 as torch computes them.
fn inverse_frequencies(dim: usize, theta: f32) -> Vec<f32> {
    (0..dim / 2)
        .map(|i| {
            let exponent = (2 * i) as f32 / dim as f32;
            1.0 / theta.powf(exponent)
        })
        .collect()
}

fn push(table: &mut Vec<f32>, position: f32, inverse: f32) {
    let angle = position * inverse;
    table.push(f64::from(angle).cos() as f32);
    table.push(f64::from(angle).sin() as f32);
}

/// The table of the encoder (Qwen3-VL text tower): one position for each row, and `head_dim / 2` pairs `(i, i +
/// head_dim / 2)`. Text has the same position on the three axes of its multimodal rotary positions, so the sections
/// do not change the angles.
pub fn half_table(positions: &[i64], head_dim: usize, theta: f32) -> Vec<f32> {
    let inverse = inverse_frequencies(head_dim, theta);
    let mut table = Vec::with_capacity(positions.len() * head_dim);
    for &position in positions {
        for &f in &inverse {
            push(&mut table, position as f32, f);
        }
    }
    table
}

/// The table of the transformer: three positions for each row, one for each axis, and the pairs `(2i, 2i + 1)`. The
/// axes have `axes[0]`, `axes[1]` and `axes[2]` dimensions, in that order (`transformer.py:655-713`).
pub fn axes_table(positions: &[[i64; 3]], axes: [usize; 3], theta: f32) -> Vec<f32> {
    let inverse: Vec<Vec<f32>> = axes.iter().map(|&dim| inverse_frequencies(dim, theta)).collect();
    let pairs: usize = axes.iter().sum::<usize>() / 2;
    let mut table = Vec::with_capacity(positions.len() * pairs * 2);
    for position in positions {
        for (axis, frequencies) in inverse.iter().enumerate() {
            for &f in frequencies {
                push(&mut table, position[axis] as f32, f);
            }
        }
    }
    table
}

/// The positions of a step's rows in the transformer (spec 11.9c, "The steps"): `conditioning` rows with position
/// `j` on each axis, then the latent tokens of an `h` by `w` grid in row order. A latent token `(y, x)` has the
/// position `conditioning` on the first axis, and `y - (h - h / 2)` and `x - (w - w / 2)` on the other two
/// (`transformer.py:695-696`).
pub fn step_positions(conditioning: usize, h: usize, w: usize) -> Vec<[i64; 3]> {
    let mut positions: Vec<[i64; 3]> = (0..conditioning as i64).map(|j| [j, j, j]).collect();
    let (top, left) = ((h - h / 2) as i64, (w - w / 2) as i64);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            positions.push([conditioning as i64, y - top, x - left]);
        }
    }
    positions
}

/// The sinusoid of a timestep `t` in [0, 1]: `dim` values, the cos half first (`transformer.py:136-154`).
pub fn timestep_sinusoid(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let scaled = 1000.0f32 * t;
    let frequencies: Vec<f32> = (0..half)
        .map(|i| (-(10000.0f32.ln()) * i as f32 / half as f32).exp())
        .collect();
    let mut out: Vec<f32> = frequencies
        .iter()
        .map(|&f| f64::from(scaled * f).cos() as f32)
        .collect();
    out.extend(frequencies.iter().map(|&f| f64::from(scaled * f).sin() as f32));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_latent_grid_is_centred_as_the_reference_centres_it() {
        let positions = step_positions(3, 4, 5);
        assert_eq!(&positions[..3], &[[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
        assert_eq!(positions.len(), 3 + 20);
        // h = 4: -(4 - 2) .. 2 is -2..=1. w = 5: -(5 - 2) .. 2 is -3..=1.
        assert_eq!(positions[3], [3, -2, -3]);
        assert_eq!(positions[3 + 19], [3, 1, 1]);
    }

    #[test]
    fn the_tables_hold_a_pair_for_each_row_and_frequency() {
        let half = half_table(&[0, 7], 128, 5_000_000.0);
        assert_eq!(half.len(), 2 * 64 * 2);
        assert_eq!((half[0], half[1]), (1.0, 0.0), "position 0 is no rotation");
        // The first frequency is 1, so row 7 turns its first pair by 7 radians.
        assert!((half[128] - 7f32.cos()).abs() < 1e-6 && (half[129] - 7f32.sin()).abs() < 1e-6);
        let axes = axes_table(&[[5, -2, 3]], [16, 56, 56], 10000.0);
        assert_eq!(axes.len(), 64 * 2);
        // The first pair of each axis has frequency 1: angles 5, -2 and 3.
        assert!((axes[0] - 5f32.cos()).abs() < 1e-6);
        assert!((axes[2 * 8 + 1] - (-2f32).sin()).abs() < 1e-6);
        assert!((axes[2 * (8 + 28)] - 3f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn the_timestep_sinusoid_has_the_cos_half_first() {
        let s = timestep_sinusoid(0.5, 256);
        assert_eq!(s.len(), 256);
        // The first frequency is 1, so its angle is 500.
        assert!((s[0] - 500f32.cos()).abs() < 1e-5);
        assert!((s[128] - 500f32.sin()).abs() < 1e-5);
        assert_eq!(timestep_sinusoid(0.0, 256)[..128], vec![1.0; 128][..]);
    }
}
