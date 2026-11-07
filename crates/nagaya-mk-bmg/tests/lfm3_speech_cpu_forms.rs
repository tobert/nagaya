//! The CPU forms of the speech ops of the LFM3 binary against closed forms. The card test `xe_lfm3_speech_ops.rs`
//! compares the kernels with these forms, so a wrong form would hide a wrong kernel. They need no card.

mod common;

use common::lfm3::{rel, values};
use common::lfm3_speech::*;

#[test]
fn a_group_attends_only_to_its_own_rows_in_both_directions() {
    let (n, heads, kv, d, group) = (8, 4, 2, 16, 4);
    let q = values(1, n * heads * d, 1.0);
    let k = values(2, n * kv * d, 1.0);
    let v = values(3, n * kv * d, 1.0);
    let out = group_attn(&q, &k, &v, n, heads, kv, d, group, 0.25);
    // A score of zero for each key gives the mean of the values of the group.
    let flat = group_attn(&vec![0.0; q.len()], &k, &v, n, heads, kv, d, group, 0.25);
    for t in 0..n {
        let lo = t / group * group;
        for h in 0..heads {
            let g = h / 2;
            for j in 0..d {
                let mean: f64 = (lo..lo + group).map(|s| v[(s * kv + g) * d + j] as f64).sum::<f64>() / group as f64;
                assert!((flat[(t * heads + h) * d + j] as f64 - mean).abs() < 1e-6);
            }
        }
    }
    // A change of a key outside the group of row 0 leaves row 0 as it was, and one inside changes it.
    let mut far = k.clone();
    far[(5 * kv) * d] += 1.0;
    let moved = group_attn(&q, &far, &v, n, heads, kv, d, group, 0.25);
    assert_eq!(moved[..heads * d], out[..heads * d]);
    let mut near = k.clone();
    near[(3 * kv) * d] += 1.0;
    let moved = group_attn(&q, &near, &v, n, heads, kv, d, group, 0.25);
    assert_ne!(
        moved[..heads * d],
        out[..heads * d],
        "row 3 is in the group of row 0, after it"
    );
}

#[test]
fn the_polar_form_gives_the_magnitude_and_the_angle() {
    let (n, f, stride) = (2, 3, 8);
    let mut a = vec![0.0f32; n * stride];
    a[0] = 2.0f32.ln();
    a[3] = std::f32::consts::FRAC_PI_2;
    a[stride + 1] = 0.0;
    a[stride + 4] = std::f32::consts::PI;
    let out = polar(&a, n, f, stride);
    assert!((out[0] - 0.0).abs() < 1e-6 && (out[3] - 2.0).abs() < 1e-6);
    assert!((out[6 + 1] + 1.0).abs() < 1e-6 && out[6 + 4].abs() < 1e-6);
}

#[test]
fn the_inverse_fft_of_one_bin_is_a_cosine_times_the_window() {
    let nf = 16;
    let f = nf / 2 + 1;
    let w = vec![1.0f32; nf];
    let mut bins = vec![0.0f32; 2 * f];
    bins[3] = 1.0;
    bins[f + 2] = 0.5;
    bins[f] = 9.0;
    bins[f + f - 1] = 9.0;
    let out = irfft(&bins, 1, nf, &w);
    for (i, got) in out.iter().enumerate() {
        let a = 2.0 * std::f64::consts::PI * i as f64 / nf as f64;
        let want = (2.0 * (3.0 * a).cos() - 2.0 * 0.5 * (2.0 * a).sin()) / nf as f64;
        assert!((*got as f64 - want).abs() < 1e-7, "sample {i}: {got} against {want}");
    }
    let hann = hann(nf);
    let windowed = irfft(&bins, 1, nf, &hann);
    assert!(
        rel(
            &windowed,
            &out.iter().zip(&hann).map(|(a, b)| a * b).collect::<Vec<f32>>()
        ) < 1e-6
    );
    assert_eq!(hann[0], 0.0);
    assert_eq!(hann[nf / 2], 1.0);
}
