//! The CPU forms of the ops of the audio encoder of LFM3-Omni-2B against the plain formula of `nn.Conv2d` in the
//! layout of the reference. The card test `xe_lfm3_audio_ops.rs` compares the kernels with these forms, so a wrong form
//! would hide a wrong kernel. They need no card.

mod common;

use common::lfm3::{from_bf16, rel};
use common::lfm3_audio::*;

#[test]
fn the_subsampling_convolution_is_conv2d_with_a_stride_of_2_on_the_channel_last_plane() {
    for (name, n, k, cin, m, silu, seed) in SUBSAMPLE_CASES {
        let (x, w, b) = subsample_inputs(n, k, cin, m, seed);
        let got = subsample(&x, &w, &b, n, k, cin, m, silu);
        // The reference holds the channels first, then mel as the height, then time as the width.
        let mut planes = vec![0.0f64; cin * k * n];
        for t in 0..n {
            for h in 0..k {
                for c in 0..cin {
                    planes[(c * k + h) * n + t] = x[(t * k + h) * cin + c] as f64;
                }
            }
        }
        let wf: Vec<f64> = w.iter().map(|v| from_bf16(*v) as f64).collect();
        let bf: Vec<f64> = b.iter().map(|v| from_bf16(*v) as f64).collect();
        let want = conv2d_reference(&planes, &wf, &bf, n, k, cin, m, silu);
        let (to_n, ho_n) = (n.div_ceil(2), k.div_ceil(2));
        let mut want_last = vec![0.0f32; to_n * ho_n * m];
        for c in 0..m {
            for ho in 0..ho_n {
                for to in 0..to_n {
                    want_last[(to * ho_n + ho) * m + c] = want[(c * ho_n + ho) * to_n + to] as f32;
                }
            }
        }
        let err = rel(&got, &want_last);
        println!("{name}: relative error {err:.3e}");
        assert!(err < 1e-6, "{name}: relative error {err} against conv2d");
    }
}
