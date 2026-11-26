//! The prepared weights of a plan, from the tensors of a checkpoint. The forms are exact but the BatchNorm fold.

use std::collections::HashMap;

use nagaya_family_beatthis::Error;
use nagaya_family_beatthis::plan::{Prep, Weight};
use nagaya_family_beatthis::weights::{self, Tensors};
use safetensors::{Dtype, tensor::TensorView};

/// A checkpoint in memory: each tensor holds `base + index` as f32.
fn file(tensors: &[(&str, &[usize], f32)]) -> Vec<u8> {
    let data: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(n, shape, base)| {
            let len: usize = shape.iter().product();
            let bytes = (0..len).flat_map(|i| (base + i as f32).to_le_bytes()).collect();
            ((*n).to_owned(), shape.to_vec(), bytes)
        })
        .collect();
    let views: HashMap<String, TensorView> = data
        .iter()
        .map(|(n, s, b)| (n.clone(), TensorView::new(Dtype::F32, s.clone(), b).unwrap()))
        .collect();
    safetensors::serialize(&views, None).unwrap()
}

fn prep(bytes: &[u8], prep: Prep) -> Result<Vec<f32>, Error> {
    let t = Tensors::parse(bytes).unwrap();
    weights::prepare(&[Weight { prep }], &t).map(|mut v| v.remove(0))
}

#[test]
fn a_plain_weight_is_the_tensor() {
    let b = file(&[("a", &[2, 3], 1.0)]);
    assert_eq!(
        prep(
            &b,
            Prep::Plain {
                name: "a".into(),
                shape: vec![2, 3]
            }
        )
        .unwrap(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn a_conv_kernel_moves_its_input_channels_last() {
    // [cout 2, cin 3, kf 2, kt 2]: the value at (co, ci, i, j) is co*12 + ci*4 + i*2 + j.
    let b = file(&[("k", &[2, 3, 2, 2], 0.0)]);
    let got = prep(
        &b,
        Prep::ConvKernel {
            name: "k".into(),
            cout: 2,
            cin: 3,
            kf: 2,
            kt: 2,
        },
    )
    .unwrap();
    for co in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                for ci in 0..3 {
                    let at = ((co * 2 + i) * 2 + j) * 3 + ci;
                    assert_eq!(got[at], (co * 12 + ci * 4 + i * 2 + j) as f32, "({co}, {i}, {j}, {ci})");
                }
            }
        }
    }
}

#[test]
fn a_column_permutation_reads_frequencies_first() {
    // [rows 2, channels 3 * freqs 2]: column c * 2 + f of the tensor goes to column f * 3 + c.
    let b = file(&[("w", &[2, 6], 0.0)]);
    let got = prep(
        &b,
        Prep::PermuteColumns {
            name: "w".into(),
            rows: 2,
            channels: 3,
            freqs: 2,
        },
    )
    .unwrap();
    for r in 0..2 {
        for c in 0..3 {
            for f in 0..2 {
                assert_eq!(got[r * 6 + f * 3 + c], (r * 6 + c * 2 + f) as f32);
            }
        }
    }
}

#[test]
fn a_batch_norm_folds_into_a_scale_and_a_shift() {
    let b = file(&[
        ("n.weight", &[2], 2.0),
        ("n.bias", &[2], 0.5),
        ("n.running_mean", &[2], 1.0),
        ("n.running_var", &[2], 3.0),
    ]);
    let scale = prep(
        &b,
        Prep::BnScale {
            prefix: "n".into(),
            channels: 2,
        },
    )
    .unwrap();
    let shift = prep(
        &b,
        Prep::BnShift {
            prefix: "n".into(),
            channels: 2,
        },
    )
    .unwrap();
    for c in 0..2 {
        let (w, bias, mean, var) = (2.0 + c as f64, 0.5 + c as f64, 1.0 + c as f64, 3.0 + c as f64);
        let s = w / (var + 1e-5).sqrt();
        assert_eq!(scale[c], s as f32);
        assert_eq!(shift[c], (bias - mean * s) as f32);
    }
}

#[test]
fn a_missing_tensor_is_an_error_that_names_it() {
    let b = file(&[("a", &[2], 0.0)]);
    let e = prep(
        &b,
        Prep::Plain {
            name: "b".into(),
            shape: vec![2],
        },
    )
    .unwrap_err();
    assert!(matches!(&e, Error::Tensor { name, .. } if name == "b"), "{e:?}");
}

#[test]
fn a_tensor_of_another_shape_is_an_error() {
    let b = file(&[("a", &[2, 3], 0.0)]);
    let e = prep(
        &b,
        Prep::Plain {
            name: "a".into(),
            shape: vec![3, 2],
        },
    )
    .unwrap_err();
    assert!(
        matches!(&e, Error::Tensor { why, .. } if why.contains("[2, 3]")),
        "{e:?}"
    );
}

#[test]
fn a_tensor_that_is_not_f32_is_an_error() {
    let bytes = vec![0u8; 4];
    let views: HashMap<String, TensorView> = [("a".to_owned(), TensorView::new(Dtype::F16, vec![2], &bytes).unwrap())]
        .into_iter()
        .collect();
    let b = safetensors::serialize(&views, None).unwrap();
    let e = prep(
        &b,
        Prep::Plain {
            name: "a".into(),
            shape: vec![2],
        },
    )
    .unwrap_err();
    assert!(matches!(&e, Error::Tensor { why, .. } if why.contains("F16")), "{e:?}");
}
