//! The gated DeltaNet core: causal convolution with silu, the gated delta rule, and the gated RMSNorm. It ports
//! `test_gdn.py` of mk (`ops/gdn.h`, `ops/gdn.hip`).
//!
//! Head order: every input stays in GGUF (tiled) order. Value head `p` uses the key and query head `p % 16`.
//!
//! The tolerance of mk stays: `REL(T) = 4 T 128 2^-24` in relative L2 error. The recurrence of the op and the chunked
//! algorithm of the reference are the same recurrence with another f32 association. Each decay is at most 0, so the error
//! grows about linearly in T.
//!
//! Changes from mk's test: the weights and the inputs come from the generator, not from layer 0 of the GGUF and numpy.
//! The reference output is in a fixture. It holds `y` and a corner of the final state of three value heads only, so the
//! relative error is over those heads. The cases with T = 64 and T = 200 are dropped: the fixture would be over its size.

use crate::fixture::Fixture;
use crate::inputs;
use crate::ops_lib::{assert_same, lib, ptr, ptr_mut, relerr, same_bits};
use crate::sym;
use std::ffi::c_void;

const KH: usize = 16;
const VH: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const CONV: usize = 4;
const CDIM: usize = 2 * KH * KD + VH * VD;

/// The weights of the layer, as `tools/op_groups/gdn.py` makes them: `ssm_a` is negative, so every decay is at most 0.
struct Weights {
    ssm_a: Vec<f32>,
    dt_bias: Vec<f32>,
    conv_w: Vec<f32>,
    gamma: Vec<f32>,
}

fn weights() -> Weights {
    let ssm_a: Vec<f32> = inputs::uniforms(9001, VH)
        .iter()
        .map(|u| -(0.5f32 + 4.0f32 * u))
        .collect();
    assert!(ssm_a.iter().all(|a| *a < 0.0), "ssm_a = -exp(A_log) must be negative");
    Weights {
        ssm_a,
        dt_bias: inputs::normals_scaled(9002, VH, 0.1),
        conv_w: inputs::normals_scaled(9003, CDIM * CONV, 0.3),
        gamma: inputs::normals(9004, VD).iter().map(|g| 1.0f32 + 0.1f32 * g).collect(),
    }
}

struct Inputs {
    qkv: Vec<f32>,
    z: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    conv_state: Vec<f32>,
    state: Vec<f32>,
}

/// The inputs of `t` tokens. `scale` is the scale of the prior state: 0 for a fresh context.
fn make_inputs(t: usize, scale: f32) -> Inputs {
    let base = 10000 + 10 * t as u64 + u64::from(scale != 0.0);
    Inputs {
        qkv: inputs::normals(base * 10 + 1, t * CDIM),
        z: inputs::normals(base * 10 + 2, t * VH * VD),
        a: inputs::normals_scaled(base * 10 + 3, t * VH, 0.5),
        b: inputs::normals_scaled(base * 10 + 4, t * VH, 0.5),
        conv_state: inputs::normals_scaled(base * 10 + 5, CDIM * (CONV - 1), scale),
        state: inputs::normals_scaled(base * 10 + 6, VH * KD * VD, scale),
    }
}

struct Output {
    y: Vec<f32>,
    conv_state: Vec<f32>,
    state: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run_op(
    w: &Weights,
    qkv: &[f32],
    z: &[f32],
    a: &[f32],
    b: &[f32],
    conv_state: &[f32],
    state: &[f32],
    grid: u32,
) -> Output {
    let t = qkv.len() / CDIM;
    let mut out = Output {
        y: vec![f32::NAN; t * VH * VD],
        conv_state: vec![f32::NAN; CDIM * (CONV - 1)],
        state: vec![f32::NAN; VH * KD * VD],
    };
    let f = sym!(
        lib("gdn"),
        "mk_gdn",
        fn(
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_void,
            u32,
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            u32,
        ) -> i32
    );
    let rc = unsafe {
        f(
            ptr(qkv),
            ptr(z),
            ptr(a),
            ptr(b),
            t as u32,
            ptr(conv_state),
            ptr(&w.conv_w),
            ptr(&w.ssm_a),
            ptr(&w.dt_bias),
            ptr(&w.gamma),
            ptr(state),
            ptr_mut(&mut out.y),
            ptr_mut(&mut out.conv_state),
            ptr_mut(&mut out.state),
            grid,
        )
    };
    assert_eq!(rc, 0, "mk_gdn");
    out
}

fn run(w: &Weights, i: &Inputs, grid: u32) -> Output {
    run_op(w, &i.qkv, &i.z, &i.a, &i.b, &i.conv_state, &i.state, grid)
}

fn tol_for(t: usize) -> f64 {
    4.0 * t as f64 * 128.0 * 2f64.powi(-24)
}

/// The new conv state is the last 3 raw inputs of the extended sequence: the old state, then the tokens. It is a copy.
fn expected_conv_state(i: &Inputs, t: usize) -> Vec<f32> {
    let mut out = vec![0f32; CDIM * (CONV - 1)];
    for c in 0..CDIM {
        for s in 0..CONV - 1 {
            // Position `t + s` of the extended row of length t + 3.
            let p = t + s;
            out[c * (CONV - 1) + s] = if p < CONV - 1 {
                i.conv_state[c * (CONV - 1) + p]
            } else {
                i.qkv[(p - (CONV - 1)) * CDIM + c]
            };
        }
    }
    out
}

#[test]
fn the_config_is_the_config_that_the_test_assumes() {
    let f = Fixture::load("gdn");
    let c = f.meta("config");
    assert_eq!(c["hidden_act"], "silu");
    assert_eq!(c["output_gate_type"], "sigmoid");
    assert_eq!(c["linear_conv_kernel_dim"], CONV);
}

#[test]
fn gdn_matches_reference() {
    let f = Fixture::load("gdn");
    let w = weights();
    let heads: Vec<usize> = f
        .meta("heads")
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h.as_u64().unwrap() as usize)
        .collect();
    let rows = f.meta("state_rows").as_u64().unwrap() as usize;
    for case in f.meta("cases").as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let t = case["t"].as_u64().unwrap() as usize;
        let from_zero = case["from_zero"].as_bool().unwrap();
        let i = make_inputs(t, if from_zero { 0.0 } else { 0.3 });
        let got = run(&w, &i, 0);
        assert!(got.y.iter().all(|v| v.is_finite()), "{name}: unwritten or non-finite y");
        assert!(
            got.conv_state.iter().all(|v| v.is_finite()),
            "{name}: unwritten or non-finite conv state"
        );
        assert!(
            got.state.iter().all(|v| v.is_finite()),
            "{name}: unwritten or non-finite recurrent state"
        );
        // The heads of the fixture.
        let mut y = Vec::new();
        for token in 0..t {
            for &h in &heads {
                y.extend_from_slice(&got.y[(token * VH + h) * VD..(token * VH + h + 1) * VD]);
            }
        }
        let mut state = Vec::new();
        for &h in &heads {
            state.extend_from_slice(&got.state[h * KD * VD..h * KD * VD + rows * VD]);
        }
        let tol = tol_for(t);
        let e_y = relerr(&y, &f.f32_flat(&format!("{name}_y")));
        let e_state = relerr(&state, &f.f32_flat(&format!("{name}_state")));
        eprintln!("{name}: y relerr {e_y:.2e}, state relerr {e_state:.2e} (tol {tol:.2e})");
        assert!(e_y < tol, "{name}: y relerr {e_y:.3e} >= tol {tol:.3e}");
        assert!(e_state < tol, "{name}: state relerr {e_state:.3e} >= tol {tol:.3e}");
        // No sum is in the conv state: it is a copy of raw inputs, so it is exact.
        assert_same(
            &got.conv_state,
            &expected_conv_state(&i, t),
            &format!("{name}: the new conv state"),
        );
    }
}

#[test]
fn gdn_continuation_bit_identical() {
    let (t1, t2) = (5, 11);
    let w = weights();
    let i = Inputs {
        qkv: inputs::normals(20001, (t1 + t2) * CDIM),
        z: inputs::normals(20002, (t1 + t2) * VH * VD),
        a: inputs::normals_scaled(20003, (t1 + t2) * VH, 0.5),
        b: inputs::normals_scaled(20004, (t1 + t2) * VH, 0.5),
        conv_state: inputs::normals_scaled(20005, CDIM * (CONV - 1), 0.2),
        state: inputs::normals_scaled(20006, VH * KD * VD, 0.2),
    };
    let once = run(&w, &i, 0);
    let p = |n: usize, v: &[f32], width: usize| -> (Vec<f32>, Vec<f32>) {
        (v[..n * width].to_vec(), v[n * width..].to_vec())
    };
    let (q1, q2) = p(t1, &i.qkv, CDIM);
    let (z1, z2) = p(t1, &i.z, VH * VD);
    let (a1, a2) = p(t1, &i.a, VH);
    let (b1, b2) = p(t1, &i.b, VH);
    let first = run_op(&w, &q1, &z1, &a1, &b1, &i.conv_state, &i.state, 0);
    let second = run_op(&w, &q2, &z2, &a2, &b2, &first.conv_state, &first.state, 0);
    let y: Vec<f32> = first.y.iter().chain(&second.y).copied().collect();
    assert_same(&once.y, &y, "y of T1 then T2 against T1+T2");
    assert_same(&once.conv_state, &second.conv_state, "conv state");
    assert_same(&once.state, &second.state, "recurrent state");
}

#[test]
fn gdn_repeatable_and_grid_independent() {
    let w = weights();
    let t = 23;
    let i = Inputs {
        qkv: inputs::normals(21001, t * CDIM),
        z: inputs::normals(21002, t * VH * VD),
        a: inputs::normals_scaled(21003, t * VH, 0.5),
        b: inputs::normals_scaled(21004, t * VH, 0.5),
        conv_state: inputs::normals_scaled(21005, CDIM * (CONV - 1), 0.1),
        state: inputs::normals_scaled(21006, VH * KD * VD, 0.1),
    };
    let base = run(&w, &i, 0);
    for repeat in 0..20 {
        let o = run(&w, &i, 0);
        assert_same(&o.y, &base.y, &format!("repeat {repeat}: y"));
        assert_same(&o.conv_state, &base.conv_state, &format!("repeat {repeat}: conv state"));
        assert_same(&o.state, &base.state, &format!("repeat {repeat}: state"));
    }
    for grid in [1, 3, 7, 48, 10240] {
        let o = run(&w, &i, grid);
        assert_same(&o.y, &base.y, &format!("grid {grid}: y"));
        assert_same(&o.conv_state, &base.conv_state, &format!("grid {grid}: conv state"));
        assert_same(&o.state, &base.state, &format!("grid {grid}: state"));
    }
    assert!(!same_bits(&base.y, &vec![0.0; base.y.len()]), "y is not all zero");
}

/// The megakernel is one function, so its register budget is the maximum over every op. That budget sets how many
/// workgroups are resident together. 64 floats of state a thread and the working set of a token fit in 128 VGPRs, with
/// nothing spilled.
#[test]
fn gdn_core_register_budget() {
    let f = sym!(lib("gdn"), "mk_gdn_core_resources", fn(*mut i32, *mut i32) -> i32);
    let (mut regs, mut scratch) = (0i32, 0i32);
    assert_eq!(unsafe { f(&mut regs, &mut scratch) }, 0);
    eprintln!("k_gdn_core: {regs} VGPRs, {scratch} B/lane scratch");
    assert_eq!(scratch, 0, "k_gdn_core spills: {scratch} B/lane of scratch");
    assert!(regs <= 128, "k_gdn_core uses {regs} VGPRs, budget 128");
}
