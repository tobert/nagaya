//! The ops of one layer of Inkling 2, in `f32` on the host, one row or one position at a time.
//!
//! Each op cites the line of `modeling_inkling2_loop.py` (transformers `046bdb8cc2`) that it follows. The sums are
//! `f32`, as in the oracle. The order of a sum is not the order of torch, so a result differs from the oracle by a
//! few units in the last place, not in its bits.

/// The width of the residual stream.
pub const HIDDEN: usize = 2304;
/// The width of one head of a query, a key or a value.
pub const HEAD_DIM: usize = 128;
/// The count of query heads.
pub const HEADS: usize = 24;
/// The count of key and value heads. Query head `h` reads key head `h / 3`.
pub const KV_HEADS: usize = 8;
/// The width of the relative state of one head.
pub const D_REL: usize = 16;
/// The count of distances that the relative bias covers, and the window of a sliding layer.
pub const REL_EXTENT: usize = 1024;
/// The taps of each short convolution.
pub const CONV_TAPS: usize = 4;
/// The count of routed experts, and the count that each position selects.
pub const EXPERTS: usize = 128;
pub const TOP_K: usize = 8;
/// The width of the inner row of one expert.
pub const INTERMEDIATE: usize = 1152;
/// The rank of a message gate.
pub const GATE_RANK: usize = 64;
/// The epsilon of every RMS norm (`rms_norm_eps` of the config).
pub const EPS: f32 = 1e-6;
/// The L2 bound of a query head and a key head (`qk_l2_clip_threshold`).
pub const QK_CLIP: f32 = 22.627_417;
/// The L2 bound of a relative state (`rel_l2_clip_threshold`).
pub const REL_CLIP: f32 = 8.0;
/// The softcaps of the expert gate, the expert up row and the logits.
pub const GATE_SOFTCAP: f32 = 4.0;
pub const UP_SOFTCAP: f32 = 25.0;
pub const LOGIT_SOFTCAP: f32 = 30.0;
/// The factor of the routing weights (`route_scale`).
pub const ROUTE_SCALE: f32 = 8.0;
/// The log scale of the full layers: `1 + ALPHA * ln(max((p + 1) / FLOOR, 1))`.
pub const LOG_ALPHA: f32 = 0.1;
pub const LOG_FLOOR: f32 = 8192.0;

/// Widen `bf16` bits to the `f32` of the same value. The result is exact.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Widen a row of little-endian `bf16` bytes into `out`.
pub fn widen_bf16_row(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(
        bytes.len(),
        out.len() * 2,
        "widen_bf16_row: {} bytes for {} values",
        bytes.len(),
        out.len()
    );
    for (o, b) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
        *o = bf16_to_f32(u16::from_le_bytes(*b));
    }
}

/// The dot product of two rows, in `f32` with 16 lanes, joined in pairs at the end.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: lengths {} and {}", a.len(), b.len());
    let mut lanes = [0f32; 16];
    let (ca, ra) = a.as_chunks::<16>();
    let (cb, rb) = b.as_chunks::<16>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..16 {
            lanes[i] += x[i] * y[i];
        }
    }
    let mut width = 16;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            lanes[i] += lanes[i + width];
        }
    }
    let mut sum = lanes[0];
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

/// `out = W x` for an `f32` matrix of `out.len()` rows, row-major.
pub fn matvec(w: &[f32], x: &[f32], out: &mut [f32]) {
    assert_eq!(
        w.len(),
        x.len() * out.len(),
        "matvec: {} weights for {}x{}",
        w.len(),
        out.len(),
        x.len()
    );
    for (o, row) in out.iter_mut().zip(w.chunks_exact(x.len())) {
        *o = dot(row, x);
    }
}

/// `out = W x` for a `bf16` matrix of `out.len()` rows, row-major, as the checkpoint holds it.
pub fn matvec_bf16(w: &[u8], x: &[f32], out: &mut [f32]) {
    assert_eq!(
        w.len(),
        2 * x.len() * out.len(),
        "matvec_bf16: {} bytes for {}x{}",
        w.len(),
        out.len(),
        x.len()
    );
    let mut row = vec![0f32; x.len()];
    for (o, bytes) in out.iter_mut().zip(w.chunks_exact(2 * x.len())) {
        widen_bf16_row(bytes, &mut row);
        *o = dot(&row, x);
    }
}

/// The mean of the squares of a row, plus `eps`, as the factor `1 / sqrt(..)`.
fn rms_factor(x: &[f32], eps: f32) -> f32 {
    let mean = dot(x, x) / x.len() as f32;
    1.0 / (mean + eps).sqrt()
}

/// The RMS norm with a weight: `weight * (x / sqrt(mean(x^2) + eps))` (`Inkling2LoopRMSNorm`, line 740).
pub fn rms_norm(x: &[f32], weight: &[f32], out: &mut [f32]) {
    assert!(
        x.len() == weight.len() && x.len() == out.len(),
        "rms_norm: lengths differ"
    );
    let f = rms_factor(x, EPS);
    for ((o, v), w) in out.iter_mut().zip(x).zip(weight) {
        *o = w * (v * f);
    }
}

/// The RMS norm with no weight, in place (`Inkling2LoopUnweightedRMSNorm`, line 72). The values of a run, the
/// stacked states of a residual and the inputs of a message gate take it.
pub fn rms_norm_unweighted(x: &mut [f32]) {
    let f = rms_factor(x, EPS);
    for v in x.iter_mut() {
        *v *= f;
    }
}

/// Scale a vector down to the L2 norm `threshold` when its norm is larger, in place (`Inkling2LoopL2Clip`, line
/// 183). The factor is `min(threshold / (norm + 1e-6), 1)`.
pub fn l2_clip(x: &mut [f32], threshold: f32) {
    let norm = dot(x, x).sqrt();
    let f = (threshold / (norm + 1e-6)).min(1.0);
    for v in x.iter_mut() {
        *v *= f;
    }
}

/// Multiply each head by its learned scale, then clip each head (`Inkling2LoopScaledL2Clip`, line 198). `x` and
/// `scale` hold whole heads of `head_dim` values.
pub fn scaled_l2_clip(x: &mut [f32], scale: &[f32], head_dim: usize, threshold: f32) {
    assert_eq!(x.len(), scale.len(), "scaled_l2_clip: lengths differ");
    for (v, s) in x.iter_mut().zip(scale) {
        *v *= s;
    }
    for head in x.chunks_exact_mut(head_dim) {
        l2_clip(head, threshold);
    }
}

/// The log scale of position `p` in the full layers: `1 + 0.1 * ln(max((p + 1) / 8192, 1))` (`Inkling2LoopLogScale`,
/// line 217). It is 1 below position 8,191.
pub fn log_scale(position: usize) -> f32 {
    let ratio = ((position + 1) as f32 / LOG_FLOOR).max(1.0);
    1.0 + LOG_ALPHA * ratio.ln()
}

/// The relative profile of each head: `states[h] @ proj`, `REL_EXTENT` values for each head (line 267). `states`
/// holds the clipped relative states, `HEADS x D_REL`, and `proj` is `D_REL x REL_EXTENT`.
pub fn relative_logits(states: &[f32], proj: &[f32], out: &mut [f32]) {
    let heads = states.len() / D_REL;
    assert_eq!(
        proj.len(),
        D_REL * REL_EXTENT,
        "relative_logits: proj has {} values",
        proj.len()
    );
    assert_eq!(
        out.len(),
        heads * REL_EXTENT,
        "relative_logits: out has {} values",
        out.len()
    );
    for h in 0..heads {
        let row = &mut out[h * REL_EXTENT..(h + 1) * REL_EXTENT];
        row.fill(0.0);
        for d in 0..D_REL {
            let s = states[h * D_REL + d];
            for (o, p) in row.iter_mut().zip(&proj[d * REL_EXTENT..(d + 1) * REL_EXTENT]) {
                *o += s * p;
            }
        }
    }
}

/// The relative bias of one head at a distance `query - key`: the profile at that distance, and 0 outside
/// `0 <= distance < REL_EXTENT` (line 268 to 270).
pub fn relative_bias(profile: &[f32], distance: i64) -> f32 {
    if (0..REL_EXTENT as i64).contains(&distance) {
        profile[distance as usize]
    } else {
        0.0
    }
}

/// The attention of one query head over its keys, with a sink (`eager_attention_forward`, lines 126 to 139).
///
/// The score of key `j` is `q . k_j / sqrt(HEAD_DIM) + bias_j`. The sink is one more score with a value of zero, so
/// it only takes weight. `keys` and `values` hold one row of `HEAD_DIM` for each entry of `bias`.
pub fn attend(q: &[f32], keys: &[&[f32]], values: &[&[f32]], bias: &[f32], sink: f32, out: &mut [f32]) {
    assert!(
        keys.len() == values.len() && keys.len() == bias.len(),
        "attend: counts differ"
    );
    let scaling = (HEAD_DIM as f32).powf(-0.5);
    let scores: Vec<f32> = keys.iter().zip(bias).map(|(k, b)| dot(q, k) * scaling + b).collect();
    let max = scores.iter().copied().fold(sink, f32::max);
    let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    // The sums are `f64`. Sums of `f32` in key order drift with the count of keys: at 65,024 positions the logits
    // moved by 3.5e-4, and at 130,560 by 4.8e-2.
    let total = weights.iter().map(|w| *w as f64).sum::<f64>() + (sink - max).exp() as f64;
    let mut sums = vec![0f64; out.len()];
    for (w, v) in weights.iter().zip(values) {
        let p = *w as f64 / total;
        for (s, x) in sums.iter_mut().zip(v.iter()) {
            *s += p * *x as f64;
        }
    }
    for (o, s) in out.iter_mut().zip(&sums) {
        *o = *s as f32;
    }
}

/// Push the input row `x` into a convolution state of 3 rows, oldest first.
pub fn conv_push(state: &mut [f32], x: &[f32]) {
    let c = x.len();
    assert_eq!(
        state.len(),
        (CONV_TAPS - 1) * c,
        "conv_push: state has {} values",
        state.len()
    );
    state.copy_within(c.., 0);
    state[(CONV_TAPS - 2) * c..].copy_from_slice(x);
}

/// One position of a short convolution with fixed taps (`Inkling2LoopShortConvolution`, lines 316 and 392).
///
/// `weight` is `C x 4` as the checkpoint holds it, and tap 3 weights the current row. `state` holds the 3 rows
/// before `x`, oldest first. The output is `(conv + x) / sqrt(5)`.
pub fn short_conv(state: &[f32], x: &[f32], weight: &[f32], out: &mut [f32]) {
    let c = x.len();
    assert!(
        state.len() == 3 * c && weight.len() == CONV_TAPS * c && out.len() == c,
        "short_conv: sizes"
    );
    let divisor = ((CONV_TAPS + 1) as f64).sqrt() as f32;
    for i in 0..c {
        let w = &weight[i * CONV_TAPS..(i + 1) * CONV_TAPS];
        let sum = w[0] * state[i] + w[1] * state[c + i] + w[2] * state[2 * c + i] + w[3] * x[i];
        out[i] = (sum + x[i]) / divisor;
    }
}

/// One position of the convolution of the key or the value of a full layer, whose taps change with each position
/// (`Inkling2LoopDynamicShortConvolution`, lines 430 to 454).
///
/// The tap at lag `t` of channel `i` is `static_w[t][i] + taps[i / HEAD_DIM][t]`, and lag 0 is the current row.
/// `taps` is `weight_proj` of the layer input, `KV_HEADS x 4`. The output is `(conv + x) / sqrt(9)`.
pub fn dynamic_conv(state: &[f32], x: &[f32], static_w: &[f32], taps: &[f32], out: &mut [f32]) {
    let c = x.len();
    assert!(
        state.len() == 3 * c && static_w.len() == CONV_TAPS * c && out.len() == c,
        "dynamic_conv: sizes"
    );
    assert_eq!(
        taps.len(),
        c / HEAD_DIM * CONV_TAPS,
        "dynamic_conv: taps has {} values",
        taps.len()
    );
    let divisor = ((2 * CONV_TAPS + 1) as f64).sqrt() as f32;
    for i in 0..c {
        let head = &taps[i / HEAD_DIM * CONV_TAPS..(i / HEAD_DIM + 1) * CONV_TAPS];
        // Lag 0 is x, lag 1 is the newest row of the state, and lag 3 its oldest.
        let rows = [x[i], state[2 * c + i], state[c + i], state[i]];
        let mut sum = 0f32;
        for lag in 0..CONV_TAPS {
            sum += rows[lag] * (static_w[lag * c + i] + head[lag]);
        }
        out[i] = (sum + x[i]) / divisor;
    }
}

/// The attention residual of one position (`Inkling2LoopBlockAttentionResidual`, lines 838 to 853).
///
/// It stacks the block outputs and the partial row, takes the RMS norm of each with no weight, and gives one row for
/// each query: the softmax over `query . row / sqrt(HIDDEN)` of the normed rows. The message input of a cycle
/// (line 158) is the same op with 4 queries.
pub fn attention_residual(blocks: &[&[f32]], partial: &[f32], queries: &[&[f32]], out: &mut [Vec<f32>]) {
    assert_eq!(
        queries.len(),
        out.len(),
        "attention_residual: {} queries, {} outputs",
        queries.len(),
        out.len()
    );
    let mut stacked: Vec<Vec<f32>> = blocks.iter().map(|b| b.to_vec()).collect();
    stacked.push(partial.to_vec());
    for row in &mut stacked {
        rms_norm_unweighted(row);
    }
    let scaling = (partial.len() as f32).powf(-0.5);
    for (q, o) in queries.iter().zip(out.iter_mut()) {
        let scores: Vec<f32> = stacked.iter().map(|s| dot(q, s) * scaling).collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f32 = weights.iter().sum();
        o.clear();
        o.resize(partial.len(), 0.0);
        for (w, s) in weights.iter().zip(&stacked) {
            let p = w / total;
            for (v, x) in o.iter_mut().zip(s) {
                *v += p * x;
            }
        }
    }
}

/// The message gate of one position: `(1 - forget) * h + gain * m` (`Inkling2LoopMessageGate`, lines 90 to 96).
///
/// The gates are `tanh` of `forget_w g` and `gain_w g`, where `g = down_w [norm(h), norm(m)]` and the norm has no
/// weight. `down_w` is `GATE_RANK x 2 HIDDEN`, and `forget_w` and `gain_w` are `HIDDEN x GATE_RANK`.
pub fn message_gate(h: &[f32], m: &[f32], down_w: &[f32], forget_w: &[f32], gain_w: &[f32], out: &mut [f32]) {
    let n = h.len();
    let mut both = Vec::with_capacity(2 * n);
    both.extend_from_slice(h);
    both.extend_from_slice(m);
    rms_norm_unweighted(&mut both[..n]);
    rms_norm_unweighted(&mut both[n..]);
    let rank = down_w.len() / (2 * n);
    let mut g = vec![0f32; rank];
    matvec(down_w, &both, &mut g);
    for i in 0..n {
        let forget = dot(&forget_w[i * rank..(i + 1) * rank], &g).tanh();
        let gain = dot(&gain_w[i * rank..(i + 1) * rank], &g).tanh();
        out[i] = (1.0 - forget) * h[i] + gain * m[i];
    }
}

/// The softcap `cap * tanh(x / cap)` (`Inkling2LoopSoftcap`, line 757).
pub fn softcap(x: f32, cap: f32) -> f32 {
    (x / cap).tanh() * cap
}

/// The sigmoid `1 / (1 + exp(-x))`.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The softcapped SwiGLU of an expert: `softcap(g, 4) * sigmoid(g) * softcap(u, 25)` (`Inkling2LoopSoftcappedSwiGLU`,
/// line 771). The sigmoid sees the raw gate.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert!(
        gate.len() == up.len() && gate.len() == out.len(),
        "swiglu: lengths differ"
    );
    for ((o, g), u) in out.iter_mut().zip(gate).zip(up) {
        *o = softcap(*g, GATE_SOFTCAP) * sigmoid(*g) * softcap(*u, UP_SOFTCAP);
    }
}

/// The router of one position (`Inkling2LoopTopKRouter`, lines 1165 to 1172).
///
/// The score of each expert is `sigmoid(logit)`. The selection takes the 8 largest `score + bias`, and the weights
/// are the scores of the selection, divided by their sum plus 1e-20, times 8. The result is in expert order.
pub fn route(logits: &[f32], bias: &[f32]) -> [(usize, f32); TOP_K] {
    assert!(logits.len() == bias.len() && logits.len() >= TOP_K, "route: lengths");
    let scores: Vec<f32> = logits.iter().map(|l| sigmoid(*l)).collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    // The order of equal keys is the lower index first, so the selection is stable.
    order.sort_by(|a, b| {
        (scores[*b] + bias[*b])
            .total_cmp(&(scores[*a] + bias[*a]))
            .then(a.cmp(b))
    });
    let mut chosen: [usize; TOP_K] = order[..TOP_K].try_into().unwrap();
    chosen.sort_unstable();
    let total: f32 = chosen.iter().map(|e| scores[*e]).sum::<f32>() + 1e-20;
    chosen.map(|e| (e, scores[e] / total * ROUTE_SCALE))
}

/// One control item on one residual row `h`, in place: the host form of `OP_INKLING2_CONTROL` (engine-abi "Control
/// vectors"). `add` gives `h + s v`, `ablate` gives `h - s (h . u) u`, and `clamp` gives `h + (s - h . u) u`, with `u =
/// v / |v|`. The dot products sum in `f64`. A vector of zeros has no direction, so it keeps the row.
pub fn control(h: &mut [f32], v: &[f32], op: nagaya_abi::ControlOp, s: f32) {
    assert_eq!(h.len(), v.len(), "control: a row and a vector of other widths");
    let hv: f64 = h.iter().zip(v).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
    let vv: f64 = v.iter().map(|b| f64::from(*b) * f64::from(*b)).sum();
    if vv == 0.0 {
        return;
    }
    let norm = vv.sqrt();
    let along = hv / norm;
    let s = f64::from(s);
    let coef = match op {
        nagaya_abi::ControlOp::Add => s,
        nagaya_abi::ControlOp::Ablate => -s * along / norm,
        nagaya_abi::ControlOp::Clamp => (s - along) / norm,
    } as f32;
    for (x, b) in h.iter_mut().zip(v) {
        *x = coef.mul_add(*b, *x);
    }
}

/// One expert on one position: `down (swiglu(gate_up x))`. `gate_up` holds the gate rows, then the up rows, each of
/// `HIDDEN` values (`Inkling2LoopExperts`, lines 802 and 803). `down` is `HIDDEN x INTERMEDIATE`.
pub fn expert(x: &[f32], gate_up: &[f32], down: &[f32], out: &mut [f32]) {
    let inner = gate_up.len() / x.len() / 2;
    let mut both = vec![0f32; 2 * inner];
    matvec(gate_up, x, &mut both);
    let mut act = vec![0f32; inner];
    swiglu(&both[..inner], &both[inner..], &mut act);
    matvec(down, &act, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host form of a control item keeps the definition of each op: an ablate of 1 leaves no part along the
    /// vector, a clamp sets that part to its strength, an add moves it by the strength times |v|, and a vector of zeros
    /// keeps the row.
    #[test]
    fn the_control_form_keeps_the_definition_of_each_operation() {
        use nagaya_abi::ControlOp;
        let v: Vec<f32> = (0..HIDDEN).map(|i| ((i * 7 % 13) as f32 - 6.0) / 5.0).collect();
        let h: Vec<f32> = (0..HIDDEN).map(|i| ((i * 5 % 11) as f32 - 4.0) / 3.0 + v[i]).collect();
        let norm = v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let along = |h: &[f32]| {
            h.iter()
                .zip(&v)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>()
                / norm
        };
        let before = along(&h);
        assert!(before.abs() > 1.0, "the row has a part along v: {before}");
        for (op, s, want) in [
            (ControlOp::Add, 2.0f32, before + 2.0 * norm),
            (ControlOp::Ablate, 1.0, 0.0),
            (ControlOp::Ablate, 0.5, before / 2.0),
            (ControlOp::Clamp, 8.0, 8.0),
        ] {
            let mut x = h.clone();
            control(&mut x, &v, op, s);
            assert!((along(&x) - want).abs() < 1e-3, "{op:?} {s}: {} not {want}", along(&x));
        }
        let mut x = h.clone();
        control(&mut x, &vec![0.0; HIDDEN], ControlOp::Clamp, 8.0);
        assert_eq!(x, h, "a vector of zeros keeps the row");
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    /// The attention over 131,072 keys gives the sums of `f64` to 1e-6. Sums of `f32` in key order drift by more at
    /// such a length.
    #[test]
    fn the_attention_over_131072_keys_keeps_its_sums_exact() {
        let n = 131_072;
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let bias: Vec<f32> = (0..n).map(|_| 2.0 * next()).collect();
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..HEAD_DIM).map(|_| 0.5 + 0.5 * next()).collect())
            .collect();
        let keys = vec![0f32; HEAD_DIM];
        let key_refs: Vec<&[f32]> = (0..n).map(|_| keys.as_slice()).collect();
        let value_refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        let q = vec![0f32; HEAD_DIM];
        let mut out = vec![0f32; HEAD_DIM];
        attend(&q, &key_refs, &value_refs, &bias, 0.0, &mut out);
        let max = bias.iter().copied().fold(0.0f32, f32::max) as f64;
        let weights: Vec<f64> = bias.iter().map(|b| ((*b - max as f32).exp()) as f64).collect();
        let total: f64 = weights.iter().sum::<f64>() + (-max).exp();
        let worst = (0..HEAD_DIM)
            .map(|d| {
                let want: f64 = weights.iter().zip(&rows).map(|(w, r)| w * r[d] as f64).sum::<f64>() / total;
                (out[d] as f64 - want).abs()
            })
            .fold(0.0, f64::max);
        assert!(
            worst < 1e-6,
            "the attention over {n} keys differs from the sums of f64 by {worst:.3e}"
        );
    }

    #[test]
    fn the_l2_clip_scales_only_a_long_vector() {
        let mut long = [3.0f32, 4.0];
        l2_clip(&mut long, 1.0);
        assert!(close(long[0], 0.6, 1e-5) && close(long[1], 0.8, 1e-5), "{long:?}");
        let mut short = [0.3f32, 0.4];
        l2_clip(&mut short, 1.0);
        assert_eq!(short, [0.3, 0.4]);
    }

    #[test]
    fn the_scaled_clip_scales_each_head_before_its_clip() {
        // Head 0: [1, 0] times 10 is [10, 0], clipped to [2, 0]. Head 1: [0, 1] times 1 stays.
        let mut x = [1.0f32, 0.0, 0.0, 1.0];
        scaled_l2_clip(&mut x, &[10.0, 10.0, 1.0, 1.0], 2, 2.0);
        assert!(
            close(x[0], 2.0, 1e-5) && x[1] == 0.0 && x[2] == 0.0 && x[3] == 1.0,
            "{x:?}"
        );
    }

    #[test]
    fn the_log_scale_is_one_below_the_floor_and_grows_after() {
        assert_eq!(log_scale(0), 1.0);
        assert_eq!(log_scale(8190), 1.0);
        assert_eq!(log_scale(8191), 1.0);
        assert!(
            close(log_scale(16383), 1.0 + 0.1 * 2f32.ln(), 1e-6),
            "{}",
            log_scale(16383)
        );
        assert!(close(log_scale(81919), 1.0 + 0.1 * 10f32.ln(), 1e-6));
    }

    #[test]
    fn the_relative_bias_reads_the_profile_and_is_zero_outside_the_extent() {
        let mut proj = vec![0f32; D_REL * REL_EXTENT];
        for d in 0..REL_EXTENT {
            proj[d] = d as f32; // profile 0 is the distance
            proj[REL_EXTENT + d] = 1.0; // profile 1 is flat
        }
        let mut states = vec![0f32; 2 * D_REL];
        states[0] = 2.0; // head 0 takes 2 times profile 0
        states[D_REL + 1] = -3.0; // head 1 takes -3 times profile 1
        let mut out = vec![0f32; 2 * REL_EXTENT];
        relative_logits(&states, &proj, &mut out);
        let (h0, h1) = out.split_at(REL_EXTENT);
        assert_eq!(relative_bias(h0, 0), 0.0);
        assert_eq!(relative_bias(h0, 5), 10.0);
        assert_eq!(relative_bias(h0, 1023), 2046.0);
        assert_eq!(relative_bias(h0, 1024), 0.0);
        assert_eq!(relative_bias(h0, -1), 0.0);
        assert_eq!(relative_bias(h1, 7), -3.0);
        assert_eq!(relative_bias(h1, 2000), 0.0);
    }

    #[test]
    fn the_sink_takes_weight_but_gives_no_value() {
        let q = [0f32; HEAD_DIM];
        let k = [0f32; HEAD_DIM];
        let v = [1f32; HEAD_DIM];
        let mut out = [0f32; HEAD_DIM];
        // All scores are 0. With a sink of 0, the one key gets half the weight.
        attend(&q, &[&k], &[&v], &[0.0], 0.0, &mut out);
        assert!(out.iter().all(|o| close(*o, 0.5, 1e-6)), "{}", out[0]);
        // A sink of ln(3) takes 3 of 4 parts.
        attend(&q, &[&k], &[&v], &[0.0], 3f32.ln(), &mut out);
        assert!(close(out[0], 0.25, 1e-6), "{}", out[0]);
        // A bias of ln(3) on the key gives it 3 of 4 parts against a sink of 0.
        attend(&q, &[&k], &[&v], &[3f32.ln()], 0.0, &mut out);
        assert!(close(out[0], 0.75, 1e-6), "{}", out[0]);
    }

    #[test]
    fn the_attention_scales_the_score_by_the_head_width() {
        let mut q = [0f32; HEAD_DIM];
        q[0] = (HEAD_DIM as f32).sqrt() * 2f32.ln();
        let mut k1 = [0f32; HEAD_DIM];
        k1[0] = 1.0;
        let k0 = [0f32; HEAD_DIM];
        let v0 = [0f32; HEAD_DIM];
        let v1 = [1f32; HEAD_DIM];
        let mut out = [0f32; HEAD_DIM];
        // Scores: ln 2 and 0, sink very low: weights 2/3 and 1/3.
        attend(&q, &[&k1, &k0], &[&v1, &v0], &[0.0, 0.0], -1e30, &mut out);
        assert!(close(out[0], 2.0 / 3.0, 1e-6), "{}", out[0]);
    }

    #[test]
    fn the_short_conv_weights_tap_3_on_the_current_row() {
        // Two channels. Channel 0 takes only tap 3, channel 1 only tap 0 (the oldest row).
        let weight = [0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let state = [10.0f32, 20.0, 11.0, 21.0, 12.0, 22.0];
        let x = [1.0f32, 2.0];
        let mut out = [0f32; 2];
        short_conv(&state, &x, &weight, &mut out);
        let s5 = 5f32.sqrt();
        assert!(close(out[0], 2.0 / s5, 1e-6), "{out:?}");
        assert!(close(out[1], 22.0 / s5, 1e-6), "{out:?}");
        let mut state = state;
        conv_push(&mut state, &x);
        assert_eq!(state, [11.0, 21.0, 12.0, 22.0, 1.0, 2.0]);
    }

    #[test]
    fn the_dynamic_conv_adds_the_taps_of_its_head_to_the_fixed_taps() {
        // Two heads of 128 channels; channel 0 is in head 0, channel 128 in head 1.
        let c = 2 * HEAD_DIM;
        let mut state = vec![0f32; 3 * c];
        state[0] = 5.0; // channel 0, lag 3 (oldest)
        state[2 * c] = 7.0; // channel 0, lag 1 (newest)
        state[2 * c + 128] = 3.0; // channel 128, lag 1
        let mut x = vec![0f32; c];
        x[0] = 1.0;
        x[128] = 2.0;
        let mut static_w = vec![0f32; CONV_TAPS * c];
        static_w[0] = 0.5; // lag 0 of channel 0
        static_w[c + 128] = 1.0; // lag 1 of channel 128
        let taps = [0.0f32, 1.0, 0.0, 2.0, 0.25, 0.0, 0.0, 0.0];
        let mut out = vec![0f32; c];
        dynamic_conv(&state, &x, &static_w, &taps, &mut out);
        let s9 = 3.0f32;
        // Channel 0: 1 * 0.5 + 7 * 1 + 5 * 2 = 17.5, plus x = 1.
        assert!(close(out[0], 18.5 / s9, 1e-6), "{}", out[0]);
        // Channel 128: 2 * 0.25 + 3 * 1 = 3.5, plus x = 2.
        assert!(close(out[128], 5.5 / s9, 1e-6), "{}", out[128]);
    }

    #[test]
    fn the_attention_residual_mixes_the_normed_rows_by_the_softmax_of_the_query() {
        let n = 4;
        let a = [2.0f32, 0.0, 0.0, 0.0]; // norm: [2, 0, 0, 0]
        let b = [0.0f32, 0.0, 0.0, 3.0]; // norm: [0, 0, 0, 2]
        // The query scores a with 2 * 2 / 2 = 2 and b with 0, so a takes e^2 / (e^2 + 1).
        let q = [2.0f32, 0.0, 0.0, 0.0];
        let zero = [0f32; 4];
        let mut out = vec![Vec::new(), Vec::new()];
        attention_residual(&[&a], &b, &[&q, &zero], &mut out);
        let p = 2f32.exp() / (2f32.exp() + 1.0);
        assert!(close(out[0][0], 2.0 * p, 1e-5), "{:?}", out[0]);
        assert!(close(out[0][3], 2.0 * (1.0 - p), 1e-5), "{:?}", out[0]);
        // A zero query takes the mean of the normed rows.
        assert!(
            close(out[1][0], 1.0, 1e-5) && close(out[1][3], 1.0, 1e-5),
            "{:?}",
            out[1]
        );
        assert_eq!(out[1].len(), n);
    }

    #[test]
    fn the_message_gate_mixes_the_row_and_the_message() {
        let n = 2;
        let rank = 1;
        // g = down . [norm h, norm m]; h = [1, 1] has norm [1, 1], m = [2, 2] has norm [1, 1]. g = 1.
        let down = [0.25f32; 4];
        // forget = tanh(0.5 g), gain = tanh(-1 g) for both channels.
        let forget = [0.5f32; 2];
        let gain = [-1.0f32; 2];
        let mut out = [0f32; 2];
        message_gate(&[1.0, 1.0], &[2.0, 2.0], &down, &forget, &gain, &mut out);
        let g = 1.0f32;
        let want = (1.0 - (0.5 * g).tanh()) + (-g).tanh() * 2.0;
        assert!(
            close(out[0], want, 1e-5) && close(out[1], want, 1e-5),
            "{out:?} want {want}"
        );
        assert_eq!(rank * n, 2);
    }

    #[test]
    fn the_swiglu_caps_the_gate_and_the_up_row_but_not_the_sigmoid() {
        let mut out = [0f32; 3];
        swiglu(&[0.0, 8.0, -2.0], &[1.0, 100.0, 50.0], &mut out);
        assert_eq!(out[0], 0.0);
        let want1 = 4.0 * 2f32.tanh() * sigmoid(8.0) * 25.0 * 4f32.tanh();
        assert!(close(out[1], want1, 1e-6), "{} {want1}", out[1]);
        let want2 = 4.0 * (-0.5f32).tanh() * sigmoid(-2.0) * 25.0 * 2f32.tanh();
        assert!(close(out[2], want2, 1e-6), "{} {want2}", out[2]);
    }

    #[test]
    fn the_router_selects_by_score_and_bias_and_weights_by_score() {
        let logits: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let mut bias = vec![0f32; 16];
        // The bias moves expert 0 into the selection, and the weight of expert 0 is still its own score.
        bias[0] = 10.0;
        let chosen = route(&logits, &bias);
        let ids: Vec<usize> = chosen.iter().map(|c| c.0).collect();
        assert_eq!(ids, [0, 9, 10, 11, 12, 13, 14, 15]);
        let scores: Vec<f32> = ids.iter().map(|e| sigmoid(logits[*e])).collect();
        let total: f32 = scores.iter().sum();
        for ((_, w), s) in chosen.iter().zip(&scores) {
            assert!(close(*w, s / total * 8.0, 1e-6), "{w} {s}");
        }
        let sum: f32 = chosen.iter().map(|c| c.1).sum();
        assert!(close(sum, 8.0, 1e-6), "{sum}");
    }

    #[test]
    fn the_expert_reads_the_gate_rows_then_the_up_rows() {
        // x of 1 value; 2 inner values. Gate rows [1, 2], up rows [3, 4]; down 1 x 2 = [1, 1].
        let x = [1.0f32];
        let gate_up = [1.0f32, 2.0, 3.0, 4.0];
        let down = [1.0f32, 1.0];
        let mut out = [0f32];
        expert(&x, &gate_up, &down, &mut out);
        let mut act = [0f32; 2];
        swiglu(&[1.0, 2.0], &[3.0, 4.0], &mut act);
        assert!(close(out[0], act[0] + act[1], 1e-6), "{} {}", out[0], act[0] + act[1]);
    }

    #[test]
    fn the_norms_follow_their_formulas() {
        let mut x = [3.0f32, 4.0];
        let mut out = [0f32; 2];
        rms_norm(&x, &[2.0, 1.0], &mut out);
        let f = 1.0 / (12.5f32 + EPS).sqrt();
        assert!(close(out[0], 6.0 * f, 1e-6) && close(out[1], 4.0 * f, 1e-6), "{out:?}");
        rms_norm_unweighted(&mut x);
        assert!(close(x[0], 3.0 * f, 1e-6), "{x:?}");
        assert!(close(softcap(1e6, 30.0), 30.0, 1e-6) && close(softcap(0.3, 30.0), 0.3, 1e-4));
        assert!(close(dot(&[1.0; 37], &[2.0; 37]), 74.0, 1e-7));
    }
}
