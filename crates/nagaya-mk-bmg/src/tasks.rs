//! Task descriptors for the ops of the speech binary, one function for each op (`kernels/qwen3tts_ops.cl`,
//! `kernels/megakernel.cl`). Each function puts its arguments in the fields that the op reads, so the field layout of
//! an op is in one place. Addresses are USM pointers of one card, as `u64`.

use crate::protocol::*;

fn task(opcode: u16) -> MkTaskDescriptor {
    MkTaskDescriptor {
        opcode,
        ..Default::default()
    }
}

/// `y[t, r] (+)= x[t] . w[r]` for `n` rows of `k` against `m` BF16 rows.
pub fn matmul(x: u64, w: u64, y: u64, n: u32, m: u32, k: u32, accumulate: bool) -> MkTaskDescriptor {
    MkTaskDescriptor {
        flags: if accumulate { FLAG_ACCUMULATE } else { 0 },
        a: x,
        b: w,
        c: y,
        m,
        n,
        k,
        ..task(OP_MATMUL)
    }
}

/// `y[t, r] = silu(x[t] . w[r]) * (x[t] . w[m + r])` for `m` gate rows followed by `m` up rows (BF16).
pub fn swiglu(x: u64, w: u64, y: u64, n: u32, m: u32, k: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: x,
        b: w,
        c: y,
        m,
        n,
        k,
        ..task(OP_GEGLU)
    }
}

/// RMSNorm of `n` rows of `k` with eps and BF16 weights (`w` 0: no weights), into `y`.
pub fn rmsnorm(x: u64, w: u64, y: u64, n: u32, k: u32, eps: f32, f32_weights: bool) -> MkTaskDescriptor {
    MkTaskDescriptor {
        flags: if f32_weights { FLAG_F32 } else { 0 },
        a: x,
        b: w,
        c: y,
        n,
        k,
        x: eps.to_bits() as u64,
        ..task(OP_RMSNORM)
    }
}

/// The norm and the RoPE of head rows in place: `n` rows of `d`, `heads` rows for each token, the first token at
/// position `p0`. `full` is Qwen3's RoPE (theta 1e6 over the head); else theta 1e4 over the head. `norm` false rotates
/// with no normalization. Each rotated row is then scaled by `scale`.
#[allow(clippy::too_many_arguments)]
pub fn qk_rope(
    x: u64,
    w: u64,
    n: u32,
    heads: u32,
    d: u32,
    p0: u32,
    full: bool,
    norm: bool,
    eps: f32,
    scale: f32,
) -> MkTaskDescriptor {
    let mut flags = FLAG_ROPE;
    if full {
        flags |= FLAG_ROPE_FULL;
    }
    if !norm {
        flags |= FLAG_NO_NORM;
    }
    MkTaskDescriptor {
        flags,
        a: x,
        b: w,
        c: x,
        m: heads,
        n,
        k: d,
        p0,
        x: eps.to_bits() as u64 | (scale.to_bits() as u64) << 32,
        ..task(OP_RMSNORM)
    }
}

/// One-token attention at position `p0` over the cache `kv` (K plane, then V plane after `max_seq` rows), then the
/// merge of its chunks into `out`. `window` 0 means no window.
#[allow(clippy::too_many_arguments)]
pub fn attend_one(
    q: u64,
    kv: u64,
    out: u64,
    heads: u32,
    kv_heads: u32,
    d: u32,
    p0: u32,
    max_seq: u32,
    window: u32,
) -> [MkTaskDescriptor; 2] {
    let x = (max_seq as u64) << 32 | window as u64;
    [
        MkTaskDescriptor {
            a: q,
            b: kv,
            m: heads | kv_heads << 16,
            k: d,
            p0,
            x,
            ..task(OP_ATTN)
        },
        MkTaskDescriptor {
            c: out,
            m: heads,
            k: d,
            p0,
            x,
            ..task(OP_ATTN_COMBINE)
        },
    ]
}

/// The floats of the score buffer of a block of `n` tokens at cache rows `p0..`: `heads` rows for each token.
pub fn score_floats(heads: u32, n: u32, p0: u32) -> usize {
    let row = (p0 + n).div_ceil(16) * 16 + 16;
    heads as usize * n as usize * row as usize
}

/// Attention of a block of `n` tokens at cache rows `p0..` (scores into `s`, softmax, the sum against V into `out`).
#[allow(clippy::too_many_arguments)]
pub fn attend_block(
    q: u64,
    kv: u64,
    s: u64,
    out: u64,
    n: u32,
    heads: u32,
    kv_heads: u32,
    d: u32,
    p0: u32,
    max_seq: u32,
    window: u32,
) -> [MkTaskDescriptor; 4] {
    let base = MkTaskDescriptor {
        m: heads | kv_heads << 16,
        n,
        k: d,
        p0,
        x: (max_seq as u64) << 32 | window as u64,
        ..Default::default()
    };
    [
        MkTaskDescriptor {
            opcode: OP_ATTN_SCORES,
            a: q,
            b: kv,
            c: s,
            ..base
        },
        MkTaskDescriptor {
            opcode: OP_ATTN_SOFTMAX,
            c: s,
            ..base
        },
        MkTaskDescriptor {
            opcode: OP_ATTN_PV,
            a: s,
            b: kv,
            c: out,
            ..base
        },
        MkTaskDescriptor {
            opcode: OP_ATTN_PV_SUM,
            c: out,
            ..base
        },
    ]
}

/// The sampler of one row of `vocab` logits (`OP_SAMPLE`): `state` is the sampler state, `out` gets the id.
#[allow(clippy::too_many_arguments)]
pub fn sample(
    logits: u64,
    state: u64,
    out: u64,
    vocab: u32,
    top_k: u32,
    temperature: f32,
    penalty: f32,
    suppress_from: u32,
    keep: u32,
    min_new: u32,
    history: bool,
) -> MkTaskDescriptor {
    MkTaskDescriptor {
        flags: if history { FLAG_HISTORY } else { 0 },
        a: logits,
        b: state,
        c: out,
        m: vocab,
        k: top_k,
        p0: suppress_from,
        layer: keep | min_new << 16,
        x: temperature.to_bits() as u64 | (penalty.to_bits() as u64) << 32,
        ..task(OP_SAMPLE)
    }
}

/// `y[t] (+)= table[ids[t * stride]]` for `n` rows of `k` (BF16 table, or F32).
#[allow(clippy::too_many_arguments)]
pub fn embed(
    ids: u64,
    table: u64,
    y: u64,
    n: u32,
    k: u32,
    stride: u32,
    f32_table: bool,
    accumulate: bool,
) -> MkTaskDescriptor {
    let mut flags = if f32_table { FLAG_F32 } else { 0 };
    if accumulate {
        flags |= FLAG_ACCUMULATE;
    }
    MkTaskDescriptor {
        flags,
        a: ids,
        b: table,
        c: y,
        m: stride,
        n,
        k,
        ..task(OP_EMBED)
    }
}

/// Zero the rows of `y` whose id is absent.
pub fn mask_rows(ids: u64, y: u64, n: u32, k: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: ids,
        c: y,
        n,
        k,
        ..task(OP_MASK_ROWS)
    }
}

/// `y[t, r] += bias[r]` (BF16, or F32).
pub fn bias(b: u64, y: u64, n: u32, k: u32, f32_bias: bool) -> MkTaskDescriptor {
    MkTaskDescriptor {
        flags: if f32_bias { FLAG_F32 } else { 0 },
        b,
        c: y,
        n,
        k,
        ..task(OP_BIAS)
    }
}

/// `y = silu(a) * b`, or `silu(a)` with `b` 0.
pub fn silu_mul(a: u64, b: u64, y: u64, n: u32, k: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a,
        b,
        c: y,
        n,
        k,
        ..task(OP_SILU_MUL)
    }
}

/// A causal convolution with F32 weights over `steps` input rows of `cin`, the first `skip` rows state.
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    x: u64,
    w: u64,
    b: u64,
    y: u64,
    steps: u32,
    cin: u32,
    cout: u32,
    taps: u32,
    dilation: u32,
    groups: u32,
    skip: u32,
    accumulate: bool,
) -> MkTaskDescriptor {
    MkTaskDescriptor {
        flags: if accumulate { FLAG_ACCUMULATE } else { 0 },
        a: x,
        b: w,
        c: y,
        x: b,
        m: cout,
        n: steps,
        k: cin,
        p0: groups,
        layer: taps | dilation << 8 | skip << 16,
        ..task(OP_CONV1D)
    }
}

/// A transposed convolution with F32 weights, stride `stride`, over `steps` input rows of `cin`, the first `skip`
/// rows state.
#[allow(clippy::too_many_arguments)]
pub fn convt1d(
    x: u64,
    w: u64,
    b: u64,
    y: u64,
    steps: u32,
    cin: u32,
    cout: u32,
    taps: u32,
    stride: u32,
    skip: u32,
) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: x,
        b: w,
        c: y,
        x: b,
        m: cout,
        n: steps,
        k: cin,
        layer: taps | stride << 8 | skip << 16,
        ..task(OP_CONVT1D)
    }
}

/// LayerNorm of `n` rows of `k` with F32 weights and bias.
pub fn layernorm(x: u64, w: u64, b: u64, y: u64, n: u32, k: u32, eps: f32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: x,
        b: w,
        c: y,
        x: b,
        n,
        k,
        p0: eps.to_bits(),
        ..task(OP_LAYERNORM)
    }
}

/// `y = gelu(x)` with erf.
pub fn gelu(x: u64, y: u64, n: u32, k: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: x,
        c: y,
        n,
        k,
        ..task(OP_GELU)
    }
}

/// SnakeBeta with F32 alpha and beta of each channel.
pub fn snake(x: u64, alpha: u64, beta: u64, y: u64, n: u32, k: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: x,
        b: alpha,
        c: y,
        x: beta,
        n,
        k,
        ..task(OP_SNAKE)
    }
}

/// `y[t, j] += g[j] * x[t, j]` with an F32 `g`.
pub fn scale_add(x: u64, g: u64, y: u64, n: u32, k: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: x,
        b: g,
        c: y,
        n,
        k,
        ..task(OP_SCALE_ADD)
    }
}

/// `n` 32-bit words from `a` to `c`.
pub fn copy(a: u64, c: u64, words: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a,
        c,
        n: words,
        ..task(OP_COPY)
    }
}

/// `n` signed 16-bit samples from `n` floats.
pub fn pcm(x: u64, y: u64, n: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: x,
        c: y,
        n,
        ..task(OP_PCM)
    }
}
