//! Task descriptors for the ops of the speech and image binaries, one function for each op (`kernels/qwen3tts_ops.cl`,
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

// The ops of the image binary (`kernels/sdxl_ops.cl`).

/// The dtype flag of the weights of an image op: F16 (the UNet and the text encoders) or F32 (the VAE).
pub fn weight_flag(f32_weights: bool) -> u16 {
    if f32_weights { FLAG_F32 } else { FLAG_F16 }
}

/// A 2D convolution over `images` images of `h` by `w` pixels, pixel-major, with weights `[cout][cin][k][k]` and a
/// bias (0: none). A 1 x 1 kernel is a linear layer over rows: pass the rows as `h` and 1 as `w`.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    x: u64,
    w: u64,
    b: u64,
    y: u64,
    images: u32,
    h: u32,
    w_px: u32,
    cin: u32,
    cout: u32,
    kernel: u32,
    stride: u32,
    pad: u32,
    f32_weights: bool,
    accumulate: bool,
) -> MkTaskDescriptor {
    assert!(
        h < 1 << 16 && w_px < 1 << 16,
        "a conv2d input is at most 65535 by 65535"
    );
    let mut flags = weight_flag(f32_weights);
    if accumulate {
        flags |= FLAG_ACCUMULATE;
    }
    MkTaskDescriptor {
        flags,
        a: x,
        b: w,
        c: y,
        x: b,
        n: h | w_px << 16,
        m: cout,
        k: cin,
        p0: images,
        layer: kernel | kernel << 8 | stride << 16 | pad << 24,
        ..task(OP_CONV2D)
    }
}

/// A linear layer over `rows` rows: weights `[cout][cin]` and a bias (0: none). Each row is an image of one pixel, so
/// the count of rows has 32 bits.
#[allow(clippy::too_many_arguments)]
pub fn linear(
    x: u64,
    w: u64,
    b: u64,
    y: u64,
    rows: u32,
    cin: u32,
    cout: u32,
    f32_weights: bool,
    accumulate: bool,
) -> MkTaskDescriptor {
    conv2d(x, w, b, y, rows, 1, 1, cin, cout, 1, 1, 0, f32_weights, accumulate)
}

/// GroupNorm of `images` images of `pixels` rows of `k` channels in `groups` groups, with an optional silu.
#[allow(clippy::too_many_arguments)]
pub fn groupnorm(
    x: u64,
    w: u64,
    b: u64,
    y: u64,
    images: u32,
    pixels: u32,
    k: u32,
    groups: u32,
    eps: f32,
    f32_weights: bool,
    silu: bool,
) -> MkTaskDescriptor {
    let mut flags = weight_flag(f32_weights);
    if silu {
        flags |= FLAG_SILU;
    }
    MkTaskDescriptor {
        flags,
        a: x,
        b: w,
        c: y,
        x: b,
        n: pixels,
        k,
        m: images,
        layer: groups,
        p0: eps.to_bits(),
        ..task(OP_GROUPNORM)
    }
}

/// Attention with no mask over `images` images: `nq` query rows and `nk` key rows of `heads * d` values.
#[allow(clippy::too_many_arguments)]
pub fn sdpa(
    q: u64,
    k: u64,
    v: u64,
    y: u64,
    images: u32,
    nq: u32,
    nk: u32,
    heads: u32,
    d: u32,
    causal: bool,
) -> MkTaskDescriptor {
    MkTaskDescriptor {
        flags: if causal { FLAG_CAUSAL } else { 0 },
        a: q,
        b: k,
        x: v,
        c: y,
        n: nq,
        m: nk,
        k: heads * d,
        p0: d,
        layer: images,
        ..task(OP_SDPA)
    }
}

/// The nearest up sampler of `images` images of `h` by `w` pixels of `k` channels.
pub fn upsample2x(x: u64, y: u64, images: u32, h: u32, w: u32, k: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: x,
        c: y,
        n: h | w << 16,
        k,
        layer: images,
        ..task(OP_UPSAMPLE2X)
    }
}

/// `y = x * sigmoid(1.702 x)` for `n` rows of `k`.
pub fn quick_gelu(x: u64, y: u64, n: u32, k: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: x,
        c: y,
        n,
        k,
        ..task(OP_QUICK_GELU)
    }
}

/// `y[t, j] = x[t, j] * gelu(x[t, k + j])` for `n` rows: `x` has rows of `2k`, `y` rows of `k`.
pub fn geglu_half(x: u64, y: u64, n: u32, k: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: x,
        c: y,
        n,
        k,
        ..task(OP_GEGLU_HALF)
    }
}

/// `y[t, offset + j] = x[t, j]` for `n` rows: `x` has rows of `k`, `y` rows of `m`.
pub fn copy_cols(x: u64, y: u64, n: u32, k: u32, m: u32, offset: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: x,
        c: y,
        n,
        k,
        m,
        p0: offset,
        ..task(OP_COPY_COLS)
    }
}

/// `y[b, p, j] += rows[b, j]` for `images` images of `pixels` rows of `k`.
pub fn add_image_row(rows: u64, y: u64, images: u32, pixels: u32, k: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: rows,
        c: y,
        n: pixels,
        k,
        layer: images,
        ..task(OP_ADD_IMAGE_ROW)
    }
}

/// The weight planes of `OP_GEMM` for weights `[cout][cin][k][k]` (or `[cout][cin]` with `kernel` 1): the taps
/// relaid as `[cout][tap][cin]`, then split into `planes` BF16 planes whose sum is the weight. Each plane takes the
/// top 16 bits of what the planes before it left, and the last one rounds, as `bf16_split3` of the kernel does. Two
/// planes hold an F16 weight exactly; three hold an F32 weight to 24 bits.
pub fn gemm_planes(weights: &[f32], cout: usize, cin: usize, kernel: usize, planes: usize) -> Vec<u16> {
    let taps = kernel * kernel;
    assert_eq!(weights.len(), cout * cin * taps, "the weights are [cout][cin][k][k]");
    assert!((2..=3).contains(&planes), "a weight has 2 or 3 planes");
    let k = taps * cin;
    let mut out = vec![0u16; planes * cout * k];
    for o in 0..cout {
        for tap in 0..taps {
            for i in 0..cin {
                let mut rest = weights[(o * cin + i) * taps + tap];
                let at = o * k + tap * cin + i;
                for p in 0..planes {
                    let bits = rest.to_bits();
                    let top = if p + 1 == planes {
                        // The last plane rounds to the nearest even.
                        bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000
                    } else {
                        bits & 0xFFFF_0000
                    };
                    out[p * cout * k + at] = (top >> 16) as u16;
                    rest -= f32::from_bits(top);
                }
            }
        }
    }
    out
}

/// A convolution or a linear layer on the matrix units: `x` pixel-major with `cin` inputs, the planes of
/// [`gemm_planes`], a bias (0: none; F32, or F16 with `f16_bias`), over `images` images of `h` by `w`.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    x: u64,
    planes_at: u64,
    bias: u64,
    y: u64,
    images: u32,
    h: u32,
    w: u32,
    cin: u32,
    cout: u32,
    (kernel, stride, pad): (u32, u32, u32),
    planes: u32,
    f16_bias: bool,
    accumulate: bool,
) -> MkTaskDescriptor {
    assert!(
        cin.is_multiple_of(64) && cout.is_multiple_of(16),
        "OP_GEMM takes inputs of 64s and outputs of 16s"
    );
    assert!(h < 1 << 16 && w < 1 << 16, "an OP_GEMM input is at most 65535 by 65535");
    let mut flags = if f16_bias { FLAG_F16 } else { FLAG_F32 };
    if accumulate {
        flags |= FLAG_ACCUMULATE;
    }
    MkTaskDescriptor {
        flags,
        a: x,
        b: planes_at,
        c: y,
        x: bias,
        n: h | w << 16,
        m: cout,
        k: cin,
        p0: images,
        layer: kernel | stride << 8 | pad << 16 | planes << 24,
        ..task(OP_GEMM)
    }
}

/// The ushorts of the planes of [`attn_prep`] for `images` images of `nk` key rows of `heads * d` values.
pub fn attn_planes_len(images: usize, nk: usize, heads: usize, d: usize) -> usize {
    images * heads * 6 * nk.div_ceil(16) * 16 * d
}

/// Split the keys and the values of an attention into the BF16 planes of [`attn_xmx`].
pub fn attn_prep(k: u64, v: u64, planes: u64, images: u32, nk: u32, heads: u32, d: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: k,
        x: v,
        c: planes,
        n: nk,
        k: heads * d,
        p0: d,
        layer: images,
        ..task(OP_ATTN_PREP)
    }
}

/// Attention with no mask on the matrix units, from the planes of [`attn_prep`]. `d` is 16, 32, 48 or 64.
pub fn attn_xmx(q: u64, planes: u64, y: u64, images: u32, nq: u32, nk: u32, heads: u32, d: u32) -> MkTaskDescriptor {
    assert!(
        d.is_multiple_of(16) && d <= 64,
        "OP_ATTN_XMX takes heads of 16, 32, 48 or 64 values"
    );
    MkTaskDescriptor {
        a: q,
        b: planes,
        c: y,
        n: nq,
        m: nk,
        k: heads * d,
        p0: d,
        layer: images,
        ..task(OP_ATTN_XMX)
    }
}

// The ops of the embedding binary (`kernels/embeddinggemma2_ops.cl`).

/// Attention in both directions over `n` tokens: `heads` query heads share `kv_heads` key and value heads of `d`
/// values. `window` 0 gives every key; else the keys within `window` positions of the query, both sides. `scale`
/// multiplies each score.
#[allow(clippy::too_many_arguments)]
pub fn biattn(
    q: u64,
    k: u64,
    v: u64,
    y: u64,
    n: u32,
    heads: u32,
    kv_heads: u32,
    d: u32,
    window: u32,
    scale: f32,
) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: q,
        b: k,
        x: v,
        c: y,
        n,
        m: heads | kv_heads << 16,
        k: window,
        p0: d,
        layer: scale.to_bits(),
        ..task(OP_BIATTN)
    }
}

/// `y[t, r] = gelu_tanh(x[t] . w[r]) * (x[t] . w[m + r])` for `m` gate rows followed by `m` up rows (BF16). It is
/// `OP_GEGLU` of the embedding binary, whose GLU is GeGLU.
pub fn geglu(x: u64, w: u64, y: u64, n: u32, m: u32, k: u32) -> MkTaskDescriptor {
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

/// `y[t, j] = gelu_tanh(a[t, j]) * b[t * stride + j]` for `n` rows of `k`; `stride` 0 means `k`.
pub fn gelu_mul(a: u64, b: u64, y: u64, n: u32, k: u32, stride: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a,
        b,
        c: y,
        n,
        k,
        m: stride,
        ..task(OP_GELU_MUL)
    }
}

/// `y = x * s` for `n` rows of `k`; `x` may be `y`.
pub fn scale(x: u64, y: u64, n: u32, k: u32, s: f32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: x,
        c: y,
        n,
        k,
        p0: s.to_bits(),
        ..task(OP_SCALE)
    }
}
