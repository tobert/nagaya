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

/// The key segments of `OP_ATTN_PV` for `n` tokens of `heads` heads of `d` values over `key_rows` keys, as
/// `attn_pv_segments` in `kernels/megakernel.cl` counts them. The count reads no slot size, so each slot size gives
/// the same bits for the same feed (spec 6.7).
pub fn attn_pv_segments(heads: u32, n: u32, d: u32, key_rows: u32) -> u32 {
    let items = heads * n.div_ceil(16) * (d / 32);
    let segs = 4096u32.div_ceil(items);
    segs.min((key_rows.div_ceil(16) / 2).max(1)).max(1)
}

/// The floats of scratch that `OP_ATTN_PV` writes for a block of up to `max_block` tokens: each segment holds the
/// partial sums of the block. The count of segments is at most `4096 / items + 1`, and the items are at least
/// `n * heads * d / 512`. Thus the partial sums take at most `2^21 + n * heads * d` floats.
pub fn attn_pv_scratch_floats(max_block: usize, heads: usize, d: usize) -> usize {
    (1 << 21) + max_block * heads * d
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

/// The keys of one round of `OP_ATTN_STREAM` (`ATTN_STREAM_KEYS`, megakernel.cl).
pub const ATTN_STREAM_KEYS: u32 = 128;

/// The most rounds in one chunk of `OP_ATTN_STREAM`, 8,192 keys (`ATTN_STREAM_ROUNDS`, megakernel.cl).
pub const ATTN_STREAM_ROUNDS: u32 = 64;

/// The floats of scratch of one team of `OP_ATTN_STREAM` (`ATTN_STREAM_TEAM_FLOATS`, megakernel.cl): the
/// probabilities of one round for 16 tokens, then a max and a sum of each token for each of the 4 subgroups.
pub const ATTN_STREAM_TEAM_FLOATS: usize = 16 * ATTN_STREAM_KEYS as usize + 2 * 4 * 16;

/// The rounds in one chunk of `OP_ATTN_STREAM`, as `attn_stream_rounds` in `kernels/megakernel.cl` counts them. A
/// block of `n` tokens of `heads` heads reads a span of `span_keys` keys, a multiple of 128. A short span has small
/// chunks, so a small block still has about 1,024 work items, two for each team. The count reads no slot size, so
/// each slot size gives the same bits for the same feed (spec 6.7).
pub fn attn_stream_rounds(heads: u32, n: u32, span_keys: u32) -> u32 {
    let items = heads * n.div_ceil(16);
    let want = 1024u32.div_ceil(items);
    (span_keys.div_ceil(ATTN_STREAM_KEYS) / want).clamp(1, ATTN_STREAM_ROUNDS)
}

/// The chunks of `OP_ATTN_STREAM` for the same block: each chunk holds `attn_stream_rounds` rounds.
pub fn attn_stream_chunks(heads: u32, n: u32, span_keys: u32) -> u32 {
    span_keys
        .div_ceil(ATTN_STREAM_KEYS)
        .div_ceil(attn_stream_rounds(heads, n, span_keys))
}

/// The floats of scratch that `OP_ATTN_STREAM` uses for a block of up to `max_block` tokens in a slot of `max_seq`
/// positions. It writes two things:
///
/// - The partial sums: a max, a sum and an output of `d` values for each chunk, token and head. The chunks are at
///   most `2 * (1024 / items + 1) + 2`, or one for each 8,192 keys of the slot. The items are at least
///   `n * heads / 16`.
/// - The scratch of each team: a team is 4 subgroups of the launch.
pub fn attn_stream_scratch_floats(max_block: usize, heads: usize, d: usize, max_seq: usize) -> usize {
    let rounds = max_seq.div_ceil(ATTN_STREAM_KEYS as usize) + 1;
    let rows = max_block * heads;
    let chunks_of_a_long_span = rounds / ATTN_STREAM_ROUNDS as usize + 3;
    let partial = (d + 2) * (2 * (16384 + rows + 15 * heads) + rows * chunks_of_a_long_span);
    partial.next_multiple_of(16) + crate::protocol::MEGAKERNEL_SUBGROUPS / 4 * ATTN_STREAM_TEAM_FLOATS
}

/// Attention of a block of `n` tokens at cache rows `p0..` with no score buffer (`OP_ATTN_STREAM`), then the merge of
/// its chunks into `out`. The fields are those of `attend_block`. The result has other bits than `attend_block` has.
#[allow(clippy::too_many_arguments)]
pub fn attend_stream(
    q: u64,
    kv: u64,
    out: u64,
    n: u32,
    heads: u32,
    kv_heads: u32,
    d: u32,
    p0: u32,
    max_seq: u32,
    window: u32,
) -> [MkTaskDescriptor; 2] {
    assert!(
        d == 128 && heads.is_multiple_of(kv_heads),
        "OP_ATTN_STREAM takes heads of 128 values, and a whole count of heads for each KV head"
    );
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
            opcode: OP_ATTN_STREAM,
            a: q,
            b: kv,
            ..base
        },
        MkTaskDescriptor {
            opcode: OP_ATTN_STREAM_SUM,
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

/// The convolution `task` with its pad at the bottom and the right only. diffusers' down sampler of the VAE encoder
/// pads one row and one column at the end, then runs a 3 x 3 kernel with stride 2 (`downsampling.py:139-145`).
pub fn pad_end(mut task: MkTaskDescriptor) -> MkTaskDescriptor {
    assert!(
        task.opcode == OP_CONV2D || task.opcode == OP_GEMM,
        "FLAG_PAD_END marks OP_CONV2D or OP_GEMM, not op {}",
        task.opcode
    );
    task.flags |= FLAG_PAD_END;
    task
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
    // Each of 16 lanes keeps at most 32 values of a head, and a head shorter than its 16s leaves values unwritten.
    assert!(
        (16..=512).contains(&d) && d.is_multiple_of(16),
        "OP_BIATTN takes a head of 16 to 512 values in 16s, and it got {d}"
    );
    // The kernel reads the bits 0 as no scale.
    assert!(
        scale != 0.0,
        "OP_BIATTN reads a scale of 0.0 as 1.0, so it takes no scale of 0.0"
    );
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

// The ops of the LFM3 binary (`kernels/lfm3_ops.cl`).

/// The gated short convolution of LFM3 over `n` positions of `d` channels. `bcx` holds the in_proj rows `[n][3d]` in
/// the chunks B, C and x. `w` holds the weights `[d][taps]` (BF16, or F32). `state` holds the last `taps - 1` inputs
/// B * x `[taps - 1][d]`, and the op writes the new ones into it. `y` gets `[n][d]`.
#[allow(clippy::too_many_arguments)]
pub fn short_conv(
    bcx: u64,
    w: u64,
    state: u64,
    y: u64,
    n: u32,
    d: u32,
    taps: u32,
    f32_weights: bool,
) -> MkTaskDescriptor {
    assert!(n >= 1, "OP_SHORT_CONV takes at least one position");
    assert!(
        d >= 16 && d.is_multiple_of(16),
        "OP_SHORT_CONV takes channels in 16s, and it got {d}"
    );
    // The kernel keeps the window in an array of 8 values (SHORT_CONV_MAX_TAPS).
    assert!(
        (2..=8).contains(&taps),
        "OP_SHORT_CONV takes 2 to 8 taps, and it got {taps}"
    );
    // The op reads the rows and the state of each channel while it writes the output and the state.
    let span = |at: u64, floats: u32| at..at + 4 * floats as u64;
    let (rows, held, out) = (span(bcx, n * 3 * d), span(state, (taps - 1) * d), span(y, n * d));
    for (a, b, what) in [
        (&out, &rows, "the output overlaps the in_proj rows"),
        (&out, &held, "the output overlaps the state"),
        (&held, &rows, "the state overlaps the in_proj rows"),
    ] {
        assert!(a.end <= b.start || b.end <= a.start, "OP_SHORT_CONV: {what}");
    }
    MkTaskDescriptor {
        flags: if f32_weights { FLAG_F32 } else { 0 },
        a: bcx,
        b: w,
        c: y,
        x: state,
        n,
        k: d,
        layer: taps,
        ..task(OP_SHORT_CONV)
    }
}

/// The rotation `task` of [`qk_rope`] with only the first `pairs` pairs (i, i + d / 2) turned, as the full attention
/// layers of LFM3 do. Their frequencies follow the whole head, and the other pairs keep their values
/// (modeling_lfm3.py:105-111). Only the LFM3 binary reads the pairs.
pub fn partial_rotation(mut task: MkTaskDescriptor, pairs: u32) -> MkTaskDescriptor {
    assert!(
        task.opcode == OP_RMSNORM && task.flags & FLAG_ROPE != 0,
        "partial_rotation marks an OP_RMSNORM with FLAG_ROPE, not op {} with the flags {:#x}",
        task.opcode,
        task.flags
    );
    // The kernel reads 0 as the count that the flags give.
    let half = task.k / 2;
    assert!(
        (1..=half).contains(&pairs),
        "a head of {} values has 1 to {half} pairs, and partial_rotation got {pairs}",
        task.k
    );
    task.layer = pairs;
    task
}

// The ops of the audio encoder of LFM3-Omni-2B (`kernels/lfm3_audio_ops.cl`).

/// The time rows and the mel rows of the output of [`subsample`] for an input of `n` by `k`: half of each, rounded up.
pub fn subsample_out(n: u32, k: u32) -> (u32, u32) {
    (n.div_ceil(2), k.div_ceil(2))
}

/// One 3 x 3 convolution with a stride of 2 and a pad of 1 of the audio downsampler, channel-last. `x` holds
/// `[n][k][cin]`: `n` time rows of `k` mel rows. `w` holds the BF16 taps `[m][3][3]` (mel, then time), and `bias` the
/// BF16 bias `[m]`. With `cin` 1, each of the `m` channels reads channel 0; with `cin` equal to `m`, the convolution
/// is depthwise. `y` gets `[ceil(n / 2)][ceil(k / 2)][m]`, with silu when `silu` is true.
#[allow(clippy::too_many_arguments)]
pub fn subsample(x: u64, w: u64, bias: u64, y: u64, n: u32, k: u32, cin: u32, m: u32, silu: bool) -> MkTaskDescriptor {
    assert!(
        n >= 1 && k >= 1,
        "OP_SUBSAMPLE takes at least one time row and one mel row, and it got {n} by {k}"
    );
    assert!(
        m >= 1 && (cin == 1 || cin == m),
        "OP_SUBSAMPLE reads 1 channel or {m} channels, and it got {cin}"
    );
    let (to, ho) = subsample_out(n, k);
    let input = x..x + 4 * (n as u64 * k as u64 * cin as u64);
    let output = y..y + 4 * (to as u64 * ho as u64 * m as u64);
    assert!(
        output.end <= input.start || input.end <= output.start,
        "OP_SUBSAMPLE: the output overlaps the input"
    );
    MkTaskDescriptor {
        a: x,
        b: w,
        c: y,
        x: bias,
        n,
        k,
        m,
        p0: cin,
        layer: silu as u32,
        ..task(OP_SUBSAMPLE)
    }
}

/// One control item on `n` rows of `k` values of the residual at `h`, in place (`OP_CONTROL`, engine-abi "Control
/// vectors"). `v` is the vector of the layer, `k` F32 values. A vector of zeros leaves each row as it is.
pub fn control(h: u64, v: u64, n: u32, k: u32, op: ControlOp, strength: f32) -> MkTaskDescriptor {
    assert!(
        strength.is_finite(),
        "a control strength is finite, and it got {strength}"
    );
    assert!(
        strength != 0.0 || op == ControlOp::Clamp,
        "an add or an ablate with strength 0 changes nothing; the engine skips the item"
    );
    MkTaskDescriptor {
        a: h,
        b: v,
        n,
        k,
        p0: op as u32,
        m: strength.to_bits(),
        ..task(OP_CONTROL)
    }
}

/// One control item of the Inkling 2 binary (`OP_INKLING2_CONTROL`): the task of [`control`] under the opcode of that
/// binary, which does not hold `OP_CONTROL`.
pub fn inkling2_control(h: u64, v: u64, n: u32, k: u32, op: ControlOp, strength: f32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        opcode: OP_INKLING2_CONTROL,
        ..control(h, v, n, k, op, strength)
    }
}

/// The sigmoid gate of each head: `y[t, h, j] = a[t, h, j] * sigmoid(gates[t * stride + h])` for `n` positions of
/// `heads` heads of `d` values. `a` may be `y`.
pub fn head_gate(a: u64, gates: u64, y: u64, n: u32, heads: u32, d: u32, stride: u32) -> MkTaskDescriptor {
    // The kernel reads a stride of 0 as `heads`.
    assert!(
        stride >= heads,
        "OP_HEAD_GATE reads rows of at least {heads} gates, and it got a stride of {stride}"
    );
    MkTaskDescriptor {
        a,
        b: gates,
        c: y,
        n,
        m: heads,
        k: d,
        p0: stride,
        ..task(OP_HEAD_GATE)
    }
}

/// Attention in both directions inside groups of `group` rows (`OP_LFM3_GROUP_ATTN`): `heads` query heads share
/// `kv_heads` key and value heads of `d` values, and each score is times `scale`. The depth decoder of LFM3 attends
/// over the 12 codebooks of each frame.
#[allow(clippy::too_many_arguments)]
pub fn group_attn(
    q: u64,
    k: u64,
    v: u64,
    y: u64,
    n: u32,
    heads: u32,
    kv_heads: u32,
    d: u32,
    group: u32,
    scale: f32,
) -> MkTaskDescriptor {
    // Each of 16 lanes keeps at most 32 values of a head, and a head shorter than its 16s leaves values unwritten.
    assert!(
        (16..=512).contains(&d) && d.is_multiple_of(16),
        "OP_LFM3_GROUP_ATTN takes a head of 16 to 512 values in 16s, and it got {d}"
    );
    assert!(
        kv_heads > 0 && heads.is_multiple_of(kv_heads) && heads < 1 << 16 && kv_heads < 1 << 16,
        "OP_LFM3_GROUP_ATTN takes query heads in groups of each key head, and it got {heads} and {kv_heads}"
    );
    // The kernel reads a group of 0 as every row, and the bits 0 as no scale.
    assert!(
        group >= 1 && n.is_multiple_of(group),
        "OP_LFM3_GROUP_ATTN takes whole groups, and it got {n} rows in groups of {group}"
    );
    assert!(
        scale != 0.0,
        "OP_LFM3_GROUP_ATTN reads a scale of 0.0 as 1.0, so it takes no scale of 0.0"
    );
    MkTaskDescriptor {
        a: q,
        b: k,
        x: v,
        c: y,
        n,
        m: heads | kv_heads << 16,
        k: group,
        p0: d,
        layer: scale.to_bits(),
        ..task(OP_LFM3_GROUP_ATTN)
    }
}

/// The complex bins of `n` rows of `stride` values (`OP_LFM3_POLAR`): `bins` log magnitudes, then `bins` angles, into
/// `[n][2 * bins]` of real parts, then imaginary parts.
pub fn polar(a: u64, c: u64, n: u32, bins: u32, stride: u32) -> MkTaskDescriptor {
    assert!(
        bins >= 1 && stride >= 2 * bins,
        "OP_LFM3_POLAR reads {bins} magnitudes and {bins} angles, and a row of {stride} values does not hold them"
    );
    MkTaskDescriptor {
        a,
        c,
        n,
        k: bins,
        m: stride,
        ..task(OP_LFM3_POLAR)
    }
}

/// The inverse real FFT of `n_fft` samples of each of `n` rows of bins, times the F32 window (`OP_LFM3_IRFFT`). The
/// rows hold `n_fft / 2 + 1` real parts, then as many imaginary parts. `table` is [`irfft_table`] of `n_fft`.
pub fn irfft(bins: u64, window: u64, table: u64, c: u64, n: u32, n_fft: u32) -> MkTaskDescriptor {
    assert!(
        n_fft >= 4 && n_fft.is_multiple_of(2),
        "OP_LFM3_IRFFT takes an even FFT size of at least 4, and it got {n_fft}"
    );
    MkTaskDescriptor {
        a: bins,
        b: window,
        x: table,
        c,
        n,
        k: n_fft,
        p0: n_fft / 2 + 1,
        ..task(OP_LFM3_IRFFT)
    }
}

/// The table of `OP_LFM3_IRFFT`: the cosine and the sine of `2 pi j / n_fft` for each `j < n_fft`, in f64 and then
/// rounded to f32.
pub fn irfft_table(n_fft: usize) -> Vec<f32> {
    (0..n_fft)
        .flat_map(|j| {
            let a = 2.0 * std::f64::consts::PI * j as f64 / n_fft as f64;
            [a.cos() as f32, a.sin() as f32]
        })
        .collect()
}

// The ops of the LFM3 vision tower (`kernels/lfm3_vision_ops.cl`).

/// Check a grid of patches of the vision tower: each side even and below 65,536, so the merge takes 2 by 2 patches and
/// the kernel reads each side from 16 bits. Returns the patches of one frame.
fn vision_frame(op: &str, grid_h: u32, grid_w: u32) -> u32 {
    assert!(
        grid_h >= 2 && grid_w >= 2 && grid_h.is_multiple_of(2) && grid_w.is_multiple_of(2),
        "{op} takes a grid with even sides, and it got {grid_h} by {grid_w}"
    );
    assert!(
        grid_h < 1 << 16 && grid_w < 1 << 16,
        "{op} takes a grid of at most 65535 by 65535"
    );
    grid_h * grid_w
}

/// The norm and the axial rotation of the head rows of the vision attention, in place: `rows` rows of `d` values,
/// `heads` rows for each patch, with the BF16 norm weights `w`. The patches are whole frames of `grid_h` by `grid_w`
/// in merge-block order. The first quarter of the pairs turns by the down coordinate, the rest by the across
/// coordinate, with the frequencies `theta^(-2j / (d / 2))`.
#[allow(clippy::too_many_arguments)]
pub fn vision_rope(
    x: u64,
    w: u64,
    rows: u32,
    heads: u32,
    d: u32,
    grid_h: u32,
    grid_w: u32,
    eps: f32,
    theta: f32,
) -> MkTaskDescriptor {
    let frame = vision_frame("OP_VISION_ROPE", grid_h, grid_w);
    assert!(
        d >= 4 && d.is_multiple_of(4),
        "OP_VISION_ROPE takes a head of a multiple of 4 values, and it got {d}"
    );
    assert!(
        heads >= 1 && rows >= 1 && rows.is_multiple_of(heads * frame),
        "OP_VISION_ROPE takes whole frames of {frame} patches of {heads} heads, and it got {rows} rows"
    );
    MkTaskDescriptor {
        a: x,
        b: w,
        c: x,
        n: rows,
        m: heads,
        k: d,
        p0: grid_w | grid_h << 16,
        x: eps.to_bits() as u64 | (theta.to_bits() as u64) << 32,
        ..task(OP_VISION_ROPE)
    }
}

/// The attention of the vision tower: `tasks::biattn` with every key, with the same bits. `q` holds the query rows
/// `[n][heads * d]`, `k` and `v` hold the key rows and the value rows `[n][kv_heads * d]`, and `y` gets `[n][heads * d]`.
/// A subgroup takes a tile of query rows of one head, so it reads each key row one time for the tile.
#[allow(clippy::too_many_arguments)]
pub fn vision_attn(
    q: u64,
    k: u64,
    v: u64,
    y: u64,
    n: u32,
    heads: u32,
    kv_heads: u32,
    d: u32,
    scale: f32,
) -> MkTaskDescriptor {
    // Each of 16 lanes keeps 4 values of a head for each query row of its tile.
    assert!(
        (16..=64).contains(&d) && d.is_multiple_of(16),
        "OP_VISION_ATTN takes a head of 16 to 64 values in 16s, and it got {d}"
    );
    // The kernel reads the bits 0 as no scale.
    assert!(
        scale != 0.0,
        "OP_VISION_ATTN reads a scale of 0.0 as 1.0, so it takes no scale of 0.0"
    );
    assert!(n > 0, "OP_VISION_ATTN takes at least one row");
    MkTaskDescriptor {
        a: q,
        b: k,
        x: v,
        c: y,
        n,
        m: heads | kv_heads << 16,
        p0: d,
        layer: scale.to_bits(),
        ..task(OP_VISION_ATTN)
    }
}

/// The gated depthwise 3 by 3 convolution of a vision conv layer over `n` patch rows of `d` channels: `bcx` holds the
/// rows of in_proj `[n][3d]` in the chunks B, C and x, `w` holds the BF16 taps `[d][9]`, and `y` gets
/// `C * conv(B * x)` `[n][d]`. The rows are whole frames of `grid_h` by `grid_w` in merge-block order.
pub fn vision_conv(bcx: u64, w: u64, y: u64, n: u32, d: u32, grid_h: u32, grid_w: u32) -> MkTaskDescriptor {
    let frame = vision_frame("OP_VISION_CONV", grid_h, grid_w);
    assert!(
        d >= 16 && d.is_multiple_of(16),
        "OP_VISION_CONV takes channels in 16s, and it got {d}"
    );
    assert!(
        n >= 1 && n.is_multiple_of(frame),
        "OP_VISION_CONV takes whole frames of {frame} patches, and it got {n} rows"
    );
    // A lane reads the rows of the neighbours while other lanes write the output.
    let rows = bcx..bcx + 4 * n as u64 * 3 * d as u64;
    let out = y..y + 4 * n as u64 * d as u64;
    assert!(
        out.end <= rows.start || rows.end <= out.start,
        "OP_VISION_CONV: the output overlaps the in_proj rows"
    );
    MkTaskDescriptor {
        a: bcx,
        b: w,
        c: y,
        n,
        k: d,
        p0: grid_w | grid_h << 16,
        ..task(OP_VISION_CONV)
    }
}

/// The gate and up rows of the experts of `plan` for the gathered rows `x` (one row of `k` for each entry, in plan
/// order) into `y` (one row of `m` for each entry): the softcapped SwiGLU of Inkling 2 (`OP_EXPERT_UP`). `slab` holds
/// the repacked experts of one layer, with `2m` rows of `k` each; `mxfp8` names its format.
pub fn expert_up(x: u64, slab: u64, y: u64, plan: u64, m: u32, k: u32, mxfp8: bool) -> MkTaskDescriptor {
    assert!(
        m.is_multiple_of(16) && k.is_multiple_of(32),
        "OP_EXPERT_UP reads whole tiles of 16 rows and scales of 32 columns, and it got {m} by {k}"
    );
    MkTaskDescriptor {
        flags: if mxfp8 { FLAG_MXFP8 } else { 0 },
        a: x,
        b: slab,
        c: y,
        x: plan,
        m,
        k,
        ..task(OP_EXPERT_UP)
    }
}

/// The down rows of the experts of `plan` for the rows `x` of `OP_EXPERT_UP` into the slot rows of `y` (8 rows of `m`
/// for each token), each times the weight of its pick (`OP_EXPERT_DOWN`). `slab` holds `m` rows of `k` for each expert.
pub fn expert_down(x: u64, slab: u64, y: u64, plan: u64, m: u32, k: u32, mxfp8: bool) -> MkTaskDescriptor {
    assert!(
        m.is_multiple_of(16) && k.is_multiple_of(32),
        "OP_EXPERT_DOWN reads whole tiles of 16 rows and scales of 32 columns, and it got {m} by {k}"
    );
    MkTaskDescriptor {
        flags: if mxfp8 { FLAG_MXFP8 } else { 0 },
        a: x,
        b: slab,
        c: y,
        x: plan,
        m,
        k,
        ..task(OP_EXPERT_DOWN)
    }
}

/// `y[t] = sum of slots[8 t + pick]` over the 8 picks in order, for `n` tokens of `m` (`OP_EXPERT_SUM`).
pub fn expert_sum(slots: u64, y: u64, n: u32, m: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: slots,
        c: y,
        n,
        m,
        ..task(OP_EXPERT_SUM)
    }
}

/// The picks of `routes` (one `MoERouteDecision` for each of `n` tokens) grouped by expert into `plan`, a `MoEPlan`
/// of `MOE_PLAN_BYTES` (`OP_MOE_PLAN`).
pub fn moe_plan(routes: u64, plan: u64, n: u32) -> MkTaskDescriptor {
    assert!(
        (1..=MAX_PLAN_TOKENS as u32).contains(&n),
        "OP_MOE_PLAN takes 1 to {MAX_PLAN_TOKENS} tokens, and it got {n}"
    );
    MkTaskDescriptor {
        a: routes,
        c: plan,
        n,
        ..task(OP_MOE_PLAN)
    }
}

/// `y[q] = x[t]` for each entry `q` of `plan`, `t` its token: the rows of `k` floats that the expert ops read, in plan
/// order, for `n` tokens (`OP_MOE_GATHER`).
pub fn moe_gather(x: u64, y: u64, plan: u64, n: u32, k: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        a: x,
        c: y,
        x: plan,
        n,
        k,
        ..task(OP_MOE_GATHER)
    }
}

// The layer-part ops of the Inkling 2 binary (`kernels/inkling2_layer_ops.cl`). The host form of each op is in
// `src/host/ops.rs` of the Inkling 2 crate.

/// The bytes of `rows` rows of `width` floats, `stride` floats apart, from `at`.
fn rows_span(at: u64, rows: u32, stride: u32, width: u32) -> std::ops::Range<u64> {
    let floats = if rows == 0 {
        0
    } else {
        (rows as u64 - 1) * stride as u64 + width as u64
    };
    at..at + 4 * floats
}

fn disjoint(a: &std::ops::Range<u64>, b: &std::ops::Range<u64>) -> bool {
    a.end <= b.start || b.end <= a.start
}

/// The attention residual of `n` positions of `k` values (`OP_ATTN_RESIDUAL`, host `attention_residual`). The rows of
/// a position are its `blocks` block outputs and its partial row, each with the RMS norm of no weight. Query `q`
/// gives the sum of the normed rows, weighted by the softmax of `q . row / sqrt(k)`.
///
/// `block_rows` holds `blocks` planes of `plane_rows` rows of `k`, and `partial` holds `[n][k]`. `queries` holds the
/// `f32` queries `[queries][k]`. `out` gets `queries` planes of `plane_rows` rows. The messages of a cycle are this op
/// with the 4 message queries.
#[allow(clippy::too_many_arguments)]
pub fn attn_residual(
    block_rows: u64,
    blocks: u32,
    partial: u64,
    queries: u64,
    query_count: u32,
    out: u64,
    n: u32,
    k: u32,
    plane_rows: u32,
) -> MkTaskDescriptor {
    assert!(n >= 1, "OP_ATTN_RESIDUAL takes at least one position");
    assert!(
        blocks <= 4,
        "a position of Inkling 2 has at most 4 block outputs, and OP_ATTN_RESIDUAL got {blocks}"
    );
    assert!(
        (1..=4).contains(&query_count),
        "OP_ATTN_RESIDUAL takes 1 to 4 queries, and it got {query_count}"
    );
    assert!(
        k >= 16 && k.is_multiple_of(16),
        "OP_ATTN_RESIDUAL sums rows in steps of 16 values, and it got a width of {k}"
    );
    assert!(
        plane_rows >= n,
        "a plane of OP_ATTN_RESIDUAL holds the {n} positions, and it has {plane_rows} rows"
    );
    let plane = plane_rows * k;
    let outputs = rows_span(out, query_count, plane, n * k);
    assert!(
        disjoint(&outputs, &rows_span(block_rows, blocks, plane, n * k))
            && disjoint(&outputs, &rows_span(partial, n, k, k)),
        "OP_ATTN_RESIDUAL: the outputs overlap the block outputs or the partial rows"
    );
    MkTaskDescriptor {
        a: block_rows,
        b: partial,
        c: out,
        x: queries,
        m: blocks | query_count << 16,
        n,
        k,
        p0: plane_rows,
        layer: (k as f32).powf(-0.5).to_bits(),
        ..task(OP_ATTN_RESIDUAL)
    }
}

/// The message gate of `n` positions of `k` values (`OP_MESSAGE_GATE`, host `message_gate`): `y = (1 - tanh(F g)) *
/// h + tanh(G g) * m`, with `g = D [norm(h), norm(m)]`. `weights` holds the `bf16` matrices of the gate one after
/// the other: `D [64][2k]`, `F [k][64]` and `G [k][64]`. `y` may be `h`.
pub fn message_gate(h: u64, m: u64, weights: u64, y: u64, n: u32, k: u32) -> MkTaskDescriptor {
    assert!(n >= 1, "OP_MESSAGE_GATE takes at least one position");
    assert!(
        k >= 16 && k.is_multiple_of(16),
        "OP_MESSAGE_GATE sums rows in steps of 16 values, and it got a width of {k}"
    );
    let out = rows_span(y, n, k, k);
    let weight_bytes = 2 * (64 * 2 * k as u64 + 2 * k as u64 * 64);
    assert!(
        y == h || disjoint(&out, &rows_span(h, n, k, k)),
        "OP_MESSAGE_GATE: the output overlaps the rows h, and it is not h"
    );
    assert!(
        disjoint(&out, &rows_span(m, n, k, k)) && disjoint(&out, &(weights..weights + weight_bytes)),
        "OP_MESSAGE_GATE: the output overlaps the messages or the weights"
    );
    MkTaskDescriptor {
        a: h,
        b: m,
        c: y,
        x: weights,
        n,
        k,
        ..task(OP_MESSAGE_GATE)
    }
}

/// What [`head_clip`] does to each head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadClip {
    /// Multiply each value by its `bf16` scale (`scale` 0: no scale), then clip the head to the L2 norm `bound`
    /// (host `scaled_l2_clip` and `l2_clip`).
    L2 { scale: u64, bound: f32 },
    /// The RMS norm of no weight (host `rms_norm_unweighted`), as for each value head.
    Rms,
}

/// The clip or the norm of each head of `n` positions of `heads` heads of `d` values (`OP_HEAD_CLIP`). `log_p0` is
/// the position of row 0 when the rows take the log scale of a full layer (host `log_scale`); with a token table,
/// each row takes the position of its entry. `y` may be `x`.
pub fn head_clip(x: u64, y: u64, n: u32, heads: u32, d: u32, clip: HeadClip, log_p0: Option<u32>) -> MkTaskDescriptor {
    assert!(
        n >= 1 && heads >= 1,
        "OP_HEAD_CLIP takes at least one head of one position, and it got {heads} heads of {n} positions"
    );
    assert!(
        d >= 16 && d.is_multiple_of(16),
        "OP_HEAD_CLIP sums heads in steps of 16 values, and it got heads of {d}"
    );
    let rows = rows_span(x, n, heads * d, heads * d);
    let out = rows_span(y, n, heads * d, heads * d);
    assert!(
        y == x || disjoint(&out, &rows),
        "OP_HEAD_CLIP: the output overlaps the input, and it is not the input"
    );
    let (scale, bound, mode) = match clip {
        HeadClip::L2 { scale, bound } => {
            assert!(
                bound.is_finite() && bound > 0.0,
                "the L2 bound of OP_HEAD_CLIP is a positive number, and it got {bound}"
            );
            (scale, bound, 0)
        }
        HeadClip::Rms => {
            // Inkling 2 gives the log scale to the queries and the relative states only.
            assert!(
                log_p0.is_none(),
                "OP_HEAD_CLIP gives the log scale only to an L2 clip, as Inkling 2 does"
            );
            (0, 0.0, 1)
        }
    };
    MkTaskDescriptor {
        a: x,
        b: scale,
        c: y,
        x: bound.to_bits() as u64,
        n,
        m: heads,
        k: d,
        p0: log_p0.unwrap_or(0),
        layer: mode | if log_p0.is_some() { 256 } else { 0 },
        ..task(OP_HEAD_CLIP)
    }
}

/// The gate of the attention output (`OP_OUTPUT_GATE`): `y[i] = a[i] * sigmoid(gate[i])` for `n` rows of `k`. `y` may
/// be `a`.
pub fn output_gate(a: u64, gate: u64, y: u64, n: u32, k: u32) -> MkTaskDescriptor {
    assert!(n >= 1 && k >= 1, "OP_OUTPUT_GATE takes at least one value");
    let out = rows_span(y, n, k, k);
    assert!(
        y == a || disjoint(&out, &rows_span(a, n, k, k)),
        "OP_OUTPUT_GATE: the output overlaps the input, and it is not the input"
    );
    assert!(
        disjoint(&out, &rows_span(gate, n, k, k)),
        "OP_OUTPUT_GATE: the output overlaps the gates"
    );
    MkTaskDescriptor {
        a,
        b: gate,
        c: y,
        n,
        k,
        ..task(OP_OUTPUT_GATE)
    }
}

/// The taps of [`residual_conv`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvTaps {
    /// Fixed taps, `bf16 [k][4]`, with tap 3 on the current row (host `short_conv`).
    Fixed(u64),
    /// Fixed taps by lag, `bf16 [4][k]` with lag 0 on the current row, plus the 4 taps of each head of 128 channels.
    /// The taps of a position follow its `k` channels in its input row (host `dynamic_conv`).
    Dynamic(u64),
}

/// The short convolution of kernel 4 with its residual (`OP_RESIDUAL_CONV`), for `n` positions of `k` channels from
/// position `p0`. The input row of each position is `input_stride` floats apart. `state` holds the 3 inputs before
/// position `p0`, oldest first, in rows `state_stride` floats apart, and the op writes the last 3 inputs into it. At
/// `p0` 0 the op reads the state as zeros. `y` gets `[n][k]`, or `accumulate` adds the result to `y`.
#[allow(clippy::too_many_arguments)]
pub fn residual_conv(
    input: u64,
    input_stride: u32,
    taps: ConvTaps,
    state: u64,
    state_stride: u32,
    y: u64,
    n: u32,
    k: u32,
    p0: u32,
    accumulate: bool,
) -> MkTaskDescriptor {
    assert!(n >= 1, "OP_RESIDUAL_CONV takes at least one position");
    assert!(
        k >= 16 && k.is_multiple_of(16),
        "OP_RESIDUAL_CONV takes channels in 16s, and it got {k}"
    );
    let (weights, dynamic, width) = match taps {
        ConvTaps::Fixed(w) => (w, false, k),
        ConvTaps::Dynamic(w) => {
            assert!(
                k.is_multiple_of(128),
                "the dynamic taps of OP_RESIDUAL_CONV are by head of 128 channels, and it got {k} channels"
            );
            (w, true, k + k / 128 * 4)
        }
    };
    // The kernel reads each stride from 16 bits.
    assert!(
        input_stride >= width && input_stride <= 0xFFFF,
        "an input row of OP_RESIDUAL_CONV holds {width} floats in a stride of at most 65535, and it got {input_stride}"
    );
    assert!(
        state_stride >= k && state_stride <= 0xFFFF,
        "a state row of OP_RESIDUAL_CONV holds {k} floats in a stride of at most 65535, and it got {state_stride}"
    );
    let rows = rows_span(input, n, input_stride, width);
    let held = rows_span(state, 3, state_stride, k);
    let out = rows_span(y, n, k, k);
    for (a, b, what) in [
        (&out, &rows, "the output overlaps the inputs"),
        (&out, &held, "the output overlaps the state"),
        (&held, &rows, "the state overlaps the inputs"),
    ] {
        assert!(disjoint(a, b), "OP_RESIDUAL_CONV: {what}");
    }
    MkTaskDescriptor {
        flags: if accumulate { FLAG_ACCUMULATE } else { 0 },
        a: input,
        b: weights,
        c: y,
        x: state,
        m: input_stride | state_stride << 16,
        n,
        k,
        p0,
        layer: dynamic as u32,
        ..task(OP_RESIDUAL_CONV)
    }
}

/// The router of `n` positions (`OP_SIGMOID_ROUTER`, host `route`): sigmoid scores of the 128 `f32` logits, the 8
/// largest score + `bias`, and the weights score / (sum + 1e-20) * 8. `decisions` gets one `MoERouteDecision` for each
/// position, with its picks in ascending expert order and `layer` as its layer.
pub fn sigmoid_router(logits: u64, bias: u64, decisions: u64, n: u32, layer: u32) -> MkTaskDescriptor {
    assert!(n >= 1, "OP_SIGMOID_ROUTER takes at least one position");
    let out = decisions..decisions + n as u64 * std::mem::size_of::<MoERouteDecision>() as u64;
    assert!(
        disjoint(&out, &rows_span(logits, n, NUM_EXPERTS as u32, NUM_EXPERTS as u32))
            && disjoint(&out, &rows_span(bias, 1, NUM_EXPERTS as u32, NUM_EXPERTS as u32)),
        "OP_SIGMOID_ROUTER: the decisions overlap the logits or the bias"
    );
    MkTaskDescriptor {
        a: logits,
        b: bias,
        c: decisions,
        n,
        layer,
        ..task(OP_SIGMOID_ROUTER)
    }
}

/// The `bf16` bits of the repacked NVFP4 planes `codes` and `scales` of `rows` rows and `cols` columns, into `y`, with
/// the factor of the expert. The rows of `y` are in the order of the planes (`OP_EXPAND_NVFP4`).
pub fn expand_nvfp4(codes: u64, scales: u64, y: u64, rows: u32, cols: u32, factor: f32) -> MkTaskDescriptor {
    assert!(
        rows.is_multiple_of(16) && cols.is_multiple_of(16),
        "OP_EXPAND_NVFP4 reads whole tiles of 16 rows and steps of 16 columns, and it got {rows} by {cols}"
    );
    MkTaskDescriptor {
        a: codes,
        b: scales,
        c: y,
        m: rows,
        k: cols,
        p0: factor.to_bits(),
        ..task(OP_EXPAND_NVFP4)
    }
}

/// The `bf16` bits of the repacked MXFP8 planes `values` and `scales` of `rows` rows and `cols` columns, into `y`
/// (`OP_EXPAND_MXFP8`).
pub fn expand_mxfp8(values: u64, scales: u64, y: u64, rows: u32, cols: u32) -> MkTaskDescriptor {
    assert!(
        rows.is_multiple_of(16) && cols.is_multiple_of(32),
        "OP_EXPAND_MXFP8 reads whole tiles of 16 rows and scales of 32 columns, and it got {rows} by {cols}"
    );
    MkTaskDescriptor {
        a: values,
        b: scales,
        c: y,
        m: rows,
        k: cols,
        ..task(OP_EXPAND_MXFP8)
    }
}

/// The keys of one round of `OP_INKLING2_ATTN` (`INKLING2_ATTN_KEYS`, `kernels/inkling2_attn_helpers.cl`).
pub const INKLING2_ATTN_KEYS: u32 = 128;

/// The floats of scratch of one team of `OP_INKLING2_ATTN`: the probabilities of one round for 16 positions, then a
/// max and a sum of each position for each of the 4 subgroups (`INKLING2_ATTN_TEAM_FLOATS`).
pub const INKLING2_ATTN_TEAM_FLOATS: usize = 16 * INKLING2_ATTN_KEYS as usize + 2 * 4 * 16;

/// The window of a sliding layer of Inkling 2: a query at position `q` reads the keys `q - 1023` to `q`.
pub const INKLING2_WINDOW: u32 = 1024;

/// The bits of the f32 of `128^-0.5`, the scale of `q . k` in the kernel and in the host form.
pub const INKLING2_ATTN_SCALE_BITS: u32 = 0x3DB5_04F3;

/// The largest block of `OP_INKLING2_ATTN`.
pub const INKLING2_ATTN_MAX_BLOCK: u32 = 512;

/// The positions that `OP_INKLING2_ATTN` counts: 2^24, 16 times the 1,048,576 positions of the model. Below it, the
/// sums of positions do not wrap in `u32`, and the distances fit an `int`.
pub const INKLING2_ATTN_MAX_POSITIONS: u32 = 1 << 24;

/// The keys of one chunk of a sliding layer. **Proposed** (Claude Opus 5.5, 2026-11-21): small chunks spread a decode
/// step over 5 chunks of each head, and a block of 512 takes at most 7 chunks of scratch.
pub const INKLING2_SLIDING_CHUNK_KEYS: u32 = 256;

/// The keys of one chunk of a full layer. **Proposed** (Claude Opus 5.5, 2026-11-21): a block of 512 in a slot of
/// 32,768 positions takes 16 chunks, about 102 MiB of scratch. A decode step walks at most 16 rounds in each chunk.
pub const INKLING2_FULL_CHUNK_KEYS: u32 = 2048;

/// The keys of one chunk for a layer with a ring of `ring` rows (a sliding layer), or with cells (`ring` 0). The
/// chunks of a layer must not depend on the block, so that a position gets the same bits in each split.
pub fn inkling2_chunk_keys(ring: u32) -> u32 {
    if ring > 0 {
        INKLING2_SLIDING_CHUNK_KEYS
    } else {
        INKLING2_FULL_CHUNK_KEYS
    }
}

/// The attention of one run of Inkling 2 over a block of `n` query positions from `p0`: `OP_INKLING2_ATTN`, then
/// `OP_INKLING2_ATTN_SUM`. A decode step is a block of one position, and it gets the bits that the position gets in
/// a block with the same `chunk_keys` (`kernels/inkling2_attn_ops.cl`, "Bits and splits").
///
/// The heads are those of Inkling 2: 24 query heads and 8 KV heads of 128 values, with 16 relative states.
#[derive(Clone, Copy, Debug)]
pub struct Inkling2Attention {
    /// The queries, `[n][24][128]` f32, after the clip and the log scale.
    pub q: u64,
    /// The key plane, rows of `[8][128]` f32, then the value plane: at `max_seq` rows for cells, or at `ring + 1024`
    /// rows for a ring.
    pub kv: u64,
    /// The relative states, `[n][24][16]` f32, after the clip and the log scale.
    pub rel: u64,
    /// The 16 profiles of the distance, `[16][1024]` bf16, as the checkpoint holds `rel_logits_proj.proj`.
    pub proj: u64,
    /// The sink of each position and head, `[n][24]` f32.
    pub sinks: u64,
    /// The output, `[n][24][128]` f32, before the output gate.
    pub out: u64,
    pub n: u32,
    pub p0: u32,
    /// The rows of a cell plane: the positions of the slot. A ring does not read it.
    pub max_seq: u32,
    /// The rows of the ring of a sliding layer, or 0 for the cells of a full layer.
    pub ring: u32,
    /// The keys of one chunk, a multiple of 128 (`inkling2_chunk_keys`).
    pub chunk_keys: u32,
}

impl Inkling2Attention {
    /// The first chunk of the block and the count of its chunks, as `inkling2_attn_chunks` of the kernel counts them.
    pub fn chunks(&self) -> (u32, u32) {
        self.check();
        let lowest = if self.ring > 0 {
            (self.p0 + 1).saturating_sub(INKLING2_WINDOW)
        } else {
            0
        };
        let first = lowest / self.chunk_keys;
        (first, (self.p0 + self.n - 1) / self.chunk_keys - first + 1)
    }

    /// The floats of scratch that the two tasks use: the partial sums of each chunk, position and head, then the
    /// scratch of each team.
    pub fn scratch_floats(&self) -> usize {
        let partial = self.chunks().1 as usize * self.n as usize * 24 * 130;
        partial.next_multiple_of(16) + crate::protocol::MEGAKERNEL_SUBGROUPS / 4 * INKLING2_ATTN_TEAM_FLOATS
    }

    /// Refuse a block that the kernel cannot take. The kernel sums positions in `u32` and takes distances as `int`,
    /// so the positions stay below `INKLING2_ATTN_MAX_POSITIONS`.
    fn check(&self) {
        assert!(
            (1..=INKLING2_ATTN_MAX_BLOCK).contains(&self.n),
            "OP_INKLING2_ATTN takes 1 to {INKLING2_ATTN_MAX_BLOCK} positions, and it got {}",
            self.n
        );
        assert!(
            self.chunk_keys > 0
                && self.chunk_keys.is_multiple_of(INKLING2_ATTN_KEYS)
                && self.chunk_keys <= INKLING2_ATTN_MAX_POSITIONS,
            "a chunk of OP_INKLING2_ATTN holds a whole count of rounds of 128 keys, and it got {}",
            self.chunk_keys
        );
        assert!(
            self.p0 as u64 + self.n as u64 <= INKLING2_ATTN_MAX_POSITIONS as u64,
            "the block of OP_INKLING2_ATTN ends at position {}, past the {INKLING2_ATTN_MAX_POSITIONS} positions \
             that the kernel counts",
            self.p0 as u64 + self.n as u64 - 1
        );
        if self.ring > 0 {
            assert!(
                self.ring >= INKLING2_WINDOW - 1 + self.n,
                "a ring of OP_INKLING2_ATTN holds the window and the block: {} rows for {} positions",
                self.ring,
                self.n
            );
            assert!(
                self.ring.is_multiple_of(INKLING2_ATTN_KEYS),
                "a ring of OP_INKLING2_ATTN has a whole count of rounds of 128 rows, and it has {}",
                self.ring
            );
        } else {
            assert!(
                self.p0 + self.n <= self.max_seq,
                "the block of OP_INKLING2_ATTN ends past the cells of the slot: positions {} to {} of {}",
                self.p0,
                self.p0 + self.n - 1,
                self.max_seq
            );
        }
    }

    /// The two tasks. The keys and values of the block must be in the planes, and the spill rows of a ring in place.
    /// Each row of the planes below the key count of the block holds finite values (see the op comment).
    pub fn tasks(&self) -> [MkTaskDescriptor; 2] {
        self.check();
        let base = MkTaskDescriptor {
            n: self.n,
            k: self.chunk_keys,
            p0: self.p0,
            layer: self.ring,
            ..Default::default()
        };
        [
            MkTaskDescriptor {
                opcode: OP_INKLING2_ATTN,
                a: self.q,
                b: self.kv,
                c: self.rel,
                x: self.proj,
                m: self.max_seq,
                ..base
            },
            MkTaskDescriptor {
                opcode: OP_INKLING2_ATTN_SUM,
                a: self.sinks,
                c: self.out,
                ..base
            },
        ]
    }
}

/// The floats of scratch that the attention of Inkling 2 uses for a block of up to `max_block` positions in a slot of
/// `max_seq` positions, with the chunks of `inkling2_chunk_keys`. A sliding block reads `n + 1023` keys at most, and
/// a full block reads `max_seq` keys at most.
pub fn inkling2_attn_scratch_floats(max_block: usize, max_seq: usize) -> usize {
    let sliding = (max_block + INKLING2_WINDOW as usize - 1).div_ceil(INKLING2_SLIDING_CHUNK_KEYS as usize) + 1;
    let full = max_seq.div_ceil(INKLING2_FULL_CHUNK_KEYS as usize);
    let partial = sliding.max(full) * max_block * 24 * 130;
    partial.next_multiple_of(16) + crate::protocol::MEGAKERNEL_SUBGROUPS / 4 * INKLING2_ATTN_TEAM_FLOATS
}

// The ops of the vision tower of Inkling 2 (`kernels/inkling2_vision_ops.cl`). The host form is
// `src/host/vision.rs` of the Inkling 2 crate.

/// The positions of the rotary table of `OP_INKLING2_VISION_ROPE`: the class row and 206 patch lines
/// (`INKLING2_VISION_POSITIONS`, kernels/inkling2_vision_helpers.cl).
pub const INKLING2_VISION_POSITIONS: usize = 207;
/// The frequencies of one axis of a head of 64 values (`INKLING2_VISION_FREQS`).
pub const INKLING2_VISION_FREQS: usize = 16;

/// Check the grid of an image of `rows x columns` image tokens: each side 1 or more, and at most 103 tokens, so its
/// patch lines fit the rotary table.
fn vision_grid(op: &str, rows: u32, columns: u32) {
    let most = (INKLING2_VISION_POSITIONS as u32 - 1) / 2;
    assert!(
        (1..=most).contains(&rows) && (1..=most).contains(&columns),
        "{op} takes an image of 1 to {most} image tokens on each side, and it got {rows} by {columns}"
    );
}

/// The bytes of the taps of `OP_INKLING2_VISION_EMBED` for an image of `rows x columns` image tokens: 16 for each
/// patch line, then 16 for each patch column.
pub fn inkling2_vision_axes_bytes(rows: u32, columns: u32) -> usize {
    16 * 2 * (rows + columns) as usize
}

/// The position embedding of one image of `rows x columns` image tokens: `y[1 + p] = patches[p] + the bilinear sum
/// of 4 rows of the bf16 table`, for `4 * rows * columns` patches of `k` values, with the taps and weights `axes`
/// of the host (`OP_INKLING2_VISION_EMBED`). Row 0 of `y` is the class row, which the op does not write.
#[allow(clippy::too_many_arguments)]
pub fn inkling2_vision_embed(
    patches: u64,
    table: u64,
    axes: u64,
    y: u64,
    rows: u32,
    columns: u32,
    k: u32,
) -> MkTaskDescriptor {
    vision_grid("OP_INKLING2_VISION_EMBED", rows, columns);
    assert!(k > 0, "OP_INKLING2_VISION_EMBED takes rows of 1 value or more");
    let n = 4 * rows * columns;
    let rows_in = patches..patches + 4 * n as u64 * k as u64;
    let out = y..y + 4 * (n as u64 + 1) * k as u64;
    assert!(
        out.end <= rows_in.start || rows_in.end <= out.start,
        "OP_INKLING2_VISION_EMBED: the output overlaps the patch rows"
    );
    MkTaskDescriptor {
        a: patches,
        b: table,
        x: axes,
        c: y,
        n,
        k,
        p0: rows | columns << 16,
        ..task(OP_INKLING2_VISION_EMBED)
    }
}

/// The rotary embedding in two axes of the query rows `q` and the key rows `k` of one image, in place: `n` rows of
/// `width` values, in heads of 64, the class row first. `columns` is the width of the image in image tokens, and
/// `table` holds the cosines `[207][16]`, then the sines (`OP_INKLING2_VISION_ROPE`).
pub fn inkling2_vision_rope(q: u64, k: u64, table: u64, n: u32, width: u32, columns: u32) -> MkTaskDescriptor {
    assert!(
        width >= 64 && width.is_multiple_of(64),
        "OP_INKLING2_VISION_ROPE takes heads of 64 values, and it got rows of {width}"
    );
    assert!(
        columns >= 1 && n >= 5 && (n - 1).is_multiple_of(4 * columns),
        "OP_INKLING2_VISION_ROPE takes a class row and whole lines of {columns} image tokens, and it got {n} rows"
    );
    vision_grid("OP_INKLING2_VISION_ROPE", (n - 1) / (4 * columns), columns);
    assert_ne!(
        q, k,
        "OP_INKLING2_VISION_ROPE turns the queries and the keys once each, so they are two buffers"
    );
    MkTaskDescriptor {
        a: q,
        b: k,
        x: table,
        n,
        k: width,
        p0: columns,
        ..task(OP_INKLING2_VISION_ROPE)
    }
}

/// The attention of the vision tower of Inkling 2 over the `n` rows of one image, in both directions: `q` holds
/// `[n][heads * d]`, `k` and `v` hold `[n][kv_heads * d]`, and `y` gets `[n][heads * d]`. It is `vision_attn` of the
/// LFM3 binary under the opcode of this binary (`OP_INKLING2_VISION_ATTN`).
#[allow(clippy::too_many_arguments)]
pub fn inkling2_vision_attn(
    q: u64,
    k: u64,
    v: u64,
    y: u64,
    n: u32,
    heads: u32,
    kv_heads: u32,
    d: u32,
    scale: f32,
) -> MkTaskDescriptor {
    MkTaskDescriptor {
        opcode: OP_INKLING2_VISION_ATTN,
        ..vision_attn(q, k, v, y, n, heads, kv_heads, d, scale)
    }
}
