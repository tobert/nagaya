//! The layer-part ops of Inkling 2 on a B70 card against their host forms (engine-abi "The first Inkling 2 binding",
//! "The gate of each op"): the attention residual, the message gate, the clips of the heads, the output gate, the
//! convolutions with a residual and the router. The inputs are fixed values from a small generator, so the test
//! needs no model file.
//!
//! Each op must give a position the same bits alone, in a block of 5 and in a block of 37. A convolution must give
//! the same bits for a block fed whole and fed in parts. Each sum of an op has the order of the host `dot`, with no
//! fused multiply-add, so a gap from the host comes from the divisions, the square roots and the transcendental
//! functions of the card. Each tolerance below names that source. The card gate has not run yet.
//!
//! Run it alone, on a card that no engine of the active profile names (`AGENTS.md`, "The engines on the B70 cards"):
//! `NAGAYA_TEST_CARDS=<pci> cargo test -p nagaya-mk-inkling2-bmg --features xe --test xe_layer_ops --
//! --test-threads=1`.
#![cfg(feature = "xe")]

mod common;

use std::ffi::c_void;

use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::{MkTaskDescriptor, MoERouteDecision};
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::tasks::{self, ConvTaps, HeadClip};
use nagaya_mk_bmg::{INKLING2_KERNEL, KERNEL_ENTRY};
use nagaya_mk_inkling2_bmg::host::ops::{
    self, CONV_TAPS, D_REL, EXPERTS, GATE_RANK, HEAD_DIM, HEADS, HIDDEN, KV_HEADS, QK_CLIP, REL_CLIP, TOP_K,
};

/// The block sizes of each check: a position alone, a short block and a block of more than 32 positions, so that the
/// positions spread over more than one round of the work-groups.
const BLOCKS: [usize; 3] = [1, 5, 37];
const MOST: usize = 37;

/// The attention residual, as the relative L2 gap of a row. The card and the host take each RMS factor through a
/// division and a square root of a few units in the last place (ulp) each, then an exp and a division for each
/// weight. About 10 ulp of 6e-8 reach each output value.
const TOL_RESIDUAL: f64 = 1e-5;
/// The message gate. The sums of g have the order of the host, so g moves only with the two RMS factors. Then tanh
/// on the card differs by a few ulp, and the output mixes two products. As for the residual: about 10 ulp.
const TOL_GATE: f64 = 1e-5;
/// The clips. The sum of squares has the bits of the host. The factor takes a square root and a division (and the
/// log of the log scale), so each value is a few ulp from the host: 4 ulp is 2.4e-7.
const TOL_CLIP: f64 = 1e-6;
/// The output gate: the exp, the add and the division of the sigmoid, a few ulp, then one product.
const TOL_OUTPUT_GATE: f64 = 1e-6;
/// The convolutions. The sum has the bits of the host, and only the division by sqrt(5) or 3 differs: 2.5 ulp at most
/// in OpenCL C.
const TOL_CONV: f64 = 1e-6;
/// The weights of the router: the sigmoid of each score and the division by their sum, a few ulp each.
const TOL_ROUTE: f64 = 2e-6;

/// A buffer in device memory, which the test fills and reads through host memory.
struct Buf(*mut u8, usize);

impl Buf {
    fn zeros(card: &B70Card, bytes: usize) -> Self {
        let ptr = card.alloc_device_usm::<u8>(bytes.max(64));
        card.fill_zero_sync(ptr as *mut c_void, bytes.max(64));
        Self(ptr, bytes)
    }

    fn of<T: Copy>(card: &B70Card, data: &[T]) -> Self {
        let bytes = std::mem::size_of_val(data);
        let buf = Self::zeros(card, bytes);
        let stage = card.alloc_host_usm::<u8>(bytes.max(64));
        // SAFETY: `stage` holds `bytes` bytes of host USM, and `data` is `bytes` bytes long.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, stage, bytes) };
        card.memcpy_sync(buf.0 as *mut c_void, stage as *const c_void, bytes);
        card.free(stage as *mut c_void);
        buf
    }

    fn read<T: Copy + Default>(&self, card: &B70Card, n: usize) -> Vec<T> {
        let bytes = n * std::mem::size_of::<T>();
        assert!(bytes <= self.1.max(64), "the read is longer than the buffer");
        let stage = card.alloc_host_usm::<u8>(bytes.max(64));
        card.memcpy_sync(stage as *mut c_void, self.0 as *const c_void, bytes);
        let mut out = vec![T::default(); n];
        // SAFETY: `stage` holds `bytes` bytes, and `out` has room for them.
        unsafe { std::ptr::copy_nonoverlapping(stage, out.as_mut_ptr() as *mut u8, bytes) };
        card.free(stage as *mut c_void);
        out
    }

    fn at(&self) -> u64 {
        self.0 as u64
    }
}

/// The card of the test, its ring, and the buffers to free at the end.
struct Card {
    cluster: L0Cluster,
    ring: Ring,
    bufs: Vec<*mut u8>,
}

impl Card {
    fn open() -> Self {
        let cluster = L0Cluster::init_test(INKLING2_KERNEL, KERNEL_ENTRY);
        let ring = Ring::new(&cluster.cards[0], 1 << 16);
        Self {
            cluster,
            ring,
            bufs: Vec::new(),
        }
    }

    fn card(&self) -> &B70Card {
        &self.cluster.cards[0]
    }

    fn buf<T: Copy>(&mut self, data: &[T]) -> Buf {
        let b = Buf::of(self.card(), data);
        self.bufs.push(b.0);
        b
    }

    fn zeros(&mut self, bytes: usize) -> Buf {
        let b = Buf::zeros(self.card(), bytes);
        self.bufs.push(b.0);
        b
    }

    fn run(&mut self, tasks: &[MkTaskDescriptor]) {
        self.ring.set(tasks);
        // SAFETY: each check sizes its buffers for its tasks.
        unsafe { self.ring.run(&self.cluster.cards[0]) };
    }
}

impl Drop for Card {
    fn drop(&mut self) {
        for b in self.bufs.drain(..) {
            self.cluster.cards[0].free(b as *mut c_void);
        }
    }
}

/// `len` values in `[-spread, spread)` from the seed.
fn values(seed: u32, len: usize, spread: f32) -> Vec<f32> {
    let mut s = 0x9E37_79B9u32 ^ seed.wrapping_mul(0x85EB_CA6B);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32 * 2.0 - 1.0) * spread
        })
        .collect()
}

/// `len` `bf16` weights in `[-spread, spread)`, as bits and widened.
fn bf16s(seed: u32, len: usize, spread: f32) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = values(seed, len, spread)
        .iter()
        .map(|v| (v.to_bits() >> 16) as u16)
        .collect();
    let wide = bits.iter().map(|b| ops::bf16_to_f32(*b)).collect();
    (bits, wide)
}

/// The rows of positions `0..n`: row `t` comes from seed `seed + t`, so a block holds the rows of a shorter one.
fn rows(seed: u32, n: usize, width: usize, spread: f32) -> Vec<f32> {
    (0..n)
        .flat_map(|t| values(seed + t as u32 * 7919, width, spread))
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Check that each block size gives the rows of the longest block in its bits, and return the longest.
fn same_in_each_block(what: &str, width: usize, mut run: impl FnMut(usize) -> Vec<f32>) -> Vec<f32> {
    let outs: Vec<Vec<f32>> = BLOCKS.iter().map(|n| run(*n)).collect();
    let longest = outs.last().unwrap();
    for (n, out) in BLOCKS.iter().zip(&outs) {
        assert_eq!(out.len(), n * width, "{what}: a block of {n} gave {} values", out.len());
        assert!(
            bits(out) == bits(&longest[..n * width]),
            "{what}: a block of {n} and a block of {MOST} give other bits"
        );
    }
    longest.clone()
}

/// The worst relative L2 gap of the rows, printed for the gate report.
fn gap(what: &str, got: &[f32], want: &[f32], width: usize, tol: f64) {
    let worst = common::worst_row(got, want, width);
    eprintln!("{what}: the worst row is {worst:.3e} from the host form (relative L2, tolerance {tol:e})");
    assert!(worst <= tol, "{what}: {worst:.3e} > {tol:e}");
}

fn row(a: &[f32], t: usize, width: usize) -> &[f32] {
    &a[t * width..(t + 1) * width]
}

/// The attention residual of one shape: `blocks` block outputs and `queries` queries.
fn check_attn_residual(card: &mut Card, blocks: usize, queries: usize) {
    let what = format!("the attention residual of {blocks} blocks and {queries} queries");
    // Rows of other sizes, so that the norm matters, and queries that give scores of about 1.
    let planes: Vec<f32> = (0..blocks)
        .flat_map(|j| rows(100 + j as u32, MOST, HIDDEN, 0.5 + j as f32))
        .collect();
    let partial = rows(200, MOST, HIDDEN, 3.0);
    let q = values(300 + blocks as u32, queries * HIDDEN, 2.0);
    let (bb, pb, qb) = (card.buf(&planes), card.buf(&partial), card.buf(&q));
    let ob = card.zeros(queries * MOST * HIDDEN * 4);
    let got = same_in_each_block(&what, queries * HIDDEN, |n| {
        card.run(&[tasks::attn_residual(
            bb.at(),
            blocks as u32,
            pb.at(),
            qb.at(),
            queries as u32,
            ob.at(),
            n as u32,
            HIDDEN as u32,
            MOST as u32,
        )]);
        let planes: Vec<f32> = ob.read(card.card(), queries * MOST * HIDDEN);
        // Position-major: the rows of each query of position t.
        (0..n)
            .flat_map(|t| (0..queries).flat_map(move |i| (i * MOST + t) * HIDDEN..(i * MOST + t + 1) * HIDDEN))
            .map(|i| planes[i])
            .collect()
    });
    let qs: Vec<&[f32]> = q.as_chunks::<HIDDEN>().0.iter().map(|r| r.as_slice()).collect();
    let mut want = Vec::new();
    let mut out = vec![Vec::new(); queries];
    for t in 0..MOST {
        let blk: Vec<&[f32]> = (0..blocks).map(|j| row(&planes, j * MOST + t, HIDDEN)).collect();
        ops::attention_residual(&blk, row(&partial, t, HIDDEN), &qs, &mut out);
        for o in &out {
            want.extend_from_slice(o);
        }
    }
    gap(&what, &got, &want, HIDDEN, TOL_RESIDUAL);
}

#[test]
fn the_attention_residual_on_the_card_gives_the_host_form_in_any_block() {
    let mut card = Card::open();
    // Layer 0's expert input, a sliding layer, a full layer, the messages of a cycle, and the most block outputs.
    for (blocks, queries) in [(1, 1), (2, 1), (1, 3), (1, 4), (4, 4), (0, 1)] {
        check_attn_residual(&mut card, blocks, queries);
    }
}

#[test]
fn the_message_gate_on_the_card_gives_the_host_form_in_any_block_and_in_place() {
    let mut card = Card::open();
    let h = rows(400, MOST, HIDDEN, 2.0);
    let m = rows(500, MOST, HIDDEN, 0.7);
    // Weights that give g and the gates of about 1, so that tanh is not flat.
    let (down_b, down) = bf16s(600, GATE_RANK * 2 * HIDDEN, 0.03);
    let (forget_b, forget) = bf16s(601, HIDDEN * GATE_RANK, 0.2);
    let (gain_b, gain) = bf16s(602, HIDDEN * GATE_RANK, 0.2);
    let weights: Vec<u16> = [down_b, forget_b, gain_b].concat();
    let (hb, mb, wb) = (card.buf(&h), card.buf(&m), card.buf(&weights));
    let yb = card.zeros(MOST * HIDDEN * 4);
    let got = same_in_each_block("the message gate", HIDDEN, |n| {
        card.run(&[tasks::message_gate(
            hb.at(),
            mb.at(),
            wb.at(),
            yb.at(),
            n as u32,
            HIDDEN as u32,
        )]);
        yb.read(card.card(), n * HIDDEN)
    });
    let mut want = vec![0f32; MOST * HIDDEN];
    for t in 0..MOST {
        ops::message_gate(
            row(&h, t, HIDDEN),
            row(&m, t, HIDDEN),
            &down,
            &forget,
            &gain,
            &mut want[t * HIDDEN..(t + 1) * HIDDEN],
        );
    }
    gap("the message gate", &got, &want, HIDDEN, TOL_GATE);
    // In place: the output replaces h, with the same bits.
    let ib = card.buf(&h);
    card.run(&[tasks::message_gate(
        ib.at(),
        mb.at(),
        wb.at(),
        ib.at(),
        MOST as u32,
        HIDDEN as u32,
    )]);
    let in_place: Vec<f32> = ib.read(card.card(), MOST * HIDDEN);
    assert!(
        bits(&in_place) == bits(&got),
        "the message gate in place gives other bits"
    );
}

/// The clip of one kind of row on the card and on the host.
#[allow(clippy::too_many_arguments)]
fn check_clip(
    card: &mut Card,
    what: &str,
    heads: usize,
    d: usize,
    spread: f32,
    scale: Option<&[u16]>,
    clip: HeadClip,
    p0: Option<u32>,
    host: impl Fn(&mut [f32], usize),
) -> (Vec<f32>, Vec<f32>) {
    let width = heads * d;
    let x = rows(700 + width as u32, MOST, width, spread);
    let xb = card.buf(&x);
    let yb = card.zeros(MOST * width * 4);
    let clip = match (clip, scale) {
        (HeadClip::L2 { bound, .. }, Some(s)) => HeadClip::L2 {
            scale: card.buf(s).at(),
            bound,
        },
        (c, _) => c,
    };
    let got = same_in_each_block(what, width, |n| {
        card.run(&[tasks::head_clip(
            xb.at(),
            yb.at(),
            n as u32,
            heads as u32,
            d as u32,
            clip,
            p0,
        )]);
        yb.read(card.card(), n * width)
    });
    let mut want = x.clone();
    for t in 0..MOST {
        host(&mut want[t * width..(t + 1) * width], t);
    }
    gap(what, &got, &want, width, TOL_CLIP);
    (got, want)
}

#[test]
fn the_clips_on_the_card_give_the_host_form_in_any_block() {
    let mut card = Card::open();
    let qd = HEADS * HEAD_DIM;
    let (q_scale_b, q_scale) = bf16s(800, qd, 1.5);
    let (r_scale_b, r_scale) = bf16s(801, HEADS * D_REL, 3.0);
    // Most query heads pass the bound, and about half of the relative states do. Positions from 20,000 take a log
    // scale above 1.
    for p0 in [None, Some(20_000u32)] {
        let log = |t: usize| p0.map_or(1.0, |p| ops::log_scale(p as usize + t));
        check_clip(
            &mut card,
            &format!("the clip of the query, log scale {p0:?}"),
            HEADS,
            HEAD_DIM,
            6.0,
            Some(&q_scale_b),
            HeadClip::L2 {
                scale: 0,
                bound: QK_CLIP,
            },
            p0,
            |x, t| {
                ops::scaled_l2_clip(x, &q_scale, HEAD_DIM, QK_CLIP);
                x.iter_mut().for_each(|v| *v *= log(t));
            },
        );
        check_clip(
            &mut card,
            &format!("the clip of the relative states, log scale {p0:?}"),
            HEADS,
            D_REL,
            2.0,
            Some(&r_scale_b),
            HeadClip::L2 {
                scale: 0,
                bound: REL_CLIP,
            },
            p0,
            |x, t| {
                ops::scaled_l2_clip(x, &r_scale, D_REL, REL_CLIP);
                x.iter_mut().for_each(|v| *v *= log(t));
            },
        );
    }
    let (got, want) = check_clip(
        &mut card,
        "the clip of the key",
        KV_HEADS,
        HEAD_DIM,
        3.4,
        None,
        HeadClip::L2 {
            scale: 0,
            bound: QK_CLIP,
        },
        None,
        |x, _| {
            x.as_chunks_mut::<HEAD_DIM>()
                .0
                .iter_mut()
                .for_each(|h| ops::l2_clip(h, QK_CLIP))
        },
    );
    // A head under the bound keeps its values: its factor is 1, so the card gives the bits of the host. A spread of
    // 3.4 gives heads of 128 a norm of about 22, near the bound.
    let input = rows(700 + (KV_HEADS * HEAD_DIM) as u32, MOST, KV_HEADS * HEAD_DIM, 3.4);
    let mut short = 0;
    for ((g, w), x) in got
        .as_chunks::<HEAD_DIM>()
        .0
        .iter()
        .zip(want.as_chunks::<HEAD_DIM>().0)
        .zip(input.as_chunks::<HEAD_DIM>().0)
    {
        if ops::dot(x, x).sqrt() < 0.99 * QK_CLIP {
            short += 1;
            assert!(bits(g) == bits(w), "a key head under the bound changed on the card");
        }
    }
    assert!(
        short > 0 && short < MOST * KV_HEADS,
        "the inputs must hold heads on both sides of the bound"
    );
    check_clip(
        &mut card,
        "the norm of the value",
        KV_HEADS,
        HEAD_DIM,
        3.0,
        None,
        HeadClip::Rms,
        None,
        |x, _| {
            x.as_chunks_mut::<HEAD_DIM>()
                .0
                .iter_mut()
                .for_each(|h| ops::rms_norm_unweighted(h))
        },
    );
}

#[test]
fn the_output_gate_on_the_card_gives_the_host_form_in_any_block() {
    let mut card = Card::open();
    let qd = HEADS * HEAD_DIM;
    let attn = rows(900, MOST, qd, 2.0);
    let gate = rows(901, MOST, qd, 8.0);
    let (ab, gb) = (card.buf(&attn), card.buf(&gate));
    let yb = card.zeros(MOST * qd * 4);
    let got = same_in_each_block("the output gate", qd, |n| {
        card.run(&[tasks::output_gate(ab.at(), gb.at(), yb.at(), n as u32, qd as u32)]);
        yb.read(card.card(), n * qd)
    });
    let want: Vec<f32> = attn.iter().zip(&gate).map(|(a, g)| a * ops::sigmoid(*g)).collect();
    gap("the output gate", &got, &want, qd, TOL_OUTPUT_GATE);
}

/// One convolution on the host over `n` positions from `state`: the outputs, plus `residual` when it is given, and
/// the state after the last position.
fn conv_on_host(
    inputs: &[f32],
    stride: usize,
    c: usize,
    weights: &[f32],
    dynamic: bool,
    mut state: Vec<f32>,
    residual: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let n = inputs.len() / stride;
    let mut out = vec![0f32; n * c];
    for t in 0..n {
        let r = &inputs[t * stride..(t + 1) * stride];
        let o = &mut out[t * c..(t + 1) * c];
        if dynamic {
            ops::dynamic_conv(&state, &r[..c], weights, &r[c..c + c / HEAD_DIM * CONV_TAPS], o);
        } else {
            ops::short_conv(&state, &r[..c], weights, o);
        }
        ops::conv_push(&mut state, &r[..c]);
        if let Some(res) = residual {
            for (v, x) in o.iter_mut().zip(&res[t * c..(t + 1) * c]) {
                *v += x;
            }
        }
    }
    (out, state)
}

/// One convolution on the card, in one task for each part of `split`. The state sits at column `offset` of rows of
/// `state_stride`, as a part of the conv state of a run does. Returns the outputs, the state, and the whole state rows.
#[allow(clippy::too_many_arguments)]
fn conv_on_card(
    card: &mut Card,
    inputs: &[f32],
    stride: usize,
    c: usize,
    weights: &[u16],
    dynamic: bool,
    state_rows: &[f32],
    state_stride: usize,
    offset: usize,
    p0: usize,
    residual: Option<&[f32]>,
    split: &[usize],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n: usize = split.iter().sum();
    let (ib, wb, sb) = (card.buf(&inputs[..n * stride]), card.buf(weights), card.buf(state_rows));
    let yb = match residual {
        Some(r) => card.buf(&r[..n * c]),
        None => card.zeros(n * c * 4),
    };
    let taps = if dynamic {
        ConvTaps::Dynamic(wb.at())
    } else {
        ConvTaps::Fixed(wb.at())
    };
    let mut at = 0;
    for part in split {
        card.run(&[tasks::residual_conv(
            ib.at() + (at * stride * 4) as u64,
            stride as u32,
            taps,
            sb.at() + (offset * 4) as u64,
            state_stride as u32,
            yb.at() + (at * c * 4) as u64,
            *part as u32,
            c as u32,
            (p0 + at) as u32,
            residual.is_some(),
        )]);
        at += part;
    }
    let out = yb.read(card.card(), n * c);
    let all: Vec<f32> = sb.read(card.card(), 3 * state_stride);
    let state = (0..3)
        .flat_map(|j| all[j * state_stride + offset..j * state_stride + offset + c].to_vec())
        .collect();
    (out, state, all)
}

/// The checks of one convolution: the block sizes, the splits, the state after the block, and the host form.
fn check_conv(card: &mut Card, what: &str, c: usize, dynamic: bool, p0: usize, with_residual: bool) {
    let stride = if dynamic { c + c / HEAD_DIM * CONV_TAPS } else { c };
    // The taps of a dynamic row are small, as the projection of a normed row gives.
    let inputs = rows(1000 + c as u32 + dynamic as u32, MOST, stride, 2.0);
    let (wb, w) = bf16s(1100 + c as u32, CONV_TAPS * c, 0.6);
    // The state of a conv part: 3 rows of 6,656 values, with this convolution at an offset. The other columns hold
    // values that the op must not change.
    let state_stride = 6656;
    let offset = 1024;
    let state_rows = values(1200, 3 * state_stride, 1.5);
    let state: Vec<f32> = (0..3)
        .flat_map(|j| state_rows[j * state_stride + offset..j * state_stride + offset + c].to_vec())
        .collect();
    let residual = with_residual.then(|| rows(1300, MOST, c, 4.0));
    let res = residual.as_deref();
    let run = |card: &mut Card, split: &[usize]| {
        conv_on_card(
            card,
            &inputs,
            stride,
            c,
            &wb,
            dynamic,
            &state_rows,
            state_stride,
            offset,
            p0,
            res,
            split,
        )
    };
    // From p0 0 the op reads the state as zeros, whatever the state holds.
    let start = if p0 == 0 { vec![0f32; 3 * c] } else { state };
    let (whole, whole_state, all) = run(card, &[MOST]);
    for n in BLOCKS {
        let (out, after, _) = run(card, &[n]);
        let (want, want_state) = conv_on_host(&inputs[..n * stride], stride, c, &w, dynamic, start.clone(), res);
        assert!(
            bits(&after) == bits(&want_state),
            "{what}: the state after {n} positions is not the last inputs"
        );
        assert!(
            bits(&out) == bits(&whole[..n * c]),
            "{what}: a block of {n} and a block of {MOST} give other bits"
        );
        gap(&format!("{what}, a block of {n}"), &out, &want, c, TOL_CONV);
    }
    for split in [&[1usize, 1, 1, 34][..], &[2, 3, 32], &[5, 32], &[36, 1]] {
        let (out, state, _) = run(card, split);
        assert!(
            bits(&out) == bits(&whole),
            "{what}: the split {split:?} gives other bits than one block"
        );
        assert!(
            bits(&state) == bits(&whole_state),
            "{what}: the split {split:?} leaves another state"
        );
    }
    for j in 0..3 {
        for (i, v) in all[j * state_stride..(j + 1) * state_stride].iter().enumerate() {
            if !(offset..offset + c).contains(&i) {
                assert_eq!(
                    v.to_bits(),
                    state_rows[j * state_stride + i].to_bits(),
                    "{what}: column {i} of the state"
                );
            }
        }
    }
}

#[test]
fn the_convolutions_on_the_card_give_the_host_form_in_any_block_and_any_split() {
    let mut card = Card::open();
    let kv = KV_HEADS * HEAD_DIM;
    check_conv(
        &mut card,
        "the fixed key convolution from a state",
        kv,
        false,
        100,
        false,
    );
    check_conv(
        &mut card,
        "the fixed key convolution from position 0",
        kv,
        false,
        0,
        false,
    );
    check_conv(
        &mut card,
        "the dynamic key convolution from a state",
        kv,
        true,
        100,
        false,
    );
    check_conv(
        &mut card,
        "the dynamic key convolution from position 0",
        kv,
        true,
        0,
        false,
    );
    check_conv(
        &mut card,
        "the convolution after the attention, with the residual",
        HIDDEN,
        false,
        100,
        true,
    );
}

fn decisions(card: &mut Card, logits: &Buf, bias: &Buf, n: usize, layer: u32) -> Vec<MoERouteDecision> {
    let db = card.zeros(n * std::mem::size_of::<MoERouteDecision>());
    card.run(&[tasks::sigmoid_router(logits.at(), bias.at(), db.at(), n as u32, layer)]);
    db.read(card.card(), n)
}

#[test]
fn the_router_on_the_card_gives_the_picks_of_the_host_in_expert_order() {
    let mut card = Card::open();
    let logits = rows(1400, MOST, EXPERTS, 4.0);
    let bias = values(1401, EXPERTS, 0.3);
    let (lb, bb) = (card.buf(&logits), card.buf(&bias));
    let flat = |d: &[MoERouteDecision]| -> Vec<f32> {
        d.iter()
            .flat_map(|r| {
                r.top_k_indices
                    .iter()
                    .map(|i| f32::from_bits(*i))
                    .chain(r.top_k_weights)
            })
            .collect()
    };
    let got = same_in_each_block("the router", 2 * TOP_K, |n| flat(&decisions(&mut card, &lb, &bb, n, 7)));
    let all = decisions(&mut card, &lb, &bb, MOST, 7);
    let mut worst = 0f64;
    for (t, d) in all.iter().enumerate() {
        let want = ops::route(row(&logits, t, EXPERTS), &bias);
        let ids: Vec<u32> = want.iter().map(|w| w.0 as u32).collect();
        assert_eq!(d.top_k_indices.to_vec(), ids, "position {t}: the picks");
        let ws: Vec<f32> = want.iter().map(|w| w.1).collect();
        worst = worst.max(common::rel(&d.top_k_weights, &ws));
        assert_eq!(
            (d.layer_idx, d.token),
            (7, t as u32),
            "position {t}: the layer and the token"
        );
    }
    eprintln!(
        "the router: the worst weights are {worst:.3e} from the host form (relative L2, tolerance {TOL_ROUTE:e})"
    );
    assert!(worst <= TOL_ROUTE, "the router: {worst:.3e} > {TOL_ROUTE:e}");
    assert_eq!(got.len(), MOST * 2 * TOP_K);

    // Equal keys go to the lower index. Row 0: all equal. Row 1: four experts above the rest, then the lowest.
    let mut tied = vec![0.5f32; 2 * EXPERTS];
    for (e, v) in [(90, 9.0f32), (3, 8.0), (127, 7.0), (64, 6.0)] {
        tied[EXPERTS + e] = v;
    }
    let (tb, zb) = (card.buf(&tied), card.buf(&[0f32; EXPERTS]));
    let d = decisions(&mut card, &tb, &zb, 2, 0);
    assert_eq!(d[0].top_k_indices, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(d[1].top_k_indices, [0, 1, 2, 3, 4, 64, 90, 127]);
    for (t, r) in d.iter().enumerate() {
        let want = ops::route(row(&tied, t, EXPERTS), &[0f32; EXPERTS]);
        assert_eq!(
            r.top_k_indices.to_vec(),
            want.iter().map(|w| w.0 as u32).collect::<Vec<_>>()
        );
    }
}
