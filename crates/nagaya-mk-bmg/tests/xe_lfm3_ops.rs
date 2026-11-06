//! The ops of the LFM3 binary on a B70 card (milestone 24 step 5): the gated short convolution, the partial rotation,
//! the sliding attention with a ring, and the gate of the heads. Each test compares the kernel with the fixture of
//! plain torch (`tools/lfm3_op_fixture.py`) and with the CPU forms of `common/lfm3.rs`, which `lfm3_cpu_forms.rs`
//! checks on the host. The ops of the LFM3 backbone give the same bits for any split of a feed, and the tests check it.
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-bmg --features xe --test xe_lfm3_ops -- --test-threads=1`.
#![cfg(feature = "xe")]

mod common;

use std::ffi::c_void;

use common::lfm3::*;
use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::*;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::tasks;
use nagaya_mk_bmg::{KERNEL_ENTRY, LFM3_KERNEL};

/// A buffer in device memory, which the test fills and reads through host memory.
struct Buf {
    ptr: *mut u8,
    bytes: usize,
}

impl Buf {
    fn zeros(card: &B70Card, bytes: usize) -> Self {
        let ptr = card.alloc_device_usm::<u8>(bytes.max(64));
        card.fill_zero_sync(ptr as *mut c_void, bytes.max(64));
        Self { ptr, bytes }
    }

    fn of<T: Copy>(card: &B70Card, data: &[T]) -> Self {
        let bytes = std::mem::size_of_val(data);
        let buf = Self::zeros(card, bytes);
        buf.write(card, data);
        buf
    }

    fn write<T: Copy>(&self, card: &B70Card, data: &[T]) {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= self.bytes.max(64));
        let stage = card.alloc_host_usm::<u8>(bytes.max(64));
        // SAFETY: `stage` holds `bytes` bytes of host USM, and `data` is `bytes` bytes long.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, stage, bytes) };
        card.memcpy_sync(self.ptr as *mut c_void, stage as *const c_void, bytes);
        card.free(stage as *mut c_void);
    }

    fn read(&self, card: &B70Card, n: usize) -> Vec<f32> {
        let bytes = n * 4;
        assert!(bytes <= self.bytes.max(64));
        let stage = card.alloc_host_usm::<u8>(bytes.max(64));
        card.memcpy_sync(stage as *mut c_void, self.ptr as *const c_void, bytes);
        let mut out = vec![0.0f32; n];
        // SAFETY: `stage` holds `bytes` bytes, and `out` has room for them.
        unsafe { std::ptr::copy_nonoverlapping(stage, out.as_mut_ptr() as *mut u8, bytes) };
        card.free(stage as *mut c_void);
        out
    }

    fn at(&self) -> u64 {
        self.ptr as u64
    }

    fn free(self, card: &B70Card) {
        card.free(self.ptr as *mut c_void);
    }
}

struct Card {
    cluster: L0Cluster,
    ring: Ring,
}

impl Card {
    fn open() -> Self {
        let cluster = L0Cluster::init(LFM3_KERNEL, KERNEL_ENTRY);
        assert!(!cluster.cards.is_empty(), "no Level Zero card");
        let ring = Ring::new(&cluster.cards[0], 8 << 20);
        Self { cluster, ring }
    }

    fn card(&self) -> &B70Card {
        &self.cluster.cards[0]
    }

    fn run(&mut self, tasks: &[MkTaskDescriptor]) {
        self.ring.set(tasks);
        // SAFETY: each test sizes its buffers for its tasks.
        unsafe { self.ring.run(&self.cluster.cards[0]) };
    }

    fn buf<T: Copy>(&self, data: &[T]) -> Buf {
        Buf::of(self.card(), data)
    }
}

/// The short convolution of `bcx` from `state` on the card, in one call for each part of `split` (the counts of
/// positions). Returns y and the state after the last part.
fn short_conv_on_card(
    card: &mut Card,
    bcx: &[f32],
    w: &[u16],
    state: &[f32],
    d: usize,
    taps: usize,
    split: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let t: usize = split.iter().sum();
    assert_eq!(bcx.len(), t * 3 * d);
    let (xb, wb, sb) = (card.buf(bcx), card.buf(w), card.buf(state));
    let yb = Buf::zeros(card.card(), t * d * 4);
    let mut p = 0usize;
    for &n in split {
        card.run(&[tasks::short_conv(
            xb.at() + (p * 3 * d * 4) as u64,
            wb.at(),
            sb.at(),
            yb.at() + (p * d * 4) as u64,
            n as u32,
            d as u32,
            taps as u32,
            false,
        )]);
        p += n;
    }
    let out = (yb.read(card.card(), t * d), sb.read(card.card(), (taps - 1) * d));
    for b in [xb, wb, sb, yb] {
        b.free(card.card());
    }
    out
}

#[test]
fn short_conv_matches_the_reference_and_its_cpu_form_to_the_bit() {
    let mut card = Card::open();
    let fx = Fixture::load();
    for (t, seed) in CONV_CASES {
        let (w, bcx, state) = conv_inputs(t, seed);
        let (y, next) = short_conv_on_card(&mut card, &bcx, &w, &state, CONV_D, CONV_TAPS, &[t]);
        let err = rel(&y, fx.get(&format!("short_conv.{t}.y")));
        assert!(err < 1e-6, "{t} positions: relative error {err} against the reference");
        let (cy, cs) = short_conv(&bcx, &w, &state, t, CONV_D, CONV_TAPS);
        assert_eq!(
            first_other_bits(&y, &cy),
            None,
            "{t} positions: the output differs from the CPU form"
        );
        assert_eq!(
            first_other_bits(&next, &cs),
            None,
            "{t} positions: the state differs from the CPU form"
        );
    }
}

/// A conv state that is wrong in one row moves only the outputs of the next `taps - 1` positions, so the test splits
/// the feed at each offset 1 to 6 from the end. It also splits at the chunks of the kernel, and decodes the tail.
#[test]
fn short_conv_gives_the_same_bits_for_any_split_of_a_feed() {
    let mut card = Card::open();
    // The width of the backbone, 2,048 channels, over 300 positions: 5 chunks of 64.
    let (d, taps, t) = (2048usize, CONV_TAPS, 300usize);
    let w: Vec<u16> = values(500, d * taps, 0.5).into_iter().map(bf16).collect();
    let bcx = values(501, t * 3 * d, 1.0);
    let state = values(502, (taps - 1) * d, 1.0);
    let (whole, whole_state) = short_conv_on_card(&mut card, &bcx, &w, &state, d, taps, &[t]);
    let (cy, cs) = short_conv(&bcx, &w, &state, t, d, taps);
    assert_eq!(
        first_other_bits(&whole, &cy),
        None,
        "one call differs from the CPU form"
    );
    assert_eq!(
        first_other_bits(&whole_state, &cs),
        None,
        "the state of one call differs from the CPU form"
    );
    let mut splits: Vec<Vec<usize>> = (1..=6).map(|back| vec![t - back, back]).collect();
    splits.extend([
        vec![1, t - 1],
        vec![63, 1, t - 64],
        vec![64, t - 64],
        vec![65, t - 65],
        vec![2, 3, t - 5],
    ]);
    // A prefill, then one position for each call: the decode of the last 20 positions.
    let mut decode = vec![t - 20];
    decode.extend(std::iter::repeat_n(1, 20));
    splits.push(decode);
    for split in splits {
        let (y, s) = short_conv_on_card(&mut card, &bcx, &w, &state, d, taps, &split);
        if let Some(i) = first_other_bits(&y, &whole) {
            panic!(
                "split {split:?}: position {} channel {} differs from one call",
                i / d,
                i % d
            );
        }
        assert_eq!(
            first_other_bits(&s, &whole_state),
            None,
            "split {split:?}: the state differs from one call"
        );
    }
}

#[test]
fn short_conv_reads_f32_weights_and_the_widths_from_2_to_8() {
    let mut card = Card::open();
    let (d, t) = (32usize, 11usize);
    for taps in [2usize, 3, 4, 7, 8] {
        let wf = values(600 + taps as u64, d * taps, 0.5);
        let w16: Vec<u16> = wf.iter().map(|v| bf16(*v)).collect();
        // The F32 weights hold BF16 values, so they give the bits of the BF16 weights.
        let wr: Vec<f32> = w16.iter().map(|b| from_bf16(*b)).collect();
        let bcx = values(610 + taps as u64, t * 3 * d, 1.0);
        let state = values(620 + taps as u64, (taps - 1) * d, 1.0);
        let (wb, xb, sb) = (card.buf(&wr), card.buf(&bcx), card.buf(&state));
        let yb = Buf::zeros(card.card(), t * d * 4);
        card.run(&[tasks::short_conv(
            xb.at(),
            wb.at(),
            sb.at(),
            yb.at(),
            t as u32,
            d as u32,
            taps as u32,
            true,
        )]);
        let (cy, cs) = short_conv(&bcx, &w16, &state, t, d, taps);
        assert_eq!(
            first_other_bits(&yb.read(card.card(), t * d), &cy),
            None,
            "{taps} taps: the output"
        );
        assert_eq!(
            first_other_bits(&sb.read(card.card(), (taps - 1) * d), &cs),
            None,
            "{taps} taps: the state"
        );
        for b in [wb, xb, sb, yb] {
            b.free(card.card());
        }
    }
}

#[test]
fn qk_rope_rotates_half_the_head_on_full_layers_and_all_of_it_on_sliding_layers() {
    let mut card = Card::open();
    let fx = Fixture::load();
    for (name, theta, pairs, seed) in ROPE_CASES {
        let (x, w) = rope_inputs(seed);
        let n = ROPE_TOKENS * ROPE_HEADS;
        let full = theta == 1e6;
        let rope = |x: u64, w: u64, n: u32, p0: u32, scale: f32| {
            let task = tasks::qk_rope(
                x,
                w,
                n,
                ROPE_HEADS as u32,
                ROPE_D as u32,
                p0,
                full,
                true,
                ROPE_EPS,
                scale,
            );
            tasks::partial_rotation(task, pairs as u32)
        };
        let (xb, wb) = (card.buf(&x), card.buf(&w));
        card.run(&[rope(xb.at(), wb.at(), n as u32, ROPE_P0, 1.0)]);
        let got = xb.read(card.card(), n * ROPE_D);
        let err = rel(&got, fx.get(&format!("rope.{name}")));
        assert!(
            err < 1e-3,
            "the {name} layers: relative error {err} against the reference"
        );
        let cpu = norm_rope(&x, &w, ROPE_HEADS, ROPE_D, ROPE_P0, theta, pairs, ROPE_EPS, 1.0);
        let err = rel(&got, &cpu);
        // At the last position one ulp of an f32 angle is 2^-12 rad (4,095 is in [2^11, 2^12)). One ulp in `pow`,
        // the multiply or `sin` turns a value by that much. Thus the bound is two of those ulps, not a bound of f32 noise.
        let angle_ulp = f32::EPSILON * ((ROPE_P0 + ROPE_TOKENS as u32 - 1) as f32).log2().floor().exp2();
        println!("the {name} layers: relative error {err} against the CPU form; one angle ulp {angle_ulp}");
        assert!(
            err < 2.0 * angle_ulp,
            "the {name} layers: relative error {err} against the CPU form"
        );
        // The pairs past the rotated ones keep the normed values: the rotation leaves them as they are.
        let half = ROPE_D / 2;
        for r in 0..n {
            for i in (pairs..half).chain(half + pairs..ROPE_D) {
                let (g, c) = (got[r * ROPE_D + i], cpu[r * ROPE_D + i]);
                assert!(
                    (g - c).abs() <= 1e-6 * c.abs().max(1.0),
                    "the {name} layers: row {r} value {i} turned"
                );
            }
        }
        // The scale of the queries, and the same bits for a position alone and in a block.
        xb.write(card.card(), &x);
        card.run(&[rope(xb.at(), wb.at(), n as u32, ROPE_P0, 128f32.powf(-0.5))]);
        let block = xb.read(card.card(), n * ROPE_D);
        // The card's own rotation without the scale: thus the check sees the scale alone, not the angle ulps.
        let want: Vec<f32> = got.iter().map(|v| v * 128f32.powf(-0.5)).collect();
        let err = rel(&block, &want);
        assert!(err < 1e-5, "the {name} layers with the scale: relative error {err}");
        for tok in 0..ROPE_TOKENS {
            let row = ROPE_HEADS * ROPE_D;
            let one = card.buf(&x[tok * row..(tok + 1) * row]);
            card.run(&[rope(
                one.at(),
                wb.at(),
                ROPE_HEADS as u32,
                ROPE_P0 + tok as u32,
                128f32.powf(-0.5),
            )]);
            let alone = one.read(card.card(), row);
            assert_eq!(
                first_other_bits(&alone, &block[tok * row..(tok + 1) * row]),
                None,
                "the {name} layers: position {} alone differs from the block",
                ROPE_P0 as usize + tok
            );
            one.free(card.card());
        }
        xb.free(card.card());
        wb.free(card.card());
    }
}

/// A KV cache of `rows` rows for keys `k` and values `v` (`[keys][kv][d]`): key j sits at row j, or at row j % ring
/// for the keys `from..` with a ring. The value plane starts `plane` rows after the key plane.
fn cache(k: &[f32], v: &[f32], kv: usize, d: usize, plane: usize, ring: usize, from: usize) -> Vec<f32> {
    let keys = k.len() / (kv * d);
    let mut out = vec![0.0f32; 2 * plane * kv * d];
    for j in from..keys {
        let row = if ring > 0 { j % ring } else { j };
        let at = row * kv * d;
        out[at..at + kv * d].copy_from_slice(&k[j * kv * d..(j + 1) * kv * d]);
        let at = (plane + row) * kv * d;
        out[at..at + kv * d].copy_from_slice(&v[j * kv * d..(j + 1) * kv * d]);
    }
    out
}

/// Block attention of the queries `q` of positions `p0..p0 + n` over a cache on the card. Returns `[n][h][d]`.
#[allow(clippy::too_many_arguments)]
fn attend_block_on_card(
    card: &mut Card,
    q: &[f32],
    cache: &[f32],
    n: usize,
    p0: usize,
    max_seq: usize,
    ring: usize,
    window: usize,
) -> Vec<f32> {
    let (h, kv, d) = (ATTN_H, ATTN_KV, ATTN_D);
    let (qb, cb) = (card.buf(q), card.buf(cache));
    let sb = Buf::zeros(card.card(), tasks::score_floats(h as u32, n as u32, p0 as u32) * 4);
    let ob = Buf::zeros(card.card(), n * h * d * 4);
    let mut ops = tasks::attend_block(
        qb.at(),
        cb.at(),
        sb.at(),
        ob.at(),
        n as u32,
        h as u32,
        kv as u32,
        d as u32,
        p0 as u32,
        max_seq as u32,
        window as u32,
    );
    for op in &mut ops {
        op.layer = ring as u32;
    }
    card.run(&ops);
    let out = ob.read(card.card(), n * h * d);
    for b in [qb, cb, sb, ob] {
        b.free(card.card());
    }
    out
}

/// One-token attention of the query `q` at position `p0` over a cache on the card. Returns `[h][d]`.
fn attend_one_on_card(
    card: &mut Card,
    q: &[f32],
    cache: &[f32],
    p0: usize,
    max_seq: usize,
    ring: usize,
    window: usize,
) -> Vec<f32> {
    let (h, kv, d) = (ATTN_H, ATTN_KV, ATTN_D);
    let (qb, cb) = (card.buf(q), card.buf(cache));
    let ob = Buf::zeros(card.card(), h * d * 4);
    let mut ops = tasks::attend_one(
        qb.at(),
        cb.at(),
        ob.at(),
        h as u32,
        kv as u32,
        d as u32,
        p0 as u32,
        max_seq as u32,
        window as u32,
    );
    for op in &mut ops {
        op.layer = ring as u32;
    }
    card.run(&ops);
    let out = ob.read(card.card(), h * d);
    for b in [qb, cb, ob] {
        b.free(card.card());
    }
    out
}

/// The sliding layers of LFM3 attend to the last 256 positions (`kv_idx > q_idx - sliding_window` in transformers),
/// and the full layers to all. Both paths of the interpreter, the block and the one token, have heads of 128 here.
#[test]
fn attention_matches_the_reference_with_a_window_of_256_on_both_paths() {
    let mut card = Card::open();
    let fx = Fixture::load();
    let (q, k, v) = attn_inputs();
    let (h, kv, d, t) = (ATTN_H, ATTN_KV, ATTN_D, ATTN_T);
    let max_seq = 320usize;
    let flat = cache(&k, &v, kv, d, max_seq, 0, 0);
    for (name, window) in ATTN_CASES {
        let rows = fx.rows(name);
        let want = fx.get(&format!("attention.{name}"));
        let block = attend_block_on_card(&mut card, &q, &flat, t, 0, max_seq, 0, window);
        let got: Vec<f32> = rows
            .iter()
            .flat_map(|&r| block[r * h * d..(r + 1) * h * d].to_vec())
            .collect();
        let err = rel(&got, want);
        assert!(err < 1e-5, "the {name} layers, one block: relative error {err}");
        let mut one = Vec::new();
        for &r in &rows {
            one.extend(attend_one_on_card(
                &mut card,
                &q[r * h * d..(r + 1) * h * d],
                &flat,
                r,
                max_seq,
                0,
                window,
            ));
        }
        let err = rel(&one, want);
        assert!(err < 1e-5, "the {name} layers, one token: relative error {err}");
    }
}

/// A sliding layer of a long slot keeps its keys in a ring (spec 11.11, "The held state of LFM3"). A ring holds the
/// same values as a flat cache at other rows, so the same block gives the same bits.
#[test]
fn a_ring_gives_the_bits_of_a_flat_cache_with_a_window_of_256() {
    let mut card = Card::open();
    let (h, kv, d, window) = (ATTN_H, ATTN_KV, ATTN_D, 256usize);
    // 600 positions; the last block of 32 starts at 568 and reads the keys 313 to 599.
    let (keys, n, p0) = (600usize, 32usize, 568usize);
    // A ring of 288 rows holds the window and the block: 288 >= 256 + 32, in rows of 32.
    let ring = 288usize;
    let q = values(700, n * h * d, 1.0);
    let k = values(701, keys * kv * d, 1.0);
    let v = values(702, keys * kv * d, 1.0);
    let max_seq = 608usize;
    let flat = cache(&k, &v, kv, d, max_seq, 0, 0);
    let ringed = cache(&k, &v, kv, d, ring + 1024, ring, keys - ring);
    let a = attend_block_on_card(&mut card, &q, &flat, n, p0, max_seq, 0, window);
    let b = attend_block_on_card(&mut card, &q, &ringed, n, p0, max_seq, ring, window);
    assert_eq!(
        first_other_bits(&b, &a),
        None,
        "the block over a ring differs from the flat cache"
    );
    // The CPU form indexes the queries by position, so the block's queries sit at p0.. of a longer array.
    let mut qp = vec![0.0f32; (p0 + n) * h * d];
    qp[p0 * h * d..].copy_from_slice(&q);
    let mut want = Vec::new();
    for t in 0..n {
        for head in 0..h {
            want.extend(attend_row(&qp, &k, &v, p0 + t, head, h, kv, d, window));
        }
    }
    let err = rel(&a, &want);
    assert!(err < 1e-5, "the block over the window: relative error {err}");
    let last = &q[(n - 1) * h * d..];
    let one_flat = attend_one_on_card(&mut card, last, &flat, keys - 1, max_seq, 0, window);
    let one_ring = attend_one_on_card(&mut card, last, &ringed, keys - 1, max_seq, ring, window);
    assert_eq!(
        first_other_bits(&one_ring, &one_flat),
        None,
        "one token over a ring differs from the flat cache"
    );
}

#[test]
fn head_gate_matches_the_reference_in_place() {
    let mut card = Card::open();
    let fx = Fixture::load();
    let (a, g) = gate_inputs();
    let (ab, gb) = (card.buf(&a), card.buf(&g));
    card.run(&[tasks::head_gate(
        ab.at(),
        gb.at() + (GATE_OFFSET * 4) as u64,
        ab.at(),
        GATE_T as u32,
        GATE_H as u32,
        GATE_D as u32,
        GATE_STRIDE as u32,
    )]);
    let got = ab.read(card.card(), a.len());
    let err = rel(&got, fx.get("head_gate"));
    assert!(err < 1e-6, "relative error {err} against the reference");
    let err = rel(&got, &head_gate(&a, &g, GATE_H, GATE_D, GATE_STRIDE, GATE_OFFSET));
    assert!(err < 1e-6, "relative error {err} against the CPU form");
}
