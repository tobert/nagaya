//! The attention of Inkling 2 on a B70 card against the host form (engine-abi "The first Inkling 2 binding", "The
//! attention"). `OP_INKLING2_ATTN` and `OP_INKLING2_ATTN_SUM` take synthetic queries, keys, values, relative states,
//! proj and sinks, and `host::ops::attend` with `relative_logits` and `relative_bias` gives the oracle.
//!
//! The tests need no model file. Run them alone, on a card that no engine of the active profile names (`AGENTS.md`,
//! "The engines on the B70 cards"):
//! `NAGAYA_TEST_CARDS=<pci> cargo test -p nagaya-mk-inkling2-bmg --features xe --test xe_attention -- --test-threads=1`.
#![cfg(feature = "xe")]

use std::ffi::c_void;

use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::tasks::{self, Inkling2Attention};
use nagaya_mk_bmg::{INKLING2_KERNEL, KERNEL_ENTRY};
use nagaya_mk_inkling2_bmg::host::ops::{self, D_REL, HEAD_DIM, HEADS, KV_HEADS, REL_EXTENT};
use rayon::prelude::*;

const Q_ROW: usize = HEADS * HEAD_DIM;
const KV_ROW: usize = KV_HEADS * HEAD_DIM;
const REL_ROW: usize = HEADS * D_REL;
/// The rows of the ring of a sliding layer, and the spill rows after it (engine-abi "Inkling 2", "The parts").
const RING: usize = 2048;
const SPILL: usize = 1024;
/// The unit roundoff of `f32`.
const U: f32 = 1.0 / 16_777_216.0;

/// A buffer in device memory.
struct Buf(*mut u8, usize);

impl Buf {
    fn of(card: &B70Card, data: &[u8]) -> Self {
        let ptr = card.alloc_device_usm::<u8>(data.len().max(64));
        let stage = card.alloc_host_usm::<u8>(data.len().max(64));
        // SAFETY: `stage` holds `data.len()` bytes of host memory.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), stage, data.len()) };
        card.memcpy_sync(ptr as *mut c_void, stage as *const c_void, data.len());
        card.free(stage as *mut c_void);
        Self(ptr, data.len())
    }

    fn of_f32(card: &B70Card, data: &[f32]) -> Self {
        Self::of(card, &data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
    }

    fn zeros(card: &B70Card, bytes: usize) -> Self {
        let ptr = card.alloc_device_usm::<u8>(bytes.max(64));
        card.fill_zero_sync(ptr as *mut c_void, bytes.max(64));
        Self(ptr, bytes)
    }

    fn read_f32(&self, card: &B70Card) -> Vec<f32> {
        let stage = card.alloc_host_usm::<u8>(self.1.max(64));
        card.memcpy_sync(stage as *mut c_void, self.0 as *const c_void, self.1);
        let mut out = vec![0f32; self.1 / 4];
        // SAFETY: `stage` holds `self.1` bytes, and `out` has room for them.
        unsafe { std::ptr::copy_nonoverlapping(stage, out.as_mut_ptr() as *mut u8, self.1) };
        card.free(stage as *mut c_void);
        out
    }

    fn at(&self) -> u64 {
        self.0 as u64
    }

    fn free(self, card: &B70Card) {
        card.free(self.0 as *mut c_void);
    }
}

/// splitmix64: the same seed gives the same inputs on each run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 8_388_608.0 - 1.0
    }

    /// A head of `len` values with an L2 norm from `lo` to `hi`: the clip leaves each norm at 22.63 or below.
    fn head(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        let v: Vec<f32> = (0..len).map(|_| self.unit()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let target = lo + (hi - lo) * (self.unit() + 1.0) / 2.0;
        v.iter().map(|x| x * target / norm).collect()
    }
}

/// The inputs of one run at each position of a context: the row of each position does not depend on the split.
struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    rel: Vec<f32>,
    sink: Vec<f32>,
    /// proj `[16][1024]` as the bf16 bits of the checkpoint, and their f32 values for the host form.
    proj_bits: Vec<u16>,
    proj: Vec<f32>,
}

impl Inputs {
    fn random(positions: usize, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let mut x = Self {
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            rel: Vec::new(),
            sink: Vec::new(),
            proj_bits: Vec::new(),
            proj: Vec::new(),
        };
        for _ in 0..positions {
            for _ in 0..HEADS {
                x.q.extend(rng.head(HEAD_DIM, 4.0, ops::QK_CLIP));
                x.rel.extend(rng.head(D_REL, 1.0, ops::REL_CLIP));
                x.sink.push(3.0 * rng.unit());
            }
            for _ in 0..KV_HEADS {
                x.k.extend(rng.head(HEAD_DIM, 4.0, ops::QK_CLIP));
                x.v.extend((0..HEAD_DIM).map(|_| rng.unit()));
            }
        }
        x.proj_bits = (0..D_REL * REL_EXTENT)
            .map(|_| (rng.unit().to_bits() >> 16) as u16)
            .collect();
        x.proj = x.proj_bits.iter().map(|&b| ops::bf16_to_f32(b)).collect();
        x
    }

    /// The bias at the distance 1,023 dominates each score that has it: each relative state is 0.5, and each value
    /// of proj at that distance is 2, so the bias there is 16. Thus a key at a distance of 1,023 or 1,024 that the
    /// op reads or drops by mistake moves the output by a large amount.
    fn with_a_large_bias_at_1023(mut self) -> Self {
        self.rel.iter_mut().for_each(|r| *r = 0.5);
        for s in 0..D_REL {
            self.proj_bits[s * REL_EXTENT + REL_EXTENT - 1] = (2f32.to_bits() >> 16) as u16;
        }
        self.proj = self.proj_bits.iter().map(|&b| ops::bf16_to_f32(b)).collect();
        self
    }
}

/// The layout of the keys and values of one run: cells of `max_seq` rows, or a ring (a sliding layer).
#[derive(Clone, Copy)]
enum Layout {
    Cells(usize),
    Ring,
}

impl Layout {
    fn plane_rows(self) -> usize {
        match self {
            Layout::Cells(max_seq) => max_seq,
            Layout::Ring => RING + SPILL,
        }
    }

    fn ring(self) -> u32 {
        match self {
            Layout::Cells(_) => 0,
            Layout::Ring => RING as u32,
        }
    }

    fn max_seq(self) -> u32 {
        match self {
            Layout::Cells(max_seq) => max_seq as u32,
            Layout::Ring => 0,
        }
    }

    fn lowest(self, pos: usize) -> usize {
        match self {
            Layout::Cells(_) => 0,
            Layout::Ring => (pos + 1).saturating_sub(tasks::INKLING2_WINDOW as usize),
        }
    }
}

/// The planes of keys and values after positions 0 to `written - 1`, in order. A ring row holds the last position
/// that went into it, as after the spill copy of each block.
fn planes(x: &Inputs, layout: Layout, written: usize) -> Vec<f32> {
    let rows = layout.plane_rows();
    let mut out = vec![0f32; 2 * rows * KV_ROW];
    for pos in 0..written {
        let row = match layout {
            Layout::Cells(_) => pos,
            Layout::Ring => pos % RING,
        };
        out[row * KV_ROW..(row + 1) * KV_ROW].copy_from_slice(&x.k[pos * KV_ROW..(pos + 1) * KV_ROW]);
        out[(rows + row) * KV_ROW..(rows + row + 1) * KV_ROW].copy_from_slice(&x.v[pos * KV_ROW..(pos + 1) * KV_ROW]);
    }
    out
}

/// The host form of positions `p0 .. p0 + n`, `[n][24][128]`, and the bound of the gap of each head.
fn host(x: &Inputs, layout: Layout, p0: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let scale = (HEAD_DIM as f32).powf(-0.5);
    let rows: Vec<(Vec<f32>, f32)> = (0..n * HEADS)
        .into_par_iter()
        .map(|i| {
            let (pos, h) = (p0 + i / HEADS, i % HEADS);
            let kvh = h / (HEADS / KV_HEADS);
            let mut profiles = vec![0f32; HEADS * REL_EXTENT];
            ops::relative_logits(&x.rel[pos * REL_ROW..(pos + 1) * REL_ROW], &x.proj, &mut profiles);
            let profile = &profiles[h * REL_EXTENT..(h + 1) * REL_EXTENT];
            let q = &x.q[pos * Q_ROW + h * HEAD_DIM..pos * Q_ROW + (h + 1) * HEAD_DIM];
            let rel = &x.rel[pos * REL_ROW + h * D_REL..pos * REL_ROW + (h + 1) * D_REL];
            let (mut keys, mut values, mut bias) = (Vec::new(), Vec::new(), Vec::new());
            // The size of the largest score: the bound of the rounding of the sums of each score.
            let mut size = 0f32;
            let mut vmax = 0f32;
            for j in layout.lowest(pos)..=pos {
                let at = j * KV_ROW + kvh * HEAD_DIM;
                let (k, v) = (&x.k[at..at + HEAD_DIM], &x.v[at..at + HEAD_DIM]);
                keys.push(k);
                values.push(v);
                let d = (pos - j) as i64;
                bias.push(ops::relative_bias(profile, d));
                let qk: f32 = q.iter().zip(k).map(|(a, b)| (a * b).abs()).sum::<f32>() * scale;
                let b: f32 = if d < REL_EXTENT as i64 {
                    (0..D_REL)
                        .map(|s| (rel[s] * x.proj[s * REL_EXTENT + d as usize]).abs())
                        .sum()
                } else {
                    0.0
                };
                size = size.max(qk + b);
                vmax = v.iter().fold(vmax, |m, a| m.max(a.abs()));
            }
            let sink = x.sink[pos * HEADS + h];
            size = size.max(sink.abs());
            let mut out = vec![0f32; HEAD_DIM];
            ops::attend(q, &keys, &values, &bias, sink, &mut out);
            (out, bound(size, vmax, keys.len()))
        })
        .collect();
    let mut out = Vec::with_capacity(n * Q_ROW);
    let mut bounds = Vec::with_capacity(n * HEADS);
    for (o, b) in rows {
        out.extend(o);
        bounds.push(b);
    }
    (out, bounds)
}

/// The bound of the gap between the card and the host form for one head, in `f32` with the unit roundoff `U`. It
/// counts the worst case of each rounding, so it is a provisional tolerance: the card run measures the real gap, and
/// the test prints it as a part of the bound.
///
/// - A score sums 128 products of q . k, a scale, and 16 products of the bias. The card and the host each round
///   each step, so a score differs by at most `2 (128 + 16 + 4) U size`.
/// - The XMX sums of three bf16 parts drop `mid * lo`, `lo * mid` and `lo * lo`. The parts are cut, not rounded,
///   so the drop is at most 8 U of each product (8 U of the size).
/// - `exp(s - max)` rounds the difference and the exponential (3 ulp on the card): at most `(2 size + 8) U` more of
///   each weight. The size counts the sink, which can be the max.
/// - A change `ds` of the scores moves a softmax output by at most `2 ds vmax`.
/// - The sums of the numerator and of the denominator over `keys` terms round once for each term, on each side:
///   `4 (keys + 16) U vmax`. The XMX sum of the probabilities and the values drops 8 U of each product.
/// - Each round and each chunk of the card scales its sums by `exp(old - new)`: 4 U for each of at most
///   `keys / 16 + 16` of them.
fn bound(size: f32, vmax: f32, keys: usize) -> f32 {
    let ds = 2.0 * (128.0 + 16.0 + 4.0) * U * size + 8.0 * U * size + (2.0 * size + 8.0) * U;
    let sums = 4.0 * (keys as f32 + 16.0) * U + 8.0 * U + 4.0 * (keys as f32 / 16.0 + 16.0) * U;
    vmax * (2.0 * ds + sums)
}

struct Card {
    cluster: L0Cluster,
}

impl Card {
    fn open() -> Self {
        Self {
            cluster: L0Cluster::init_test(INKLING2_KERNEL, KERNEL_ENTRY),
        }
    }

    fn card(&self) -> &B70Card {
        &self.cluster.cards[0]
    }

    /// The card's output of each block `(p0, n)` over the planes after `written` positions, `[n][24][128]` each.
    fn attend(
        &self,
        x: &Inputs,
        layout: Layout,
        written: usize,
        blocks: &[(usize, usize)],
        chunk_keys: u32,
    ) -> Vec<Vec<f32>> {
        let card = self.card();
        let kv = Buf::of_f32(card, &planes(x, layout, written));
        let proj = Buf::of(
            card,
            &x.proj_bits.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>(),
        );
        let mut outs = Vec::new();
        for &(p0, n) in blocks {
            assert!(
                p0 + n <= written,
                "the block {p0} + {n} reads keys that are not written"
            );
            // A ring row holds the last position written into it, so the block must not read an older one.
            assert!(
                matches!(layout, Layout::Cells(_)) || written.saturating_sub(RING) <= layout.lowest(p0),
                "the ring after {written} positions no longer holds the keys of the block {p0} + {n}"
            );
            let q = Buf::of_f32(card, &x.q[p0 * Q_ROW..(p0 + n) * Q_ROW]);
            let rel = Buf::of_f32(card, &x.rel[p0 * REL_ROW..(p0 + n) * REL_ROW]);
            let sinks = Buf::of_f32(card, &x.sink[p0 * HEADS..(p0 + n) * HEADS]);
            let out = Buf::zeros(card, n * Q_ROW * 4);
            let attention = Inkling2Attention {
                q: q.at(),
                kv: kv.at(),
                rel: rel.at(),
                proj: proj.at(),
                sinks: sinks.at(),
                out: out.at(),
                n: n as u32,
                p0: p0 as u32,
                max_seq: layout.max_seq(),
                ring: layout.ring(),
                chunk_keys,
            };
            let mut ring = Ring::new(card, attention.scratch_floats());
            ring.set(&attention.tasks());
            // SAFETY: the buffers hold the rows of the block, the planes hold `plane_rows` rows of each, and the
            // scratch holds `scratch_floats`.
            unsafe { ring.run(card) };
            outs.push(out.read_f32(card));
            ring.free(card);
            for buf in [q, rel, sinks, out] {
                buf.free(card);
            }
        }
        kv.free(card);
        proj.free(card);
        outs
    }
}

/// The largest gap of a block and its ratio to the bound; it fails on a gap past the bound of its head.
fn check(name: &str, card: &[f32], host: &[f32], bounds: &[f32], p0: usize) -> (f32, f32) {
    let (mut worst, mut ratio) = (0f32, 0f32);
    for (i, (c, h)) in card.iter().zip(host).enumerate() {
        let b = bounds[i / HEAD_DIM];
        let gap = (c - h).abs();
        assert!(
            gap <= b,
            "{name}: position {} head {} value {}: card {c}, host {h}, gap {gap} past the bound {b}",
            p0 + i / Q_ROW,
            (i / HEAD_DIM) % HEADS,
            i % HEAD_DIM
        );
        worst = worst.max(gap);
        ratio = ratio.max(gap / b);
    }
    eprintln!("{name}: largest gap {worst:.3e}, {ratio:.3e} of the bound");
    (worst, ratio)
}

/// The values whose bits differ between two outputs, and the largest gap.
fn other_bits(a: &[f32], b: &[f32]) -> (usize, f32) {
    assert_eq!(a.len(), b.len(), "two outputs of other lengths");
    let count = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let gap = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    (count, gap)
}

/// Each block against the host form.
fn gate(
    name: &str,
    card: &Card,
    x: &Inputs,
    layout: Layout,
    written: usize,
    blocks: &[(usize, usize)],
    chunk_keys: u32,
) {
    let outs = card.attend(x, layout, written, blocks, chunk_keys);
    for (&(p0, n), out) in blocks.iter().zip(&outs) {
        let (want, bounds) = host(x, layout, p0, n);
        check(&format!("{name}, block {p0} + {n}"), out, &want, &bounds, p0);
    }
}

/// A full layer reads every key from position 0. The contexts hold 1, 37 and 600 positions; 600 positions take a
/// block of 512 and a block of 88. One chunk of 2,048 keys, and chunks of 128 keys, so the merge joins 5 chunks.
///
/// Mutations to run on the card: a mask that drops key 0; a scale of the score of 1.0.
#[test]
fn a_full_layer_gives_the_host_form_at_1_37_and_600_positions() {
    let card = Card::open();
    let x = Inputs::random(600, 1);
    let layout = Layout::Cells(1024);
    for chunk_keys in [tasks::INKLING2_FULL_CHUNK_KEYS, 128] {
        let name = format!("full, chunks of {chunk_keys}");
        gate(&name, &card, &x, layout, 1, &[(0, 1)], chunk_keys);
        gate(&name, &card, &x, layout, 37, &[(0, 37), (36, 1)], chunk_keys);
        gate(
            &name,
            &card,
            &x,
            layout,
            600,
            &[(0, 512), (512, 88), (599, 1)],
            chunk_keys,
        );
    }
}

/// A block of 512 on a full layer that does not start at position 0.
#[test]
fn a_full_layer_gives_the_host_form_for_a_block_of_512() {
    let card = Card::open();
    let x = Inputs::random(612, 2);
    gate(
        "full",
        &card,
        &x,
        Layout::Cells(1024),
        612,
        &[(100, 512)],
        tasks::INKLING2_FULL_CHUNK_KEYS,
    );
}

/// A sliding layer past 2,048 positions: the ring wraps. A block of 512 crosses the end of the ring (positions 2,048
/// to 2,311 are in rows 0 to 263), decode steps read both sides of the end, and a block starts past the end.
///
/// Mutations to run on the card: a ring row of `j` in place of `j % ring`; a window of 1,025 keys.
#[test]
fn a_sliding_layer_gives_the_host_form_where_the_ring_wraps() {
    let card = Card::open();
    let x = Inputs::random(2600, 3);
    gate(
        "sliding",
        &card,
        &x,
        Layout::Ring,
        2600,
        &[(1800, 512), (2100, 1), (2599, 1), (2088, 512)],
        tasks::INKLING2_SLIDING_CHUNK_KEYS,
    );
}

/// The bias ends at a distance of 1,023, and a sliding layer reads 1,024 keys. With a bias of 16 at the distance
/// 1,023, position 1,023 reads key 0 with the bias, and position 1,024 reads key 0 with no bias on a full layer, and
/// not at all on a sliding layer.
///
/// Mutations to run on the card: a bias at distance 1,024 (the test `dist <= 1024`); a bias of 0 at 1,023.
#[test]
fn the_bias_and_the_window_end_at_the_distances_1023_and_1024() {
    let card = Card::open();
    let x = Inputs::random(1100, 4).with_a_large_bias_at_1023();
    let blocks = [(1016, 16), (1023, 1), (1024, 1), (1000, 100)];
    gate(
        "full, distance",
        &card,
        &x,
        Layout::Cells(2048),
        1100,
        &blocks,
        tasks::INKLING2_FULL_CHUNK_KEYS,
    );
    gate(
        "full, distance, chunks of 128",
        &card,
        &x,
        Layout::Cells(2048),
        1100,
        &blocks,
        128,
    );
    gate(
        "sliding, distance",
        &card,
        &x,
        Layout::Ring,
        1100,
        &blocks,
        tasks::INKLING2_SLIDING_CHUNK_KEYS,
    );
}

/// A sliding layer reads 1,024 keys: position 1,023 reads key 0, and position 1,024 does not. Each score is 0 and only
/// the values of key 0 are not 0, so an output is the weight of key 0, about 1 / 1,025, or exactly 0. The bias test
/// cannot show a key at the distance 1,024 that the window keeps by mistake, because the bias there is 0.
///
/// Mutations to run on the card: a window of 1,025 keys; a window of 1,023 keys.
#[test]
fn a_sliding_layer_reads_key_0_at_position_1023_and_not_at_1024() {
    let card = Card::open();
    let mut x = Inputs::random(1100, 6);
    x.q.iter_mut().for_each(|v| *v = 0.0);
    x.rel.iter_mut().for_each(|v| *v = 0.0);
    x.sink.iter_mut().for_each(|v| *v = 0.0);
    x.v.iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = if i < KV_ROW { 1.0 } else { 0.0 });
    let outs = card.attend(
        &x,
        Layout::Ring,
        1100,
        &[(1016, 16), (1023, 1), (1024, 1)],
        tasks::INKLING2_SLIDING_CHUNK_KEYS,
    );
    for (out, (p0, n)) in outs.iter().zip([(1016, 16), (1023, 1), (1024, 1)]) {
        let (want, bounds) = host(&x, Layout::Ring, p0, n);
        check(&format!("sliding, window, block {p0} + {n}"), out, &want, &bounds, p0);
    }
    // Position 1,023 is row 7 of the block of 16, and 1,024 is row 8.
    assert!(outs[0][7 * Q_ROW] > 1.0 / 2048.0, "position 1,023 did not read key 0");
    assert_eq!(outs[0][8 * Q_ROW], 0.0, "position 1,024 read key 0");
    assert_eq!(outs[2][0], 0.0, "the decode step at 1,024 read key 0");
}

/// The same block twice gives the same bits. Other splits of the same positions use the same grid of chunks, so the
/// op comment expects the same bits from them too: a split at a multiple of 16, a split at 100 positions, and decode
/// steps next to the ends of a key tile, a round, a chunk and the ring. The test prints the count of values of other
/// bits, and each output must stay within the bound of the host form. Another size of chunk is another split of the
/// keys, and the test prints its gap too.
#[test]
fn the_same_split_gives_the_same_bits_and_another_split_stays_in_the_bound() {
    let card = Card::open();
    let x = Inputs::random(2400, 5);
    let ck = tasks::INKLING2_SLIDING_CHUNK_KEYS;
    let (want, bounds) = host(&x, Layout::Ring, 1800, 512);
    let whole = card.attend(&x, Layout::Ring, 2400, &[(1800, 512), (1800, 512)], ck);
    assert_eq!(
        other_bits(&whole[0], &whole[1]).0,
        0,
        "the same block of a sliding layer gave other bits on a second run"
    );
    for (name, blocks) in [
        ("256 + 256", vec![(1800, 256), (2056, 256)]),
        ("100 + 412", vec![(1800, 100), (1900, 412)]),
    ] {
        let split: Vec<f32> = card.attend(&x, Layout::Ring, 2400, &blocks, ck).concat();
        let (count, gap) = other_bits(&whole[0], &split);
        eprintln!("sliding, 512 against {name}: {count} values of other bits, largest gap {gap:.3e}");
        check(&format!("sliding, {name}"), &split, &want, &bounds, 1800);
    }
    // The ends of a key tile (1,807 and 1,808), a round (1,919 and 1,920), a chunk of 256 (2,047 and 2,048, also
    // the end of the ring), and the last position.
    let decode_at = [1800, 1807, 1808, 1919, 1920, 2047, 2048, 2311];
    let steps = card.attend(&x, Layout::Ring, 2400, &decode_at.map(|p| (p, 1)), ck);
    let mut decode_other = 0;
    for (p, step) in decode_at.iter().zip(&steps) {
        let t = p - 1800;
        decode_other += other_bits(&whole[0][t * Q_ROW..(t + 1) * Q_ROW], step).0;
        check(
            &format!("sliding, decode at {p}"),
            step,
            &want[t * Q_ROW..(t + 1) * Q_ROW],
            &bounds[t * HEADS..(t + 1) * HEADS],
            *p,
        );
    }
    eprintln!("sliding, decode steps at {decode_at:?} against the block: {decode_other} values of other bits");

    let full = Layout::Cells(1024);
    let one = card.attend(&x, full, 600, &[(0, 512), (0, 512)], tasks::INKLING2_FULL_CHUNK_KEYS);
    assert_eq!(
        other_bits(&one[0], &one[1]).0,
        0,
        "the same block of a full layer gave other bits on a second run"
    );
    let small = card.attend(&x, full, 600, &[(0, 512)], 128);
    let (count, gap) = other_bits(&one[0], &small[0]);
    eprintln!("full, chunks of 2,048 against chunks of 128: {count} values of other bits, largest gap {gap:.3e}");
}
