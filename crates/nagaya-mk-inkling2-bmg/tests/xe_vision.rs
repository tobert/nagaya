//! The vision tower of Inkling 2 on a B70 card (engine-abi "The images of Inkling 2", "The choices of the tower"):
//! the new ops against the host form, each stage of the tower of image 0 against the oracle and the host form, the
//! projector rows of image 1, and the bits of an image alone against two images in one ring.
//!
//! The op gates need no model file. The tower gates need the Mini folder and the oracle of the stage `vision`
//! (`tools/inkling2_golden.sh --stages vision`), and each skips when a file is absent. Each gate prints its gaps.
//! The card gate has not run yet.
//!
//! Run it alone, on a card that no engine of the active profile names (`AGENTS.md`, "The engines on the B70 cards"):
//! `NAGAYA_TEST_CARDS=<pci> INKLING2_MODEL_DIR=<folder> cargo test -p nagaya-mk-inkling2-bmg --features xe --test
//! xe_vision -- --test-threads=1`.
#![cfg(feature = "xe")]

mod common;

use std::collections::HashMap;

use common::Which;
use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::MkTaskDescriptor;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::safetensors::MappedSafeTensors;
use nagaya_mk_bmg::tasks;
use nagaya_mk_bmg::{INKLING2_KERNEL, KERNEL_ENTRY};
use nagaya_mk_inkling2_bmg::device::{Device, OnCard};
use nagaya_mk_inkling2_bmg::host::ops::HIDDEN;
use nagaya_mk_inkling2_bmg::host::vision::{self, Grid, Vision, VisionTrace, WIDTH};
use nagaya_mk_inkling2_bmg::host::weights::Checkpoint;
use nagaya_mk_inkling2_bmg::vision::{self as tower, Bufs, Mark, Stage};

const ORACLE: &str = "inkling2-mini.vision.safetensors";

/// The position embedding: the card takes each bilinear weight through one division, which OpenCL C bounds by 2.5
/// ulp. A weight then moves by a few ulp (2.4e-7 for 4 ulp), and the sum of 4 taps and the patch row by as much.
const TOL_EMBED: f64 = 1e-6;
/// The first stages against the host form. The patch embedding is `OP_MATMUL`: its XMX sums take `x` in three
/// `bf16` parts, about 24 bits, and they sum 1,200 terms in another order. That is about sqrt(1200) * 6e-8 = 2e-6
/// of a row. The LayerNorm adds its `f32` mean and variance, where the host takes `f64`: a few ulp more.
const TOL_FRONT: f64 = 1e-5;
/// A layer adds about the gap of the front stages: 4 matmuls of 1,024 or 4,096 terms, the exp of the online softmax
/// and the erf of the card, each of a few ulp. 23 layers that add 2e-6 each give about 5e-5; the gate allows twice
/// that. The projector adds two matmuls of 4,096 terms.
const TOL_LAYERS: f64 = 1e-4;
/// The card against the oracle: the gap of the host form to the oracle (below 1e-4, `tests/host_vision.rs`) and the
/// gap of the card to the host form.
const TOL_ORACLE: f64 = 2e-4;

/// The card of a test, its ring, and the device memory of the test.
struct Card {
    cluster: L0Cluster,
    ring: Ring,
    device: Device,
}

impl Card {
    fn open() -> Self {
        let cluster = L0Cluster::init_test(INKLING2_KERNEL, KERNEL_ENTRY);
        let ring = Ring::new(&cluster.cards[0], 1 << 16);
        let device = Device::new(&cluster.cards[0]);
        Self { cluster, ring, device }
    }

    fn card(&self) -> &B70Card {
        &self.cluster.cards[0]
    }

    fn zeros(&mut self, bytes: usize) -> u64 {
        let card = &self.cluster.cards[0];
        self.device.zeros(card, bytes)
    }

    fn put(&mut self, values: &[f32]) -> u64 {
        let at = self.zeros(values.len() * 4);
        self.write(at, values);
        at
    }

    fn write(&self, at: u64, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device.write(self.card(), at, &bytes);
    }

    fn read(&self, at: u64, len: usize) -> Vec<f32> {
        self.device
            .read(self.card(), at, len * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn run(&mut self, tasks: &[MkTaskDescriptor]) {
        self.ring.set(tasks);
        // SAFETY: each task points at device memory of this test, sized for its op.
        unsafe { self.ring.run(&self.cluster.cards[0]) };
    }
}

impl Drop for Card {
    fn drop(&mut self) {
        let card = &self.cluster.cards[0];
        self.device.free_all(card);
        self.ring.free(card);
    }
}

/// `len` values in `[-spread, spread)` from the seed.
fn values(seed: u32, len: usize, spread: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32 * 2.0 - 1.0) * spread
        })
        .collect()
}

fn bits_differ(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len(), "the lengths differ");
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

#[test]
fn the_rotation_gives_the_bits_of_the_host_form() {
    let mut c = Card::open();
    for g in [
        Grid { rows: 1, columns: 1 },
        Grid { rows: 2, columns: 3 },
        Grid { rows: 5, columns: 2 },
    ] {
        let n = g.patches() + 1;
        let (q0, k0) = (values(1, n * WIDTH, 3.0), values(2, n * WIDTH, 3.0));
        let (q, k) = (c.put(&q0), c.put(&k0));
        let table = c.put(&tower::rope_table());
        c.run(&[tasks::inkling2_vision_rope(
            q,
            k,
            table,
            n as u32,
            WIDTH as u32,
            g.columns as u32,
        )]);
        for (name, at, start) in [("queries", q, &q0), ("keys", k, &k0)] {
            let mut want = start.clone();
            for (t, row) in want.as_chunks_mut::<WIDTH>().0.iter_mut().enumerate() {
                let (line, column) = if t == 0 {
                    (0, 0)
                } else {
                    let (l, c) = vision::patch_position(t - 1, g);
                    (l + 1, c + 1)
                };
                for head in row.as_chunks_mut::<{ vision::HEAD_DIM }>().0.iter_mut() {
                    for (j, pair) in head.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                        let a = vision::rope_angle(j, line, column);
                        let (cs, sn) = (a.cos(), a.sin());
                        let (x1, x2) = (pair[0], pair[1]);
                        pair[0] = x1 * cs + -x2 * sn;
                        pair[1] = x2 * cs + x1 * sn;
                    }
                }
            }
            let differ = bits_differ(&c.read(at, n * WIDTH), &want);
            eprintln!("rope {g:?}, {name}: {differ} of {} values differ in bits", n * WIDTH);
            // The card reads the host's table and computes `x1 * c + -x2 * s` with no fused multiply-add, as the host
            // does. Thus the gate is equal bits.
            assert_eq!(differ, 0, "rope {g:?}: the {name} differ from the host form");
        }
    }
}

#[test]
fn the_class_row_and_the_position_embedding_match_the_host_form() {
    let mut c = Card::open();
    let table_rows = vision::GRID * vision::GRID;
    // The table as bf16 bits, and its f32 values for the host.
    let table_f32: Vec<f32> = values(3, table_rows * WIDTH, 1.0)
        .iter()
        .map(|v| f32::from_bits(v.to_bits() & 0xFFFF_0000))
        .collect();
    let table_bits: Vec<u8> = table_f32
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let table = c.zeros(table_bits.len());
    c.device.write(c.card(), table, &table_bits);
    let class = values(4, WIDTH, 1.0);
    let class_at = c.put(&class);
    for g in [
        Grid { rows: 1, columns: 1 },
        Grid { rows: 3, columns: 6 },
        Grid { rows: 12, columns: 16 },
    ] {
        let p = g.patches();
        let patches = values(5, p * WIDTH, 2.0);
        let (a, y) = (c.put(&patches), c.zeros((p + 1) * WIDTH * 4));
        c.run(&[tasks::inkling2_vision_embed(
            a,
            table,
            class_at,
            y,
            g.rows as u32,
            g.columns as u32,
            WIDTH as u32,
        )]);
        let got = c.read(y, (p + 1) * WIDTH);
        assert_eq!(bits_differ(&got[..WIDTH], &class), 0, "{g:?}: the class row is row 0");
        let mut want = Vec::with_capacity(p * WIDTH);
        for n in 0..p {
            let (line, column) = vision::patch_position(n, g);
            let mut pos = vec![0f32; WIDTH];
            for (ht, hw) in vision::axis_taps(line, 2 * g.rows) {
                for (wt, ww) in vision::axis_taps(column, 2 * g.columns) {
                    let weight = hw * ww;
                    let row = &table_f32[(ht * vision::GRID + wt) * WIDTH..][..WIDTH];
                    for (o, v) in pos.iter_mut().zip(row) {
                        *o += v * weight;
                    }
                }
            }
            want.extend(patches[n * WIDTH..][..WIDTH].iter().zip(&pos).map(|(a, b)| a + b));
        }
        let gap = common::worst_row(&got[WIDTH..], &want, WIDTH);
        let differ = bits_differ(&got[WIDTH..], &want);
        eprintln!(
            "embed {g:?}: worst row {gap:.3e}, {differ} of {} values differ in bits",
            p * WIDTH
        );
        assert!(gap < TOL_EMBED, "embed {g:?}: the rows differ by {gap:.3e}");
    }
}

/// The model, its weights on the card, and the buffers of the tower.
struct Tower {
    card: Card,
    weights: tower::Weights,
    bufs: Bufs,
    host: Vision,
    oracle: HashMap<String, common::Tensor>,
}

fn absent() -> bool {
    common::model_absent(Which::Mini) || common::golden_absent(ORACLE)
}

/// The two images of the oracle need 192 image tokens at most.
const TEST_TOKENS: usize = 192;

impl Tower {
    fn open() -> Self {
        let mut card = Card::open();
        let path = common::model_weights(Which::Mini);
        let map = MappedSafeTensors::open(&path).unwrap();
        let view = map.view().unwrap();
        let weights = {
            let (device, c) = (&mut card.device, &card.cluster.cards[0]);
            tower::load(&mut OnCard { device, card: c }, &view).unwrap_or_else(|e| panic!("the load: {e}"))
        };
        let bufs = Bufs::new(&mut |bytes| card.zeros(bytes), TEST_TOKENS);
        let host = Vision::open(&Checkpoint::open(&path).unwrap()).unwrap();
        Self {
            card,
            weights,
            bufs,
            host,
            oracle: common::golden_tensors(ORACLE),
        }
    }

    /// The grid and the patch rows of image `i` of the oracle, from the rules of the processor.
    fn prepared(&self, i: usize) -> (Grid, Vec<f32>) {
        let t = &self.oracle[&format!("image{i}.rgb")];
        let (h, w) = (t.shape[0], t.shape[1]);
        let rgb: Vec<u8> = t.values.iter().map(|v| *v as u8).collect();
        let (g, px) = vision::pixel_values(&rgb, w, h).unwrap();
        (g, vision::patch_rows(&px, g))
    }

    /// Run the first `count` tasks of the tower of `g` with its rows at `rows`, and read the rows of `mark`.
    fn stage(&mut self, g: Grid, rows: u64, out: u64, mark: &Mark) -> Vec<f32> {
        let run = tower::tower(&self.weights, &self.bufs, g, rows, out);
        self.card.run(&run.tasks[..mark.tasks]);
        self.card.read(mark.at, mark.rows * mark.width)
    }
}

#[test]
fn each_stage_of_the_tower_of_image_0_matches_the_oracle_and_the_host_form() {
    if absent() {
        return;
    }
    let mut t = Tower::open();
    let (g, rows) = t.prepared(0);
    let mut trace = VisionTrace::default();
    let host_out = t.host.image_rows(&rows, g, Some(&mut trace));
    let rows_at = t.card.put(&rows);
    let out = t.card.zeros(tower::out_bytes(g.tokens()));
    let marks = tower::tower(&t.weights, &t.bufs, g, rows_at, out).marks;
    let mut failures = Vec::new();
    for mark in &marks {
        let got = t.stage(g, rows_at, out, mark);
        let (name, host, oracle, tol) = match mark.stage {
            Stage::PatchEmbed => (
                "patch_embed".to_owned(),
                &trace.patch_embed,
                "image0.patch_embed",
                TOL_FRONT,
            ),
            Stage::Embeddings => (
                "embeddings".to_owned(),
                &trace.embeddings,
                "image0.embeddings",
                TOL_FRONT,
            ),
            Stage::PreLayernorm => (
                "pre_layernorm".to_owned(),
                &trace.pre_layernorm,
                "image0.pre_layernorm",
                TOL_FRONT,
            ),
            Stage::Layer(l) => (format!("layer{l:02}"), &trace.layers[l], "", TOL_LAYERS),
            Stage::Projector => ("projector".to_owned(), &host_out, "image0.projector", TOL_LAYERS),
        };
        let oracle_name = if oracle.is_empty() {
            format!("image0.{name}")
        } else {
            oracle.to_owned()
        };
        let to_host = common::worst_row(&got, host, mark.width);
        let to_oracle = common::worst_row(&got, &t.oracle[&oracle_name].f32(), mark.width);
        eprintln!("image 0, {name}: worst row {to_host:.3e} to the host form, {to_oracle:.3e} to the oracle");
        if to_host >= tol || to_oracle >= TOL_ORACLE {
            failures.push(format!(
                "{name}: {to_host:.3e} to the host form, {to_oracle:.3e} to the oracle"
            ));
        }
    }
    assert!(failures.is_empty(), "the stages that differ: {failures:?}");
}

#[test]
fn the_projector_rows_of_image_1_match_the_oracle_and_the_host_form() {
    if absent() {
        return;
    }
    let mut t = Tower::open();
    let (g, rows) = t.prepared(1);
    let host = t.host.image_rows(&rows, g, None);
    let rows_at = t.card.put(&rows);
    let out = t.card.zeros(tower::out_bytes(g.tokens()));
    let run = tower::tower(&t.weights, &t.bufs, g, rows_at, out);
    let started = std::time::Instant::now();
    t.card.run(&run.tasks);
    eprintln!(
        "image 1: {} patches in {:.1} ms on the card",
        g.patches(),
        started.elapsed().as_secs_f64() * 1e3
    );
    let got = t.card.read(out, g.tokens() * HIDDEN);
    let to_host = common::worst_row(&got, &host, HIDDEN);
    let to_oracle = common::worst_row(&got, &t.oracle["image1.projector"].f32(), HIDDEN);
    eprintln!("image 1, projector: worst row {to_host:.3e} to the host form, {to_oracle:.3e} to the oracle");
    assert!(
        to_host < TOL_LAYERS,
        "image 1: the projector differs from the host form by {to_host:.3e}"
    );
    assert!(
        to_oracle < TOL_ORACLE,
        "image 1: the projector differs from the oracle by {to_oracle:.3e}"
    );
}

#[test]
fn an_image_alone_and_two_images_in_one_ring_give_the_same_bits() {
    if absent() {
        return;
    }
    let mut t = Tower::open();
    let images = [t.prepared(0), t.prepared(1)];
    let rows: Vec<u64> = images.iter().map(|(_, r)| t.card.put(r)).collect();
    let outs = |t: &mut Tower| -> Vec<u64> {
        images
            .iter()
            .map(|(g, _)| t.card.zeros(tower::out_bytes(g.tokens())))
            .collect()
    };
    let (alone, joined) = (outs(&mut t), outs(&mut t));
    for (i, (g, _)) in images.iter().enumerate() {
        let run = tower::tower(&t.weights, &t.bufs, *g, rows[i], alone[i]);
        t.card.run(&run.tasks);
    }
    let mut both = Vec::new();
    for (i, (g, _)) in images.iter().enumerate() {
        both.extend(tower::tower(&t.weights, &t.bufs, *g, rows[i], joined[i]).tasks);
    }
    t.card.run(&both);
    for (i, (g, _)) in images.iter().enumerate() {
        let n = g.tokens() * HIDDEN;
        let differ = bits_differ(&t.card.read(alone[i], n), &t.card.read(joined[i], n));
        eprintln!("image {i}: {differ} of {n} projector values differ between alone and joined");
        assert_eq!(differ, 0, "image {i}: the bits depend on the other image of the ring");
    }
    // A second run of the same ring gives the same bits too.
    let first = t.card.read(joined[1], images[1].0.tokens() * HIDDEN);
    t.card.run(&both);
    assert_eq!(
        bits_differ(&first, &t.card.read(joined[1], first.len())),
        0,
        "a second run differs"
    );
}
