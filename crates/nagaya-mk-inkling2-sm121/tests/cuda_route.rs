//! GPU gate of `OP_ROUTE` (milestone 37 step 9, wave B lane M): the router matmul, the sigmoid, the top 8 with the tie
//! rule, and the list of pairs. The weights are the real router of Inkling 2 Mini, layers 0 and 20, and the oracle is
//! `ops::route` of the B70 crate.
//!
//! The gates:
//! - The logits are within the bound of the MMA of the host dot products.
//! - The picks equal the host picks, except where two keys are closer than the bound of the router. Each such case is
//!   printed. The weights are within a relative bound of the host weights.
//! - The list of pairs is the list that the plan rules give for the picks of the device, exactly.
//! - The bits of a row do not depend on the count of rows (1, 16, 512) or on the grid (48, 24, 7, 1), and a second run
//!   with no new zeroing gives the same bits: each level reset its counter.
//! - A tie of keys takes the lower id.
//!
//! `NAGAYA_TEST_CARDS=<pci address> INKLING2_MODEL_DIR=<folder of Mini> cargo test -p nagaya-mk-inkling2-sm121
//!  --features cuda --test cuda_route -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;
mod moe_support;

use common::gpu::{GRIDS, download, open, open_profile, run_with_watchdog, upload};
use moe_support::{Checkpoint, Rng, activation_row, download_pairs, model_dir, widen_bf16};
use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::experts::{
    ROUTE_COUNTER_WORDS, ROUTE_ITEMS_PER_ROW_TILE, ROUTE_TABLE_BIAS, ROUTE_TABLE_NORM_WEIGHT, ROUTE_TABLE_ROW_FACTOR,
    route_items, route_pair_list_bytes, route_picks_bytes, route_scratch_words,
};
use nagaya_mk_cuda::protocol::{
    EXPERT_MAX_ROWS, ExpertPair, FLAG_NORM, FLAG_PAIRS, MkTaskDescriptor, MkTaskTable, OP_PROBE_READ, OP_ROUTE,
    PROBE_ITEM_BYTES, PROBE_READ_NC, PROFILE_WORDS, TOP_K, expert_pair_bound,
};
use nagaya_mk_cuda::repack::{MatmulShape, repack_bf16};
use nagaya_mk_cuda::ring::{DEFAULT_BLOCK_THREADS, Ring, Task};
use nagaya_mk_inkling2_bmg::host::ops;

const EXPERTS: usize = 128;
const HIDDEN: usize = 2304;
const ROUTE_SCALE: f32 = 8.0;

/// The bound of the logits: the error over the sum of the absolute terms of the dot product. The MMA of the card does
/// not round as a sequence of `f32` adds, and the host sums in 16 lanes. The gate of the matmul lane measured 1.24e-7
/// for one MMA, and a router sums 144 of them.
const LOGIT_BOUND: f64 = 4.0e-7;

/// Two keys closer than this are a tie that the order of a sum can flip. The bound is a few times the error of the
/// logit through the sigmoid, and of the `expf` of the device against the `expf` of the host.
const KEY_TIE: f64 = 2.0e-6;

struct Router {
    name: String,
    /// The `bf16` bytes as the checkpoint has them, and the widened values.
    bytes: Vec<u8>,
    wide: Vec<f32>,
    bias: Vec<f32>,
    norm_weight: Vec<f32>,
}

fn routers() -> Option<Vec<Router>> {
    let dir = model_dir("cuda_route")?;
    let mut checkpoint = Checkpoint::open(&dir);
    Some(
        [0usize, 20]
            .iter()
            .map(|&layer| {
                let bytes = checkpoint.all(&format!("model.llm.layers.{layer}.mlp.gate.weight"));
                Router {
                    name: format!("layer {layer}"),
                    wide: widen_bf16(&bytes),
                    bytes,
                    bias: checkpoint.f32s(&format!("model.llm.layers.{layer}.mlp.gate.bias"), 0, EXPERTS),
                    norm_weight: checkpoint.bf16_as_f32(&format!("model.llm.layers.{layer}.mlp_norm.weight")),
                }
            })
            .collect(),
    )
}

/// A router for the tie gate: every value of row `e` is the same power of two, so that the sums are exact.
fn synthetic(class: impl Fn(usize) -> usize) -> Router {
    let wide: Vec<f32> = (0..EXPERTS)
        .flat_map(|e| std::iter::repeat_n(f32::from_bits(((126 - 9) as u32) << 23) * (1 + class(e)) as f32, HIDDEN))
        .collect();
    let bytes = wide
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    Router {
        name: "synthetic".into(),
        bytes,
        wide,
        bias: vec![0.0; EXPERTS],
        norm_weight: vec![1.0; HIDDEN],
    }
}

/// The rows of a gate: `count` rows, with the seed `seed`, as a normalized hidden row looks.
fn rows_of(count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(0x7007_0000 + seed);
    (0..count)
        .map(|i| activation_row(&mut rng, HIDDEN, [1.0f32, 0.3, 2.0][i % 3]))
        .collect()
}

struct Picks {
    logits: Vec<f32>,
    experts: Vec<u32>,
    weights: Vec<f32>,
    pairs: Vec<ExpertPair>,
    scratch_head: Vec<u32>,
}

/// The router on the device: the weights repacked and uploaded once.
struct Device<'d> {
    dev: &'d CudaDevice,
    weight: CUdeviceptr,
    bias: CUdeviceptr,
    norm_weight: CUdeviceptr,
}

impl<'d> Device<'d> {
    fn new(dev: &'d CudaDevice, router: &Router) -> Self {
        let shape = MatmulShape::new(EXPERTS, HIDDEN).unwrap();
        Self {
            dev,
            weight: upload(dev, &repack_bf16(&router.bytes, &shape).unwrap()),
            bias: upload(dev, &router.bias),
            norm_weight: upload(dev, &router.norm_weight),
        }
    }

    fn free(self) {
        for ptr in [self.weight, self.bias, self.norm_weight] {
            self.dev.free_device(ptr).unwrap();
        }
    }

    /// The descriptor of the op for `rows` rows. `norm` holds the row factors when the op applies the norm.
    #[allow(clippy::too_many_arguments)]
    fn task(
        &self,
        rows: usize,
        x: CUdeviceptr,
        picks: CUdeviceptr,
        pairs: CUdeviceptr,
        scratch: CUdeviceptr,
        table: CUdeviceptr,
        norm: bool,
    ) -> MkTaskDescriptor {
        MkTaskDescriptor {
            opcode: OP_ROUTE,
            flags: FLAG_PAIRS | if norm { FLAG_NORM } else { 0 },
            items: route_items(rows) as u32,
            rows: rows as u32,
            a: x,
            b: self.weight,
            c: picks,
            d: table,
            e: pairs,
            f: scratch,
            n: EXPERTS as u32,
            k: HIDDEN as u32,
            kt: 768,
            group_items: 3,
            // The producer prefetches the item of its block before the wait, for up to 16 rows.
            item_bytes: if rows <= 16 { 24_576 } else { 0 },
            p0: ROUTE_SCALE,
            ..Default::default()
        }
    }

    /// Run the op over `x` at `grid`, `repeat` times with no zeroing between the runs, and return the last result.
    /// With `factors` the op reads `x` as the rows before the norm.
    fn run(&self, x: &[Vec<f32>], factors: Option<&[f32]>, grid: u32, repeat: usize) -> Picks {
        let rows = x.len();
        let flat: Vec<f32> = x.iter().flatten().copied().collect();
        let (x_ptr, scratch) = (
            upload(self.dev, &flat),
            upload(self.dev, &vec![0u32; route_scratch_words(rows)]),
        );
        let picks = upload(self.dev, &vec![0xFFu8; route_picks_bytes(rows)]);
        let pairs = upload(self.dev, &vec![0u8; route_pair_list_bytes(rows)]);
        let rf = upload(self.dev, factors.unwrap_or(&[0.0]));
        let mut table = MkTaskTable::default();
        table.ptr[ROUTE_TABLE_BIAS] = self.bias;
        table.ptr[ROUTE_TABLE_NORM_WEIGHT] = self.norm_weight;
        table.ptr[ROUTE_TABLE_ROW_FACTOR] = rf;
        let table_ptr = upload(self.dev, &[table]);
        let mut ring = Ring::new(self.dev, 1, false).unwrap();
        ring.set(&[
            Task::with_items(self.task(rows, x_ptr, picks, pairs, scratch, table_ptr, factors.is_some()))
                .reads(x_ptr, flat.len() * 4)
                .writes(picks, route_picks_bytes(rows))
                .writes(pairs, route_pair_list_bytes(rows))
                .writes(scratch, route_scratch_words(rows) * 4),
        ]);
        let mut out = None;
        for _ in 0..repeat {
            ring.zero_device().unwrap();
            run_with_watchdog(
                &ring,
                grid,
                DEFAULT_BLOCK_THREADS,
                self.dev.max_dynamic_shared_bytes().unwrap(),
            );
            let words = route_scratch_words(rows);
            let all: Vec<u32> = download(self.dev, scratch, words);
            let counter_words = ROUTE_COUNTER_WORDS;
            let raw: Vec<u32> = download(self.dev, picks, rows * TOP_K * 2);
            out = Some(Picks {
                logits: all[counter_words..counter_words + rows * EXPERTS]
                    .iter()
                    .map(|b| f32::from_bits(*b))
                    .collect(),
                experts: raw[..rows * TOP_K].to_vec(),
                weights: raw[rows * TOP_K..].iter().map(|b| f32::from_bits(*b)).collect(),
                pairs: download_pairs(self.dev, pairs, expert_pair_bound(rows)),
                scratch_head: all[..counter_words].to_vec(),
            });
        }
        ring.free().unwrap();
        for ptr in [x_ptr, scratch, picks, pairs, rf, table_ptr] {
            self.dev.free_device(ptr).unwrap();
        }
        out.unwrap()
    }
}

impl Device<'_> {
    /// Run the op over `before`, then over `x`, on one scratch that nothing zeroes between the runs, and return the
    /// result of `x`. A call of other rows must leave the counters at 0.
    fn run_after(&self, before: &[Vec<f32>], x: &[Vec<f32>], grid: u32) -> Picks {
        let most = before.len().max(x.len());
        let scratch = upload(self.dev, &vec![0u32; route_scratch_words(most)]);
        let rf = upload(self.dev, &[0.0f32]);
        let mut table = MkTaskTable::default();
        table.ptr[ROUTE_TABLE_BIAS] = self.bias;
        table.ptr[ROUTE_TABLE_NORM_WEIGHT] = self.norm_weight;
        table.ptr[ROUTE_TABLE_ROW_FACTOR] = rf;
        let table_ptr = upload(self.dev, &[table]);
        let mut ring = Ring::new(self.dev, 1, false).unwrap();
        let mut out = None;
        for rows_in in [before, x] {
            let rows = rows_in.len();
            let flat: Vec<f32> = rows_in.iter().flatten().copied().collect();
            let x_ptr = upload(self.dev, &flat);
            let picks = upload(self.dev, &vec![0xFFu8; route_picks_bytes(rows)]);
            let pairs = upload(self.dev, &vec![0u8; route_pair_list_bytes(rows)]);
            ring.set(&[
                Task::with_items(self.task(rows, x_ptr, picks, pairs, scratch, table_ptr, false))
                    .reads(x_ptr, flat.len() * 4)
                    .writes(picks, route_picks_bytes(rows))
                    .writes(pairs, route_pair_list_bytes(rows))
                    .writes(scratch, route_scratch_words(rows) * 4),
            ]);
            ring.zero_device().unwrap();
            run_with_watchdog(
                &ring,
                grid,
                DEFAULT_BLOCK_THREADS,
                self.dev.max_dynamic_shared_bytes().unwrap(),
            );
            let all: Vec<u32> = download(self.dev, scratch, route_scratch_words(most));
            let counter_words = ROUTE_COUNTER_WORDS;
            let raw: Vec<u32> = download(self.dev, picks, rows * TOP_K * 2);
            out = Some(Picks {
                logits: all[counter_words..counter_words + rows * EXPERTS]
                    .iter()
                    .map(|b| f32::from_bits(*b))
                    .collect(),
                experts: raw[..rows * TOP_K].to_vec(),
                weights: raw[rows * TOP_K..].iter().map(|b| f32::from_bits(*b)).collect(),
                pairs: download_pairs(self.dev, pairs, expert_pair_bound(rows)),
                scratch_head: all[..counter_words].to_vec(),
            });
            for ptr in [x_ptr, picks, pairs] {
                self.dev.free_device(ptr).unwrap();
            }
        }
        ring.free().unwrap();
        for ptr in [scratch, rf, table_ptr] {
            self.dev.free_device(ptr).unwrap();
        }
        out.unwrap()
    }
}

/// The list of pairs that the rules give for the picks `experts` (8 for each row, ascending) and their weights.
fn expected_pairs(experts: &[u32], weights: &[f32]) -> Vec<ExpertPair> {
    let rows = experts.len() / TOP_K;
    let mut out = Vec::new();
    for e in 0..EXPERTS as u32 {
        let mut hits: Vec<(usize, usize)> = Vec::new();
        for r in 0..rows {
            if let Some(j) = experts[r * TOP_K..(r + 1) * TOP_K].iter().position(|x| *x == e) {
                hits.push((r, j));
            }
        }
        for chunk in hits.chunks(EXPERT_MAX_ROWS) {
            let mut pair = ExpertPair {
                expert: e,
                count: chunk.len() as u32,
                ..Default::default()
            };
            for (i, (r, j)) in chunk.iter().enumerate() {
                pair.x_row[i] = *r as u16;
                pair.y_row[i] = (r * TOP_K + j) as u16;
                pair.weight[i] = weights[r * TOP_K + j];
            }
            out.push(pair);
        }
    }
    out
}

/// Compare the live entries of two pairs: the entries past the count are not defined.
fn same_pair(a: &ExpertPair, b: &ExpertPair) -> bool {
    a.expert == b.expert
        && a.count == b.count
        && (0..a.count as usize).all(|i| {
            a.x_row[i] == b.x_row[i] && a.y_row[i] == b.y_row[i] && a.weight[i].to_bits() == b.weight[i].to_bits()
        })
}

fn host_logits(router: &Router, row: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; EXPERTS];
    ops::matvec(&router.wide, row, &mut out);
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Check one run against the host. Return the count of rows with a pick that differs at a tie of keys.
fn check_against_host(router: &Router, x: &[Vec<f32>], got: &Picks, label: &str) -> usize {
    let mut ties = 0;
    let (mut worst_logit, mut worst_weight) = (0.0f64, 0.0f64);
    for (r, row) in x.iter().enumerate() {
        let host = host_logits(router, row);
        for e in 0..EXPERTS {
            let terms: f64 = (0..HIDDEN)
                .map(|k| (f64::from(router.wide[e * HIDDEN + k]) * f64::from(row[k])).abs())
                .sum();
            let error = (f64::from(got.logits[r * EXPERTS + e]) - f64::from(host[e])).abs();
            let ratio = error / terms.max(1e-30);
            worst_logit = worst_logit.max(ratio);
            assert!(
                ratio <= LOGIT_BOUND,
                "{label}: row {r} expert {e}: gpu {:e}, host {:e}: the error is {ratio:e} of the terms; the bound is {LOGIT_BOUND:e}",
                got.logits[r * EXPERTS + e],
                host[e]
            );
        }
        // The host picks come from the host logits. The device picks must be the same set, except at a tie of keys.
        let want = ops::route(&host, &router.bias);
        let want_ids: Vec<u32> = want.iter().map(|(e, _)| *e as u32).collect();
        let got_ids = &got.experts[r * TOP_K..(r + 1) * TOP_K];
        assert!(
            got_ids.windows(2).all(|w| w[0] < w[1]),
            "{label}: row {r}: the picks {got_ids:?} are not ascending"
        );
        if want_ids == got_ids {
            for j in 0..TOP_K {
                let (w, g) = (f64::from(want[j].1), f64::from(got.weights[r * TOP_K + j]));
                let ratio = (w - g).abs() / w.abs();
                worst_weight = worst_weight.max(ratio);
                assert!(
                    ratio <= 4.0e-6,
                    "{label}: row {r} pick {j} (expert {}): weight gpu {g:e}, host {w:e}",
                    want[j].0
                );
            }
        } else {
            // The keys of the host, in `f64`: the differing experts must be at the eighth place within the bound.
            let key = |e: usize| 1.0 / (1.0 + (-f64::from(host[e])).exp()) + f64::from(router.bias[e]);
            let mut order: Vec<usize> = (0..EXPERTS).collect();
            order.sort_by(|a, b| key(*b).total_cmp(&key(*a)).then(a.cmp(b)));
            let eighth = key(order[7]);
            for e in want_ids.iter().chain(got_ids).map(|e| *e as usize) {
                let in_want = want_ids.contains(&(e as u32));
                let in_got = got_ids.contains(&(e as u32));
                if in_want != in_got {
                    let gap = (key(e) - eighth).abs();
                    assert!(
                        gap < KEY_TIE,
                        "{label}: row {r}: expert {e} is picked by only one side, and its key is {gap:e} from the eighth key; that is not a tie (bound {KEY_TIE:e}). host {want_ids:?}, gpu {got_ids:?}"
                    );
                }
            }
            eprintln!(
                "cuda_route: {label}: row {r}: a pick differs at a tie of keys (gap under {KEY_TIE:e}): host {want_ids:?}, gpu {got_ids:?}"
            );
            ties += 1;
        }
    }
    eprintln!(
        "cuda_route: {label}: {} rows: the worst logit error is {worst_logit:e} of the terms, the worst weight error is {worst_weight:e}, {ties} rows differ at a tie",
        x.len()
    );
    ties
}

fn check_pairs(got: &Picks, label: &str) {
    let want = expected_pairs(&got.experts, &got.weights);
    assert_eq!(
        got.pairs.len(),
        want.len(),
        "{label}: the live count of pairs is {}, and the picks give {}",
        got.pairs.len(),
        want.len()
    );
    for (i, (g, w)) in got.pairs.iter().zip(&want).enumerate() {
        assert!(same_pair(g, w), "{label}: pair {i} is {g:?}, and the rules give {w:?}");
    }
    assert!(
        got.scratch_head.iter().all(|c| *c == 0),
        "{label}: a counter was not reset"
    );
}

#[test]
fn the_picks_the_weights_and_the_pairs_follow_the_host_router() {
    let Some(routers) = routers() else { return };
    let dev = open();
    for router in &routers {
        let device = Device::new(&dev, router);
        for rows in [1usize, 16, 512] {
            let x = rows_of(rows, rows as u64);
            let got = device.run(&x, None, 48, 1);
            let label = format!("{} at {rows} rows", router.name);
            check_against_host(router, &x, &got, &label);
            check_pairs(&got, &label);
            assert!(got.pairs.len() <= expert_pair_bound(rows));
            if rows == 1 {
                assert_eq!(got.pairs.len(), TOP_K, "{label}: a decode row gives 8 pairs of 1 row");
            }
        }
        device.free();
    }
}

#[test]
fn the_bits_do_not_depend_on_the_count_of_rows_or_the_grid_and_a_rerun_gives_the_same_bits() {
    let Some(routers) = routers() else { return };
    let dev = open();
    for router in &routers {
        let device = Device::new(&dev, router);
        let big = rows_of(512, 99);
        let reference = device.run(&big, None, 48, 1);
        // The grid, and a second run with no zeroing in between.
        for grid in GRIDS {
            let got = device.run(&big, None, grid, 2);
            assert_eq!(
                bits(&got.logits),
                bits(&reference.logits),
                "{}: logits at grid {grid}",
                router.name
            );
            assert_eq!(got.experts, reference.experts, "{}: picks at grid {grid}", router.name);
            assert_eq!(
                bits(&got.weights),
                bits(&reference.weights),
                "{}: weights at grid {grid}",
                router.name
            );
            assert_eq!(got.pairs.len(), reference.pairs.len());
            for (a, b) in got.pairs.iter().zip(&reference.pairs) {
                assert!(same_pair(a, b), "{}: a pair at grid {grid}", router.name);
            }
        }
        // The count of rows: the first row alone, the first 16 rows, and rows 7 and 300 alone.
        for (label, take) in [("1 row", 0..1usize), ("16 rows", 0..16), ("7", 7..8), ("300", 300..301)] {
            for grid in [48u32, 7, 1] {
                let sub: Vec<Vec<f32>> = big[take.clone()].to_vec();
                let got = device.run(&sub, None, grid, 1);
                for (i, r) in take.clone().enumerate() {
                    assert_eq!(
                        bits(&got.logits[i * EXPERTS..(i + 1) * EXPERTS]),
                        bits(&reference.logits[r * EXPERTS..(r + 1) * EXPERTS]),
                        "{}: logits of row {r} alone ({label}) at grid {grid}",
                        router.name
                    );
                    assert_eq!(
                        got.experts[i * TOP_K..(i + 1) * TOP_K],
                        reference.experts[r * TOP_K..(r + 1) * TOP_K]
                    );
                    assert_eq!(
                        bits(&got.weights[i * TOP_K..(i + 1) * TOP_K]),
                        bits(&reference.weights[r * TOP_K..(r + 1) * TOP_K]),
                        "{}: weights of row {r} ({label}) at grid {grid}",
                        router.name
                    );
                }
                check_pairs(&got, &format!("{} {label} at grid {grid}", router.name));
            }
        }
        eprintln!(
            "cuda_route: {}: the bits equal at 1, 16, 512 rows and at the grids {GRIDS:?}",
            router.name
        );
        device.free();
    }
}

/// The hashes of the logits and the picks of the standard case (512 rows of `rows_of(512, 99)`, grid 48), for the routers of
/// layer 0 and layer 20. The hash pins the cut of K, the tree of the warps, the order of the slices, and the picks: they are
/// part of the engine identity, and a change of one of them changes the hash. The weights are not in the hash, because the
/// `expf` of a CUDA version can change their last bit.
const PINNED_HASHES: [&str; 2] = [
    "e7f1e660433d538dd1ca5179540872b40ccaa92eb0c4433e951a5b6680b90712",
    "5b7c97c65910cde4ce479f55df7aff7253bd198148a12c03df58dd9a9a14f8ce",
];

/// A call of few rows leaves logits where a call of more rows has its counters. The counters sit at a place that does
/// not depend on the rows, so the second call gives the bits of a call on a fresh scratch.
#[test]
fn a_call_after_a_call_of_other_rows_on_the_same_scratch_gives_the_bits_of_a_fresh_scratch() {
    let Some(routers) = routers() else { return };
    let dev = open();
    for router in &routers {
        let device = Device::new(&dev, router);
        for (first, second) in [(1usize, 17usize), (9, 17), (17, 1), (16, 512)] {
            let a = rows_of(first, 5);
            let b = rows_of(second, 6);
            let fresh = device.run(&b, None, 48, 1);
            let after = device.run_after(&a, &b, 48);
            assert_eq!(
                bits(&after.logits),
                bits(&fresh.logits),
                "{}: {second} rows after {first} rows: the logits differ from a fresh scratch",
                router.name
            );
            assert_eq!(
                after.experts, fresh.experts,
                "{}: {second} rows after {first}: picks",
                router.name
            );
            assert_eq!(
                bits(&after.weights),
                bits(&fresh.weights),
                "{}: {second} rows after {first}: weights",
                router.name
            );
        }
        device.free();
    }
}

#[test]
fn the_bits_of_the_logits_and_the_picks_are_pinned() {
    let Some(routers) = routers() else { return };
    let dev = open();
    let print_only = std::env::var("CUDA_MOE_PRINT_HASHES").is_ok();
    for (index, router) in routers.iter().enumerate() {
        let device = Device::new(&dev, router);
        let got = device.run(&rows_of(512, 99), None, 48, 1);
        let mut hasher = blake3::Hasher::new();
        for v in &got.logits {
            hasher.update(&v.to_le_bytes());
        }
        for e in &got.experts {
            hasher.update(&e.to_le_bytes());
        }
        let hash = hasher.finalize().to_hex().to_string();
        if print_only {
            eprintln!("cuda_route pinned: {index} {} {hash}", router.name);
        } else {
            assert_eq!(
                hash, PINNED_HASHES[index],
                "{}: the bits of the standard case changed",
                router.name
            );
        }
        device.free();
    }
}

#[test]
fn the_norm_prologue_gives_the_bits_of_a_normed_row() {
    let Some(routers) = routers() else { return };
    let dev = open();
    for router in &routers {
        let device = Device::new(&dev, router);
        let raw = rows_of(40, 5);
        let factors: Vec<f32> = raw
            .iter()
            .map(|row| 1.0 / (ops::dot(row, row) / HIDDEN as f32 + 1e-6).sqrt())
            .collect();
        let normed: Vec<Vec<f32>> = raw
            .iter()
            .map(|row| {
                let mut out = vec![0f32; HIDDEN];
                ops::rms_norm(row, &router.norm_weight, &mut out);
                out
            })
            .collect();
        for (n, f) in normed.iter().zip(raw.iter().zip(&factors)) {
            for k in 0..HIDDEN {
                assert_eq!(
                    n[k].to_bits(),
                    (router.norm_weight[k] * (f.0[k] * f.1)).to_bits(),
                    "the host norm of the gate"
                );
            }
        }
        let with_norm = device.run(&raw, Some(&factors), 48, 1);
        let plain = device.run(&normed, None, 48, 1);
        assert_eq!(
            bits(&with_norm.logits),
            bits(&plain.logits),
            "{}: FLAG_NORM",
            router.name
        );
        assert_eq!(with_norm.experts, plain.experts);
        device.free();
    }
}

#[test]
fn the_plan_holds_the_most_rows_and_rows_that_all_pick_the_same_experts() {
    let Some(routers) = routers() else { return };
    let dev = open();
    let router = &routers[0];
    let device = Device::new(&dev, router);
    // 1,024 rows (the most), and 1,024 equal rows: the same 8 experts for each row, 64 pairs for each expert.
    let varied = rows_of(1024, 11);
    let got = device.run(&varied, None, 48, 1);
    check_pairs(&got, "1,024 rows");
    let same: Vec<Vec<f32>> = (0..1024).map(|_| varied[0].clone()).collect();
    let got = device.run(&same, None, 7, 1);
    check_pairs(&got, "1,024 equal rows");
    assert_eq!(
        got.pairs.len(),
        8 * 64,
        "8 experts with 1,024 rows each, 16 rows for a pair"
    );
    assert!(got.pairs.len() <= expert_pair_bound(1024));
    assert!(got.pairs.iter().all(|p| p.count as usize == EXPERT_MAX_ROWS));
    device.free();
}

#[test]
fn a_tie_of_keys_takes_the_lower_id() {
    let Some(_) = model_dir("cuda_route") else { return };
    let dev = open();
    let ones = vec![vec![1.0f32; HIDDEN]];
    // A: 25 experts (ids 4, 9, ..., 124) share the top key. The 8 lowest ids win.
    let router = synthetic(|e| if e % 5 == 4 { 4 } else { e % 4 });
    let device = Device::new(&dev, &router);
    let got = device.run(&ones, None, 48, 1);
    assert_eq!(
        got.experts,
        (0..8).map(|i| 4 + 5 * i).collect::<Vec<u32>>(),
        "A: the top class has 25 equal keys"
    );
    let want = ops::route(&host_logits(&router, &ones[0]), &router.bias);
    assert_eq!(got.experts, want.iter().map(|(e, _)| *e as u32).collect::<Vec<_>>());
    assert!(
        got.weights.windows(2).all(|w| w[0].to_bits() == w[1].to_bits()),
        "A: equal keys give equal weights"
    );
    device.free();
    // B: 5 experts of class 4 (ids 10, 20, 30, 40, 50), then 60 of class 3: the eighth place is a tie that the lowest
    // 3 ids of class 3 win. The ids of class 3 are those under 128 that are not of class 4 and are odd.
    let class4 = [10usize, 20, 30, 40, 50];
    let router = synthetic(|e| {
        if class4.contains(&e) {
            4
        } else if e % 2 == 1 {
            3
        } else {
            0
        }
    });
    let device = Device::new(&dev, &router);
    let got = device.run(&ones, None, 7, 1);
    let want = ops::route(&host_logits(&router, &ones[0]), &router.bias);
    assert_eq!(
        got.experts,
        want.iter().map(|(e, _)| *e as u32).collect::<Vec<_>>(),
        "B"
    );
    assert_eq!(
        got.experts,
        vec![1, 3, 5, 10, 20, 30, 40, 50],
        "B: the eighth place goes to the lowest ids of the tie"
    );
    device.free();
}

/// The nanoseconds from the first block that has an item and passes the wait of the task `task` to the last block that
/// ends it. A block with no item of the task does not wait, so its time says nothing: the blocks `0..items` have items.
fn task_ns(profile: &[u64], blocks: usize, task: usize, items: usize) -> u64 {
    let slots = &profile[task * blocks * PROFILE_WORDS..(task + 1) * blocks * PROFILE_WORDS];
    let begin = slots.chunks(PROFILE_WORDS).take(items).map(|s| s[1]).min().unwrap();
    let end = slots.chunks(PROFILE_WORDS).take(items).map(|s| s[2]).max().unwrap();
    end - begin
}

#[test]
fn speed_of_the_router_at_1_row() {
    let Some(routers) = routers() else { return };
    // `MOE_CUBIN` names a profile cubin to measure instead of the built one (to cut the op short at a level).
    let dev = match std::env::var("MOE_CUBIN") {
        Ok(path) => CudaDevice::open_by_pci(
            &nagaya_mk_cuda::test_cards::card(),
            &std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}")),
        )
        .expect("open the GPU"),
        Err(_) => open_profile(),
    };
    let blocks = dev.sm_count() as usize;
    let flush_bytes = 1usize << 30;
    let flush = dev.alloc_device(flush_bytes).unwrap();
    let checks = dev.alloc_device(blocks * 4).unwrap();
    for router in &routers {
        // Two copies of the weights: the first run of the op meets cold code and cold weights, and the second meets warm
        // code (the flush of 1 GiB takes the code out of L2) and weights that no task has read.
        let (first, second) = (Device::new(&dev, router), Device::new(&dev, router));
        let x = upload(&dev, &rows_of(1, 3)[0]);
        let scratch = upload(&dev, &vec![0u32; route_scratch_words(1)]);
        let picks = upload(&dev, &vec![0u8; route_picks_bytes(1)]);
        let pairs = upload(&dev, &vec![0u8; route_pair_list_bytes(1)]);
        let mut table = MkTaskTable::default();
        table.ptr[ROUTE_TABLE_BIAS] = first.bias;
        let table_ptr = upload(&dev, &[table]);
        let mut ring = Ring::new(&dev, 1, true).unwrap();
        let read = Task::per_block(MkTaskDescriptor {
            opcode: OP_PROBE_READ,
            a: flush,
            c: checks,
            n: (flush_bytes / PROBE_ITEM_BYTES) as u32,
            k: u32::from(PROBE_READ_NC),
            ..Default::default()
        });
        let cold = Task::with_items(first.task(1, x, picks, pairs, scratch, table_ptr, false)).after(0);
        let warm = Task::with_items(second.task(1, x, picks, pairs, scratch, table_ptr, false)).after(1);
        ring.set(&[read, cold, warm]);
        let (cold_index, warm_index) = (ring.ring_index(1), ring.ring_index(2));
        let (mut cold_spans, mut warm_spans) = (Vec::new(), Vec::new());
        for _ in 0..9 {
            ring.zero_device().unwrap();
            run_with_watchdog(
                &ring,
                blocks as u32,
                DEFAULT_BLOCK_THREADS,
                dev.max_dynamic_shared_bytes().unwrap(),
            );
            let profile = ring.read_profile().unwrap();
            cold_spans.push(task_ns(&profile, blocks, cold_index, ROUTE_ITEMS_PER_ROW_TILE));
            warm_spans.push(task_ns(&profile, blocks, warm_index, ROUTE_ITEMS_PER_ROW_TILE));
        }
        cold_spans.sort_unstable();
        warm_spans.sort_unstable();
        eprintln!(
            "cuda_route speed: {}: 1 row: {:.1} us with cold code (best {:.1}), {:.1} us with warm code (best {:.1}); {:.2} MB of weights, {} items; the budget is 3.5 us",
            router.name,
            cold_spans[4] as f64 / 1e3,
            cold_spans[0] as f64 / 1e3,
            warm_spans[4] as f64 / 1e3,
            warm_spans[0] as f64 / 1e3,
            (EXPERTS * HIDDEN * 2) as f64 / 1e6,
            ROUTE_ITEMS_PER_ROW_TILE
        );
        ring.free().unwrap();
        for ptr in [x, scratch, picks, pairs, table_ptr] {
            dev.free_device(ptr).unwrap();
        }
        first.free();
        second.free();
    }
    dev.free_device(flush).unwrap();
    dev.free_device(checks).unwrap();
}
