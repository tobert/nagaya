//! The experts of one layer on a B70 card against the host form (engine-abi "The first Inkling 2 binding", "The
//! experts on the card"): the plan, the gather, the gate and up rows with the softcapped SwiGLU, the down rows with
//! the weight of each pick, and the sum of the 8 picks. A block of 1 to 8 tokens takes the row path, and a block of 9
//! or more the XMX path. On each path, a token gets the same bits in any block. A test also times the row path.
//!
//! Run it alone, on a card that no engine of the active profile names (`AGENTS.md`, "The engines on the B70 cards"):
//! `NAGAYA_TEST_CARDS=<pci> cargo test -p nagaya-mk-inkling2-bmg --features xe --test xe_experts -- --test-threads=1`.
#![cfg(feature = "xe")]

mod common;

use std::collections::HashMap;
use std::ffi::c_void;

use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::{MOE_PLAN_BYTES, MoERouteDecision};
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::{INKLING2_KERNEL, KERNEL_ENTRY, tasks};
use nagaya_mk_inkling2_bmg::host::ops::{EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, expert, matvec, swiglu};
use nagaya_mk_inkling2_bmg::host::weights::{ExpertFormat, Weights};
use nagaya_mk_inkling2_bmg::repack::expert_slab;

/// The largest relative L2 gap of a token row of the card from the host form. The two sum in other orders: the card
/// in steps of 16 columns with x in three bf16 parts, the host in one f32 sum.
const TOLERANCE: f64 = 1e-5;

/// A buffer in device memory.
struct Buf(*mut u8);

impl Buf {
    fn of(card: &B70Card, data: &[u8]) -> Self {
        let buf = Self::zeros(card, data.len());
        let stage = card.alloc_host_usm::<u8>(data.len().max(64));
        // SAFETY: `stage` holds `data.len()` bytes of host memory.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), stage, data.len()) };
        card.memcpy_sync(buf.0 as *mut c_void, stage as *const c_void, data.len());
        card.free(stage as *mut c_void);
        buf
    }

    fn zeros(card: &B70Card, bytes: usize) -> Self {
        let ptr = card.alloc_device_usm::<u8>(bytes.max(64));
        card.fill_zero_sync(ptr as *mut c_void, bytes.max(64));
        Self(ptr)
    }

    fn read_f32(&self, card: &B70Card, n: usize) -> Vec<f32> {
        let stage = card.alloc_host_usm::<u8>((n * 4).max(64));
        card.memcpy_sync(stage as *mut c_void, self.0 as *const c_void, n * 4);
        let mut out = vec![0f32; n];
        // SAFETY: `stage` holds `n * 4` bytes, and `out` has room for them.
        unsafe { std::ptr::copy_nonoverlapping(stage, out.as_mut_ptr() as *mut u8, n * 4) };
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

fn bytes_of<T: Copy>(data: &[T]) -> Vec<u8> {
    // SAFETY: `T` is plain data, and the slice holds `size_of_val(data)` bytes.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }.to_vec()
}

/// A fixed input row for token `t`: values of an RMS near 1, from a small generator.
fn token_row(t: usize) -> Vec<f32> {
    let mut s = 0x9E37_79B9u32 ^ (t as u32).wrapping_mul(0x85EB_CA6B);
    (0..HIDDEN)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32 - 0.5) * 3.4
        })
        .collect()
}

/// The 8 picks of token `t`, in expert order, with their weights.
fn picks(t: usize) -> [(usize, f32); TOP_K] {
    let mut experts: Vec<usize> = (0..TOP_K).map(|j| (t * 37 + j * 17 + j * j) % EXPERTS).collect();
    experts.sort_unstable();
    experts.dedup();
    let mut e = 0;
    while experts.len() < TOP_K {
        if !experts.contains(&e) {
            experts.push(e);
        }
        e += 1;
    }
    experts.sort_unstable();
    let mut out = [(0, 0.0); TOP_K];
    for (j, &e) in experts.iter().enumerate() {
        out[j] = (e, 0.2 + 0.15 * ((t + j) % 7) as f32);
    }
    out
}

fn route(t: usize) -> MoERouteDecision {
    let p = picks(t);
    let mut r = MoERouteDecision::default();
    for (j, (e, w)) in p.iter().enumerate() {
        r.top_k_indices[j] = *e as u32;
        r.top_k_weights[j] = *w;
    }
    r.token = t as u32;
    r
}

/// The slabs of the gate and up rows and of the down rows of one layer, as the load builds them.
fn slabs(w: &Weights, layer: usize) -> (Vec<u8>, Vec<u8>, bool) {
    let e = &w.layers[layer].experts;
    let bytes = w.checkpoint.file_bytes();
    let (mx, f13, f2) = match &e.format {
        ExpertFormat::Nvfp4 { w13_factor, w2_factor } => (false, Some(&w13_factor[..]), Some(&w2_factor[..])),
        ExpertFormat::Mxfp8 => (true, None, None),
    };
    let up = expert_slab(
        &bytes[e.w13.clone()],
        &bytes[e.w13_scale.clone()],
        f13,
        EXPERTS,
        2 * INTERMEDIATE,
        HIDDEN,
        mx,
        true,
    )
    .unwrap();
    let down = expert_slab(
        &bytes[e.w2.clone()],
        &bytes[e.w2_scale.clone()],
        f2,
        EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        mx,
        false,
    )
    .unwrap();
    (up, down, mx)
}

/// The outputs of each expert op of `n` tokens on the card, and the order of the plan.
struct Run {
    /// The rows of `OP_EXPERT_UP`, one row of `INTERMEDIATE` for each entry of the plan.
    act: Vec<f32>,
    /// The rows of `OP_EXPERT_DOWN`, one row of `HIDDEN` for each slot `8 t + pick`.
    slots: Vec<f32>,
    /// The rows of `OP_EXPERT_SUM`, one row of `HIDDEN` for each token.
    out: Vec<f32>,
    /// The slot of each entry of the plan, in plan order.
    entry_slot: Vec<u32>,
}

/// The word of `MoEPlan` where `entry_slot` starts: after `count`, `offset`, `n_active` and `active`.
const ENTRY_SLOT_WORD: usize = EXPERTS + EXPERTS + 1 + 1 + EXPERTS;

/// The tasks of the experts of `n` tokens on the card.
fn on_card(card: &B70Card, ring: &mut Ring, up: &Buf, down: &Buf, mx: bool, n: usize) -> Run {
    let x: Vec<f32> = (0..n).flat_map(token_row).collect();
    let routes: Vec<MoERouteDecision> = (0..n).map(route).collect();
    let (xb, rb) = (Buf::of(card, &bytes_of(&x)), Buf::of(card, &bytes_of(&routes)));
    let plan = Buf::zeros(card, MOE_PLAN_BYTES);
    let gathered = Buf::zeros(card, n * TOP_K * HIDDEN * 4);
    let act = Buf::zeros(card, n * TOP_K * INTERMEDIATE * 4);
    let slots = Buf::zeros(card, n * TOP_K * HIDDEN * 4);
    let out = Buf::zeros(card, n * HIDDEN * 4);
    let (n32, h, i) = (n as u32, HIDDEN as u32, INTERMEDIATE as u32);
    ring.set(&[
        tasks::moe_plan(rb.at(), plan.at(), n32),
        tasks::moe_gather(xb.at(), gathered.at(), plan.at(), n32, h),
        tasks::expert_up(gathered.at(), up.at(), act.at(), plan.at(), i, h, mx),
        tasks::expert_down(act.at(), down.at(), slots.at(), plan.at(), h, i, mx),
        tasks::expert_sum(slots.at(), out.at(), n32, h),
    ]);
    // SAFETY: each buffer holds the rows of its task for `n` tokens.
    unsafe { ring.run(card) };
    let words = plan.read_f32(card, ENTRY_SLOT_WORD + n * TOP_K);
    let run = Run {
        act: act.read_f32(card, n * TOP_K * INTERMEDIATE),
        slots: slots.read_f32(card, n * TOP_K * HIDDEN),
        out: out.read_f32(card, n * HIDDEN),
        entry_slot: words[ENTRY_SLOT_WORD..].iter().map(|v| v.to_bits()).collect(),
    };
    for b in [xb, rb, plan, gathered, act, slots, out] {
        b.free(card);
    }
    run
}

/// The gate and up rows, then the down rows, of expert `e` of `layer`, as the host form reads them.
fn host_expert(w: &Weights, layer: usize, e: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gate_up = vec![0f32; 2 * INTERMEDIATE * HIDDEN];
    for (r, row) in gate_up.chunks_mut(HIDDEN).enumerate() {
        w.expert_row(layer, e, false, r, row);
    }
    let mut down = vec![0f32; HIDDEN * INTERMEDIATE];
    for (r, row) in down.chunks_mut(INTERMEDIATE).enumerate() {
        w.expert_row(layer, e, true, r, row);
    }
    (gate_up, down)
}

/// The host form of one token: the sum of the weighted experts in expert order.
fn on_host(w: &Weights, layer: usize, t: usize, cache: &mut HashMap<usize, (Vec<f32>, Vec<f32>)>) -> Vec<f32> {
    let x = token_row(t);
    let mut sum = vec![0f32; HIDDEN];
    for (e, weight) in picks(t) {
        let (gate_up, down) = cache.entry(e).or_insert_with(|| host_expert(w, layer, e));
        let mut out = vec![0f32; HIDDEN];
        expert(&x, gate_up, down, &mut out);
        for (s, o) in sum.iter_mut().zip(&out) {
            *s += o * weight;
        }
    }
    sum
}

/// The bits of a row.
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|v| v.to_bits()).collect()
}

/// The worst relative L2 gap of each op of `run` from the host form: the up rows of each entry against the host
/// SwiGLU, and the down rows of each slot against the host down rows of the card's up row, times the weight. The
/// third value is the count of down rows whose bits are not those of the host.
fn op_gaps(
    w: &Weights,
    layer: usize,
    run: &Run,
    cache: &mut HashMap<usize, (Vec<f32>, Vec<f32>)>,
) -> (f64, f64, usize) {
    let (mut up_gap, mut down_gap, mut other) = (0f64, 0f64, 0);
    for (q, &slot) in run.entry_slot.iter().enumerate() {
        let (t, pick) = (slot as usize / TOP_K, slot as usize % TOP_K);
        let (e, weight) = picks(t)[pick];
        let (gate_up, down) = cache.entry(e).or_insert_with(|| host_expert(w, layer, e));
        let mut both = vec![0f32; 2 * INTERMEDIATE];
        matvec(gate_up, &token_row(t), &mut both);
        let mut want = vec![0f32; INTERMEDIATE];
        swiglu(&both[..INTERMEDIATE], &both[INTERMEDIATE..], &mut want);
        let act = &run.act[q * INTERMEDIATE..(q + 1) * INTERMEDIATE];
        up_gap = up_gap.max(common::rel(act, &want));
        let mut want = vec![0f32; HIDDEN];
        matvec(down, act, &mut want);
        want.iter_mut().for_each(|v| *v *= weight);
        let got = &run.slots[slot as usize * HIDDEN..(slot as usize + 1) * HIDDEN];
        down_gap = down_gap.max(common::rel(got, &want));
        other += usize::from(bits(got) != bits(&want));
    }
    (up_gap, down_gap, other)
}

/// A block of 1 to 8 tokens takes the row path of the expert ops, and a block of 9 or more the XMX path. Each path
/// gives a token the same bits in any block of its counts. Each op of each path follows the host form, in an NVFP4
/// layer (0) and an MXFP8 layer (20).
#[test]
fn the_experts_of_a_layer_on_the_card_give_the_host_form_and_the_same_bits_on_each_path() {
    if common::absent() {
        return;
    }
    let w = Weights::open(&common::weights_path()).unwrap();
    let cluster = L0Cluster::init_test(INKLING2_KERNEL, KERNEL_ENTRY);
    let card = &cluster.cards[0];
    let mut ring = Ring::new(card, 1 << 16);
    for (layer, want_mx) in [(0usize, false), (20, true)] {
        let (up, down, mx) = slabs(&w, layer);
        assert_eq!(mx, want_mx, "layer {layer}: the format of its experts");
        let (up, down) = (Buf::of(card, &up), Buf::of(card, &down));
        let runs: Vec<(usize, Run)> = [1usize, 2, 8, 9, 40]
            .iter()
            .map(|&n| (n, on_card(card, &mut ring, &up, &down, mx, n)))
            .collect();
        let out = |n: usize| &runs.iter().find(|(m, _)| *m == n).unwrap().1.out;
        for n in [2, 8] {
            assert_eq!(
                bits(out(1)),
                bits(&out(n)[..HIDDEN]),
                "layer {layer}: token 0 alone and in a block of {n}"
            );
        }
        assert_eq!(
            bits(out(9)),
            bits(&out(40)[..9 * HIDDEN]),
            "layer {layer}: tokens 0 to 8 in a block of 9 and of 40"
        );
        let mut cache = HashMap::new();
        for (n, run) in &runs {
            let (up_gap, down_gap, other) = op_gaps(&w, layer, run, &mut cache);
            eprintln!(
                "layer {layer}, {n} tokens: up {up_gap:.3e}, down {down_gap:.3e} from the host form (relative L2); \
                 {other} of {} down rows with other bits",
                run.entry_slot.len()
            );
            assert!(
                up_gap <= TOLERANCE,
                "layer {layer}, {n} tokens: up {up_gap:.3e} > {TOLERANCE:e}"
            );
            assert!(
                down_gap <= TOLERANCE,
                "layer {layer}, {n} tokens: down {down_gap:.3e} > {TOLERANCE:e}"
            );
        }
        let mut worst = 0f64;
        for t in 0..40 {
            let want = on_host(&w, layer, t, &mut cache);
            worst = worst.max(common::rel(&out(40)[t * HIDDEN..(t + 1) * HIDDEN], &want));
        }
        let alone = common::rel(out(1), &on_host(&w, layer, 0, &mut cache));
        eprintln!(
            "layer {layer}: the worst token of 40 is {worst:.3e}, and token 0 alone {alone:.3e}, from the host form \
             (relative L2)"
        );
        assert!(worst <= TOLERANCE, "layer {layer}: {worst:.3e} > {TOLERANCE:e}");
        assert!(
            alone <= TOLERANCE,
            "layer {layer}: token 0 alone {alone:.3e} > {TOLERANCE:e}"
        );
        up.free(card);
        down.free(card);
    }
}

/// The longest time of one pair of expert ops of one token, in us. The goal is about 8 ms for
/// the 48 pairs of a decode step at 3 cycles, 167 us each. The row path took 172 us in NVFP4 and 165 us in MXFP8 on
/// 2026-11-22, and the XMX path 680 us. The bound leaves room for the noise of a shared card.
const PAIR_US: f64 = 200.0;

/// The 16 plans of one token whose picks cover the 128 experts once: plan r picks the experts 8r to 8r + 7. A pass of
/// these pairs thus reads each expert once, and no pair reads the bytes of another from the cache.
fn disjoint_route(r: usize) -> MoERouteDecision {
    let mut d = MoERouteDecision::default();
    for j in 0..TOP_K {
        d.top_k_indices[j] = (TOP_K * r + j) as u32;
        d.top_k_weights[j] = 0.25;
    }
    d
}

/// The expert ops of one token read the packed planes near the bandwidth: a pair of up and down takes at most
/// `PAIR_US`, in an NVFP4 and in an MXFP8 layer. The test times a ring of 48 pairs against a ring with none.
#[test]
fn the_expert_ops_of_one_token_read_the_planes_near_the_bandwidth() {
    if common::absent() {
        return;
    }
    let w = Weights::open(&common::weights_path()).unwrap();
    let cluster = L0Cluster::init_test(INKLING2_KERNEL, KERNEL_ENTRY);
    let card = &cluster.cards[0];
    let mut ring = Ring::new(card, 1 << 16);
    let plans = EXPERTS / TOP_K;
    let pairs = 48;
    let mut slow = Vec::new();
    for layer in [0usize, 20] {
        let (up_bytes, down_bytes, mx) = slabs(&w, layer);
        let pair_bytes = (up_bytes.len() + down_bytes.len()) as f64 / plans as f64;
        let (up, down) = (Buf::of(card, &up_bytes), Buf::of(card, &down_bytes));
        let x = Buf::of(card, &bytes_of(&token_row(0)));
        let mut each = Vec::new();
        for r in 0..plans {
            let route = Buf::of(card, &bytes_of(&[disjoint_route(r)]));
            let plan = Buf::zeros(card, MOE_PLAN_BYTES);
            let gathered = Buf::zeros(card, TOP_K * HIDDEN * 4);
            let act = Buf::zeros(card, TOP_K * INTERMEDIATE * 4);
            let slots = Buf::zeros(card, TOP_K * HIDDEN * 4);
            each.push((route, plan, gathered, act, slots));
        }
        let (h, i) = (HIDDEN as u32, INTERMEDIATE as u32);
        let mut base = Vec::new();
        for (route, plan, gathered, _, _) in &each {
            base.push(tasks::moe_plan(route.at(), plan.at(), 1));
            base.push(tasks::moe_gather(x.at(), gathered.at(), plan.at(), 1, h));
        }
        let mut ups = base.clone();
        let mut full = base.clone();
        for p in 0..pairs {
            let (_, plan, gathered, act, slots) = &each[p % plans];
            let up_task = tasks::expert_up(gathered.at(), up.at(), act.at(), plan.at(), i, h, mx);
            ups.push(up_task);
            full.push(up_task);
            full.push(tasks::expert_down(act.at(), down.at(), slots.at(), plan.at(), h, i, mx));
        }
        let mut time = |tasks: &[nagaya_mk_bmg::protocol::MkTaskDescriptor]| {
            ring.set(tasks);
            let mut best = f64::MAX;
            for _ in 0..5 {
                let started = std::time::Instant::now();
                // SAFETY: each buffer holds the rows of its task for one token.
                unsafe { ring.run(card) };
                best = best.min(started.elapsed().as_secs_f64() * 1e6);
            }
            best
        };
        let (base_us, ups_us, full_us) = (time(&base), time(&ups), time(&full));
        let pair_us = (full_us - base_us) / pairs as f64;
        let up_us = (ups_us - base_us) / pairs as f64;
        eprintln!(
            "layer {layer} ({}): a pair of expert ops of one token takes {pair_us:.1} us (up {up_us:.1}, down {:.1}), \
             {:.0} GB/s of the planes",
            if mx { "MXFP8" } else { "NVFP4" },
            pair_us - up_us,
            pair_bytes / pair_us / 1e3
        );
        slow.push((layer, pair_us));
        for (route, plan, gathered, act, slots) in each {
            for b in [route, plan, gathered, act, slots] {
                b.free(card);
            }
        }
        for b in [x, up, down] {
            b.free(card);
        }
    }
    for (layer, pair_us) in slow {
        assert!(
            pair_us <= PAIR_US,
            "layer {layer}: a pair takes {pair_us:.1} us > {PAIR_US:.1} us"
        );
    }
}

/// The time of a pair of expert ops for blocks of 1 to 16 tokens, in an NVFP4 layer. Blocks of 1 to 8 tokens take
/// the row path, and 9 or more the XMX path. It prints the times and checks nothing: it is the measure for the choice of the counts of the row path. Each pair reads one plan, so a part of its planes can
/// come from the cache.
#[test]
fn the_time_of_the_expert_ops_for_each_count_of_tokens() {
    if common::absent() {
        return;
    }
    let w = Weights::open(&common::weights_path()).unwrap();
    let cluster = L0Cluster::init_test(INKLING2_KERNEL, KERNEL_ENTRY);
    let card = &cluster.cards[0];
    let mut ring = Ring::new(card, 1 << 16);
    let pairs = 48;
    let (up_bytes, down_bytes, mx) = slabs(&w, 0);
    let (up, down) = (Buf::of(card, &up_bytes), Buf::of(card, &down_bytes));
    for n in [1usize, 2, 4, 8, 9, 16] {
        let x: Vec<f32> = (0..n).flat_map(token_row).collect();
        let routes: Vec<MoERouteDecision> = (0..n).map(route).collect();
        let (xb, rb) = (Buf::of(card, &bytes_of(&x)), Buf::of(card, &bytes_of(&routes)));
        let plan = Buf::zeros(card, MOE_PLAN_BYTES);
        let gathered = Buf::zeros(card, n * TOP_K * HIDDEN * 4);
        let act = Buf::zeros(card, n * TOP_K * INTERMEDIATE * 4);
        let slots = Buf::zeros(card, n * TOP_K * HIDDEN * 4);
        let (n32, h, i) = (n as u32, HIDDEN as u32, INTERMEDIATE as u32);
        let base = vec![
            tasks::moe_plan(rb.at(), plan.at(), n32),
            tasks::moe_gather(xb.at(), gathered.at(), plan.at(), n32, h),
        ];
        let mut full = base.clone();
        for _ in 0..pairs {
            full.push(tasks::expert_up(gathered.at(), up.at(), act.at(), plan.at(), i, h, mx));
            full.push(tasks::expert_down(act.at(), down.at(), slots.at(), plan.at(), h, i, mx));
        }
        let mut time = |tasks: &[nagaya_mk_bmg::protocol::MkTaskDescriptor]| {
            ring.set(tasks);
            let mut best = f64::MAX;
            for _ in 0..5 {
                let started = std::time::Instant::now();
                // SAFETY: each buffer holds the rows of its task for `n` tokens.
                unsafe { ring.run(card) };
                best = best.min(started.elapsed().as_secs_f64() * 1e6);
            }
            best
        };
        let pair_us = (time(&full) - time(&base)) / pairs as f64;
        let experts: std::collections::BTreeSet<usize> = (0..n).flat_map(|t| picks(t).map(|(e, _)| e)).collect();
        eprintln!(
            "{n} tokens ({} experts, {}): a pair of expert ops takes {pair_us:.1} us",
            experts.len(),
            if n <= 8 { "row path" } else { "XMX path" }
        );
        for b in [xb, rb, plan, gathered, act, slots] {
            b.free(card);
        }
    }
    up.free(card);
    down.free(card);
}
