//! The experts of one layer on a B70 card against the host form (engine-abi "The first Inkling 2 binding", "The
//! experts on the card"): the plan, the gather, the gate and up rows with the softcapped SwiGLU, the down rows with
//! the weight of each pick, and the sum of the 8 picks. A token gets the same bits in any block.
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
use nagaya_mk_inkling2_bmg::host::ops::{EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, expert};
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

/// The output rows of `n` tokens on the card.
fn on_card(card: &B70Card, ring: &mut Ring, up: &Buf, down: &Buf, mx: bool, n: usize) -> Vec<f32> {
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
    let rows = out.read_f32(card, n * HIDDEN);
    for b in [xb, rb, plan, gathered, act, slots, out] {
        b.free(card);
    }
    rows
}

/// The host form of one token: the sum of the weighted experts in expert order.
fn on_host(w: &Weights, layer: usize, t: usize, cache: &mut HashMap<usize, (Vec<f32>, Vec<f32>)>) -> Vec<f32> {
    let x = token_row(t);
    let mut sum = vec![0f32; HIDDEN];
    for (e, weight) in picks(t) {
        let (gate_up, down) = cache.entry(e).or_insert_with(|| {
            let mut gate_up = vec![0f32; 2 * INTERMEDIATE * HIDDEN];
            for (r, row) in gate_up.chunks_mut(HIDDEN).enumerate() {
                w.expert_row(layer, e, false, r, row);
            }
            let mut down = vec![0f32; HIDDEN * INTERMEDIATE];
            for (r, row) in down.chunks_mut(INTERMEDIATE).enumerate() {
                w.expert_row(layer, e, true, r, row);
            }
            (gate_up, down)
        });
        let mut out = vec![0f32; HIDDEN];
        expert(&x, gate_up, down, &mut out);
        for (s, o) in sum.iter_mut().zip(&out) {
            *s += o * weight;
        }
    }
    sum
}

#[test]
fn the_experts_of_a_layer_on_the_card_give_the_host_form_and_the_same_bits_in_any_block() {
    if common::absent() {
        return;
    }
    let w = Weights::open(&common::weights_path()).unwrap();
    let cluster = L0Cluster::init_test(INKLING2_KERNEL, KERNEL_ENTRY);
    let card = &cluster.cards[0];
    let mut ring = Ring::new(card, 1 << 16);
    for layer in [0usize, 20] {
        let (up, down, mx) = slabs(&w, layer);
        let (up, down) = (Buf::of(card, &up), Buf::of(card, &down));
        let one = on_card(card, &mut ring, &up, &down, mx, 1);
        let five = on_card(card, &mut ring, &up, &down, mx, 5);
        let forty = on_card(card, &mut ring, &up, &down, mx, 40);
        assert_eq!(
            one.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            five[..HIDDEN].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "layer {layer}: token 0 alone and in a block of 5"
        );
        assert_eq!(
            five.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            forty[..5 * HIDDEN].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "layer {layer}: tokens 0 to 4 in a block of 5 and of 40"
        );
        let mut cache = HashMap::new();
        let mut worst = 0f64;
        for t in 0..40 {
            let want = on_host(&w, layer, t, &mut cache);
            let gap = common::rel(&forty[t * HIDDEN..(t + 1) * HIDDEN], &want);
            worst = worst.max(gap);
        }
        eprintln!("layer {layer}: the worst token of 40 is {worst:.3e} from the host form (relative L2)");
        assert!(worst <= TOLERANCE, "layer {layer}: {worst:.3e} > {TOLERANCE:e}");
        up.free(card);
        down.free(card);
    }
}
