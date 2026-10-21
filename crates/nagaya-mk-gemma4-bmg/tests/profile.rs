//! Profiles, not checks. Run each one alone, in its own process:
//! `cargo test --release --features xe --test profile -- --ignored --exact --nocapture <name>`.
//!
//! mkviolet's `src/golden.rs`, copied with no change in behavior (milestone 15s, step 4).
#![cfg(feature = "xe")]
// `!(err < tol)` fails on NaN too, as mkviolet meant. The other lints wait for the port, as in lib.rs.
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::chunks_exact_to_as_chunks,
    clippy::needless_range_loop
)]

mod common;

use common::*;
use nagaya_mk_gemma4_bmg::gemma4::*;
use nagaya_mk_gemma4_bmg::protocol::{MkTaskDescriptor, NO_EVENT, OP_MATMUL, OP_NOP};
use nagaya_mk_gemma4_bmg::sharding::Checkpoint;

/// Not a check: decodes 32 single tokens after prompts of several lengths on both cards and
/// prints where a token's wall time goes: card 0's ring, card 1's ring with the head, and the
/// host.
#[test]
#[ignore = "profile: not a check; run it alone"]
fn profile_decode() {
    if common::absent(&[]) {
        return;
    }
    let cl = common::cluster();
    // The server's shape: the council's slot and generation's, blocks of up to 256.
    let engine = Gemma4Engine::load_slots(&cl, &common::model_dir(), &[4096, 16384], 256).unwrap();
    engine.set_slot(1);
    for context in [20usize, 300, 2000, 8000] {
        let prompt: Vec<u32> = (0..context as u32).map(|i| 1000 + i % 5000).collect();
        let (mut logits, _) = engine.forward(&cl, &prompt, 0).unwrap();
        let (mut wall, mut c0, mut c1) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..32 {
            let next = top1(&logits) as u32;
            let (l, t) = engine.forward(&cl, &[next], prompt.len() + i).unwrap();
            logits = l;
            wall.push(t.wall_us);
            c0.push(t.card_us[0]);
            c1.push(t.card_us[1]);
        }
        let med = |v: &mut Vec<u128>| {
            v.sort();
            v[v.len() / 2]
        };
        let (w, a, b) = (med(&mut wall), med(&mut c0), med(&mut c1));
        println!(
            "decode after {} tokens, median of 32: wall {} us ({:.1} tok/s); card0 {} us, card1 {} us, host {} us",
            context,
            w,
            1e6 / w as f64,
            a,
            b,
            w as i128 - a as i128 - b as i128
        );
    }
}

/// Not a check: one block's card time by size, then a 4096-token prefill through both cards
/// at several block sizes.
#[test]
#[ignore = "profile: not a check; run it alone"]
fn profile_prefill_blocks() {
    if common::absent(&[]) {
        return;
    }
    let cl = common::cluster();
    let engine = Gemma4Engine::load_slots(&cl, &common::model_dir(), &[4096], 1024).unwrap();
    let prompt: Vec<u32> = (0..4096u32).map(|i| (1000 + i * 7919) % 200000).collect();
    // One block alone at position 1000: what a pipeline stage costs per card.
    for n in [8usize, 16, 32, 64, 128, 256] {
        let us: Vec<u128> = (0..3)
            .map(|_| engine.forward_blocks(&cl, &prompt[..n], 1000, n).unwrap().1.card_us[0])
            .collect();
        println!(
            "one block of {} tokens at 1000: card0 {} us",
            n,
            us.iter().min().unwrap()
        );
    }
    for block in [128usize, 256, 512, 1024] {
        let (_, t) = engine.forward_blocks(&cl, &prompt, 0, block).unwrap();
        println!(
            "prefill 4096 in blocks of {}: wall {} ms ({:.0} tok/s); card0 {} ms, card1 {} ms",
            block,
            t.wall_us / 1000,
            4096e6 / t.wall_us as f64,
            t.card_us[0] / 1000,
            t.card_us[1] / 1000
        );
    }
}

/// Not a check: what a decode step costs with n tokens and logits for every one (a
/// speculative verification of n - 1 drafted tokens), after a 3000-token prompt.
#[test]
#[ignore = "profile: not a check; run it alone"]
fn profile_verify_steps() {
    if common::absent(&[]) {
        return;
    }
    let cl = common::cluster();
    let engine = Gemma4Engine::load_slots(&cl, &common::model_dir(), &[4096], 256).unwrap();
    let prompt: Vec<u32> = (0..3000u32).map(|i| (1000 + i * 7919) % 200000).collect();
    engine.forward(&cl, &prompt, 0).unwrap();
    let next: Vec<u32> = (0..16u32).map(|i| (5000 + i * 104729) % 200000).collect();
    for n in [1usize, 2, 3, 4, 5, 6, 8, 12, 16] {
        let mut walls: Vec<u128> = (0..9)
            .map(|_| {
                let t = std::time::Instant::now();
                let jobs = [PipeJob::plain(&next[..n], 3000, (0..n).collect())];
                engine.run_jobs(&cl, &jobs, &mut PassTiming::default());
                t.elapsed().as_micros()
            })
            .collect();
        walls.sort();
        println!(
            "step of {} tokens with {} logit rows: {} us (median of 9)",
            n, n, walls[4]
        );
    }
}

/// Not a check: a 16384-token prefill in blocks of 256 through both cards, the size of
/// generation's KV slot, timed by quarter to show how cost grows with position.
#[test]
#[ignore = "profile: not a check; run it alone"]
fn profile_prefill_long() {
    if common::absent(&[]) {
        return;
    }
    let cl = common::cluster();
    let engine = Gemma4Engine::load_slots(&cl, &common::model_dir(), &[16384], 256).unwrap();
    let prompt: Vec<u32> = (0..16384u32).map(|i| (1000 + i * 7919) % 200000).collect();
    let mut total = 0u128;
    for q in 0..4 {
        let part = &prompt[q * 4096..(q + 1) * 4096];
        let (_, t) = engine.forward_blocks(&cl, part, q * 4096, 256).unwrap();
        total += t.wall_us;
        println!(
            "tokens {}..{}: wall {} ms ({:.0} tok/s)",
            q * 4096,
            (q + 1) * 4096,
            t.wall_us / 1000,
            4096e6 / t.wall_us as f64
        );
    }
    println!(
        "prefill 16384 in blocks of 256: {} ms ({:.0} tok/s)",
        total / 1000,
        16384e6 / total as f64
    );
}

/// Not a check: the service's shape with a generation slot of $GEMMA4_PROFILE_SEQ positions
/// (default 65536): a prefill to the slot's end in blocks of 512, timed by eighths, then
/// decode speed at its end.
#[test]
#[ignore = "profile: not a check; run it alone"]
fn profile_long_slot() {
    if common::absent(&[]) {
        return;
    }
    let seq: usize = std::env::var("GEMMA4_PROFILE_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(65536);
    let cl = common::cluster();
    let engine = Gemma4Engine::load_slots(&cl, &common::model_dir(), &[4096, seq], 512).unwrap();
    let vram: Vec<usize> = engine.shards.iter().map(|s| s.vram_bytes >> 20).collect();
    println!("slots [4096, {}]: VRAM per card {:?} MB", seq, vram);
    engine.set_slot(1);
    let n = seq - 64;
    let prompt: Vec<u32> = (0..n as u32).map(|i| (1000 + i * 7919) % 200000).collect();
    let step = n / 8;
    let mut logits = Vec::new();
    for q in 0..8 {
        let end = if q == 7 { n } else { (q + 1) * step };
        let (l, t) = engine
            .forward_blocks(&cl, &prompt[q * step..end], q * step, 512)
            .unwrap();
        logits = l;
        println!(
            "tokens {}..{}: wall {} ms ({:.0} tok/s)",
            q * step,
            end,
            t.wall_us / 1000,
            (end - q * step) as f64 * 1e6 / t.wall_us as f64
        );
    }
    let mut wall = Vec::new();
    for i in 0..32 {
        let (l, t) = engine.forward(&cl, &[top1(&logits) as u32], n + i).unwrap();
        logits = l;
        wall.push(t.wall_us);
    }
    wall.sort();
    println!(
        "decode after {} tokens, median of 32: {} us ({:.1} tok/s)",
        n,
        wall[16],
        1e6 / wall[16] as f64
    );
}

/// Not a check: launches each task of layers 0-5 on its own and prints its time, for 1 and
/// 64 tokens at position 0 and 80 tokens at position 600, to show where a block's time goes.
#[test]
#[ignore = "profile: not a check; run it alone"]
fn profile_ops() {
    if common::absent(&[]) {
        return;
    }
    let cl = common::cluster();
    let card = &cl.cards[0];
    let ckpt = Checkpoint::open_dir(&common::model_dir()).unwrap();
    let shard = CardShard::new(card, &ckpt, 0..6, false, 1024, 128).unwrap();
    for (n, pos0) in [(1usize, 0usize), (64, 0), (80, 600)] {
        let labels = unsafe { shard.build_ring(n, pos0, &[], None) };
        shard.write(card, Buf::X, 0, &vec![0.01f32; n * HIDDEN]);
        let whole = unsafe { shard.run(card, labels.len()) };
        let empty = unsafe { shard.run_range(card, 0, 0) };
        let mut total = 0u128;
        let mut rows = Vec::new();
        for (i, l) in labels.iter().enumerate() {
            let us = unsafe { shard.run_range(card, i, i + 1) };
            total += us;
            rows.push(format!("{}:{}", l, us));
        }
        println!(
            "n={} pos0={} whole ring {} us for {} tasks; empty launch {} us; per-task launches sum {} us",
            n,
            pos0,
            whole,
            labels.len(),
            empty,
            total
        );
        println!("  {}", rows.join(" "));
    }
    // Inside the ring, n=1: each task's cost as the growth of the ring prefix ending at it
    // (fastest of 3), summed by op over the 6 layers.
    let labels = unsafe { shard.build_ring(1, 0, &[], None) };
    shard.write(card, Buf::X, 0, &vec![0.01f32; HIDDEN]);
    let prefix = |end: usize| (0..3).map(|_| unsafe { shard.run_range(card, 0, end) }).min().unwrap() as i64;
    let mut by_op: Vec<(String, i64)> = Vec::new();
    let mut prev = prefix(0);
    for (i, l) in labels.iter().enumerate() {
        let now = prefix(i + 1);
        let op = l.split_once('.').map(|(_, o)| o.to_string()).unwrap_or(l.clone());
        match by_op.iter_mut().find(|e| e.0 == op) {
            Some(e) => e.1 += now - prev,
            None => by_op.push((op, now - prev)),
        }
        prev = now;
    }
    let total: i64 = by_op.iter().map(|e| e.1).sum();
    println!(
        "in-ring n=1, 6 layers, {} us: {}",
        total,
        by_op
            .iter()
            .map(|e| format!("{}:{}", e.0, e.1))
            .collect::<Vec<_>>()
            .join(" ")
    );
    // The same number of tasks that do nothing: what the task boundaries cost on their own.
    let n_tasks = unsafe { shard.build_ring(1, 0, &[], None) }.len();
    let nop = MkTaskDescriptor {
        opcode: OP_NOP,
        ..Default::default()
    };
    let mut ring = Vec::new();
    for i in 0..n_tasks {
        ring.push(MkTaskDescriptor {
            wait_event: if i == 0 { NO_EVENT } else { (i - 1) as u32 },
            signal_event: i as u32,
            ..nop
        });
    }
    let us = unsafe { shard.run_tasks(card, &ring) };
    println!(
        "{} no-op tasks: {} us, {:.1} us per task boundary",
        n_tasks,
        us,
        us as f64 / n_tasks as f64
    );
    // Matmuls of growing height over layer 0's expert weights, for 1 to 64 tokens: streaming rate
    // against how many rows the card has to spread.
    for (m, n) in [
        (1024usize, 1u32),
        (2048, 1),
        (4096, 1),
        (16384, 1),
        (131072, 1),
        (16384, 8),
        (16384, 32),
        (16384, 64),
    ] {
        let t = MkTaskDescriptor {
            opcode: OP_MATMUL,
            a: shard.buf_ptr(Buf::Norm, 0),
            b: shard.weights[0].experts_gate_up,
            c: shard.buf_ptr(Buf::MoeOut, 0),
            m: m as u32,
            n,
            k: HIDDEN as u32,
            wait_event: NO_EVENT,
            signal_event: 0,
            ..Default::default()
        };
        let us = (0..3).map(|_| unsafe { shard.run_tasks(card, &[t]) }).min().unwrap();
        let bytes = m * HIDDEN * 2;
        println!(
            "matmul n={} m={}: {} us, {:.0} GB/s",
            n,
            m,
            us,
            bytes as f64 / us as f64 / 1e3
        );
    }
}
