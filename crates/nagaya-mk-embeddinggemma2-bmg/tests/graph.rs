//! The ring of one input, the checks of the ids and the mean, with no card and no checkpoint.

use nagaya_mk_bmg::protocol::*;
use nagaya_mk_embeddinggemma2_bmg::model::{Bufs, ids_problem, mean, pass, pass_until};
use nagaya_mk_embeddinggemma2_bmg::spec::*;
use nagaya_mk_embeddinggemma2_bmg::weights::{Layer, Weights};

/// Weights at distinct fake addresses, so a test can tell the tensors of each task apart.
fn weights() -> Weights {
    let mut next = 0x1000u64;
    let mut at = || {
        next += 0x1000;
        next
    };
    let layers = (0..LAYERS)
        .map(|l| Layer {
            input_norm: at(),
            q: at(),
            k: at(),
            v: at(),
            q_norm: at(),
            k_norm: at(),
            o: at(),
            post_attn_norm: at(),
            pre_ff_norm: at(),
            gate_up: at(),
            down: at(),
            post_ff_norm: at(),
            ple_gate: at(),
            ple_proj: at(),
            ple_norm: at(),
            scalar: 0.25 + l as f32,
        })
        .collect();
    Weights {
        embed: at(),
        ple_proj: at(),
        ple_norm: at(),
        layers,
        norm: at(),
        projection: at(),
    }
}

fn bufs() -> Bufs {
    let mut next = 0x1000_0000u64;
    Bufs::new(
        &mut |bytes| {
            let at = next;
            next += bytes as u64;
            at
        },
        MAX_TOKENS,
    )
}

#[test]
fn a_pass_runs_the_prologue_24_layers_of_19_tasks_and_the_tail() {
    let (w, b) = (weights(), bufs());
    let tasks = pass(&w, &b, 19);
    assert_eq!(tasks.len(), 5 + LAYERS * 19 + 2);
    assert_eq!(tasks[0].opcode, OP_EMBED);
    assert_eq!(tasks[0].b, w.embed);
    assert_eq!(tasks[1].opcode, OP_SCALE);
    assert_eq!(f32::from_bits(tasks[1].p0), 512f32.sqrt());
    assert_eq!((tasks[2].opcode, tasks[2].m), (OP_MATMUL, 24 * 512));
    assert_eq!(f32::from_bits(tasks[3].p0), 512f32.powf(-0.5));
    assert_eq!((tasks[4].opcode, tasks[4].n, tasks[4].k), (OP_RMSNORM, 19 * 24, 512));
    let last = tasks.last().unwrap();
    assert_eq!(
        (last.opcode, last.b, last.c, last.m),
        (OP_MATMUL, w.projection, b.out, 768)
    );
}

#[test]
fn each_layer_takes_its_attention_kind_its_slice_and_its_scalar() {
    let (w, b) = (weights(), bufs());
    let tasks = pass(&w, &b, 7);
    for l in 0..LAYERS {
        let t = &tasks[5 + 19 * l..5 + 19 * (l + 1)];
        let full = FULL_LAYERS.contains(&l);
        let (kv, d, window) = if full { (1, 512, 0) } else { (2, 256, 512) };
        // The query and key norms rotate the whole head; theta 1e6 for a full layer.
        for (task, heads) in [(&t[4], 4), (&t[5], kv)] {
            assert_eq!(task.opcode, OP_RMSNORM);
            assert_eq!((task.n, task.m, task.k, task.p0), (7 * heads, heads, d, 0));
            assert_ne!(task.flags & FLAG_ROPE, 0);
            assert_eq!(task.flags & FLAG_ROPE_FULL != 0, full, "layer {l}");
            assert_eq!(task.flags & FLAG_ROPE_GLOBAL, 0, "layer {l} rotates the whole head");
        }
        // The value norm has no weights.
        assert_eq!((t[6].opcode, t[6].b, t[6].n, t[6].k), (OP_RMSNORM, 0, 7 * kv, d));
        assert_eq!(t[7].opcode, OP_BIATTN);
        assert_eq!((t[7].m, t[7].k, t[7].p0), (4 | kv << 16, window, d), "layer {l}");
        assert_eq!(f32::from_bits(t[7].layer), 1.0);
        // The three residual norms add to the stream.
        for i in [9, 13, 17] {
            assert_eq!((t[i].opcode, t[i].c), (OP_RMSNORM, b.x), "layer {l} task {i}");
            assert_ne!(t[i].flags & FLAG_ACCUMULATE, 0, "layer {l} task {i}");
        }
        assert_eq!((t[11].opcode, t[11].b, t[11].m), (OP_GEGLU, w.layers[l].gate_up, 2048));
        assert_eq!(t[15].opcode, OP_GELU_MUL);
        assert_eq!(
            (t[15].b, t[15].m),
            (b.ple + (l * 512 * 4) as u64, 24 * 512),
            "layer {l}"
        );
        assert_eq!(t[17].b, w.layers[l].ple_norm);
        assert_eq!((t[18].opcode, t[18].a, t[18].c), (OP_SCALE, b.x, b.x), "layer {l}");
        assert_eq!(f32::from_bits(t[18].p0), 0.25 + l as f32);
    }
}

#[test]
fn a_probe_stops_after_its_layers_with_no_tail() {
    let (w, b) = (weights(), bufs());
    assert_eq!(pass_until(&w, &b, 3, 0, false).len(), 5);
    assert_eq!(pass_until(&w, &b, 3, 2, false).len(), 5 + 2 * 19);
}

#[test]
fn the_engine_refuses_no_ids_too_many_ids_and_an_id_past_the_vocabulary() {
    assert!(ids_problem(&[]).is_some());
    assert!(ids_problem(&vec![2; MAX_TOKENS + 1]).is_some());
    assert!(ids_problem(&[2, VOCAB as u32, 1]).unwrap().contains("vocabulary"));
    assert_eq!(ids_problem(&vec![2; MAX_TOKENS]), None);
    assert_eq!(ids_problem(&[2, VOCAB as u32 - 1, 1]), None);
}

#[test]
fn the_mean_sums_each_column_in_f64_in_row_order() {
    // 1e8, then 15 rows of 1: a sum in f32 loses each 1, and a sum in f64 keeps them.
    let n = 16;
    let mut rows = vec![0.0f32; n * DIMENSION];
    rows[0] = 1.0e8;
    for t in 1..n {
        rows[t * DIMENSION] = 1.0;
        rows[t * DIMENSION + 1] = t as f32;
    }
    let m = mean(&rows, n);
    assert_eq!(m.len(), DIMENSION);
    assert_eq!(m[0], 6_250_001.0, "(1e8 + 15) / 16 rounded to f32");
    assert_eq!(m[1], 120.0 / 16.0);
    assert_eq!(m[2], 0.0);
}
