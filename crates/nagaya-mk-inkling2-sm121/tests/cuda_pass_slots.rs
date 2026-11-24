//! GPU gates of a pass with many slots (milestone 37 step 11): a call names its slot, its cycles and its logit rows.
//!
//! - A pass on one slot leaves the planes and the conv states of the others as they were, and each slot gives the bits of
//!   a lone slot.
//! - The cycles of a call give the bits of a pass that is fixed at that count.
//! - A call gives logits for any rows inside a span of 16 rows, or none, with the same bits as the call that gives all of
//!   them. A refused call changes nothing.
//! - A read of the parts, a write into another slot and a pass give the bits of the run with no snapshot, in each cache
//!   format.
//! - A clear, then a pass, give the bits of a fresh slot. A pass allocates nothing on the device.
//!
//! `INKLING2_MODEL_DIR=<folder> INKLING2_GOLDEN_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p
//!  nagaya-mk-inkling2-sm121 --features cuda --test cuda_pass_slots -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;

use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_inkling2_sm121::pass::step::{Pass, PassConfig};
use nagaya_mk_inkling2_sm121::pass::weights::DeviceWeights;
use nagaya_mk_inkling2_sm121::spec::{self, LAYERS, LOGITS};

const SEQ: usize = 1024;

fn model_dir(who: &str) -> Option<PathBuf> {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(d) if PathBuf::from(&d).join(spec::WEIGHTS_FILE).exists() => Some(d.into()),
        _ => {
            assert!(
                !require,
                "INKLING2_MODEL_DIR has no checkpoint and NAGAYA_REQUIRE_MODELS is set"
            );
            eprintln!("{who}: skipped, INKLING2_MODEL_DIR has no checkpoint");
            None
        }
    }
}

fn chat_ids() -> Vec<u32> {
    let dir = std::env::var_os("INKLING2_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/inkling2-golden")));
    let path = dir.join("inkling2-mini.quant.oracle.json");
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap();
    json["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "chat")
        .expect("the golden has the prompt chat")["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

/// `rows` ids that depend on `seed`, so that two slots hold other rows.
fn ids_of(chat: &[u32], seed: usize, at: usize, rows: usize) -> Vec<u32> {
    (at..at + rows)
        .map(|i| chat[(i * 7 + 3 + seed * 5) % chat.len()])
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// One hash for each buffer of the first `states` states of slot `slot`: the planes and the conv state.
fn snapshot(pass: &Pass<'_>, slot: usize, states: usize) -> Vec<[u8; 32]> {
    let mut out = Vec::new();
    for run in pass.slots[slot].runs.iter().take(states) {
        for buf in [&run.k_plane, &run.v_plane, &run.conv] {
            out.push(*blake3::hash(&buf.read(0, buf.bytes()).unwrap()).as_bytes());
        }
    }
    out
}

/// A lone pass: one slot, and `cycles` as its only count. It runs the calls and returns the last logits and the slot.
fn lone(
    dev: &CudaDevice,
    weights: &DeviceWeights<'_>,
    cycles: usize,
    format: CacheFormat,
    calls: &[Vec<u32>],
) -> (Vec<f32>, Vec<[u8; 32]>) {
    let mut pass = Pass::with_cache_format(dev, weights, SEQ, 512, cycles, format).unwrap();
    let mut out = Vec::new();
    for ids in calls {
        out = pass.run(ids, 1).unwrap();
    }
    let states = LAYERS + (cycles - 1) * spec::CYCLE_LAYERS;
    (out, snapshot(&pass, 0, states))
}

fn open_slots<'d>(
    dev: &'d CudaDevice,
    weights: &'d DeviceWeights<'d>,
    seqs: &[usize],
    format: CacheFormat,
) -> Pass<'d> {
    Pass::open(
        dev,
        weights,
        &PassConfig {
            seqs: seqs.to_vec(),
            max_rows: 512,
            max_cycles: 3,
            profile: false,
            format,
        },
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn a_pass_on_one_slot_leaves_the_others_and_each_slot_gives_the_bits_of_a_lone_slot() {
    let Some(dir) = model_dir("cuda_pass_slots") else {
        return;
    };
    let dev = common::gpu::open();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let chat = chat_ids();
    let call = |seed: usize, at: usize, rows: usize| ids_of(&chat, seed, at, rows);
    // Slot 0 and slot 1 hold other rows; the calls alternate.
    let (a, b) = (
        vec![call(0, 0, 17), call(0, 17, 1), call(0, 18, 1)],
        vec![call(1, 0, 16), call(1, 16, 1), call(1, 17, 1)],
    );
    let (want_a, snap_a) = lone(&dev, &weights, 3, CacheFormat::F32, &a);
    let (want_b, snap_b) = lone(&dev, &weights, 3, CacheFormat::F32, &b);
    assert_ne!(
        bits(&want_a),
        bits(&want_b),
        "the two slots must differ for the gate to mean something"
    );
    let mut pass = open_slots(&dev, &weights, &[SEQ, SEQ], CacheFormat::F32);
    let (mut got_a, mut got_b) = (Vec::new(), Vec::new());
    for i in 0..3 {
        let before_1 = snapshot(&pass, 1, 48);
        let live = dev.live_buffers();
        got_a = pass.run_slot(0, 3, &a[i], &[a[i].len() - 1]).unwrap();
        assert_eq!(dev.live_buffers(), live, "a call allocated on the device");
        assert_eq!(
            snapshot(&pass, 1, 48),
            before_1,
            "a pass on slot 0 moved slot 1 (call {i})"
        );
        let before_0 = snapshot(&pass, 0, 48);
        got_b = pass.run_slot(1, 3, &b[i], &[b[i].len() - 1]).unwrap();
        assert_eq!(dev.live_buffers(), live, "a call allocated on the device");
        assert_eq!(
            snapshot(&pass, 0, 48),
            before_0,
            "a pass on slot 1 moved slot 0 (call {i})"
        );
    }
    assert_eq!(bits(&got_a), bits(&want_a), "slot 0 differs from a lone slot");
    assert_eq!(bits(&got_b), bits(&want_b), "slot 1 differs from a lone slot");
    assert_eq!(
        snapshot(&pass, 0, 48),
        snap_a,
        "the planes of slot 0 differ from a lone slot"
    );
    assert_eq!(
        snapshot(&pass, 1, 48),
        snap_b,
        "the planes of slot 1 differ from a lone slot"
    );
    assert_eq!(pass.slots[0].position, 19);
    // The slots can differ in length.
    let mut mixed = open_slots(&dev, &weights, &[SEQ, 512], CacheFormat::F32);
    let got = mixed.run_slot(1, 3, &b[0], &[15]).unwrap();
    let live = dev.live_buffers();
    assert_eq!(got.len(), LOGITS);
    let e = mixed.run_slot(1, 3, &ids_of(&chat, 1, 0, 512), &[0]).unwrap_err();
    assert!(e.contains("slot 1 holds 512 positions"), "{e}");
    assert_eq!(mixed.slots[1].position, 16, "a refused call moved the position");
    assert_eq!(dev.live_buffers(), live);
}

#[test]
fn the_cycles_of_a_call_give_the_bits_of_a_pass_fixed_at_that_count() {
    let Some(dir) = model_dir("cuda_pass_slots") else {
        return;
    };
    let dev = common::gpu::open();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let chat = chat_ids();
    let calls = vec![
        ids_of(&chat, 0, 0, 17),
        ids_of(&chat, 0, 17, 1),
        ids_of(&chat, 0, 18, 1),
    ];
    let mut pass = open_slots(&dev, &weights, &[SEQ, SEQ, SEQ], CacheFormat::F32);
    // Slot `c - 1` runs at `c` cycles, in calls that also move the shape of the ring and the count of cycles of the
    // other slots.
    let mut got: Vec<Vec<f32>> = vec![Vec::new(); 3];
    for ids in &calls {
        for cycles in (1..=3).rev() {
            got[cycles - 1] = pass.run_slot(cycles - 1, cycles, ids, &[ids.len() - 1]).unwrap();
        }
    }
    for cycles in 1..=3 {
        let (want, snap) = lone(&dev, &weights, cycles, CacheFormat::F32, &calls);
        assert_eq!(
            bits(&got[cycles - 1]),
            bits(&want),
            "{cycles} cycles: the logits differ"
        );
        let states = LAYERS + (cycles - 1) * spec::CYCLE_LAYERS;
        assert_eq!(
            snapshot(&pass, cycles - 1, states),
            snap,
            "{cycles} cycles: the slot differs"
        );
        eprintln!("cycles {cycles}: the logits and the {states} states have the bits of a pass fixed at {cycles}");
    }
    // A count outside the open is refused, and the slot stays as it was.
    for bad in [0usize, 4] {
        let e = pass.run_slot(0, bad, &calls[1], &[0]).unwrap_err();
        assert!(e.contains("from 1 to 3 cycles"), "{e}");
    }
    assert_eq!(pass.slots[0].position, 19);
}

#[test]
fn the_rows_of_the_logits_are_any_set_inside_a_span_or_none() {
    let Some(dir) = model_dir("cuda_pass_slots") else {
        return;
    };
    let dev = common::gpu::open();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let chat = chat_ids();
    let ids = ids_of(&chat, 0, 0, 17);
    let mut pass = open_slots(&dev, &weights, &[SEQ], CacheFormat::F32);
    let all: Vec<usize> = (1..17).collect();
    let full = pass.run_slot(0, 3, &ids, &all).unwrap();
    assert_eq!(full.len(), 16 * LOGITS);
    let snap = snapshot(&pass, 0, 48);
    for set in [vec![], vec![16], vec![1, 4, 16], vec![0, 15], vec![5], vec![2, 3]] {
        pass.slots[0].clear().unwrap();
        let live = dev.live_buffers();
        let got = pass.run_slot(0, 3, &ids, &set).unwrap();
        assert_eq!(dev.live_buffers(), live);
        assert_eq!(got.len(), set.len() * LOGITS, "{set:?}");
        for (i, row) in set.iter().enumerate() {
            // Row 0 is outside the full call; a later check compares it.
            if *row == 0 {
                continue;
            }
            let reference = full[(row - 1) * LOGITS..row * LOGITS].to_vec();
            assert_eq!(
                bits(&got[i * LOGITS..(i + 1) * LOGITS]),
                bits(&reference),
                "set {set:?} row {row}"
            );
        }
        // The planes and the conv states do not depend on the logits.
        assert_eq!(snapshot(&pass, 0, 48), snap, "set {set:?}: the slot differs");
        assert_eq!(pass.slots[0].position, 17);
    }
    // Row 0 and row 15 in one span of 16 rows: compare with the single-row calls at the same position.
    pass.slots[0].clear().unwrap();
    let two = pass.run_slot(0, 3, &ids, &[0, 15]).unwrap();
    for (i, row) in [0usize, 15].into_iter().enumerate() {
        pass.slots[0].clear().unwrap();
        let one = pass.run_slot(0, 3, &ids, &[row]).unwrap();
        assert_eq!(bits(&two[i * LOGITS..(i + 1) * LOGITS]), bits(&one), "row {row}");
    }
    // The refusals change nothing: the position, the slot and the count of buffers.
    pass.slots[0].clear().unwrap();
    let live = dev.live_buffers();
    let before = snapshot(&pass, 0, 48);
    for (set, text) in [
        (vec![0, 16], "lie in 17 rows"),
        (vec![17], "has no row 17"),
        (vec![3, 3], "rise one by one"),
        (vec![4, 2], "rise one by one"),
    ] {
        let e = pass.run_slot(0, 3, &ids, &set).unwrap_err();
        assert!(e.contains(text), "{set:?}: {e}");
    }
    assert!(pass.run_slot(0, 3, &ids, &(0..17).collect::<Vec<_>>()).is_err());
    assert!(pass.run_slot(2, 3, &ids, &[0]).unwrap_err().contains("names slot 2"));
    assert_eq!((pass.slots[0].position, dev.live_buffers()), (0, live));
    assert_eq!(snapshot(&pass, 0, 48), before);
}

#[test]
fn a_clear_then_a_pass_give_the_bits_of_a_fresh_slot() {
    let Some(dir) = model_dir("cuda_pass_slots") else {
        return;
    };
    let dev = common::gpu::open();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let chat = chat_ids();
    let first: Vec<Vec<u32>> = (0..4).map(|i| ids_of(&chat, 2, i * 17, 17)).collect();
    let second = vec![
        ids_of(&chat, 0, 0, 17),
        ids_of(&chat, 0, 17, 1),
        ids_of(&chat, 0, 18, 1),
    ];
    let (want, _) = lone(&dev, &weights, 3, CacheFormat::F32, &second);
    let mut pass = open_slots(&dev, &weights, &[SEQ], CacheFormat::F32);
    // The first run fills 68 positions of the planes with other values. The clear keeps them.
    for ids in &first {
        pass.run_slot(0, 3, ids, &[16]).unwrap();
    }
    let stale = snapshot(&pass, 0, 48);
    pass.slots[0].clear().unwrap();
    assert_eq!(snapshot(&pass, 0, 48).len(), stale.len());
    assert_ne!(
        snapshot(&pass, 0, 48),
        stale,
        "a clear that changed nothing cannot show the gate"
    );
    let mut got = Vec::new();
    for ids in &second {
        got = pass.run_slot(0, 3, ids, &[ids.len() - 1]).unwrap();
    }
    assert_eq!(
        bits(&got),
        bits(&want),
        "a pass after a clear differs from a pass in a fresh slot"
    );
    // A clear zeroes the poison.
    pass.slots[0].poisoned = true;
    assert!(pass.run_slot(0, 3, &second[1], &[0]).unwrap_err().contains("poisoned"));
    pass.slots[0].clear().unwrap();
    assert!(pass.run_slot(0, 3, &second[0], &[0]).is_ok());
}

/// The planes and the conv states of the first `states` states of slot `slot`, as the parts of a snapshot: the key and the
/// value rows of the positions below `n`, and the conv. A sliding state gives its rows below `n`, which fit the ring here.
fn parts(pass: &Pass<'_>, slot: usize, states: usize, n: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for s in 0..states {
        out.push(pass.slots[slot].read_cells(s, false, 0, n).unwrap());
        out.push(pass.slots[slot].read_cells(s, true, 0, n).unwrap());
        out.push(pass.slots[slot].read_conv(s).unwrap());
    }
    out
}

#[test]
fn a_snapshot_restored_into_another_slot_gives_the_bits_of_the_run_with_no_snapshot() {
    let Some(dir) = model_dir("cuda_pass_slots") else {
        return;
    };
    let chat = chat_ids();
    for format in [CacheFormat::F32, CacheFormat::Bf16, CacheFormat::E4m3] {
        let dev = common::gpu::open_cache(format);
        let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
        let mut pass = open_slots(&dev, &weights, &[SEQ, SEQ], format);
        let prompt = ids_of(&chat, 0, 0, 17);
        pass.run_slot(0, 3, &prompt, &[16]).unwrap();
        // Dirty slot 1 first, then clear it: a restore must not need a clean plane.
        pass.run_slot(1, 3, &ids_of(&chat, 3, 0, 33), &[32]).unwrap();
        pass.slots[1].clear().unwrap();
        let live = dev.live_buffers();
        for s in 0..48 {
            let k = pass.slots[0].read_cells(s, false, 0, 17).unwrap();
            let v = pass.slots[0].read_cells(s, true, 0, 17).unwrap();
            let conv = pass.slots[0].read_conv(s).unwrap();
            pass.slots[1].write_cells(s, false, 0, 17, &k).unwrap();
            pass.slots[1].write_cells(s, true, 0, 17, &v).unwrap();
            pass.slots[1].write_conv(s, &conv).unwrap();
        }
        pass.slots[1].set_position(17).unwrap();
        assert_eq!(dev.live_buffers(), live, "a part call allocated on the device");
        assert_eq!(
            parts(&pass, 1, 48, 17),
            parts(&pass, 0, 48, 17),
            "{format:?}: the parts differ after a restore"
        );
        let step = ids_of(&chat, 0, 17, 1);
        let want = pass.run_slot(0, 3, &step, &[0]).unwrap();
        let got = pass.run_slot(1, 3, &step, &[0]).unwrap();
        assert_eq!(
            bits(&got),
            bits(&want),
            "{format:?}: a pass after a restore differs from the run with no snapshot"
        );
        assert_eq!(
            parts(&pass, 1, 48, 18),
            parts(&pass, 0, 48, 18),
            "{format:?}: the state differs after the pass"
        );
        eprintln!("{format:?}: a restore of 48 states and one pass give the bits of the run with no snapshot");
    }
}
