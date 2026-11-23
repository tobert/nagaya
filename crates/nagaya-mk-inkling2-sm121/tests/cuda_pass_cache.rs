//! GPU gate: the pass that finds its ring in the cache gives the bits of a pass that builds the ring again
//! (milestone 37 step 10). A call of a shape that the pass has run loads the ring of that shape and patches only the
//! fields that move: the position of the first row, and the chunks of the attention. The gate runs decode steps at the
//! positions 17 to 40, a prefill that crosses the chunk boundary of 256 keys, and calls of other row counts between
//! decode steps. It compares the logits of each call and the whole slot after the last call, at the grids 48 and 7.
//!
//! `INKLING2_MODEL_DIR=<folder> INKLING2_GOLDEN_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p
//!  nagaya-mk-inkling2-sm121 --features cuda --test cuda_pass_cache -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;

use nagaya_mk_inkling2_sm121::pass::step::Pass;
use nagaya_mk_inkling2_sm121::pass::weights::DeviceWeights;
use nagaya_mk_inkling2_sm121::spec;

const SEQ: usize = 1024;

fn slot_hashes(pass: &Pass<'_>) -> Vec<[u8; 32]> {
    let mut out = Vec::new();
    for run in &pass.slot.runs {
        for buf in [&run.k_plane, &run.v_plane, &run.conv] {
            out.push(*blake3::hash(&buf.read(0, buf.bytes()).unwrap()).as_bytes());
        }
    }
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
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

/// The ids of a call of `rows` rows that starts at the position `at` of the sequence: the chat ids, cycled.
fn ids_of(chat: &[u32], at: usize, rows: usize) -> Vec<u32> {
    (at..at + rows).map(|i| chat[(i * 7 + 3) % chat.len()]).collect()
}

/// Run the calls `rows_list` in a zeroed slot. With `fresh` the pass forgets its rings before each call. The result is
/// the logits of each call and the hashes of the slot after the last.
fn run_calls(pass: &mut Pass<'_>, chat: &[u32], rows_list: &[usize], fresh: bool) -> (Vec<Vec<u32>>, Vec<[u8; 32]>) {
    pass.slot.zero().unwrap();
    pass.forget_rings();
    let mut at = 0;
    let mut logits = Vec::new();
    for &rows in rows_list {
        if fresh {
            pass.forget_rings();
        }
        let out = pass.run(&ids_of(chat, at, rows), 1).unwrap_or_else(|e| panic!("{e}"));
        assert!(out.iter().all(|v| v.is_finite()));
        logits.push(bits(&out));
        at += rows;
    }
    (logits, slot_hashes(pass))
}

fn compare(pass: &mut Pass<'_>, chat: &[u32], rows_list: &[usize], label: &str) {
    for grid in [48u32, 7] {
        pass.grid = grid;
        let started = std::time::Instant::now();
        let (fresh, fresh_slot) = run_calls(pass, chat, rows_list, true);
        let (cached, cached_slot) = run_calls(pass, chat, rows_list, false);
        eprintln!(
            "cuda_pass_cache: {label} at grid {grid}: {} calls, {:.1} s",
            rows_list.len(),
            started.elapsed().as_secs_f64()
        );
        for (i, (a, b)) in fresh.iter().zip(&cached).enumerate() {
            assert_eq!(
                a, b,
                "{label} at grid {grid}: call {i} of {} rows: the cached ring gives other bits",
                rows_list[i]
            );
        }
        assert!(
            fresh_slot == cached_slot,
            "{label} at grid {grid}: the slot after the cached calls differs from the slot after the fresh calls"
        );
    }
}

#[test]
fn a_cached_ring_gives_the_bits_of_a_fresh_ring() {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    let dir = match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(d) if PathBuf::from(&d).join(spec::WEIGHTS_FILE).exists() => PathBuf::from(d),
        _ => {
            assert!(
                !require,
                "INKLING2_MODEL_DIR has no checkpoint and NAGAYA_REQUIRE_MODELS is set"
            );
            eprintln!("cuda_pass_cache: skipped, INKLING2_MODEL_DIR has no checkpoint");
            return;
        }
    };
    let dev = common::gpu::open();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let mut pass = Pass::new(&dev, &weights, SEQ, 512, 3).unwrap_or_else(|e| panic!("{e}"));
    let chat = chat_ids();
    assert_eq!(chat.len(), 17);
    // The prompt, then decode steps at the positions 17 to 40.
    let mut decode = vec![17usize];
    decode.extend([1usize; 23]);
    compare(&mut pass, &chat, &decode, "decode steps at the positions 17 to 40");
    // Other row counts between decode steps: each switch of the shape loads a ring that an earlier call built.
    compare(
        &mut pass,
        &chat,
        &[17, 1, 1, 16, 1, 1, 9, 8, 1, 16, 1, 7, 1],
        "blocks of 17, 16, 9, 8 and 7 rows between decode steps",
    );
    // A prefill that ends before the chunk of 256 keys, and decode steps that cross it.
    let mut crossing = vec![250usize];
    crossing.extend([1usize; 14]);
    compare(
        &mut pass,
        &chat,
        &crossing,
        "decode steps across the chunk boundary of 256 keys",
    );
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// The host time and the wall time of a call, as the medians of `calls` calls of `rows` rows after a warm call.
fn time_calls(pass: &mut Pass<'_>, chat: &[u32], rows: usize, calls: usize) -> (f64, f64, f64) {
    pass.slot.zero().unwrap();
    pass.forget_rings();
    let mut at = 0;
    let (mut host, mut patch, mut wall) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..=calls {
        let ids = ids_of(chat, at, rows);
        let started = std::time::Instant::now();
        pass.run(&ids, 1).unwrap_or_else(|e| panic!("{e}"));
        let elapsed = started.elapsed();
        at += rows;
        // The first call builds the ring.
        if i > 0 {
            host.push(pass.host.call.as_secs_f64() * 1e3);
            patch.push(pass.host.patch.as_secs_f64() * 1e3);
            wall.push(elapsed.as_secs_f64() * 1e3);
        }
    }
    (median(host), median(patch), median(wall))
}

/// The host time of a call that finds its ring is under 1 ms, and the end to end rate follows the kernel.
#[test]
fn the_host_time_of_a_cached_call_is_reported_and_under_a_millisecond() {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    let dir = match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(d) if PathBuf::from(&d).join(spec::WEIGHTS_FILE).exists() => PathBuf::from(d),
        _ => {
            assert!(
                !require,
                "INKLING2_MODEL_DIR has no checkpoint and NAGAYA_REQUIRE_MODELS is set"
            );
            eprintln!("cuda_pass_cache: skipped, INKLING2_MODEL_DIR has no checkpoint");
            return;
        }
    };
    let dev = common::gpu::open();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let mut pass = Pass::new(&dev, &weights, SEQ, 512, 3).unwrap_or_else(|e| panic!("{e}"));
    let chat = chat_ids();
    // The build of a ring, for the report.
    pass.slot.zero().unwrap();
    pass.forget_rings();
    pass.run(&ids_of(&chat, 0, 1), 1).unwrap();
    eprintln!(
        "cuda_pass_cache: the first call of 1 row builds the ring: host {:.2} ms, of it the build {:.2} ms",
        pass.host.call.as_secs_f64() * 1e3,
        pass.host.build.as_secs_f64() * 1e3
    );
    let mut worst = 0f64;
    for rows in [1usize, 16] {
        let (host, patch, wall) = time_calls(&mut pass, &chat, rows, 20);
        let gpu = wall - host;
        eprintln!(
            "cuda_pass_cache: {rows} rows: host {host:.3} ms (patch {patch:.3} ms), wall {wall:.2} ms, kernel {gpu:.2} ms: \
             {:.1} tokens/s end to end, {:.1} tokens/s kernel only",
            rows as f64 * 1e3 / wall,
            rows as f64 * 1e3 / gpu
        );
        if rows == 1 {
            worst = host;
        }
    }
    assert!(
        worst < 1.0,
        "the host time of a decode step is {worst:.3} ms, over 1 ms"
    );
}
