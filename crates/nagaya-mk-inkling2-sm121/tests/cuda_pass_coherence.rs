//! GPU gate of the whole pass by coherence. The card does not reproduce the CPU
//! reference bit for bit: the CPU reference rounds the weights of the NVFP4 experts to bf16, the card does not, and a
//! quantizer can flip a code. So the gate is the coherence of the logits with the golden, and it prints the gaps that
//! the binding writes as thresholds. It asserts only what is clear: the top token and the greedy tokens.
//!
//! For each prompt of the golden (`inkling2-mini.quant.oracle.json`, 3 cycles; `quant.cycles.json`, 1 to 5 cycles) the
//! test prefills the ids in blocks of up to 512 rows, and reads the logits of the last row. The gap is the worst
//! `|d|` over the 64 `top_ids` and `|d|` of the logsumexp. The decode file (`quant.decode`) gives a prompt, 8 greedy
//! tokens and their logits: the test prefills the prompt and runs 7 decode steps with the tokens of the golden.
//! The logits of the last row of the tape are a second realization of the CPU reference, and the test prints the gap
//! to it too.
//!
//! `INKLING2_MODEL_DIR=<folder> INKLING2_GOLDEN_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p
//!  nagaya-mk-inkling2-sm121 --features cuda --test cuda_pass_coherence -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;
mod pass_support;

use std::path::PathBuf;
use std::time::Instant;

use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_inkling2_sm121::pass::step::Pass;
use nagaya_mk_inkling2_sm121::pass::weights::DeviceWeights;
use nagaya_mk_inkling2_sm121::spec::{self, LOGITS};
use pass_support::tape;

const MAX_ROWS: usize = 512;
/// The bounds of the gaps: twice the worst gap that the measurement gave (3.0 over the 64 top ids, 2.3 for the logsumexp,
/// both at 4 and 5 cycles). The gap of the two realizations of the CPU reference (the golden and the tape) is 1.9.
const TOP_GAP_BOUND: f64 = 6.0;
const LSE_GAP_BOUND: f64 = 4.6;
/// A greedy token of the card can differ from the golden only where the golden has a near tie: its top-2 margin is under
/// this value (measured: one step in 8, margin 0.095).
const TIE_MARGIN: f32 = 0.5;
const SEQ: usize = 1024;

fn golden_dir() -> PathBuf {
    std::env::var_os("INKLING2_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/inkling2-golden")))
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::MIN), |b, (i, x)| if *x > b.1 { (i, *x) } else { b })
        .0
}

fn logsumexp(v: &[f32]) -> f64 {
    let top = f64::from(v[argmax(v)]);
    top + v.iter().map(|x| (f64::from(*x) - top).exp()).sum::<f64>().ln()
}

/// The watchdog of a pass: a pass that hangs ends the process, so that the GPU is free.
fn with_watchdog<T>(seconds: u64, body: impl FnOnce() -> T) -> T {
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for _ in 0..seconds * 10 {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if done.load(std::sync::atomic::Ordering::SeqCst) {
                    return;
                }
            }
            eprintln!("cuda_pass_coherence: the pass did not finish within {seconds} s; it hangs");
            std::process::exit(2);
        });
        // The guard ends the watchdog when the body returns or panics.
        struct Done<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let _guard = Done(&done);
        body()
    })
}

/// Prefill `ids` in blocks and return the logits of the last row.
fn prefill(pass: &mut Pass<'_>, ids: &[u32]) -> Vec<f32> {
    pass.slots[0].zero().unwrap();
    let mut out = Vec::new();
    for block in ids.chunks(MAX_ROWS) {
        out = pass.run(block, 1).unwrap_or_else(|e| panic!("{e}"));
    }
    out
}

struct Golden {
    name: String,
    ids: Vec<u32>,
    top_ids: Vec<usize>,
    top_logits: Vec<f64>,
    logsumexp: f64,
    cycles: usize,
}

fn read_prompts(file: &str, default_cycles: usize) -> Vec<Golden> {
    let path = golden_dir().join(file);
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap();
    json["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| Golden {
            name: p["name"].as_str().unwrap().to_owned(),
            ids: p["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect(),
            top_ids: p["top_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect(),
            top_logits: p["top_logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect(),
            logsumexp: p["logsumexp"].as_f64().unwrap(),
            cycles: p
                .get("cycles")
                .and_then(|c| c.as_u64())
                .map_or(default_cycles, |c| c as usize),
        })
        .collect()
}

/// The gap of the logits of a row to a golden: the worst `|d|` over the top ids, the `|d|` of the logsumexp, and the
/// margin between the first and the second token of the golden.
fn gap(logits: &[f32], g: &Golden) -> (f64, f64, f64) {
    let worst = g
        .top_ids
        .iter()
        .zip(&g.top_logits)
        .map(|(i, want)| (f64::from(logits[*i]) - want).abs())
        .fold(0.0, f64::max);
    (
        worst,
        (logsumexp(logits) - g.logsumexp).abs(),
        g.top_logits[0] - g.top_logits[1],
    )
}

fn open(format: CacheFormat) -> Option<(nagaya_mk_cuda::driver::CudaDevice, std::path::PathBuf)> {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    let dir = match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(d) if PathBuf::from(&d).join(spec::WEIGHTS_FILE).exists() => PathBuf::from(d),
        _ => {
            assert!(
                !require,
                "INKLING2_MODEL_DIR has no checkpoint and NAGAYA_REQUIRE_MODELS is set"
            );
            eprintln!("cuda_pass_coherence: skipped, INKLING2_MODEL_DIR has no checkpoint");
            return None;
        }
    };
    Some((common::gpu::open_cache(format), dir))
}

/// The coherence of the prefill logits with the golden, for a cache in `format`. Each gap is printed. The test asserts
/// the bounds after the last prompt, so that one run reports every prompt.
fn prefill_coherence(format: CacheFormat) {
    let Some((dev, dir)) = open(format) else { return };
    let Some(mut tape) = tape("cuda_pass_coherence") else {
        return;
    };
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let mut worst_gap = 0.0f64;
    let mut wrong_top = 0usize;
    let mut broken: Vec<String> = Vec::new();
    let mut cases = read_prompts("inkling2-mini.quant.oracle.json", 3);
    cases.extend(read_prompts("inkling2-mini.quant.cycles.json", 3));
    let tape_last = tape.f32s("logits_last");
    let tape_ids = tape.ids();
    for g in &cases {
        let started = Instant::now();
        let mut pass =
            Pass::with_cache_format(&dev, &weights, SEQ, MAX_ROWS, g.cycles, format).unwrap_or_else(|e| panic!("{e}"));
        let logits = with_watchdog(900, || prefill(&mut pass, &g.ids));
        assert_eq!(logits.len(), LOGITS);
        let top = argmax(&logits);
        let (d_top, d_lse, margin) = gap(&logits, g);
        let tape_note = if g.ids == tape_ids && g.cycles == 3 {
            let d = logits
                .iter()
                .zip(&tape_last)
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                .fold(0.0, f64::max);
            format!(
                " | against the logits_last of the tape: max |d| {d:.3e}, top {}",
                argmax(&tape_last)
            )
        } else {
            String::new()
        };
        eprintln!(
            "{:>14} cycles {} ({:>3} ids): top {top} (golden {}) margin {margin:.3} | max |d| over the 64 top ids {d_top:.3e}, |d| of logsumexp {d_lse:.3e}{tape_note} | {:.1} s",
            g.name,
            g.cycles,
            g.ids.len(),
            g.top_ids[0],
            started.elapsed().as_secs_f64()
        );
        if d_top >= TOP_GAP_BOUND {
            broken.push(format!(
                "{}: the logits of the top ids differ from the golden by {d_top:.3}",
                g.name
            ));
        }
        if d_lse >= LSE_GAP_BOUND {
            broken.push(format!(
                "{}: the logsumexp differs from the golden by {d_lse:.3}",
                g.name
            ));
        }
        worst_gap = worst_gap.max(d_top);
        wrong_top += usize::from(top != g.top_ids[0]);
        if !logits.iter().all(|v| v.is_finite()) {
            broken.push(format!("{}: a logit is not finite", g.name));
        }
    }
    eprintln!(
        "{format:?}: worst gap over the 64 top ids: {worst_gap:.3e}; prompts with the wrong top token: {wrong_top} of {}",
        cases.len()
    );
    assert!(broken.is_empty(), "{format:?}: {}", broken.join("; "));
    assert_eq!(
        wrong_top, 0,
        "{format:?}: the top token differs from the golden on a prompt"
    );
}

#[test]
fn the_logits_of_the_pass_are_coherent_with_the_golden() {
    prefill_coherence(CacheFormat::F32);
}

#[test]
fn the_logits_of_an_e4m3_cache_are_coherent_with_the_golden() {
    prefill_coherence(CacheFormat::E4m3);
}

#[test]
fn the_logits_of_a_bf16_cache_are_coherent_with_the_golden() {
    prefill_coherence(CacheFormat::Bf16);
}

fn decode_coherence(format: CacheFormat) {
    let Some((dev, dir)) = open(format) else { return };
    let path = golden_dir().join("inkling2-mini.quant.decode.safetensors");
    if !path.exists() {
        assert!(
            std::env::var_os("NAGAYA_REQUIRE_MODELS").is_none(),
            "{} is absent",
            path.display()
        );
        eprintln!("cuda_pass_coherence: skipped, {} is absent", path.display());
        return;
    }
    let header = nagaya_mk_cuda::safetensors::SafeTensorsHeader::read(&path).unwrap();
    let mut file = std::fs::File::open(&path).unwrap();
    let ids: Vec<u32> = header
        .read_tensor(&mut file, "ids")
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| i64::from_le_bytes(*b) as u32)
        .collect();
    let tokens: Vec<u32> = header
        .read_tensor(&mut file, "tokens")
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| i64::from_le_bytes(*b) as u32)
        .collect();
    let golden: Vec<f32> = header
        .read_tensor(&mut file, "logits")
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let mut pass = Pass::with_cache_format(&dev, &weights, SEQ, MAX_ROWS, 3, format).unwrap_or_else(|e| panic!("{e}"));
    let mut worst = 0.0f64;
    let mut mismatches = 0usize;
    let mut broken: Vec<String> = Vec::new();
    let mut logits = with_watchdog(600, || prefill(&mut pass, &ids));
    for step in 0..tokens.len() {
        let want = &golden[step * LOGITS..(step + 1) * LOGITS];
        let d = logits
            .iter()
            .zip(want)
            .map(|(a, b)| f64::from(*a - *b).abs())
            .fold(0.0, f64::max);
        let (got, expect) = (argmax(&logits) as u32, tokens[step]);
        let margin = {
            let mut top = want.to_vec();
            top.sort_by(|a, b| b.partial_cmp(a).unwrap());
            top[0] - top[1]
        };
        eprintln!(
            "decode step {step}: token {got} (golden {expect}), max |d| over all logits {d:.3e}, golden top-2 margin {:.3}",
            {
                let mut top = want.to_vec();
                top.sort_by(|a, b| b.partial_cmp(a).unwrap());
                top[0] - top[1]
            }
        );
        worst = worst.max(d);
        if got != expect {
            if margin >= TIE_MARGIN {
                broken.push(format!(
                    "step {step}: token {got}, golden {expect}, and the golden margin is {margin}, not a near tie"
                ));
            }
            mismatches += 1;
        }
        if step + 1 < tokens.len() {
            logits = with_watchdog(600, || pass.run(&[tokens[step]], 1)).unwrap_or_else(|e| panic!("{e}"));
        }
    }
    eprintln!(
        "{format:?}: worst max |d| over all logits: {worst:.3e}; steps with another token: {mismatches} of {}",
        tokens.len()
    );
    assert!(broken.is_empty(), "{format:?}: {}", broken.join("; "));
    assert!(
        mismatches <= 1,
        "{format:?}: {mismatches} greedy tokens differ from the golden"
    );
}

#[test]
fn a_prefill_and_seven_decode_steps_give_the_greedy_tokens_of_the_golden() {
    decode_coherence(CacheFormat::F32);
}

#[test]
fn an_e4m3_cache_gives_the_greedy_tokens_of_the_golden() {
    decode_coherence(CacheFormat::E4m3);
}

#[test]
fn a_bf16_cache_gives_the_greedy_tokens_of_the_golden() {
    decode_coherence(CacheFormat::Bf16);
}
