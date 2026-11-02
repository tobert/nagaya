//! Measure the engine on gfx1151host: the open, and the time and the peak memory of each phase of a job at each size. The
//! peak is the GPU's VRAM and GTT use from sysfs, sampled every 10 ms. Stop `nagaya.service` first (spec 3.1).
//!
//! Run: `QWENIMAGE21_MODEL_DIR=... QWENIMAGE21_ENCODER=... QWENIMAGE21_LORA=... cargo run --release -p
//! nagaya-mk-qwenimage21-gfx1151 --features hip --example measure -- 1024x1024 1440x1440`.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use nagaya_abi::{Engine, ImageJob, ImagePrompt, ImageSampler, ImageSpacing};
use nagaya_mk_qwenimage21_gfx1151::engine::{Config, QwenImageEngine};

const SYSFS: &str = "/sys/class/drm/card1/device";

fn used() -> u64 {
    ["mem_info_vram_used", "mem_info_gtt_used"]
        .iter()
        .map(|name| {
            std::fs::read_to_string(format!("{SYSFS}/{name}"))
                .unwrap_or_else(|e| panic!("read {SYSFS}/{name}: {e}"))
                .trim()
                .parse::<u64>()
                .unwrap()
        })
        .sum()
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn main() {
    let env = |name: &str| PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")));
    let config = Config {
        model: env("QWENIMAGE21_MODEL_DIR"),
        encoder: env("QWENIMAGE21_ENCODER"),
        lora: Some((env("QWENIMAGE21_LORA"), 1.0)),
        plan_dir: std::env::var_os("QWENIMAGE21_PLAN_DIR").map(PathBuf::from),
        flash: std::env::var_os("QWENIMAGE21_FLASH").is_some(),
    };
    let peak = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let (peak, stop) = (Arc::clone(&peak), Arc::clone(&stop));
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                peak.fetch_max(used(), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(10));
            }
        })
    };
    let before = used();
    println!("before the open: {:.2} GiB used", gib(before));
    let started = Instant::now();
    let mut engine = QwenImageEngine::open(&config).unwrap();
    let loaded = used();
    println!(
        "open: {:.1} s, {:.2} GiB more after the open",
        started.elapsed().as_secs_f64(),
        gib(loaded - before)
    );
    // The ids of a prompt of 30 words: the template of the family is 14 system ids, then the prompt.
    let ids: Vec<u32> = (0..14).chain((0..40).map(|i| 1000 + i)).chain(0..5).collect();
    for size in std::env::args().skip(1) {
        let (w, h) = size.split_once('x').expect("a size is WxH");
        let (w, h): (u32, u32) = (w.parse().unwrap(), h.parse().unwrap());
        for round in 0..2 {
            let job = ImageJob {
                prompt: ImagePrompt::Tokens {
                    ids: ids.clone(),
                    cut: 14,
                },
                width: w,
                height: h,
                steps: 4,
                cfg: 1.0,
                sampler: ImageSampler::FlowEuler,
                spacing: ImageSpacing::Leading,
                count: 1,
                seed: 7,
                previews: false,
                loras: Vec::new(),
                conditioning: None,
                init: None,
            };
            peak.store(used(), Ordering::Relaxed);
            let image = engine.image().unwrap();
            let t = Instant::now();
            image.image_begin(0, &job).unwrap();
            let begin = t.elapsed().as_secs_f64();
            let mut step_times = Vec::new();
            for _ in 0..4 {
                let t = Instant::now();
                image.image_step(0).unwrap();
                step_times.push(t.elapsed().as_secs_f64());
            }
            let t = Instant::now();
            let images = image.image_decode(0).unwrap();
            let decode = t.elapsed().as_secs_f64();
            image.image_clear(0);
            assert_eq!(images.len(), 1);
            let top = peak.load(Ordering::Relaxed);
            println!(
                "{w}x{h} round {round}: begin {begin:.2} s, steps {:?} s, decode {decode:.2} s; peak {:.2} GiB, {:.2} GiB \
                 over the loaded engine",
                step_times.iter().map(|s| format!("{s:.2}")).collect::<Vec<_>>(),
                gib(top),
                gib(top.saturating_sub(loaded))
            );
        }
    }
    stop.store(true, Ordering::Relaxed);
    sampler.join().unwrap();
}
