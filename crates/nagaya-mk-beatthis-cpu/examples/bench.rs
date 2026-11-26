//! The time of one chunk on the CPU: `cargo run --release --example bench -- [--profile] [runs]`.
//!
//! It reads `small0` and `final0` from `BEATTHIS_MODEL_DIR`, and runs chunks of 500 frames (10 s) and 1500 frames
//! (30 s) on 8 and 16 threads. After one warm-up run, it gives the median and the least time of `runs` runs (default
//! 7). The machine can be shared, so the least time is the better measure of a change. The input is a fixed
//! pseudo-random log-mel; the time does not depend on the values. With `--profile` it also gives the median time of
//! each task kind for each configuration.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use nagaya_family_beatthis::chunks::CHUNK;
use nagaya_family_beatthis::mel::MELS;
use nagaya_family_beatthis::plan::{self, Config, Task};
use nagaya_family_beatthis::weights::{self, Tensors};
use nagaya_mk_beatthis_cpu as cpu;

fn kind(t: &Task) -> &'static str {
    match t {
        Task::Affine { .. } => "affine",
        Task::Conv { .. } => "conv",
        Task::Gelu { .. } => "gelu",
        Task::RmsNorm { .. } => "rms_norm",
        Task::Linear { .. } => "linear",
        Task::Attention { .. } => "attention",
        Task::AddInto { .. } => "add_into",
    }
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let profile = args.iter().any(|a| a == "--profile");
    let runs: usize = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map(|a| a.parse().expect("runs is not a number"))
        .unwrap_or(7);
    let dir = PathBuf::from(std::env::var_os("BEATTHIS_MODEL_DIR").expect("set BEATTHIS_MODEL_DIR"));
    println!("model  frames threads  median_ms  min_ms");
    for name in ["small0", "final0"] {
        let path = dir.join(format!("{name}.safetensors"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let config = Config::from_json(&Tensors::hyper_parameters(&bytes).unwrap()).unwrap();
        let tensors = Tensors::parse(&bytes).unwrap();
        let w = weights::prepare(&plan::build(&config, CHUNK).weights, &tensors).unwrap();
        for frames in [500, 1500] {
            let p = plan::build(&config, frames);
            let mut s = 12345u32;
            let input: Vec<f32> = (0..frames * MELS)
                .map(|_| {
                    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (s >> 8) as f32 / (1u32 << 24) as f32 * 8.0 - 4.0
                })
                .collect();
            for threads in [8, 16] {
                cpu::run(&p, &w, &input, threads, |_, _| {});
                let mut whole = Vec::new();
                let mut kinds: BTreeMap<&str, Vec<Duration>> = BTreeMap::new();
                for _ in 0..runs {
                    if profile {
                        let mut bufs: Vec<Vec<f32>> = p.buffers.iter().map(|&n| vec![0.0f32; n]).collect();
                        bufs[p.input.0].copy_from_slice(&input);
                        let mut per: BTreeMap<&str, Duration> = BTreeMap::new();
                        let start = Instant::now();
                        for t in &p.tasks {
                            let ts = Instant::now();
                            cpu::step(t, &mut bufs, &w, threads);
                            *per.entry(kind(t)).or_default() += ts.elapsed();
                        }
                        whole.push(start.elapsed());
                        for (k, d) in per {
                            kinds.entry(k).or_default().push(d);
                        }
                    } else {
                        let start = Instant::now();
                        std::hint::black_box(cpu::run(&p, &w, &input, threads, |_, _| {}));
                        whole.push(start.elapsed());
                    }
                }
                let min = whole.iter().min().copied().unwrap_or_default();
                println!(
                    "{name:6} {frames:6} {threads:7} {:10.1} {:7.1}",
                    median(whole).as_secs_f64() * 1e3,
                    min.as_secs_f64() * 1e3
                );
                for (k, d) in kinds {
                    println!("    {k:10} {:8.1} ms", median(d).as_secs_f64() * 1e3);
                }
            }
        }
    }
}
