//! Times Beat This! on the card: one chunk of 500 frames (10 s) and of 1,500 frames (30 s) for each checkpoint, the
//! median of 9 runs after 2 warm runs, and the share of each task kind from a run with a wait after each task.
//!
//! A card program. Stop `nagaya.service` first (spec 3.1):
//! `BEATTHIS_MODEL_DIR=<dir of final0.safetensors> cargo run --release -p nagaya-mk-beatthis-gfx1151 --features hip --example bench`

use std::collections::BTreeMap;
use std::time::Instant;

use nagaya_family_beatthis::plan::{self, Config, Task};
use nagaya_family_beatthis::weights::{self, Tensors};
use nagaya_mk_beatthis_gfx1151::gpu::Gpu;

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

fn main() {
    let dir = std::path::PathBuf::from(std::env::var_os("BEATTHIS_MODEL_DIR").expect("set BEATTHIS_MODEL_DIR"));
    for name in ["small0", "final0"] {
        let bytes = std::fs::read(dir.join(format!("{name}.safetensors"))).expect("read the weights");
        let config = Config::from_json(&Tensors::hyper_parameters(&bytes).unwrap()).unwrap();
        let t = Tensors::parse(&bytes).unwrap();
        let prepared = weights::prepare(&plan::build(&config, 1500).weights, &t).unwrap();
        let gpu = Gpu::new(&prepared).unwrap();
        for frames in [500usize, 1500] {
            let p = plan::build(&config, frames);
            let input: Vec<f32> = (0..frames * 128).map(|i| ((i * 7919) % 1000) as f32 / 500.0).collect();
            for _ in 0..2 {
                gpu.run(&p, &input, false, |_, _| {}).unwrap();
            }
            let mut ms: Vec<f64> = (0..9)
                .map(|_| {
                    let s = Instant::now();
                    gpu.run(&p, &input, false, |_, _| {}).unwrap();
                    s.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            ms.sort_by(f64::total_cmp);
            let mut times = Vec::new();
            gpu.run_timed(&p, &input, false, |_, _| {}, Some(&mut times)).unwrap();
            let mut by: BTreeMap<&str, f64> = BTreeMap::new();
            for (task, t) in p.tasks.iter().zip(&times) {
                *by.entry(kind(task)).or_default() += t;
            }
            let total: f64 = times.iter().sum();
            let shares: Vec<String> = by
                .iter()
                .map(|(k, v)| format!("{k} {:.0}%", 100.0 * v / total))
                .collect();
            println!(
                "{name} {frames:4} frames: median {:7.1} ms | {}",
                ms[4],
                shares.join(", ")
            );
        }
    }
}
