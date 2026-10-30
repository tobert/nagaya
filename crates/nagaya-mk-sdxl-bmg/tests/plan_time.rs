//! A probe of the time of `plan` on real checkpoints. Ignored: run it by hand with the model variables set.
//! `SDXL_MODEL_DIR=... SDXL_PLAN_FILES=a.safetensors:b.safetensors cargo test --release -p nagaya-mk-sdxl-bmg --test
//! plan_time -- --ignored --nocapture`.

use std::path::PathBuf;
use std::time::Instant;

use nagaya_mk_sdxl_bmg::{Config, plan};

#[test]
#[ignore]
fn the_time_of_a_plan() {
    let dir = PathBuf::from(std::env::var("SDXL_MODEL_DIR").expect("set SDXL_MODEL_DIR"));
    let mut configs = vec![Config::folder(dir.clone())];
    for file in std::env::var("SDXL_PLAN_FILES")
        .unwrap_or_default()
        .split(':')
        .filter(|f| !f.is_empty())
    {
        configs.push(Config {
            model_dir: dir.clone(),
            checkpoint: Some(PathBuf::from(file)),
        });
    }
    for config in &configs {
        for run in 0..2 {
            let at = Instant::now();
            plan(config).expect("the plan");
            eprintln!("{:?} run {run}: {} ms", config.checkpoint, at.elapsed().as_millis());
        }
    }
}
