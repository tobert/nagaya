//! Integration test planning a 512K slot against the real Inkling 2 Mini checkpoint when `INKLING2_MODEL_DIR` is set.

use std::path::PathBuf;

use nagaya_mk_inkling2_sm121::spec::{self, Config, DEFAULT_EXTRA_RUNS, DEFAULT_MAX_BLOCK};

const VAR: &str = "INKLING2_MODEL_DIR";

fn model_dir_or_skip() -> Option<PathBuf> {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    match std::env::var_os(VAR) {
        None => {
            if require {
                panic!("{VAR} is not set and NAGAYA_REQUIRE_MODELS is set");
            }
            eprintln!("skipping spec_model: {VAR} is not set");
            None
        }
        Some(path) => {
            let dir = PathBuf::from(path);
            if !dir.exists() {
                if require {
                    panic!("{VAR} points to absent path `{}`", dir.display());
                }
                eprintln!("skipping spec_model: `{}` is absent", dir.display());
                return None;
            }
            Some(dir)
        }
    }
}

#[test]
fn real_checkpoint_plans_512k_slot_without_reading_weights() {
    let Some(dir) = model_dir_or_skip() else {
        return;
    };
    let cfg = Config {
        model_dir: dir,
        slots: vec![524_288],
        max_block: DEFAULT_MAX_BLOCK,
        extra_runs: DEFAULT_EXTRA_RUNS,
    };
    let lim = spec::plan(&cfg).unwrap();
    lim.check().unwrap();
    let weights = lim
        .memory
        .mandatory
        .iter()
        .find(|t| t.name == "weights")
        .expect("weights term")
        .bytes;
    let kv = lim
        .memory
        .mandatory
        .iter()
        .find(|t| t.name == "kv")
        .expect("kv term")
        .bytes;
    // Inkling 2 Mini packed LLM weights ~ 18-20 GB; 512K slot at 3 cycles ~ 35.4 GB.
    assert!(
        (17_000_000_000..=21_000_000_000).contains(&weights),
        "weights={weights}"
    );
    assert!((34_000_000_000..=37_000_000_000).contains(&kv), "kv={kv}");
}
