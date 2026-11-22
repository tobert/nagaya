//! Integration test planning a 512K slot against the real Inkling 2 checkpoint when `INKLING2_MODEL_DIR` is set.

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
    let bytes = |name: &str| lim.memory.mandatory.iter().find(|t| t.name == name).expect(name).bytes;
    // The slot is exact for any checkpoint of the two models: 8 full and 40 sliding states at 3 cycles.
    assert_eq!(bytes("kv"), 35_370_205_184);
    assert_eq!(lim.slots[0].bytes, 35_370_205_184);
    // The sums of the header of Inkling 2 Mini, each tensor rounded up to 64 bytes. The research model differs.
    if spec::Model::read(&cfg.model_dir).unwrap() == spec::Model::Mini {
        assert_eq!(bytes("weights"), 18_889_360_896);
        assert_eq!(bytes("vision"), 636_512_256);
    }
}
