//! The canary records of the gfx1151 engine against the engine on the real checkpoints (spec 8.1). A missing or other
//! record fails, and the message gives the line for `src/canary.rs`.
//!
//! A card test. Stop `nagaya.service` first (spec 3.1):
//! `cargo test -p nagaya-mk-beatthis-gfx1151 --features hip --test hip_canary -- --test-threads=1`.
#![cfg(feature = "hip")]

#[path = "../../nagaya-family-beatthis/tests/common/mod.rs"]
mod common;

use nagaya_family_beatthis::canary;
use nagaya_mk_beatthis_gfx1151::canary::{TABLE, TARGET};
use nagaya_mk_beatthis_gfx1151::engine::BeatThisGpuEngine;

#[test]
fn each_checkpoint_has_its_record() {
    let Some(_) = common::weights("final0") else { return };
    let dir = std::path::PathBuf::from(std::env::var_os(common::MODEL_VARIABLE).unwrap());
    let mut lines = Vec::new();
    for name in ["final0", "small0"] {
        let mut engine = BeatThisGpuEngine::open_unchecked(&dir.join(format!("{name}.safetensors"))).unwrap();
        let measured = engine.measure_canary().unwrap();
        let again = engine.measure_canary().unwrap();
        assert_eq!(measured, again, "{name}: two canary runs gave other bits");
        let digest = engine.checkpoint_digest().to_owned();
        if let Err(e) = canary::verify(TABLE, TARGET, &digest, &measured) {
            lines.push(format!("{name}: {e}\n    (\"{digest}\", \"{measured}\"),"));
        }
    }
    assert!(
        lines.is_empty(),
        "records to add or fix in src/canary.rs:\n{}",
        lines.join("\n")
    );
}
