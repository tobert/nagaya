//! The canary records of the CPU engine against the engine on the real checkpoints (spec 8.1). A missing or other
//! record fails, and the message gives the line for `src/canary.rs`.

#[path = "../../nagaya-family-beatthis/tests/common/mod.rs"]
mod common;

use nagaya_family_beatthis::canary;
use nagaya_mk_beatthis_cpu::canary::{TABLE, TARGET};
use nagaya_mk_beatthis_cpu::engine::BeatThisEngine;

#[test]
fn each_checkpoint_has_its_record() {
    let Some(_) = common::weights("final0") else { return };
    let dir = std::path::PathBuf::from(std::env::var_os(common::MODEL_VARIABLE).unwrap());
    let mut lines = Vec::new();
    for name in ["final0", "small0"] {
        let mut engine = BeatThisEngine::open_unchecked(&dir.join(format!("{name}.safetensors")), 4).unwrap();
        let measured = engine.measure_canary().unwrap();
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

/// The same tensors with one more metadata key: other file bytes, so another digest with no record. The open must
/// refuse it as a canary failure, whatever the table holds.
#[test]
fn an_open_of_a_checkpoint_with_no_record_is_refused_as_a_canary() {
    let Some(bytes) = common::weights("small0") else { return };
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    let mut md = meta.metadata().clone().unwrap();
    md.insert("canary_test".into(), "a file with no record".into());
    let tensors: Vec<(String, safetensors::tensor::TensorView)> = st.tensors();
    let out = safetensors::serialize(tensors, Some(md)).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("small0-other.safetensors");
    std::fs::write(&path, out).unwrap();
    let e = BeatThisEngine::open(&path, 2)
        .err()
        .expect("a checkpoint with no record opened");
    assert!(
        matches!(
            e,
            nagaya_mk_beatthis_cpu::engine::OpenFailure::Canary(canary::Refusal::NoRecord { .. })
        ),
        "{e}"
    );
}

/// One record serves each machine, because the canary gives the same hash at each instruction set (spec 3.2c). A level
/// that the CPU lacks is skipped by name.
#[test]
fn the_canary_is_the_same_at_each_instruction_set() {
    use nagaya_mk_beatthis_cpu::isa::{self, Level};
    let Some(_) = common::weights("small0") else { return };
    let dir = std::path::PathBuf::from(std::env::var_os(common::MODEL_VARIABLE).unwrap());
    let mut engine = BeatThisEngine::open_unchecked(&dir.join("small0.safetensors"), 4).unwrap();
    let mut hashes = Vec::new();
    for level in Level::ALL {
        if level > isa::detected() {
            eprintln!(
                "skipped: the canary at {}: the CPU does not have {}",
                level.name(),
                level.name()
            );
            continue;
        }
        hashes.push((level, isa::with_level(level, || engine.measure_canary().unwrap())));
    }
    let want = canary::record(TABLE, engine.checkpoint_digest()).expect("small0 has a record");
    for (level, h) in &hashes {
        assert_eq!(h, want, "the canary at {}", level.name());
    }
}
