//! The lab files of a profile (`docs/lab.md`, "The rules of the J-Lens"): the model folder of each LFM3 engine, and
//! the J-Lens that its `[engines.lab]` table names.

use std::collections::BTreeMap;
use std::path::PathBuf;

use nagaya::lfm3::KIND;
use nagaya::profile::PlacementProfile;
use nagaya::serving::lab_files;

fn profile(entries: &[(&str, &str, &str)]) -> (tempfile::TempDir, PlacementProfile) {
    let dir = tempfile::tempdir().unwrap();
    let mut text = String::new();
    for (name, kind, lab) in entries {
        text.push_str(&format!(
            "[[engines]]\nname = \"{name}\"\nkind = \"{kind}\"\ndevices = [0]\nshare = 1\n\n[engines.config]\n\
             model_dir = \"{}\"\n\n{lab}\n",
            dir.path().display()
        ));
    }
    text.push_str("[[budgets]]\ntiers = [\"device(0)\"]\nbytes = 1000\n");
    (dir, PlacementProfile::parse(&text).unwrap())
}

#[test]
fn an_lfm3_engine_gives_its_folder_and_the_lens_of_its_lab_table() {
    let (dir, profile) = profile(&[
        ("lfm3", KIND, "[engines.lab]\njlens = \"/lenses/lfm3.safetensors\""),
        ("lfm3-omni", KIND, ""),
        ("other", "fake", ""),
    ]);
    let (folders, lenses) = lab_files(&profile).unwrap();
    assert_eq!(
        folders,
        BTreeMap::from([
            ("lfm3".to_owned(), dir.path().to_path_buf()),
            ("lfm3-omni".to_owned(), dir.path().to_path_buf()),
        ])
    );
    assert_eq!(
        lenses,
        BTreeMap::from([("lfm3".to_owned(), PathBuf::from("/lenses/lfm3.safetensors"))])
    );
}

#[test]
fn a_lab_table_on_another_kind_or_with_another_key_stops_the_start_with_the_reason() {
    for (kind, lab, needle) in [
        ("fake", "[engines.lab]\njlens = \"/lenses/a.safetensors\"", "fake"),
        (KIND, "[engines.lab]\ntuned = \"/lenses/a.safetensors\"", "tuned"),
        (KIND, "[engines.lab]\njlens = 3", "jlens"),
    ] {
        let (_dir, profile) = profile(&[("one", kind, lab)]);
        let error = lab_files(&profile).unwrap_err();
        assert!(error.contains(needle) && error.contains("one"), "{lab}: {error}");
    }
}
