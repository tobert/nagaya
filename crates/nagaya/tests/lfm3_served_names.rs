//! An LFM3 engine serves under its engine name (spec 11.11, "The served name of an LFM3 engine"). The start of
//! 12:59 on 2026-11-10 exited 3, because `lfm3` and `lfm3-1` of the B70 machine both served `lfm3`.
//!
//! It needs `LFM3_MODEL_DIR` and `LFM3_OMNI_MODEL_DIR`. An absent folder skips the test, and fails it with
//! `NAGAYA_REQUIRE_MODELS=1`. No card is needed: the test plans the family as the start does, from a fake identity.

use std::path::PathBuf;

use nagaya::process::OpenedEngine;
use nagaya::serving::FamilyRegistry;
use nagaya_abi::Engine;

fn folder(variable: &str) -> Option<PathBuf> {
    match std::env::var_os(variable) {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("{variable} is absent and NAGAYA_REQUIRE_MODELS is set")
        }
        None => {
            eprintln!("skipped: {variable} is absent");
            None
        }
    }
}

/// The opened engine `name` on the folder `dir`, with the identity that the LFM3 engine reports: the `model` of its
/// checkpoint and the tokenizer hash of its folder.
fn opened(name: &str, dir: &std::path::Path) -> OpenedEngine {
    let family = nagaya_text_lfm3::Lfm3Family::from_dir(dir).unwrap();
    let fake = nagaya_abi::fake::FakeAlways::new().with_family("lfm3");
    let mut identity = fake.identity();
    identity.model = match family.kind() {
        nagaya_text_lfm3::Kind::Text => "lfm3",
        nagaya_text_lfm3::Kind::Omni => "lfm3-omni",
    }
    .to_owned();
    identity.tokenizer_hash = family.tokenizer_hash().to_owned();
    OpenedEngine {
        name: name.to_owned(),
        thread: std::thread::current().id(),
        identity,
        limits: fake.limits(),
        handoff: None,
    }
}

#[test]
fn two_engines_of_one_lfm3_checkpoint_serve_their_engine_names() {
    let (Some(text), Some(omni)) = (folder("LFM3_MODEL_DIR"), folder("LFM3_OMNI_MODEL_DIR")) else {
        return;
    };
    let mut registry = FamilyRegistry::new();
    registry.register_with_objects("lfm3", nagaya::lfm3::open_family_with);
    let (mut served, mut checked) = (Vec::new(), Vec::new());
    for (name, dir) in [("lfm3-omni", &omni), ("lfm3", &text), ("lfm3-1", &text)] {
        let table: toml::Table = format!("model_dir = \"{}\"", dir.display()).parse().unwrap();
        let plan = registry
            .plan(&opened(name, dir), &table)
            .unwrap()
            .expect("the LFM3 family is registered");
        assert_eq!(plan.config.engine, name);
        served.push(plan.config.model);
        checked.extend(nagaya::lfm3::served_names(name, &table).unwrap_or_default());
    }
    assert_eq!(served, ["lfm3-omni", "lfm3", "lfm3-1"]);
    // The profile check reads the same names before the open (spec 2.3, "The check of a name").
    assert_eq!(checked, served);
}
