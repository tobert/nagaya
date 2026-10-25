//! The Qwen3-TTS kind: the config of the profile, the plan with no card, and the speech family (spec 2.4, 11.8).
//!
//! A test that needs the checkpoint skips when `QWEN3TTS_MODEL_DIR` is not set, and fails then with
//! `NAGAYA_REQUIRE_MODELS=1` (AGENTS "Tests that need a model file").

use std::path::PathBuf;

use nagaya::admission::{GroupDevice, Registry};
use nagaya::qwen3tts::{Family, KIND, config, plan};
use nagaya::speech::{SpeechError, SpeechFamily};
use nagaya_abi::{Capability, Tier};
use nagaya_text_qwen3tts::Qwen3TtsFamily;

fn checkpoint() -> Option<PathBuf> {
    match std::env::var_os("QWEN3TTS_MODEL_DIR") {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("QWEN3TTS_MODEL_DIR is not set and NAGAYA_REQUIRE_MODELS is set; name the Qwen3-TTS checkpoint")
        }
        None => {
            eprintln!("skipped: QWEN3TTS_MODEL_DIR is not set; set it to the Qwen3-TTS checkpoint directory");
            None
        }
    }
}

fn table(text: &str) -> toml::Table {
    text.parse().expect("a TOML table")
}

fn device(id: u32) -> GroupDevice {
    GroupDevice {
        id,
        name: "xe".to_owned(),
        pci_address: format!("0000:0{id}:00.0"),
        memory_bytes: 32 << 30,
    }
}

#[test]
fn the_config_takes_its_keys_with_defaults_and_refuses_others() {
    let dir = std::env::temp_dir();
    let given = config(&table(&format!(
        "model_dir = {:?}\nslots = 3\nmax_frames = 600",
        dir.display().to_string()
    )))
    .unwrap();
    assert_eq!(given.model_dir, dir);
    assert_eq!(
        (
            given.slots,
            given.max_frames,
            given.max_prompt,
            given.frames_per_quantum
        ),
        (3, 600, 512, 8)
    );
    let error = config(&table("model_dir = \"/tmp\"\nvoice = \"vivian\"")).unwrap_err();
    assert!(error.message.contains("`voice`"), "{}", error.message);
    let error = config(&table("model_dir = \"/tmp\"\nslots = -1")).unwrap_err();
    assert!(error.message.contains("`slots`"), "{}", error.message);
    let error = config(&table("model_dir = \"/tmp\"\nframes_per_quantum = 0")).unwrap_err();
    assert!(error.message.contains("frames_per_quantum"), "{}", error.message);
}

#[test]
fn a_group_of_two_devices_is_refused() {
    let error = plan(&table("model_dir = \"/tmp\""), &[device(0), device(1)]).unwrap_err();
    assert!(error.message.contains("one card"), "{}", error.message);
}

#[test]
fn the_kind_exists_only_with_the_feature_xe() {
    assert_eq!(Registry::compiled_in().factory(KIND).is_some(), cfg!(feature = "xe"));
}

#[test]
fn the_plan_gives_speech_on_one_card_with_no_card() {
    let Some(dir) = checkpoint() else { return };
    let limits = plan(
        &table(&format!("model_dir = {:?}", dir.display().to_string())),
        &[device(1)],
    )
    .unwrap();
    assert!(limits.capabilities.contains(&Capability::Speech));
    assert_eq!(limits.devices, vec![0]);
    let weights = limits.memory.mandatory.iter().find(|t| t.name == "weights").unwrap();
    assert_eq!(weights.tier, Tier::Device(0));
}

#[test]
fn the_family_names_the_language_param_of_an_unknown_language() {
    let Some(dir) = checkpoint() else { return };
    let family = Family(Qwen3TtsFamily::from_dir(&dir).unwrap());
    let prompt = family
        .prompt("The nagaya is awake.", "A calm voice.", "english")
        .unwrap();
    assert!(prompt.len() > 10);
    match family.prompt("Hi.", "A calm voice.", "klingon") {
        Err(SpeechError::Invalid { param, .. }) => assert_eq!(param, "language"),
        other => panic!("an unknown language gave {other:?}"),
    }
}
