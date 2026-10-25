//! The plan of the speech engine, with no card (spec 2.4).

mod common;

use std::path::PathBuf;

use nagaya_abi::{Capability, Tier};
use nagaya_mk_bmg::protocol::MAX_TASKS_PER_CARD;
use nagaya_mk_bmg::safetensors::MappedSafeTensors;
use nagaya_mk_qwen3tts_bmg::model::{self, Slot, Work, bytes_of};
use nagaya_mk_qwen3tts_bmg::plan::{limits, weights_bytes};
use nagaya_mk_qwen3tts_bmg::weights::{Count, load};
use nagaya_mk_qwen3tts_bmg::{Config, plan};

fn config(model_dir: PathBuf) -> Config {
    Config {
        model_dir,
        slots: 2,
        max_frames: 1500,
        max_prompt: 512,
        frames_per_quantum: 8,
    }
}

#[test]
fn a_zero_or_a_quantum_larger_than_a_request_is_refused() {
    let good = config(PathBuf::from("/nowhere"));
    assert_eq!(good.check(), Ok(()));
    for broken in [
        Config {
            slots: 0,
            ..good.clone()
        },
        Config {
            max_frames: 0,
            ..good.clone()
        },
        Config {
            max_prompt: 0,
            ..good.clone()
        },
        Config {
            frames_per_quantum: 0,
            ..good.clone()
        },
        Config {
            frames_per_quantum: 1501,
            ..good.clone()
        },
    ] {
        assert!(broken.check().is_err(), "{broken:?} passed the check");
    }
}

#[test]
fn the_limits_name_speech_one_card_and_the_slots() {
    let config = config(PathBuf::from("/nowhere"));
    let limits = limits(&config, 1000);
    assert_eq!(limits.capabilities, vec![Capability::Speech, Capability::Close]);
    assert_eq!(limits.devices, vec![0]);
    assert!(limits.stages.is_empty());
    assert_eq!(limits.slots.len(), 1);
    assert_eq!(limits.slots[0].count, 2);
    assert_eq!(limits.slots[0].tokens, 512 + 1500 + 1);
    let term = |name: &str| limits.memory.mandatory.iter().find(|t| t.name == name).unwrap().clone();
    assert_eq!(term("weights").bytes, 1000);
    assert_eq!(term("weights").tier, Tier::Device(0));
    let slot = bytes_of(|z| Slot::new(z, &config)) as u64;
    assert_eq!(term("slots").bytes, 2 * slot);
    assert_eq!(limits.slots[0].bytes, slot);
    // The talker's KV of one slot: 28 layers, K and V, 2013 positions, 8 heads of 128, f32.
    assert!(
        slot > 28 * 2 * 2013 * 8 * 128 * 4,
        "a slot of {slot} bytes has no room for its KV"
    );
    let work = bytes_of(|z| Work::new(z, &config)) as u64;
    assert!(term("work").bytes > work);
}

#[test]
fn the_plan_reads_the_headers_of_the_checkpoint() {
    if common::absent() {
        return;
    }
    let config = config(common::model_dir());
    let limits = plan(&config).expect("plan the checkpoint");
    let weights = limits
        .memory
        .mandatory
        .iter()
        .find(|t| t.name == "weights")
        .unwrap()
        .bytes;
    assert_eq!(weights, weights_bytes(&config).unwrap());
    // The talker and the code predictor are 1.9 G BF16 values; the codec and its tables are F32.
    assert!(
        (3_500_000_000..5_500_000_000).contains(&weights),
        "the weights are {weights} bytes"
    );
}

#[test]
fn another_checkpoint_is_refused_by_name() {
    if common::absent() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("nagaya-qwen3tts-plan-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("speech_tokenizer")).unwrap();
    let source = common::model_dir();
    let mut talker: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(source.join("config.json")).unwrap()).unwrap();
    talker["talker_config"]["num_hidden_layers"] = 20.into();
    std::fs::write(dir.join("config.json"), talker.to_string()).unwrap();
    std::fs::copy(
        source.join("speech_tokenizer/config.json"),
        dir.join("speech_tokenizer/config.json"),
    )
    .unwrap();
    let error = plan(&config(dir.clone())).unwrap_err();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(error.contains("talker_config.num_hidden_layers = 20"), "{error}");
}

#[test]
fn each_pass_fits_one_ring() {
    if common::absent() {
        return;
    }
    let config = config(common::model_dir());
    let model_map = MappedSafeTensors::open(&config.model_dir.join("model.safetensors")).unwrap();
    let codec_map = MappedSafeTensors::open(&config.model_dir.join("speech_tokenizer/model.safetensors")).unwrap();
    let weights = load(
        &mut Count::default(),
        &model_map.view().unwrap(),
        &codec_map.view().unwrap(),
    )
    .unwrap();
    let work = Work::new(&mut |_| 0, &config);
    let slot = Slot::new(&mut |_| 0, &config);
    let sampling = Some(model::Sampling {
        temperature: 0.9,
        top_k: 50,
        penalty: 1.05,
    });
    let passes = [
        (
            "prefill",
            model::prefill(&weights, &work, &slot, &config, 512, sampling).0.len(),
        ),
        (
            "frame",
            model::frame(&weights, &work, &slot, &config, 512, 1499, sampling)
                .0
                .len(),
        ),
        ("codec", model::codec(&weights, &work, &slot, &config, 0, 8, 0).0.len()),
    ];
    for (name, tasks) in passes {
        eprintln!("{name}: {tasks} tasks");
        assert!(
            tasks < MAX_TASKS_PER_CARD,
            "the {name} pass has {tasks} tasks, and a ring holds {}",
            MAX_TASKS_PER_CARD - 1
        );
    }
}
