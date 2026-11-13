//! The plan of the speech path of Omni with no card (engine-abi "The speech calls of LFM3"): the term `speech`, the
//! capability `speech_frames`, the numbers of the configs, and the tensors of the depth decoder and the codec.

mod common;

use common::Which;
use nagaya_abi::Capability;
use nagaya_mk_bmg::safetensors::{MappedSafeTensors, SafeTensorsView};
use nagaya_mk_lfm3_bmg::spec::{self, Config};
use nagaya_mk_lfm3_bmg::{codec, depth, frames};

fn config(which: Which, slots: usize) -> Config {
    Config {
        model_dir: common::model_dir(which),
        slots: vec![spec::SLOT_TOKENS; slots],
        max_block: 512,
    }
}

#[test]
fn omni_plans_the_term_speech_and_the_text_model_plans_none() {
    if common::absent(Which::Omni) {
        return;
    }
    let one = spec::plan(&config(Which::Omni, 1)).unwrap();
    let two = spec::plan(&config(Which::Omni, 2)).unwrap();
    one.check().unwrap();
    assert!(one.capabilities.contains(&Capability::SpeechFrames));
    let term = |l: &nagaya_abi::Limits| l.memory.mandatory.iter().find(|t| t.name == "speech").unwrap().bytes;
    let weights = frames::weights_bytes(&common::model_dir(Which::Omni)).unwrap();
    println!(
        "speech term: {} bytes with 1 slot; weights {weights}, buffers {}, each slot {}",
        term(&one),
        frames::bufs_bytes(),
        frames::slot_bytes()
    );
    assert_eq!(
        term(&two) - term(&one),
        frames::slot_bytes() as u64,
        "each slot adds its codec state"
    );
    assert_eq!(
        term(&one),
        (weights + frames::bufs_bytes() + frames::slot_bytes()) as u64
    );
    // The depth decoder, the codebook table and the decoder of the codec are BF16; the window and the table are F32.
    let depth_params: usize = depth::tensors().iter().map(|(_, s)| s.iter().product::<usize>()).sum();
    let codec_params: usize = codec::tensors().iter().map(|(_, s)| s.iter().product::<usize>()).sum();
    assert!(
        weights >= 2 * (depth_params + codec_params),
        "{weights} bytes hold every tensor"
    );
    if common::absent(Which::Text) {
        return;
    }
    let text = spec::plan(&config(Which::Text, 1)).unwrap();
    assert!(!text.capabilities.contains(&Capability::SpeechFrames));
    assert!(text.memory.mandatory.iter().all(|t| t.name != "speech"));
}

#[test]
fn the_configs_of_omni_hold_the_numbers_of_the_speech_path() {
    if common::absent(Which::Omni) {
        return;
    }
    let dir = common::model_dir(Which::Omni);
    frames::check_configs(&dir).unwrap();
    // A copy of the configs with one other number is refused, with the file and the key.
    let tmp = std::env::temp_dir().join(format!("nagaya-lfm3-speech-{}", std::process::id()));
    std::fs::create_dir_all(tmp.join("audio_tokenizer")).unwrap();
    let copy = |name: &str, edit: &dyn Fn(&mut serde_json::Value)| {
        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join(name)).unwrap()).unwrap();
        edit(&mut value);
        std::fs::write(tmp.join(name), value.to_string()).unwrap();
    };
    copy("audio_tokenizer/config.json", &|_| {});
    copy("config.json", &|v| v["maskgit_schedule"] = serde_json::json!([6, 6]));
    let error = frames::check_configs(&tmp).unwrap_err();
    assert!(error.contains("config.json has maskgit_schedule [6,6]"), "{error}");
    copy("config.json", &|v| v["depth_config"]["head_dim"] = 64.into());
    let error = frames::check_configs(&tmp).unwrap_err();
    assert!(error.contains("depth_config.head_dim 64"), "{error}");
    copy("config.json", &|_| {});
    copy("audio_tokenizer/config.json", &|v| v["decoder_window"] = 32.into());
    let error = frames::check_configs(&tmp).unwrap_err();
    assert!(
        error.contains("audio_tokenizer/config.json has decoder_window 32"),
        "{error}"
    );
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn the_files_of_omni_hold_each_tensor_of_the_depth_decoder_and_the_codec() {
    if common::absent(Which::Omni) {
        return;
    }
    let dir = common::model_dir(Which::Omni);
    let map = MappedSafeTensors::open(&dir.join(spec::WEIGHTS_FILE)).unwrap();
    depth::check(&map.view().unwrap()).unwrap();
    let map = MappedSafeTensors::open(&dir.join(codec::WEIGHTS_FILE)).unwrap();
    codec::check(&map.view().unwrap()).unwrap();
}

#[test]
fn a_file_without_a_tensor_of_the_speech_path_is_refused_with_its_name() {
    let bytes = SafeTensorsView::build_synthetic_safetensors(&[(
        "model.depth_decoder.mask_embedding",
        "BF16",
        &[1024],
        &[0u8; 2048],
    )]);
    let view = SafeTensorsView::parse(&bytes).unwrap();
    let error = depth::check(&view).unwrap_err();
    assert!(
        error.contains("no tensor model.depth_decoder.depth_linear.weight"),
        "{error}"
    );
    let error = codec::check(&view).unwrap_err();
    assert!(error.contains("no tensor quantizer.project_out.weight"), "{error}");
}
