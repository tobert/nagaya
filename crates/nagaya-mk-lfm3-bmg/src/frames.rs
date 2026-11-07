//! The plan of the speech path of Omni (milestone 24 step 8; engine-abi "The speech calls of LFM3"): the numbers of
//! the configs that the depth decoder and the codec take, and the device bytes of the term `speech`. It needs no
//! card, and it reads no weight bytes (spec 2.4).

use std::path::Path;

use nagaya_abi::{Capability, Limits, MandatoryTerm, Tier};
use nagaya_mk_bmg::safetensors::MappedSafeTensors;

use crate::codes::{CODEBOOKS, CODES, END_CODE, SCHEDULE};
use crate::spec::WEIGHTS_FILE;
use crate::weights::{Count, rounded};
use crate::{codec, depth};

/// The name of the memory term of the speech path.
pub const TERM: &str = "speech";

/// The placeholder token of a frame of audio out (`audio_out_token_id`): a fed frame takes its position.
pub const AUDIO_OUT_TOKEN: u32 = 125_032;

/// The config of the codec in the folder of Omni.
pub const CODEC_CONFIG: &str = "audio_tokenizer/config.json";

fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{} is not JSON: {e}", path.display()))
}

/// Check the numbers of `config.json` and `audio_tokenizer/config.json` of Omni in `dir` that the speech path takes.
/// A value that the speech path cannot run is an error that names the file and the key.
pub fn check_configs(dir: &Path) -> Result<(), String> {
    let root = read_json(&dir.join("config.json"))?;
    let codec_config = read_json(&dir.join(CODEC_CONFIG))?;
    let want = |file: &str, key: &str, got: &serde_json::Value, need: serde_json::Value| -> Result<(), String> {
        if *got == need {
            Ok(())
        } else {
            Err(format!("{file} has {key} {got}, and the speech path needs {need}"))
        }
    };
    let c = "config.json";
    want(c, "num_codebooks", &root["num_codebooks"], CODEBOOKS.into())?;
    want(c, "codebook_size", &root["codebook_size"], CODES.into())?;
    want(c, "audio_eos_code", &root["audio_eos_code"], END_CODE.into())?;
    want(
        c,
        "maskgit_schedule",
        &root["maskgit_schedule"],
        serde_json::json!(SCHEDULE),
    )?;
    want(
        c,
        "audio_out_token_id",
        &root["audio_out_token_id"],
        AUDIO_OUT_TOKEN.into(),
    )?;
    let d = &root["depth_config"];
    for (key, need) in [
        ("hidden_size", depth::HIDDEN),
        ("num_attention_heads", depth::HEADS),
        ("num_key_value_heads", depth::KV_HEADS),
        ("head_dim", depth::HEAD_DIM),
        ("intermediate_size", depth::INTERMEDIATE),
        ("num_hidden_layers", depth::LAYERS),
    ] {
        want(c, &format!("depth_config.{key}"), &d[key], need.into())?;
    }
    want(
        c,
        "depth_config.layer_types",
        &d["layer_types"],
        serde_json::json!(["full_attention", "full_attention", "full_attention"]),
    )?;
    want(c, "depth_config.norm_eps", &d["norm_eps"], 1e-5.into())?;
    want(
        c,
        "depth_config.use_sandwich_norm",
        &d["use_sandwich_norm"],
        false.into(),
    )?;
    let rope = &d["rope_parameters"]["full_attention"];
    want(
        c,
        "depth_config.rope_parameters.full_attention.rope_theta",
        &rope["rope_theta"],
        1e6.into(),
    )?;
    want(
        c,
        "depth_config.rope_parameters.full_attention.partial_rotary_factor",
        &rope["partial_rotary_factor"],
        1.0.into(),
    )?;
    let k = CODEC_CONFIG;
    want(k, "n_q", &codec_config["n_q"], CODEBOOKS.into())?;
    want(
        k,
        "fsq_levels",
        &codec_config["fsq_levels"],
        serde_json::json!(codec::LEVELS),
    )?;
    want(
        k,
        "decoder_window",
        &codec_config["decoder_window"],
        codec::WINDOW.into(),
    )?;
    want(
        k,
        "frame_samples",
        &codec_config["frame_samples"],
        codec::FRAME_SAMPLES.into(),
    )?;
    want(
        k,
        "stft_frame_stack",
        &codec_config["stft_frame_stack"],
        codec::STACK.into(),
    )?;
    want(k, "stft_n_fft", &codec_config["stft_n_fft"], codec::N_FFT.into())?;
    want(
        k,
        "stft_win_length",
        &codec_config["stft_win_length"],
        codec::N_FFT.into(),
    )?;
    want(
        k,
        "stft_hop_length",
        &codec_config["stft_hop_length"],
        codec::HOP.into(),
    )?;
    want(k, "encoder_dim", &codec_config["encoder_dim"], codec::HIDDEN.into())?;
    want(
        k,
        "quantizer_dim",
        &codec_config["quantizer_dim"],
        codec::QUANTIZER.into(),
    )?;
    Ok(())
}

/// The device bytes of the speech path of one slot: the hidden row of its last position and its codec state.
pub fn slot_bytes() -> usize {
    rounded(depth::BACKBONE * 4) + codec::Slot::bytes()
}

/// The device bytes of the buffers of the speech path, shared by the slots.
pub fn bufs_bytes() -> usize {
    depth::Bufs::bytes() + codec::Bufs::bytes()
}

/// The device bytes of the weights of the speech path in `dir`, from the tensor headers.
pub fn weights_bytes(dir: &Path) -> Result<usize, String> {
    let map = MappedSafeTensors::open(&dir.join(WEIGHTS_FILE))?;
    let codec_map = MappedSafeTensors::open(&dir.join(codec::WEIGHTS_FILE))?;
    let mut count = Count::default();
    depth::load(&mut count, &map.view()?)?;
    codec::load(&mut count, &codec_map.view()?)?;
    Ok(count.bytes)
}

/// The device bytes of the term `speech` for `slots` slots, with the weights of `weights` bytes.
pub fn term_bytes(weights: usize, slots: usize) -> u64 {
    (weights + bufs_bytes() + slots * slot_bytes()) as u64
}

/// Add the capability `speech_frames` and the term `speech` of `bytes` device bytes to the limits of Omni.
pub fn declare(limits: &mut Limits, bytes: u64) {
    limits.capabilities.push(Capability::SpeechFrames);
    let kinds = limits
        .memory
        .mandatory
        .first()
        .map(|term| term.kinds.clone())
        .unwrap_or_default();
    limits.memory.mandatory.push(MandatoryTerm {
        name: TERM.to_owned(),
        tier: Tier::Device(0),
        bytes,
        lifetime: nagaya_abi::Lifetime::Engine,
        kinds,
    });
}

/// The term `speech` of the plan of Omni in `dir`.
pub fn plan_bytes(dir: &Path, slots: usize) -> Result<u64, String> {
    check_configs(dir)?;
    Ok(term_bytes(weights_bytes(dir)?, slots))
}
