//! The shapes of Qwen3-TTS 12 Hz 1.7B VoiceDesign, the config of the engine, and its limits. Nothing here needs a
//! card, so the engine plans with no card (engine-abi "Core").

use std::path::{Path, PathBuf};

use serde_json::Value;

/// The model name in the identity: the served speech model (spec 11.8).
pub const MODEL: &str = "Qwen3-TTS-12Hz-1.7B-VoiceDesign";

/// The engine kind in a placement profile.
pub const KIND: &str = "qwen3tts-bmg";

/// The talker: a Qwen3 decoder over text and codec embeddings.
pub const HIDDEN: usize = 2048;
pub const LAYERS: usize = 28;
pub const HEADS: usize = 16;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const INTER: usize = 6144;
pub const CODEC_VOCAB: usize = 3072;
pub const TEXT_VOCAB: usize = 151936;
pub const EPS: f32 = 1e-6;

/// The code predictor: a small Qwen3 decoder that samples codebooks 1 to 15 of a frame.
pub const CP_HIDDEN: usize = 1024;
pub const CP_LAYERS: usize = 5;
pub const CP_INTER: usize = 3072;
pub const CP_VOCAB: usize = 2048;

/// The codebooks of a frame.
pub const CODEBOOKS: usize = 16;

/// The codec decoder.
pub const CODEBOOK_SIZE: usize = 2048;
pub const VQ_DIM: usize = 256;
pub const CODEBOOK_DIM: usize = 512;
pub const LATENT: usize = 1024;
pub const DEC_HIDDEN: usize = 512;
pub const DEC_LAYERS: usize = 8;
pub const DEC_HEADS: usize = 16;
pub const DEC_HEAD_DIM: usize = 64;
pub const DEC_INTER: usize = 1024;
pub const DEC_WINDOW: usize = 72;
pub const DEC_EPS: f32 = 1e-5;
pub const DECODER_DIM: usize = 1536;
pub const UPSAMPLING_RATIOS: [usize; 2] = [2, 2];
pub const UPSAMPLE_RATES: [usize; 4] = [8, 5, 4, 3];
/// The audio samples of one frame at 24 kHz: 80 ms.
pub const SAMPLES_PER_FRAME: usize = 1920;
pub const SAMPLE_RATE: u32 = 24000;

/// The codec ids that the engine reads at decode time (`config.json`, `talker_config`).
pub const CODEC_EOS: u32 = 2150;
/// The text id of the trailing text of each frame in non-streaming mode: `tts_pad`.
pub const TTS_PAD: u32 = 151671;
/// The codec ids from here are suppressed in codebook 0, except `CODEC_EOS` (spec 11.8 sampler).
pub const SUPPRESS_FROM: u32 = (CODEC_VOCAB - 1024) as u32;
/// End-of-speech stays suppressed until this many ids are in the history (the reference's min-new-tokens).
pub const MIN_NEW: u32 = 2;

/// The config of one engine, from its profile entry.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// The checkpoint directory: `config.json`, `model.safetensors`, and `speech_tokenizer/`.
    pub model_dir: PathBuf,
    /// The requests that the engine holds at once.
    pub slots: u32,
    /// The most frames of one request: 1,500 frames are 120 s of audio.
    pub max_frames: u32,
    /// The most positions of one prompt.
    pub max_prompt: u32,
    /// The most frames of one `speech_frames` call (one quantum).
    pub frames_per_quantum: u32,
}

impl Config {
    /// The positions of the talker's KV for one slot.
    pub fn max_positions(&self) -> u32 {
        self.max_prompt + self.max_frames + 1
    }
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "cannot read {}: {error}; give the Qwen3-TTS VoiceDesign checkpoint directory",
            path.display()
        )
    })?;
    serde_json::from_str(&text).map_err(|error| format!("{} is not JSON: {error}", path.display()))
}

/// Check that the checkpoint in `model_dir` has the shapes of this engine. Another checkpoint is an error that names
/// the first value that differs.
pub fn check_config(model_dir: &Path) -> Result<(), String> {
    let config = read_json(&model_dir.join("config.json"))?;
    let codec = read_json(&model_dir.join("speech_tokenizer/config.json"))?;
    let talker = &config["talker_config"];
    let cp = &talker["code_predictor_config"];
    let dec = &codec["decoder_config"];
    let want: [(&str, &Value, usize); 22] = [
        ("talker_config.hidden_size", &talker["hidden_size"], HIDDEN),
        ("talker_config.num_hidden_layers", &talker["num_hidden_layers"], LAYERS),
        (
            "talker_config.num_attention_heads",
            &talker["num_attention_heads"],
            HEADS,
        ),
        (
            "talker_config.num_key_value_heads",
            &talker["num_key_value_heads"],
            KV_HEADS,
        ),
        ("talker_config.head_dim", &talker["head_dim"], HEAD_DIM),
        ("talker_config.intermediate_size", &talker["intermediate_size"], INTER),
        ("talker_config.vocab_size", &talker["vocab_size"], CODEC_VOCAB),
        ("talker_config.text_vocab_size", &talker["text_vocab_size"], TEXT_VOCAB),
        ("talker_config.num_code_groups", &talker["num_code_groups"], CODEBOOKS),
        (
            "talker_config.codec_eos_token_id",
            &talker["codec_eos_token_id"],
            CODEC_EOS as usize,
        ),
        ("tts_pad_token_id", &config["tts_pad_token_id"], TTS_PAD as usize),
        ("code_predictor_config.hidden_size", &cp["hidden_size"], CP_HIDDEN),
        (
            "code_predictor_config.num_hidden_layers",
            &cp["num_hidden_layers"],
            CP_LAYERS,
        ),
        (
            "code_predictor_config.intermediate_size",
            &cp["intermediate_size"],
            CP_INTER,
        ),
        ("code_predictor_config.vocab_size", &cp["vocab_size"], CP_VOCAB),
        ("decoder_config.latent_dim", &dec["latent_dim"], LATENT),
        ("decoder_config.hidden_size", &dec["hidden_size"], DEC_HIDDEN),
        (
            "decoder_config.num_hidden_layers",
            &dec["num_hidden_layers"],
            DEC_LAYERS,
        ),
        ("decoder_config.head_dim", &dec["head_dim"], DEC_HEAD_DIM),
        ("decoder_config.sliding_window", &dec["sliding_window"], DEC_WINDOW),
        ("decoder_config.decoder_dim", &dec["decoder_dim"], DECODER_DIM),
        ("decoder_config.codebook_size", &dec["codebook_size"], CODEBOOK_SIZE),
    ];
    for (name, got, want) in want {
        if got.as_u64() != Some(want as u64) {
            return Err(format!(
                "the checkpoint has {name} = {got}, and this engine runs {want}; use the Qwen3-TTS 12 Hz 1.7B \
                 VoiceDesign checkpoint"
            ));
        }
    }
    let rates: Vec<u64> = dec["upsample_rates"]
        .as_array()
        .map(|values| values.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default();
    if rates != UPSAMPLE_RATES.map(|r| r as u64) {
        return Err(format!(
            "the checkpoint has decoder_config.upsample_rates = {rates:?}, and this engine runs {UPSAMPLE_RATES:?}"
        ));
    }
    if config["tts_model_type"] != "voice_design" {
        return Err(format!(
            "the checkpoint is tts_model_type {}, and this engine runs voice_design",
            config["tts_model_type"]
        ));
    }
    Ok(())
}

/// The channels at the input of decoder block `b` (spec of the codec: 1536, then halves).
pub fn block_in(b: usize) -> usize {
    DECODER_DIM >> b
}

/// The audio steps of `frames` frames after the codec's two ConvNeXt stages, before the decoder blocks.
pub fn upsampled(frames: usize) -> usize {
    frames * UPSAMPLING_RATIOS.iter().product::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rates_give_the_samples_of_a_frame() {
        assert_eq!(
            upsampled(1) * UPSAMPLE_RATES.iter().product::<usize>(),
            SAMPLES_PER_FRAME
        );
        assert_eq!(block_in(3), 192);
    }
}
