//! The shape of SDXL base 1.0 and the config of the engine (spec 11.9, 11.9a).

use std::path::PathBuf;

/// The channels of the latent.
pub const LATENT: usize = 4;
/// The input channels of an inpainting UNet: the scaled latent, the latent mask, and the masked image latent (spec
/// 11.9b, "The 9-channel UNet"). A UNet for text to image has `LATENT`.
pub const INPAINT_CHANNELS: usize = 9;
/// The factor from pixels to latent rows and columns.
pub const VAE_SCALE: usize = 8;
/// The factor of a latent before the VAE decode: the latent is divided by it.
pub const VAE_SCALING_FACTOR: f32 = 0.13025;
/// The tokens of one chunk with its start and end ids.
pub const CHUNK_IDS: usize = 77;
/// The most chunks of a prompt (spec 11.9a).
pub const MAX_CHUNKS: usize = 3;
/// The most images of a job (spec 11.9a).
pub const MAX_IMAGES: usize = 4;
/// The most pixels of one image (spec 11.9a).
pub const MAX_IMAGE_PIXELS: usize = 1_572_864;
/// The values of a head of attention in the UNet and in the text encoders.
pub const HEAD: usize = 64;
/// The groups of each GroupNorm.
pub const GROUPS: u32 = 32;

/// A CLIP text encoder.
#[derive(Clone, Copy, Debug)]
pub struct TextShape {
    pub hidden: usize,
    pub layers: usize,
    pub mlp: usize,
    /// True for CLIP ViT-L, whose activation is quick gelu; false for OpenCLIP ViT-bigG, whose activation is gelu.
    pub quick_gelu: bool,
}

pub const TEXT1: TextShape = TextShape {
    hidden: 768,
    layers: 12,
    mlp: 3072,
    quick_gelu: true,
};
pub const TEXT2: TextShape = TextShape {
    hidden: 1280,
    layers: 32,
    mlp: 5120,
    quick_gelu: false,
};
/// The features of a row of the text context: both encoders, joined.
pub const CONTEXT: usize = 768 + 1280;
/// The pooled output of the second encoder.
pub const POOLED: usize = 1280;

/// The channels of each level of the UNet.
pub const LEVELS: [usize; 3] = [320, 640, 1280];
/// The transformer blocks of each attention of a level; level 0 has none.
pub const DEPTH: [usize; 3] = [0, 2, 10];
/// The time embedding of the UNet.
pub const TIME: usize = 1280;
/// The sinusoidal features of a timestep, and of each of the 6 time ids.
pub const TIME_PROJ: usize = 320;
pub const TIME_ID_PROJ: usize = 256;
/// The input of the add embedding: the pooled text and the 6 time ids.
pub const ADD_IN: usize = POOLED + 6 * TIME_ID_PROJ;

/// The channels of each level of the VAE decoder, from the latent side.
pub const VAE_LEVELS: [usize; 4] = [512, 512, 256, 128];
/// The channels of each level of the VAE encoder, from the image side: `block_out_channels` of `vae/config.json`.
pub const VAE_ENCODER_LEVELS: [usize; 4] = [128, 256, 512, 512];

/// The model name of the identity.
pub const MODEL: &str = "stable-diffusion-xl-base-1.0";

/// The engine kind in a placement profile (spec 11.9, "The served image model").
pub const KIND: &str = "mk-sdxl-bmg";

/// The config of the engine, from the `[engines.config]` table of the profile.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// A checkpoint in the folder layout of diffusers. With `checkpoint`, only its CLIP tokenizer (`tokenizer/`).
    pub model_dir: PathBuf,
    /// A single file with Stability's tensor names, such as a checkpoint from Civitai. The weights then come from it.
    pub checkpoint: Option<PathBuf>,
}

impl Config {
    /// The config of a checkpoint in the folder layout of diffusers.
    pub fn folder(model_dir: PathBuf) -> Config {
        Config {
            model_dir,
            checkpoint: None,
        }
    }
}
