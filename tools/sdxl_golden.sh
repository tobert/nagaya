#!/usr/bin/env bash
# Writes the reference files of the image oracle (spec 8.3, 11.9) for SDXL base 1.0.
#
# The reference is diffusers, on the CPU in float32 from the fp16 weights of the checkpoint. Do not run it with a GPU
# test or with another big Python job: it holds about 20 GB of RAM.
#
#   SDXL_MODEL_DIR=<snapshot of stabilityai/stable-diffusion-xl-base-1.0> tools/sdxl_golden.sh
#
# REF_PYTHON overrides the python (default ~/.venvs/sdxl-ref/bin/python). The files go to $SDXL_GOLDEN_DIR, or to
# target/sdxl-golden. Do not commit them. Each file carries the request, the scheduler config and the versions in its
# safetensors metadata.
#
# The request: prompt "a red fox in a snowy forest, watercolor", negative "blurry, low quality", 256 wide by 320
# high, 4 steps, CFG 7, seed 1234. The image is below the request minimum of 512 (spec 11.9a) to keep the CPU time
# small. At that size and step count SDXL gives no picture, and plain diffusers gives none either. The same encoding
# at 512 by 512 with 15 Euler steps gives the fox (checked 2026-10-28). The engine has its own random stream, so the noise comes from torch.Generator().manual_seed(1234): first the
# start latent, then one draw for each step. Every sampler gets the same noise.
#
# H = 40 and W = 32 are the latent size. Row 0 of a batch is the negative prompt, row 1 the prompt. Floats are f32.
#
# encode.safetensors: the text encoders, with the A1111 encoding of spec 11.9 and no token weights.
#   ids1, ids2                      [2, 77]        i64 ids of the CLIP tokenizer and of tokenizer_2, padded
#   clip_skip{1,2}.prompt_embeds    [2, 77, 2048]  hidden state -(clip_skip + 1) of text_encoder (768 features) and
#                                                  of text_encoder_2 (1280), joined along the features
#   clip_skip{1,2}.pooled           [2, 1280]      text_embeds of text_encoder_2; clip_skip does not change it
#   add_time_ids                    [6]            original size, crop origin, target size, each as height, width
#
# samplers.safetensors: the oracle. S is euler, euler_a, dpmpp_2m_karras or dpmpp_2m_sde_karras (spec 11.9a).
#   start                   [1, 4, H, W]          the start noise, before the scheduler scales it
#   S.sigmas                [5]                   the sigmas of the steps and the final 0
#   S.timesteps             [4]                   the timestep of each step
#   S.step_noise            [D, 1, 4, H, W]       the noise of each draw; D is 0 for euler and dpmpp_2m_karras
#   S.latents               [5, 1, 4, H, W]       the latent before each step (start * init_noise_sigma first),
#                                                 then after the last step
#   S.unet_in               [4, 2, 4, H, W]       the UNet input of each step, after scale_model_input
#   S.unet_out              [4, 2, 4, H, W]       the noise prediction of each step, negative first
#   S.guided                [4, 1, 4, H, W]       negative + 7 * (prompt - negative)
#   S.final_latent          [1, 4, H, W]          the latent after the last step
#   S.image                 [3, 320, 256]         the VAE decode, in [-1, 1] before the clamp
#   S.rgb                   [320, 256, 3]         u8 RGB: round(clamp(image / 2 + 0.5, 0, 1) * 255)
#
# parts.safetensors: references for kernel bring-up, from step 0 of euler.
#   unet.in                 [2, 4, H, W]          the UNet input; unet.timestep [1]; unet.encoder_hidden_states
#                                                 [2, 77, 2048]
#   unet.conv_in            [2, 320, H, W]
#   unet.time_embedding     [2, 1280]             unet.add_embedding [2, 1280]
#   unet.down0..2           the hidden state after each down block: [2, 320, 20, 16], [2, 640, 10, 8],
#                           [2, 1280, 10, 8]
#   unet.mid                [2, 1280, 10, 8]
#   unet.up0..2             [2, 1280, 20, 16], [2, 640, 40, 32], [2, 320, 40, 32]
#   unet.conv_norm_out      [2, 320, H, W]        after the GroupNorm; the SiLU follows
#   unet.conv_out           [2, 4, H, W]          equal to samplers euler.unet_out[0]
#   vae.in                  [1, 4, H, W]          the final latent of euler / the scaling factor 0.13025
#   vae.post_quant_conv, vae.conv_in, vae.mid, vae.up0..3, vae.conv_norm_out, vae.conv_out
#                           the VAE decoder stages; vae.conv_out is equal to samplers euler.image
#
# S.rgb.npy: the same RGB bytes, for a look with any viewer of numpy files.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${REF_PYTHON:-$HOME/.venvs/sdxl-ref/bin/python}"
MODEL="${SDXL_MODEL_DIR:?set SDXL_MODEL_DIR to the SDXL base 1.0 checkpoint directory}"
OUT="${SDXL_GOLDEN_DIR:-target/sdxl-golden}"
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
nice -n 10 "$PY" -I tools/sdxl_reference.py --model "$MODEL" --out "$OUT"
