#!/usr/bin/env bash
# Writes the reference files of img2img and inpainting (spec 8.3, 11.9b) for SDXL base 1.0 and the inpainting UNet.
#
# The reference is diffusers, on the CPU in float32 from the fp16 weights. Do not run it with a GPU test or with
# another big Python job: it holds about 20 GB of RAM.
#
#   SDXL_MODEL_DIR=<snapshot of stabilityai/stable-diffusion-xl-base-1.0> \
#   SDXL_INPAINT_MODEL_DIR=<snapshot of diffusers/stable-diffusion-xl-1.0-inpainting-0.1> \
#   tools/sdxl_inpaint_golden.sh
#
# REF_PYTHON overrides the python (default ~/.venvs/sdxl-ref/bin/python). The files go to $SDXL_GOLDEN_DIR, or to
# target/sdxl-golden, beside the files of tools/sdxl_golden.sh. Do not commit them. Each safetensors file carries the
# request, the runs, the scheduler config and the versions in its metadata.
#
# The request is that of tools/sdxl_golden.sh: the same prompts, 256 wide by 320 high, 4 steps, CFG 7, seed 1234,
# and clip_skip 2. The size is below the request minimum of 512 (spec 11.9a) to keep the CPU time small. mask_blur
# is 4. The cases:
#
#   img2img   the base UNet; each sampler at strength 0.5 (n = 2), and euler at 1.0 (n = 4)
#   inpaint   the base UNet with the latent blend; each sampler at 0.99 (n = 3), and euler and dpmpp_2m_karras at 1.0
#   inpaint9  the 9-channel UNet, with every part from SDXL_INPAINT_MODEL_DIR; the same runs as inpaint. The VAE of
#             that folder is madebyollin's sdxl-vae-fp16-fix, not the VAE of base; it runs in f32 here.
#
# The input image, RGBA, x from 0 to 255 and y from 0 to 319: R = x * 255 / 255, G = y * 255 / 319 (integer
# division), B = 255 where (x / 32 + y / 32) is even and 64 elsewhere, A = 128 + (x + y) % 128.
# The mask, RGBA: white in the ellipse ((x - 128) / 72)^2 + ((y - 160) / 96)^2 <= 1, black elsewhere. Rows 32 to 63
# are R = 7x % 256, G = (13y + x) % 256, B = x^2 % 256, so their gray values cross 128. A = 255 for x < 128, else 100.
# The script converts the mask to gray, then blurs it, then gives it to diffusers (spec 11.9b, "The mask").
#
# The noise. The engine has its own random stream, so each file records the draws with their purposes. The draws of a
# run come from torch.Generator().manual_seed(1234), in the order of diffusers:
#   1. encode         the z of the image latent, [1, 4, H, W]. Absent in an inpaint9 run at strength 1.0, because
#                     diffusers then does not encode the image (inpaint.py:805). The base UNet's inpaint encodes it.
#   2. start          the start noise, [1, 4, H, W]. The blend uses it for each `known` too.
#   3. encode_masked  the z of the masked image latent (inpaint and inpaint9 only). diffusers encodes the masked
#                     image for the base UNet too (inpaint.py:887), but that UNet does not read it.
#   4. step           one draw for each step of euler_a and dpmpp_2m_sde_karras, [1, 4, H, W] each.
# Thus every run of a case starts with the same draws. The engine draws `encode` and `encode masked` by purpose.
#
# H = 40 and W = 32 are the latent size. Row 0 of a CFG batch is the negative prompt, row 1 the prompt. Floats are
# f32 unless a line says otherwise.
#
# image_inputs.safetensors: the inputs, the processed mask, the schedule table and the VAE encoder stages.
#   init.png, mask.png    [bytes]          u8, the PNG files that a client sends; also written as init.png and mask.png
#   init.rgba             [320, 256, 4]    u8, the decoded image; init.rgb [320, 256, 3] drops the alpha
#   init.tensor           [3, 320, 256]    2 * v / 255 - 1 of init.rgb
#   mask.rgba             [320, 256, 4]    u8, the decoded mask
#   mask.gray             [320, 256]       u8, (19595 R + 38470 G + 7471 B + 32768) >> 16
#   mask.blurred          [320, 256]       u8, Pillow's GaussianBlur(4) of mask.gray: the `a` of the paste
#   mask.binary           [320, 256]       1.0 where mask.blurred >= 128, else 0.0
#   mask.latent           [40, 32]         mask.binary at pixel (8x, 8y)
#   masked.tensor         [3, 320, 256]    init.tensor with 0 where mask.binary is 1
#   blur.rR               [320, 256]       u8, GaussianBlur(R) of mask.gray, for R in 0, 1, 2, 3, 4, 8, 64
#   prompt_embeds         [2, 77, 2048]    the encoding of clip_skip 2; pooled [2, 1280]; add_time_ids [6]
#   schedule.cases        [K, 2]           f64 (steps, strength); schedule.n [K] and schedule.begin [K] are i64,
#                                          from diffusers' get_timesteps, the same for each sampler. n 0 is a 400.
#   vae_enc.in            [1, 3, 320, 256] init.tensor as the encoder takes it
#   vae_enc.conv_in, vae_enc.down0..3, vae_enc.mid, vae_enc.conv_norm_out, vae_enc.conv_out
#                                          the stages of the VAE encoder of base; conv_norm_out is before the SiLU
#   vae_enc.quant_conv    [1, 8, 40, 32]   the moments: the mean in channels 0 to 3, the raw logvar in 4 to 7
#
# img2img.safetensors, inpaint.safetensors, inpaint9.safetensors: each key is S.T.name. S is the sampler, as in
# tools/sdxl_golden.sh. T is the strength in percent: s50, s99 or s100. n is the count of steps that run.
#   S.T.n, S.T.begin        [1]                 i64, n = int(4 * strength) and steps - n; S.T.strength [1] is f64
#   S.T.sigmas              [5]                 the whole schedule and its final sigma; S.T.timesteps [4]
#   S.T.run_sigmas          [n + 1]             the sigmas from the begin index; S.T.run_timesteps [n]
#   S.T.noise.encode        [1, 4, H, W]        the draws above; noise.step is [D, 1, 4, H, W]
#   S.T.noise.start, S.T.noise.encode_masked, S.T.noise.step
#   S.T.encode.mean         [1, 4, H, W]        the moments of the VAE encoder for the image; encode.logvar is raw,
#   S.T.encode.logvar                           before the clamp to [-30, 20]
#   S.T.encode_masked.mean, S.T.encode_masked.logvar   the same for the masked image
#   S.T.image_latent        [1, 4, H, W]        0.13025 * (mean + exp(0.5 * clamp(logvar, -30, 20)) * z)
#   S.T.masked_latent       [1, 4, H, W]        the same for the masked image, with the draw encode_masked
#   S.T.latents             [n + 1, 1, 4, H, W] the latent before each step, then after the last step. Row 0 is
#                                               add_noise(image_latent, start) at the begin index, or
#                                               start * init_noise_sigma for inpainting at strength 1.0
#   S.T.unet_in             [n, 2, C, H, W]     the UNet input of each step. C is 9 for inpaint9: the scaled latent,
#                                               the latent mask, the masked latent
#   S.T.unet_out            [n, 2, 4, H, W]     the noise prediction, negative first
#   S.T.guided              [n, 1, 4, H, W]     negative + 7 * (prompt - negative)
#   S.T.stepped             [n, 1, 4, H, W]     the output of the sampler, before the blend
#   S.T.known               [n, 1, 4, H, W]     inpaint only: add_noise(image_latent, start) at the next index;
#                                               image_latent itself after the last step. After step i the latent is
#                                               (1 - mask.latent) * known[i] + mask.latent * stepped[i]
#   S.T.final_latent        [1, 4, H, W]        the latent after the last step, as plain diffusers gives it
#   S.T.image               [3, 320, 256]       the VAE decode, in [-1, 1] before the clamp, with no paste
#   S.T.rgb                 [320, 256, 3]       u8 RGB: round(clamp(image / 2 + 0.5, 0, 1) * 255)
#   S.T.paste               [320, 256, 3]       inpaint and inpaint9: (a * rgb + (255 - a) * init.rgb + 127) / 255
#                                               in integers, with a = mask.blurred (spec 11.9b, "The paste")
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${REF_PYTHON:-$HOME/.venvs/sdxl-ref/bin/python}"
MODEL="${SDXL_MODEL_DIR:?set SDXL_MODEL_DIR to the SDXL base 1.0 checkpoint directory}"
INPAINT="${SDXL_INPAINT_MODEL_DIR:?set SDXL_INPAINT_MODEL_DIR to the SDXL inpainting 0.1 checkpoint directory}"
OUT="${SDXL_GOLDEN_DIR:-target/sdxl-golden}"
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
nice -n 19 ionice -c3 "$PY" -I tools/sdxl_inpaint_reference.py --model "$MODEL" --inpaint-model "$INPAINT" --out "$OUT"
