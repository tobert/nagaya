#!/usr/bin/env bash
# Writes the reference file of the VAE encoder of SDXL base 1.0 (milestone 20 step 3, spec 11.9b).
#
# The reference is diffusers 0.42.1, on the CPU in float32 from `vae/` of the checkpoint. It holds about 1 GB of RAM
# and runs in a few seconds. Do not run it with a GPU test.
#
#   SDXL_MODEL_DIR=<snapshot of stabilityai/stable-diffusion-xl-base-1.0> tools/sdxl_vae_encode_golden.sh
#
# REF_PYTHON overrides the python (default ~/.venvs/sdxl-ref/bin/python). The file goes to $SDXL_GOLDEN_DIR, or to
# target/sdxl-golden, beside the files of tools/sdxl_golden.sh. Do not commit it.
#
# The input is a fixed RGB image of 128 wide by 192 high that the script makes. h = 24 and w = 16 are the latent size.
# The seed of z is 1234, from torch.Generator().manual_seed(1234). Floats are f32.
#
# vae_encode.safetensors:
#   rgb       [192, 128, 3]  u8, the input image, row after row
#   image     [1, 3, 192, 128]  the input of the encoder: 2 * (v / 255) - 1
#   moments   [1, 8, 24, 16]    the output of quant_conv: the mean of 4 channels, then their log variance
#   mean      [1, 4, 24, 16]    latent_dist.mean
#   logvar    [1, 4, 24, 16]    latent_dist.logvar, clamped to -30 and 20
#   z         [1, 4, 24, 16]    the draw of latent_dist.sample
#   latent    [1, 4, 24, 16]    0.13025 * (mean + exp(0.5 * logvar) * z), as img2img.py:740-751 makes it
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${REF_PYTHON:-$HOME/.venvs/sdxl-ref/bin/python}"
MODEL="${SDXL_MODEL_DIR:?set SDXL_MODEL_DIR to the SDXL base 1.0 checkpoint directory}"
OUT="${SDXL_GOLDEN_DIR:-target/sdxl-golden}"
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
nice -n 19 "$PY" -I tools/sdxl_vae_encode_reference.py --model "$MODEL" --out "$OUT"
