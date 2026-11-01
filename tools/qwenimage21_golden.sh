#!/usr/bin/env bash
# Writes the reference files of the image oracle (spec 8.3, 11.9) for Qwen-Image 2.1 with the FUSAL LoRA.
#
# The reference is diffusers, on the CPU in float32 from the bf16 weights. It loads one model at a time, and its peak
# is about 33 GiB of RAM. Run it with nagaya.service stopped and with no GPU job (AGENTS.md, milestone 22).
#
#   QWENIMAGE21_MODEL_DIR=<Qwen-Image-2.1> QWENIMAGE21_FUSAL_DIR=<Qwen-Image-2.1-Anime-Fusal> \
#       tools/qwenimage21_golden.sh
#
# REF_PYTHON overrides the python (default ~/src/diffusepal/.venv/bin/python). The files go to
# $QWENIMAGE21_GOLDEN_DIR, or to target/qwenimage21-golden. Do not commit them. Each file carries the request, the
# template, the scheduler config, mu, the image shapes and the versions in its safetensors metadata.
#
# The request: prompt "fusal style. a red fox in a snowy forest, watercolor", 256 by 256, 4 steps, seed 1234, true CFG
# 1.0 (no negative prompt), the Heretic text encoder, and the FUSAL LoRA at weight 1.0. The image is small to keep the
# CPU time small. At 4 steps it may show no clear picture. The engine has its own random stream, so the start latent
# comes from torch.Generator().manual_seed(1234). FlowMatch Euler draws no other noise.
#
# H = W = 16 are the latent size, and N = 256 = H * W is the count of image tokens. L is the count of prompt tokens
# after the cut of the system tokens. Floats are f32.
#
# encode.safetensors: the text encoder.
#   ids                  [1, T]         i64 ids of the whole template, with the system tokens
#   drop                 [1]            the count of leading system tokens that the cut removes
#   prompt_embeds        [1, L, 4096]   the input of the encoder's final norm, after the cut
#   prompt_embeds_bf16   [1, L, 4096]   the same from the encoder in bf16, as diffusepal runs it: the baseline
#                                       of the conditioning gate (spec 8.3)
#
# steps.safetensors: the oracle.
#   start                [1, 1, 64, H, W]   the start noise
#   packed               [1, N, 64]         the start noise as tokens: the flatten of H and W
#   sigmas               [5]                the sigmas of the steps and the final 0; the last step's sigma
#                                           is shift_terminal, 0.02
#   timesteps            [4, 1]             the timestep that the transformer gets at each step (t / 1000)
#   latents              [5, 1, N, 64]      the latent before each step, then after the last step
#   transformer_out      [4, 1, N, 64]      the transformer output of each step (the velocity)
#   final_latent         [1, N, 64]         the latent after the last step
#   image                [4, 256, 256]      the VAE decode, RGBA in [-1, 1] after the VAE's clamp
#   rgba                 [256, 256, 4]      u8 RGBA: round(clamp(image / 2 + 0.5, 0, 1) * 255)
#   transformer_out_bf16 [4, 1, N, 64]      each call of the transformer again in bf16, from the f32 inputs
#                                           rounded to bf16: the baseline of the step gate
#   latents_bf16         [5, 1, N, 64]      the whole run in bf16 from the same start: the baseline of the run gate
#   rgba_bf16            [256, 256, 4]      the decode of final_latent in bf16, through the pipeline's postprocess:
#                                           the baseline of the decode gate
#
# parts.safetensors: references for kernel bring-up.
#   enc.hidden0, enc.hidden1, enc.hidden18, enc.hidden36
#                        [1, L, 4096]   the token embedding and the outputs of layers 0, 17 and 35, after the cut
#   enc.hidden_all_tokens_last
#                        [1, T, 4096]   the input of the final norm, with the system tokens
#   tr.img_in, tr.txt_in, tr.time_text_embed, tr.modulation, tr.block0, tr.block15, tr.block31, tr.norm_out,
#   tr.proj_out         the outputs of these transformer modules at step 0. time_text_embed and modulation
#                        have 2 rows: the timestep of the step, then t = 0 for the condition tokens
#   tr.step0_no_lora     [1, N, 64]         step 0 with the LoRA at weight 0
#   tr.timestep0         [1]
#   vae.in               [1, 64, 1, H, W]   the final latent, scaled by latents_std and latents_mean
#   vae.post_quant_conv, vae.conv_in, vae.mid_block, vae.up_blocks0..4, vae.norm_out, vae.conv_out
#                        the stages of the VAE decoder; conv_out has 4 channels, RGBA
#
# rgba.npy: the same RGBA bytes, for a look with any viewer of numpy files.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${REF_PYTHON:-$HOME/src/diffusepal/.venv/bin/python}"
MODEL="${QWENIMAGE21_MODEL_DIR:?set QWENIMAGE21_MODEL_DIR to the Qwen-Image-2.1 directory}"
FUSAL="${QWENIMAGE21_FUSAL_DIR:?set QWENIMAGE21_FUSAL_DIR to the Qwen-Image-2.1-Anime-Fusal directory}"
OUT="${QWENIMAGE21_GOLDEN_DIR:-target/qwenimage21-golden}"

if systemctl --user is-active --quiet nagaya.service; then
    echo "qwenimage21_golden: nagaya.service is active. Stop it first: the reference needs 33 GiB of RAM." >&2
    exit 1
fi
avail_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
need_kib=$((40 * 1024 * 1024))
if [ "$avail_kib" -lt "$need_kib" ]; then
    echo "qwenimage21_golden: MemAvailable is ${avail_kib} KiB, less than 40 GiB. Free memory first." >&2
    exit 1
fi

mkdir -p "$OUT"
HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
    nice -n 10 "$PY" -I tools/qwenimage21_reference.py --model "$MODEL" --fusal "$FUSAL" --out "$OUT"
