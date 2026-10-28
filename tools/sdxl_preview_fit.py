# SPDX-License-Identifier: MIT
"""Fit the linear preview of SDXL latents (spec 11.9): RGB = A @ latent + b, by least squares.

The tool makes a few images with diffusers on the CPU at 512 by 512, decodes their final latents with the stock VAE,
and averages each 8 by 8 block of the image in [0, 1]. Each latent pixel and its block give one sample. It prints the
3 x 4 matrix A, the bias b, and the error of the fit, for `PREVIEW` in src/host.rs of nagaya-mk-sdxl-bmg.

    HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 ~/.venvs/sdxl-ref/bin/python -I tools/sdxl_preview_fit.py \
        --model DIR
"""
import argparse

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline

PROMPTS = [
    "a red fox in a snowy forest, watercolor",
    "a bowl of lemons on a blue table, photo",
    "a neon city street at night, rain",
    "green hills under a cloudy sky, oil painting",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model, variant="fp16", torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    xs, ys = [], []
    for i, prompt in enumerate(PROMPTS):
        latent = pipe(prompt, width=512, height=512, num_inference_steps=15, guidance_scale=7.0,
                      generator=torch.Generator().manual_seed(i), output_type="latent").images
        image = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)[0]
        blocks = image.reshape(3, 64, 8, 64, 8).mean(dim=(2, 4))
        xs.append(latent[0].reshape(4, -1).T.numpy())
        ys.append(blocks.reshape(3, -1).T.numpy())
        print(f"image {i} done", flush=True)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fit = design @ coef
    print("A (rows R, G, B; columns the 4 latent channels):")
    for row in coef[:4].T:
        print("    [" + ", ".join(f"{v:.4f}" for v in row) + "],")
    print("b:", "[" + ", ".join(f"{v:.4f}" for v in coef[4]) + "]")
    print(f"mean absolute error {np.abs(fit - y).mean():.4f} of the range [0, 1], over {len(x)} samples")


if __name__ == "__main__":
    main()
