#!/usr/bin/env python3
"""f32 CPU reference for the VAE encoder of SDXL base 1.0 (milestone 20 step 3, spec 11.9b "The image latent").

Run it through tools/sdxl_vae_encode_golden.sh. The direct command is:

  HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 ~/.venvs/sdxl-ref/bin/python -I \
      tools/sdxl_vae_encode_reference.py --model "$SDXL_MODEL_DIR" --out target/sdxl-golden

The script uses diffusers 0.42.1 in ~/.venvs/sdxl-ref, on the CPU only. It loads `vae/` of the checkpoint in f32,
as the engine does. The input is a fixed RGB image that the script makes, so it needs no file. The tensor names,
shapes and meanings are in tools/sdxl_vae_encode_golden.sh.

The engine draws `z` from its own random stream (spec 11.9b), so the script saves the `z` of torch with the latent.
The test gives that `z` to the host form of the draw and compares the result with the latent of diffusers.

The script stops with an AssertionError when a self check fails. The checks:

  - the image of VaeImageProcessor is 2 * (v / 255) - 1 of the bytes, bit for bit;
  - the moments split into the mean and the clamped log variance of `latent_dist`;
  - `retrieve_latents` of img2img with the generator gives (mean + std * z) * scaling_factor, with the saved z.
"""

import argparse
import os
import time

import numpy as np
import PIL.Image
import torch
from safetensors.torch import save_file

import diffusers
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import retrieve_latents
from diffusers.utils.torch_utils import randn_tensor

# The size is a multiple of 64, as the engine checks, and is not square, so a transposed axis fails the test.
WIDTH, HEIGHT = 128, 192
SEED = 1234


def check(ok, what):
    """Stop with the name of the self check that failed."""
    if not ok:
        raise AssertionError(f"self check failed: {what}")


def image_bytes():
    """A fixed RGB image of HEIGHT by WIDTH: smooth ramps, a disc and a band of fine stripes, so each level of the
    encoder sees structure."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    r = (255 * x / (WIDTH - 1)).astype(np.float64)
    g = (255 * y / (HEIGHT - 1)).astype(np.float64)
    b = 128 + 127 * np.sin(x / 5.0) * np.cos(y / 7.0)
    disc = (x - 80) ** 2 + (y - 60) ** 2 < 30**2
    r[disc], g[disc], b[disc] = 230, 40, 30
    stripes = (y > 140) & (y < 170)
    g[stripes] = np.where(x[stripes] % 4 < 2, 250, 10)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)
    started = time.time()

    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    vae.eval()
    rgb = image_bytes()
    processor = VaeImageProcessor(vae_scale_factor=8)
    image = processor.preprocess(PIL.Image.fromarray(rgb, "RGB"))
    check(tuple(image.shape) == (1, 3, HEIGHT, WIDTH), f"the image shape {tuple(image.shape)}")
    mapped = 2.0 * (torch.from_numpy(rgb).float() / 255.0) - 1.0
    check(torch.equal(image[0].permute(1, 2, 0), mapped), "the image is 2 * (v / 255) - 1 of the bytes")

    # The encoder and quant_conv (autoencoder_kl.py:158-168), then the distribution of vae.py:690-709.
    moments = vae._encode(image)
    dist = vae.encode(image).latent_dist
    mean, logvar = torch.chunk(moments, 2, dim=1)
    check(torch.equal(dist.mean, mean), "the mean is the first 4 channels of the moments")
    check(torch.equal(dist.logvar, torch.clamp(logvar, -30.0, 20.0)), "the clamped log variance")

    # img2img.py:740 and :751: retrieve_latents with a generator samples, then the scaling factor.
    latent = retrieve_latents(vae.encode(image), generator=torch.Generator().manual_seed(SEED))
    latent = vae.config.scaling_factor * latent
    z = randn_tensor(dist.mean.shape, generator=torch.Generator().manual_seed(SEED), dtype=torch.float32)
    check(torch.equal(latent, vae.config.scaling_factor * (dist.mean + dist.std * z)), "the latent with the saved z")

    tensors = {
        "rgb": torch.from_numpy(rgb),
        "image": image,
        "moments": moments,
        "mean": dist.mean,
        "logvar": dist.logvar,
        "z": z,
        "latent": latent,
    }
    meta = {
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "seed": str(SEED),
        "diffusers": diffusers.__version__,
        "torch": torch.__version__,
        "vae_scaling_factor": str(vae.config.scaling_factor),
    }
    path = os.path.join(args.out, "vae_encode.safetensors")
    save_file({k: v.contiguous() for k, v in tensors.items()}, path, metadata=meta)
    print(f"wrote {path} in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
