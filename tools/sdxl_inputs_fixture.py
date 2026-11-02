# SPDX-License-Identifier: MIT
"""Make the input fixture of the SDXL family library (spec 11.9b; milestone 20).

The tool writes crates/nagaya-text-sdxl/tests/fixtures/inputs.json. It holds small PNG files and what Pillow makes
of them:

- `pngs`: each PNG of a kind that the nagaya reads, with its pixels after `convert("RGB")` and after `convert("L")`.
  diffusers takes an image with `convert("RGB")` and a mask with `convert("L")` (`image_processor.py:250` and `:267`).
- `refused`: each PNG of a kind that the nagaya refuses: 16 bits, or fewer than 8 bits in a sample.
- `blurs`: gray masks and their `ImageFilter.GaussianBlur(radius)` for some radii, as diffusers' `blur` makes them
  (`image_processor.py:284`). The mask `sweep` has each radius from 0 to 64.
- `unit`: the f32 bits of each byte value after diffusers' `pil_to_numpy` and `normalize`, and `binary`: each byte
  value after `binarize` (`image_processor.py:167`, `:220` and `:536-537`).
- `latent`: a binary mask and its latent mask, from torch's `interpolate` in the mode `nearest`
  (`inpaint.py:862-864`).

Each byte string is hex. The images are small and of odd sizes, so the edges of the box blur get tested. A radius
larger than the image tests the other branch of Pillow's line blur.

The tool needs Pillow, numpy, torch and diffusers. It uses no GPU and little RAM. Run it with the venv of the reference (Pillow 13.0.0):

    PYTHONDONTWRITEBYTECODE=1 ~/.venvs/sdxl-ref/bin/python -I tools/sdxl_inputs_fixture.py [--out DIR]
"""
import argparse
import io
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import PIL  # noqa: E402
import torch  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "crates/nagaya-text-sdxl/tests/fixtures"

RADII = [0, 1, 2, 3, 4, 5, 8, 16, 33, 64]

# Each `mask_blur` of a request (spec 11.9b), for one small mask.
SWEEP = list(range(65))


def png_bytes(image, **params):
    out = io.BytesIO()
    image.save(out, format="PNG", **params)
    return out.getvalue()


def noise(rng, count):
    return bytes(rng.randrange(256) for _ in range(count))


def pngs(rng):
    """The PNGs that the nagaya reads, each with 8 bits in a sample."""
    cases = []
    gray = Image.frombytes("L", (13, 7), noise(rng, 13 * 7))
    cases.append(("gray", gray, {}))
    cases.append(("gray_alpha", Image.frombytes("LA", (13, 7), noise(rng, 13 * 7 * 2)), {}))
    cases.append(("rgb", Image.frombytes("RGB", (37, 23), noise(rng, 37 * 23 * 3)), {}))
    cases.append(("rgba", Image.frombytes("RGBA", (37, 23), noise(rng, 37 * 23 * 4)), {}))
    # A palette of 256 colors, so Pillow writes 8 bits for each index.
    palette = Image.frombytes("P", (17, 11), noise(rng, 17 * 11))
    palette.putpalette(noise(rng, 256 * 3))
    cases.append(("palette", palette, {}))
    cases.append(("palette_trns", palette.copy(), {"transparency": bytes([0, 128, 255] * 10)}))
    cases.append(("gray_trns", gray.copy(), {"transparency": 5}))
    rgb_trns = Image.frombytes("RGB", (9, 5), noise(rng, 9 * 5 * 3))
    cases.append(("rgb_trns", rgb_trns, {"transparency": rgb_trns.getpixel((0, 0))}))
    out = []
    for name, image, params in cases:
        data = png_bytes(image, **params)
        read = Image.open(io.BytesIO(data))
        read.load()
        out.append({
            "name": name,
            "mode": read.mode,
            "width": read.width,
            "height": read.height,
            "png": data.hex(),
            "rgb": read.convert("RGB").tobytes().hex(),
            "gray": read.convert("L").tobytes().hex(),
        })
    return out


def refused(rng):
    """The PNGs that the nagaya refuses."""
    cases = []
    deep = Image.frombytes("I;16", (5, 3), noise(rng, 5 * 3 * 2))
    cases.append(("gray16", deep))
    cases.append(("bilevel", Image.frombytes("L", (8, 4), noise(rng, 32)).convert("1")))
    # A palette of 4 colors, so Pillow writes 2 bits for each index.
    small = Image.frombytes("P", (8, 4), bytes(b % 4 for b in noise(rng, 32)))
    small.putpalette(noise(rng, 4 * 3))
    cases.append(("palette2", small))
    out = []
    for name, image in cases:
        data = png_bytes(image)
        out.append({"name": name, "png": data.hex(), "depth": data[24]})
    return out


def blurs(rng):
    """Gray masks and their Gaussian blurs."""
    masks = []
    shapes = Image.new("L", (37, 23), 0)
    draw = ImageDraw.Draw(shapes)
    draw.rectangle((4, 3, 15, 12), fill=255)
    draw.ellipse((18, 6, 33, 20), fill=255)
    masks.append(("shapes", shapes))
    masks.append(("noise", Image.frombytes("L", (13, 7), noise(rng, 13 * 7))))
    masks.append(("tiny", Image.frombytes("L", (5, 3), noise(rng, 5 * 3))))
    band = Image.new("L", (40, 24), 0)
    ImageDraw.Draw(band).rectangle((0, 8, 39, 15), fill=255)
    masks.append(("band", band))
    masks = [(name, mask, RADII) for name, mask in masks]
    sweep = Image.new("L", (16, 9), 0)
    ImageDraw.Draw(sweep).ellipse((3, 1, 12, 7), fill=255)
    masks.append(("sweep", sweep, SWEEP))
    out = []
    for name, mask, radii in masks:
        out.append({
            "name": name,
            "width": mask.width,
            "height": mask.height,
            "gray": mask.tobytes().hex(),
            "blurred": {str(r): mask.filter(ImageFilter.GaussianBlur(r)).tobytes().hex() for r in radii},
        })
    return out


def values():
    """The map of each byte value to f32, and its binary value, as diffusers makes them."""
    ramp = Image.frombytes("L", (256, 1), bytes(range(256)))
    array = VaeImageProcessor.pil_to_numpy(ramp)
    unit = VaeImageProcessor.normalize(array).astype(np.float32).reshape(-1)
    binary = VaeImageProcessor(do_binarize=True).binarize(VaeImageProcessor.pil_to_numpy(ramp)).reshape(-1)
    return [f"{int(v):08x}" for v in unit.view(np.uint32)], [int(v) for v in binary]


def latent(rng):
    """A binary mask of 40 by 24 pixels, and its latent mask of 5 by 3."""
    mask = np.array([rng.randrange(2) for _ in range(40 * 24)], dtype=np.float32).reshape(1, 1, 24, 40)
    small = torch.nn.functional.interpolate(torch.from_numpy(mask), size=(24 // 8, 40 // 8))
    return {
        "width": 40,
        "height": 24,
        "binary": [int(v) for v in mask.reshape(-1)],
        "latent": [int(v) for v in small.numpy().reshape(-1)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rng = random.Random(20261102)
    fixture = {
        "pillow": PIL.__version__,
        "pngs": pngs(rng),
        "refused": refused(rng),
        "blurs": blurs(rng),
    }
    fixture["unit"], fixture["binary"] = values()
    fixture["latent"] = latent(rng)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "inputs.json"
    path.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {path} with Pillow {PIL.__version__}")


if __name__ == "__main__":
    main()
