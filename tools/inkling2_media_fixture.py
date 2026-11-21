# SPDX-License-Identifier: MIT
"""Make the media fixture of the Inkling 2 family library (engine-abi "The images of Inkling 2").

The tool writes these files into crates/nagaya-text-inkling2/tests/fixtures:

- media.json: for each small image file below, the grid of the reference processor, and the SHA-256 of the RGB
  bytes after its convert step and of its f32 pixel values, little-endian. A JPEG also has a PNG of Pillow's RGB,
  because a JPEG decoder of another library can differ.
- gray.png, gray_alpha.png, palette.png, palette_trns.png, rgb.png, rgba.png, photo.jpg, photo_pil.png: the files.

Each image is 83 by 45 pixels, so no side is a multiple of 40 and the processor pads both sides. The pattern has the
channel c of the pixel at row y and column x:

  noise(y, x, c) = ((x * 2654435761 + y * 40503 + c * 97 + x * y * 7) mod 2^32) >> 24
  smooth(y, x, c) = (x + y + 85 * c) mod 256
  value = noise on the squares of 16 pixels where (x / 16 + y / 16) is even, else smooth

The alpha channel is (x * 3 + y * 5) mod 256, so it holds 0, 255 and the values between them.

The processor is transformers `Inkling2ImageProcessor` at 046bdb8cc2. The tool loads no weights, reads no model
folder and uses no GPU. Run it with the reference venv:

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 <reference venv>/bin/python -I \
        tools/inkling2_media_fixture.py [--out DIR]
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "crates/nagaya-text-inkling2/tests/fixtures"
WIDTH, HEIGHT = 83, 45


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pattern(height, width):
    import numpy as np

    y, x = np.mgrid[0:height, 0:width].astype(np.uint64)
    planes = []
    for c in range(3):
        noise = ((x * 2654435761 + y * 40503 + c * 97 + x * y * 7) % (1 << 32)) >> 24
        smooth = (x + y + 85 * c) % 256
        squares = ((x // 16 + y // 16) % 2) == 0
        planes.append(np.where(squares, noise, smooth))
    rgb = np.stack(planes, axis=-1).astype(np.uint8)
    alpha = ((x * 3 + y * 5) % 256).astype(np.uint8)
    return rgb, alpha


def files():
    import numpy as np
    from PIL import Image

    rgb, alpha = pattern(HEIGHT, WIDTH)
    gray = rgb[..., 0]
    out = {}

    def png(image, **kw):
        buf = io.BytesIO()
        image.save(buf, format="PNG", **kw)
        return buf.getvalue()

    out["gray.png"] = png(Image.fromarray(gray, "L"))
    out["gray_alpha.png"] = png(Image.fromarray(np.stack([gray, alpha], axis=-1), "LA"))
    palette = Image.fromarray(rgb, "RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    out["palette.png"] = png(palette)
    trns = palette.copy()
    out["palette_trns.png"] = png(trns, transparency=bytes((i * 4) % 256 for i in range(64)))
    out["rgb.png"] = png(Image.fromarray(rgb, "RGB"))
    out["rgba.png"] = png(Image.fromarray(np.concatenate([rgb, alpha[..., None]], axis=-1), "RGBA"))
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="JPEG", quality=90)
    out["photo.jpg"] = buf.getvalue()
    out["photo_pil.png"] = png(Image.open(io.BytesIO(out["photo.jpg"])).convert("RGB"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    import numpy as np
    from PIL import Image
    from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor

    processor = Inkling2ImageProcessor()
    args.out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, data in files().items():
        (args.out / name).write_bytes(data)
        image = Image.open(io.BytesIO(data))
        image.load()
        mode = "RGBA" if image.has_transparency_data else "RGB"
        converted = np.asarray(image.convert(mode))
        if mode == "RGBA":
            # The convert step of the processor: the checkerboard under the alpha (image_processing_inkling2.py:63).
            import torch

            tensor = torch.from_numpy(converted).permute(2, 0, 1).contiguous()
            converted = processor.convert_to_rgb(tensor).permute(1, 2, 0).numpy()
        result = processor(images=[image], return_tensors="pt")
        values = result["pixel_values"].float().numpy().astype("<f4")
        entries[name] = {
            "width": image.width,
            "height": image.height,
            "alpha": mode == "RGBA",
            "grid": result["image_grid_thw"][0].tolist(),
            "rgb_sha256": sha256(np.ascontiguousarray(converted, dtype=np.uint8).tobytes()),
            "pixel_values_sha256": sha256(values.tobytes()),
        }
        print(f"{name}: {image.width}x{image.height} {mode} grid {entries[name]['grid']}", file=sys.stderr)
    fixture = {
        "source": "transformers Inkling2ImageProcessor 046bdb8cc2, tools/inkling2_media_fixture.py",
        "images": entries,
    }
    (args.out / "media.json").write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
