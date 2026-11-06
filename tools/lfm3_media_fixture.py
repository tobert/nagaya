# SPDX-License-Identifier: MIT
"""Make the media fixture of the LFM3 family library (spec 11.11, "The media of a request").

The tool writes these files into crates/nagaya-text-lfm3/tests/fixtures:

- media.json: the results of the reference processors of LFM3-Omni-2B on fixed inputs.
  - images: for each size, the processor's resized size, grid and position count, and the SHA-256 of the resized
    RGB bytes and of the f32 pixel values. The input is the pattern image below, so the fixture holds no pixels.
  - decodes: for each small image file of this tool, the SHA-256 of the RGB bytes that Pillow's convert("RGB")
    gives. A JPEG also has a PNG of Pillow's RGB, because a JPEG decoder of another library can differ.
  - clips: for each short clip of the pattern samples below, the mel frame count, the position count, and the mel
    features of the reference feature extractor, bins first.
- gray.png, gray_alpha.png, palette.png, rgba.png, photo.jpg, photo_pil.png: the small image files of "decodes".

The pattern image has the channel c of the pixel at row y and column x:

  noise(y, x, c) = ((x * 2654435761 + y * 40503 + c * 97 + x * y * 7) mod 2^32) >> 24
  smooth(y, x, c) = (x + y + 85 * c) mod 256
  value = noise on the squares of 16 pixels where (x / 16 + y / 16) is even, else smooth

The pattern clip has the sample i = ((i * 7919 + 12345) mod 65536) - 32768 as PCM in 16 bits.

The image processor and the feature extractor are the checkpoint's own code (`trust_remote_code`). The tool loads no
weights and uses no GPU. It reads the Omni folder from --omni or LFM3_OMNI_MODEL_DIR; the code holds no default
path. Run it with the reference venv:

    PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 nice -n 19 ionice -c3 ~/.venvs/nagaya-lfm3-ref/bin/python -I \
        tools/lfm3_media_fixture.py [--omni DIR] [--out DIR]
"""
import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "crates/nagaya-text-lfm3/tests/fixtures"

# The 9 sizes of the resize table of the oracle (facts.json), and one small size with odd sides.
SIZES = [(96, 160), (300, 420), (256, 256), (64, 64), (480, 640), (1080, 1920), (3000, 4000), (100, 5000),
         (2048, 2048), (37, 53)]
# Clips below one window, at one window, and just above it.
CLIPS = [1, 250, 399, 400, 401, 1000]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pattern_image(height, width):
    import numpy as np

    y, x = np.mgrid[0:height, 0:width].astype(np.uint64)
    planes = []
    for c in range(3):
        noise = ((x * 2654435761 + y * 40503 + c * 97 + x * y * 7) % (1 << 32)) >> 24
        smooth = (x + y + 85 * c) % 256
        squares = ((x // 16 + y // 16) % 2) == 0
        planes.append(np.where(squares, noise, smooth))
    return np.stack(planes, axis=-1).astype(np.uint8)


def pattern_clip(n):
    import numpy as np

    i = np.arange(n, dtype=np.int64)
    return (((i * 7919 + 12345) % 65536) - 32768).astype(np.int16)


def images(processor):
    import torch
    from PIL import Image

    out = []
    for height, width in SIZES:
        rgb = pattern_image(height, width)
        inputs = processor(images=[Image.fromarray(rgb, mode="RGB")], return_tensors="pt")
        pixels, grid = inputs["pixel_values"], inputs["image_grid_thw"]
        t, gh, gw = (int(v) for v in grid[0])
        resized = processor.resize(
            torch.from_numpy(rgb).permute(2, 0, 1)[None],
            size=processor.size,
            resample=processor.resample,
            factor=processor.patch_size * processor.merge_size,
        )[0]
        assert tuple(resized.shape) == (3, 16 * gh, 16 * gw), "the resize gives the size of the grid"
        out.append({
            "source": [height, width],
            "resized": [16 * gh, 16 * gw],
            "grid": [t, gh, gw],
            "positions": t * gh * gw // 4,
            "resized_sha256": sha256(resized.permute(1, 2, 0).contiguous().numpy().tobytes()),
            "pixel_values_sha256": sha256(pixels.float().contiguous().numpy().tobytes()),
        })
        print(f"image {height}x{width} -> {16 * gh}x{16 * gw}", file=sys.stderr)
    return out


def decodes(out_dir):
    import numpy as np
    from PIL import Image

    rgb = pattern_image(24, 40)
    files = {}
    files["gray.png"] = Image.fromarray(rgb[:, :, 0], mode="L")
    gray_alpha = Image.fromarray(np.stack([rgb[:, :, 0], rgb[:, :, 1]], axis=-1), mode="LA")
    files["gray_alpha.png"] = gray_alpha
    files["rgba.png"] = Image.fromarray(np.concatenate([rgb, rgb[:, :, :1]], axis=-1), mode="RGBA")
    files["palette.png"] = Image.fromarray(rgb, mode="RGB").quantize(colors=37)
    out = []
    for name, image in files.items():
        path = out_dir / name
        image.save(path, format="PNG")
        decoded = Image.open(path).convert("RGB")
        out.append({"file": name, "mode": Image.open(path).mode, "rgb_sha256": sha256(np.asarray(decoded).tobytes())})
    photo = pattern_image(48, 64)
    buffer = io.BytesIO()
    Image.fromarray(photo, mode="RGB").save(buffer, format="JPEG", quality=90)
    (out_dir / "photo.jpg").write_bytes(buffer.getvalue())
    decoded = Image.open(out_dir / "photo.jpg").convert("RGB")
    decoded.save(out_dir / "photo_pil.png", format="PNG")
    out.append({"file": "photo.jpg", "mode": "RGB", "rgb_sha256": sha256(np.asarray(decoded).tobytes()),
                "pillow_png": "photo_pil.png"})
    return out


def clips(extractor):
    import numpy as np

    out = []
    for n in CLIPS:
        samples = pattern_clip(n).astype(np.float32) / 32768.0
        features = extractor([samples], sampling_rate=16000, return_tensors="pt")
        mel = features["input_features"][0].float()
        frames = int(features["feature_attention_mask"].sum())
        assert frames == mel.shape[1] == extractor.num_frames(n)
        out.append({
            "samples": n,
            "frames": frames,
            "positions": extractor.num_encoder_tokens(frames),
            "mel": [[float(v) for v in row] for row in mel.tolist()],
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--omni", default=os.environ.get("LFM3_OMNI_MODEL_DIR"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if not args.omni:
        sys.exit("set --omni or LFM3_OMNI_MODEL_DIR to the folder of LFM3-Omni-2B")
    # transformers copies the remote code into HF_MODULES_CACHE. Keep it out of ~/.cache.
    os.environ.setdefault("HF_MODULES_CACHE", tempfile.mkdtemp(prefix="lfm3-media-modules-"))

    import PIL
    import torch
    import torchvision
    import transformers
    from transformers import AutoFeatureExtractor, AutoImageProcessor

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(args.omni, trust_remote_code=True)
    extractor = AutoFeatureExtractor.from_pretrained(args.omni, trust_remote_code=True)
    fixture = {
        "format": 1,
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
            "pillow": PIL.__version__,
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        },
        "images": images(processor),
        "decodes": decodes(out_dir),
        "clips": clips(extractor),
    }
    (out_dir / "media.json").write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {out_dir / 'media.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
