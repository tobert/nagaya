"""Write the fixture of the Inkling 2 weight formats for `nagaya-mk-inkling2-bmg`.

It reads small slices of the real tensors of Inkling 2 Mini with `safetensors.safe_open`. It expands each slice with
the reference op itself: `Inkling2LoopDequantizeExperts.convert` of transformers (commit 046bdb8cc2), on a slice
that holds one expert and a few rows. The op needs no whole expert, because it works on any slice of the three axes.
The op for `w13` is followed by `Interleave(dim=1).convert`, as in `conversion_mapping.py`.

Synthetic cases go through the same op. They reach the edges that a few real rows do not: every E2M1 code, the
E4M3 codes and the UE8M0 scales 0 and 254.

Run it with the venv of the reference:
    PYTHONDONTWRITEBYTECODE=1 ~/.venvs/inkling2-ref/bin/python -I tools/inkling2_formats_fixture.py MODEL_DIR OUT.json
"""

import json
import sys
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave

OP = Inkling2LoopDequantizeExperts()
PATTERNS = ["w", "s", "a", "d"]


def expand(weight, scale, amax=None, denominator=None):
    """Run the reference op. A missing `amax` gives the MXFP8 form."""
    got = {"w": weight, "s": scale}
    if amax is not None:
        got["a"], got["d"] = amax, denominator
    return OP.convert(got, source_patterns=PATTERNS, target_patterns=["t"], model=None)["t"]


def interleave(tensor):
    return Interleave(dim=1).convert({"t": tensor}, source_patterns=["t"], target_patterns=["t"])["t"]


def bits(tensor):
    return [f"{v & 0xFFFF:04x}" for v in tensor.contiguous().view(torch.int16).flatten().tolist()]


def raw(tensor):
    return bytes(tensor.contiguous().view(torch.uint8).flatten().tolist()).hex()


def real_case(f, layer, kind, name, expert, rows, fmt):
    base = f"model.llm.layers.{layer}.mlp.experts.{name}"
    r0, r1 = rows
    weight = f.get_slice(base)[expert : expert + 1, r0:r1, :]
    scale = f.get_slice(base + "_scale")[expert : expert + 1, r0:r1, :]
    case = {
        "format": fmt,
        "kind": kind,
        "source": {"tensor": base, "expert": expert, "rows": [r0, r1]},
        "weight": raw(weight),
        "scale": raw(scale),
    }
    if fmt == "nvfp4":
        amax = f.get_tensor(base + "_amax")[expert : expert + 1]
        denominator = f.get_tensor(base + "_outer_scale_denominator")
        out = expand(weight, scale, amax, denominator)
        case["amax"] = float(amax[0])
        case["amax_bits"] = f"{amax.view(torch.int32)[0].item() & 0xFFFFFFFF:08x}"
        case["denominator_bits"] = f"{denominator.view(torch.int32)[0].item() & 0xFFFFFFFF:08x}"
        # The loader casts each source of the converter to the dtype of the parameter (core_model_loading.py,
        # `_materialize_copy`). So `amax` and the denominator may reach the op rounded to bf16. Record both results.
        low = expand(weight, scale, amax.to(torch.bfloat16), denominator.to(torch.bfloat16))
        case["amax_bf16_bits"] = f"{amax.to(torch.bfloat16).view(torch.int16)[0].item() & 0xFFFF:04x}"
        case["denominator_bf16_bits"] = f"{denominator.to(torch.bfloat16).view(torch.int16)[0].item() & 0xFFFF:04x}"
        case["expected_loader_cast"] = bits(low)
    else:
        out = expand(weight, scale)
    case["expected"] = bits(out)
    if kind == "w13":
        case["expected_interleaved"] = bits(interleave(out))
    return case


def synthetic_nvfp4():
    # 16 codes x 16 columns = one block of 16 values per row. Row r uses the scale byte in SCALES[r].
    codes = torch.arange(16, dtype=torch.uint8)
    pair = codes[0::2] | (codes[1::2] << 4)  # codes 0..15 as 8 bytes, even element low
    scales = [0x00, 0x01, 0x08, 0x38, 0x77, 0x7E, 0x80, 0xB8, 0xFE, 0x30, 0x06, 0x07]
    weight = torch.cat([pair, pair.flip(0)]).repeat(len(scales), 1).reshape(1, len(scales), 16)
    scale = torch.tensor(scales, dtype=torch.uint8).reshape(1, len(scales), 1).view(torch.float8_e4m3fn)
    out = []
    for amax, den in ((6.0, 6.0), (448.0, 1.0), (1.0e-30, 3.0e4), (3.4e38, 1.0)):
        a, d = torch.tensor([amax], dtype=torch.float32), torch.tensor([den], dtype=torch.float32)
        weight2 = weight.reshape(1, len(scales), 16)
        # Two blocks of 16 values per row: 16 bytes of codes hold 32 values, so cols = 32 and 2 scales per row.
        scale2 = scale.repeat(1, 1, 2)
        res = expand(weight2, scale2, a, d)
        out.append(
            {
                "format": "nvfp4",
                "kind": "synthetic",
                "weight": raw(weight2),
                "scale": raw(scale2),
                "amax_bits": f"{a.view(torch.int32)[0].item() & 0xFFFFFFFF:08x}",
                "denominator_bits": f"{d.view(torch.int32)[0].item() & 0xFFFFFFFF:08x}",
                "expected": bits(res),
            }
        )
    return out


def synthetic_mxfp8():
    # 32 values per row with the codes of E4M3 that are finite; each row has the scale byte in SCALES[r].
    finite = [c for c in range(256) if c not in (0x7F, 0xFF)]
    scales = [0, 1, 126, 127, 128, 200, 253, 254]
    out = []
    for start in range(0, len(finite), 32):
        part = finite[start : start + 32]
        part += [0] * (32 - len(part))
        weight = torch.tensor(part, dtype=torch.uint8).repeat(len(scales), 1).reshape(1, len(scales), 32)
        weight = weight.view(torch.float8_e4m3fn)
        scale = torch.tensor(scales, dtype=torch.uint8).reshape(1, len(scales), 1)
        res = expand(weight, scale)
        out.append({"format": "mxfp8", "kind": "synthetic", "weight": raw(weight), "scale": raw(scale),
                    "expected": bits(res)})
    return out


def main():
    model_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    path = model_dir / "model-00001-of-00001.safetensors"
    cases = []
    with safe_open(path, framework="pt") as f:
        for layer, fmt in ((0, "nvfp4"), (20, "mxfp8")):
            cases.append(real_case(f, layer, "w13", "w13_weight", 0, (0, 4), fmt))
            cases.append(real_case(f, layer, "w13", "w13_weight", 127, (2300, 2304), fmt))
            cases.append(real_case(f, layer, "w2", "w2_weight", 0, (0, 2), fmt))
            cases.append(real_case(f, layer, "w2", "w2_weight", 127, (2302, 2304), fmt))
    cases += synthetic_nvfp4() + synthetic_mxfp8()
    doc = {
        "header": {
            "model": "thinkingmachines/Inkling-2-Mini",
            "revision": model_dir.name,
            "transformers": transformers.__version__,
            "transformers_commit": "046bdb8cc2",
            "reference": "Inkling2LoopDequantizeExperts.convert, then Interleave(dim=1) for w13",
            "layout": "weight and scale are raw bytes in hex; expected holds bf16 bits, row by row",
        },
        "cases": cases,
    }
    out_path.write_text(json.dumps(doc, separators=(",", ":")))


main()
