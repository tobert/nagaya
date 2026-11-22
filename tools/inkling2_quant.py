# SPDX-License-Identifier: MIT
"""The activation quantizers of the Inkling 2 experts, in f32 on the CPU: the rules of the checkpoint.

The checkpoint requires the quantization of the expert inputs (`required_activation_quantization` in
`hf_quant_config.json`). transformers quantizes nothing, so `tools/inkling2_reference.py --quant` calls this module.
The rules are those of the Triton kernels of vLLM commit 186882d9e8 (`nvfp4.py` and `mxfp8.py`), as
`crates/nagaya-mk-inkling2-sm121/src/quant.rs` states them. Each function here runs the same f32 operations in the
same order as `quant.rs`: each one is a separate element-wise torch call, so nothing fuses into an FMA.
`tools/test_inkling2_quant.py` compares the bytes with the fixture of the vLLM kernels.

Where torch could differ from Rust, and how this module stays equal:

- A constant such as 1/1536 is a Python double, and `torch.tensor(..., dtype=float32)` rounds it once to f32.
  `quant.rs` does the same: the f64 quotient, rounded once.
- torch rounds an f32 to e4m3 by nearest even, but it turns a value above 448 into NaN. The code clamps to 448
  first, as `quant.rs` does. The test checks the cast against the exhaustive search of `quant.rs`.
- The code never sets the flush of denormals. A scale near 2^-127 and a reciprocal of FLT_MAX stay subnormal.

The module also gives the fake quantizer of the reference: it quantizes a row and then expands it to f32 as
`e2m1 * block_scale * outer` for NVFP4 (the outer scale is `amax * (1/1536)`), and as `e4m3 * 2^(byte - 127)` for MXFP8.
The expanded weights of the reference hold their exact values. Thus a f32 product of a fake quantized row and a
weight row is the quantized dot product, up to the order of the f32 sums.
"""

import json
import re
from pathlib import Path

import torch

F32 = torch.float32
BLOCK_NVFP4 = 16
BLOCK_MXFP8 = 32
E4M3_MAX = 448.0
FP32_MAX = torch.finfo(F32).max
# The constants of the kernels, as f32. A Python float is an f64, and `torch.tensor` rounds it once.
RECIP_1536 = torch.tensor(1.0 / 1536.0, dtype=F32)
RECIP_6 = torch.tensor(1.0 / 6.0, dtype=F32)
RECIP_448 = torch.tensor(1.0 / 448.0, dtype=F32)
TOP_1536 = torch.tensor(1536.0, dtype=F32)
ONE = torch.tensor(1.0, dtype=F32)
QUARTER = torch.tensor(0.25, dtype=F32)
# The exact values of the e2m1 codes 0 to 15. Codes 8 to 15 are negative copies, and code 8 is -0.0.
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=F32)
SMALLEST_UE8M0 = 1 << 22  # the bits of 2^-127, the scale of the byte 0

NVFP4_RULE = {
    "amax_type": "per_token",
    "block_size": 16,
    "format": "nvfp4_e2m1",
    "rounding": "nearest",
    "scale_dtype": "float8_e4m3fn",
    "scale_selection_rule": "fouroversix_mae",
}
MXFP8_RULE = {"block_size": 32, "format": "mxfp8_e4m3", "rounding": "nearest", "scale_dtype": "ue8m0"}


def require_finite(x):
    """A value that is not finite is an error (spec 8.5), as in `quant.rs`."""
    if not torch.isfinite(x).all():
        raise ValueError("a non-finite f32 value reached the activation quantizer; the row is corrupt")


def e4m3_bytes(value):
    """The e4m3fn bytes of f32 values, nearest even, clamped to [-448, 448]. Returns uint8."""
    clamped = torch.clamp(value, -E4M3_MAX, E4M3_MAX)
    return clamped.to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_values(code):
    """The f32 values of e4m3fn bytes."""
    if bool(((code & 0x7F) == 0x7F).any()):
        raise ValueError("an E4M3FN code is NaN; the activation buffer is corrupt")
    return code.view(torch.float8_e4m3fn).to(F32)


def e2m1_codes(value):
    """The e2m1 codes (0 to 15) of f32 values: nearest even, saturated at 6.0, the sign bit kept (-0.0 is code 8)."""
    if bool(torch.isnan(value).any()):
        raise ValueError("a NaN reached the e2m1 encoder; the row is corrupt")
    magnitude = value.abs()
    code = (
        (magnitude > 0.25).to(torch.int64)
        + (magnitude >= 0.75)
        + (magnitude > 1.25)
        + (magnitude >= 1.75)
        + (magnitude > 2.5)
        + (magnitude >= 3.5)
        + (magnitude > 5.0)
    )
    return code + 8 * torch.signbit(value).to(torch.int64)


def tree_sum16(error):
    """The sum of 16 values on the last axis as a pairwise tree: adjacent pairs, then pairs of pairs."""
    while error.shape[-1] > 1:
        error = error[..., 0::2] + error[..., 1::2]
    return error[..., 0]


def quantize_nvfp4(x):
    """NVFP4 `fouroversix_mae` of f32 rows `[R, K]`: the e2m1 codes `[R, K]`, the e4m3 scale bytes `[R, K/16]`, the
    row amax `[R]`. Each row has its own amax (`per_token`). A block takes the scale of 4 only when its error is
    less than the error of the scale of 6.
    """
    x = x.to(F32)
    require_finite(x)
    rows, width = x.shape
    if width % BLOCK_NVFP4:
        raise ValueError(f"an NVFP4 row has {width} values; the length must be a multiple of {BLOCK_NVFP4}")
    amax = x.abs().amax(dim=1)
    encoder = torch.minimum(TOP_1536 / amax, torch.tensor(FP32_MAX, dtype=F32))
    decoder = ONE / encoder
    outer = amax * RECIP_1536
    blocks = x.reshape(rows, width // BLOCK_NVFP4, BLOCK_NVFP4)
    block_max = blocks.abs().amax(dim=-1)
    candidates = []
    for factor in (RECIP_6, QUARTER):
        byte = e4m3_bytes(torch.minimum(block_max * (encoder[:, None] * factor), torch.tensor(E4M3_MAX, dtype=F32)))
        scale = e4m3_values(byte)
        inverse = torch.minimum(ONE / (scale * decoder[:, None]), torch.tensor(FP32_MAX, dtype=F32))
        codes = e2m1_codes(blocks * inverse[..., None])
        rebuilt = (E2M1[codes] * scale[..., None]) * outer[:, None, None]
        error = tree_sum16((rebuilt - blocks).abs())
        candidates.append((byte, codes, error))
    (byte6, codes6, error6), (byte4, codes4, error4) = candidates
    pick4 = error4 < error6
    byte = torch.where(pick4, byte4, byte6)
    codes = torch.where(pick4[..., None], codes4, codes6)
    return codes.reshape(rows, width).to(torch.uint8), byte, amax


def dequantize_nvfp4(codes, scale_bytes, amax):
    """The f32 rows of NVFP4 codes: `e2m1 * block_scale * outer`, with `outer = amax * (1/1536)`."""
    rows, width = codes.shape
    scale = e4m3_values(scale_bytes)
    blocks = E2M1[codes.to(torch.int64)].reshape(rows, width // BLOCK_NVFP4, BLOCK_NVFP4)
    return ((blocks * scale[..., None]) * (amax * RECIP_1536)[:, None, None]).reshape(rows, width)


def ue8m0_values(byte):
    """The f32 scales `2^(byte - 127)` of ue8m0 bytes. The byte 0 is 2^-127, a subnormal. The byte 255 is an error."""
    if bool((byte == 255).any()):
        raise ValueError("a UE8M0 scale is 255 (NaN); the activation buffer is corrupt")
    bits = torch.where(byte == 0, torch.tensor(SMALLEST_UE8M0, dtype=torch.int32), byte.to(torch.int32) << 23)
    return bits.view(F32)


def quantize_mxfp8(x):
    """MXFP8 `ue8m0` of f32 rows `[R, K]`: the e4m3 codes `[R, K]` and the ue8m0 scale bytes `[R, K/32]`."""
    x = x.to(F32)
    require_finite(x)
    rows, width = x.shape
    if width % BLOCK_MXFP8:
        raise ValueError(f"an MXFP8 row has {width} values; the length must be a multiple of {BLOCK_MXFP8}")
    blocks = x.reshape(rows, width // BLOCK_MXFP8, BLOCK_MXFP8)
    raw = blocks.abs().amax(dim=-1) * RECIP_448
    # The exponent of the raw scale, rounded up: a nonzero mantissa carries into the exponent field.
    exponent = ((raw.view(torch.int32) + 0x007F_FFFF) & 0x7F80_0000) >> 23
    byte = torch.where(raw <= torch.tensor(SMALLEST_UE8M0, dtype=torch.int32).view(F32), 0, exponent.clamp(max=254))
    byte = byte.to(torch.uint8)
    inverse = torch.where(byte == 0, torch.tensor(2.0**127, dtype=F32), ONE / ue8m0_values(byte))
    codes = e4m3_bytes(blocks * inverse[..., None])
    return codes.reshape(rows, width), byte


def dequantize_mxfp8(codes, byte):
    """The f32 rows of MXFP8 codes: `e4m3 * 2^(byte - 127)`."""
    rows, width = codes.shape
    blocks = e4m3_values(codes).reshape(rows, width // BLOCK_MXFP8, BLOCK_MXFP8)
    return (blocks * ue8m0_values(byte)[..., None]).reshape(rows, width)


def fake_quantize(x, fmt):
    """Quantize f32 rows `[R, K]` in the format `fmt` ("nvfp4" or "mxfp8"), and expand them to f32."""
    if fmt == "nvfp4":
        codes, scales, amax = quantize_nvfp4(x)
        return dequantize_nvfp4(codes, scales, amax)
    if fmt == "mxfp8":
        codes, scales = quantize_mxfp8(x)
        return dequantize_mxfp8(codes, scales)
    raise ValueError(f"the activation format {fmt!r} is not 'nvfp4' or 'mxfp8'")


def pack_nibbles(codes):
    """Pack e2m1 codes `[R, K]` into bytes `[R, K/2]`: the even element in the low nibble."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


EXPERT_NAME = re.compile(r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight$")


def layer_formats(folder, layers):
    """The activation format of each expert layer, from `hf_quant_config.json` of a model folder.

    Returns a list with one entry per layer, "nvfp4" or "mxfp8". Both expert tensors of a layer must name the same
    rule, and the rule must be the one that this module implements. Any other case is an error: the module does not
    guess a layer split.
    """
    config = json.loads((Path(folder) / "hf_quant_config.json").read_text())
    found = {}
    for group in config["model"]["parameter_groups"]:
        rule = group["required_activation_quantization"]
        for tensor in group["tensors"]:
            match = EXPERT_NAME.match(tensor)
            if not match:
                continue
            if rule == NVFP4_RULE:
                fmt = "nvfp4"
            elif rule == MXFP8_RULE:
                fmt = "mxfp8"
            else:
                raise ValueError(f"{tensor} requires the activation rule {rule}, which this module does not implement")
            layer = int(match.group(1))
            if found.setdefault(layer, fmt) != fmt:
                raise ValueError(f"layer {layer} has two activation formats")
    if sorted(found) != list(range(layers)):
        raise ValueError(f"hf_quant_config.json names expert tensors for the layers {sorted(found)}, not 0 to {layers - 1}")
    return [found[layer] for layer in range(layers)]
