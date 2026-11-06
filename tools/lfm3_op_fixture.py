#!/usr/bin/env python3
"""Write the fixture of the LFM3 ops of nagaya-mk-bmg (milestone 24 step 5): the gated short convolution, the
partial rotation, the sliding attention and the gate of the heads.

The reference is plain torch on the CPU, in f32. Each function below copies the arithmetic of LiquidAI's remote code
at the pinned revision of LFM3-Omni-2B, and the comment of each names the lines. The tool does not import the remote
code. The inputs come from a generator with no state, so Rust makes them again. The fixture holds only the outputs.

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 ~/.venvs/nagaya-ref/bin/python -I tools/lfm3_op_fixture.py

The tool writes `lfm3_ops.json` and `lfm3_ops.bin` into `crates/nagaya-mk-bmg/tests/fixtures`, in the format of
`tools/op_fixtures.py`. It runs no GPU job.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "crates/nagaya-mk-bmg/tests/fixtures"
REVISION = "21d75123f86c1f7ffb4bef1ea202710c4e7ebe67"
AUTHOR = "written by Claude Opus 5.5 for this tool, 2026-11-06"

# ---- the generator: the same as tests/common/lfm3.rs ------------------------------------------------------------------

GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _mix(z):
    z = np.asarray(z, dtype=np.uint64)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def values(seed: int, n: int, scale: float = 1.0) -> np.ndarray:
    """`n` values in [-scale, scale): the top 24 bits of splitmix64 of the seed and the index."""
    with np.errstate(over="ignore"):
        key = _mix(np.uint64(seed) * GOLDEN)
        u = _mix(key + (np.arange(n, dtype=np.uint64) + np.uint64(1)) * GOLDEN)
    unit = (u >> np.uint64(40)).astype(np.float32) / np.float32(8388608.0) - np.float32(1.0)
    return unit * np.float32(scale)


def bf16(x: np.ndarray) -> np.ndarray:
    """`x` rounded to BF16 and back to f32, to the nearest even, as `bf16` in the Rust tests."""
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))) & np.uint64(0xFFFF0000)
    return u.astype(np.uint32).view(np.float32)


# ---- the fixture file -------------------------------------------------------------------------------------------------


class Fixture:
    def __init__(self):
        self.meta, self.arrays, self.blob = {}, {}, bytearray()

    def add(self, name: str, array) -> None:
        a = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
        assert name not in self.arrays, name
        raw = a.tobytes()
        self.arrays[name] = {"dtype": "f32", "shape": list(a.shape), "offset": len(self.blob), "len": len(raw)}
        self.blob += raw

    def write(self, out: Path, header: dict) -> None:
        out.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"header": header, "meta": self.meta, "arrays": self.arrays}, sort_keys=True, indent=1)
        (out / "lfm3_ops.json").write_text(text + "\n", encoding="ascii")
        (out / "lfm3_ops.bin").write_bytes(bytes(self.blob))
        print(f"lfm3_ops: {len(self.blob)} bytes of arrays, {len(text)} bytes of index")


# ---- the reference ops -------------------------------------------------------------------------------------------------


def short_conv(bcx: torch.Tensor, weight: torch.Tensor, state: torch.Tensor):
    """Lfm3ShortConv.forward after in_proj (modeling_lfm3.py:371-373, 422) with the cached window of
    causal_conv1d_update (:299-302): the state is the last `taps - 1` inputs of the conv, B * x.

    bcx is [T][3D] (time-major), weight [D][taps], state [taps - 1][D]. Returns y [T][D] and the new state."""
    t, d = bcx.shape[0], bcx.shape[1] // 3
    b, c, x = bcx.transpose(0, 1).unsqueeze(0).chunk(3, dim=-2)  # (1, D, T) each, as BCx.chunk(3, dim=-2)
    bx = b * x
    window = torch.cat([state.transpose(0, 1).unsqueeze(0), bx], dim=-1)
    new_state = window[:, :, -state.shape[0]:]
    out = F.conv1d(window, weight.unsqueeze(1), None, padding=0, groups=d)[:, :, -t:]
    y = c * out
    return y[0].transpose(0, 1), new_state[0].transpose(0, 1)


def inv_freq(theta: float, factor: float, dim: int) -> torch.Tensor:
    """Lfm3RotaryEmbedding.compute_default_rope_parameters (modeling_lfm3.py:105-111): the zero-tail layout."""
    rope_angles = int(factor * dim) // 2
    f = 1.0 / (theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / dim))
    return torch.cat((f, torch.zeros(dim // 2 - rope_angles, dtype=torch.float)))


def rotate_half(x):
    """modeling_lfm3.py:160-164."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def norm_rope(x: torch.Tensor, weight: torch.Tensor, positions: torch.Tensor, theta: float, factor: float, eps: float):
    """q_norm or k_norm (Lfm3RMSNorm, modeling_lfm3.py:61-66), then the rotation of apply_rotary_pos_emb (:204-210)
    with the cos and sin of Lfm3RotaryEmbedding.forward (:119-122). x is [T][heads][D]."""
    variance = x.pow(2).mean(-1, keepdim=True)
    normed = weight * (x * torch.rsqrt(variance + eps))
    freqs = positions[:, None].float() * inv_freq(theta, factor, x.shape[-1])[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    return normed * cos + rotate_half(normed) * sin


def sliding_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int):
    """eager_attention_forward (modeling_lfm3.py:179-201) with scaling 1.0 and the causal mask of transformers, with
    the overlay `kv_idx > q_idx - sliding_window` (masking_utils.py, sliding_window_overlay) when window > 0.
    q is [T][H][D], k and v [T][KV][D]. Returns [T][H][D]."""
    t, h, _ = q.shape
    kvh = k.shape[1]
    qi = torch.arange(t)[:, None]
    ki = torch.arange(t)[None, :]
    allowed = ki <= qi
    if window > 0:
        allowed = allowed & (ki > qi - window)
    mask = torch.zeros(t, t).masked_fill(~allowed, float("-inf"))
    qh = q.transpose(0, 1).unsqueeze(0)  # (1, H, T, D)
    kr = k.transpose(0, 1).unsqueeze(0).repeat_interleave(h // kvh, dim=1)  # repeat_kv
    vr = v.transpose(0, 1).unsqueeze(0).repeat_interleave(h // kvh, dim=1)
    weights = torch.matmul(qh, kr.transpose(2, 3)) + mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32)
    out = torch.matmul(weights, vr)
    return out[0].transpose(0, 1)


def head_gate(attn: torch.Tensor, gate: torch.Tensor):
    """The sigmoid gate of each head (modeling_lfm3.py:270-271). attn is [T][H][D], gate [T][H]."""
    return attn * torch.sigmoid(gate).unsqueeze(-1)


# ---- the cases ---------------------------------------------------------------------------------------------------------

# The gated short convolution: D channels, 7 taps, a state of 6 rows. Each case has its own seeds.
CONV_D, CONV_TAPS = 48, 7
CONV_CASES = [(20, 100), (3, 110), (1, 120)]  # (positions, first seed)
# The rotation: heads of 128, the full layers (theta 1e6, half the head) and the sliding layers (theta 1e4, all of it).
ROPE_D, ROPE_HEADS, ROPE_TOKENS, ROPE_P0, ROPE_EPS = 128, 2, 3, 4093, 1e-5
ROPE_CASES = [("full", 1e6, 0.5, 200), ("sliding", 1e4, 1.0, 210)]
# The attention: the shape of the LFM3 backbone (16 heads on 4 KV heads of 128) over 300 positions.
ATTN_T, ATTN_H, ATTN_KV, ATTN_D, ATTN_SEED = 300, 16, 4, 128, 300
ATTN_CASES = [("sliding", 256, [0, 1, 2, 254, 255, 256, 257, 298, 299]), ("full", 0, [0, 150, 299])]
# The gate of the heads: the gates sit in rows of GATE_STRIDE values, from GATE_OFFSET.
GATE_T, GATE_H, GATE_D, GATE_STRIDE, GATE_OFFSET, GATE_SEED = 3, 4, 32, 20, 3, 400


def tensor(seed: int, shape, scale: float = 1.0) -> torch.Tensor:
    return torch.from_numpy(values(seed, int(np.prod(shape)), scale).reshape(shape).copy())


def make(fx: Fixture) -> None:
    for t, seed in CONV_CASES:
        weight = torch.from_numpy(bf16(values(seed, CONV_D * CONV_TAPS, 0.5)).reshape(CONV_D, CONV_TAPS).copy())
        bcx = tensor(seed + 1, (t, 3 * CONV_D))
        state = tensor(seed + 2, (CONV_TAPS - 1, CONV_D))
        y, new_state = short_conv(bcx, weight, state)
        fx.add(f"short_conv.{t}.y", y.numpy())
        fx.add(f"short_conv.{t}.state", new_state.numpy())
    for name, theta, factor, seed in ROPE_CASES:
        x = tensor(seed, (ROPE_TOKENS, ROPE_HEADS, ROPE_D))
        weight = torch.from_numpy(bf16(values(seed + 1, ROPE_D, 1.0) + np.float32(1.0)).copy())
        positions = torch.arange(ROPE_P0, ROPE_P0 + ROPE_TOKENS)
        fx.add(f"rope.{name}", norm_rope(x, weight, positions, theta, factor, ROPE_EPS).numpy())
    q = tensor(ATTN_SEED, (ATTN_T, ATTN_H, ATTN_D))
    k = tensor(ATTN_SEED + 1, (ATTN_T, ATTN_KV, ATTN_D))
    v = tensor(ATTN_SEED + 2, (ATTN_T, ATTN_KV, ATTN_D))
    for name, window, rows in ATTN_CASES:
        out = sliding_attention(q, k, v, window)
        fx.add(f"attention.{name}", out[rows].numpy())
        fx.meta[f"attention.{name}.rows"] = rows
    attn = tensor(GATE_SEED, (GATE_T, GATE_H, GATE_D))
    gates = tensor(GATE_SEED + 1, (GATE_T, GATE_STRIDE), 4.0)
    fx.add("head_gate", head_gate(attn, gates[:, GATE_OFFSET:GATE_OFFSET + GATE_H]).numpy())
    fx.meta["cases"] = {
        "short_conv": {"d": CONV_D, "taps": CONV_TAPS, "cases": CONV_CASES},
        "rope": {"d": ROPE_D, "heads": ROPE_HEADS, "tokens": ROPE_TOKENS, "p0": ROPE_P0, "eps": ROPE_EPS,
                 "cases": ROPE_CASES},
        "attention": {"t": ATTN_T, "h": ATTN_H, "kv": ATTN_KV, "d": ATTN_D, "seed": ATTN_SEED},
        "head_gate": {"t": GATE_T, "h": GATE_H, "d": GATE_D, "stride": GATE_STRIDE, "offset": GATE_OFFSET,
                      "seed": GATE_SEED},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    header = {
        "model": "LiquidAI/LFM3-Omni-2B",
        "revision": REVISION,
        "reference": "modeling_lfm3.py of the revision, copied into plain torch",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "generator": "tools/lfm3_op_fixture.py",
        "author": AUTHOR,
    }
    fx = Fixture()
    make(fx)
    fx.write(args.out, header)


main()
