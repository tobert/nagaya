#!/usr/bin/env python3
"""Write the fixtures of the op tests of nagaya-mk-qwen38-gfx1151 from mk's Python (milestone 15, step 2).

The Rust tests in `crates/nagaya-mk-qwen38-gfx1151/tests/ops` port the op tests of mk (`tests/ops`). Rust remakes the
inputs with the generator below, which has no state: each value is a function of a seed and an index. This tool
uses the same generator, runs mk's reference code on the same inputs, and writes the expected outputs. No Python runs at
test time. The tool runs no GPU job and writes nothing into mk.

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 \
        ~/src/megakernel-qwen38-flashnext-strixhalo/.venv/bin/python -I tools/op_fixtures.py [--out DIR] [GROUP ...]

A group is a pair of files, `<group>.json` and `<group>.bin`. The JSON file holds the header, the metadata and an index of
the arrays. The BIN file holds the little-endian bytes of the arrays. With no group the tool writes every group.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

MK = Path(os.environ.get("MK_REPO", Path.home() / "src/megakernel-qwen38-flashnext-strixhalo"))
GGUF_PY = Path.home() / "src/wt/llama.cpp-master-0924/gguf-py"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "crates/nagaya-mk-qwen38-gfx1151/tests/fixtures/ops"
AUTHOR = "written by Claude Sonnet 5.5 for this tool, 2026-10-22"

# ---- the generator: the same as inputs.rs in the tests --------------------------------------------------------------------

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _mix(z):
    z = np.asarray(z, dtype=np.uint64)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def u64_stream(seed: int, n: int) -> np.ndarray:
    """The values at indices 0..n-1 of the stream of `seed`."""
    with np.errstate(over="ignore"):
        key = _mix(np.uint64(seed) * GOLDEN ^ np.uint64(0x6F70735F74657374))
        index = np.arange(n, dtype=np.uint64)
        return _mix(key + (index + np.uint64(1)) * GOLDEN)


_SD = math.sqrt(4.0 * (65536.0 * 65536.0 - 1.0) / 12.0)


def normals(seed: int, n: int) -> np.ndarray:
    u = u64_stream(seed, n)
    s = ((u & np.uint64(0xFFFF)) + ((u >> np.uint64(16)) & np.uint64(0xFFFF)) + ((u >> np.uint64(32)) & np.uint64(0xFFFF))
         + ((u >> np.uint64(48)) & np.uint64(0xFFFF))).astype(np.float64)
    return ((s - 4.0 * 32767.5) / _SD).astype(np.float32)


def uniforms(seed: int, n: int) -> np.ndarray:
    return (u64_stream(seed, n) >> np.uint64(40)).astype(np.float32) / np.float32(16777216.0)


def integers(seed: int, n: int, bound: int) -> np.ndarray:
    return (u64_stream(seed, n) % np.uint64(bound)).astype(np.uint32)


def permutation(seed: int, n: int) -> np.ndarray:
    u = u64_stream(seed, n)
    return np.lexsort((np.arange(n), u)).astype(np.uint32)


def stream_bytes(seed: int, n: int) -> np.ndarray:
    u = u64_stream(seed, (n + 7) // 8)
    return u.astype("<u8").view(np.uint8)[:n].copy()


# type name -> (values per block, bytes per block, offset of the f16 d or None)
QTYPES = {
    "F32": (1, 4, None), "BF16": (1, 2, None), "Q8_0": (32, 34, 0), "IQ4_NL": (32, 18, 0), "Q6_K": (256, 210, 208),
    "IQ3_S": (256, 110, 0), "IQ4_XS": (256, 136, 0),
}


def raw_rows(qtype: str, seed: int, rows: int, k: int) -> np.ndarray:
    """`rows` rows of `k` values in the GGUF layout, as bytes (inputs.rs `raw_rows`)."""
    values, block, d_off = QTYPES[qtype]
    assert k % values == 0
    if qtype == "F32":
        return normals(seed, rows * k).astype("<f4").view(np.uint8).reshape(rows, k * 4)
    if qtype == "BF16":
        bits = (normals(seed, rows * k).view(np.uint32) >> np.uint32(16)).astype("<u2")
        return bits.view(np.uint8).reshape(rows, k * 2)
    row = k // values * block
    out = stream_bytes(seed, rows * row).reshape(-1, block)
    d = (np.uint64(0x1400) + u64_stream(seed + 0x10000000, out.shape[0]) % np.uint64(0x1000)).astype("<u2")
    out[:, d_off:d_off + 2] = d.view(np.uint8).reshape(-1, 2)
    return out.reshape(rows, row)


# ---- the fixture files -------------------------------------------------------------------------------------------------

DTYPES = {np.dtype("<f4"): "f32", np.dtype("<u4"): "u32", np.dtype("<i4"): "i32", np.dtype("u1"): "u8"}


class Group:
    def __init__(self, name: str):
        self.name, self.meta, self.arrays, self.blob = name, {}, {}, bytearray()

    def add(self, name: str, array) -> None:
        a = np.ascontiguousarray(array)
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        assert a.dtype in DTYPES, f"{name}: dtype {a.dtype}"
        assert name not in self.arrays, name
        raw = a.tobytes()
        self.arrays[name] = {"dtype": DTYPES[a.dtype], "shape": list(a.shape), "offset": len(self.blob), "len": len(raw)}
        self.blob += raw

    def write(self, out: Path, header: dict) -> None:
        out.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"header": header, "meta": self.meta, "arrays": self.arrays}, sort_keys=True, indent=1)
        (out / f"{self.name}.json").write_text(text + "\n", encoding="ascii")
        (out / f"{self.name}.bin").write_bytes(bytes(self.blob))
        print(f"{self.name}: {len(self.blob)} bytes of arrays, {len(text)} bytes of index")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


# ---- the groups --------------------------------------------------------------------------------------------------------

GROUPS = {}


def group(fn):
    GROUPS[fn.__name__.removeprefix("make_")] = fn
    return fn


@group
def make_gen(g: Group) -> None:
    """The first values of each generator function, so that the Rust test can check its own generator."""
    g.add("u64_head", u64_stream(7, 16).astype("<u8").view(np.uint32))
    g.add("normals", normals(9, 64))
    g.add("uniforms", uniforms(11, 64))
    g.add("integers", integers(13, 64, 300).astype("<u4"))
    g.add("permutation", permutation(15, 100).astype("<u4"))
    g.add("bytes", stream_bytes(17, 100))
    g.meta["raw"] = []
    for qtype, (values, block, _) in QTYPES.items():
        rows, k = 3, 512
        raw = raw_rows(qtype, 19, rows, k)
        # Keep only the head and the tail of each row, so that the fixture stays small.
        g.add(f"raw_{qtype}", raw.reshape(-1)[:300])
        g.meta["raw"].append({"type": qtype, "seed": 19, "rows": rows, "k": k, "len": 300})


@group
def make_qmm(g: Group) -> None:
    """Dequantized values of gguf-py for the raw rows of the generator, one case for each type (test_qmm.py)."""
    sys.path.insert(0, str(GGUF_PY))
    import gguf
    types = {"F32": gguf.GGMLQuantizationType.F32, "BF16": gguf.GGMLQuantizationType.BF16,
             "Q8_0": gguf.GGMLQuantizationType.Q8_0, "IQ4_NL": gguf.GGMLQuantizationType.IQ4_NL,
             "Q6_K": gguf.GGMLQuantizationType.Q6_K, "IQ3_S": gguf.GGMLQuantizationType.IQ3_S,
             "IQ4_XS": gguf.GGMLQuantizationType.IQ4_XS}
    g.meta["dequant_cases"] = []
    for n, (qtype, ty) in enumerate(types.items()):
        rows, k, seed = 4, 512, 500 + n
        raw = raw_rows(qtype, seed, rows, k)
        want = gguf.quants.dequantize(raw, ty).astype(np.float32).reshape(rows, k)
        assert np.isfinite(want).all(), qtype
        g.add(f"dequant_{qtype}", want)
        g.meta["dequant_cases"].append({"type": qtype, "rows": rows, "k": k, "seed": seed})


# Every mk file that a group reads, for the header.
MK_FILES = ["tests/ops/test_qmm.py"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="*")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    names = args.groups or list(GROUPS)
    for name in names:
        if name not in GROUPS:
            sys.exit(f"no group {name}; the groups are {', '.join(GROUPS)}")
    header = {
        "mk_rev": git(MK, "rev-parse", "HEAD"),
        "mk_files": MK_FILES,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "generator": "tools/op_fixtures.py",
        "author": AUTHOR,
    }
    for name in names:
        g = Group(name)
        GROUPS[name](g)
        g.write(args.out, header)


main()
