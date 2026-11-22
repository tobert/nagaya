# SPDX-License-Identifier: MIT
"""Write the fixture of the Inkling 2 expert activation quantization for `nagaya-mk-inkling2-sm121`.

It runs the Triton kernels that vLLM ships, on the GPU of this machine, with no change to their source:
`nvfp4_quantize` (nvfp4.py) and `mxfp8_quantize` (mxfp8.py) of `vllm/models/inkling2/nvidia/ops`. A Rust test
compares its own quantizer with the bytes that these kernels return, bit for bit.

The tool does not import vLLM. It loads the `ops` folder of a vLLM checkout as a package, and it puts small stubs
into `sys.modules` for the three vLLM names that the kernels import: `vllm.platforms.current_platform`,
`vllm.triton_utils` (`tl`, `tldevice`, `triton`) and nothing else. The stub `is_device_capability_family` tells
the truth for the GPU: it compares the capability of the device, as vLLM does.

Rows are bf16, so each input is exact in f32. There are three sets:
- `nvfp4_k2304`: NVFP4 with K = 2304, the hidden size of Inkling 2 Mini.
- `nvfp4_k1152`: NVFP4 with K = 1152, the expert intermediate size.
- `mxfp8_k2304`: MXFP8 with K = 2304.

Each set has the tensors `x` (bf16 inputs), and the outputs as the kernel returns them, as raw bytes:
NVFP4 `values` (u8, two e2m1 codes each, the low nibble first), `scales` (e4m3 bytes) and `amax` (f32 per row);
MXFP8 `values` (e4m3 bytes) and `scales` (ue8m0 bytes). The scales are row-major (`TILE` = 0).
The metadata holds the vLLM commit, the SHA-256 of each loaded kernel file, the library versions and the class
of each row.

Versions of the recorded run: Python 3.12, torch 2.14.1+cu130, triton 3.8.0, CUDA 13.0 runtime, NVIDIA GB10
(capability 12.1). The recorded vLLM commit is 186882d9e865ccfabc9f75146a708c4a68524c8e.

Run it with a venv that has torch (CUDA), triton and safetensors:
    PYTHONDONTWRITEBYTECODE=1 VENV/bin/python -I tools/inkling2_quant_fixture.py \\
        --vllm VLLM_CHECKOUT --out crates/nagaya-mk-inkling2-sm121/tests/fixtures/quant_kernels.safetensors

The tool stops, and writes nothing, when a kernel fails to load or when two runs give different bytes.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors.torch import save_file

OPS = "vllm/models/inkling2/nvidia/ops"
KERNEL_FILES = ["__init__.py", "block_scales.py", "nvfp4.py", "mxfp8.py"]
SEED = 20261122
BF16_MAX = 3.3895313892515355e38
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


# ---------------------------------------------------------------- load the kernels


def install_stubs():
    """Put the stubs of the three vLLM names into `sys.modules`."""
    major, minor = torch.cuda.get_device_capability()

    class Platform:
        def is_device_capability_family(self, family):
            return (major * 10 + minor) // 10 == family // 10

    from triton.language.extra.cuda import libdevice

    vllm = types.ModuleType("vllm")
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = Platform()
    utils = types.ModuleType("vllm.triton_utils")
    utils.tl, utils.tldevice, utils.triton = tl, libdevice, triton
    vllm.platforms, vllm.triton_utils = platforms, utils
    sys.modules.update({"vllm": vllm, "vllm.platforms": platforms, "vllm.triton_utils": utils})


def load_ops(checkout):
    """Load the `ops` folder as the package `inkling2_ops`, with its relative imports. Return it and the hashes."""
    folder = Path(checkout) / OPS
    spec = importlib.util.spec_from_file_location(
        "inkling2_ops", folder / "__init__.py", submodule_search_locations=[str(folder)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["inkling2_ops"] = package
    spec.loader.exec_module(package)
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in KERNEL_FILES}
    nvfp4 = importlib.import_module("inkling2_ops.nvfp4")
    mxfp8 = importlib.import_module("inkling2_ops.mxfp8")
    return nvfp4.nvfp4_quantize, mxfp8.mxfp8_quantize, hashes


def checkout_commit(checkout):
    out = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"cannot read the commit of {checkout}: {out.stderr.strip()}")
    return out.stdout.strip()


# ---------------------------------------------------------------- build the rows


def bf16_bits_add(values, delta):
    """Move each bf16 value by `delta` units in the last place (toward larger magnitude when positive)."""
    bits = values.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)
    return (bits + delta).to(torch.int16).view(torch.bfloat16).to(torch.float32)


def e4m3_midpoints():
    """The midpoints of neighbours of the positive e4m3 grid: exact in bf16, and ties for the e4m3 conversion."""
    codes = torch.arange(0, 126, dtype=torch.uint8)
    grid = codes.view(torch.float8_e4m3fn).to(torch.float32)
    return ((grid[:-1] + grid[1:]) / 2)[8:]


class Rows:
    """A list of f32 rows, each with a class name."""

    def __init__(self, k, block, gen):
        self.k, self.block, self.gen = k, block, gen
        self.rows, self.classes = [], []

    def add(self, name, row):
        assert row.numel() == self.k, (name, row.numel())
        self.rows.append(row.to(torch.float32))
        self.classes.append(name)

    def normal(self, scale=1.0):
        return torch.randn(self.k, generator=self.gen) * scale

    def blocks(self, row):
        return row.view(self.k // self.block, self.block)


def shared_rows(r):
    """The classes that both formats use."""
    for magnitude in (1e-3, 1.0, 30.0, 1e4):
        for _ in range(4):
            r.add(f"normal_{magnitude:g}", r.normal(magnitude))
    r.add("zero_row", torch.zeros(r.k))
    for position in (0, r.k // r.block // 2, r.k // r.block - 1):
        row = r.normal()
        r.blocks(row)[position] = 0.0
        r.add("zero_block", row)
    # The block that holds the row amax, and blocks far below it: the block scale goes into the subnormal range.
    for _ in range(4):
        row = r.normal()
        shifts = torch.arange(r.k // r.block) % 26
        r.blocks(row).mul_(torch.exp2(-shifts.to(torch.float32))[:, None])
        r.blocks(row)[0] = r.normal(1.0)[: r.block]
        r.blocks(row)[0, 0] = 1.0
        r.add("far_below_amax", row)
    # A different spread of block magnitudes per block.
    for _ in range(6):
        row = r.normal()
        r.blocks(row).mul_(torch.exp(torch.randn(r.k // r.block, 1, generator=r.gen) * 3.0))
        r.add("block_spread", row)
    # Tiny and large rows.
    r.add("tiny_1e-30", r.normal(1e-30))
    r.add("tiny_1e-38_subnormal", r.normal(1e-38))
    row = torch.zeros(r.k)
    row[7] = 2.0**-133
    r.add("tiny_smallest_subnormal", row)
    row = torch.zeros(r.k)
    row[:] = torch.randint(0, 4, (r.k,), generator=r.gen).to(torch.float32) * 2.0**-133
    r.add("tiny_subnormal_codes", row)
    r.add("large_near_max", (torch.rand(r.k, generator=r.gen) * 0.5 + 0.5) * BF16_MAX)
    row = torch.zeros(r.k)
    row[3] = BF16_MAX
    r.add("large_one_max", row)
    r.add("large_1e38", r.normal(1e38).clamp(-BF16_MAX, BF16_MAX))
    row = torch.zeros(r.k)
    row[5] = 1.0
    r.add("one_hot", row)
    r.add("constant", torch.full((r.k,), 1.0))
    r.add("constant_negative", torch.full((r.k,), -0.4375))


def nvfp4_rows(k, gen):
    r = Rows(k, 16, gen)
    shared_rows(r)
    nblocks = k // 16
    # Ties of the e2m1 grid after scaling. With the row amax `a` in block 0, the scaled values are
    # `x * 6 / a` for the 6-based scale and `x * 4 / a` for the 4-based scale. A power of two for `a` keeps both exact.
    ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    for top in (6.0, 4.0):
        pattern = torch.tensor([top] + ties + [-t for t in ties] + [0.0])
        for variant in range(2):
            row = torch.empty(k)
            for j in range(nblocks):
                block = pattern.clone()
                if variant == 1:
                    block = block[torch.randperm(16, generator=gen)]
                row[j * 16 : (j + 1) * 16] = block * 2.0 ** -(j % 20)
            r.add(f"tie_grid_amax{top:g}", row)
    # Ties at the midpoints, moved one step in the last place: just below and just above the tie.
    for delta in (-1, 1):
        pattern = torch.tensor([6.0] + ties + [-t for t in ties] + [0.0])
        shifted = pattern.clone()
        shifted[1:15] = bf16_bits_add(pattern[1:15].abs(), delta) * torch.sign(pattern[1:15])
        row = shifted.repeat(nblocks)
        r.add("tie_grid_near_ulp", row)
    # Blocks with one amax value and fifteen equal values: the 4-based and 6-based errors come close or equal.
    for lo, hi, top in ((0.05, 1.0, 1.0), (0.3, 0.9, 6.0), (0.5, 0.8, 4.0), (0.0625, 0.95, 3.0)):
        row = torch.empty(k)
        ratios = torch.linspace(lo, hi, nblocks)
        for j in range(nblocks):
            row[j * 16 : (j + 1) * 16] = top * ratios[j]
            row[j * 16] = top
        r.add("near_equal_error", row)
    # Equal values at each e2m1 multiple of a unit: the two candidates may round to the same grid.
    row = torch.empty(k)
    for j in range(nblocks):
        row[j * 16 : (j + 1) * 16] = E2M1[j % 8] * (1.0 + (j // 8) * 0.0625)
        row[j * 16] = 6.0
    r.add("grid_multiples", row)
    # A row whose amax sits in the last block, and one in the middle.
    for position in (nblocks - 1, nblocks // 3):
        row = r.normal() * 0.01
        r.blocks(row)[position, 5] = 50.0
        r.add("amax_in_block", row)
    return r


def mxfp8_rows(k, gen):
    r = Rows(k, 32, gen)
    shared_rows(r)
    nblocks = k // 32
    # Blocks whose amax/448 is an exact power of two, just below it, and just above it.
    for delta in (0, -1, 1):
        row = torch.empty(k)
        for j in range(nblocks):
            exponent = j % 60 - 30
            amax = torch.tensor([448.0 * 2.0**exponent])
            amax = bf16_bits_add(amax, delta) if delta else amax
            block = torch.randn(32, generator=gen) * float(amax) * 0.25
            block[j % 32] = float(amax) * (1 if j % 2 == 0 else -1)
            row[j * 32 : (j + 1) * 32] = block
        r.add(["amax_pow2", "amax_pow2_below_ulp", "amax_pow2_above_ulp"][[0, -1, 1].index(delta)], row)
    # Scaled values on the ties of the e4m3 grid (amax 448 makes the scale 1).
    mid = e4m3_midpoints()
    for _ in range(3):
        row = torch.empty(k)
        for j in range(nblocks):
            pick = mid[torch.randint(0, mid.numel(), (32,), generator=gen)]
            pick = pick * (torch.randint(0, 2, (32,), generator=gen) * 2 - 1)
            pick[j % 32] = 448.0
            row[j * 32 : (j + 1) * 32] = pick * 2.0 ** (j % 7 - 3)
        r.add("tie_e4m3_midpoints", row)
    # Scales at the two ends of the ue8m0 range.
    row = r.normal(2.0**-120)
    r.add("scale_exponent_low", row)
    row = r.normal(2.0**120)
    r.add("scale_exponent_high", row)
    return r


# ---------------------------------------------------------------- host checks


def e2m1_round(values):
    """Round f32 values to the e2m1 grid, to nearest, ties to the even code, with saturation. Return the values."""
    grid = torch.tensor(E2M1, dtype=torch.float64)
    a = values.abs().to(torch.float64)
    distance = (a[..., None] - grid).abs()
    best = distance.min(-1, keepdim=True).values
    # Among equal distances the lower index is even-first only for index pairs (0,1),(2,3): pick the even code.
    candidates = distance == best
    index = torch.arange(8)
    even = candidates & (index % 2 == 0)
    chosen = torch.where(even.any(-1, keepdim=True), even, candidates)
    code = (chosen * (8 - index)).argmax(-1)
    return grid[code] * torch.where(values < 0, -1.0, 1.0).to(torch.float64)


def nvfp4_statistics(x, scales):
    """Count the blocks that chose the 4-based or the 6-based scale. Emulate both candidates on the CPU in f64."""
    x = x.to(torch.float32).cpu()
    rows, k = x.shape
    blocks = x.view(rows, k // 16, 16)
    amax = x.abs().max(1).values
    stats = dict(blocks=0, zero_block=0, chose4=0, chose6=0, same_scale=0, error_equal=0, emulation_mismatch=0)
    for i in range(rows):
        if amax[i] == 0:
            stats["blocks"] += k // 16
            stats["zero_block"] += k // 16
            continue
        encoder = min(float(torch.tensor(1536.0) / amax[i]), 3.4028234663852886e38)
        decoder = 1.0 / encoder
        outer = float(amax[i]) * (1.0 / 1536.0)
        got = scales[i].cpu().view(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
        for j in range(k // 16):
            stats["blocks"] += 1
            b = blocks[i, j].to(torch.float64)
            bm = float(b.abs().max())
            if bm == 0:
                stats["zero_block"] += 1
                continue
            cand = {}
            for name, factor in (("6", 1.0 / 6.0), ("4", 0.25)):
                s = torch.tensor(min(bm * (encoder * factor), 448.0), dtype=torch.float32)
                s = s.to(torch.float8_e4m3fn).to(torch.float32)
                inverse = min(1.0 / (float(s) * decoder), 3.4028234663852886e38) if s > 0 else 3.4028234663852886e38
                rounded = e2m1_round((b * inverse).to(torch.float32))
                error = (rounded * float(s) * outer - b).abs().sum()
                cand[name] = (float(s), float(error))
            if cand["4"][0] == cand["6"][0]:
                stats["same_scale"] += 1
                continue
            if cand["4"][1] == cand["6"][1]:
                stats["error_equal"] += 1
            expected4 = cand["4"][1] < cand["6"][1]
            if float(got[j]) == cand["4"][0] and expected4:
                stats["chose4"] += 1
            elif float(got[j]) == cand["6"][0] and not expected4:
                stats["chose6"] += 1
            else:
                stats["emulation_mismatch"] += 1
    return stats


# ---------------------------------------------------------------- run


def to_bf16_checked(rows):
    """Stack the rows as bf16. Return the tensor and the number of values that bf16 changed."""
    x32 = torch.stack(rows)
    x = x32.to(torch.bfloat16)
    return x, int((x.to(torch.float32) != x32).sum())


def run_twice(fn, x):
    """Run a kernel twice, and with the rows padded past 512 (another launch shape). Stop when the bytes differ."""
    first = fn(x)
    second = fn(x)
    padded = fn(torch.cat([x, x[:1].repeat(520, 1)]))
    for a, b, c in zip(first, second, padded):
        if not torch.equal(a.view(torch.uint8) if a.dtype != torch.float32 else a, b.view(torch.uint8) if b.dtype != torch.float32 else b):
            sys.exit("the kernel gave different bytes on two runs of the same rows")
        got = c[: x.shape[0]]
        same = torch.equal(a.view(torch.uint8), got.view(torch.uint8)) if a.dtype != torch.float32 else torch.equal(a, got)
        if not same:
            sys.exit("the kernel gave different bytes with the rows padded past 512 rows")
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--vllm", required=True, help="a vLLM checkout")
    parser.add_argument("--out", required=True, help="the safetensors file to write")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device: the kernels need a GPU")
    install_stubs()
    try:
        nvfp4_quantize, mxfp8_quantize, hashes = load_ops(args.vllm)
        # Force the compile of the launch path now, so a missing Triton feature stops here.
        probe = torch.zeros(1, 32, dtype=torch.bfloat16, device="cuda")
        nvfp4_quantize(probe)
        mxfp8_quantize(probe)
        torch.cuda.synchronize()
    except Exception as error:  # report the exact cause, and do not edit the kernel
        sys.exit(f"cannot run the vLLM kernels: {type(error).__name__}: {error}")

    gen = torch.Generator().manual_seed(SEED)
    sets = {
        "nvfp4_k2304": ("nvfp4", nvfp4_rows(2304, gen)),
        "nvfp4_k1152": ("nvfp4", nvfp4_rows(1152, gen)),
        "mxfp8_k2304": ("mxfp8", mxfp8_rows(2304, gen)),
    }
    tensors, classes, report = {}, {}, {}
    for name, (kind, rows) in sets.items():
        x, changed = to_bf16_checked(rows.rows)
        xd = x.cuda()
        if kind == "nvfp4":
            values, scales, amax = run_twice(nvfp4_quantize, xd)
            tensors[f"{name}.values"] = values.cpu().contiguous()
            tensors[f"{name}.scales"] = scales.cpu().view(torch.uint8).contiguous()
            tensors[f"{name}.amax"] = amax.cpu().contiguous()
            report[name] = nvfp4_statistics(x, scales)
        else:
            values, scales = run_twice(mxfp8_quantize, xd)
            tensors[f"{name}.values"] = values.cpu().view(torch.uint8).contiguous()
            tensors[f"{name}.scales"] = scales.cpu().contiguous()
        tensors[f"{name}.x"] = x.contiguous()
        classes[name] = rows.classes
        report.setdefault(name, {})["rows"] = len(rows.rows)
        report[name]["values_changed_by_bf16"] = changed
        counts = {}
        for c in rows.classes:
            counts[c] = counts.get(c, 0) + 1
        report[name]["classes"] = counts

    metadata = {
        "vllm_commit": checkout_commit(args.vllm),
        "kernel_sha256": json.dumps(hashes, sort_keys=True),
        "versions": json.dumps(
            {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "triton": triton.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
            }
        ),
        "seed": str(SEED),
        "layout": json.dumps(
            {
                "nvfp4": "values u8 [rows, k/2] low nibble first; scales e4m3 bytes [rows, k/16] row-major; amax f32 [rows]",
                "mxfp8": "values e4m3 bytes [rows, k]; scales ue8m0 bytes [rows, k/32] row-major",
            }
        ),
        "classes": json.dumps(classes),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=metadata)
    print(json.dumps(report, indent=1))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
