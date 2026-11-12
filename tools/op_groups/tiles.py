"""The group `tiles`: dequantized values of gguf-py for raw rows of the generator, for the round trips of the tile layouts."""
import sys

import numpy as np

SOURCE_FILES = ["tests/ops/test_tiles.py"]

# (format, type, rows, k, seed). A tile needs rows % 16 == 0. mk used rows of real tensors; these shapes are small.
CASES = [
    ("iq3s", "IQ3_S", 16, 512, 701),
    ("iq3s", "IQ3_S", 32, 256, 702),
    ("q8", "Q8_0", 16, 320, 711),
    ("q8", "Q8_0", 32, 128, 712),
    ("q8", "Q8_0", 48, 64, 713),
    ("q6", "Q6_K", 16, 256, 721),
    ("q6", "Q6_K", 48, 256, 722),
    ("iq4", "IQ4_NL", 32, 256, 731),
]


def make(g, lib):
    sys.path.insert(0, str(lib.GGUF_PY))
    import gguf
    g.meta["cases"] = []
    for i, (fmt, qtype, rows, k, seed) in enumerate(CASES):
        raw = lib.raw_rows(qtype, seed, rows, k)
        want = gguf.quants.dequantize(raw, getattr(gguf.GGMLQuantizationType, qtype)).astype(np.float32).reshape(rows, k)
        assert np.isfinite(want).all(), (fmt, i)
        g.add(f"dq_{i}", want)
        g.meta["cases"].append({"fmt": fmt, "type": qtype, "rows": rows, "k": k, "seed": seed})
