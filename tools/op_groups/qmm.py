"""The group `qmm`: dequantized values of gguf-py for the raw rows of the generator, one case for each type."""

import numpy as np

SOURCE_FILES = ["tools/op_groups/qmm.py"]


def make(g, lib):
    raw_rows = lib.raw_rows
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
