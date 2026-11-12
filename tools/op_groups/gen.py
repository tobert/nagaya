"""The group `gen`: the first values of each generator function, so that the Rust test can check its own generator."""
import numpy as np

SOURCE_FILES = []


def make(g, lib):
    normals, uniforms, integers, permutation, stream_bytes, u64_stream, raw_rows, QTYPES = (
        lib.normals, lib.uniforms, lib.integers, lib.permutation, lib.stream_bytes, lib.u64_stream, lib.raw_rows, lib.QTYPES)
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
        # Keep only the head of the bytes, so that the fixture stays small.
        g.add(f"raw_{qtype}", raw.reshape(-1)[:300])
        g.meta["raw"].append({"type": qtype, "seed": 19, "rows": rows, "k": k, "len": 300})
