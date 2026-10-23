"""The group `moe`: the expected outputs of the MoE router and of the whole MoE block (test_moe_route.py, test_moe.py).

Route: for each logit maker and token count, the exact answers of mk's `expected` function. The integer arrays are
stored as SHA-256 digests of their little-endian bytes, and the weights as f32 values.

Block: the MoE block in float64 from the dequantized weights of two synthetic layers. mk compares against the HF module
`Qwen4ExpTextSparseMoeBlock` with the real weights of layers 0 and 3. The HF module needs the 512 experts as dense float
tensors, so this group follows the same maths in numpy: a softmax router, the top 10 renormalized, SwiGLU experts, and a
shared expert with a sigmoid gate. Only the experts that the router selects are dequantized.
"""
import hashlib
import sys

import numpy as np

MK_FILES = ["tests/ops/test_moe_route.py", "tests/ops/test_moe.py"]

E, K = 512, 10
GU_ROWS, DN_ROWS = 640, 2560
TILES = {0: (8, 8), 1: (128, 16)}
ROUTE_TS = [1, 7, 16, 17, 80, 81, 160, 256]
BLOCK_T = 6
POOL_BASES = {"a": 7000, "b": 7100}
ROUTER_SCALE = np.float32(0.02)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).astype("<u4" if a.dtype.kind == "u" else "<i4").tobytes()).hexdigest()


def route_logits(lib, maker, t):
    seed = t if maker == "grid" else 1000 + t
    lg = ((lib.integers(seed, t * E, 49).astype(np.int64) - 24) / 8).astype(np.float32).reshape(t, E)
    if maker == "skewed":
        lg[:, :12] += np.float32(4.0)
    return lg


def route_expected(logits, fast):
    t = logits.shape[0]
    idx = np.stack([np.lexsort((np.arange(E), -row))[:K] for row in logits]).astype(np.int32)
    z = logits.astype(np.float64)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    sel = np.take_along_axis(p, idx, 1)
    w = sel / sel.sum(1, keepdims=True)
    flat = idx.reshape(-1)
    cnt = np.bincount(flat, minlength=E).astype(np.uint32)
    off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.uint32)
    asg = np.concatenate([np.flatnonzero(flat == e) for e in range(E)]).astype(np.uint32)
    tok = (asg // K).astype(np.uint32)
    rows, toks = TILES[fast]
    tiles = (cnt + toks - 1) // toks
    igu = np.concatenate([[0], np.cumsum(2 * (GU_ROWS // rows) * tiles)]).astype(np.uint32)
    idn = np.concatenate([[0], np.cumsum((DN_ROWS // rows) * tiles)]).astype(np.uint32)
    return idx, w, cnt, off, asg, tok, igu, idn


def u64_range(lib, seed, start, n):
    """The values at indices start .. start + n - 1 of the stream of `seed` (lib.u64_stream from an offset)."""
    with np.errstate(over="ignore"):
        key = lib._mix(np.uint64(seed) * lib.GOLDEN ^ np.uint64(0x6F70735F74657374))
        index = np.arange(start, start + n, dtype=np.uint64)
        return lib._mix(key + (index + np.uint64(1)) * lib.GOLDEN)


def raw_slice(lib, qtype, seed, row0, nrows, k):
    """Rows row0 .. row0 + nrows - 1 of `raw_rows(qtype, seed, total_rows, k)`, as bytes [nrows, row bytes]."""
    values, block, d_off = lib.QTYPES[qtype]
    row = k // values * block
    start = row0 * row
    first, last = start // 8, (start + nrows * row + 7) // 8
    u = u64_range(lib, seed, first, last - first).astype("<u8").view(np.uint8)
    out = u[start - first * 8:start - first * 8 + nrows * row].copy().reshape(-1, block)
    j0 = start // block
    d = (np.uint64(0x1400) + u64_range(lib, seed + 0x10000000, j0, out.shape[0]) % np.uint64(0x1000)).astype("<u2")
    out[:, d_off:d_off + 2] = d.view(np.uint8).reshape(-1, 2)
    return out.reshape(nrows, row)


def block_reference(lib, gguf, types, base, t):
    s = base
    x = lib.normals(s + 9, t * 2560).reshape(t, 2560)
    router = (lib.normals(s + 7, E * 2560) * ROUTER_SCALE).reshape(E, 2560)
    shg = lib.normals(s + 8, 2560) * ROUTER_SCALE

    def deq(qtype, seed, row0, nrows, k):
        raw = raw_slice(lib, qtype, seed, row0, nrows, k)
        return gguf.quants.dequantize(raw, types[qtype]).astype(np.float64).reshape(nrows, k)

    logits = x.astype(np.float64) @ router.astype(np.float64).T
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    order = np.argsort(-p, axis=1, kind="stable")
    idx = order[:, :K]
    p10, p11 = np.take_along_axis(p, order[:, 9:10], 1)[:, 0], np.take_along_axis(p, order[:, 10:11], 1)[:, 0]
    w = np.take_along_axis(p, idx, 1)
    w /= w.sum(1, keepdims=True)
    silu = lambda v: v / (1.0 + np.exp(-v))
    xd = x.astype(np.float64)
    out = np.zeros((t, 2560))
    for e in sorted(set(idx.reshape(-1).tolist())):
        wg = deq("IQ3_S", s + 1, e * GU_ROWS, GU_ROWS, 2560)
        wu = deq("IQ3_S", s + 2, e * GU_ROWS, GU_ROWS, 2560)
        wd = deq("IQ4_NL", s + 3, e * DN_ROWS, DN_ROWS, 640)
        for ti, slot in zip(*np.where(idx == e)):
            h = silu(wg @ xd[ti]) * (wu @ xd[ti])
            out[ti] += w[ti, slot] * (wd @ h)
    sg, su, sd = deq("Q8_0", s + 4, 0, 640, 2560), deq("Q8_0", s + 5, 0, 640, 2560), deq("Q8_0", s + 6, 0, 2560, 640)
    shared = (sd @ (silu(xd @ sg.T) * (xd @ su.T)).T).T
    gate = 1.0 / (1.0 + np.exp(-(xd @ shg.astype(np.float64))))
    out += gate[:, None] * shared
    return idx.astype(np.int32), p10.astype(np.float32), p11.astype(np.float32), out.astype(np.float32)


def make(g, lib):
    g.meta["route_cases"] = []
    for maker in ("grid", "skewed"):
        for t in ROUTE_TS:
            lg = route_logits(lib, maker, t)
            e = [route_expected(lg, f) for f in (0, 1)]
            idx, w, cnt, off, asg, tok, igu0, idn0 = e[0]
            _, _, _, _, _, _, igu1, idn1 = e[1]
            assert all(np.array_equal(a, b) for a, b in zip(e[0][:6], e[1][:6]))
            g.add(f"route_w_{maker}_{t}", w.astype(np.float32))
            g.meta["route_cases"].append({
                "maker": maker, "t": t,
                "digest": {"idx": digest(idx), "cnt": digest(cnt), "off": digest(off), "asg": digest(asg),
                           "tok": digest(tok), "igu0": digest(igu0), "idn0": digest(idn0), "igu1": digest(igu1),
                           "idn1": digest(idn1)}})
    sys.path.insert(0, str(lib.GGUF_PY))
    import gguf
    T = gguf.GGMLQuantizationType
    types = {"IQ3_S": T.IQ3_S, "IQ4_NL": T.IQ4_NL, "Q8_0": T.Q8_0}
    g.meta["block_t"] = BLOCK_T
    g.meta["pool_bases"] = POOL_BASES
    for name, base in POOL_BASES.items():
        idx, p10, p11, out = block_reference(lib, gguf, types, base, BLOCK_T)
        g.add(f"block_idx_{name}", idx)
        g.add(f"block_p10_{name}", p10)
        g.add(f"block_p11_{name}", p11)
        g.add(f"block_out_{name}", out)
