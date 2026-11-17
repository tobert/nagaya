# SPDX-License-Identifier: MIT
"""The float64 reference of the attention cell mask: cell ranges [lo, hi) of the context that no attention read may use.
Masked cells keep their positions; they stop being visible. `tools/op_groups/attn_a.py` writes its fixture from it.

The definition: for query token t (n_vis = p0 + t + 1 visible cells, nb = n_vis // 4 complete blocks) a block is clear
if none of its cells is masked; only clear blocks are scored and compete. Dense (every unmasked visible cell) when the
clear blocks number <= budget / 4; else the top budget / 4 clear blocks by (score desc, index asc), plus the unmasked
cells of blocks that straddle a mask edge (always taken, like the tail), plus the unmasked tail cells. Emitted in
ascending cell order, -1 padding to the masked stride budget + 3 + 2 * 4 * 3.

The reference works from the raw projections: RMSNorm + rope of Q, K and the index Q, the new cells written by prep, the
block scores of the clear blocks, and attention over exactly the selected cells (masked cells removed). These functions
come unchanged from the op test of the first Qwen3.8 engine; the test half, which called the HIP op, did not come.
"""
from __future__ import annotations

import numpy as np

TOL = 5e-4
SCORE_C = 8.0
U = 2.0 ** -24
H, KVH, HD, IH, ID = 24, 2, 256, 4, 128
RATIO = 4
EPS = 1e-6
MASK_MAX = 4


def max_sel_masked(budget):
    return budget + RATIO - 1 + 2 * MASK_MAX * (RATIO - 1)


def inv_freq():
    return (1.0 / np.power(np.float32(1e7), np.arange(0, 64, 2, dtype=np.float32) / np.float32(64))).astype(np.float32)


# ---- the reference (float64) ------------------------------------------------------------------------------------

def masked_cells(ranges, n):
    m = np.zeros(n, bool)
    for lo, hi in ranges:
        m[lo:hi] = True
    return m


def norm_rope(x, g, pos, inv):
    x = np.asarray(x, np.float64)
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * g.astype(np.float64)
    ang = (np.float32(pos) * inv).astype(np.float64)  # the op's f32 angle
    cs, sn = np.cos(ang), np.sin(ang)
    a, b = y[..., :32].copy(), y[..., 32:64].copy()
    y[..., :32] = a * cs - b * sn
    y[..., 32:64] = b * cs + a * sn
    return y


def reference(c, p0, t):
    """Q [t, 24, 256], gate, K/V cache [p0 + t, 2, 2, 256] and raw index keys with the new cells, index Q [t, 4, 128]."""
    kv = c["kv"].astype(np.float64)
    ik = c["ik"].astype(np.float64)
    q = np.zeros((t, H, HD))
    gate = np.zeros((t, H, HD))
    iq = np.zeros((t, IH, ID))
    for i in range(t):
        pos = p0 + i
        qr = c["q_raw"][i].reshape(H, 2 * HD)
        q[i] = norm_rope(qr[:, :HD], c["q_g"], pos, c["inv"])
        gate[i] = qr[:, HD:]
        kv[pos, 0] = norm_rope(c["k_raw"][i].reshape(KVH, HD), c["k_g"], pos, c["inv"])
        kv[pos, 1] = c["v_raw"][i].reshape(KVH, HD)
        iq[i] = norm_rope(c["iq_raw"][i].reshape(IH, ID), c["iq_g"], pos, c["inv"])
        ik[pos] = c["ik_raw"][i]
    return q, gate, kv, ik, iq


def ref_scores(iq_t, ik, nb, c):
    """(scores [nb], S_abs [nb]) of every block from the reference's raw keys."""
    pooled = ik[: nb * RATIO].reshape(nb, RATIO, ID).mean(1)
    k = np.stack([norm_rope(pooled[b], c["ik_g"], b * RATIO, c["inv"]) for b in range(nb)]) if nb else np.zeros((0, ID))
    dots = k @ iq_t.T  # [nb, 4]
    s = np.maximum(dots, 0).sum(1) / np.sqrt(ID)
    sabs = (np.abs(k)[:, None, :] * np.abs(iq_t)[None, :, :]).sum((1, 2)) / np.sqrt(ID)
    return s, sabs


def definition(n_vis, scores_row, ranges, budget):
    """The masked selection (the module docstring) from a row of block scores: the ascending cell list."""
    masked = masked_cells(ranges, n_vis)
    nb = n_vis // RATIO
    clear = np.array([not masked[b * RATIO:(b + 1) * RATIO].any() for b in range(nb)], bool)
    topk = budget // RATIO
    chosen = clear.copy()
    if clear.sum() > topk:
        idx = np.flatnonzero(clear)
        s = scores_row[idx].astype(np.float64)
        order = np.lexsort((idx, -s))
        chosen = np.zeros(nb, bool)
        chosen[idx[order[:topk]]] = True
    return [cc for cc in range(n_vis) if not masked[cc] and (cc >= nb * RATIO or not clear[cc // RATIO] or chosen[cc // RATIO])]


def ref_attend(q, gate, kv, cells):
    out = np.zeros((H, HD))
    for h in range(H):
        kh = h // (H // KVH)
        k, v = kv[cells, 0, kh], kv[cells, 1, kh]
        x = k @ q[h] / 16.0
        p = np.exp(x - x.max())
        out[h] = (p @ v / p.sum()) / (1.0 + np.exp(-gate[h]))
    return out


def relerr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# (budget, p0, t, ranges, label). Toy budgets make the sparse path reachable at small sizes.
CASES = [
    (2048, 60, 8, [(10, 23)], "dense, one range straddling both ends"),
    (16, 60, 8, [(5, 14)], "sparse toy, straddling both ends"),
    (16, 61, 7, [(0, 4), (9, 10), (20, 33), (40, 61)], "sparse toy, four ranges: block-aligned, one cell, hi = p0"),
    (16, 40, 16, [(0, 40)], "everything before p0 masked"),
    (32, 100, 5, [(3, 97)], "dense after the mask (few clear blocks)"),
    (16, 77, 1, [(13, 51)], "sparse toy, T = 1 (the decode shape)"),
    (2048, 3000, 4, [(0, 1958)], "real budget, dense after the mask (the lab's coding seam)"),
    (2048, 4400, 3, [(0, 1958)], "real budget, sparse, the coding seam"),
    (2048, 4400, 2, [(101, 303), (2000, 2001), (2500, 2600)], "real budget, sparse, three ranges"),
]
