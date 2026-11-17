"""The group `attn_a`: the expected values of the masked attention op (test_attn_mask.py of the reference).

The inputs of a case come from the stateless generator, and the float64 reference model runs on them. A case stores, for each
token, the reference score and the score bound input `S_abs` of every block, the SHA-256 of the cells that the reference
selects, and the reference `attn_out` of two heads only (head 0 of the first key head, head 12 of the second). Two heads
keep the fixture small: the Rust test takes the relative L2 error over those two heads of each token.
"""
import hashlib

import numpy as np

SOURCE_FILES = ["tools/op_groups/attn_a.py", "tools/qwen38_reference/attn_mask.py"]
HEADS = (0, 12)


def case_arrays(lib, idx, p0, t, m):
    """The inputs of case `idx`: the same seeds and shapes as `make_case` in tests/ops/attn_a.rs."""
    base = 9000 + idx * 16
    h, kvh, hd, ih, idim = m.H, m.KVH, m.HD, m.IH, m.ID
    n = lambda s, *shape: lib.normals(base + s, int(np.prod(shape))).reshape(shape)
    gamma = lambda s, d: (np.float32(1) + np.float32(0.1) * n(s, d)).astype(np.float32)
    c = {
        "q_raw": n(0, t, h * 2 * hd), "k_raw": n(1, t, kvh * hd), "v_raw": n(2, t, kvh * hd),
        "iq_raw": n(3, t, ih * idim), "ik_raw": n(4, t, idim),
        "q_g": gamma(5, hd), "k_g": gamma(6, hd), "iq_g": gamma(7, idim), "ik_g": gamma(8, idim),
        "inv": m.inv_freq(),
        "kv": np.zeros((p0 + t, 2, kvh, hd), np.float32), "ik": np.zeros((p0 + t, idim), np.float32),
    }
    c["kv"][:p0] = n(9, p0, 2, kvh, hd)
    c["ik"][:p0] = n(10, p0, idim)
    return c


def make(g, lib):
    from qwen38_reference import attn_mask as m

    g.add("inv_freq", m.inv_freq())
    g.meta["cases"] = []
    for idx, (budget, p0, t, ranges, label) in enumerate(m.CASES):
        c = case_arrays(lib, idx, p0, t, m)
        q, gate, kv, ik, iq = m.reference(c, p0, t)
        scores, sabs, outs, digests = [], [], [], []
        for i in range(t):
            n_vis = p0 + i + 1
            nb = n_vis // m.RATIO
            s_ref, s_abs = m.ref_scores(iq[i], ik, nb, c)
            want = m.definition(n_vis, s_ref, ranges, budget)
            out = m.ref_attend(q[i], gate[i], kv, want)
            scores.append(s_ref.astype(np.float32))
            sabs.append(s_abs.astype(np.float32))
            outs.append(out[list(HEADS)].astype(np.float32))
            digests.append(hashlib.sha256(np.asarray(want, "<i4").tobytes()).hexdigest())
        g.add(f"score_{idx}", np.concatenate(scores))
        g.add(f"sabs_{idx}", np.concatenate(sabs))
        g.add(f"out_{idx}", np.stack(outs))
        g.meta["cases"].append({"label": label, "budget": budget, "p0": p0, "t": t, "ranges": ranges, "sha256": digests})
