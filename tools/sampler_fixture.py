# SPDX-License-Identifier: MIT
"""Make the sampler fixture (spec 11.2, milestone 12 step 2).

The tool runs mk's `service/sampling.py` on the CPU, with no service and no GPU. For each case it records the
row of logits, the settings, the survivors with their probabilities, the uniform value that mk drew, mk's token, p and h.
The tool imports only `service.sampling` and `service.common` from mk's repository, and it never writes there.
Run it with mk's venv:

    PYTHONDONTWRITEBYTECODE=1 ~/src/megakernel-qwen38-flashnext-strixhalo/.venv/bin/python -I \
        tools/sampler_fixture.py crates/nagaya-council/tests/fixtures/sampler.json

Logits are float32 values, as the engine gives them. JSON keeps each one exactly.
"""
import json
import os
import sys
from pathlib import Path

MK = Path(os.environ.get("MK_REPO", Path.home() / "src/megakernel-qwen38-flashnext-strixhalo"))
sys.path.insert(0, str(MK))

import numpy as np  # noqa: E402

from service.sampling import SampleParams, Sampler  # noqa: E402

AUTHOR = "written by Claude Sonnet 5.5 for this tool, 2026-10-20"


def f32(values):
    return np.asarray(values, dtype=np.float32)


def rows():
    """The fixed logit rows. Each has a name."""
    rng = np.random.default_rng(20261020)
    realistic = (rng.standard_normal(1000) * 3.0 + 1.0).astype(np.float32)
    realistic[7] += 9.0  # a clear leader
    return [
        ("small", f32([2.0, 1.0, 0.5, -1.0, 0.0])),
        ("tie_all", f32([1.5] * 6)),
        ("tie_top", f32([3.0, 1.0, 3.0, 3.0, 0.0, 1.0, 3.0])),
        ("tie_at_cut", f32([5.0, 4.0, 4.0, 4.0, 1.0, 0.0])),
        ("two", f32([0.0, 0.0])),
        ("one", f32([-3.25])),
        ("extreme", f32([3.0e38, -3.0e38, 1.0e4, -1.0e4, 80.0, -80.0, 0.0])),
        ("wide_gap", f32([100.0, 0.0, -100.0, 99.5, 50.0])),
        ("realistic_1k", realistic),
    ]


def settings():
    """The fixed settings. A seed is added for each case."""
    base = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}
    out = [("plain_t1", dict(base))]
    for k in (1, 2, 3, 4, 20, 5000):
        out.append((f"top_k_{k}", dict(base, top_k=k)))
    for p in (0.1, 0.5, 0.9, 0.95, 0.999, 1.0):
        out.append((f"top_p_{p}", dict(base, top_p=p)))
    for m in (0.0, 0.01, 0.1, 0.5, 0.9, 0.999):
        out.append((f"min_p_{m}", dict(base, min_p=m)))
    for t in (0.0, 0.3, 0.6, 1.0, 2.5):
        out.append((f"temp_{t}", dict(base, temperature=t)))
    out.append(("defaults", {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}))
    out.append(("all", {"temperature": 0.8, "top_p": 0.9, "top_k": 3, "min_p": 0.05}))
    out.append(("all_hot", {"temperature": 1.7, "top_p": 0.8, "top_k": 50, "min_p": 0.02}))
    out.append(("k1_p_small", {"temperature": 1.0, "top_p": 0.01, "top_k": 1, "min_p": 0.0}))
    out.append(("greedy_with_filters", {"temperature": 0.0, "top_p": 0.5, "top_k": 2, "min_p": 0.5}))
    return out


SEEDS = (0, 1, 42, 2**63 + 12345, 2**64 - 1)


def main(out_path):
    cases = []
    for row_name, logits in rows():
        for set_name, s in settings():
            for seed in SEEDS:
                params = SampleParams(s["temperature"], s["top_p"], s["top_k"], s["min_p"], seed)
                sampler = Sampler(params)
                ids, probs = sampler.candidates(logits)
                u = float(np.random.Generator(np.random.PCG64(seed)).random())
                token, p, h = sampler.pick(logits)
                cases.append(
                    {
                        "name": f"{row_name}/{set_name}/{seed}",
                        "row": row_name,
                        "settings": dict(s, seed=seed),
                        "survivor_ids": [int(i) for i in ids],
                        "survivor_probs": [float(x) for x in probs],
                        "uniform": u,
                        "greedy": s["temperature"] == 0.0,
                        "token": int(token),
                        "p": float(p),
                        "h": float(h),
                    }
                )
    row_table = {name: [float(x) for x in logits] for name, logits in rows()}
    doc = {"author": AUTHOR, "source": "mk service/sampling.py", "rows": row_table, "cases": cases}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: sampler_fixture.py OUT.json")
    main(sys.argv[1])
