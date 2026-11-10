"""Writes the fixture of the ensemble rules (docs/lab.md, "`ensemble`"): three random logit rows, and the combined row
of each rule, from NumPy in f64. The Rust test compares its f32 rows with these.

    python3 -I tools/ensemble_fixture.py crates/nagaya-council/tests/fixtures/ensemble.json
"""
import json
import sys

import numpy as np


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def main(out):
    rng = np.random.default_rng(29)
    rows = [(rng.standard_normal(256) * s).astype(np.float32) for s in (2.0, 4.0, 1.0)]
    weights = [1.0, 0.5, 2.0]
    ls = [log_softmax(r) for r in rows]
    logprob_sum = sum(w * l for w, l in zip(weights, ls))
    w = np.array(weights) / sum(weights)
    prob_mean = np.log(sum(wk * np.exp(l) for wk, l in zip(w, ls)))
    strength = 0.5
    contrast = (1 + strength) * ls[0] - strength * ls[1]
    fixture = {
        "rows": [r.tolist() for r in rows],
        "weights": weights,
        "strength": strength,
        "logprob_sum": logprob_sum.tolist(),
        "prob_mean": prob_mean.tolist(),
        "contrast": contrast.tolist(),
        "numpy": np.__version__,
    }
    with open(out, "w") as f:
        json.dump(fixture, f)


if __name__ == "__main__":
    main(sys.argv[1])
