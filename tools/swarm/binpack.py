"""Online bin packing, after FunSearch. A program defines priority(item, bins), which gives one score for each bin
that can hold the item. The evaluator puts each item into the bin with the highest score. The score is the mean
percent of excess bins over the L1 lower bound; lower is better. The worker reads a program on stdin and prints JSON.
The sandbox runs this file alone, so it imports only numpy and the standard library."""
import json, math, sys
import numpy as np

CAP = 100


def instances(seed, count=5, n=5000):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        items = np.clip(np.round(45 * rng.weibull(3.0, n)), 1, CAP).astype(float)
        out.append(items)
    return out


def pack(priority, items):
    bins = np.full(len(items), float(CAP))
    for item in items:
        valid = np.nonzero(bins - item >= 0)[0]
        scores = np.asarray(priority(item, bins[valid]), dtype=float)
        if scores.shape != valid.shape:
            raise ValueError(f"priority gave shape {scores.shape} for {valid.shape}")
        if np.any(np.isnan(scores)):
            raise ValueError("priority gave NaN")
        bins[valid[int(np.argmax(scores))]] -= item
    return int(np.sum(bins != CAP))


def excess(priority, seed):
    total = 0.0
    sets = instances(seed)
    for items in sets:
        used = pack(priority, items)
        lb = math.ceil(items.sum() / CAP)
        total += 100.0 * (used - lb) / lb
    return total / len(sets)


BEST_FIT = """import numpy as np
def priority(item, bins):
    return -(bins - item)
"""

if __name__ == "__main__":
    seed = int(sys.argv[1])
    src = sys.stdin.read()
    try:
        env = {}
        exec(compile(src, "<program>", "exec"), env)
        print(json.dumps({"excess": excess(env["priority"], seed)}))
    except Exception as e:  # noqa: BLE001 - the worker reports any failure of a program
        print(json.dumps({"fail": f"{type(e).__name__}: {str(e)[:200]}"}))
