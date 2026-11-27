# SPDX-License-Identifier: MIT
"""Runs the DSpark head (inkling2_dspark.py) over texts whose trunk features are saved, and counts accepted drafts.

  python3 inkling2_dspark_run.py [--cost rows:steps,...] [--head-cost STEPS] MODEL_DIR TEXTS_JSON FEATS_DIR NAME...

TEXTS_JSON is the list of {"name", "kind", "ids", "start"} of prompt_lookup_sim.py. FEATS_DIR/NAME.safetensors holds
"feat" [T, 4, 2304] bf16: the trunk's features at every position of ids (inputs of layer visits 5, 23, 41 and the
final-norm output; the lane's features.py writes them with the f32 reference of inkling2_reference.py).
The blocks follow the real process: the first anchor is ids[start] (the first token the model wrote), the head drafts 7
tokens, the longest prefix equal to the text is accepted, and the next anchor is the token after the accepted ones.
Each block is exact given its features: a position's context K/V come from the features at that position only.
"""
import json
import sys
import time

import torch
from safetensors.torch import load_file

import inkling2_dspark as dspark
import prompt_lookup_sim as lookup


def run_text(head, ids, start, feats):
    """List of (drafted, accepted, seconds) per block over ids[start:]."""
    n = len(ids)
    kv = head.context_kv(feats, torch.arange(n))
    blocks, a = [], start
    while a + 1 < n:
        t = time.time()
        p = a - 1
        drafts = head.block(ids[a], a, kv, feats[p:p + 1])
        truth = ids[a + 1:a + 1 + len(drafts)]
        k = dspark.accepted(drafts, truth)
        blocks.append((len(truth), k, time.time() - t))
        a += k + 1
    return blocks


def head_speed(acc, drafted, pts, head_cost):
    """Tokens per cost of the head's blocks, relative to plain decode. A block of `drafted` rows + 1 costs
    cost(rows) + head_cost steps."""
    tokens = sum(a + 1 for a in acc)
    return tokens / sum(lookup.cost(d + 1, pts) + head_cost for d in drafted)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Count the drafts of the DSpark head that the text of the model accepts.")
    ap.add_argument("model_dir", help="the checkpoint folder (holds mtp.safetensors and the model weights)")
    ap.add_argument("texts", help="JSON list of {name, kind, ids, start} (see prompt_lookup_sim.py)")
    ap.add_argument("feats_dir", help="folder of NAME.safetensors files with the trunk features 'feat' [T, 4, 2304]")
    ap.add_argument("names", nargs="+", help="names of texts in TEXTS_JSON")
    ap.add_argument("--cost", default=None, help="cost table rows:steps,... (default: the doc's estimate 1:1.0,8:1.8,16:2.5)")
    ap.add_argument("--head-cost", type=float, default=0.26,
                    help="steps that the head adds to a block (default 0.26: the doc's estimate of 7 ms over a 27 ms step)")
    args = ap.parse_args()
    try:
        pts = lookup.parse_points(args.cost) if args.cost else lookup.COST_POINTS
        texts = {t["name"]: t for t in json.load(open(args.texts))}
    except (ValueError, OSError) as e:
        sys.exit(f"inkling2_dspark_run: {e}")
    missing = [n for n in args.names if n not in texts]
    if missing:
        sys.exit(f"inkling2_dspark_run: {missing} not in {args.texts}; it holds {sorted(texts)}")
    label = "measured" if args.cost else "estimated"
    head = dspark.Head(args.model_dir)
    rows = []
    for name in args.names:
        t = texts[name]
        feats = load_file(f"{args.feats_dir}/{name}.safetensors")["feat"]
        blocks = run_text(head, t["ids"], t["start"], feats)
        if not blocks:
            print(f"{name} {t['kind']}: no blocks (no generated tokens after start)", flush=True)
            continue
        acc = [b[1] for b in blocks]
        look = lookup.run(t["ids"], t["start"] + 1, 7, 1, 4)
        ls = lookup.summarize(look, pts)
        print(f"{name} {t['kind']}: {len(blocks)} blocks of 7 drafts; head accepted/block {sum(acc) / len(acc):.2f} "
              f"(hist {[acc.count(i) for i in range(8)]}), speed {head_speed(acc, [b[0] for b in blocks], pts, args.head_cost):.2f} "
              f"({label} cost, head {args.head_cost}); lookup K=7 accepted/block {ls['accepted_per_block']:.2f}, "
              f"speed {ls['speed']:.2f}; host CPU time {sum(b[2] for b in blocks) / len(blocks):.2f} s/block "
              f"(oracle cost, not a GPU time)", flush=True)
        rows.append({"name": name, "kind": t["kind"], "head": acc, "lookup": [a for _, a in look]})
    json.dump(rows, open(f"{args.feats_dir}/head-results.json", "w"))


if __name__ == "__main__":
    main()
