# SPDX-License-Identifier: MIT
"""Runs the DSpark head (inkling2_dspark.py) over texts whose trunk features are saved, and counts accepted drafts.

  python3 inkling2_dspark_run.py MODEL_DIR TEXTS_JSON FEATS_DIR NAME...

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


def main():
    folder, texts_path, feats_dir, *names = sys.argv[1:]
    texts = {t["name"]: t for t in json.load(open(texts_path))}
    head = dspark.Head(folder)
    rows = []
    for name in names:
        t = texts[name]
        feats = load_file(f"{feats_dir}/{name}.safetensors")["feat"]
        blocks = run_text(head, t["ids"], t["start"], feats)
        acc = [b[1] for b in blocks]
        look = lookup.run(t["ids"], t["start"] + 1, 7, 1, 4)
        ls = lookup.summarize(look)
        print(f"{name} {t['kind']}: {len(blocks)} blocks, head accepted/block {sum(acc) / len(acc):.2f} "
              f"(hist {[acc.count(i) for i in range(8)]}), {sum(b[2] for b in blocks) / len(blocks):.2f} s/block; "
              f"lookup K=7 accepted/block {ls['accepted_per_block']:.2f}", flush=True)
        rows.append({"name": name, "kind": t["kind"], "head": acc, "lookup": [a for _, a in look]})
    json.dump(rows, open(f"{feats_dir}/head-results.json", "w"))


if __name__ == "__main__":
    main()
