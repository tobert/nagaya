# SPDX-License-Identifier: MIT
"""Make the oracle fixture for the Qwen3.8 family (spec 8.3, milestone 11.5 spike).

The tool runs mk's reference on the CPU. The reference is the model's own transformers code, which streams one layer
at a time over the GGUF weights. For each fixed prompt the tool records the token ids and, for the last position, the
top-64 logits (ids and f32 values) and the logsumexp over the whole vocabulary.

The tool imports only from mk's repository, by an explicit path, and it never writes there. Run it with mk's venv:

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 \
        ~/src/megakernel-qwen38-flashnext-strixhalo/.venv/bin/python -I tools/oracle_fixture.py OUT.json [--layers N]

With --layers N the reference stops after N layers. That output is a smoke test, not an oracle.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

MK = Path(os.environ.get("MK_REPO", Path.home() / "src/megakernel-qwen38-flashnext-strixhalo"))
TOP = 64

# Three short, fixed prompts. They differ in language, in shape and in length.
PROMPTS = [
    ("plain", "The capital of France is"),
    ("code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return"),
    ("chat", "<|im_start|>user\nName three colors.<|im_end|>\n<|im_start|>assistant\n"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--layers", type=int, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(MK))
    import torch
    from harness import reference
    from harness.paths import SHARDS
    from harness.tokenizer import Tokenizer
    import transformers

    torch.set_num_threads(int(os.environ.get("MK_REF_THREADS", "16")))
    tokenizer = Tokenizer.from_gguf(SHARDS[0])
    ids = [tokenizer.encode(text) for _, text in PROMPTS]
    model = reference.GGUFModel()
    started = time.time()

    def on_layer(il, _layer, _states):
        print(f"[{time.time() - started:6.0f}s] layer {il}", flush=True)

    sequences = [torch.tensor([row]) for row in ids]
    hidden = reference.stream_forward(model.config, sequences, model.embed, model.layer, model.head(),
                                      on_layer=on_layer, n_layers=args.layers)
    lm_head = model.lm_head()
    entries = []
    for (name, text), row, state in zip(PROMPTS, ids, hidden):
        logits = state[0, -1] @ lm_head.T
        top = torch.topk(logits, TOP)
        entries.append({
            "name": name,
            "text": text,
            "ids": row,
            "top_ids": top.indices.tolist(),
            "top_logits": top.values.tolist(),
            "logsumexp": torch.logsumexp(logits.double(), -1).item(),
        })
    document = {
        "format": 1,
        "layers": args.layers if args.layers is not None else model.config.num_hidden_layers,
        "vocab": int(logits.shape[0]),
        "transformers": transformers.__version__,
        "reference_sha": hashlib.sha256((MK / "harness" / "reference.py").read_bytes()).hexdigest()[:16],
        "seconds": round(time.time() - started),
        "prompts": entries,
    }
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {args.out} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
