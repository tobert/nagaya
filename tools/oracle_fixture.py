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

MK = Path(os.environ.get("QWEN38_REFERENCE_REPO", Path.home() / "src/megakernel-qwen38-flashnext-strixhalo"))
TOP = 64

# The most tokens that the mk binding takes in one context of the small class: 4096 (spec.rs CONTEXT_CLASSES). A prompt
# longer than 256 tokens (mk kMaxPassTokens) takes several passes.
MAX_TOKENS = 4096

AUTHOR = "written by Claude Sonnet 5.5 for this tool, 2026-10-20"

# Three short prompts and one long prompt, all fixed. The long prompt is two paragraphs of ordinary prose.
LONG = (
    'A harbour town keeps its own time. In the early morning the fishing boats come in with the tide, and the men on the quay stack grey crates of mackerel and herring beside the ice house. By seven the bakers have opened their shutters, and the smell of warm bread drifts down the hill past the church, the post office and the little school where the children line up in pairs. Gulls wheel overhead and quarrel over scraps. An old woman sweeps the steps of the chandlery, as she has done every day for forty years, and nods to each person who passes.\n'
    '\n'
    'The harbour master works in a narrow room above the customs shed. On his desk lie a brass barometer, a ledger of arrivals and a mug of tea that has gone cold. He writes down each vessel by name, by tonnage and by the hour of its mooring. When the weather turns, he rings the bell on the pier, and the boats still at sea make for the shelter of the breakwater. In winter the storms can last for days. The waves climb the sea wall and break over the road, and the town waits, with its lamps lit and its doors barred, until the wind drops.'
)

# Five paragraphs, 596 tokens: three passes of mk (256, 256 and 84). A sixth paragraph makes 721 tokens, and there mk
# and the reference disagree by 0.23 in a logit. The break is between 630 and 655 tokens (spec 15, question 8).
LONGER = (
    'The railway workshop stands at the end of the line, where the tracks fan out into sidings and the grass grows between the sleepers. Before dawn the lamps are lit in the long iron shed, and the first shift of fitters walks in out of the cold with their tin boxes under their arms. They hang their coats on the hooks beside the tool racks, and each man checks that his own spanners are where he left them. A locomotive waits on the pit road with its fire dropped and its boiler still warm, and a boy climbs along its frames with a can of oil, feeding every joint in turn.\n\n'
    'The foreman keeps a board by the door with chalk marks for every engine in the yard. A circle means that the engine is waiting for a part. A cross means that it is ready for the road. A line means that the men are working on it now. He rubs out the marks with his sleeve and writes them again as the day goes on, and the men glance at the board when they pass, as sailors glance at the sky. Nobody remembers who first drew the system, and nobody has ever proposed to change it.\n\n'
    'In the machine bay the lathes turn all day, and the floor shakes a little under the heavy cuts. A turner named Okafor is making a new axle box from a block of bronze. He measures it with calipers every few minutes and speaks to nobody. Beside him an apprentice sharpens a cutting tool on a wheel, throwing a fan of orange sparks across the concrete. When the axle box is done, it will be carried across the yard on a handcart, fitted to the wheel, and tested with a hammer. A good fit rings like a bell. A bad fit gives a dull note, and the work begins again.\n\n'
    'At noon the whistle blows, and the men sit on upturned crates in the sun to eat bread and cold meat. They talk about the price of coal, the football on Saturday, and the new timetable that the company has promised for the spring. The older men do not trust the timetable. They have seen three of them come and go, and each one was drawn by a clerk who had never stood on a platform in the rain. The younger men laugh at this, but they listen, because the older men are right more often than not.\n\n'
    'In the afternoon a train comes in late from the north, and the whole yard stirs. The driver leans from the cab and shouts that the brakes are binding on the third coach. Two fitters run along the track with their bags, and a third unhooks the coach and rolls it onto the siding by hand. They find a cracked shoe and a worn pin, and within the hour they have replaced both. The driver signs the slip without reading it, because he knows these men, and he knows that the coach will go out tonight as sound as the day it was built.'
)

PROMPTS = [
    ("plain", "The capital of France is", "fixed in this tool"),
    ("code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return", "fixed in this tool"),
    ("chat", "<|im_start|>user\nName three colors.<|im_end|>\n<|im_start|>assistant\n", "fixed in this tool"),
    ("long", LONG, "prose " + AUTHOR),
    ("longer", LONGER, "prose " + AUTHOR),
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

    torch.set_num_threads(int(os.environ.get("QWEN38_REFERENCE_THREADS", "16")))
    tokenizer = Tokenizer.from_gguf(SHARDS[0])
    ids = [tokenizer.encode(text) for _, text, _ in PROMPTS]
    for (name, _, _), row in zip(PROMPTS, ids):
        if len(row) > MAX_TOKENS:
            sys.exit(f"prompt {name} has {len(row)} tokens; the mk binding takes {MAX_TOKENS}")
    model = reference.GGUFModel()
    started = time.time()

    def on_layer(il, _layer, _states):
        print(f"[{time.time() - started:6.0f}s] layer {il}", flush=True)

    sequences = [torch.tensor([row]) for row in ids]
    hidden = reference.stream_forward(model.config, sequences, model.embed, model.layer, model.head(),
                                      on_layer=on_layer, n_layers=args.layers)
    lm_head = model.lm_head()
    entries = []
    for (name, text, source), row, state in zip(PROMPTS, ids, hidden):
        logits = state[0, -1] @ lm_head.T
        top = torch.topk(logits, TOP)
        entries.append({
            "name": name,
            "source": source,
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
