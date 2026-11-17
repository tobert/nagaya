# SPDX-License-Identifier: MIT
"""Make the oracle fixture for the Qwen3.8 family (spec 8.3, milestone 11.5 spike).

The tool runs the reference on the CPU. The reference is the model's own transformers code, which streams one layer
at a time over the GGUF weights. For each fixed prompt the tool records the token ids and, for the last position, the
top-64 logits (ids and f32 values) and the logsumexp over the whole vocabulary.

The reference is the package `tools/qwen38_reference`. Run the tool with the venv of its `requirements.txt`, with
`QWEN38_SHARDS` set, and with no GPU job. It needs about 30 GiB of memory:

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 <venv>/bin/python -I tools/oracle_fixture.py OUT.json [--layers N]

With --layers N the reference stops after N layers. That output is a smoke test, not an oracle.

The option --residual OUT.json makes the oracle of the residual (engine-abi "The residual of the Qwen3.8 engine"). One
reference run takes three sequences: `plain`, `chat`, and `plain` again with an `add` (`plain_add`). A hook after each
layer records the 4 streams (10,240 f32 values) at the last position. The `add` of the fixed vector `v[l][j] =
(((31*j + 17*l) mod 257) - 128) / 256` goes to every position of the third sequence at the layers of ADD_LAYERS,
before the record. The tool also records the row that the head reads (the fold, 2,560 values) and the top-64 logits.
The JSON holds norms, samples and hashes. The full rows go into --golden DIR as little-endian f32 files,
`residual.<run>.f32` (48 x 10,240 values) and `fold.<run>.f32`. The tool prints the norms of each layer. Without
--strength, it runs the first two sequences only, prints the norms so that a person can choose a strength, and writes
nothing:

    ... tools/oracle_fixture.py --residual OUT.json --golden target/qwen38-golden --strength S
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
TOP = 64

# The most tokens that the engine binding takes in one context of the small class: 4096 (spec.rs CONTEXT_CLASSES). A
# prompt longer than 256 tokens (engine kMaxPassTokens) takes several passes.
MAX_TOKENS = 4096

AUTHOR = "written by Claude Sonnet 5.5 for this tool, 2026-10-20"

# Three short prompts and one long prompt, all fixed. The long prompt is two paragraphs of ordinary prose.
LONG = (
    'A harbour town keeps its own time. In the early morning the fishing boats come in with the tide, and the men on the quay stack grey crates of mackerel and herring beside the ice house. By seven the bakers have opened their shutters, and the smell of warm bread drifts down the hill past the church, the post office and the little school where the children line up in pairs. Gulls wheel overhead and quarrel over scraps. An old woman sweeps the steps of the chandlery, as she has done every day for forty years, and nods to each person who passes.\n'
    '\n'
    'The harbour master works in a narrow room above the customs shed. On his desk lie a brass barometer, a ledger of arrivals and a mug of tea that has gone cold. He writes down each vessel by name, by tonnage and by the hour of its mooring. When the weather turns, he rings the bell on the pier, and the boats still at sea make for the shelter of the breakwater. In winter the storms can last for days. The waves climb the sea wall and break over the road, and the town waits, with its lamps lit and its doors barred, until the wind drops.'
)

# Five paragraphs, 596 tokens: three passes of the engine (256, 256 and 84). A sixth paragraph makes 721 tokens, and
# there the engine and the reference disagree by 0.23 in a logit. The break is between 630 and 655 tokens (spec 15,
# question 8).
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


# ---- the oracle of the residual (--residual) ----------------------------------------------------------------------

STREAMS = 4
STREAM_WIDTH = 2560
WIDTH = STREAMS * STREAM_WIDTH  # the residual of a token: 10,240 f32 values
LAYERS = 48
ADD_LAYERS = (8, 24, 40)
SAMPLE = 64
VECTOR_FORMULA = "v[l][j] = (((31*j + 17*l) mod 257) - 128) / 256"
# The runs, in order: a name, the prompt (an index into PROMPTS) and whether the `add` applies.
RUNS = (("plain", 0, False), ("chat", 2, False), ("plain_add", 0, True))


def vector_numerator(layer: int, j: int) -> int:
    """The integer of `v[layer][j] * 256`. The value is exact in f32."""
    return ((31 * j + 17 * layer) % 257) - 128


def vector_value(layer: int, j: int) -> float:
    return vector_numerator(layer, j) / 256.0


def control_vector(layer: int, width: int = WIDTH) -> list:
    return [vector_value(layer, j) for j in range(width)]


def sample_indices(width: int) -> list:
    """64 fixed indices over `width`. The JSON names them, so a reader needs no rule."""
    step = width // SAMPLE
    return [i * step + 3 for i in range(SAMPLE)]


def require_float32(dtype_name: str) -> None:
    """Stop the tool when the hidden state is not f32. The rows are f32 bytes, and a cast would hide the break."""
    if dtype_name != "float32":
        sys.exit(f"the hidden state is {dtype_name}, not float32, so the tool writes no residual oracle: the rows are "
                 "f32 little-endian bytes; run the reference in float32")


def f32_bytes(values) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def f32_values(raw: bytes) -> tuple:
    if len(raw) % 4:
        raise ValueError(f"{len(raw)} bytes are not a whole number of f32 values")
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def norm(values) -> float:
    return math.sqrt(math.fsum(float(x) * float(x) for x in values))


def residual_entry(layer: int, raw: bytes, indices: list) -> dict:
    """The record of the 4 streams after one layer: the norm of each stream, 64 values and the hash of the bytes."""
    values = f32_values(raw)
    if len(values) != WIDTH:
        raise ValueError(f"layer {layer} has {len(values)} values; a residual has {WIDTH}")
    return {
        "layer": layer,
        "stream_norms": [norm(values[k * STREAM_WIDTH:(k + 1) * STREAM_WIDTH]) for k in range(STREAMS)],
        "sample": [values[i] for i in indices],
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def fold_entry(raw: bytes, indices: list) -> dict:
    """The record of the row that the head reads."""
    values = f32_values(raw)
    if len(values) != STREAM_WIDTH:
        raise ValueError(f"the fold has {len(values)} values; it has {STREAM_WIDTH}")
    return {"norm": norm(values), "sample": [values[i] for i in indices], "sha256": hashlib.sha256(raw).hexdigest()}


def residual_document(runs: list, header: dict, strength: float) -> dict:
    """The JSON of the residual oracle. `runs` holds one dict for each of RUNS (see `residual_main`)."""
    document = {
        "format": 1,
        "kind": "residual",
        "dtype": "float32",
        "layers": LAYERS,
        "width": WIDTH,
        "streams": STREAMS,
        "fold_width": STREAM_WIDTH,
        "sample_indices": sample_indices(WIDTH),
        "fold_sample_indices": sample_indices(STREAM_WIDTH),
        "add": {
            "formula": VECTOR_FORMULA,
            "strength": strength,
            "strength_f32": struct.unpack("<f", struct.pack("<f", strength))[0],
            "layers": list(ADD_LAYERS),
            "run": RUNS[-1][0],
            "where": "every position of the run, after the layer and before the record",
        },
    }
    document.update(header)
    document["runs"] = runs
    return document


def residual_files(run: str) -> tuple:
    """The names of the golden files of one run."""
    return f"residual.{run}.f32", f"fold.{run}.f32"


def residual_main(args) -> None:
    if args.layers is not None:
        sys.exit("--residual needs all 48 layers: do not give --layers")
    if args.strength is None:
        print("no --strength: the tool runs `plain` and `chat` only, prints the norms and writes nothing", flush=True)
    elif not math.isfinite(args.strength):
        sys.exit("--strength must be a finite number")
    if args.golden is None:
        sys.exit("--residual needs --golden DIR for the full rows")
    sys.path.insert(0, str(TOOLS))
    import torch
    import qwen38_reference
    from qwen38_reference import reference
    from qwen38_reference.tokenizer import Tokenizer
    import transformers

    torch.set_num_threads(int(os.environ.get("QWEN38_REFERENCE_THREADS", "16")))
    tokenizer = Tokenizer.from_gguf(qwen38_reference.shards_from_env()[0])
    runs = [run for run in RUNS if run[2] is False or args.strength is not None]
    ids = [tokenizer.encode(PROMPTS[prompt][1]) for _, prompt, _ in runs]
    model = reference.GGUFModel()
    started = time.time()
    print("norm of the vector at the layers of the add: " + ", ".join(
        f"layer {l} {norm(control_vector(l)):.3f}" for l in ADD_LAYERS), flush=True)
    indices = sample_indices(WIDTH)
    rows = [[] for _ in runs]  # the raw bytes of the residual after each layer, for each run
    scaled = {}
    if args.strength is not None:
        strength = torch.tensor(args.strength, dtype=torch.float32)
        for l in ADD_LAYERS:
            scaled[l] = torch.tensor(control_vector(l), dtype=torch.float32) * strength

    def on_layer(il, _layer, states):
        for k, (state, (name, _, add)) in enumerate(zip(states, runs)):
            require_float32(str(state.hidden.dtype).removeprefix("torch."))
            if add and il in scaled:
                state.hidden = state.hidden + scaled[il]
            last = state.hidden[0, -1].contiguous().numpy().astype("<f4", copy=False)
            rows[k].append(last.tobytes())
        stream_norms = " | ".join(
            f"{name} " + " ".join(f"{norm(f32_values(rows[k][-1])[s * STREAM_WIDTH:(s + 1) * STREAM_WIDTH]):8.2f}"
                                  for s in range(STREAMS))
            for k, (name, _, _) in enumerate(runs))
        print(f"[{time.time() - started:6.0f}s] layer {il:2d} stream norms: {stream_norms}", flush=True)

    sequences = [torch.tensor([row]) for row in ids]
    hidden = reference.stream_forward(model.config, sequences, model.embed, model.layer, model.head(),
                                      on_layer=on_layer)
    if args.strength is None:
        print("survey done: choose a strength from the norms, and run again with --strength", flush=True)
        return
    lm_head = model.lm_head()
    golden = Path(args.golden)
    golden.mkdir(parents=True, exist_ok=True)
    entries = []
    for k, ((name, prompt, add), row, state) in enumerate(zip(runs, ids, hidden)):
        fold = state[0, -1].contiguous().numpy().astype("<f4", copy=False).tobytes()
        logits = state[0, -1] @ lm_head.T
        top = torch.topk(logits, TOP)
        residual_name, fold_name = residual_files(name)
        (golden / residual_name).write_bytes(b"".join(rows[k]))
        (golden / fold_name).write_bytes(fold)
        entries.append({
            "name": name,
            "prompt": PROMPTS[prompt][0],
            "text": PROMPTS[prompt][1],
            "ids": row,
            "add": add,
            "residual_file": residual_name,
            "fold_file": fold_name,
            "rows": [residual_entry(l, raw, indices) for l, raw in enumerate(rows[k])],
            "fold": fold_entry(fold, sample_indices(STREAM_WIDTH)),
            "top_ids": top.indices.tolist(),
            "top_logits": top.values.tolist(),
            "logsumexp": torch.logsumexp(logits.double(), -1).item(),
        })
    header = {
        "transformers": transformers.__version__,
        "reference_sha": qwen38_reference.digest(),
        "seconds": round(time.time() - started),
        "vocab": int(logits.shape[0]),
        "author": AUTHOR,
    }
    Path(args.out).write_text(json.dumps(residual_document(entries, header, args.strength), indent=1) + "\n")
    print(f"wrote {args.out} and the rows in {golden} in {time.time() - started:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--residual", metavar="OUT.json", default=None,
                        help="make the oracle of the residual, the fold and an add (see the top of this file)")
    parser.add_argument("--golden", metavar="DIR", default=None, help="with --residual: the folder of the full rows")
    parser.add_argument("--strength", type=float, default=None, help="with --residual: the strength of the add")
    args = parser.parse_args()
    if args.residual is not None:
        if args.out is not None:
            parser.error("--residual takes its own output; give no OUT.json beside it")
        args.out = args.residual
        residual_main(args)
        return
    if args.out is None:
        parser.error("the following arguments are required: out")
    if args.golden is not None or args.strength is not None:
        parser.error("--golden and --strength belong to --residual")

    sys.path.insert(0, str(TOOLS))
    import torch
    import qwen38_reference
    from qwen38_reference import reference
    from qwen38_reference.tokenizer import Tokenizer
    import transformers

    torch.set_num_threads(int(os.environ.get("QWEN38_REFERENCE_THREADS", "16")))
    tokenizer = Tokenizer.from_gguf(qwen38_reference.shards_from_env()[0])
    ids = [tokenizer.encode(text) for _, text, _ in PROMPTS]
    for (name, _, _), row in zip(PROMPTS, ids):
        if len(row) > MAX_TOKENS:
            sys.exit(f"prompt {name} has {len(row)} tokens; the engine binding takes {MAX_TOKENS}")
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
        "reference_sha": qwen38_reference.digest(),
        "seconds": round(time.time() - started),
        "prompts": entries,
    }
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {args.out} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
