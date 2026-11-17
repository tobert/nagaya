# SPDX-License-Identifier: MIT
"""Make the oracle fixture for the Gemma 4 family (spec 8.3, "The Gemma 4 oracle", Proposed).

The tool runs the reference of `tools/gemma4_reference.py` on the CPU. The reference is HF transformers' own
Gemma 4 layers in f32, run one layer at a time over the checkpoint. For each fixed prompt the tool records the token
ids and, for the last position, the top-64 logits (ids and f32 values) and the logsumexp over the vocabulary. The
format is the format of the Qwen3.8 fixture (`tools/oracle_fixture.py`).

The tool reads the checkpoint, and it never writes into it. Each prompt runs the reference in its own
process, which writes its tensors into a temporary directory. Run it with the reference venv and no GPU job:

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 <reference venv>/bin/python -I tools/gemma4_oracle_fixture.py \
        OUT.json [--model DIR] [--layers N]

--model defaults to GEMMA4_MODEL_DIR. One of them must name the checkpoint folder.

With --layers N the reference stops after N layers and gives no logits. That run checks only the plumbing.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
TOP = 64
REFERENCE = ROOT / "tools/gemma4_reference.py"
TEMPLATE = ROOT / "crates/nagaya-text-gemma4/assets/gemma4_chat_template.jinja"


def qwen_prose():
    """The two prose prompts of the Qwen3.8 fixture, so that both families read the same text."""
    spec = importlib.util.spec_from_file_location("oracle_fixture", ROOT / "tools/oracle_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LONG, module.LONGER, module.AUTHOR


def chat_prompt():
    """A chat turn rendered with Google's template as transformers renders it, thinking off."""
    from transformers.utils.chat_template_utils import _compile_jinja_template

    template = _compile_jinja_template(TEMPLATE.read_text())
    return template.render(messages=[{"role": "user", "content": "Name three colors."}], add_generation_prompt=True,
                           enable_thinking=False, preserve_thinking=True, bos_token="<bos>")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--model", type=Path, default=os.environ.get("GEMMA4_MODEL_DIR"))
    parser.add_argument("--layers", type=int, default=None)
    args = parser.parse_args()
    if args.model is None:
        parser.error("no checkpoint folder: neither --model nor GEMMA4_MODEL_DIR names one. "
                     "Give the folder of the Gemma 4 checkpoint.")

    import torch
    import transformers
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    long, longer, author = qwen_prose()
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    bos = tokenizer.token_to_id("<bos>")
    chat = chat_prompt()
    prompts = [
        ("plain", "The capital of France is", "fixed in this tool"),
        ("code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return", "fixed in this tool"),
        ("chat", chat, "Google's template, rendered by transformers"),
        ("long", long, "prose " + author),
        ("longer", longer, "prose " + author),
    ]
    started = time.time()
    entries = []
    vocab = None
    with tempfile.TemporaryDirectory() as scratch:
        for name, text, source in prompts:
            # The chat text holds its own <bos>. The reference adds none for --ids, so the tool adds it to the others.
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if not ids or ids[0] != bos:
                ids = [bos] + ids
            out = Path(scratch) / f"{name}.safetensors"
            command = [sys.executable, "-I", str(REFERENCE), "--model", str(args.model), "--out", str(out),
                       "--ids", ",".join(map(str, ids))]
            if args.layers is not None:
                command += ["--layers", str(args.layers)]
            print(f"[{time.time() - started:6.0f}s] {name}: {len(ids)} tokens", flush=True)
            subprocess.run(command, check=True)
            tensors = load_file(str(out))
            entry = {"name": name, "source": source, "text": text, "ids": ids}
            if "logits_last" in tensors:
                logits = tensors["logits_last"]
                vocab = int(logits.shape[0])
                top = torch.topk(logits, TOP)
                entry.update({
                    "top_ids": top.indices.tolist(),
                    "top_logits": top.values.tolist(),
                    "logsumexp": torch.logsumexp(logits.double(), -1).item(),
                })
            entries.append(entry)
    with open(args.model / "config.json") as f:
        layers = json.load(f)["text_config"]["num_hidden_layers"]
    document = {
        "format": 1,
        "layers": args.layers if args.layers is not None else layers,
        "vocab": vocab,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "reference_sha": hashlib.sha256(REFERENCE.read_bytes()).hexdigest()[:16],
        "tokenizer_sha256": hashlib.sha256((args.model / "tokenizer.json").read_bytes()).hexdigest(),
        "seconds": round(time.time() - started),
        "prompts": entries,
    }
    Path(args.out).write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
