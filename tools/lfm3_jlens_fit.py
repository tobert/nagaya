"""Fits the J-Lens of LFM3-1.7B on the CPU, and writes it as a lens object (docs/lab.md, "The J-Lens in the nagaya").

The fit is Anthropic's reference, `jacobian-lens` (Gurnee et al., 2026), imported from the clone that
--jlens-src or $JLENS_SRC names. The model is transformers with the checkpoint's own code, in float32
with eager attention, as tools/lfm3_reference.py loads it. Do not run it with a GPU job. The fit
runs on the CPU.

  <reference venv>/bin/python -I tools/lfm3_jlens_fit.py --model "$LFM3_MODEL_DIR" --jlens-src <clone> \
      --corpus prompts.json --weight-hash blake3:… --out lfm3.jlens.safetensors [--n 100]

--corpus is a JSON list of prose paragraphs. The first J-Lens fit, of Gemma 4 12B, used the WikiText-103
paragraphs of more than 700 characters. The tool takes the first --n of one fixed permutation,
so a larger --n resumes the checkpoint. --weight-hash is the `weight_hash` of the served model in
`GET /council/v2/identity`. The route refuses a lens of other weights.

The cost on a Ryzen 9 9950X (16 threads) was 2.18 s for each backward pass, 256 passes for each prompt:
9.3 minutes a prompt (2026-11-08). The checkpoint goes beside --out, every 5 prompts.

The output is canonical safetensors (engine-abi D10). It holds one F16 tensor `layers.<l>` of shape [2048, 2048]
for each layer l from 0 to 22. The lens of the last layer is the identity, so the file does not hold it. Its
metadata names the model, the weight hash, the corpus and the estimator.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

REVISION = "0dc4268b561499c648c7ba7c9466bf3d8d01517a"
SEED = 20261011
ALIGN = 4096


def detok(text):
    """WikiText's spacing back to prose, as the first J-Lens fit does: ' , ' to ', ', '( x )' to '(x)', '1 @,@ 000' to
    '1,000'."""
    text = re.sub(r" @(.)@ ", r"\1", text)
    text = re.sub(r" ([,.;:!?)\]}]|'s|'t|'re|'ve|'ll|'d|n't)", r"\1", text)
    text = re.sub(r"([(\[{]) ", r"\1", text)
    return re.sub(r'" (.*?) "', r'"\1"', text)


def choose(paragraphs, n):
    """The first n of one fixed permutation, so a larger n adds prompts and the checkpoint resumes."""
    order = np.random.default_rng(SEED).permutation(len(paragraphs))
    return [paragraphs[i] for i in order[:n]]


def canonical(tensors, metadata):
    """The bytes of a canonical safetensors file (engine-abi D10): sorted keys, a compact header padded with spaces
    to 4 KiB, the tensors in byte order of their names, and no padding between tensors."""
    header = {}
    offset = 0
    names = sorted(tensors, key=lambda name: name.encode())
    for name in names:
        dtype, shape, data = tensors[name]
        header[name] = {"data_offsets": [offset, offset + len(data)], "dtype": dtype, "shape": list(shape)}
        offset += len(data)
    header["__metadata__"] = {key: str(value) for key, value in metadata.items()}
    text = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    padding = (ALIGN - (8 + len(text)) % ALIGN) % ALIGN
    out = bytearray(struct.pack("<Q", len(text) + padding))
    out += text + b" " * padding
    for name in names:
        out += tensors[name][2]
    return bytes(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--weight-hash", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--jlens-src", type=Path, default=os.environ.get("JLENS_SRC"),
                        help="the clone of jacobian-lens (default: $JLENS_SRC)")
    args = parser.parse_args()
    if args.jlens_src is None:
        sys.exit("the clone of jacobian-lens is not known. Give --jlens-src <folder>, or set JLENS_SRC.")
    if not (Path(args.jlens_src) / "jlens").is_dir():
        sys.exit(f"{args.jlens_src} holds no folder jlens, so it is not a clone of jacobian-lens. Give the folder "
                 "of the clone.")
    if args.model.resolve().name != REVISION:
        sys.exit(f"{args.model} is not the pinned revision {REVISION} of LFM3-1.7B (spec 11.11)")
    if not re.fullmatch(r"blake3:[0-9a-f]{64}", args.weight_hash):
        sys.exit(f"--weight-hash {args.weight_hash!r} is not a blake3 content id; copy it from the identity")

    import torch

    sys.path.insert(0, str(args.jlens_src))
    import jlens
    from jlens.hf import Layout
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    corpus_bytes = args.corpus.read_bytes()
    prompts = choose([detok(p) for p in json.loads(corpus_bytes)], args.n)
    started = time.time()
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager", trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # LFM3 names its final norm `embedding_norm`, so the adapter of the reference does not find it by itself.
    model = jlens.from_hf(hf, tokenizer, layout=Layout("model", norm="embedding_norm"))
    print(f"{model}; {len(prompts)} prompts; loaded in {time.time() - started:.0f} s", flush=True)
    checkpoint = args.out.with_name(args.out.name + ".ckpt.pt")
    lens = jlens.fit(model, prompts=prompts, dim_batch=args.dim_batch, max_seq_len=args.max_seq_len,
                     checkpoint_path=str(checkpoint), checkpoint_every=5)

    tensors = {}
    for layer, matrix in lens.jacobians.items():
        data = matrix.to(torch.float16).contiguous().numpy().astype("<f2").tobytes()
        tensors[f"layers.{layer}"] = ("F16", tuple(matrix.shape), data)
    metadata = {
        "model": "lfm3",
        "weight_hash": args.weight_hash,
        "nagaya.lens": "jlens",
        "nagaya.lens.target_layer": hf.config.num_hidden_layers - 1,
        "nagaya.lens.prompts": lens.n_prompts,
        "nagaya.lens.corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "nagaya.lens.max_seq_len": args.max_seq_len,
        "nagaya.lens.estimator": "jacobian-lens fit: cotangents summed over target positions, mean over sources",
    }
    args.out.write_bytes(canonical(tensors, metadata))
    took = time.time() - started
    print(f"-> {args.out} ({took:.0f} s, {took / max(1, lens.n_prompts):.0f} s a prompt)", flush=True)
    probe = "Fact: The currency used in the country shaped like a boot is"
    lens_logits, model_logits, _ = lens.apply(model, probe, positions=[-2])
    for layer in sorted(lens_logits)[::4] + [max(lens_logits)]:
        print(f"  L{layer}: {[tokenizer.decode([t]) for t in lens_logits[layer][0].topk(6).indices]}", flush=True)
    print(f"  model: {[tokenizer.decode([t]) for t in model_logits[0].topk(6).indices]}", flush=True)


if __name__ == "__main__":
    main()
