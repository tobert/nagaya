"""Writes the reference files of the embedding oracle (spec 8.3) for EmbeddingGemma 2.

Run it through tools/embeddinggemma2_golden.sh, which names the files and their contents. The reference is
sentence-transformers on the CPU, in float32 from the bf16 weights, with eager attention. A second load in bf16 gives
the difference of the two dtypes, for the tolerance of the engine.
"""

import argparse
import json
import os
import sys
import time

import numpy
import safetensors
import sentence_transformers
import torch
import transformers
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer

TASKS = [
    "query",
    "document",
    "Clustering",
    "Classification",
    "SentenceSimilarity",
    "CodeRetrieval",
    "QuestionAnswering",
    "FactChecking",
]
CUTS = [512, 256, 128]
# The case "near_limit" has this many tokens with the prompt of "document" and the special tokens. The model takes
# 262,144 positions, and sentence-transformers cuts nothing, so 2048 is a choice of this oracle, not of the model.
NEAR_LIMIT = 2040
PARAGRAPH = (
    "The nagaya is one process per machine that hosts our engines. It admits each call before it allocates, and it "
    "keeps its store of content ids, recipes and entries on the disk. A ledger holds the budgets of the devices. "
)


def cases(tok):
    out = {
        "ascii": "The quick brown fox jumps over the lazy dog.",
        "code": 'fn main() { println!("hello, {}", 42); }',
        "cjk": "東京は日本の首都です。春には桜が咲きます。",
        "emoji": "I love pizza 🍕 and tacos 🌮! Do you? 🤔",
        "space": " ",
        "empty": "",
        # Past the sliding window of 512 positions on each side, so the window mask changes the result.
        "long": PARAGRAPH * 16,
    }
    prompt = "title: none | text: "
    words = (PARAGRAPH * 64).split(" ")
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        n = len(tok(prompt + " ".join(words[:mid]))["input_ids"])
        if n <= NEAR_LIMIT:
            lo = mid
        else:
            hi = mid - 1
    out["near_limit"] = " ".join(words[:lo])
    return out


def load(model, dtype):
    return SentenceTransformer(
        model, device="cpu", model_kwargs={"dtype": dtype, "attn_implementation": "eager"}
    )


def capture_ids(st):
    """Records the input ids of each forward of the Transformer module of sentence-transformers."""
    seen = []

    def hook(_module, args, kwargs):
        features = kwargs.get("features", args[0] if args else None)
        mask = features.get("attention_mask")
        seen.append((features["input_ids"].clone(), None if mask is None else mask.clone()))

    st[0].register_forward_pre_hook(hook, with_kwargs=True)
    return seen


def encode(st, text, task, dim=None):
    # With truncate_dim alone, sentence-transformers cuts after its Normalize module and does not normalize again.
    # normalize_embeddings=True normalizes the cut, as the rule of M19 says.
    return st.encode(
        [text], prompt_name=task, convert_to_tensor=True, truncate_dim=dim, normalize_embeddings=dim is not None
    )[0].float()


def parts(st, text, task):
    """The hidden states of one input, for the bring-up of the engine's kernels."""
    lm = st[0].model.language_model
    got = {}
    hooks = [
        lm.embed_tokens.register_forward_hook(lambda m, i, o: got.__setitem__("embed", o[0].clone())),
        lm.ple.register_forward_hook(lambda m, i, o: got.__setitem__("ple", o[0].clone())),
        lm.norm.register_forward_hook(lambda m, i, o: got.__setitem__("final_norm", o[0].clone())),
        lm.embedding_projection.register_forward_hook(
            lambda m, i, o: got.__setitem__("token_embeddings", o[0].clone())
        ),
    ]
    for n, layer in enumerate(lm.layers):
        hooks.append(
            layer.register_forward_hook(lambda m, i, o, n=n: got.__setitem__(f"layer{n}", o[0].clone()))
        )
    pooled = {}
    hooks.append(
        st[1].register_forward_hook(lambda m, i, o: pooled.__setitem__("v", o["sentence_embedding"][0].clone()))
    )
    vector = encode(st, text, task)
    for h in hooks:
        h.remove()
    got["pooled"] = pooled["v"]
    got["vector"] = vector
    return {f"parts.{k}": v.float().contiguous() for k, v in got.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    start = time.time()

    st = load(args.model, torch.float32)
    tok = st[0].processor.tokenizer if hasattr(st[0].processor, "tokenizer") else st[0].processor
    prompts = st.prompts
    seen = capture_ids(st)
    texts = cases(tok)

    tensors = {}
    meta_cases = []
    for name, text in texts.items():
        for task in TASKS:
            seen.clear()
            v = encode(st, text, task)
            if len(seen) != 1:
                sys.exit(f"{name}/{task}: expected one forward, got {len(seen)}")
            ids = seen[0][0][0]
            key = f"{name}.{task}"
            tensors[f"{key}.ids"] = ids.to(torch.int64).contiguous()
            tensors[f"{key}.v768"] = v.contiguous()
            for d in CUTS:
                cut = v[:d] / v[:d].norm()
                tensors[f"{key}.v{d}"] = cut.contiguous()
                st_cut = encode(st, text, task, d)
                diff = (st_cut - cut).abs().max().item()
                if diff > 1e-6:
                    sys.exit(f"{key}: the cut to {d} differs from sentence-transformers by {diff}")
            if name in ("ascii", "near_limit") or task in ("query", "document"):
                meta_cases.append(
                    {
                        "case": name,
                        "task": task,
                        "prompt": prompts[task],
                        "text": text if name != "near_limit" else f"<{len(text)} chars>",
                        "ids": ids.tolist() if len(ids) <= 64 else f"<{len(ids)} ids>",
                        "tokens": len(ids),
                    }
                )
            print(f"{key}: {len(ids)} tokens", flush=True)

    # The prompt joins the text as a plain prefix, with <bos> before and <eos> after.
    joined = tok(prompts["query"] + texts["ascii"])["input_ids"]
    if joined != tensors["ascii.query.ids"].tolist():
        sys.exit("the ids of the prompt joined with the text differ from the ids that the model saw")

    # A batch with padding gives the same vectors as one input at a time.
    batch_texts = [texts[n] for n in ("ascii", "cjk", "long", "empty")]
    batch = st.encode(batch_texts, prompt_name="document", convert_to_tensor=True, batch_size=4).float()
    batch_diff = max(
        (batch[i] - tensors[f"{n}.document.v768"]).abs().max().item()
        for i, n in enumerate(("ascii", "cjk", "long", "empty"))
    )

    tensors.update(parts(st, texts["ascii"], "query"))
    if (tensors["parts.vector"] - tensors["ascii.query.v768"]).abs().max().item() != 0.0:
        sys.exit("the vector of the parts run differs from the vector of the case")

    del st
    bf = load(args.model, torch.bfloat16)
    worst_abs, worst_cos = 0.0, 1.0
    for name, text in texts.items():
        for task in ("query", "document"):
            v = encode(bf, text, task)
            ref = tensors[f"{name}.{task}.v768"]
            tensors[f"{name}.{task}.bf16"] = v.contiguous()
            worst_abs = max(worst_abs, (v - ref).abs().max().item())
            worst_cos = min(worst_cos, torch.nn.functional.cosine_similarity(v, ref, dim=0).item())

    meta = {
        "model": os.path.realpath(args.model),
        "dtype": "float32 from the bf16 weights, eager attention, CPU",
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "safetensors": safetensors.__version__,
            "numpy": numpy.__version__,
        },
        "max_seq_length": str(bf.max_seq_length),
        "tasks": {t: prompts[t] for t in TASKS},
        "cuts": CUTS,
        "near_limit_tokens": NEAR_LIMIT,
        "batch_vs_single_max_abs": batch_diff,
        "bf16_vs_f32_max_abs": worst_abs,
        "bf16_vs_f32_min_cosine": worst_cos,
        "parts_input": {"case": "ascii", "task": "query"},
        "seconds": round(time.time() - start, 1),
        "cases": meta_cases,
    }
    os.makedirs(args.out, exist_ok=True)
    save_file(tensors, os.path.join(args.out, "golden.safetensors"), metadata={"meta": json.dumps(meta)})
    with open(os.path.join(args.out, "cases.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "cases"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
