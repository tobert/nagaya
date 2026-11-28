# SPDX-License-Identifier: MIT
"""GPU-capable, prompt-shardable J-Lens fitter for Inkling-2-Mini and Qwen3.8-Flash-Next.

Each worker processes its assigned shard of prompts, accumulating exact float64 sums on CPU and
checkpointing periodically. After all shards finish, run tools/jlens/merge.py to combine shards
into a canonical D10 .jlens.safetensors file.

Example (shard 0 of 8 on Kubernetes IndexedJob):
  python3 -m tools.jlens.fit \
    --model-kind inkling2 \
    --weight-hash blake3:... \
    --n 100 \
    --shard-index "${JOB_COMPLETION_INDEX:-0}" \
    --num-shards 8 \
    --out-dir /data/runs/inkling2
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

from .adapters import DEFAULT_REPOS, load_model
from .common import SEED, load_corpus, shard_indices, validate_weight_hash
from .estimator import estimate_prompt_jacobians
from .merge import merge_shards

log = logging.getLogger(__name__)


def _save_atomic(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit a J-Lens shard on GPU or CPU.")
    parser.add_argument("--model-kind", required=True, choices=sorted(DEFAULT_REPOS))
    parser.add_argument("--model", default=None, help="HF repo id or local snapshot path (default per kind)")
    parser.add_argument("--revision", default=None, help="Optional HF git revision to pin")
    parser.add_argument(
        "--weight-hash",
        required=True,
        help="blake3:<64 hex> weight_hash from nagaya GET /council/v2/identity",
    )
    parser.add_argument(
        "--corpus",
        default="wikitext-103",
        help="Local JSON list of prose paragraphs, or 'wikitext-103' to pull from HF (default)",
    )
    parser.add_argument("--corpus-cache", type=Path, default=None, help="Optional path to cache extracted corpus JSON")
    parser.add_argument("--n", type=int, default=100, help="Total prompts across all shards (default: 100)")
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="0-based shard index (default: $JOB_COMPLETION_INDEX, or 0 when --num-shards 1)",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards (default: 1)")
    parser.add_argument("--dim-batch", type=int, default=16, help="Cotangent batch size per VJP call (default: 16)")
    parser.add_argument("--vjp-mode", choices=("auto", "batched", "loop"), default="auto")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Max tokens per prompt (default: 128)")
    parser.add_argument("--dtype", choices=("bf16", "fp32", "fp16"), default="bf16")
    parser.add_argument("--no-ple-cpu", action="store_true", help="Keep Qwen3.8 PLE table on GPU instead of CPU RAM")
    parser.add_argument("--max-memory-gib", type=int, default=None, help="Per-GPU memory cap in GiB for device_map")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save shard checkpoint every N prompts")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for shard checkpoints")
    parser.add_argument(
        "--export-safetensors",
        type=Path,
        default=None,
        help="If --num-shards 1, also export the final canonical .jlens.safetensors here",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_weight_hash(args.weight_hash)

    if args.shard_index is None:
        env_idx = os.environ.get("JOB_COMPLETION_INDEX")
        if env_idx is not None:
            args.shard_index = int(env_idx)
        elif args.num_shards == 1:
            args.shard_index = 0
        else:
            sys.exit(
                f"--num-shards is {args.num_shards}, but neither --shard-index nor JOB_COMPLETION_INDEX is set; "
                "refusing to default all workers to shard 0"
            )

    shard_idxs = shard_indices(args.n, args.shard_index, args.num_shards)
    ckpt_path = args.out_dir / f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}.pt"

    corpus_cache = args.corpus_cache or (args.out_dir / f"corpus-shard-{args.shard_index:03d}.json")
    prompts, corpus_sha256, corpus_desc = load_corpus(
        args.corpus, args.n, seed=SEED, cache_json=corpus_cache
    )
    log.info(
        "shard %d/%d: %d prompts (indices %s..%s), corpus=%s (%s)",
        args.shard_index,
        args.num_shards,
        len(shard_idxs),
        shard_idxs[0] if shard_idxs else "-",
        shard_idxs[-1] if shard_idxs else "-",
        corpus_desc,
        corpus_sha256[:12],
    )

    done_indices: list[int] = []
    sum_j: list[torch.Tensor] | None = None
    layer_ids: list[int] | None = None
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        for field, expected in (
            ("model_kind", args.model_kind),
            ("weight_hash", args.weight_hash),
            ("corpus_sha256", corpus_sha256),
            ("max_seq_len", args.max_seq_len),
            ("dtype", args.dtype),
        ):
            if ckpt.get(field) != expected:
                sys.exit(
                    f"checkpoint {ckpt_path} mismatch on {field!r}: found {ckpt.get(field)!r}, expected {expected!r}; "
                    "remove the checkpoint or match the original fit arguments"
                )
        if args.revision and ckpt.get("source_revision") and ckpt.get("source_revision") != args.revision:
            sys.exit(
                f"checkpoint {ckpt_path} was fitted on revision {ckpt.get('source_revision')!r}, "
                f"not {args.revision!r}"
            )
        done_indices = list(ckpt["prompt_indices"])
        sum_j = ckpt["sum_j"]
        layer_ids = list(ckpt["layer_ids"])
        log.info("resuming %s: %d/%d prompts already completed", ckpt_path.name, len(done_indices), len(shard_idxs))

    remaining = [idx for idx in shard_idxs if idx not in set(done_indices)]
    if not remaining:
        log.info("shard %d/%d already complete (%d prompts)", args.shard_index, args.num_shards, len(done_indices))
    else:
        t0 = time.time()
        loaded = load_model(
            args.model_kind,
            args.model,
            revision=args.revision,
            dtype_name=args.dtype,
            ple_on_cpu=not args.no_ple_cpu,
            max_memory_gib=args.max_memory_gib,
        )
        log.info("model loaded in %.1fs (input_device=%s)", time.time() - t0, loaded.input_device)

        for step, p_idx in enumerate(remaining, start=1):
            p_start = time.time()
            encoded = loaded.tokenizer(
                prompts[p_idx],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_len,
            )
            input_ids = encoded["input_ids"].to(loaded.input_device)
            jacs, run_layer_ids = estimate_prompt_jacobians(
                loaded.hooked,
                input_ids,
                dim_batch=args.dim_batch,
                vjp_mode=args.vjp_mode,
            )
            if sum_j is None:
                sum_j = jacs
                layer_ids = run_layer_ids
            else:
                if run_layer_ids != layer_ids:
                    raise RuntimeError(f"prompt {p_idx} produced layer_ids {run_layer_ids}, expected {layer_ids}")
                for k in range(len(sum_j)):
                    sum_j[k] += jacs[k]
            done_indices.append(p_idx)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elapsed = time.time() - p_start
            log.info(
                "shard %d/%d prompt %d (%d/%d, T=%d): %.1fs",
                args.shard_index,
                args.num_shards,
                p_idx,
                len(done_indices),
                len(shard_idxs),
                input_ids.shape[1],
                elapsed,
            )

            if step % args.checkpoint_every == 0 or step == len(remaining):
                _save_atomic(
                    {
                        "schema": 1,
                        "model_kind": loaded.kind,
                        "model_name": loaded.model_name,
                        "source_repo": loaded.repo_or_path,
                        "source_revision": loaded.revision or "",
                        "dtype": args.dtype,
                        "weight_hash": args.weight_hash,
                        "corpus_sha256": corpus_sha256,
                        "corpus_desc": corpus_desc,
                        "max_seq_len": args.max_seq_len,
                        "layer_ids": layer_ids,
                        "prompt_indices": done_indices,
                        "n_prompts": len(done_indices),
                        "sum_j": sum_j,
                    },
                    ckpt_path,
                )
                log.info("saved checkpoint %s (%d prompts)", ckpt_path, len(done_indices))

    if args.export_safetensors is not None:
        if args.num_shards != 1:
            sys.exit("--export-safetensors is only for --num-shards 1; use tools/jlens/merge.py for multi-shard runs")
        merge_shards([ckpt_path], args.export_safetensors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
