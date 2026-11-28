# SPDX-License-Identifier: MIT
"""Merge shard checkpoints into a canonical D10 .jlens.safetensors file.

Supports two views for looped models (Inkling-2-Mini):
  - 'runs'   : all 48 execution runs as runs.0 .. runs.47 (default for inkling2)
  - 'layers' : last run of each physical layer as layers.0 .. layers.23 (default for qwen38)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from .common import canonical_safetensors, validate_weight_hash

log = logging.getLogger(__name__)


def merge_shards(
    shard_paths: list[Path],
    out_path: Path,
    *,
    view: str = "auto",
    expect_prompts: int | None = None,
) -> Path:
    """Merge one or more shard checkpoints into a canonical D10 safetensors lens file."""
    if not shard_paths:
        raise ValueError("no shard checkpoint files provided")
    if view not in ("auto", "runs", "layers"):
        raise ValueError(f"unsupported view {view!r}; expected 'auto', 'runs', or 'layers'")

    seen_prompts: set[int] = set()
    sum_j: list[torch.Tensor] | None = None
    ref: dict | None = None

    for path in sorted(shard_paths):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("schema") != 1:
            raise ValueError(f"{path}: unsupported checkpoint schema {ckpt.get('schema')!r}")
        ckpt_sum = ckpt.pop("sum_j")
        if ref is None:
            ref = ckpt
            validate_weight_hash(ref["weight_hash"])
            sum_j = [t.to(dtype=torch.float64).clone() for t in ckpt_sum]
        else:
            for key in ("model_kind", "weight_hash", "corpus_sha256", "max_seq_len", "layer_ids"):
                if ckpt.get(key) != ref.get(key):
                    raise ValueError(f"{path}: mismatch in {key!r}: {ckpt.get(key)!r} != {ref.get(key)!r}")
            for k, t in enumerate(ckpt_sum):
                sum_j[k] += t.to(dtype=torch.float64)
        del ckpt_sum

        idxs = list(ckpt["prompt_indices"])
        if len(idxs) != int(ckpt["n_prompts"]):
            raise ValueError(f"{path}: len(prompt_indices)={len(idxs)} != n_prompts={ckpt['n_prompts']}")
        overlap = seen_prompts.intersection(idxs)
        if overlap:
            raise ValueError(f"{path}: duplicate prompt indices across shards: {sorted(overlap)[:10]}")
        seen_prompts.update(idxs)

    assert ref is not None and sum_j is not None
    total_prompts = len(seen_prompts)
    if total_prompts == 0:
        raise ValueError("merged shards contain 0 prompts")
    if expect_prompts is not None and seen_prompts != set(range(expect_prompts)):
        missing = sorted(set(range(expect_prompts)) - seen_prompts)[:10]
        extra = sorted(seen_prompts - set(range(expect_prompts)))[:10]
        raise ValueError(
            f"expected prompt indices 0..{expect_prompts - 1}, got {total_prompts} prompts "
            f"(missing={missing}, extra={extra})"
        )

    layer_ids: list[int] = list(ref["layer_ids"])
    is_looped = len(layer_ids) != len(set(layer_ids))
    resolved_view = ("runs" if is_looped else "layers") if view == "auto" else view

    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    if resolved_view == "runs":
        for run_idx, acc in enumerate(sum_j):
            mean_f16 = (acc / float(total_prompts)).to(torch.float16).contiguous()
            data = mean_f16.numpy().astype("<f2").tobytes()
            tensors[f"runs.{run_idx}"] = ("F16", tuple(mean_f16.shape), data)
    else:
        # Map each physical layer id to its final execution run index.
        last_run_for_layer: dict[int, int] = {}
        for run_idx, phys_layer in enumerate(layer_ids):
            last_run_for_layer[phys_layer] = run_idx
        for phys_layer in sorted(last_run_for_layer):
            acc = sum_j[last_run_for_layer[phys_layer]]
            mean_f16 = (acc / float(total_prompts)).to(torch.float16).contiguous()
            data = mean_f16.numpy().astype("<f2").tobytes()
            tensors[f"layers.{phys_layer}"] = ("F16", tuple(mean_f16.shape), data)

    metadata = {
        "model": ref["model_name"],
        "weight_hash": ref["weight_hash"],
        "nagaya.lens": "jlens",
        "nagaya.lens.view": resolved_view,
        "nagaya.lens.fit_dtype": ref.get("dtype", "bf16"),
        "nagaya.lens.prompts": total_prompts,
        "nagaya.lens.corpus_sha256": ref["corpus_sha256"],
        "nagaya.lens.max_seq_len": ref["max_seq_len"],
        "nagaya.lens.source_repo": ref.get("source_repo", ""),
        "nagaya.lens.source_revision": ref.get("source_revision", ""),
        "nagaya.lens.estimator": "jacobian-lens fit: cotangents summed over target positions, mean over sources",
    }
    if resolved_view == "runs":
        metadata["nagaya.lens.run_layers"] = ",".join(str(x) for x in layer_ids)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(canonical_safetensors(tensors, metadata))
    tmp.replace(out_path)
    log.info(
        "merged %d shards (%d prompts, view=%s, %d tensors) -> %s (%.1f MiB)",
        len(shard_paths),
        total_prompts,
        resolved_view,
        len(tensors),
        out_path,
        out_path.stat().st_size / (1024 * 1024),
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge J-Lens shard checkpoints into canonical safetensors.")
    parser.add_argument("--shards-dir", type=Path, required=True, help="Directory holding shard-*.pt checkpoints")
    parser.add_argument("--glob", default="shard-*.pt", help="Glob pattern inside --shards-dir (default: shard-*.pt)")
    parser.add_argument("--out", type=Path, required=True, help="Output .jlens.safetensors path")
    parser.add_argument(
        "--view",
        choices=("auto", "runs", "layers"),
        default="auto",
        help="'runs' keeps all looped runs (runs.0..); 'layers' keeps last run per physical layer (layers.0..)",
    )
    parser.add_argument(
        "--expect-prompts",
        type=int,
        default=None,
        help="Optional exact total prompt count to verify contiguous 0..N-1 shard coverage",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    shard_paths = sorted(args.shards_dir.glob(args.glob))
    if not shard_paths:
        raise SystemExit(f"no files matched {args.glob!r} in {args.shards_dir}")
    merge_shards(shard_paths, args.out, view=args.view, expect_prompts=args.expect_prompts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
