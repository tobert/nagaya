# SPDX-License-Identifier: MIT
"""Validate a canonical .jlens.safetensors file, run optional interpretability probes, and upload to Hugging Face.

Examples:
  # Fast structural + finiteness + D10 alignment check (CPU only, no weights needed):
  python3 -m tools.jlens.validate --lens out/qwen38.jlens.safetensors --weight-hash blake3:...

  # Full model-backed probe check + upload to Hugging Face:
  python3 -m tools.jlens.validate --lens out/inkling2-runs.jlens.safetensors \
      --model-kind inkling2 --probe --upload-repo my-org/inkling2-mini-jlens
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .adapters import DEFAULT_REPOS, load_model
from .common import parse_canonical_safetensors, validate_weight_hash

log = logging.getLogger(__name__)

DEFAULT_PROBES = [
    "Fact: The currency used in the country shaped like a boot is",
    "The capital of France is",
    "In Python, to read a UTF-8 text file you call",
]


def validate_lens_file(
    lens_path: Path,
    *,
    expected_weight_hash: str | None = None,
) -> dict[str, Any]:
    """Run strict D10 canonical safetensors, metadata, shape, and numeric sanity checks."""
    raw = lens_path.read_bytes()
    tensors, meta = parse_canonical_safetensors(raw)

    if meta.get("nagaya.lens") != "jlens":
        raise ValueError(f"metadata nagaya.lens is {meta.get('nagaya.lens')!r}, expected 'jlens'")
    weight_hash = meta.get("weight_hash", "")
    validate_weight_hash(weight_hash)
    if expected_weight_hash is not None and weight_hash != expected_weight_hash:
        raise ValueError(f"lens weight_hash {weight_hash} != expected {expected_weight_hash}")

    if not tensors:
        raise ValueError("lens file holds zero tensors")

    shapes = {tuple(arr.shape) for arr in tensors.values()}
    if len(shapes) != 1:
        raise ValueError(f"inconsistent tensor shapes in lens file: {sorted(shapes)}")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError(f"expected 2D Jacobian matrices, got shape {shape}")

    norms = {}
    zero_layers = []
    for name, arr in tensors.items():
        frob = float(np.linalg.norm(arr.astype(np.float32)))
        norms[name] = round(frob, 4)
        if frob == 0.0:
            zero_layers.append(name)
    if zero_layers:
        raise ValueError(f"zero-norm Jacobian matrices found: {zero_layers}")

    return {
        "ok": True,
        "path": str(lens_path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "tensor_count": len(tensors),
        "shape": list(shape),
        "metadata": meta,
        "frobenius_norm_min": min(norms.values()),
        "frobenius_norm_max": max(norms.values()),
    }


@torch.no_grad()
def run_probes(
    lens_path: Path,
    model_kind: str,
    model_id: str | None = None,
    *,
    revision: str | None = None,
    dtype_name: str = "bf16",
    probes: list[str] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run interpretability probes through the model and compare J-Lens readouts against the final layer."""
    if model_kind == "qwen38" and not torch.cuda.is_available():
        raise RuntimeError(
            "--probe for qwen38 requires CUDA GPUs (~250 GiB BF16 MoE + ~102 GiB PLE); "
            "run --probe on a GPU node rather than a CPU-only merge pod"
        )
    tensors, meta = parse_canonical_safetensors(lens_path.read_bytes())
    loaded = load_model(model_kind, model_id, revision=revision, dtype_name=dtype_name)
    view = meta.get("nagaya.lens.view", "layers")

    norm = loaded.hooked.norm
    lm_head = loaded.lm_head
    softcap = getattr(loaded.hf_model, "logit_softcap", None)
    norm_device = next(norm.parameters()).device
    norm_dtype = next(norm.parameters()).dtype

    results = []
    for text in probes or DEFAULT_PROBES:
        encoded = loaded.tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(loaded.input_device)

        captured: list[tuple[int, torch.Tensor]] = []

        def make_hook(layer_idx: int):
            def hook(_mod, _args, out):
                h = out[0] if isinstance(out, tuple) else out
                # Last token position, shape [D_in]
                captured.append((layer_idx, h[0, -1].detach().to(device="cpu", dtype=torch.float32)))

            return hook

        handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(loaded.hooked.layers)]
        try:
            out = loaded.hooked.run(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        model_logits = out.logits[0, -1].detach().float().cpu()
        model_top = [loaded.tokenizer.decode([int(i)]) for i in model_logits.topk(top_k).indices]

        # Build key -> activation vector matching the lens view.
        if view == "runs":
            act_map = {f"runs.{i}": vec for i, (_, vec) in enumerate(captured)}
        else:
            act_map = {}
            for _, (layer_idx, vec) in enumerate(captured):
                act_map[f"layers.{layer_idx}"] = vec

        keys = sorted(tensors.keys(), key=lambda k: int(k.split(".")[1]))
        sample_indices = sorted(set(list(range(0, len(keys), max(1, len(keys) // 8))) + [len(keys) - 1]))
        layer_reads = {}
        for idx in sample_indices:
            key = keys[idx]
            j_mat = torch.from_numpy(tensors[key].astype(np.float32))
            projected = j_mat @ act_map[key]
            normed = norm(projected.to(device=norm_device, dtype=norm_dtype).unsqueeze(0))
            logits = lm_head(normed)
            if callable(softcap):
                logits = softcap(logits)
            logits = logits[0].detach().float().cpu()
            layer_reads[key] = [loaded.tokenizer.decode([int(i)]) for i in logits.topk(top_k).indices]

        results.append({"prompt": text, "model_top": model_top, "jlens_top": layer_reads})
    return results


def _render_model_card(report: dict[str, Any], probes: list[dict[str, Any]] | None) -> str:
    meta = report["metadata"]
    lines = [
        "---",
        "tags:",
        "  - j-lens",
        "  - interpretability",
        "  - nagaya",
        "---",
        f"# J-Lens for `{meta.get('model', 'unknown')}`",
        "",
        "Fitted with `tools/jlens` from [nagaya](https://github.com/tobert/nagaya) using the",
        "reverse-mode Jacobian lens estimator (Gurnee et al., 2026).",
        "",
        "## Summary",
        "",
        f"- **Model:** `{meta.get('model')}` (`{meta.get('nagaya.lens.source_repo')}`)",
        f"- **Weight hash:** `{meta.get('weight_hash')}`",
        f"- **View:** `{meta.get('nagaya.lens.view', 'layers')}` ({report['tensor_count']} tensors of shape `{report['shape']}`, F16)",
        f"- **Prompts:** `{meta.get('nagaya.lens.prompts')}` (max_seq_len `{meta.get('nagaya.lens.max_seq_len')}`)",
        f"- **File SHA-256:** `{report['sha256']}`",
        "",
    ]
    if probes:
        lines.append("## Probe readout")
        lines.append("")
        for p in probes:
            lines.append(f"### `{p['prompt']}`")
            lines.append(f"- **Model:** `{p['model_top']}`")
            for k, toks in p["jlens_top"].items():
                lines.append(f"- **{k}:** `{toks}`")
            lines.append("")
    return "\n".join(lines)


def upload_to_hf(
    lens_path: Path,
    repo_id: str,
    report: dict[str, Any],
    probes: list[dict[str, Any]] | None = None,
    *,
    private: bool = False,
) -> str:
    """Upload validated .jlens.safetensors, validation JSON report, and README model card to Hugging Face."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(lens_path),
        path_in_repo=lens_path.name,
        repo_id=repo_id,
        repo_type="model",
    )
    report_bytes = json.dumps({"validation": report, "probes": probes}, indent=2).encode("utf-8")
    api.upload_file(
        path_or_fileobj=report_bytes,
        path_in_repo=f"{lens_path.stem}.validation.json",
        repo_id=repo_id,
        repo_type="model",
    )
    card_bytes = _render_model_card(report, probes).encode("utf-8")
    api.upload_file(
        path_or_fileobj=card_bytes,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    return f"https://huggingface.co/{repo_id}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and optionally probe/upload a J-Lens safetensors file.")
    parser.add_argument("--lens", type=Path, required=True, help="Path to .jlens.safetensors")
    parser.add_argument("--weight-hash", default=None, help="Optional expected blake3 weight_hash")
    parser.add_argument("--probe", action="store_true", help="Run interpretability probes using the model")
    parser.add_argument("--model-kind", choices=sorted(DEFAULT_REPOS), default=None)
    parser.add_argument("--model", default=None, help="HF repo id or local path for --probe")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp32", "fp16"), default="bf16")
    parser.add_argument("--report-out", type=Path, default=None, help="Write JSON validation report here")
    parser.add_argument("--upload-repo", default=None, help="Upload validated lens to this HF repo (e.g. org/name)")
    parser.add_argument("--private", action="store_true", help="Create HF repo as private if it does not exist")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = validate_lens_file(args.lens, expected_weight_hash=args.weight_hash)
    log.info(
        "structural validation OK: %d tensors %s, norm range [%.2f, %.2f], sha256=%s",
        report["tensor_count"],
        report["shape"],
        report["frobenius_norm_min"],
        report["frobenius_norm_max"],
        report["sha256"][:12],
    )

    probes = None
    if args.probe:
        if args.model_kind is None:
            raise SystemExit("--probe requires --model-kind (inkling2 or qwen38)")
        probes = run_probes(
            args.lens,
            args.model_kind,
            args.model,
            revision=args.revision,
            dtype_name=args.dtype,
        )
        for p in probes:
            print(f"\nProbe: {p['prompt']}", flush=True)
            for k, toks in p["jlens_top"].items():
                print(f"  {k:10s}: {toks}", flush=True)
            print(f"  {'model':10s}: {p['model_top']}", flush=True)

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps({"validation": report, "probes": probes}, indent=2) + "\n")

    if args.upload_repo is not None:
        url = upload_to_hf(args.lens, args.upload_repo, report, probes, private=args.private)
        log.info("uploaded to %s", url)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
