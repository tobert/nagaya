# SPDX-License-Identifier: MIT
"""Model adapters for loading Inkling-2-Mini and Qwen3.8-Flash-Next for J-Lens fitting and probing.

Key architecture details:
  - Inkling-2-Mini (thinkingmachines/Inkling-2-Mini): 24 physical layers executed as 48 runs at
    3 cycles (0..5, 6..17 x3, 18..23), hidden_size=2304. ~37-40 GiB in BF16 (1x GPU/replica).
  - Qwen3.8-Flash-Next (Qwen/Qwen3.8-Flash-Next): 48 layers, 4-stream mHC residual (width 10240)
    folded to 2560 before final norm. ~250 GiB MoE + ~102 GiB PLE n-gram table in BF16.
    Keeping the PLE n-gram table on CPU RAM drops GPU VRAM to ~250 GiB (4x H100 / 2x B200 / 1x B300).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .estimator import HookedModel

log = logging.getLogger(__name__)

DEFAULT_REPOS = {
    "inkling2": "thinkingmachines/Inkling-2-Mini",
    "qwen38": "Qwen/Qwen3.8-Flash-Next",
}

DEFAULT_NAMES = {
    "inkling2": "inkling2-mini",
    "qwen38": "qwen38",
}


@dataclass
class LoadedModel:
    """Loaded HF model, tokenizer, HookedModel handle, and readout modules for validation probes."""

    kind: str
    model_name: str
    repo_or_path: str
    revision: str | None
    hf_model: nn.Module
    tokenizer: Any
    hooked: HookedModel
    lm_head: nn.Module
    input_device: torch.device


def _find_text_backbone(hf_model: nn.Module) -> nn.Module:
    """Locate the text transformer module holding `.layers` and `.norm` across VLM and CausalLM wrappers."""
    candidates = [
        hf_model,
        getattr(hf_model, "model", None),
        getattr(getattr(hf_model, "model", None), "language_model", None),
        getattr(hf_model, "language_model", None),
    ]
    for cand in candidates:
        if cand is not None and hasattr(cand, "layers") and hasattr(cand, "norm"):
            return cand
    raise RuntimeError(
        f"could not locate text backbone with .layers and .norm on {type(hf_model).__name__}"
    )


def _first_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def _disable_parameter_grads(model: nn.Module) -> None:
    """Disable parameter grads so autograd only tracks activation gradients (saves huge memory)."""
    for param in model.parameters():
        param.requires_grad_(False)


def _enable_embedding_output_grad(backbone: nn.Module) -> list[Any]:
    """Ensure the embedding output requires_grad=True so activation gradients flow through all layers."""
    embed = getattr(backbone, "embed_tokens", None) or getattr(backbone, "embed", None)
    if embed is None:
        raise RuntimeError(f"could not locate embed_tokens on {type(backbone).__name__}")

    def hook(_module: nn.Module, _args: tuple, output: torch.Tensor) -> torch.Tensor:
        if not output.requires_grad:
            return output.detach().requires_grad_(True)
        return output

    return [embed.register_forward_hook(hook)]


def _auto_model_cls():
    """Return AutoModelForImageTextToText (used by both Inkling2Loop and Qwen4Exp VLMs)."""
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText


def _build_qwen38_device_map(
    model_id: str,
    *,
    revision: str | None,
    ple_on_cpu: bool,
    max_memory_gib: int | None,
) -> Any:
    """Build a multi-GPU device_map for Qwen3.8 that pins the 51B PLE n-gram table to CPU RAM."""
    if not torch.cuda.is_available():
        return "cpu"
    n_gpus = torch.cuda.device_count()
    if n_gpus == 1 and not ple_on_cpu:
        return "cuda:0"

    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import AutoConfig

    auto_cls = _auto_model_cls()
    cfg = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    with init_empty_weights():
        empty = auto_cls.from_config(cfg, dtype=torch.bfloat16, trust_remote_code=True)

    max_memory: dict[Any, str] = {}
    for idx in range(n_gpus):
        if max_memory_gib is not None:
            gib = max_memory_gib
        else:
            total_gib = torch.cuda.get_device_properties(idx).total_memory // (1024**3)
            # Leave ~15% headroom for autograd activations and workspace.
            gib = max(1, int(total_gib * 0.85))
        max_memory[idx] = f"{gib}GiB"
    max_memory["cpu"] = "1024GiB"

    no_split = list(getattr(empty, "_no_split_modules", []) or [])
    dmap = infer_auto_device_map(
        empty,
        max_memory=max_memory,
        no_split_module_classes=no_split,
        dtype=torch.bfloat16,
    )
    if ple_on_cpu:
        # Locate actual PLE / n-gram embedding modules on the instantiated meta model and pin them to CPU.
        pinned_modules: list[str] = []
        pinned_bytes = 0
        for name, mod in empty.named_modules():
            parts = name.split(".")
            if any(p in ("ple", "ple_embedding", "ngram_embedding") for p in parts) and (
                parts[-1] in ("ple", "ple_embedding", "ngram_embedding")
            ):
                # Avoid double-counting child modules if a parent PLE module was already pinned.
                if any(name.startswith(parent + ".") for parent in pinned_modules):
                    continue
                dmap[name] = "cpu"
                pinned_modules.append(name)
                pinned_bytes += sum(p.numel() * 2 for p in mod.parameters())
        if not pinned_modules:
            raise RuntimeError(
                "ple_on_cpu=True was requested for Qwen3.8, but no PLE/n-gram module was found on "
                f"{type(empty).__name__}; pass --no-ple-cpu if running a model without PLE"
            )
        log.info(
            "pinned Qwen3.8 PLE modules %s to CPU RAM (~%.1f GiB BF16 saved in VRAM)",
            pinned_modules,
            pinned_bytes / (1024**3),
        )
    return dmap


def load_model(
    kind: str,
    model_id: str | None = None,
    *,
    revision: str | None = None,
    dtype_name: str = "bf16",
    ple_on_cpu: bool = True,
    max_memory_gib: int | None = None,
) -> LoadedModel:
    """Load Inkling-2-Mini or Qwen3.8-Flash-Next ready for activation-only reverse-mode J-Lens fitting."""
    if kind not in DEFAULT_REPOS:
        raise ValueError(f"unsupported model kind {kind!r}; expected one of {sorted(DEFAULT_REPOS)}")
    repo = model_id or DEFAULT_REPOS[kind]
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[dtype_name]

    from transformers import AutoTokenizer

    auto_cls = _auto_model_cls()
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)

    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": dtype,
        "attn_implementation": "eager",
        "trust_remote_code": True,
    }
    if kind == "qwen38":
        load_kwargs["device_map"] = _build_qwen38_device_map(
            repo, revision=revision, ple_on_cpu=ple_on_cpu, max_memory_gib=max_memory_gib
        )
    elif torch.cuda.is_available():
        load_kwargs["device_map"] = "auto" if torch.cuda.device_count() > 1 else {"": 0}

    log.info("loading %s (%s, rev=%s, dtype=%s)", kind, repo, revision or "default", dtype_name)
    hf_model = auto_cls.from_pretrained(repo, **load_kwargs).eval()
    _disable_parameter_grads(hf_model)

    backbone = _find_text_backbone(hf_model)
    # Keep the hook handle attached to the model so it lives as long as hf_model.
    hf_model._jlens_embed_hooks = _enable_embedding_output_grad(backbone)

    lm_head = getattr(hf_model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError(f"could not locate lm_head on {type(hf_model).__name__}")

    def forward_fn(input_ids: torch.Tensor) -> object:
        return hf_model(input_ids=input_ids, use_cache=False)

    resolved_revision = revision or getattr(hf_model.config, "_commit_hash", None)
    hooked = HookedModel(
        model=hf_model,
        layers=list(backbone.layers),
        norm=backbone.norm,
        forward_fn=forward_fn,
    )
    return LoadedModel(
        kind=kind,
        model_name=DEFAULT_NAMES[kind],
        repo_or_path=repo,
        revision=resolved_revision,
        hf_model=hf_model,
        tokenizer=tokenizer,
        hooked=hooked,
        lm_head=lm_head,
        input_device=_first_device(backbone),
    )
