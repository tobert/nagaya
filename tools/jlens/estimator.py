# SPDX-License-Identifier: MIT
"""Reverse-mode VJP Jacobian estimator for J-Lens (Gurnee et al., 2026).

Estimator: for a prompt of length T, target pre-final-norm activation y [1, T, D_out], and each
hooked residual activation h_k [1, T, D_in]:

    J_k[i, :] = (1 / T) * sum_{s=0}^{T-1} d(sum_{t=0}^{T-1} y[0, t, i]) / d(h_k[0, s, :])

Because reverse-mode autodiff computes gradients with respect to every hooked layer/run h_k in a
single backward pass, the cost scales with D_out (2304 for Inkling-2-Mini, 2560 for Qwen3.8)
rather than D_in (10240 for Qwen3.8's 4-stream residual) or the number of layers/runs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import nn

log = logging.getLogger(__name__)

VjpMode = Literal["auto", "batched", "loop"]


@dataclass
class HookedModel:
    """Thin handle over a causal LM exposing the decoder layers to hook and the final norm module."""

    model: nn.Module
    layers: list[nn.Module]
    norm: nn.Module
    forward_fn: Callable[[torch.Tensor], object] | None = None

    def run(self, input_ids: torch.Tensor) -> object:
        if self.forward_fn is not None:
            return self.forward_fn(input_ids)
        return self.model(input_ids)


def _extract_hidden(output: object) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"unexpected layer output type {type(output).__name__}; expected Tensor or tuple")


def _forward_with_hooks(
    hooked: HookedModel,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
    """Run one forward pass with autograd enabled, capturing every layer/run output and the pre-norm target."""
    activations: list[torch.Tensor] = []
    layer_ids: list[int] = []
    target_box: list[torch.Tensor] = []

    def make_layer_hook(layer_idx: int):
        def hook(_module: nn.Module, _args: tuple, output: object):
            h = _extract_hidden(output)
            if not h.requires_grad:
                raise RuntimeError(
                    f"hooked activation at layer {layer_idx} does not require grad; "
                    "ensure torch.enable_grad() is active and inputs/weights preserve autograd"
                )
            activations.append(h)
            layer_ids.append(layer_idx)

        return hook

    def norm_pre_hook(_module: nn.Module, args: tuple):
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("final norm pre-hook did not receive a Tensor as args[0]")
        target_box.append(args[0])

    handles = [layer.register_forward_hook(make_layer_hook(i)) for i, layer in enumerate(hooked.layers)]
    handles.append(hooked.norm.register_forward_pre_hook(norm_pre_hook))
    try:
        with torch.enable_grad():
            hooked.run(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    if not target_box:
        raise RuntimeError("final norm pre-hook never fired during forward pass")
    if not activations:
        raise RuntimeError("no layer forward hooks fired during forward pass")
    # If the norm ran multiple times (rare), take the last call before lm_head.
    target = target_box[-1]
    return target, activations, layer_ids


def estimate_prompt_jacobians(
    hooked: HookedModel,
    input_ids: torch.Tensor,
    *,
    dim_batch: int = 16,
    vjp_mode: VjpMode = "auto",
) -> tuple[list[torch.Tensor], list[int]]:
    """Compute float64 CPU Jacobians [D_out, D_in] for every hooked run on one prompt.

    Returns (jacobians, layer_ids), where jacobians[k] has shape [D_out, D_in] in float64 on CPU,
    and layer_ids[k] is the physical layer index that produced activation k.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected input_ids shape [1, T], got {tuple(input_ids.shape)}")
    if dim_batch < 1:
        raise ValueError(f"dim_batch must be >= 1, got {dim_batch}")

    target, activations, layer_ids = _forward_with_hooks(hooked, input_ids)
    if target.ndim != 3 or target.shape[0] != 1:
        raise ValueError(f"expected target shape [1, T, D_out], got {tuple(target.shape)}")

    t_len = target.shape[1]
    d_out = target.shape[2]
    scale = 1.0 / float(t_len)
    results = [
        torch.zeros((d_out, act.shape[-1]), dtype=torch.float64, device="cpu")
        for act in activations
    ]

    use_batched = vjp_mode in ("auto", "batched")
    start = 0
    while start < d_out:
        end = min(start + dim_batch, d_out)
        bsz = end - start
        retain = end < d_out

        if use_batched:
            grad_outputs = torch.zeros(
                (bsz, 1, t_len, d_out),
                dtype=target.dtype,
                device=target.device,
            )
            for b in range(bsz):
                grad_outputs[b, 0, :, start + b] = scale
            # Keep the graph alive if auto-fallback might need to re-run this chunk in loop mode.
            batched_retain = True if vjp_mode == "auto" else retain
            try:
                grads = torch.autograd.grad(
                    outputs=target,
                    inputs=activations,
                    grad_outputs=grad_outputs,
                    retain_graph=batched_retain,
                    is_grads_batched=True,
                    allow_unused=True,
                )
            except (RuntimeError, NotImplementedError, ValueError) as exc:
                if vjp_mode == "batched":
                    raise
                log.warning(
                    "batched VJP failed (%s); falling back to sequential VJP loop for remaining dims",
                    exc,
                )
                use_batched = False
                continue

            for k, grad in enumerate(grads):
                if grad is None:
                    raise RuntimeError(
                        f"hooked activation {k} (layer {layer_ids[k]}) received no gradient from final norm input; "
                        "check that the hook captures a tensor on the active autograd path"
                    )
                # grad has shape [bsz, 1, T, D_in]; cotangent already carries 1/T.
                results[k][start:end] = grad[:, 0].sum(dim=1).to(device="cpu", dtype=torch.float64)
            start = end
        else:
            for dim in range(start, end):
                retain_dim = dim < d_out - 1
                grad_out = torch.zeros_like(target)
                grad_out[0, :, dim] = scale
                grads = torch.autograd.grad(
                    outputs=target,
                    inputs=activations,
                    grad_outputs=grad_out,
                    retain_graph=retain_dim,
                    allow_unused=True,
                )
                for k, grad in enumerate(grads):
                    if grad is None:
                        raise RuntimeError(
                            f"hooked activation {k} (layer {layer_ids[k]}) received no gradient from final norm input; "
                            "check that the hook captures a tensor on the active autograd path"
                        )
                    results[k][dim] = grad[0].sum(dim=0).to(device="cpu", dtype=torch.float64)
            start = end

    return results, layer_ids
