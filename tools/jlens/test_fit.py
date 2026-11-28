# SPDX-License-Identifier: MIT
"""Unit tests for tools/jlens: canonical D10 safetensors, exact VJP Jacobians, sharding, merging, and validation.

Run with a Python that has torch and numpy (CPU is enough):
  python3 -m unittest tools/jlens/test_fit.py
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.jlens.common import (  # noqa: E402
    ALIGN,
    canonical_safetensors,
    choose,
    detok,
    parse_canonical_safetensors,
    shard_indices,
)
from tools.jlens.estimator import HookedModel, estimate_prompt_jacobians  # noqa: E402
from tools.jlens.merge import merge_shards  # noqa: E402
from tools.jlens.validate import validate_lens_file  # noqa: E402


class TinyMHCBlock(nn.Module):
    """Tiny causal residual block over 4 streams (width = 4 * d), mimicking Qwen3.8's mHC geometry."""

    def __init__(self, d: int, hc: int = 4):
        super().__init__()
        self.w = d * hc
        self.proj1 = nn.Linear(self.w, self.w, bias=False)
        self.proj2 = nn.Linear(self.w, self.w, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal prefix mix (cumulative average) so cross-position terms exist in the Jacobian.
        t = x.shape[1]
        mask = torch.tril(torch.ones(t, t, dtype=x.dtype, device=x.device))
        mask = mask / mask.sum(dim=-1, keepdim=True)
        mixed = torch.einsum("ts,bsd->btd", mask, x)
        return x + 0.1 * self.proj2(torch.tanh(self.proj1(mixed)))


class TinyMHCModel(nn.Module):
    """Synthetic 3-layer, 4-stream mHC model with a 4->1 stream fold before final RMSNorm."""

    def __init__(self, vocab: int = 32, d: int = 8, hc: int = 4, n_layers: int = 3):
        super().__init__()
        self.d = d
        self.hc = hc
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([TinyMHCBlock(d, hc) for _ in range(n_layers)])
        self.fold = nn.Linear(d * hc, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids).repeat(1, 1, self.hc)
        for layer in self.layers:
            h = layer(h)
        folded = self.fold(h)
        return self.lm_head(self.norm(folded))


class TinyLoopedBlock(nn.Module):
    """Tiny causal block for a looped transformer mimicking Inkling-2-Mini."""

    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        mask = torch.tril(torch.ones(t, t, dtype=x.dtype, device=x.device))
        mask = mask / mask.sum(dim=-1, keepdim=True)
        mixed = torch.einsum("ts,bsd->btd", mask, x)
        return x + 0.1 * self.fc2(torch.nn.functional.silu(self.fc1(mixed)))


class TinyLoopedModel(nn.Module):
    """Synthetic 6-layer looped model: layers 0..1 once, 2..3 for 3 cycles, 4..5 once -> 10 runs + fold."""

    def __init__(self, vocab: int = 32, d: int = 8, cycles: int = 3):
        super().__init__()
        self.d = d
        self.cycles = cycles
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([TinyLoopedBlock(d) for _ in range(6)])
        self.fold_w = nn.Parameter(torch.linspace(0.2, 1.0, 2))
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        h = self.layers[0](h)
        h = self.layers[1](h)
        early = h
        for _ in range(self.cycles):
            h = self.layers[2](h)
            h = self.layers[3](h)
        h = self.layers[4](h)
        h = self.layers[5](h)
        folded = self.fold_w[0] * early + self.fold_w[1] * h
        return self.lm_head(self.norm(folded))


def _exact_finite_diff_jacobian(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_hook_idx: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Computes the exact mean-over-sources, sum-over-targets Jacobian for one hooked run via central differences."""
    model.eval()
    t_len = input_ids.shape[1]

    def run_with_delta(delta: torch.Tensor | None) -> tuple[torch.Tensor, int]:
        seen = []
        captured_y = []

        def make_layer_hook():
            def hook(_mod, _args, out):
                idx = len(seen)
                seen.append(out.shape[-1])
                if delta is not None and idx == target_hook_idx:
                    return out + delta
                return out

            return hook

        def norm_pre_hook(_mod, args):
            captured_y.append(args[0])

        handles = [layer.register_forward_hook(make_layer_hook()) for layer in model.layers]
        handles.append(model.norm.register_forward_pre_hook(norm_pre_hook))
        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            for h in handles:
                h.remove()
        return captured_y[0][0], seen[target_hook_idx]

    y0, d_in = run_with_delta(None)
    d_out = y0.shape[-1]
    jac = torch.zeros(d_out, d_in, dtype=torch.float64)
    # Perturb every source position simultaneously in coordinate j -> sum_s d(sum_t y_t)/d(h_{s,j}), then divide by T.
    for j in range(d_in):
        delta = torch.zeros(1, t_len, d_in, dtype=y0.dtype)
        delta[0, :, j] = eps
        y_plus, _ = run_with_delta(delta)
        y_minus, _ = run_with_delta(-delta)
        # Sum over target positions t, divide by 2*eps and by T (mean over source positions s).
        jac[:, j] = ((y_plus.double() - y_minus.double()).sum(dim=0) / (2.0 * eps * t_len))
    return jac


class CommonTests(unittest.TestCase):
    def test_detok_and_choose_determinism(self):
        self.assertEqual(detok("Hello , world ! 1 @,@ 000 ( x ) ."), "Hello, world! 1,000 (x).")
        items = [f"p{i}" for i in range(50)]
        a = choose(items, 10)
        b = choose(items, 20)
        self.assertEqual(a, b[:10])
        self.assertEqual(len(set(b)), 20)

    def test_shard_indices_partition_exactly(self):
        for total in (1, 7, 100):
            for num_shards in (1, 2, 3, 8):
                all_idx = []
                for s in range(num_shards):
                    part = shard_indices(total, s, num_shards)
                    all_idx.extend(part)
                self.assertEqual(sorted(all_idx), list(range(total)))

    def test_canonical_safetensors_roundtrip_and_alignment(self):
        arr0 = np.arange(16, dtype=np.float16).reshape(4, 4)
        arr1 = np.ones((2, 8), dtype=np.float16)
        tensors = {
            "layers.1": ("F16", (2, 8), arr1.astype("<f2").tobytes()),
            "layers.0": ("F16", (4, 4), arr0.astype("<f2").tobytes()),
        }
        meta = {"model": "test", "weight_hash": "blake3:" + "a" * 64, "nagaya.lens": "jlens"}
        blob = canonical_safetensors(tensors, meta)
        header_len = struct.unpack("<Q", blob[:8])[0]
        self.assertEqual((8 + header_len) % ALIGN, 0)
        parsed_tensors, parsed_meta = parse_canonical_safetensors(blob)
        self.assertEqual(parsed_meta["nagaya.lens"], "jlens")
        np.testing.assert_array_equal(parsed_tensors["layers.0"], arr0)
        np.testing.assert_array_equal(parsed_tensors["layers.1"], arr1)

    def test_canonical_safetensors_rejects_bad_alignment_and_non_finite(self):
        arr = np.array([[1.0, float("nan")]], dtype=np.float16)
        blob = canonical_safetensors(
            {"layers.0": ("F16", (1, 2), arr.astype("<f2").tobytes())},
            {"nagaya.lens": "jlens", "weight_hash": "blake3:" + "0" * 64},
        )
        with self.assertRaises(ValueError):
            parse_canonical_safetensors(blob)

        # Unaligned header should also fail strict D10 check.
        raw_hdr = json.dumps({"__metadata__": {}, "a": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
        bad = struct.pack("<Q", len(raw_hdr)) + raw_hdr + np.array([1.0], dtype="<f2").tobytes()
        with self.assertRaises(ValueError):
            parse_canonical_safetensors(bad)


class EstimatorTests(unittest.TestCase):
    def test_mhc_rectangular_jacobian_matches_finite_diff_both_vjp_modes(self):
        torch.manual_seed(42)
        model = TinyMHCModel(vocab=32, d=6, hc=4, n_layers=3).double().eval()
        ids = torch.tensor([[1, 5, 9, 2, 7]], dtype=torch.long)
        hooked = HookedModel(model=model, layers=list(model.layers), norm=model.norm)

        jac_batched, layer_ids_b = estimate_prompt_jacobians(hooked, ids, dim_batch=3, vjp_mode="batched")
        jac_loop, layer_ids_l = estimate_prompt_jacobians(hooked, ids, dim_batch=3, vjp_mode="loop")
        self.assertEqual(layer_ids_b, [0, 1, 2])
        self.assertEqual(layer_ids_l, [0, 1, 2])
        self.assertEqual(len(jac_batched), 3)

        for idx in range(3):
            self.assertEqual(tuple(jac_batched[idx].shape), (6, 24))
            exact = _exact_finite_diff_jacobian(model, ids, idx, eps=1e-6)
            torch.testing.assert_close(jac_batched[idx], exact, atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(jac_loop[idx], exact, atol=1e-7, rtol=1e-6)

    def test_looped_model_jacobians_match_finite_diff(self):
        torch.manual_seed(7)
        model = TinyLoopedModel(vocab=32, d=6, cycles=3).double().eval()
        ids = torch.tensor([[3, 8, 1, 4]], dtype=torch.long)
        hooked = HookedModel(model=model, layers=list(model.layers), norm=model.norm)

        jacs, layer_ids = estimate_prompt_jacobians(hooked, ids, dim_batch=2, vjp_mode="batched")
        self.assertEqual(layer_ids, [0, 1, 2, 3, 2, 3, 2, 3, 4, 5])
        self.assertEqual(len(jacs), 10)
        for idx in (0, 3, 7, 9):
            self.assertEqual(tuple(jacs[idx].shape), (6, 6))
            exact = _exact_finite_diff_jacobian(model, ids, idx, eps=1e-6)
            torch.testing.assert_close(jacs[idx], exact, atol=1e-7, rtol=1e-6)


class ShardMergeValidateTests(unittest.TestCase):
    def test_shard_merge_equals_single_worker_and_supports_views(self):
        torch.manual_seed(123)
        model = TinyLoopedModel(vocab=32, d=6, cycles=3).eval()
        hooked = HookedModel(model=model, layers=list(model.layers), norm=model.norm)
        prompts = [
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[5, 6, 7, 8]]),
            torch.tensor([[2, 4, 6, 8]]),
            torch.tensor([[9, 1, 3, 5]]),
        ]

        # Single worker sum
        single_sum = None
        layer_ids = None
        for ids in prompts:
            jacs, layer_ids = estimate_prompt_jacobians(hooked, ids, dim_batch=3, vjp_mode="loop")
            if single_sum is None:
                single_sum = [j.clone() for j in jacs]
            else:
                for i in range(len(jacs)):
                    single_sum[i] += jacs[i]

        # Two shards of 2 prompts each
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            weight_hash = "blake3:" + "b" * 64
            for shard_id in (0, 1):
                idxs = shard_indices(len(prompts), shard_id, 2)
                shard_sum = None
                for p_idx in idxs:
                    jacs, _ = estimate_prompt_jacobians(hooked, prompts[p_idx], dim_batch=3, vjp_mode="loop")
                    if shard_sum is None:
                        shard_sum = [j.clone() for j in jacs]
                    else:
                        for i in range(len(jacs)):
                            shard_sum[i] += jacs[i]
                torch.save(
                    {
                        "schema": 1,
                        "model_kind": "inkling2",
                        "model_name": "inkling2-mini",
                        "weight_hash": weight_hash,
                        "corpus_sha256": "deadbeef",
                        "max_seq_len": 128,
                        "layer_ids": layer_ids,
                        "prompt_indices": idxs,
                        "n_prompts": len(idxs),
                        "sum_j": shard_sum,
                    },
                    tmp_path / f"shard-{shard_id:03d}.pt",
                )

            out_runs = tmp_path / "inkling2-runs.jlens.safetensors"
            merge_shards(
                shard_paths=[tmp_path / "shard-000.pt", tmp_path / "shard-001.pt"],
                out_path=out_runs,
                view="runs",
            )
            report_runs = validate_lens_file(out_runs, expected_weight_hash=weight_hash)
            self.assertTrue(report_runs["ok"])
            self.assertEqual(report_runs["tensor_count"], 10)

            tensors_runs, meta_runs = parse_canonical_safetensors(out_runs.read_bytes())
            self.assertEqual(meta_runs["nagaya.lens.prompts"], "4")
            for i in range(10):
                expected_f16 = (single_sum[i] / 4.0).to(torch.float16).numpy()
                np.testing.assert_array_equal(tensors_runs[f"runs.{i}"], expected_f16)

            # Also test "layers" view (last run of each physical layer 0..5).
            out_layers = tmp_path / "inkling2-layers.jlens.safetensors"
            merge_shards(
                shard_paths=[tmp_path / "shard-000.pt", tmp_path / "shard-001.pt"],
                out_path=out_layers,
                view="layers",
            )
            report_layers = validate_lens_file(out_layers, expected_weight_hash=weight_hash)
            self.assertTrue(report_layers["ok"])
            self.assertEqual(report_layers["tensor_count"], 6)
            tensors_layers, _ = parse_canonical_safetensors(out_layers.read_bytes())
            # Layer 2's last run is index 6; layer 3's last run is index 7.
            np.testing.assert_array_equal(tensors_layers["layers.2"], tensors_runs["runs.6"])
            np.testing.assert_array_equal(tensors_layers["layers.3"], tensors_runs["runs.7"])

    def test_merge_rejects_duplicate_prompt_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = {
                "schema": 1,
                "model_kind": "qwen38",
                "model_name": "qwen38",
                "weight_hash": "spot-sha256:" + "c" * 64,
                "corpus_sha256": "deadbeef",
                "max_seq_len": 128,
                "layer_ids": [0, 1],
                "prompt_indices": [0, 1],
                "n_prompts": 2,
                "sum_j": [torch.zeros(4, 16, dtype=torch.float64), torch.zeros(4, 16, dtype=torch.float64)],
            }
            p0 = tmp_path / "s0.pt"
            p1 = tmp_path / "s1.pt"
            torch.save(ckpt, p0)
            torch.save(ckpt, p1)
            with self.assertRaises(ValueError):
                merge_shards([p0, p1], tmp_path / "out.safetensors", view="layers")

            # Also verify spot-sha256: works on valid merge and --expect-prompts catches missing shards.
            out_ok = tmp_path / "qwen38.jlens.safetensors"
            merge_shards([p0], out_ok, view="layers", expect_prompts=2)
            with self.assertRaises(ValueError):
                merge_shards([p0], tmp_path / "incomplete.safetensors", view="layers", expect_prompts=4)


if __name__ == "__main__":
    unittest.main()
