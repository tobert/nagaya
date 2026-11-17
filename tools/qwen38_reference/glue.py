# SPDX-License-Identifier: MIT
"""GGUF -> transformers parameter glue for Qwen3.8-Flash-Next (qwen4exp, UD-IQ4_XS).

The GGUF carries the converter's transforms (llama.cpp 5cf3a3528 `conversion/qwen4exp.py`,
`conversion/qwen.py` `_LinearAttentionVReorderBase` and `Qwen3NextModel.modify_tensors`), which
this module undoes so the vendored transformers `qwen4_exp` code runs on the shipped weights:

1. GDN V heads: HF stores the 48 value heads grouped by key head ([k0: v0 v1 v2, k1: ...]); the
   converter tiles them ([v0 of k0..k15, v1 of k0..k15, v2 ...]) in in_proj_qkv's V rows,
   in_proj_z rows, in_proj_a/b rows, A_log, dt_bias, conv1d's V channels and out_proj's columns.
2. A_log is stored as ssm_a = -exp(A_log).
3. Zero-centred RMSNorm gammas get +1: every `*norm.weight` except `linear_attn.norm.weight`
   (hc norms, q/k norms), plus the PLE norms and the indexer q/k layernorms.
4. The indexer's index_qk_proj is split into q_proj and k_proj.
5. conv1d weights are squeezed ([C, 1, K] -> [C, K]); experts' gate_up_proj is split in two.

Values are the shipped ones: dequantized, never requantized. Everything returned is float32.
"""
from __future__ import annotations

import math
from functools import cached_property

import numpy as np
import torch

from . import shards_from_env


N_K_HEADS, N_V_HEADS, HEAD_K, HEAD_V = 16, 48, 128, 128
V_PER_K = N_V_HEADS // N_K_HEADS
N_LAYER = 48
PLE_LAYER = 1  # 0-based (HF ple_layer_ids = [2], 1-based)


def is_attn_layer(il: int) -> bool:
    return (il + 1) % 4 == 0


# ---- the converter's transforms and their inverses ----

def tile_v(t: torch.Tensor, dim: int, head_dim: int) -> torch.Tensor:
    """The converter's grouped -> tiled V-head reorder (`_reorder_v_heads`), for tests."""
    shape = list(t.shape)
    new = shape[:dim] + [N_K_HEADS, V_PER_K, head_dim] + shape[dim + 1:]
    return t.reshape(new).transpose(dim, dim + 1).contiguous().reshape(shape)


def untile_v(t: torch.Tensor, dim: int, head_dim: int) -> torch.Tensor:
    """Inverse of tile_v: tiled (GGUF) -> grouped (HF)."""
    shape = list(t.shape)
    if shape[dim] != N_V_HEADS * head_dim:
        raise ValueError(f"untile_v: dim {dim} of {shape} is not {N_V_HEADS}x{head_dim}")
    new = shape[:dim] + [V_PER_K, N_K_HEADS, head_dim] + shape[dim + 1:]
    return t.reshape(new).transpose(dim, dim + 1).contiguous().reshape(shape)


def unshift_norm(t: torch.Tensor) -> torch.Tensor:
    """GGUF stores 1 + w for zero-centred gammas; HF's Qwen4ExpTextRMSNorm computes (1 + w)."""
    return t - 1.0


# ---- weight source: the GGUF, dequantized ----

class GGUFSource:
    """Dequantized tensors of the 3 shards, in numpy order (reversed ggml ne: last dim = ne0).

    gguf-py reads and dequantizes the GGUF. It is llama.cpp's own Python reader, and it is independent of the
    engine's dequant, which an oracle needs. The tensors stay memory-mapped: a call dequantizes only its rows."""

    _TYPES = {"F32", "BF16", "Q8_0", "Q6_K", "IQ3_S", "IQ4_NL", "IQ4_XS"}

    def __init__(self, shards=None):
        self.shards = list(shards) if shards is not None else shards_from_env()

    @cached_property
    def _tensors(self):
        import gguf

        out = {}
        for p in self.shards:
            for t in gguf.GGUFReader(p).tensors:
                if t.name in out:
                    raise ValueError(f"{p}: the tensor {t.name} is in two shards")
                out[t.name] = t
        return out

    def _info(self, name: str):
        t = self._tensors[name]
        if t.tensor_type.name not in self._TYPES:
            raise ValueError(f"{name}: the type {t.tensor_type.name} is not a type of this model")
        return t

    def ne(self, name: str) -> list[int]:
        return [int(n) for n in self._info(name).shape]

    def qtype(self, name: str) -> str:
        return self._info(name).tensor_type.name

    def dequant_rows(self, name: str, start: int, stop: int) -> np.ndarray:
        """Rows [start, stop) of the tensor as float32, with the rows of all outer dims in one run."""
        import gguf

        t = self._info(name)
        ne = self.ne(name)
        n_rows = math.prod(ne[1:])
        if not 0 <= start <= stop <= n_rows:
            raise IndexError(f"{name}: rows [{start}, {stop}) are not in [0, {n_rows})")
        raw = np.asarray(t.data).reshape(n_rows, -1)[start:stop]
        rows = gguf.quants.dequantize(raw, t.tensor_type).reshape(stop - start, ne[0])
        if rows.dtype != np.float32:
            raise TypeError(f"{name}: gguf-py gave {rows.dtype}, not float32")
        return rows

    def get(self, name: str, index=None) -> torch.Tensor:
        """Whole tensor, or `data[index]` along the outermost numpy dim (rows / experts): an
        int, a slice, or an array of row ids (2-D tensors)."""
        ne = self.ne(name)
        n_rows = math.prod(ne[1:])
        if index is None:
            arr = self.dequant_rows(name, 0, n_rows).reshape(list(reversed(ne)))
        elif len(ne) == 3:
            if not isinstance(index, (int, np.integer)):
                raise TypeError(f"{name}: 3-D tensors take one expert index, got {type(index)}")
            e = int(index)
            arr = self.dequant_rows(name, e * ne[1], (e + 1) * ne[1])
        elif isinstance(index, slice):
            start, stop, step = index.indices(n_rows)
            if step != 1:
                raise ValueError("strided slices are not supported")
            arr = self.dequant_rows(name, start, stop)
        elif isinstance(index, (int, np.integer)):
            arr = self.dequant_rows(name, int(index), int(index) + 1)[0]
        else:
            rows = np.asarray(index).reshape(-1)
            arr = np.stack([self.dequant_rows(name, int(r), int(r) + 1)[0] for r in rows]) if rows.size else \
                np.empty((0, ne[0]), np.float32)
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


# ---- per-layer HF state dicts ----

def layer_state(src: GGUFSource, il: int, with_experts: bool = True) -> dict[str, torch.Tensor]:
    """Qwen4ExpTextDecoderLayer state-dict entries for layer `il` (module-relative names).
    The PLE n-gram table is not included (it's gathered on demand; see reference.py)."""
    b = f"blk.{il}."
    g = lambda n, index=None: src.get(b + n, index)
    sd: dict[str, torch.Tensor] = {}
    for hc, pre in (("attn", "attn_hyper_connection"), ("ffn", "mlp_hyper_connection")):
        sd[f"{pre}.hc_norm.weight"] = unshift_norm(g(f"hc_{hc}_norm.weight"))
        sd[f"{pre}.input_mix_weight_down.weight"] = g(f"hc_{hc}_down.weight")
        sd[f"{pre}.input_mix_weight_up.weight"] = g(f"hc_{hc}_up.weight")
        sd[f"{pre}.block_inject_weight.weight"] = g(f"hc_{hc}_inject.weight")

    if is_attn_layer(il):
        a = "self_attn."
        sd[a + "q_proj.weight"] = g("attn_q.weight")
        sd[a + "k_proj.weight"] = g("attn_k.weight")
        sd[a + "v_proj.weight"] = g("attn_v.weight")
        sd[a + "o_proj.weight"] = g("attn_output.weight")
        sd[a + "q_norm.weight"] = unshift_norm(g("attn_q_norm.weight"))
        sd[a + "k_norm.weight"] = unshift_norm(g("attn_k_norm.weight"))
        sd[a + "indexer.index_qk_proj.weight"] = torch.cat([g("indexer.q_proj.weight"), g("indexer.k_proj.weight")], 0)
        sd[a + "indexer.q_layernorm.weight"] = unshift_norm(g("indexer.q_norm.weight"))
        sd[a + "indexer.k_layernorm.weight"] = unshift_norm(g("indexer.k_norm.weight"))
    else:
        l = "linear_attn."
        qkv = g("attn_qkv.weight")
        kd = N_K_HEADS * HEAD_K
        sd[l + "in_proj_qkv.weight"] = torch.cat([qkv[: 2 * kd], untile_v(qkv[2 * kd:], 0, HEAD_V)], 0)
        sd[l + "in_proj_z.weight"] = untile_v(g("attn_gate.weight"), 0, HEAD_V)
        sd[l + "in_proj_a.weight"] = untile_v(g("ssm_alpha.weight"), 0, 1)
        sd[l + "in_proj_b.weight"] = untile_v(g("ssm_beta.weight"), 0, 1)
        ssm_a = g("ssm_a").double()
        if not bool((ssm_a < 0).all()):
            raise ValueError(f"layer {il}: ssm_a has non-negative entries; can't invert -exp(A_log)")
        sd[l + "A_log"] = untile_v(torch.log(-ssm_a).float().unsqueeze(-1), 0, 1).squeeze(-1)
        sd[l + "dt_bias"] = untile_v(g("ssm_dt.bias").unsqueeze(-1), 0, 1).squeeze(-1)
        conv = g("ssm_conv1d.weight")  # [C, K]
        sd[l + "conv1d.weight"] = torch.cat([conv[: 2 * kd], untile_v(conv[2 * kd:], 0, HEAD_V)], 0).unsqueeze(1)
        sd[l + "norm.weight"] = g("ssm_norm.weight")
        sd[l + "out_proj.weight"] = untile_v(g("ssm_out.weight"), 1, HEAD_V)

    m = "mlp."
    sd[m + "gate.weight"] = g("ffn_gate_inp.weight")
    sd[m + "shared_expert_gate.weight"] = g("ffn_gate_inp_shexp.weight").unsqueeze(0)
    sd[m + "shared_expert.gate_proj.weight"] = g("ffn_gate_shexp.weight")
    sd[m + "shared_expert.up_proj.weight"] = g("ffn_up_shexp.weight")
    sd[m + "shared_expert.down_proj.weight"] = g("ffn_down_shexp.weight")
    if with_experts:
        sd[m + "experts.gate_up_proj"] = torch.cat([g("ffn_gate_exps.weight"), g("ffn_up_exps.weight")], 1)
        sd[m + "experts.down_proj"] = g("ffn_down_exps.weight")

    if il == PLE_LAYER:
        p = "ple."
        sd[p + "key_proj.weight"] = g("ple_key.weight")
        sd[p + "value_proj.weight"] = g("ple_value.weight")
        sd[p + "norm_key.weight"] = unshift_norm(g("ple_norm_key.weight"))
        sd[p + "norm_query.weight"] = unshift_norm(g("ple_norm_query.weight"))
        sd[p + "norm_conv.weight"] = unshift_norm(g("ple_norm_conv.weight"))
        sd[p + "conv1d.weight"] = g("ple_conv1d.weight").unsqueeze(1)
    return sd


def head_state(src: GGUFSource) -> dict[str, torch.Tensor]:
    """The final hyper-connection mixer (use_combine=False)."""
    return {
        "hc_norm.weight": unshift_norm(src.get("output_hc_norm.weight")),
        "input_mix_weight_down.weight": src.get("output_hc_down.weight"),
        "input_mix_weight_up.weight": src.get("output_hc_up.weight"),
    }


# GGUF name for each HF (model.language_model.)layers.{il}.<key>, for the glue-vs-original test.
def hf_key(il: int | None, key: str) -> str:
    if il is None:
        return f"model.language_model.{key}"
    return f"model.language_model.layers.{il}.{key}"
