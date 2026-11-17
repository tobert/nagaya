# SPDX-License-Identifier: MIT
"""The reference: transformers' own `qwen4_exp` code (5.17.0, pinned) run on the CPU, layer-streamed.

`Qwen4ExpTextModel.forward` holds every layer at once (~62 GiB quantized, ~480 GiB in f32), so
`stream_forward` below repeats its body (modeling_qwen4_exp.py, `Qwen4ExpTextModel.forward`) with
one decoder layer materialized at a time: build the layer on the meta device, load its glued
weights, run every sequence through it, drop it. `test_reference.py` here holds this driver equal to
`Qwen4ExpTextModel.forward` on a small random model, so the loop can't drift from the model's
definition unnoticed.

The PLE n-gram table (16 heads x ~20M rows x 160) can't be an nn.Embedding; `GatherEmbedding`
dequantizes only the rows a batch asks for, from the GGUF mmap.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from . import glue

# The model's config.json, from the Hugging Face repository of Qwen3.8-Flash-Next (Apache-2.0), unchanged.
HF_CONFIG = Path(__file__).resolve().parent / "config.json"


def text_config(**overrides):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    tc = dict(json.loads(HF_CONFIG.read_text())["text_config"])
    for k in ("mtp", "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings", "mamba_ssm_dtype", "dtype",
              "model_type"):
        tc.pop(k, None)
    tc.update(overrides)
    cfg = Qwen4ExpTextConfig(**tc)
    cfg._attn_implementation = "eager"
    if hasattr(cfg, "_experts_implementation"):
        cfg._experts_implementation = "eager"
    return cfg


class GatherEmbedding(nn.Module):
    """Stands in for ple_embedding.ngram_embedding: rows dequantized on demand from a source."""

    def __init__(self, fetch_rows: Callable[[np.ndarray], torch.Tensor], num_embeddings: int, dim: int):
        super().__init__()
        self.fetch_rows = fetch_rows
        self.num_embeddings = num_embeddings
        self.embedding_dim = dim
        self.weight = torch.empty(0)  # the model reads .weight.device

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).cpu().numpy()
        if flat.min() < 0 or flat.max() >= self.num_embeddings:
            raise IndexError(f"PLE row out of range: [{flat.min()}, {flat.max()}] of {self.num_embeddings}")
        uniq, inv = np.unique(flat, return_inverse=True)
        rows = self.fetch_rows(uniq)  # [len(uniq), dim] f32
        return rows[torch.from_numpy(inv)].reshape(*ids.shape, self.embedding_dim)


def gguf_ple_rows(src: glue.GGUFSource):
    def fetch(rows: np.ndarray) -> torch.Tensor:
        return src.get("per_layer_token_embd.weight", rows)
    return fetch


def _materialize_buffers(layer: nn.Module, config) -> None:
    """Buffers built on the meta device: recompute them the way the modules' __init__ does."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M

    for mod in layer.modules():
        if isinstance(mod, M.Qwen4ExpTextNGramEmbedding):
            mod.layer_multipliers = M._build_layer_multipliers(mod.unigram_vocab_size, mod.ngram_size,
                                                               mod.ple_layer_index, mod.seed)
            mod.ngram_heads_vocab_sizes = torch.tensor(mod.head_vocab_sizes, dtype=torch.long)
            mod.ngram_heads_offsets = torch.tensor(mod.head_offsets, dtype=torch.long)


def build_layer(config, il: int, state: dict[str, torch.Tensor], ple_rows=None) -> nn.Module:
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M

    with torch.device("meta"):
        layer = M.Qwen4ExpTextDecoderLayer(config, il)
    lazy = "mlp.experts.gate_up_proj" not in state
    missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise KeyError(f"layer {il}: unexpected keys {unexpected}")
    # the table is gathered on demand; the hash constants are recomputed from the config by
    # _materialize_buffers (test_ple_constants_match_gguf holds them equal to the GGUF's)
    allowed_missing = {"ple.ple_embedding.ngram_embedding.weight", "ple.ple_embedding.layer_multipliers",
                       "ple.ple_embedding.ngram_heads_offsets", "ple.ple_embedding.ngram_heads_vocab_sizes"}
    if lazy:  # the caller attaches LazyExperts
        allowed_missing |= {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    if set(missing) - allowed_missing:
        raise KeyError(f"layer {il}: missing keys {sorted(set(missing) - allowed_missing)}")
    if layer.ple is not None:
        emb = layer.ple.ple_embedding
        if ple_rows is None:
            raise ValueError(f"layer {il} has PLE but no row source")
        if isinstance(ple_rows, nn.Module):
            emb.ngram_embedding = ple_rows
        else:
            emb.ngram_embedding = GatherEmbedding(ple_rows, emb.ngram_embedding.num_embeddings,
                                                  emb.ngram_embedding.embedding_dim)
    _materialize_buffers(layer, config)
    for n, p in list(layer.named_parameters()) + list(layer.named_buffers()):
        if p.device.type == "meta" and not (lazy and ".experts." in n):
            raise RuntimeError(f"layer {il}: {n} still on the meta device")
    return layer.eval()


class LazyExperts:
    """Stands in for experts.gate_up_proj / down_proj: `[e]` dequantizes expert e on demand, so
    a layer never holds 512 experts in f32 (~10 GiB, over the 8 GiB line in CLAUDE.md). Same
    values as glue.layer_state's. A bounded cache (the first `cap` experts asked for) saves
    re-dequantizing them for every sequence of a layer."""

    def __init__(self, get, cache: dict, cap: int):
        self.get, self.cache, self.cap = get, cache, cap

    def __getitem__(self, e):
        e = int(e)
        v = self.cache.get(e)
        if v is None:
            v = self.get(e)
            if len(self.cache) < self.cap:
                self.cache[e] = v
        return v


EXPERT_CACHE_CAP = 180  # x ~20 MiB (gate_up + down, f32) ~ 3.5 GiB per layer


def attach_lazy_experts(layer: nn.Module, src: glue.GGUFSource, il: int, cap: int = EXPERT_CACHE_CAP) -> None:
    from concurrent.futures import ThreadPoolExecutor

    ex = layer.mlp.experts
    b = f"blk.{il}."
    for name in ("gate_up_proj", "down_proj"):
        del ex._parameters[name]
    pool = ThreadPoolExecutor(3)  # the native dequant releases the GIL

    def both(e):
        fg = pool.submit(src.get, b + "ffn_gate_exps.weight", e)
        fu = pool.submit(src.get, b + "ffn_up_exps.weight", e)
        fd = pool.submit(src.get, b + "ffn_down_exps.weight", e)
        return torch.cat([fg.result(), fu.result()], 0), fd.result()

    cache: dict[int, tuple] = {}
    pair = LazyExperts(both, cache, cap)
    import weakref
    weakref.finalize(layer, pool.shutdown, wait=False)  # the pool goes with its layer (kaibo M1 review)
    ex.gate_up_proj = LazyExperts(lambda e: pair[e][0], {}, 0)
    ex.down_proj = LazyExperts(lambda e: pair[e][1], {}, 0)


class StreamState:
    """One sequence's activations between layers, plus what every layer needs."""

    def __init__(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor, config, rotary_emb):
        from transformers.masking_utils import create_causal_mask
        from transformers.models.qwen4_exp.modeling_qwen4_exp import create_recurrent_attention_mask

        # Qwen4ExpTextModel.forward, no cache, no padding:
        self.input_ids = input_ids
        position_ids = torch.arange(inputs_embeds.shape[1]).view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
        mask_kwargs = {"config": config, "inputs_embeds": inputs_embeds, "attention_mask": None,
                       "past_key_values": None, "position_ids": text_position_ids,
                       "allow_is_causal_skip": False}
        self.causal_mask_mapping = {
            "qwen_sparse_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }
        self.conv_mask = self.causal_mask_mapping.get("linear_attention")
        ple_input_ids = input_ids
        if config.ple_layer_ids and self.conv_mask is not None:
            eos = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
            ple_input_ids = torch.where(self.conv_mask.bool(), ple_input_ids, eos)
        self.ple_input_ids = ple_input_ids
        self.position_embeddings = rotary_emb(inputs_embeds, position_ids)
        self.hidden = inputs_embeds.repeat(1, 1, config.hc_count)

    def run_layer(self, layer: nn.Module) -> torch.Tensor:
        self.hidden = layer(self.hidden, position_embeddings=self.position_embeddings,
                            attention_mask=self.causal_mask_mapping["qwen_sparse_attention"],
                            conv_mask=self.conv_mask, past_key_values=None,
                            ple_input_ids=self.ple_input_ids)
        return self.hidden


@torch.no_grad()
def stream_forward(config, input_ids_list: list[torch.Tensor], embed: Callable[[torch.Tensor], torch.Tensor],
                   layer_source: Callable[[int], nn.Module], head: nn.Module,
                   on_layer: Callable[[int, nn.Module, list[StreamState]], None] | None = None,
                   n_layers: int | None = None) -> list[torch.Tensor]:
    """Runs every sequence ([1, T] ids) through the model one layer at a time. Returns each
    sequence's final hidden state after the head mixer ([1, T, hidden])."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding

    rotary = Qwen4ExpTextRotaryEmbedding(config=config)
    states = [StreamState(ids, embed(ids), config, rotary) for ids in input_ids_list]
    for il in range(config.num_hidden_layers if n_layers is None else n_layers):
        layer = layer_source(il)
        for s in states:
            s.run_layer(layer)
        if on_layer is not None:
            on_layer(il, layer, states)
        del layer
    return [head(s.hidden) for s in states]


# ---- the real model from the GGUF ----

class GGUFModel:
    """Layer source, embedding, head and lm_head rows for the shipped GGUF."""

    def __init__(self, src: glue.GGUFSource | None = None):
        self.src = src or glue.GGUFSource()
        self.config = text_config()

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).numpy()
        uniq, inv = np.unique(flat, return_inverse=True)
        rows = self.src.get("token_embd.weight", uniq)
        return rows[torch.from_numpy(inv)].reshape(*ids.shape, -1)

    def layer(self, il: int) -> nn.Module:
        ple = gguf_ple_rows(self.src) if il == glue.PLE_LAYER else None
        layer = build_layer(self.config, il, glue.layer_state(self.src, il, with_experts=False), ple)
        attach_lazy_experts(layer, self.src, il)
        return layer

    def head(self) -> nn.Module:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

        with torch.device("meta"):
            h = Qwen4ExpTextGatedResidual(self.config, use_combine=False)
        h.load_state_dict(glue.head_state(self.src), strict=True, assign=True)
        return h.eval()

    def lm_head(self) -> torch.Tensor:
        return self.src.get("output.weight")  # [vocab, hidden]


def letter_probs(logits: torch.Tensor, pieces: list[str], letters: list[str], n_probs: int = 64) -> dict[str, float]:
    """kaiseki's read (src/kaiseki/llama.py `read`): softmax over the vocab, the top n_probs
    candidates, summing those whose piece, stripped, is a letter. Unnormalized, like the record."""
    p = torch.softmax(logits.double(), -1)
    top = torch.topk(p, n_probs)
    out = {k: 0.0 for k in letters}
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        t = pieces[i].strip()
        if t in out:
            out[t] += v
    return out


def log_ratios(probs: dict[str, float], letters: list[str]) -> dict[tuple[str, str], float]:
    floor = 1e-12
    return {(a, b): math.log(max(probs[a], floor)) - math.log(max(probs[b], floor))
            for i, a in enumerate(letters) for b in letters[i + 1:]}
