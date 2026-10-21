#!/usr/bin/env python3
"""f32 CPU reference for the Gemma 4 26B-A4B text model, built from HF transformers' own modules.

Runs `Gemma4TextDecoderLayer` from `transformers.models.gemma4` one layer at a time (each layer's
BF16 weights are upcast to f32, run over every token, then freed) and writes a safetensors file
of f32 tensors that the GPU golden tests compare against:

  input_ids                 [S] (stored as F32)
  embed                     [S, 2816]   scaled token embeddings (residual entering layer 0)
  L{l}.out                  [S, 2816]   residual leaving layer l (every layer)
  final_norm                [S, 2816]   after model.language_model.norm
  logits_last               [262144]    softcapped logits at the last position
  argmax_all, top2_gap_all  [S]         with --all-positions: each position's greedy next token and
                                        the gap between its top two softcapped logits
  L{l}.<op>                 per-op intermediates for each layer in --dump-layers

Setup: uv venv --python 3.13 VENV && uv pip install --python VENV/bin/python torch \
         --index-url https://download.pytorch.org/whl/cpu && uv pip install --python VENV/bin/python transformers safetensors

Checked against HF's full Gemma4ForConditionalGeneration in BF16 on a chat-turn prompt: same top-3
next tokens, top logit 25.07 (this f32 driver) vs 25.25 (HF BF16).

Usage (needs torch, transformers >= 5.19, safetensors, tokenizers):
  python tools/gemma4_reference.py --model "$GEMMA4_MODEL_DIR" \
      --text "The capital of France is" --dump-layers 0,5 --out target/gemma4-golden/ref.safetensors
"""

import argparse
import json
import math
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import transformers.models.gemma4.modeling_gemma4 as m4
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

PREFIX = "model.language_model."


class Shards:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.handles = {
            name: safe_open(os.path.join(model_dir, name), framework="pt")
            for name in sorted(set(self.weight_map.values()))
        }

    def get(self, name):
        return self.handles[self.weight_map[name]].get_tensor(name)

    def layer_state(self, layer):
        p = f"{PREFIX}layers.{layer}."
        return {k[len(p):]: self.get(k).float() for k in self.weight_map if k.startswith(p)}


def masks(seq_len, sliding_window):
    q = torch.arange(seq_len)[:, None]
    k = torch.arange(seq_len)[None, :]
    causal = k <= q
    sliding = causal & ((q - k) < sliding_window)
    to_add = lambda allowed: torch.where(allowed, 0.0, float("-inf"))[None, None, :, :]
    return {"full_attention": to_add(causal), "sliding_attention": to_add(sliding)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="encoded with <bos> and no <eos>")
    g.add_argument("--ids", help="comma-separated token ids")
    ap.add_argument("--tokenizer", help="tokenizer.json (default: <model>/tokenizer.json)")
    ap.add_argument("--dump-layers", default="", help="comma-separated layer indices")
    ap.add_argument("--layers", type=int, default=None, help="run only the first N layers")
    ap.add_argument("--dump-routes", action="store_true", help="every layer's top-8 indices and 9 largest router probabilities")
    ap.add_argument(
        "--expert-sweep",
        default="",
        help="comma-separated layers: also run that layer's Gemma4TextExperts with every one of the 128 "
        "experts picked once (16 tokens x 8 picks, token t pick s -> expert 8t+s, weight 1/8 + s/100) "
        "on the layer's real pre_feedforward_layernorm_2 output; needs >= 16 tokens",
    )
    ap.add_argument("--all-positions", action="store_true", help="also every position's argmax and top-2 logit gap")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    with open(os.path.join(args.model, "config.json")) as f:
        config = Gemma4TextConfig(**json.load(f)["text_config"])
    config._attn_implementation = "eager"

    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    else:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(args.tokenizer or os.path.join(args.model, "tokenizer.json"))
        ids = [2] + tok.encode(args.text, add_special_tokens=False).ids
    seq_len = len(ids)
    dump_layers = {int(x) for x in args.dump_layers.split(",") if x}
    sweep_layers = {int(x) for x in args.expert_sweep.split(",") if x}
    dump_layers |= sweep_layers
    n_layers = args.layers if args.layers is not None else config.num_hidden_layers

    shards = Shards(args.model)
    out = {"input_ids": torch.tensor(ids, dtype=torch.float32)}

    embed_w = shards.get(PREFIX + "embed_tokens.weight")
    h = embed_w[torch.tensor(ids)].float() * math.sqrt(config.hidden_size)
    h = h[None, :, :]
    out["embed"] = h[0].clone()

    rotary = m4.Gemma4TextRotaryEmbedding(config)
    position_ids = torch.arange(seq_len)[None, :]
    pos_emb = {t: rotary(h, position_ids, t) for t in set(config.layer_types)}
    mask = masks(seq_len, config.sliding_window)

    captured = {}
    real_attention = m4.eager_attention_forward

    def capturing_attention(module, query, key, value, attention_mask, **kw):
        attn_out, weights = real_attention(module, query, key, value, attention_mask, **kw)
        if module.layer_idx in dump_layers:
            l = module.layer_idx
            captured[f"L{l}.q_rope"] = query[0].transpose(0, 1).contiguous()  # [S, H, D]
            captured[f"L{l}.k_rope"] = key[0].transpose(0, 1).contiguous()  # [S, KV, D]
            captured[f"L{l}.v_norm"] = value[0].transpose(0, 1).contiguous()  # [S, KV, D]
            captured[f"L{l}.attn"] = attn_out[0].contiguous()  # [S, H, D]
        return attn_out, weights

    m4.eager_attention_forward = capturing_attention

    for l in range(n_layers):
        layer = m4.Gemma4TextDecoderLayer(config, l).float()
        missing, unexpected = layer.load_state_dict(shards.layer_state(l), strict=False)
        missing = [k for k in missing if k != "layer_scalar"]
        if missing or unexpected:
            sys.exit(f"layer {l}: missing {missing}, unexpected {unexpected}")
        layer.layer_scalar.copy_(shards.get(f"{PREFIX}layers.{l}.layer_scalar").float())

        hooks = []
        if l in dump_layers:
            out[f"L{l}.in"] = h[0].clone()

            def save(name, which="out"):
                def hook(mod, inputs, output):
                    t = inputs[0] if which == "in" else output
                    if isinstance(t, tuple):
                        t = t[0]
                    captured[f"L{l}.{name}"] = t.reshape(seq_len, -1).clone() if t.dim() > 1 else t.clone()
                return hook

            named = {
                "input_layernorm": "attn_norm",
                "self_attn.q_proj": "q_proj",
                "self_attn.k_proj": "k_proj",
                "self_attn.o_proj": "o_proj",
                "post_attention_layernorm": "post_attn_norm",
                "pre_feedforward_layernorm": "pre_ffn_norm",
                "mlp": "mlp",
                "post_feedforward_layernorm_1": "post_ffn_norm_1",
                "pre_feedforward_layernorm_2": "pre_ffn_norm_2",
                "experts": "experts",
                "post_feedforward_layernorm_2": "post_ffn_norm_2",
                "post_feedforward_layernorm": "post_ffn_norm",
            }
            if layer.self_attn.v_proj is not None:
                named["self_attn.v_proj"] = "v_proj"
            for path, name in named.items():
                hooks.append(layer.get_submodule(path).register_forward_hook(save(name)))

            def router_hook(mod, inputs, output):
                _, w, idx = output
                captured[f"L{l}.router_weights"] = w.clone()
                captured[f"L{l}.router_index"] = idx.float().clone()
            hooks.append(layer.router.register_forward_hook(router_hook))

        if l in sweep_layers:
            def sweep_hook(mod, inputs, output, l=l):
                x = output.reshape(seq_len, -1)[:16]
                idx = torch.arange(128).reshape(16, 8)
                w = (1.0 / 8 + torch.arange(8, dtype=torch.float32) / 100).repeat(16, 1)
                captured[f"L{l}.sweep_in"] = x.clone()
                captured[f"L{l}.sweep_index"] = idx.float()
                captured[f"L{l}.sweep_weights"] = w.clone()
                captured[f"L{l}.sweep_out"] = layer.experts(x, idx, w).clone()
            hooks.append(layer.pre_feedforward_layernorm_2.register_forward_hook(sweep_hook))

        if args.dump_routes:
            def route_hook(mod, inputs, output, l=l):
                probs, _, idx = output
                captured[f"L{l}.route_index"] = idx.float().clone()
                captured[f"L{l}.route_p9"] = torch.topk(probs, 9, dim=-1).values.clone()
            hooks.append(layer.router.register_forward_hook(route_hook))

        h = layer(
            h,
            None,
            shared_kv_states={},
            position_embeddings=pos_emb[config.layer_types[l]],
            attention_mask=mask[config.layer_types[l]],
            position_ids=position_ids,
        )
        for hk in hooks:
            hk.remove()
        out[f"L{l}.out"] = h[0].clone()
        out.update(captured)
        captured.clear()
        del layer
        print(f"layer {l} done", file=sys.stderr)

    if n_layers == config.num_hidden_layers:
        norm = m4.Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight.copy_(shards.get(PREFIX + "norm.weight").float())
        hn = norm(h)
        out["final_norm"] = hn[0].clone()
        logits = hn[0, -1] @ embed_w.float().T
        cap = config.final_logit_softcapping
        logits = torch.tanh(logits / cap) * cap
        out["logits_last"] = logits
        if args.all_positions:
            embed_t = embed_w.float().T
            arg, gap = [], []
            for i in range(0, seq_len, 64):
                lg = torch.tanh((hn[0, i:i + 64] @ embed_t) / cap) * cap
                t2 = torch.topk(lg, 2, dim=-1)
                arg.append(t2.indices[:, 0].float())
                gap.append(t2.values[:, 0] - t2.values[:, 1])
            out["argmax_all"] = torch.cat(arg)
            out["top2_gap_all"] = torch.cat(gap)
        top = torch.topk(logits, 10)
        print("top-10 next tokens:", list(zip(top.indices.tolist(), [round(v, 4) for v in top.values.tolist()])))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file({k: v.float().contiguous() for k, v in out.items()}, args.out)
    print(f"wrote {len(out)} tensors to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
