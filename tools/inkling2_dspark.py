# SPDX-License-Identifier: MIT
"""CPU oracle of the Inkling 2 DSpark draft head (docs/gb10-research.md, D1 step 2).

The rules come from vLLM's `models/inkling2/nvidia/dspark.py` (Apache-2.0); this file is our own code. They are
written down in the lane notes (head-rules.md) and the docstrings below. The head reads the trunk's features of each
position (the inputs of layer visits 5, 23 and 41, and the final-norm output), mixes them, and drafts a block of 7 tokens
after an anchor token (the bonus token that the target has sampled and not yet run).

The pure parts (norms, mix, RoPE, window, Markov step, acceptance) take tensors and are tested in
test_inkling2_dspark.py with tiny shapes. `Head` loads the real weights.
"""
import math

import torch

HIDDEN = 2304
HEAD_DIM = 128
N_HEADS = 24
N_KV = 8
BLOCK = 8
WINDOW = 1024
EPS = 1e-6
THETA = 1_000_000.0
LOCAL = (True, False, True)
SOFTCAP = 30.0
VOCAB = 200058  # text_config.unpadded_vocab_size; the unembed has 201,024 rows


def rms(x, weight, eps=EPS):
    """RMSNorm with a plain weight, fp32 math, rounded to bf16 (flashinfer `rmsnorm` of the draft)."""
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(torch.bfloat16)


def mix(features, alpha, weight, eps=EPS):
    """RMSNorm of the alpha-weighted sum of the taps. features [T, taps, H] -> [T, H] bf16."""
    mixed = (features.float() * alpha.float()[None, :, None]).sum(1)
    return (mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(torch.bfloat16)


def rope(x, positions, theta=THETA):
    """Neox RoPE (halves rotate) on x [T, heads, D]: fp32 math, bf16 out."""
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    angle = positions.float()[:, None] * inv[None, :]
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]
    x1, x2 = x[..., : d // 2].float(), x[..., d // 2:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(torch.bfloat16)


def head_norm(x, weight, eps=EPS):
    """Per-head RMSNorm of x [T, heads, D] with a [D] weight, bf16 out."""
    return rms(x, weight, eps)


def key_range(first_block_pos, local, window=WINDOW):
    """The positions that a block starting at `first_block_pos` (= p + 1) reads, as (lo, hi) with hi exclusive.

    A local layer reads the `window` positions before the block and the block (`window + BLOCK` keys); a full layer
    reads everything from 0."""
    hi = first_block_pos + BLOCK
    return (max(0, hi - (window + BLOCK)) if local else 0), hi


def attention(q, k, v, scale=HEAD_DIM ** -0.5):
    """Bidirectional softmax attention, GQA. q [Tq, 24, D], k and v [Tk, 8, D] (bf16) -> [Tq, 24 * D] bf16 (fp32 math)."""
    group = q.shape[1] // k.shape[1]
    kk = k.float().repeat_interleave(group, dim=1)
    vv = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), kk) * scale
    out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vv)
    return out.reshape(q.shape[0], -1).to(torch.bfloat16)


def markov_gate(raw, state):
    """One recurrence of the Markov head. raw [3r] fp32 (gate, candidate, output), state [r] bf16.

    Every torch op of the reference rounds to bf16. Returns (new_state, latent), bf16."""
    r = state.shape[0]
    bf = lambda t: t.to(torch.bfloat16).float()
    gate, cand, out = bf(raw[:r]), bf(raw[r:2 * r]), bf(raw[2 * r:])
    gate = bf(torch.sigmoid(gate))
    cand = bf(torch.tanh(cand))
    keep = bf(gate * state.float())
    update = bf(bf(1.0 - gate) * cand)
    return (keep + update).to(torch.bfloat16), torch.tanh(out).to(torch.bfloat16)


def softcap(base, bias, cap=SOFTCAP):
    """`cap * tanh(bf16(base + bias) / cap)`: the sum rounds to bf16, the rest runs in fp32 and rounds once (vLLM
    `add_softcap`)."""
    x = (base.float() + bias.float()).to(torch.bfloat16).float()
    return (torch.tanh(x / cap) * cap).to(torch.bfloat16)


def greedy(logits, vocab=VOCAB):
    """The argmax over the first `vocab` rows (the rest of the padded vocab is padding), NaN as -inf."""
    return int(torch.nan_to_num(logits[:vocab].float(), nan=-math.inf).argmax())


def accepted(draft, truth):
    """The longest prefix of `draft` that equals `truth` (the true greedy tokens after the anchor)."""
    n = 0
    while n < min(len(draft), len(truth)) and draft[n] == truth[n]:
        n += 1
    return n


class Head:
    """The weights of the head, and the draft of one block."""

    def __init__(self, folder):
        from pathlib import Path
        from safetensors import safe_open

        folder = Path(folder)
        with safe_open(str(folder / "mtp.safetensors"), "pt") as f:
            self.w = {k: f.get_tensor(k) for k in f.keys()}
        import json
        index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(str(folder / index["model.llm.unembed.weight"]), "pt") as f:
            self.unembed = f.get_tensor("model.llm.unembed.weight").to(torch.bfloat16)
            self.embed = f.get_tensor("model.llm.embed.weight").to(torch.bfloat16)
            self.embed_norm = f.get_tensor("model.llm.embed_norm.weight")
        self.alpha = torch.softmax(self.w["fuse_weight"].float(), dim=-1)
        w = self.w
        self.layers = []
        for i in range(3):
            p = f"layers.{i}."
            g = lambda n: w[p + n]
            self.layers.append({
                "qkv": torch.cat([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")]),
                "kv_ctx": torch.cat([g("self_attn.k_proj_ctx.weight"), g("self_attn.v_proj_ctx.weight")]),
                "o": g("self_attn.o_proj.weight"),
                "gate_up": torch.cat([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")]),
                "down": g("mlp.down_proj.weight"),
                "in_ln": g("input_layernorm.weight"), "post_ln": g("post_attention_layernorm.weight"),
                "dflare": g("dflare_norm.weight"), "q_norm": g("self_attn.q_norm.weight"),
                "k_norm": g("self_attn.k_norm.weight"), "k_norm_ctx": g("self_attn.k_norm_ctx.weight"),
            })
        m = "markov_head."
        self.rnn_w, self.rnn_b = w[m + "rnn_proj.weight"], w[m + "rnn_proj.bias"].float()
        self.w1 = w[m + "markov_w1.weight"]
        self.m_dflare = w[m + "dflare_norm.weight"]
        # hidden_out folded into the head once, in fp64 (vLLM `fold`): [V, 256] and [V].
        proj, bias = w[m + "hidden_out.weight"].double(), w[m + "hidden_out.bias"].double()
        rows = self.unembed.shape[0]
        self.fold_w = torch.empty(rows, proj.shape[1], dtype=torch.bfloat16)
        self.fold_b = torch.empty(rows, dtype=torch.bfloat16)
        for s in range(0, rows, 4096):
            h = self.unembed[s:s + 4096].double()
            self.fold_w[s:s + 4096] = (h @ proj).to(torch.bfloat16)
            self.fold_b[s:s + 4096] = (h @ bias).to(torch.bfloat16)

    def embed_token(self, token):
        return rms(self.embed[token][None], self.embed_norm)[0]

    def context_kv(self, features, positions):
        """Per layer: the keys and values of the context positions. features [T, 4, H] bf16."""
        out = []
        for i, L in enumerate(self.layers):
            ctx = mix(features, self.alpha[i], L["dflare"])
            k, v = (ctx @ L["kv_ctx"].T).split(N_KV * HEAD_DIM, dim=-1)
            k = head_norm(k.reshape(-1, N_KV, HEAD_DIM), L["k_norm_ctx"])
            out.append((rope(k, positions), v.reshape(-1, N_KV, HEAD_DIM)))
        return out

    def block(self, anchor_token, first_pos, kv, f_last):
        """The draft of 7 tokens after `anchor_token` at position `first_pos` (= p + 1).

        kv: context_kv of positions 0..p. f_last: the mixed-features input of the Markov head's dflare, [4, H] at p."""
        mask = self.w["mask_embed"]
        x = torch.cat([self.embed_token(anchor_token)[None], mask[None].expand(BLOCK - 1, -1)])
        pos = torch.arange(first_pos, first_pos + BLOCK)
        residual, pending = None, None
        for i, L in enumerate(self.layers):
            if pending is None:
                residual = x
                h = rms(x, L["in_ln"])
            else:
                residual = (residual.float() + pending.float()).to(torch.bfloat16)
                h = rms(residual, L["in_ln"])
            qkv = h @ L["qkv"].T
            q, k, v = qkv.split([N_HEADS * HEAD_DIM, N_KV * HEAD_DIM, N_KV * HEAD_DIM], dim=-1)
            q = rope(head_norm(q.reshape(-1, N_HEADS, HEAD_DIM), L["q_norm"]), pos)
            k = rope(head_norm(k.reshape(-1, N_KV, HEAD_DIM), L["k_norm"]), pos)
            v = v.reshape(-1, N_KV, HEAD_DIM)
            lo, _ = key_range(first_pos, LOCAL[i])
            ck, cv = kv[i]
            kk = torch.cat([ck[lo:first_pos], k])
            vv = torch.cat([cv[lo:first_pos], v])
            a = attention(q, kk, vv) @ L["o"].T
            residual = (residual.float() + a.float()).to(torch.bfloat16)
            h = rms(residual, L["post_ln"])
            gu = (h @ L["gate_up"].T).float()
            g, u = gu.split(gu.shape[-1] // 2, dim=-1)
            pending = ((g / (1.0 + torch.exp(-g))) * u).to(torch.bfloat16) @ L["down"].T
        out = rms((residual.float() + pending.float()).to(torch.bfloat16), self.w["norm.weight"])
        return self.markov(anchor_token, out[1:], f_last)

    def markov(self, anchor_token, hidden, f_features):
        """Greedy chain over the 7 mask rows `hidden` [7, H] (bf16). f_features: [1, 4, H] features at p."""
        f = mix(f_features, self.alpha[3], self.m_dflare)[0]
        base = hidden @ self.unembed.T  # bf16 GEMM, [7, V]
        r = self.fold_w.shape[1]
        ctx = (torch.cat([hidden, f[None].expand_as(hidden)], dim=-1).float() @ self.rnn_w[:, 2 * r:].float().T) + self.rnn_b
        state = torch.zeros(r, dtype=torch.bfloat16)
        prev, drafts = anchor_token, []
        for i in range(hidden.shape[0]):
            raw = ctx[i] + torch.cat([state, self.w1[prev]]).float() @ self.rnn_w[:, :2 * r].float().T
            state, latent = markov_gate(raw, state)
            bias = (latent @ self.fold_w.T + self.fold_b).to(torch.bfloat16)
            logits = softcap(base[i], bias)
            prev = greedy(logits)
            drafts.append(prev)
        return drafts
