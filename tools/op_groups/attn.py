"""The group `attn`: the expected values of the HF attention module for the op `qwen38_attn`, on generator inputs.

The reference's `test_attn.py` and `test_attn_long.py` run the HF module `Qwen4ExpTextAttention` with the real weights of layer 3 on
real embeddings. Here the module has stand-ins for its projection layers: they return the raw projections of the
generator, so no weight is needed. The norms take the generator's gammas. The rest of the module is HF's own code,
and the runs follow the reference's `run_reference` and `reference_tail`.

A sequence of seed S has, for token i, the raw projections that the generator makes at index i * width of the streams
S * 10 + 1 (q, width 12288), + 2 (k, 512), + 3 (v, 512), + 4 (index q, 512) and + 5 (index k, 128). The gammas are stored
in the fixture. A sequence is the same for any prefix, and the attention is causal, so one run of 24 tokens serves the
dense cases of 1, 9 and 24 tokens and the continuation.

The outputs hold the heads 0, 11, 12 and 23 only (one on each side of the border of the two key-value heads), to keep the
fixture small.
"""
import math
import sys

import numpy as np

SOURCE_FILES = ["tests/ops/test_attn.py", "tests/ops/test_attn_long.py"]

H, KVH, HD, IH, ID = 24, 2, 256, 4, 128
HEADS = [0, 11, 12, 23]
WIDTH = {"q": 12288, "k": 512, "v": 512, "iq": 512, "ik": 128}
STREAM = {"q": 1, "k": 2, "v": 3, "iq": 4, "ik": 5}
DENSE_SEED, SPARSE_SEED, LONG_SEED = 71, 72, 73
DENSE_T, SPARSE_T, SPARSE_BUDGET = 24, 28, 16
NK = 544  # blocks kept for each token in the long cases
LONG_CASES = [(5000, 5), (8190, 3), (12000, 4), (32768 - 64, 4), (32768 - 1, 1)]
NCTX = 32768
SEL_WIDTH = {"dense": DENSE_T, "sparse": SPARSE_BUDGET + 3}


def make(g, lib):
    sys.path.insert(0, str(lib.REFERENCE))
    import torch
    from torch import nn
    from harness import reference

    torch.set_num_threads(8)

    def normals_at(seed, start, n):
        """`n` values of the generator's normal stream from index `start`."""
        with np.errstate(over="ignore"):
            key = lib._mix(np.uint64(seed) * lib.GOLDEN ^ np.uint64(0x6F70735F74657374))
            index = np.arange(start, start + n, dtype=np.uint64)
            u = lib._mix(key + (index + np.uint64(1)) * lib.GOLDEN)
        s = ((u & np.uint64(0xFFFF)) + ((u >> np.uint64(16)) & np.uint64(0xFFFF)) + ((u >> np.uint64(32)) & np.uint64(0xFFFF))
             + ((u >> np.uint64(48)) & np.uint64(0xFFFF))).astype(np.float64)
        return ((s - 4.0 * 32767.5) / lib._SD).astype(np.float32)

    def raw(seed, kind, first, count):
        w = WIDTH[kind]
        return normals_at(seed * 10 + STREAM[kind], first * w, count * w).reshape(count, w)

    def gammas(seed):
        f = lambda s, n: (np.float32(1) + np.float32(0.1) * normals_at(seed * 10 + s, 0, n)).astype(np.float32)
        return {"q": f(6, HD), "k": f(7, HD), "iq": f(8, ID), "ik": f(9, ID)}

    class Fixed(nn.Module):
        """A projection layer that returns a given tensor [1, n, width], whatever its input."""

        def __init__(self, t):
            super().__init__()
            self.t = t

        def forward(self, x):
            assert x.shape[1] == self.t.shape[1], (x.shape, self.t.shape)
            return self.t

    from transformers.models.qwen4_exp import modeling_qwen4_exp as M

    def tt(a):
        return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0)

    def build_attn(cfg, gm):
        attn = M.Qwen4ExpTextAttention(cfg, layer_idx=3)
        # Gamma of the op is the whole multiplier: the module computes x * (1 + weight). gamma - 1 is exact in f32.
        for mod, name in ((attn.q_norm, "q"), (attn.k_norm, "k"), (attn.indexer.q_layernorm, "iq"),
                          (attn.indexer.k_layernorm, "ik")):
            mod.weight.data = torch.from_numpy(gm[name] - np.float32(1))
        return attn.eval()

    def set_projections(attn, seed, first, count):
        attn.q_proj = Fixed(tt(raw(seed, "q", first, count)))
        attn.k_proj = Fixed(tt(raw(seed, "k", first, count)))
        attn.v_proj = Fixed(tt(raw(seed, "v", first, count)))
        attn.indexer.index_qk_proj = Fixed(tt(np.concatenate([raw(seed, "iq", first, count), raw(seed, "ik", first, count)], 1)))

    def pos_mask(cfg, t):
        from transformers.masking_utils import create_causal_mask
        h = torch.zeros(1, t, cfg.hidden_size)
        position_ids = torch.arange(t).view(1, 1, -1).expand(4, 1, -1)
        mask_kwargs = {"config": cfg, "inputs_embeds": h, "attention_mask": None, "past_key_values": None,
                       "position_ids": position_ids[0], "allow_is_causal_skip": False}
        causal = create_causal_mask(**mask_kwargs)
        pos_emb = M.Qwen4ExpTextRotaryEmbedding(config=cfg)(h, position_ids[1:])
        return h, pos_emb, causal

    def run_reference(attn, h, pos_emb, causal):
        with torch.no_grad():
            sel_mask = attn.indexer(h, pos_emb, causal, None)[0, 0]
            captured = {}
            handle = attn.o_proj.register_forward_hook(lambda _m, inp, _out: captured.setdefault("pre_o", inp[0]))
            attn(h, position_embeddings=pos_emb, attention_mask=causal, past_key_values=None)
            handle.remove()
        return captured["pre_o"][0].numpy(), (sel_mask > -1e30).numpy()

    def padded(visible, width):
        out = np.full((visible.shape[0], width), -1, np.int32)
        for i, row in enumerate(visible):
            cells = np.nonzero(row)[0]
            assert len(cells) <= width, (len(cells), width)
            out[i, :len(cells)] = cells
        return out

    cfg = reference.text_config()
    inv_freq = M.Qwen4ExpTextRotaryEmbedding.compute_default_rope_parameters(cfg)[0].numpy().astype(np.float32)
    assert inv_freq.shape == (32,)
    g.add("inv_freq", inv_freq)

    for name, seed, t, budget in (("dense", DENSE_SEED, DENSE_T, 2048), ("sparse", SPARSE_SEED, SPARSE_T, SPARSE_BUDGET)):
        c = reference.text_config(indexer_budget=budget)
        gm = gammas(seed)
        for k, v in gm.items():
            g.add(f"{name}_gamma_{k}", v)
        attn = build_attn(c, gm)
        set_projections(attn, seed, 0, t)
        h, pos_emb, causal = pos_mask(c, t)
        out, visible = run_reference(attn, h, pos_emb, causal)
        g.add(f"{name}_out", out.reshape(t, H, HD)[:, HEADS])
        g.add(f"{name}_sel", padded(visible, SEL_WIDTH[name]))
        print(name, "reference done", flush=True)

    # ---- the long cases (test_attn_long.py) ----
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    seed = LONG_SEED
    gm = gammas(seed)
    for k, v in gm.items():
        g.add(f"long_gamma_{k}", v)
    attn = build_attn(cfg, gm)
    h_all = torch.zeros(1, NCTX, cfg.hidden_size)
    position_ids = torch.arange(NCTX).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = M.Qwen4ExpTextRotaryEmbedding(config=cfg)(h_all, position_ids)
    del h_all
    k_raw = torch.from_numpy(raw(seed, "k", 0, NCTX)).unsqueeze(0)
    v_raw = torch.from_numpy(raw(seed, "v", 0, NCTX)).unsqueeze(0)
    ik_raw = torch.from_numpy(raw(seed, "ik", 0, NCTX)).unsqueeze(0)
    with torch.no_grad():
        k_all = attn.k_norm(k_raw.view(1, NCTX, 2, 256)).transpose(1, 2)
        k_all = apply_rotary_pos_emb(k_all, cos=cos, sin=sin)
        v_all = v_raw.view(1, NCTX, 2, 256).transpose(1, 2)

    class TailCache:
        def __init__(self, k, v, raw_keys):
            self.k, self.v, self.raw = k, v, raw_keys

        def update(self, key_states, value_states, layer_idx):
            return torch.cat([self.k, key_states], dim=-2), torch.cat([self.v, value_states], dim=-2)

        def update_indexer(self, raw_keys, layer_idx):
            return torch.cat([self.raw, raw_keys], dim=1)

    def reference_tail(p0, t):
        ttot = p0 + t
        set_projections(attn, seed, p0, t)
        hh = torch.zeros(1, t, cfg.hidden_size)
        cache = TailCache(k_all[:, :, :p0], v_all[:, :, :p0], ik_raw[:, :p0])
        mask = torch.full((1, 1, t, ttot), torch.finfo(torch.float32).min)
        for i in range(t):
            mask[0, 0, i, :p0 + i + 1] = 0.0
        pos_emb = (cos[:, :ttot], sin[:, :ttot])
        got, scores = {}, []
        orig_topk = torch.Tensor.topk

        def topk(self, *a, **kw):
            scores.append(self.detach().clone())
            return orig_topk(self, *a, **kw)

        h1 = attn.o_proj.register_forward_hook(lambda _m, _i, inp_out: None)
        h1.remove()
        h1 = attn.o_proj.register_forward_hook(lambda _m, inp, _o: got.setdefault("pre_o", inp[0]))
        torch.Tensor.topk = topk
        try:
            with torch.no_grad():
                attn(hh, position_embeddings=pos_emb, attention_mask=mask, past_key_values=cache)
        finally:
            torch.Tensor.topk = orig_topk
            h1.remove()
        assert len(scores) == t
        return got["pre_o"][0].numpy(), [s.numpy() for s in scores]

    def score_bounds(p0, t):
        ix = attn.indexer
        ttot = p0 + t
        iq = torch.from_numpy(raw(seed, "iq", p0, t)).reshape(1, t, IH, ID)
        with torch.no_grad():
            q = apply_rotary_pos_emb(ix.q_layernorm(iq), cos=cos[:, p0:ttot], sin=sin[:, p0:ttot], unsqueeze_dim=2)[0]
            nb = ttot // 4
            pooled = ix.k_layernorm(ik_raw[0, :nb * 4].view(nb, 4, ID).float().mean(dim=1))
            starts = torch.arange(nb) * 4
            k = apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos[0].index_select(0, starts),
                                     sin=sin[0].index_select(0, starts)).squeeze(1)
            s_abs = torch.einsum("thj,bj->tb", q.abs(), k.abs()) / math.sqrt(ID)
        return (8.0 * ID * 2.0 ** -24 * s_abs).numpy()

    g.meta["long_cases"] = [{"p0": p0, "t": t} for p0, t in LONG_CASES]
    g.meta["long_nk"] = NK
    for n, (p0, t) in enumerate(LONG_CASES):
        ttot = p0 + t
        out, scores = reference_tail(p0, t)
        bounds = score_bounds(p0, t)
        ids = np.full((t, NK), -1, np.int32)
        sc = np.zeros((t, NK), np.float32)
        bd = np.zeros((t, NK), np.float32)
        for i in range(t):
            s = torch.from_numpy(scores[i])
            top = torch.topk(s, min(NK, s.shape[0])).indices.numpy()
            ids[i, :len(top)] = top
            sc[i, :len(top)] = scores[i][top]
            bd[i, :len(top)] = bounds[i, top]
        g.add(f"long{n}_out", out.reshape(t, H, HD)[:, HEADS])
        g.add(f"long{n}_ids", ids)
        g.add(f"long{n}_scores", sc)
        g.add(f"long{n}_bounds", bd)
        g.add(f"long{n}_k", k_all[0, :, p0:ttot].transpose(0, 1).numpy())
        print("long case", n, "done", flush=True)
