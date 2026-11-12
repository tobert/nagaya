"""The group `hc`: the hyper-connection ops against torch (test_hc.py of mk).

`hc_pre` runs the HF module `Qwen4ExpTextGatedResidual` of mk's transformers with SYNTHETIC weights that the generator
makes at the real shapes (mk used the real weights of layer 0). The Rust test remakes the weights and calls `mk_hc_pre`.
`hc_combine` is `h + out x inject` in torch, which must match the op bit for bit.
"""
import sys
import types

import numpy as np

SOURCE_FILES = ["tests/ops/test_hc.py"]

E, R, T_PRE = 2560, 320, 3
# The f16 d of the Q8_0 weights: small, so that the activations of the module do not saturate (the Rust test does the same).
D_DOWN, D_UP = 0x0C00, 0x1C00
INJECT_SCALE = np.float32(0.01)


def fixed_d(raw, bits):
    """Set the f16 d of every Q8_0 block (34 bytes) of `raw` to `bits`."""
    flat = raw.reshape(-1, 34).copy()
    flat[:, 0:2] = np.array([bits], dtype="<u2").view(np.uint8)
    return flat.reshape(raw.shape)


def make(g, lib):
    sys.path.insert(0, str(lib.GGUF_PY))
    import gguf
    import torch

    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    d4 = 4 * E
    cfg = types.SimpleNamespace(hc_count=4, hidden_size=E, hc_lowrank=R, rms_norm_eps=1e-6)
    hc = Qwen4ExpTextGatedResidual(cfg)
    q8 = gguf.GGMLQuantizationType.Q8_0
    down = fixed_d(lib.raw_rows("Q8_0", 701, R, d4), D_DOWN)
    up = fixed_d(lib.raw_rows("Q8_0", 702, d4, R), D_UP)
    inject = (lib.normals(703, 4 * d4) * INJECT_SCALE).reshape(4, d4)
    gamma = (np.float32(1.0) + lib.normals(704, d4) * np.float32(0.1)).astype(np.float32)
    deq = lambda raw, rows, k: gguf.quants.dequantize(raw, q8).astype(np.float32).reshape(rows, k)
    with torch.no_grad():
        hc.input_mix_weight_down.weight.copy_(torch.from_numpy(deq(down, R, d4)))
        hc.input_mix_weight_up.weight.copy_(torch.from_numpy(deq(up, d4, R)))
        hc.block_inject_weight.weight.copy_(torch.from_numpy(inject))
        hc.hc_norm.weight.copy_(torch.from_numpy(gamma - np.float32(1.0)))
        base = lib.normals(705, T_PRE * E).reshape(T_PRE, E)
        noise = lib.normals(706, T_PRE * d4).reshape(T_PRE, d4)
        h = np.tile(base, (1, 4)) + np.float32(0.01) * noise
        mixed, _, inj = hc(torch.from_numpy(h.astype(np.float32)).unsqueeze(0))
    g.add("pre_mixed", mixed[0].numpy())
    g.add("pre_inject", inj[0].numpy())
    g.meta["pre"] = {"t": T_PRE, "d_down": D_DOWN, "d_up": D_UP}

    t = 2
    h = lib.normals(710, t * d4).reshape(t, d4)
    out = lib.normals(711, t * E).reshape(t, E)
    inj = lib.normals(712, t * 4).reshape(t, 4)
    want = (torch.from_numpy(h) + (torch.from_numpy(out).unsqueeze(-2) * torch.from_numpy(inj).unsqueeze(-1)).flatten(-2)).numpy()
    g.add("combine", want)
