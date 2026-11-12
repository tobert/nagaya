"""The group `gdn`: the gated DeltaNet core (test_gdn.py) against the transformers code that mk uses as its reference.

The weights and the inputs come from the generator, so Rust remakes them. The reference is `ref_gdn` of mk's test, run on
the CPU. The fixture holds the output `y` of three value heads and a corner of the final recurrent state of the same heads.
"""
import sys

import numpy as np

SOURCE_FILES = ["tests/ops/test_gdn.py"]

KH, VH, KD, VD, CONV = 16, 48, 128, 128, 4
CDIM = 2 * KH * KD + VH * VD
HEADS = [0, 17, 47]  # the value heads in the fixture
STATE_ROWS = 16  # the first rows of the K dimension of the state in the fixture
CASES = [(1, True), (1, False), (7, True), (7, False), (20, True), (20, False)]


def weights(lib):
    ssm_a = -(np.float32(0.5) + np.float32(4.0) * lib.uniforms(9001, VH))
    dt_bias = np.float32(0.1) * lib.normals(9002, VH)
    conv_w = (np.float32(0.3) * lib.normals(9003, CDIM * CONV)).reshape(CDIM, CONV)
    gamma = np.float32(1.0) + np.float32(0.1) * lib.normals(9004, VD)
    return [np.ascontiguousarray(x, np.float32) for x in (ssm_a, dt_bias, conv_w, gamma)]


def inputs(lib, t, scale):
    base = 10000 + 10 * t + (0 if scale == 0 else 1)
    f = np.float32
    qkv = lib.normals(base * 10 + 1, t * CDIM).reshape(t, CDIM)
    z = lib.normals(base * 10 + 2, t * VH * VD).reshape(t, VH * VD)
    a = (f(0.5) * lib.normals(base * 10 + 3, t * VH)).reshape(t, VH)
    b = (f(0.5) * lib.normals(base * 10 + 4, t * VH)).reshape(t, VH)
    conv_state = (lib.normals(base * 10 + 5, CDIM * (CONV - 1)) * f(scale)).reshape(CDIM, CONV - 1)
    state = (lib.normals(base * 10 + 6, VH * KD * VD) * f(scale)).reshape(VH, KD, VD)
    return [np.ascontiguousarray(x, np.float32) for x in (qkv, z, a, b, conv_state, state)]


def make(g, lib):
    sys.path.insert(0, str(lib.MK))
    import torch
    import torch.nn.functional as F
    from harness import reference
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextRMSNormGated, causal_conv1d_fn, torch_chunk_gated_delta_rule)

    c = reference.text_config()
    g.meta["config"] = {"hidden_act": c.hidden_act, "output_gate_type": c.output_gate_type,
                        "linear_conv_kernel_dim": c.linear_conv_kernel_dim}
    assert c.hidden_act == "silu" and c.output_gate_type == "sigmoid" and c.linear_conv_kernel_dim == CONV
    ssm_a, dt_bias, conv_w, gamma = weights(lib)
    assert (ssm_a < 0).all()
    g.meta["heads"] = HEADS
    g.meta["state_rows"] = STATE_ROWS
    g.meta["cases"] = []
    torch.set_num_threads(8)
    for t, from_zero in CASES:
        qkv, z, a, b, conv_state, state_in = inputs(lib, t, 0.0 if from_zero else 0.3)
        with torch.no_grad():
            x = torch.from_numpy(qkv).unsqueeze(0).transpose(1, 2)
            ext = torch.cat([torch.from_numpy(conv_state).unsqueeze(0), x], dim=-1)
            conv_out = causal_conv1d_fn(ext, torch.from_numpy(conv_w), None, activation="silu")[:, :, -t:]
            co = conv_out.squeeze(0).transpose(0, 1).contiguous()
            q16 = co[:, :KH * KD].reshape(t, KH, KD)
            k16 = co[:, KH * KD:2 * KH * KD].reshape(t, KH, KD)
            v = co[:, 2 * KH * KD:].reshape(t, VH, VD)
            kh = torch.arange(VH) % KH
            beta = torch.from_numpy(b).sigmoid()
            gg = torch.from_numpy(ssm_a).unsqueeze(0) * F.softplus(torch.from_numpy(a) + torch.from_numpy(dt_bias).unsqueeze(0))
            out, final = torch_chunk_gated_delta_rule(
                q16[:, kh, :].unsqueeze(0), k16[:, kh, :].unsqueeze(0), v.unsqueeze(0), g=gg.unsqueeze(0),
                beta=beta.unsqueeze(0), initial_state=torch.from_numpy(state_in).unsqueeze(0), output_final_state=True,
                use_qk_l2norm_in_kernel=True)
            norm = Qwen4ExpTextRMSNormGated(VD, eps=1e-6, activation="sigmoid")
            norm.weight.copy_(torch.from_numpy(gamma))
            y = norm(out.squeeze(0).reshape(-1, VD), torch.from_numpy(z).reshape(-1, VD)).reshape(t, VH, VD)
        name = f"t{t}_{'zero' if from_zero else 'prior'}"
        g.add(f"{name}_y", y.numpy()[:, HEADS, :])
        g.add(f"{name}_state", final.squeeze(0).numpy()[HEADS, :STATE_ROWS, :])
        g.meta["cases"].append({"name": name, "t": t, "from_zero": from_zero})
