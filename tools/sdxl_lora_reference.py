#!/usr/bin/env python3
"""f32 CPU reference for a LoRA on SDXL base 1.0: the oracle of the LoRA deltas (spec 8.3, 11.9a).

Run it through tools/sdxl_golden.sh's conventions. The direct command is:

  HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 <reference venv>/bin/python -I \
      tools/sdxl_lora_reference.py --model "$SDXL_MODEL_DIR" --out target/sdxl-lora-golden

The script writes a small LoRA of fixed values in kohya names, loads it with diffusers at weight 0.8, and records:

  lora.safetensors          the LoRA file that the test reads (kohya names, rank 4, alpha 4)
  lora.encode.safetensors   the prompt and negative encodings with the LoRA (clip_skip 2 in A1111's count), their pooled
                            outputs, and the encodings without the LoRA (`base_prompt_embeds`)
  lora.unet.safetensors     one UNet step at a fixed latent, with the LoRA (`unet_out`) and without it (`base_out`)

Diffusers applies a LoRA as `base(x) + weight * alpha / rank * up(down(x))`, as the engine does. The script stops with
an AssertionError when a self check fails. It runs on the CPU and holds about 20 GB of RAM. Do not run it with a GPU
test.

The modules: the UNet's attention `to_q` of level 1, the UNet's `conv1` of level 0 (a 3 x 3 convolution), the first
text encoder's `q_proj`, and the second text encoder's `k_proj`.
"""

import argparse
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdxl_reference import HEIGHT, NEGATIVE, PROMPT, WIDTH, encode  # noqa: E402

from diffusers import StableDiffusionXLPipeline  # noqa: E402

SEED = 7
RANK = 4
WEIGHT = 0.8

# The kohya name of each module, and its shape as (out, in) for a linear layer, or (out, in, k, k) for a convolution.
MODULES = {
    "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q": ("linear", 640, 640),
    "lora_unet_down_blocks_0_resnets_0_conv1": ("conv", 320, 320),
    "lora_te1_text_model_encoder_layers_0_self_attn_q_proj": ("linear", 768, 768),
    "lora_te2_text_model_encoder_layers_0_self_attn_k_proj": ("linear", 1280, 1280),
}


# The diffusers name of each module, as the underscore joined form of a kohya name gives it.
KNOWN = {
    "down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q": "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
    "down_blocks_0_resnets_0_conv1": "down_blocks.0.resnets.0.conv1",
    "text_model_encoder_layers_0_self_attn_q_proj": "text_model.encoder.layers.0.self_attn.q_proj",
    "text_model_encoder_layers_0_self_attn_k_proj": "text_model.encoder.layers.0.self_attn.k_proj",
}


def lora_file(gen):
    """The LoRA of fixed values: a down matrix and an up matrix for each module, and alpha = rank."""
    tensors = {}
    for name, (kind, cout, cin) in MODULES.items():
        if kind == "conv":
            down = torch.randn((RANK, cin, 3, 3), generator=gen) * 0.05
            up = torch.randn((cout, RANK, 1, 1), generator=gen) * 0.05
        else:
            down = torch.randn((RANK, cin), generator=gen) * 0.05
            up = torch.randn((cout, RANK), generator=gen) * 0.05
        tensors[f"{name}.lora_down.weight"] = down
        tensors[f"{name}.lora_up.weight"] = up
        tensors[f"{name}.alpha"] = torch.tensor(float(RANK))
    return tensors


def peft_names(tensors):
    """The LoRA of `tensors` under the diffusers names: `lora_A` is the down matrix, and `lora_B` the up matrix."""
    prefixes = {"lora_unet_": "unet.", "lora_te1_": "text_encoder.", "lora_te2_": "text_encoder_2."}
    out = {}
    for key, value in tensors.items():
        if key.endswith(".alpha"):
            continue
        # A kohya key has no dot in its module part, so the module is what comes before the first dot.
        module, role = key.split(".", 1)[0], key.split(".")[1]
        kohya = next(k for k in prefixes if module.startswith(k))
        dotted = KNOWN[module[len(kohya):]]
        suffix = "lora_A" if role == "lora_down" else "lora_B"
        out[f"{prefixes[kohya]}{dotted}.{suffix}.weight"] = value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)

    gen = torch.Generator().manual_seed(SEED)
    tensors = lora_file(gen)
    path = os.path.join(args.out, "lora.safetensors")
    save_file({k: v.contiguous() for k, v in tensors.items()}, path)

    pipe = StableDiffusionXLPipeline.from_pretrained(args.model, variant="fp16", torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)

    # The encodings without the LoRA, then with it. The engine's clip_skip 2 is diffusers' 1.
    base = {}
    for name, text in (("neg", NEGATIVE), ("pos", PROMPT)):
        ids1, ids2, hidden, pooled = encode(pipe, text, 1)
        base[name] = (hidden, pooled)

    # diffusers reads the same matrices under its own names. Its scale is `weight` with alpha equal to the rank, as the
    # kohya file says.
    pipe.load_lora_weights(peft_names(tensors), adapter_name="oracle")
    pipe.set_adapters(["oracle"], adapter_weights=[WEIGHT])
    lora = {}
    for name, text in (("neg", NEGATIVE), ("pos", PROMPT)):
        ids1, ids2, hidden, pooled = encode(pipe, text, 1)
        lora[f"{name}.prompt_embeds"] = hidden
        lora[f"{name}.pooled"] = pooled
        lora[f"{name}.base_prompt_embeds"] = base[name][0]
    encode_out = {
        "neg.prompt_embeds": lora["neg.prompt_embeds"],
        "pos.prompt_embeds": lora["pos.prompt_embeds"],
        "neg.base_prompt_embeds": lora["neg.base_prompt_embeds"],
        "pos.base_prompt_embeds": lora["pos.base_prompt_embeds"],
        "neg.pooled": lora["neg.pooled"],
        "pos.pooled": lora["pos.pooled"],
    }
    # The text deltas must change the encodings, or the oracle tests nothing.
    diff = (lora["pos.prompt_embeds"] - lora["pos.base_prompt_embeds"]).abs().max().item()
    assert diff > 1e-4, f"the LoRA changed no prompt encoding: {diff}"
    save_file({k: v.contiguous() for k, v in encode_out.items()}, os.path.join(args.out, "lora.encode.safetensors"))

    # One UNet step at a fixed latent and timestep. Row 0 is the negative prompt, row 1 the prompt.
    unet_gen = torch.Generator().manual_seed(SEED + 1)
    latent = torch.randn((1, 4, HEIGHT // 8, WIDTH // 8), generator=unet_gen)
    x = torch.cat([latent, latent])
    hidden = torch.stack([lora["neg.prompt_embeds"], lora["pos.prompt_embeds"]])
    pooled = torch.stack([lora["neg.pooled"], lora["pos.pooled"]])
    time_ids = torch.tensor([[HEIGHT, WIDTH, 0, 0, HEIGHT, WIDTH]] * 2, dtype=torch.float32)
    kwargs = {"text_embeds": pooled, "time_ids": time_ids}
    t = torch.tensor([999.0])
    with_lora = pipe.unet(x, t, encoder_hidden_states=hidden, added_cond_kwargs=kwargs).sample
    pipe.unload_lora_weights()
    without = pipe.unet(x, t, encoder_hidden_states=hidden, added_cond_kwargs=kwargs).sample
    diff = (with_lora - without).abs().max().item()
    assert diff > 1e-4, f"the LoRA changed no UNet output: {diff}"
    save_file(
        {
            "unet_out": with_lora.contiguous(),
            "base_out": without.contiguous(),
            "latent": latent.contiguous(),
        },
        os.path.join(args.out, "lora.unet.safetensors"),
    )
    print(f"wrote {args.out}: the LoRA moved the UNet output by {diff:.3e}")


if __name__ == "__main__":
    main()
