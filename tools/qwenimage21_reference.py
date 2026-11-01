#!/usr/bin/env python3
"""f32 CPU reference for Qwen-Image 2.1 with FUSAL: the image oracle (spec 8.3) and references for kernel bring-up.

Run it through tools/qwenimage21_golden.sh. The tensor names, shapes and meanings are in that script.

The script uses diffusers in ~/.venvs/nagaya-qwenimage21-ref (a copy of diffusepal's venv),
on the CPU only. It loads the bf16 weights as float32, one model at a
time: the text encoder, then the transformer, then the VAE. Each model is freed before the next one loads, so the peak
is about 33 GiB of RAM. The pipeline itself runs each stage: the text encoder through `_get_qwen_prompt_embeds`, and
the transformer and the scheduler through `__call__` with the recorded `prompt_embeds`.

The engine makes its noise with its own random stream (spec 11.9), so the script draws the start latent with torch,
saves it, and gives it to diffusers through `latents`. FlowMatch Euler draws no noise in a step.

The script stops with an AssertionError when a self check fails. The checks:

  - the template, the tokens and the cut of the system tokens give the pipeline's `prompt_embeds`;
  - the last hidden state is the input of the encoder's final norm, not its output;
  - each step is one Euler step: latent + (next sigma - sigma) * transformer output;
  - a step with the cached keys and values of the condition gives the step without the cache;
  - the transformer with the LoRA at weight 1 and no cache gives step 0 of the run;
  - the manual RGBA bytes are the bytes of the pipeline's image processor.
"""

import argparse
import gc
import hashlib
import json
import os
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import diffusers
import peft
import transformers
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

PROMPT = "fusal style. a red fox in a snowy forest, watercolor"
WIDTH, HEIGHT = 256, 256
STEPS = 4
SEED = 1234
LORA = "loras/fusal-qwen-image-2.1-style-hires-exact-r56.safetensors"
HERETIC = "text_encoders/qwen3-vl-8b-heretic-1.3.0_bf16.safetensors"
LORA_WEIGHT = 1.0

# The encoder layers whose output the parts file keeps. hidden_states[0] is the token embedding, and
# hidden_states[i] is the output of layer i - 1.
ENCODER_LAYERS = (1, 18, 36)
# The transformer blocks whose output the parts file keeps, at step 0.
BLOCKS = (0, 15, 31)


def heretic_to_transformers(key):
    """Map the ComfyUI key names of the Heretic file onto the transformers Qwen3-VL layout (diffusepal models.py)."""
    for old, new in (
        ("model.layers.", "model.language_model.layers."),
        ("model.embed_tokens.", "model.language_model.embed_tokens."),
        ("model.norm.", "model.language_model.norm."),
    ):
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def f32(t):
    return t.detach().to(torch.float32).contiguous().clone()


def first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise AssertionError(f"no tensor in a module output of type {type(output)}")


def close(name, got, want, atol, rtol=0.0):
    diff = (got.to(torch.float64) - want.to(torch.float64)).abs()
    bound = atol + rtol * want.to(torch.float64).abs()
    worst = float(diff.max())
    print(f"check {name}: max |diff| {worst:.3e}")
    assert bool((diff <= bound).all()), f"{name}: max |diff| {worst:.3e} is over atol {atol} rtol {rtol}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def encode(base, fusal, processor, scheduler):
    """Run the Heretic encoder in f32. Return the encode tensors, the parts tensors and the pipeline's embeds."""
    config = AutoConfig.from_pretrained(os.path.join(base, "text_encoder"))
    te = Qwen3VLForConditionalGeneration._from_config(config, dtype=torch.bfloat16)
    state = load_file(os.path.join(fusal, HERETIC))
    te.load_state_dict({heretic_to_transformers(k): v for k, v in state.items()}, strict=True)
    del state
    gc.collect()
    te = te.float().eval()

    pipe = QwenImage21Pipeline(scheduler=scheduler, vae=None, text_encoder=te, processor=processor, transformer=None)
    embeds, mask, pad = pipe._get_qwen_prompt_embeds(PROMPT, None, torch.device("cpu"))
    assert mask.shape[0] == 1 and bool(mask.all()), "one prompt has no padding"
    assert not bool(pad.any()), "a text-to-image prompt has no image tokens"

    # The same encoding by hand, with the layer outputs for bring-up.
    text = pipe.prompt_template_t2i.format(PROMPT)
    inputs = processor(text=[text], padding=True, padding_side="left", return_tensors="pt")
    ids = inputs.input_ids
    drop = pipe._drop_idx
    language = te.model.language_model
    norm_in = {}

    def norm_input(module, args, output):
        norm_in.setdefault("x", args[0])

    norm_hook = language.norm.register_forward_hook(norm_input)
    out = te(input_ids=ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
    norm_hook.remove()
    hidden = out.hidden_states
    assert len(hidden) == config.text_config.num_hidden_layers + 1, f"{len(hidden)} hidden states"
    # The pipeline wants the input of the final norm (pipeline_qwenimage21.py, the hook in _get_qwen_prompt_embeds).
    # transformers 4 gives that input as hidden_states[-1]. transformers 5 gives the norm's output there instead.
    last = norm_in["x"]
    normed = language.norm(last)
    assert float((normed - last).abs().max()) > 1e-2, "the final norm changes the hidden state"
    if torch.equal(hidden[-1], last):
        last_is = "norm input"
    elif torch.equal(hidden[-1], normed):
        last_is = "norm output"
    else:
        raise AssertionError("hidden_states[-1] is neither the input nor the output of the final norm")
    print(f"check hidden_states[-1] of transformers {transformers.__version__} is the {last_is}")
    close("prompt_embeds by hand", last[:, drop:], embeds, atol=1e-5, rtol=1e-5)

    enc = {
        "ids": ids.to(torch.int64).contiguous(),
        "drop": torch.tensor([drop], dtype=torch.int64),
        "prompt_embeds": f32(embeds),
    }
    parts = {f"enc.hidden{i}": f32(hidden[i][:, drop:]) for i in (0,) + ENCODER_LAYERS[:-1]}
    parts[f"enc.hidden{ENCODER_LAYERS[-1]}"] = f32(last[:, drop:])
    parts["enc.hidden_all_tokens_last"] = f32(last)
    template = {"t2i": pipe.prompt_template_t2i, "system": pipe.sys_prompt, "hidden_states_last": last_is}
    del pipe, te, out, hidden
    gc.collect()
    return enc, parts, embeds.detach().clone(), template


def encode_bf16(base, fusal, ids):
    """The Heretic encoder in bf16, as diffusepal runs it: the baseline of the conditioning gate (spec 8.3)."""
    config = AutoConfig.from_pretrained(os.path.join(base, "text_encoder"))
    te = Qwen3VLForConditionalGeneration._from_config(config, dtype=torch.bfloat16)
    state = load_file(os.path.join(fusal, HERETIC))
    te.load_state_dict({heretic_to_transformers(k): v for k, v in state.items()}, strict=True)
    del state
    gc.collect()
    te = te.eval()
    norm_in = {}
    hook = te.model.language_model.norm.register_forward_hook(lambda m, a, o: norm_in.setdefault("x", a[0]))
    te(input_ids=ids, attention_mask=torch.ones_like(ids))
    hook.remove()
    out = f32(norm_in["x"])
    del te
    gc.collect()
    return out


def denoise(base, fusal, processor, scheduler, embeds):
    """Run the transformer in f32 with the FUSAL LoRA. Return the steps tensors and the parts tensors."""
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        os.path.join(base, "transformer"), torch_dtype=torch.float32
    ).eval()
    pipe = QwenImage21Pipeline(
        scheduler=scheduler, vae=None, text_encoder=None, processor=processor, transformer=transformer
    )
    pipe.load_lora_weights(fusal, weight_name=LORA, adapter_name="fusal")
    pipe.set_adapters(["fusal"], adapter_weights=[LORA_WEIGHT])

    h, w = HEIGHT // 16, WIDTH // 16
    generator = torch.Generator().manual_seed(SEED)
    start = torch.randn((1, 1, 64, h, w), generator=generator, dtype=torch.float32)
    packed = start.view(1, 64, h * w).transpose(1, 2).contiguous()

    expected = {"img_in", "txt_in", "time_text_embed", "modulation", "transformer_blocks", "norm_out", "proj_out"}
    children = {name for name, _ in transformer.named_children()}
    assert expected <= children, f"the transformer lacks {sorted(expected - children)}"

    calls = []
    parts = {}
    step0 = {"on": True}

    def pre(module, args, kwargs):
        calls.append({"kwargs": dict(kwargs)})

    def post(module, args, kwargs, output):
        calls[-1]["out"] = first_tensor(output)
        step0["on"] = False

    def keep(name):
        def hook(module, args, output):
            if step0["on"]:
                parts[name] = f32(first_tensor(output))

        return hook

    handles = [
        transformer.register_forward_pre_hook(pre, with_kwargs=True),
        transformer.register_forward_hook(post, with_kwargs=True),
    ]
    for name in ("img_in", "txt_in", "time_text_embed", "modulation", "norm_out", "proj_out"):
        handles.append(getattr(transformer, name).register_forward_hook(keep(f"tr.{name}")))
    for i in BLOCKS:
        handles.append(transformer.transformer_blocks[i].register_forward_hook(keep(f"tr.block{i}")))

    latents = [f32(packed)]

    def on_step(p, i, t, kw):
        latents.append(f32(kw["latents"]))
        return {}

    pipe(
        prompt_embeds=embeds,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        latents=packed.clone(),
        output_type="latent",
        callback_on_step_end=on_step,
        callback_on_step_end_tensor_inputs=["latents"],
        use_kv_cache=True,
    )
    for handle in handles:
        handle.remove()
    assert len(calls) == STEPS and len(latents) == STEPS + 1, f"{len(calls)} calls, {len(latents)} latents"

    n = packed.shape[1]
    outs = torch.stack([f32(c["out"][:, -n:]) for c in calls])
    ins = torch.stack([f32(c["kwargs"]["hidden_states"][:, -n:]) for c in calls])
    timesteps = torch.stack([f32(c["kwargs"]["timestep"]) for c in calls])
    sigmas = f32(scheduler.sigmas)
    assert sigmas.shape[0] == STEPS + 1, f"{sigmas.shape[0]} sigmas"
    lat = torch.stack(latents)
    close("the transformer input is the latent", ins, lat[:-1], atol=0.0)
    for i in range(STEPS):
        euler = lat[i] + (sigmas[i + 1] - sigmas[i]) * outs[i]
        close(f"euler step {i}", lat[i + 1], euler, atol=1e-5, rtol=1e-5)

    # Replay steps 0 and 1 with no cache. The cache of the condition is exact, so the outputs agree.
    def replay(i):
        kwargs = dict(calls[i]["kwargs"])
        kwargs["kv_cache"] = None
        kwargs["kv_cache_mode"] = None
        return f32(first_tensor(transformer(**kwargs))[:, -n:])

    close("step 0 with no cache", replay(0), outs[0], atol=1e-4, rtol=1e-4)
    close("step 1 with no cache", replay(1), outs[1], atol=1e-4, rtol=1e-4)
    pipe.set_adapters(["fusal"], adapter_weights=[0.0])
    base_out = replay(0)
    pipe.set_adapters(["fusal"], adapter_weights=[LORA_WEIGHT])
    assert float((base_out - outs[0]).abs().max()) > 1e-3, "the LoRA changes the output of step 0"
    parts["tr.step0_no_lora"] = base_out
    parts["tr.timestep0"] = timesteps[0]

    # The baselines of the step and run gates (spec 8.3): the same transformer and LoRA in bf16, as diffusepal runs
    # them. Each call replays from its f32 inputs rounded to bf16, and the whole run starts from the same latent.
    transformer.to(torch.bfloat16)

    def bf16(value):
        return value.to(torch.bfloat16) if torch.is_tensor(value) and value.is_floating_point() else value

    def replay16(i):
        kwargs = {key: bf16(value) for key, value in calls[i]["kwargs"].items()}
        kwargs["kv_cache"] = None
        kwargs["kv_cache_mode"] = None
        return f32(first_tensor(transformer(**kwargs))[:, -n:])

    outs16 = torch.stack([replay16(i) for i in range(STEPS)])
    latents16 = [f32(packed)]

    def on_step16(p, i, t, kw):
        latents16.append(f32(kw["latents"]))
        return {}

    pipe(
        prompt_embeds=embeds.to(torch.bfloat16),
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        latents=packed.to(torch.bfloat16),
        output_type="latent",
        callback_on_step_end=on_step16,
        callback_on_step_end_tensor_inputs=["latents"],
        use_kv_cache=True,
    )
    assert len(latents16) == STEPS + 1, f"{len(latents16)} bf16 latents"

    first = calls[0]["kwargs"]
    steps = {
        "start": f32(start),
        "packed": f32(packed),
        "sigmas": sigmas,
        "timesteps": timesteps,
        "latents": lat,
        "transformer_out": outs,
        "final_latent": lat[-1].clone(),
        "transformer_out_bf16": outs16,
        "latents_bf16": torch.stack(latents16),
    }
    config = scheduler.config
    mu = calculate_shift(
        n,
        config.get("base_image_seq_len", 256),
        config.get("max_image_seq_len", 4096),
        config.get("base_shift", 0.5),
        config.get("max_shift", 1.15),
    )
    meta = {
        "img_shapes": json.dumps([list(map(list, s)) for s in first["img_shapes"]]),
        "mu": json.dumps(mu),
        "encoder_hidden_states_mask": "none" if first.get("encoder_hidden_states_mask") is None else "set",
    }
    del pipe, transformer, calls
    gc.collect()
    return steps, parts, meta


def decode(base, final_latent):
    """Run the VAE decode in f32. Return the image tensors and the parts tensors."""
    vae = AutoencoderKLQwenImage21.from_pretrained(os.path.join(base, "vae"), torch_dtype=torch.float32).eval()
    h, w = HEIGHT // 16, WIDTH // 16
    z = final_latent.transpose(1, 2).reshape(1, 64, 1, h, w)
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, 64, 1, 1, 1)
    z = z * std + mean

    parts = {"vae.in": f32(z)}

    def keep(name):
        def hook(module, args, output):
            if name not in parts:
                parts[name] = f32(first_tensor(output))

        return hook

    handles = [vae.post_quant_conv.register_forward_hook(keep("vae.post_quant_conv"))]
    for name, child in vae.decoder.named_children():
        if isinstance(child, torch.nn.ModuleList):
            for i, sub in enumerate(child):
                handles.append(sub.register_forward_hook(keep(f"vae.{name}{i}")))
        elif sum(1 for _ in child.parameters()) > 0:
            handles.append(child.register_forward_hook(keep(f"vae.{name}")))
    image = vae.decode(z, return_dict=False)[0][:, :, 0]
    for handle in handles:
        handle.remove()
    assert "vae.conv_out" in parts, f"no conv_out among {sorted(parts)}"

    image = f32(image[0])
    assert image.shape[0] == 4, f"the VAE gives RGBA, not {image.shape[0]} channels"
    rgba = torch.round(torch.clamp(image / 2 + 0.5, 0, 1) * 255).to(torch.uint8).permute(1, 2, 0).contiguous()
    processor = diffusers.image_processor.VaeImageProcessor(vae_scale_factor=16)
    pil = processor.postprocess(image[None], output_type="pil")[0]
    assert pil.mode == "RGBA" and rgba.numpy().tobytes() == pil.tobytes(), "the manual bytes are the pipeline's"

    # The baseline of the decode gate (spec 8.3): the VAE in bf16, with the latent made in bf16 as the pipeline does.
    vae.to(torch.bfloat16)
    z16 = final_latent.to(torch.bfloat16).transpose(1, 2).reshape(1, 64, 1, h, w)
    z16 = z16 * std.to(torch.bfloat16) + mean.to(torch.bfloat16)
    image16 = vae.decode(z16, return_dict=False)[0][:, :, 0]
    pil16 = processor.postprocess(image16, output_type="pil")[0]
    rgba16 = torch.frombuffer(bytearray(pil16.tobytes()), dtype=torch.uint8).view(HEIGHT, WIDTH, 4).clone()
    del vae
    gc.collect()
    return {"image": image, "rgba": rgba, "rgba_bf16": rgba16}, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the Qwen-Image-2.1 directory")
    ap.add_argument("--fusal", required=True, help="the Qwen-Image-2.1-Anime-Fusal directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert not torch.cuda.is_available(), "the reference runs on the CPU only; hide the GPU"
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)

    began = time.time()
    processor = AutoProcessor.from_pretrained(os.path.join(args.model, "processor"))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(args.model, "scheduler"))

    enc, enc_parts, embeds, template = encode(args.model, args.fusal, processor, scheduler)
    print(f"encode done, {time.time() - began:.0f} s, {embeds.shape[1]} tokens after the cut")
    enc["prompt_embeds_bf16"] = encode_bf16(args.model, args.fusal, enc["ids"])[:, int(enc["drop"][0]) :].contiguous()
    print(f"bf16 encode done, {time.time() - began:.0f} s")
    steps, tr_parts, tr_meta = denoise(args.model, args.fusal, processor, scheduler, embeds)
    print(f"denoise done, {time.time() - began:.0f} s")
    images, vae_parts = decode(args.model, steps["final_latent"])
    print(f"decode done, {time.time() - began:.0f} s")

    lora_path = os.path.join(args.fusal, LORA)
    heretic_path = os.path.join(args.fusal, HERETIC)
    meta = {
        "request": json.dumps({
            "prompt": PROMPT, "width": WIDTH, "height": HEIGHT, "steps": STEPS, "seed": SEED,
            "true_cfg_scale": 1.0, "lora": LORA, "lora_weight": LORA_WEIGHT, "encoder": HERETIC,
        }),
        "template": json.dumps(template),
        "scheduler": json.dumps(dict(scheduler.config)),
        "versions": json.dumps({
            "torch": torch.__version__, "diffusers": diffusers.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
        }),
        "lora_sha256": sha256(lora_path),
        "heretic_bytes": str(os.path.getsize(heretic_path)),
        "img_shapes": tr_meta["img_shapes"],
        "mu": tr_meta["mu"],
        "encoder_hidden_states_mask": tr_meta["encoder_hidden_states_mask"],
        "precision": "bf16 weights, f32 compute, CPU; the *_bf16 tensors are the bf16 baselines of the gates",
    }
    save_file(enc, os.path.join(args.out, "encode.safetensors"), metadata=meta)
    save_file({**steps, **images}, os.path.join(args.out, "steps.safetensors"), metadata=meta)
    save_file({**enc_parts, **tr_parts, **vae_parts}, os.path.join(args.out, "parts.safetensors"), metadata=meta)
    np.save(os.path.join(args.out, "rgba.npy"), images["rgba"].numpy())
    with safe_open(os.path.join(args.out, "parts.safetensors"), "pt") as f:
        for key in sorted(f.keys()):
            print(f"  parts {key} {list(f.get_slice(key).get_shape())}")
    print(f"done in {time.time() - began:.0f} s: {args.out}")


if __name__ == "__main__":
    main()
