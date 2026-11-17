#!/usr/bin/env python3
"""f32 CPU reference for SDXL base 1.0: the image oracle (spec 8.3, 11.9) and references for kernel bring-up.

Run it through tools/sdxl_golden.sh. The direct command is:

  HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 <reference venv>/bin/python -I \
      tools/sdxl_reference.py --model "$SDXL_MODEL_DIR" --out target/sdxl-golden

The script uses diffusers in the reference venv, on the CPU only. It loads the fp16 weights of the checkpoint as
float32. The tensor names, shapes and meanings are in tools/sdxl_golden.sh.

The engine makes its noise with its own random stream (spec 11.9), so the script draws the noise with torch, saves it,
and gives it to diffusers: the start latent through `latents`, and the noise of each step by a stand-in for
`randn_tensor` in the scheduler modules.

The script stops with an AssertionError when a self check fails. The checks:

  - the A1111 encoding with clip_skip 1 gives diffusers' `encode_prompt` for both prompts. Here `clip_skip` counts
    as diffusers does, 1 less than A1111 and the engine (docs/spec.md 11.9a);
  - each step of each sampler used the noise that the script gave it, and no other draw;
  - the explicit guidance of the recorded UNet outputs gives the latent of the next step;
  - the manual VAE decode gives the image of the pipeline;
  - the hooks of the UNet and the VAE give the output of the whole module.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from safetensors.torch import save_file

import diffusers
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.schedulers import (
    scheduling_dpmsolver_multistep,
    scheduling_euler_ancestral_discrete,
    scheduling_euler_discrete,
)

PROMPT = "a red fox in a snowy forest, watercolor"
NEGATIVE = "blurry, low quality"
WIDTH, HEIGHT = 256, 320
STEPS = 4
CFG = 7.0
SEED = 1234

# The samplers of spec 11.9 and the diffusers scheduler of each, from the config of the checkpoint.
SAMPLERS = {
    "euler": (EulerDiscreteScheduler, {}),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "dpmpp_2m_karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "dpmpp_2m_sde_karras": (
        DPMSolverMultistepScheduler,
        {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"},
    ),
}


def check(ok, what):
    """Stop with the name of the self check that failed."""
    if not ok:
        raise AssertionError(f"self check failed: {what}")


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def close(a, b, tol, what):
    err = max_abs(a, b)
    check(err <= tol, f"{what}: max abs error {err} > {tol}")
    return err


def tokens(tokenizer, text):
    """The 77 ids of one chunk, padded as the tokenizer of the checkpoint pads."""
    return tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids


@torch.no_grad()
def encode(pipe, text, clip_skip):
    """The A1111 encoding of one prompt with no weights: the hidden state of layer -(clip_skip + 1) of each encoder,
    joined along the features, and the pooled output of the second encoder."""
    ids1 = tokens(pipe.tokenizer, text)
    ids2 = tokens(pipe.tokenizer_2, text)
    out1 = pipe.text_encoder(ids1, output_hidden_states=True)
    out2 = pipe.text_encoder_2(ids2, output_hidden_states=True)
    hidden = torch.cat([out1.hidden_states[-(clip_skip + 1)], out2.hidden_states[-(clip_skip + 1)]], dim=-1)
    return ids1[0], ids2[0], hidden[0], out2.text_embeds[0]


class Draws:
    """A stand-in for `randn_tensor` that gives the saved noise of each step, in order."""

    def __init__(self, noise):
        self.noise = list(noise)
        self.used = 0

    def __call__(self, shape, generator=None, device=None, dtype=None, layout=None):
        check(self.used < len(self.noise), "a scheduler drew more noise than the steps give")
        value = self.noise[self.used]
        check(tuple(value.shape) == tuple(shape), f"noise shape {tuple(shape)}")
        self.used += 1
        return value.to(dtype=dtype or value.dtype)


def hooked(modules, record, prefix):
    """Record the output of each named module. A tuple output keeps its first item."""
    handles = []
    for name, module in modules.items():
        def hook(_m, _inp, out, name=name):
            if (prefix + name) not in record:
                value = out[0] if isinstance(out, tuple) else out
                record[prefix + name] = value.detach().clone()
        handles.append(module.register_forward_hook(hook))
    return handles


@torch.no_grad()
def run_sampler(pipe, name, embeds, start, step_noise, parts):
    """Run one sampler with the saved noise. Give its tensors, keyed by name."""
    cls, extra = SAMPLERS[name]
    pipe.scheduler = cls.from_config(pipe.scheduler_config, **extra)
    draws = Draws(step_noise)
    for module in (scheduling_euler_discrete, scheduling_euler_ancestral_discrete, scheduling_dpmsolver_multistep):
        module.randn_tensor = draws

    unet_out = []
    unet_in = []

    def unet_hook(_m, args, kwargs, out):
        unet_in.append((args[0] if args else kwargs["sample"]).detach().clone())
        unet_out.append((out[0] if isinstance(out, tuple) else out.sample).detach().clone())

    handle = pipe.unet.register_forward_hook(unet_hook, with_kwargs=True)
    handles = []
    latents = []

    def on_step(_pipe, _i, _t, kw):
        latents.append(kw["latents"].detach().clone())
        return kw

    record = {}
    if name == "euler":
        handles = hooked(parts, record, "unet.")
    try:
        result = pipe(
            prompt_embeds=embeds["pos"][None],
            negative_prompt_embeds=embeds["neg"][None],
            pooled_prompt_embeds=embeds["pos_pooled"][None],
            negative_pooled_prompt_embeds=embeds["neg_pooled"][None],
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=CFG,
            latents=start.clone(),
            output_type="latent",
            callback_on_step_end=on_step,
        )
    finally:
        handle.remove()
        for h in handles:
            h.remove()
    check(len(unet_out) == STEPS, f"{name}: one UNet pass for each step")
    sched = pipe.scheduler
    used = draws.used
    noise_tensor = torch.stack(step_noise) if used else torch.zeros(0)
    check(used in (0, STEPS - 1, STEPS), f"{name}: {used} draws of noise in {STEPS} steps")
    out = {
        f"{name}.sigmas": sched.sigmas.float(),
        f"{name}.timesteps": sched.timesteps.float(),
        f"{name}.unet_in": torch.stack(unet_in),
        f"{name}.unet_out": torch.stack(unet_out),
        f"{name}.latents": torch.stack([start * sched.init_noise_sigma] + latents),
        f"{name}.step_noise": noise_tensor[:used] if used else torch.zeros(0, *start.shape),
    }
    # The guidance of the recorded UNet outputs: uncond + cfg * (cond - uncond), uncond first.
    guided = torch.stack([u[0:1] + CFG * (u[1:2] - u[0:1]) for u in unet_out])
    out[f"{name}.guided"] = guided
    if name == "euler":
        # One Euler step of the guided output gives the next latent: x + (sigma_next - sigma) * eps.
        x0 = start * sched.init_noise_sigma
        x1 = x0 + (sched.sigmas[1] - sched.sigmas[0]) * guided[0]
        close(x1, latents[0], 1e-3, "euler step 0 from the guided output")
        out.update(record)
    return out, result.images


@torch.no_grad()
def decode(pipe, latent, parts, record):
    """Decode one latent as the pipeline does, and record the parts of the VAE decoder."""
    handles = hooked(parts, record, "vae.")
    try:
        scaled = latent / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(scaled, return_dict=False)[0]
    finally:
        for h in handles:
            h.remove()
    return scaled, image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)
    started = time.time()

    pipe = StableDiffusionXLPipeline.from_pretrained(args.model, variant="fp16", torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler_config = dict(pipe.scheduler.config)
    meta = {
        "prompt": PROMPT,
        "negative": NEGATIVE,
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "steps": str(STEPS),
        "cfg": str(CFG),
        "seed": str(SEED),
        "diffusers": diffusers.__version__,
        "torch": torch.__version__,
        "scheduler_config": json.dumps(pipe.scheduler_config, default=str),
        "vae_scaling_factor": str(pipe.vae.config.scaling_factor),
    }

    # The encoders. Row 0 is the negative prompt and row 1 the prompt, as the pipeline batches them.
    enc = {}
    embeds = {}
    for skip in (1, 2):
        n_ids1, n_ids2, n_hidden, n_pooled = encode(pipe, NEGATIVE, skip)
        p_ids1, p_ids2, p_hidden, p_pooled = encode(pipe, PROMPT, skip)
        enc[f"clip_skip{skip}.prompt_embeds"] = torch.stack([n_hidden, p_hidden])
        enc[f"clip_skip{skip}.pooled"] = torch.stack([n_pooled, p_pooled])
        if skip == 1:
            enc["ids1"] = torch.stack([n_ids1, p_ids1])
            enc["ids2"] = torch.stack([n_ids2, p_ids2])
            embeds = {"neg": n_hidden, "pos": p_hidden, "neg_pooled": n_pooled, "pos_pooled": p_pooled}
    ref = pipe.encode_prompt(PROMPT, negative_prompt=NEGATIVE, do_classifier_free_guidance=True, device="cpu")
    close(ref[0][0], embeds["pos"], 1e-5, "the prompt encoding of diffusers")
    close(ref[1][0], embeds["neg"], 1e-5, "the negative encoding of diffusers")
    close(ref[2][0], embeds["pos_pooled"], 1e-5, "the pooled prompt of diffusers")
    close(ref[3][0], embeds["neg_pooled"], 1e-5, "the pooled negative of diffusers")
    # The time ids of SDXL: original size, crop origin, target size, as height then width.
    enc["add_time_ids"] = torch.tensor([HEIGHT, WIDTH, 0, 0, HEIGHT, WIDTH], dtype=torch.float32)
    save_file({k: v.contiguous() for k, v in enc.items()}, os.path.join(args.out, "encode.safetensors"), metadata=meta)

    # The noise: the start latent, then one draw for each step of each sampler.
    gen = torch.Generator().manual_seed(SEED)
    shape = (1, 4, HEIGHT // 8, WIDTH // 8)
    start = torch.randn(shape, generator=gen)
    step_noise = [torch.randn(shape, generator=gen) for _ in range(STEPS)]

    unet = pipe.unet
    unet_parts = {"conv_in": unet.conv_in, "time_embedding": unet.time_embedding, "add_embedding": unet.add_embedding}
    unet_parts.update({f"down{i}": b for i, b in enumerate(unet.down_blocks)})
    unet_parts["mid"] = unet.mid_block
    unet_parts.update({f"up{i}": b for i, b in enumerate(unet.up_blocks)})
    unet_parts.update({"conv_norm_out": unet.conv_norm_out, "conv_out": unet.conv_out})

    vae = pipe.vae
    vae_parts = {"post_quant_conv": vae.post_quant_conv, "conv_in": vae.decoder.conv_in, "mid": vae.decoder.mid_block}
    vae_parts.update({f"up{i}": b for i, b in enumerate(vae.decoder.up_blocks)})
    vae_parts.update({"conv_norm_out": vae.decoder.conv_norm_out, "conv_out": vae.decoder.conv_out})

    samplers = {"start": start}
    parts = {}
    for name in SAMPLERS:
        out, final = run_sampler(pipe, name, embeds, start, step_noise, unet_parts)
        for key in list(out):
            if key.startswith("unet."):
                parts[key] = out.pop(key)
        samplers.update(out)
        record = {}
        scaled, image = decode(pipe, final, vae_parts if name == "euler" else {}, record)
        samplers[f"{name}.final_latent"] = final
        samplers[f"{name}.image"] = image[0]
        rgb = ((image[0] / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
        samplers[f"{name}.rgb"] = rgb
        if name == "euler":
            close(record["vae.conv_out"], image, 0.0, "the VAE hook of conv_out")
            parts["vae.in"] = scaled
            parts.update(record)
            pipe_image = pipe.image_processor.postprocess(image, output_type="pt")[0]
            close(pipe_image, (image[0] / 2 + 0.5).clamp(0, 1), 1e-6, "the pipeline's postprocess")
        print(f"{name}: done after {time.time() - started:.0f} s", flush=True)

    # The hooks of step 0 of Euler: the input of the UNet and its whole output.
    parts["unet.in"] = samplers["euler.unet_in"][0]
    parts["unet.timestep"] = samplers["euler.timesteps"][0:1]
    parts["unet.encoder_hidden_states"] = torch.stack([embeds["neg"], embeds["pos"]])
    close(parts["unet.conv_out"], samplers["euler.unet_out"][0], 0.0, "the UNet hook of conv_out")

    save_file({k: v.contiguous() for k, v in samplers.items()}, os.path.join(args.out, "samplers.safetensors"),
              metadata=meta)
    save_file({k: v.contiguous() for k, v in parts.items()}, os.path.join(args.out, "parts.safetensors"), metadata=meta)
    for name in SAMPLERS:
        np.save(os.path.join(args.out, f"{name}.rgb.npy"), samplers[f"{name}.rgb"].numpy())
    print(f"wrote {args.out} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
