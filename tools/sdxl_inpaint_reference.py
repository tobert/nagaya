#!/usr/bin/env python3
"""f32 CPU reference for the image inputs of SDXL: img2img and inpainting (spec 8.3, 11.9b).

Run it through tools/sdxl_inpaint_golden.sh. The direct command is:

  HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 <reference venv>/bin/python -I \
      tools/sdxl_inpaint_reference.py --model "$SDXL_MODEL_DIR" --inpaint-model "$SDXL_INPAINT_MODEL_DIR" \
      --out target/sdxl-golden

The script uses diffusers in the reference venv, on the CPU only. It loads the fp16 weights as float32. It takes the
prompt, the size, the step count, the CFG scale, the seed and the samplers of tools/sdxl_reference.py. The tensor
names, shapes and meanings are in tools/sdxl_inpaint_golden.sh.

The engine makes its noise with its own random stream (spec 11.9b), so the script records each draw of diffusers with
its purpose. Each draw comes from torch.Generator().manual_seed(1234) in the order of diffusers. Thus a second run of
plain diffusers with that generator must give the same latent, and the script checks that it does.

The script stops with an AssertionError when a self check fails. The checks:

  - the gray mask, the blur, the binary mask, the latent mask and the masked image of the script give the tensors of
    diffusers, bit for bit. The blur is a port of Pillow's box blur, so the engine can port it too;
  - the input tensor is 2 * v / 255 - 1 in f32, and the image latent is
    0.13025 * (mean + exp(0.5 * clamp(logvar, -30, 20)) * z), bit for bit;
  - n = int(steps * strength) and the begin index steps - n give the timesteps that diffusers runs;
  - add_noise of the spec gives the start latent and the `known` latent of each step, bit for bit;
  - the blend (1 - m) * known + m * latent gives the latent after each step, bit for bit;
  - the 9-channel UNet input is the scaled latent, the latent mask and the masked image latent;
  - the run with the hooks gives the final latent of plain diffusers, bit for bit;
  - the hooks of the VAE encoder give the output of the whole encoder.
"""

import argparse
import importlib.util
import io
import json
import os
import resource
import time

import numpy as np
import PIL
import torch
from PIL import Image
from safetensors.torch import save_file

import diffusers
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.models.autoencoders import vae as vae_module
from diffusers.pipelines.stable_diffusion_xl import pipeline_stable_diffusion_xl_img2img as img2img_module
from diffusers.pipelines.stable_diffusion_xl import pipeline_stable_diffusion_xl_inpaint as inpaint_module
from diffusers.schedulers import (
    scheduling_dpmsolver_multistep,
    scheduling_euler_ancestral_discrete,
    scheduling_euler_discrete,
)


def load_base_reference():
    """Load tools/sdxl_reference.py by its path, because `python -I` does not add the folder of the script."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sdxl_reference.py")
    spec = importlib.util.spec_from_file_location("sdxl_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = load_base_reference()
SAMPLERS = REF.SAMPLERS
PROMPT, NEGATIVE = REF.PROMPT, REF.NEGATIVE
WIDTH, HEIGHT, STEPS, CFG, SEED = REF.WIDTH, REF.HEIGHT, REF.STEPS, REF.CFG, REF.SEED
check, close = REF.check, REF.close

MASK_BLUR = 4
# The blur radii of the extra test vectors of the blur port.
BLUR_RADII = (0, 1, 2, 3, 4, 8, 64)

# The runs of each case: a sampler and a strength. The tag of a strength is its value in percent.
ALL = list(SAMPLERS)
CASES = {
    "img2img": [(s, 0.5) for s in ALL] + [("euler", 1.0)],
    "inpaint": [(s, 0.99) for s in ALL] + [("euler", 1.0), ("dpmpp_2m_karras", 1.0)],
    "inpaint9": [(s, 0.99) for s in ALL] + [("euler", 1.0), ("dpmpp_2m_karras", 1.0)],
}

# The schedule table: (steps, strength). The last four lose a step to the f64 product (spec 11.9b).
SCHEDULE_CASES = [
    (4, 0.5), (4, 0.99), (4, 1.0), (30, 0.75), (30, 0.99), (1, 0.99), (2, 0.5), (90, 0.7), (50, 0.58),
    (100, 0.29), (150, 0.82),
]

SCHEDULER_MODULES = (scheduling_euler_discrete, scheduling_euler_ancestral_discrete, scheduling_dpmsolver_multistep)
PIPELINE_MODULES = (img2img_module, inpaint_module)


def tag(strength):
    return f"s{round(strength * 100)}"


def synthetic_image():
    """The input image, RGBA, HEIGHT rows by WIDTH columns. R is a ramp along x, G a ramp along y, B a checker of 32
    pixels, and A varies so that a test sees that the alpha is ignored."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    r = x * 255 // (WIDTH - 1)
    g = y * 255 // (HEIGHT - 1)
    b = np.where(((x // 32) + (y // 32)) % 2 == 0, 255, 64)
    a = 128 + (x + y) % 128
    return np.stack([r, g, b, a], axis=-1).astype(np.uint8)


def synthetic_mask():
    """The mask, RGBA. White repaints an ellipse at the center. A band of rows 32 to 63 has colors whose gray values
    cross 128, so the gray formula and the threshold both matter. The alpha is 255 on the left half, 100 on the
    right."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    inside = ((x - 128) / 72.0) ** 2 + ((y - 160) / 96.0) ** 2 <= 1.0
    r = np.where(inside, 255, 0)
    g = np.where(inside, 255, 0)
    b = np.where(inside, 255, 0)
    band = (y >= 32) & (y < 64)
    r = np.where(band, (x * 7) % 256, r)
    g = np.where(band, (y * 13 + x) % 256, g)
    b = np.where(band, (x * x) % 256, b)
    a = np.where(x < 128, 255, 100)
    return np.stack([r, g, b, a], axis=-1).astype(np.uint8)


def png_bytes(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def gray(rgb):
    """Pillow's convert("L"): (19595 R + 38470 G + 7471 B + 32768) >> 16 (spec 11.9b, "The mask")."""
    rgb = rgb.astype(np.int64)
    return ((19595 * rgb[..., 0] + 38470 * rgb[..., 1] + 7471 * rgb[..., 2] + 32768) >> 16).astype(np.uint8)


# A port of Pillow's GaussianBlur (libImaging/BoxBlur.c): three box passes along x, then three along y. The box radius
# comes from the Gaussian radius in f32, and the sums use 24 bits of fraction. The script checks it against Pillow.
F32 = np.float32


def box_radius(radius, passes=3):
    sigma2 = F32(F32(radius) * F32(radius) / F32(passes))
    length = F32(np.sqrt(12.0 * float(sigma2) + 1.0))
    whole = F32(np.floor((float(length) - 1.0) / 2.0))
    part = F32((F32(2) * whole + F32(1)) * (whole * (whole + F32(1)) - F32(3) * sigma2))
    part = F32(part / F32(F32(6) * (sigma2 - (whole + F32(1)) * (whole + F32(1)))))
    return F32(whole + part)


def box_pass(rows, fradius):
    """One box pass along each row of a [R, X] u8 array, with the edge pixels repeated."""
    n = rows.shape[1]
    radius = int(fradius)
    ww = int(F32(F32(1 << 24) / F32(fradius * F32(2) + F32(1))))
    fw = ((1 << 24) - (radius * 2 + 1) * ww) // 2
    last = n - 1
    edge_a = min(radius + 1, n)
    edge_b = max(n - radius - 1, 0)
    src = rows.astype(np.int64)
    out = np.empty_like(rows)
    acc = src[:, 0] * (radius + 1)
    for x in range(edge_a - 1):
        acc = acc + src[:, x]
    acc = acc + src[:, last] * (radius - edge_a + 1)

    def emit(x, sub, add, left, right):
        nonlocal acc
        acc = acc + src[:, add] - src[:, sub]
        bulk = acc * ww + (src[:, left] + src[:, right]) * fw
        out[:, x] = ((bulk + (1 << 23)) >> 24).astype(np.uint8)

    if edge_a <= edge_b:
        for x in range(0, edge_a):
            emit(x, 0, x + radius, 0, x + radius + 1)
        for x in range(edge_a, edge_b):
            emit(x, x - radius - 1, x + radius, x - radius - 1, x + radius + 1)
        for x in range(edge_b, last + 1):
            emit(x, x - radius - 1, last, x - radius - 1, last)
    else:
        for x in range(0, edge_b):
            emit(x, 0, x + radius, 0, x + radius + 1)
        for x in range(edge_b, edge_a):
            emit(x, 0, last, 0, last)
        for x in range(edge_a, last + 1):
            emit(x, x - radius - 1, last, x - radius - 1, last)
    return out


def blur(gray_mask, radius, passes=3):
    """Pillow's GaussianBlur(radius) of a gray u8 image [H, W]."""
    if radius == 0:
        return gray_mask.copy()
    fradius = box_radius(radius, passes)
    img = gray_mask.copy()
    for _ in range(passes):
        img = box_pass(img, fradius)
    cols = np.ascontiguousarray(img.T)
    for _ in range(passes):
        cols = box_pass(cols, fradius)
    return np.ascontiguousarray(cols.T)


def paste(blurred, new, old):
    """The paste of spec 11.9b: (a * new + (255 - a) * old + 127) / 255 in integers, for each channel."""
    a = blurred.astype(np.int64)[..., None]
    out = (a * new.astype(np.int64) + (255 - a) * old.astype(np.int64) + 127) // 255
    return out.astype(np.uint8)


def to_tensor(rgb):
    """2 * v / 255 - 1 in f32, [1, 3, H, W], as diffusers' preprocess makes it."""
    x = rgb.astype(np.float32) / np.float32(255.0)
    return torch.from_numpy(2.0 * x - 1.0).permute(2, 0, 1)[None].contiguous()


def add_noise(name, sched, x, noise, index):
    """add_noise of spec 11.9b at the sigma of schedule index `index`."""
    sigma = sched.sigmas.to(dtype=x.dtype)[index]
    if name.startswith("euler"):
        return x + noise * sigma
    alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
    sigma_t = sigma * alpha_t
    return alpha_t * x + sigma_t * noise


def image_latent(mean, logvar, z, factor):
    """The draw of spec 11.9b, "The image latent", then the scaling factor."""
    std = torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0))
    return factor * (mean + std * z)


class Draws:
    """Stand-ins for `randn_tensor` in each module that draws. Each one gives the next value of one torch.Generator
    and records the purpose: `encode` or `encode_masked` in the VAE, `start` in the pipeline, `step` in a scheduler."""

    def __init__(self, seed):
        self.gen = torch.Generator().manual_seed(seed)
        self.log = []
        self.encode_purpose = None
        self.saved = []

    def stand_in(self, purpose):
        def randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
            name = purpose() if callable(purpose) else purpose
            check(name is not None, "a VAE draw before the VAE encode")
            value = torch.randn(tuple(shape), generator=self.gen, dtype=torch.float32)
            self.log.append((name, value.clone()))
            return value.to(dtype=dtype or value.dtype)

        return randn_tensor

    def install(self):
        targets = [(vae_module, lambda: self.encode_purpose)]
        targets += [(m, "start") for m in PIPELINE_MODULES]
        targets += [(m, "step") for m in SCHEDULER_MODULES]
        for module, purpose in targets:
            self.saved.append((module, module.randn_tensor))
            module.randn_tensor = self.stand_in(purpose)

    def remove(self):
        for module, original in self.saved:
            module.randn_tensor = original
        self.saved = []

    def of(self, purpose):
        return [v for p, v in self.log if p == purpose]


def wrap(obj, name, record):
    """Replace a method on one instance with a wrapper that records its arguments and its result."""
    original = getattr(obj, name)

    def wrapper(*args, **kwargs):
        out = original(*args, **kwargs)
        record.append((args, kwargs, out))
        return out

    setattr(obj, name, wrapper)
    return lambda: delattr(obj, name)


def hooked(modules, record, prefix):
    """Record the output of each named module once. A tuple output keeps its first item."""
    handles = []
    for name, module in modules.items():
        def hook(_m, _inp, out, name=name):
            if (prefix + name) not in record:
                value = out[0] if isinstance(out, tuple) else out
                record[prefix + name] = value.detach().clone()
        handles.append(module.register_forward_hook(hook))
    return handles


def call_pipe(pipe, case, inputs, embeds, strength, generator, on_step=None):
    kwargs = dict(
        prompt_embeds=embeds["pos"][None],
        negative_prompt_embeds=embeds["neg"][None],
        pooled_prompt_embeds=embeds["pos_pooled"][None],
        negative_pooled_prompt_embeds=embeds["neg_pooled"][None],
        image=inputs["init_pil"],
        strength=strength,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        generator=generator,
        output_type="latent",
    )
    if on_step is not None:
        kwargs["callback_on_step_end"] = on_step
    if case != "img2img":
        kwargs.update(mask_image=inputs["mask_pil"], width=WIDTH, height=HEIGHT)
    return pipe(**kwargs).images


@torch.no_grad()
def run(pipe, case, name, strength, inputs, embeds):
    """Run one sampler at one strength with the hooks, then with plain diffusers. Give its tensors by name."""
    cls, extra = SAMPLERS[name]
    pipe.scheduler = cls.from_config(pipe.scheduler_config, **extra)
    sched = pipe.scheduler
    channels = pipe.unet.config.in_channels
    factor = pipe.vae.config.scaling_factor

    draws = Draws(SEED)
    encodes, prepared, mask_prepared, stepped, noised = [], [], [], [], []
    unet_in, unet_out, after = [], [], []

    original_encode = pipe.vae.encode

    def encode(x, *args, **kwargs):
        if torch.equal(x, inputs["init"]):
            draws.encode_purpose = "encode"
        else:
            check(torch.equal(x, inputs["masked"]), f"{case} {name}: the VAE encodes the masked image")
            draws.encode_purpose = "encode_masked"
        out = original_encode(x, *args, **kwargs)
        mean, logvar = torch.chunk(out.latent_dist.parameters, 2, dim=1)
        encodes.append((draws.encode_purpose, mean.clone(), logvar.clone()))
        return out

    def unet_hook(_m, args, kwargs, out):
        unet_in.append((args[0] if args else kwargs["sample"]).detach().clone())
        unet_out.append((out[0] if isinstance(out, tuple) else out.sample).detach().clone())

    def on_step(_pipe, _i, _t, kw):
        after.append(kw["latents"].detach().clone())
        return kw

    pipe.vae.encode = encode
    undo = [lambda: delattr(pipe.vae, "encode")]
    undo.append(wrap(pipe, "prepare_latents", prepared))
    if case != "img2img":
        undo.append(wrap(pipe, "prepare_mask_latents", mask_prepared))
    undo.append(wrap(sched, "step", stepped))
    undo.append(wrap(sched, "add_noise", noised))
    handle = pipe.unet.register_forward_hook(unet_hook, with_kwargs=True)
    draws.install()
    try:
        final = call_pipe(pipe, case, inputs, embeds, strength, torch.Generator().manual_seed(SEED), on_step)
    finally:
        draws.remove()
        handle.remove()
        for u in undo:
            u()

    what = f"{case} {name} {tag(strength)}"
    n = int(STEPS * strength)
    begin = STEPS - n
    check(len(unet_out) == n, f"{what}: {len(unet_out)} UNet passes, not n = {n}")
    check(sched.begin_index == begin, f"{what}: begin index {sched.begin_index}, not {begin}")

    # The order of the draws in diffusers.
    image_encoded = case == "img2img" or case == "inpaint" or strength < 1.0
    order = [p for p, _ in draws.log]
    want = (["encode"] if image_encoded else []) + ["start"] + (["encode_masked"] if case != "img2img" else [])
    check(order[: len(want)] == want, f"{what}: draws {order[:len(want)]}, not {want}")
    check(all(p == "step" for p in order[len(want):]), f"{what}: draws after the start are steps")
    step_noise = draws.of("step")
    check(len(step_noise) in (0, n - 1, n), f"{what}: {len(step_noise)} step draws in {n} steps")
    noise = draws.of("start")[0]

    out = {
        "n": torch.tensor([n], dtype=torch.int64),
        "begin": torch.tensor([begin], dtype=torch.int64),
        "strength": torch.tensor([strength], dtype=torch.float64),
        "sigmas": sched.sigmas.float(),
        "timesteps": sched.timesteps.float(),
        "run_sigmas": sched.sigmas.float()[begin:],
        "run_timesteps": sched.timesteps.float()[begin:],
        "noise.start": noise,
        "noise.step": torch.stack(step_noise) if step_noise else torch.zeros(0, *noise.shape),
    }

    # The image latents, from the moments of the encoder and the recorded draw.
    latents_of = {}
    for purpose, mean, logvar in encodes:
        z = draws.of(purpose)[0]
        out[f"noise.{purpose}"] = z
        out[f"{purpose}.mean"] = mean
        out[f"{purpose}.logvar"] = logvar
        latents_of[purpose] = image_latent(mean, logvar, z, factor)
    check(sorted(latents_of) == sorted(set(want) - {"start"}), f"{what}: the encodes {sorted(latents_of)}")
    if "encode" in latents_of:
        out["image_latent"] = latents_of["encode"]
    if "encode_masked" in latents_of:
        out["masked_latent"] = latents_of["encode_masked"]

    # The start latent.
    result = prepared[0][2]
    start = result if case == "img2img" else result[0]
    if case != "img2img":
        close(result[1], noise, 0.0, f"{what}: the noise of the blend is the start draw")
    if case == "inpaint":
        close(result[2], out["image_latent"], 0.0, f"{what}: the image latent of diffusers")
    if strength == 1.0 and case != "img2img":
        mine = noise * sched.init_noise_sigma
    else:
        mine = add_noise(name, sched, out["image_latent"], noise, begin)
        close(noised[0][2], start, 0.0, f"{what}: add_noise made the start latent")
    close(mine, start, 0.0, f"{what}: the start latent of spec 11.9b")

    # The latent mask and the masked latent, as diffusers gives them to the UNet.
    if case != "img2img":
        mask_cfg, masked_cfg = mask_prepared[0][2]
        latent_mask = inputs["latent_mask"][None, None]
        close(mask_cfg, latent_mask.expand(2, 1, -1, -1), 0.0, f"{what}: the latent mask")
        close(masked_cfg, out["masked_latent"].expand(2, -1, -1, -1), 0.0, f"{what}: the masked latent")

    # The steps.
    latents = [start] + after
    sched_out = torch.stack([s[2][0] for s in stepped])
    guided = torch.stack([u[0:1] + CFG * (u[1:2] - u[0:1]) for u in unet_out])
    out.update({
        "latents": torch.stack(latents),
        "unet_in": torch.stack(unet_in),
        "unet_out": torch.stack(unet_out),
        "guided": guided,
        "stepped": sched_out,
    })
    check(unet_in[0].shape[1] == channels, f"{what}: the UNet input has {channels} channels")
    for i in range(n):
        close(stepped[i][0][0], guided[i], 0.0, f"{what}: step {i} takes the guided output")
        close(stepped[i][0][2], latents[i], 0.0, f"{what}: step {i} starts from the latent before it")
    if channels == 9:
        for i in range(n):
            close(unet_in[i][:, 4:5], inputs["latent_mask"][None, None].expand(2, 1, -1, -1), 0.0,
                  f"{what}: channel 4 of step {i} is the latent mask")
            close(unet_in[i][:, 5:9], out["masked_latent"].expand(2, -1, -1, -1), 0.0,
                  f"{what}: channels 5 to 8 of step {i} are the masked latent")
            close(unet_in[i][:, 0:4], sched_scaled(name, sched, latents[i], begin + i), 0.0,
                  f"{what}: channels 0 to 3 of step {i} are the scaled latent")
    if case == "inpaint":
        known = []
        m = inputs["latent_mask"][None, None]
        for i in range(n):
            if i < n - 1:
                k = add_noise(name, sched, out["image_latent"], noise, begin + i + 1)
                close(noised[1 + i - (1 if strength == 1.0 else 0)][2], k, 0.0, f"{what}: known of step {i}")
            else:
                k = out["image_latent"]
            known.append(k)
            close((1 - m) * k + m * sched_out[i], after[i], 0.0, f"{what}: the blend of step {i}")
        out["known"] = torch.stack(known)
        calls = len(noised)
        check(calls == (n - 1) + (0 if strength == 1.0 else 1), f"{what}: {calls} calls of add_noise")
    else:
        for i in range(n):
            close(sched_out[i], after[i], 0.0, f"{what}: no blend after step {i}")
    out["final_latent"] = final

    # Plain diffusers with the same generator gives the same latent.
    pipe.scheduler = cls.from_config(pipe.scheduler_config, **extra)
    plain = call_pipe(pipe, case, inputs, embeds, strength, torch.Generator().manual_seed(SEED))
    close(plain, final, 0.0, f"{what}: plain diffusers")

    # The decode, before any paste.
    image = pipe.vae.decode(final / factor, return_dict=False)[0]
    rgb = ((image[0] / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
    out["image"] = image[0]
    out["rgb"] = rgb
    if case != "img2img":
        out["paste"] = torch.from_numpy(paste(inputs["blurred_np"], rgb.numpy(), inputs["rgb_np"]))
    return {f"{name}.{tag(strength)}.{k}": v for k, v in out.items()}


def sched_scaled(name, sched, latent, index):
    """scale_model_input of the sampler at schedule index `index`, for both halves of CFG."""
    doubled = torch.cat([latent] * 2)
    if name.startswith("euler"):
        sigma = sched.sigmas[index]
        return doubled / ((sigma**2 + 1) ** 0.5)
    return doubled


def prepare_inputs(pipe):
    """The input PNGs, their tensors and the processed mask. Check each against diffusers."""
    init_rgba = synthetic_image()
    mask_rgba = synthetic_mask()
    init_png, mask_png = png_bytes(init_rgba), png_bytes(mask_rgba)
    init_dec = Image.open(io.BytesIO(init_png))
    mask_dec = Image.open(io.BytesIO(mask_png))
    check(init_dec.mode == "RGBA" and mask_dec.mode == "RGBA", "the PNGs decode as RGBA")
    check(np.array_equal(np.array(init_dec), init_rgba), "the image PNG gives its pixels")
    check(np.array_equal(np.array(mask_dec), mask_rgba), "the mask PNG gives its pixels")

    init_pil = init_dec.convert("RGB")
    rgb = np.array(init_pil)
    check(np.array_equal(rgb, init_rgba[..., :3]), "convert(\"RGB\") drops the alpha")
    gray_pil = mask_dec.convert("L")
    gray_np = np.array(gray_pil)
    check(np.array_equal(gray_np, gray(mask_rgba[..., :3])), "the gray formula of spec 11.9b")

    for radius in BLUR_RADII:
        ref = np.array(pipe.mask_processor.blur(gray_pil, blur_factor=radius))
        check(np.array_equal(blur(gray_np, radius), ref), f"the blur port at radius {radius}")
    mask_pil = pipe.mask_processor.blur(gray_pil, blur_factor=MASK_BLUR)
    blurred = np.array(mask_pil)

    init = pipe.image_processor.preprocess(init_pil, height=HEIGHT, width=WIDTH).to(torch.float32)
    close(init, to_tensor(rgb), 0.0, "the image tensor 2 * v / 255 - 1")
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    close(pipe.image_processor.preprocess(Image.fromarray(ramp, "RGB"), height=16, width=16), to_tensor(ramp), 0.0,
          "the image tensor of each byte value")

    binary_ref = pipe.mask_processor.preprocess(mask_pil, height=HEIGHT, width=WIDTH)
    binary = torch.from_numpy((blurred >= 128).astype(np.float32))
    close(binary_ref[0, 0], binary, 0.0, "the binary mask repaints 128 and more")
    latent_mask = binary[0::8, 0::8].contiguous()
    interp = torch.nn.functional.interpolate(binary_ref, size=(HEIGHT // 8, WIDTH // 8))
    close(interp[0, 0], latent_mask, 0.0, "the latent mask takes pixel (8x, 8y)")
    masked = init * (binary_ref < 0.5)
    close(masked, init * (1 - binary)[None, None], 0.0, "the masked image is 0 where the mask repaints")

    tensors = {
        "init.png": torch.frombuffer(bytearray(init_png), dtype=torch.uint8),
        "mask.png": torch.frombuffer(bytearray(mask_png), dtype=torch.uint8),
        "init.rgba": torch.from_numpy(init_rgba),
        "init.rgb": torch.from_numpy(rgb.copy()),
        "init.tensor": init[0],
        "mask.rgba": torch.from_numpy(mask_rgba),
        "mask.gray": torch.from_numpy(gray_np.copy()),
        "mask.blurred": torch.from_numpy(blurred.copy()),
        "mask.binary": binary,
        "mask.latent": latent_mask,
        "masked.tensor": masked[0],
    }
    for radius in BLUR_RADII:
        tensors[f"blur.r{radius}"] = torch.from_numpy(blur(gray_np, radius))
    inputs = {
        "init_pil": init_pil, "mask_pil": mask_pil, "init": init, "masked": masked, "latent_mask": latent_mask,
        "blurred_np": blurred, "rgb_np": rgb,
    }
    return inputs, tensors, init_png, mask_png


def schedule_table(pipe):
    """n and the begin index of each (steps, strength) of SCHEDULE_CASES, from diffusers' get_timesteps."""
    rows, ns, begins = [], [], []
    for steps, strength in SCHEDULE_CASES:
        for name, (cls, extra) in SAMPLERS.items():
            pipe.scheduler = cls.from_config(pipe.scheduler_config, **extra)
            pipe.scheduler.set_timesteps(steps)
            timesteps, n = pipe.get_timesteps(steps, strength, "cpu")
            check(n == int(steps * strength), f"{name} {steps} {strength}: n {n}")
            check(len(timesteps) == n, f"{name} {steps} {strength}: {len(timesteps)} timesteps")
            if n > 0:
                check(pipe.scheduler.begin_index == steps - n, f"{name} {steps} {strength}: the begin index")
        rows.append([float(steps), strength])
        ns.append(n)
        begins.append(steps - n)
    check(ns[SCHEDULE_CASES.index((90, 0.7))] == 62, "steps 90 at strength 0.7 run 62 steps")
    return {
        "schedule.cases": torch.tensor(rows, dtype=torch.float64),
        "schedule.n": torch.tensor(ns, dtype=torch.int64),
        "schedule.begin": torch.tensor(begins, dtype=torch.int64),
    }


@torch.no_grad()
def encoder_parts(pipe, init):
    """The stages of the VAE encoder of SDXL base for the image, for kernel bring-up."""
    vae = pipe.vae
    enc = vae.encoder
    parts = {"conv_in": enc.conv_in}
    parts.update({f"down{i}": b for i, b in enumerate(enc.down_blocks)})
    parts.update({"mid": enc.mid_block, "conv_norm_out": enc.conv_norm_out, "conv_out": enc.conv_out,
                  "quant_conv": vae.quant_conv})
    record = {}
    handles = hooked(parts, record, "vae_enc.")
    try:
        moments = vae.encode(init).latent_dist.parameters
    finally:
        for h in handles:
            h.remove()
    close(record["vae_enc.quant_conv"], moments, 0.0, "the VAE encoder hook of quant_conv")
    record["vae_enc.in"] = init
    return record


def embeddings(pipe):
    """The A1111 encoding with clip_skip 2 (diffusers' 1), as in tools/sdxl_reference.py."""
    _, _, n_hidden, n_pooled = REF.encode(pipe, NEGATIVE, 1)
    _, _, p_hidden, p_pooled = REF.encode(pipe, PROMPT, 1)
    ref = pipe.encode_prompt(PROMPT, negative_prompt=NEGATIVE, do_classifier_free_guidance=True, device="cpu")
    embeds = {"neg": n_hidden, "pos": p_hidden, "neg_pooled": n_pooled, "pos_pooled": p_pooled}
    close(ref[0][0], embeds["pos"], 1e-5, "the prompt encoding of diffusers")
    close(ref[1][0], embeds["neg"], 1e-5, "the negative encoding of diffusers")
    close(ref[2][0], embeds["pos_pooled"], 1e-5, "the pooled prompt of diffusers")
    close(ref[3][0], embeds["neg_pooled"], 1e-5, "the pooled negative of diffusers")
    return embeds


def save(path, tensors, meta):
    save_file({k: v.contiguous().clone() for k, v in tensors.items()}, path, metadata=meta)


def run_case(pipe, case, inputs, embeds, meta, out_dir, started):
    tensors = {}
    for name, strength in CASES[case]:
        tensors.update(run(pipe, case, name, strength, inputs, embeds))
        print(f"{case} {name} {tag(strength)}: done after {time.time() - started:.0f} s", flush=True)
    # The moments of the encoder do not depend on the sampler.
    for key in [k for k in tensors if k.endswith((".mean", ".logvar"))]:
        first = f"{CASES[case][0][0]}.{tag(CASES[case][0][1])}." + key.split(".", 2)[2]
        if first in tensors:
            close(tensors[key], tensors[first], 0.0, f"{case}: {key} is the same for each run")
    save(os.path.join(out_dir, f"{case}.safetensors"), tensors, meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--inpaint-model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)
    started = time.time()

    base = StableDiffusionXLPipeline.from_pretrained(args.model, variant="fp16", torch_dtype=torch.float32)
    components = {k: v for k, v in base.components.items()}
    del base
    img2img = StableDiffusionXLImg2ImgPipeline(**components)
    inpaint = StableDiffusionXLInpaintPipeline(**components)
    for each in (img2img, inpaint):
        each.set_progress_bar_config(disable=True)
        each.scheduler_config = dict(each.scheduler.config)
    del each
    check(not inpaint.config.requires_aesthetics_score, "the time ids of SDXL base")

    meta = {
        "prompt": PROMPT,
        "negative": NEGATIVE,
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "steps": str(STEPS),
        "cfg": str(CFG),
        "seed": str(SEED),
        "mask_blur": str(MASK_BLUR),
        "runs": json.dumps({case: [[s, strength] for s, strength in runs] for case, runs in CASES.items()}),
        "diffusers": diffusers.__version__,
        "torch": torch.__version__,
        "pillow": PIL.__version__,
        "scheduler_config": json.dumps(inpaint.scheduler_config, default=str),
        "vae_scaling_factor": str(inpaint.vae.config.scaling_factor),
    }

    inputs, tensors, init_png, mask_png = prepare_inputs(inpaint)
    embeds = embeddings(inpaint)
    tensors["prompt_embeds"] = torch.stack([embeds["neg"], embeds["pos"]])
    tensors["pooled"] = torch.stack([embeds["neg_pooled"], embeds["pos_pooled"]])
    tensors["add_time_ids"] = torch.tensor([HEIGHT, WIDTH, 0, 0, HEIGHT, WIDTH], dtype=torch.float32)
    tensors.update(schedule_table(inpaint))
    tensors.update(encoder_parts(inpaint, inputs["init"]))
    save(os.path.join(args.out, "image_inputs.safetensors"), tensors, meta)
    for name, data in (("init.png", init_png), ("mask.png", mask_png)):
        with open(os.path.join(args.out, name), "wb") as f:
            f.write(data)
    print(f"inputs: done after {time.time() - started:.0f} s", flush=True)

    run_case(img2img, "img2img", inputs, embeds, meta, args.out, started)
    run_case(inpaint, "inpaint", inputs, embeds, meta, args.out, started)

    # The 9-channel UNet, with every part from its own folder: the profile of spec 11.9b names that folder.
    del img2img, inpaint, components
    pipe9 = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.inpaint_model, variant="fp16", torch_dtype=torch.float32
    )
    pipe9.set_progress_bar_config(disable=True)
    pipe9.scheduler_config = dict(pipe9.scheduler.config)
    check(pipe9.unet.config.in_channels == 9, "the inpainting UNet has 9 input channels")
    check(tuple(pipe9.unet.conv_in.weight.shape) == (320, 9, 3, 3), "conv_in.weight is [320, 9, 3, 3]")
    check(not pipe9.config.requires_aesthetics_score, "the inpainting UNet takes the time ids of SDXL base")
    embeds9 = embeddings(pipe9)
    for key in embeds:
        close(embeds9[key], embeds[key], 0.0, f"the text encoders of the inpainting folder give {key}")
    meta9 = dict(meta)
    meta9["scheduler_config"] = json.dumps(pipe9.scheduler_config, default=str)
    meta9["vae_scaling_factor"] = str(pipe9.vae.config.scaling_factor)
    meta9["vae"] = json.dumps(dict(pipe9.vae.config), default=str)
    run_case(pipe9, "inpaint9", inputs, embeds, meta9, args.out, started)

    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(f"wrote {args.out} in {time.time() - started:.0f} s; CPU {usage.ru_utime + usage.ru_stime:.0f} s; "
          f"peak RSS {usage.ru_maxrss / 1048576:.1f} GiB")


if __name__ == "__main__":
    main()
