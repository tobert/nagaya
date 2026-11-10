# SPDX-License-Identifier: MIT
"""f32 CPU reference for LFM3-Omni-2B and LFM3-1.7B: the LFM3 oracle (spec 8.3, 11.11).

Run it through tools/lfm3_golden.sh, which names each file and tensor. The direct command is:

  PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 nice -n 19 ionice -c3 ~/.venvs/nagaya-lfm3-ref/bin/python -I \
      tools/lfm3_reference.py --omni "$LFM3_OMNI_MODEL_DIR" --text "$LFM3_MODEL_DIR" --out target/lfm3-golden

The reference is transformers with the checkpoint's own code (`trust_remote_code`), in float32 with eager
attention, on the CPU. The tool reads the two model folders and never writes into them. It writes only into --out.

The stages, each with its own gate:

  text     the backbone logits of both models (5 prompts each), and the residual after each layer of one prompt
  vision   two synthetic PNG images: the pixel values, the vision tower rows and the projector rows
  audio    one synthetic WAV: the 128 mel bins, the encoder rows and the projector rows
  speech   a TTS run and an interleaved run, the depth decoder with forced codes, and the codec decode
  control  LFM3-1.7B only: a control vector from two small prompt sets, and the logits of each operation (11.12).
           It is not in the default stages; name it with --stages control.
  project  LFM3-1.7B only: the vector of the control stage through the final norm's weight and the head (lab.md).
           It reads the file of the control stage. It is not in the default stages; name it with --stages project.
  latent   LFM3-1.7B only: 16 latent steps after a prompt, then 8 greedy tokens (lab.md, "`latent` and `merge`").
           It is not in the default stages; name it with --stages latent.

The tool stops with an AssertionError when a self check fails. The checks:

  - the ids of each oracle prompt start with <|startoftext|>, and the five prompts fit in 4096 tokens;
  - the hooks of each layer give the hidden state of the model, and the last layer plus the final norm gives the
    logits;
  - the count of image placeholders is grid_t * grid_h * grid_w / 4, and the count of audio placeholders is
    ceil(mel frames / 8);
  - the rows of the tower and the projectors are the rows that the model scatters into its input;
  - the teacher-forced pass of a run gives the text tokens that greedy decoding chose;
  - the summed codebook embedding is the sum of the 12 rows of the shared table;
  - the codec decode of all frames at once gives 1920 samples for each frame.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
TOP = 64
MAX_TOKENS = 4096
SEED = 1234
FRAME_SAMPLES = 1920  # 24 kHz / 12.5 Hz

TTS_SYSTEM = "Perform TTS in American English using a female voice."
TTS_TEXT = "The nagaya is awake, and every engine is ready."
MIXED_SYSTEM = "Respond with interleaved text and audio."
MIXED_TEXT = "Say hello to the nagaya in one short sentence."


def log(message):
    print(f"[{time.time() - STARTED:7.1f}s] {message}", file=sys.stderr, flush=True)


STARTED = time.time()


def check(ok, what):
    """Stop with the name of the self check that failed."""
    if not ok:
        raise AssertionError(f"self check failed: {what}")


def max_abs(a, b):
    return (a.double() - b.double()).abs().max().item()


def close(a, b, tol, what):
    check(tuple(a.shape) == tuple(b.shape), f"{what}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
    d = max_abs(a, b)
    log(f"check {what}: max abs {d:.3g} (tolerance {tol:g})")
    check(d <= tol, f"{what}: max abs {d:.3g} > {tol:g}")
    return d


def save(tensors, path, meta):
    """Writes the tensors as one safetensors file, with `meta` as JSON in its metadata. Each tensor is a copy."""
    copies = {name: tensor.detach().clone().contiguous() for name, tensor in tensors.items()}
    save_file(copies, str(path), metadata={"meta": json.dumps(meta)})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prose():
    """The two prose prompts of the Qwen3.8 fixture, so that every family reads the same text."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("oracle_fixture", ROOT / "tools/oracle_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LONG, module.LONGER, module.AUTHOR


def top_entry(name, source, text, ids, logits):
    top = torch.topk(logits.float(), TOP)
    return {
        "name": name,
        "source": source,
        "text": text,
        "ids": [int(i) for i in ids],
        "top_ids": top.indices.tolist(),
        "top_logits": top.values.tolist(),
        "logsumexp": torch.logsumexp(logits.double(), -1).item(),
    }


def versions():
    import PIL
    import tokenizers
    import torchvision
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
    }


def source_record(folder):
    """The files of a model folder that the reference reads, with their SHA-256."""
    folder = Path(folder)
    names = sorted(p.name for p in folder.iterdir() if p.suffix in (".py", ".json", ".jinja"))
    record = {name: sha256(folder / name) for name in names}
    if (folder / "audio_tokenizer/config.json").exists():
        record["audio_tokenizer/config.json"] = sha256(folder / "audio_tokenizer/config.json")
    return {"folder": str(folder), "revision": folder.name, "files": record}


class LayerTap:
    """Keeps the output of each decoder layer and of the final norm, for one forward."""

    def __init__(self, backbone):
        self.outputs = {}
        self.handles = []
        self.handles.append(backbone.embed_tokens.register_forward_hook(self.keep("embed")))
        for index, layer in enumerate(backbone.layers):
            self.handles.append(layer.register_forward_hook(self.keep(f"layer{index}")))
        self.handles.append(backbone.embedding_norm.register_forward_hook(self.keep("final_norm")))

    def keep(self, name):
        def hook(_module, _args, output):
            self.outputs[name] = (output[0] if isinstance(output, tuple) else output).detach().clone()

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()


def text_prompts(tokenizer, chat_text):
    long, longer, author = prose()
    bos = tokenizer.convert_tokens_to_ids("<|startoftext|>")
    prompts = [
        ("plain", "The capital of France is", "fixed in this tool"),
        ("code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return", "fixed in this tool"),
        ("chat", chat_text, "the checkpoint's chat_template.jinja, rendered by transformers"),
        ("long", long, "prose " + author),
        ("longer", longer, "prose " + author),
    ]
    out = []
    for name, text, source in prompts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids or ids[0] != bos:
            ids = [bos] + ids
        check(len(ids) <= MAX_TOKENS, f"prompt {name} has {len(ids)} tokens")
        check(ids[0] == bos, f"prompt {name} starts with <|startoftext|>")
        out.append((name, text, source, ids))
    return out


def run_text(model, backbone, lm_head, tokenizer, label, folder, layer_count, out, facts):
    """The oracle of one backbone: 5 prompts, top-64 logits of the last position, and the layers of `plain`."""
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Name three colors."}], add_generation_prompt=True, tokenize=False
    )
    entries = []
    vocab = None
    plain_layers = None
    for name, text, source, ids in text_prompts(tokenizer, chat_text):
        log(f"{label} {name}: {len(ids)} tokens")
        tap = LayerTap(backbone) if name == "plain" else None
        with torch.no_grad():
            hidden = backbone(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state
            logits = lm_head(hidden[0, -1])
        vocab = int(logits.shape[0])
        entries.append(top_entry(name, source, text, ids, logits))
        if tap is not None:
            tap.remove()
            tensors = {"ids": torch.tensor(ids, dtype=torch.int64), "embed": tap.outputs["embed"][0]}
            for index in range(layer_count):
                tensors[f"layer{index}"] = tap.outputs[f"layer{index}"][0]
            tensors["final_norm"] = tap.outputs["final_norm"][0]
            tensors["logits_last"] = logits
            close(tensors["final_norm"], hidden[0], 0.0, f"{label} final norm tap is the model output")
            plain_layers = tensors
    document = {
        "format": 1,
        "model": label,
        "layers": layer_count,
        "vocab": vocab,
        "dtype": "float32",
        "attention": "eager",
        "versions": versions(),
        "source": source_record(folder),
        "tokenizer_sha256": sha256(Path(folder) / "tokenizer.json"),
        "seconds": round(time.time() - STARTED),
        "prompts": entries,
    }
    path = Path(out) / f"{label}.oracle.json"
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")
    meta = {"model": label, "prompt": "plain", "text": entries[0]["text"], "versions": versions()}
    save(plain_layers, Path(out) / f"{label}.layers.safetensors", meta)
    facts[label] = {
        "chat_text": chat_text,
        "prompt_lengths": {e["name"]: len(e["ids"]) for e in entries},
        "tokenizer_sha256": document["tokenizer_sha256"],
    }
    log(f"wrote {path}")


CONTROL_POSITIVE = [
    "Describe the sound of waves on a beach.",
    "What lives in a coral reef?",
    "Write one sentence about the tide coming in.",
    "Why is the sea salty?",
]
CONTROL_NEGATIVE = [
    "Describe the sound of wind over sand dunes.",
    "What lives in a dry desert?",
    "Write one sentence about the heat at noon in the desert.",
    "Why do deserts get cold at night?",
]
CONTROL_TEST = "Tell me about a place that you like."
# The strength of each operation in the fixture: a check of the arithmetic, not a tuned value.
CONTROL_STRENGTHS = {"add": 4.0, "ablate": 1.0, "clamp": 8.0}
CONTROL_DECODE = 8


def chat_ids(tokenizer, text):
    """The ids of one user message with the generation prompt and thinking off, as the family renders a context."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    check(len(ids) <= MAX_TOKENS, f"control prompt has {len(ids)} tokens")
    return rendered, ids


class Steer:
    """Changes the output of each decoder layer by one control item, as engine-abi "Control vectors" gives it.

    The arithmetic is f32, as the card's. The hook returns the changed output, so a later hook of the same layer
    (the LayerTap) reads the changed rows.
    """

    def __init__(self, backbone, vector, op, strength):
        self.handles = []
        for index, layer in enumerate(backbone.layers):
            self.handles.append(layer.register_forward_hook(self.change(vector[index], op, strength)))

    @staticmethod
    def change(v, op, alpha):
        unit = v / torch.linalg.vector_norm(v)

        def hook(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            if op == "add":
                changed = h + alpha * v
            elif op == "ablate":
                changed = h - alpha * (h @ unit).unsqueeze(-1) * unit
            elif op == "clamp":
                changed = h + (alpha - h @ unit).unsqueeze(-1) * unit
            else:
                raise AssertionError(f"unknown op {op}")
            return (changed,) + tuple(output[1:]) if isinstance(output, tuple) else changed

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()


def last_rows(backbone, ids, layer_count):
    """The residual after each layer at the last prompt position, [layers, width], and the final hidden state."""
    tap = LayerTap(backbone)
    with torch.no_grad():
        hidden = backbone(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state
    tap.remove()
    rows = torch.stack([tap.outputs[f"layer{index}"][0, -1] for index in range(layer_count)])
    return rows, hidden


def greedy(model, ids, steps):
    """Greedy decode of `steps` tokens with the cache, and the top-64 logits of each step.

    The loop calls the model, not `generate`: `generate` applies `repetition_penalty` 1.1 and the other processors of
    generation_config.json, so its tokens were not the argmax of the logits (found by the card gate, 2026-11-07).
    """
    steps_logits, tokens = [], []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=True)
        for _ in range(steps):
            logits = out.logits[0, -1].float()
            token = int(torch.argmax(logits))
            steps_logits.append(logits)
            tokens.append(token)
            out = model(input_ids=torch.tensor([[token]]), past_key_values=out.past_key_values, use_cache=True)
    logits = torch.stack(steps_logits)
    tokens = torch.tensor(tokens, dtype=torch.int64)
    check(bool((torch.argmax(logits, -1) == tokens).all()), "each greedy token is the argmax of its step")
    top = torch.topk(logits, TOP)
    return tokens, top.indices, top.values


def run_control(model, backbone, lm_head, tokenizer, layer_count, out, facts, label="lfm3"):
    """The control vector gate of one model (spec 11.12): the means of two prompt sets, and each operation. `label` is
    `lfm3` or `lfm3-omni`; Omni's vector acts on its text passes only (AGENTS.md milestone 25 step 9)."""
    sets = {}
    tensors = {}
    for name, texts in (("positive", CONTROL_POSITIVE), ("negative", CONTROL_NEGATIVE)):
        rows = []
        for i, text in enumerate(texts):
            _, ids = chat_ids(tokenizer, text)
            tensors[f"{name}{i}.ids"] = torch.tensor(ids, dtype=torch.int64)
            r, _ = last_rows(backbone, ids, layer_count)
            tensors[f"{name}{i}.rows"] = r
            rows.append(r)
        # The sum is f64, in the order of the request (spec 11.12).
        total = torch.zeros_like(rows[0], dtype=torch.float64)
        for r in rows:
            total += r.double()
        sets[name] = total / len(rows)
    vector = (sets["positive"] - sets["negative"]).float()
    tensors["vector"] = vector
    log(f"control vector: {layer_count} layers, norms {[round(float(n), 2) for n in vector.norm(dim=-1)]}")

    rendered, ids = chat_ids(tokenizer, CONTROL_TEST)
    tensors["test.ids"] = torch.tensor(ids, dtype=torch.int64)
    plain_rows, plain_hidden = last_rows(backbone, ids, layer_count)
    with torch.no_grad():
        tensors["plain.logits"] = lm_head(plain_hidden[0, -1])
    tensors["plain.rows"] = plain_rows
    tokens, top_ids, top_values = greedy(model, ids, CONTROL_DECODE)
    tensors["plain.decode_ids"], tensors["plain.decode_top_ids"], tensors["plain.decode_top"] = (
        tokens, top_ids, top_values)

    for op, alpha in CONTROL_STRENGTHS.items():
        steer = Steer(backbone, vector, op, alpha)
        rows, hidden = last_rows(backbone, ids, layer_count)
        with torch.no_grad():
            logits = lm_head(hidden[0, -1])
        tokens, top_ids, top_values = greedy(model, ids, CONTROL_DECODE)
        steer.remove()
        units = vector / vector.norm(dim=-1, keepdim=True)
        along = (rows * units).sum(-1)
        if op == "ablate":
            check(float(along.abs().max()) < 1e-3, f"ablate 1 leaves h.v {float(along.abs().max())}")
        if op == "clamp":
            check(float((along - alpha).abs().max()) < 1e-3, f"clamp leaves h.v off {alpha}")
        check(float((logits - tensors["plain.logits"]).abs().max()) > 1e-3, f"{op} changes the logits")
        tensors[f"{op}.logits"] = logits
        tensors[f"{op}.rows"] = rows
        tensors[f"{op}.decode_ids"], tensors[f"{op}.decode_top_ids"], tensors[f"{op}.decode_top"] = (
            tokens, top_ids, top_values)
        log(f"control {op} {alpha}: decode {tokenizer.decode(tokens)!r}")

    for op in ("add", "ablate"):
        steer = Steer(backbone, vector, op, 0.0)
        rows, _ = last_rows(backbone, ids, layer_count)
        steer.remove()
        close(rows, plain_rows, 0.0, f"{op} with strength 0 is the plain model")

    meta = {"model": label, "strengths": CONTROL_STRENGTHS, "positive": CONTROL_POSITIVE,
            "negative": CONTROL_NEGATIVE, "test": CONTROL_TEST, "rendered_test": rendered,
            "decode_steps": CONTROL_DECODE, "versions": versions()}
    save(tensors, Path(out) / f"{label}.control.safetensors", meta)
    facts[f"{label}_control"] = {"strengths": CONTROL_STRENGTHS, "test_tokens": len(ids)}
    log(f"wrote {Path(out) / f'{label}.control.safetensors'}")


# The layers of the project gate: the first, one in the middle and the last.
PROJECT_LAYERS = [0, 12, 23]
# Ids whose rank the stage logs: " seawater" and <|im_end|>.
PROJECT_WATCH = {"seawater": 62590, "im_end": 124900}


def rank(scores, token):
    """The 0-based rank of `token` in `scores`, from the top: the count of scores above its score."""
    return int((scores > scores[token]).sum())


def run_project(backbone, lm_head, tokenizer, out, facts):
    """The project gate of LFM3-1.7B (docs/lab.md, "`project`"): each layer of the control vector of the control
    stage, through the weight of the final norm and the tied head, with no division by its root mean square.

    The stage uses the model's own modules. It checks that the final norm multiplies by its weight (not by 1 plus its
    weight), and that the head is the embedding table.
    """
    source = Path(out) / "lfm3.control.safetensors"
    check(source.exists(), f"{source} exists; run --stages control first")
    from safetensors.torch import load_file

    vector = load_file(str(source))["vector"].float()
    norm = backbone.embedding_norm
    weight = norm.weight.detach().float()
    check(bool(torch.equal(lm_head.weight, backbone.embed_tokens.weight)), "the head is the embedding table")
    tensors = {"vector": vector, "layers": torch.tensor(PROJECT_LAYERS, dtype=torch.int64)}
    ranks = {}
    for layer in PROJECT_LAYERS:
        v = vector[layer]
        with torch.no_grad():
            normed = norm(v.unsqueeze(0))[0]
            u = weight * v
            inverse_rms = torch.rsqrt(v.pow(2).mean() + norm.variance_epsilon)
            close(normed, u * inverse_rms, 1e-5, f"layer {layer}: the final norm is its weight times the row")
            scores = lm_head(u)
        check(tuple(scores.shape) == (128000,), f"layer {layer}: a score for each row of the head")
        tensors[f"layer{layer}.scores"] = scores
        tensors[f"layer{layer}.up_ids"] = torch.topk(scores, TOP).indices
        tensors[f"layer{layer}.down_ids"] = torch.topk(-scores, TOP).indices
        ranks[str(layer)] = {name: rank(scores, token) for name, token in PROJECT_WATCH.items()}
        up = tensors[f"layer{layer}.up_ids"][:8].tolist()
        log(f"project layer {layer}: ranks {ranks[str(layer)]}, top {[tokenizer.decode([i]) for i in up]!r}")
    meta = {"model": "lfm3", "source": "lfm3.control.safetensors vector", "layers": PROJECT_LAYERS,
            "watch": PROJECT_WATCH, "ranks": ranks, "versions": versions()}
    save(tensors, Path(out) / "lfm3.project.safetensors", meta)
    facts["lfm3_project"] = {"layers": PROJECT_LAYERS, "ranks": ranks}
    log(f"wrote {Path(out) / 'lfm3.project.safetensors'}")


def synthetic_image(height, width, seed):
    """A deterministic RGB image: two gradients, a checkerboard of 8 pixels and a ring."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    r = 255.0 * x / max(width - 1, 1)
    g = 255.0 * y / max(height - 1, 1)
    checker = ((x // 8 + y // 8) % 2) * 255.0
    dist = np.hypot(x - width * 0.6, y - height * 0.4)
    ring = (np.abs(dist - min(height, width) * 0.3) < 4) * 255.0
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 16, size=(height, width))
    b = np.clip(0.5 * checker + ring + noise, 0, 255)
    return np.stack([r, g, b], axis=-1).round().astype(np.uint8)


def synthetic_wav(path, seconds, rate):
    """A deterministic mono WAV in 16 bits: a chirp from 200 Hz to 3 kHz with a 440 Hz tone and a fade."""
    n = int(round(seconds * rate))
    t = np.arange(n) / rate
    f0, f1 = 200.0, 3000.0
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * seconds))
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.05)
    signal = (0.45 * np.sin(phase) + 0.2 * np.sin(2 * np.pi * 440.0 * t)) * fade
    pcm = np.clip(np.round(signal * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return pcm


def read_wav(path):
    """The samples of a 16-bit mono WAV as float32 in [-1, 1): each sample divided by 32768."""
    with wave.open(str(path), "rb") as handle:
        check(handle.getnchannels() == 1 and handle.getsampwidth() == 2, "the WAV is 16-bit mono")
        rate = handle.getframerate()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return pcm, (pcm.astype(np.float32) / 32768.0), rate


def omni_media_prompt(processor, parts, system=None):
    """A user turn with media parts and a question, rendered by the checkpoint's template."""
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": parts})
    return processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def run_vision(model, processor, out, facts, media_entries):
    from PIL import Image

    tensors = {}
    record = {}
    image_token = model.config.image_token_id
    for name, (height, width), seed in (("small", (96, 160), 1), ("mid", (300, 420), 2)):
        path = Path(out) / f"image_{name}.png"
        Image.fromarray(synthetic_image(height, width, seed), mode="RGB").save(path, format="PNG")
        image = Image.open(path).convert("RGB")
        text = omni_media_prompt(processor, [{"type": "image"}, {"type": "text", "text": "Describe the image."}])
        inputs = processor(text=text, images=[image], return_tensors="pt")
        pixel_values, grid = inputs["pixel_values"], inputs["image_grid_thw"]
        ids = inputs["input_ids"][0]
        t, gh, gw = (int(v) for v in grid[0])
        positions = t * gh * gw // 4
        check(int((ids == image_token).sum()) == positions, f"image {name}: placeholders are grid / merge^2")
        with torch.no_grad():
            vision = model.model.vision_tower(pixel_values, grid_thw=grid, return_dict=True)
            patches = model.model.vision_tower.embedding_norm(
                model.model.vision_tower.patchifier(pixel_values.float())
            )
            rows = model.model.multi_modal_projector(vision.last_hidden_state)
            check(rows.shape[0] == positions, f"image {name}: projector rows")
            output = model(**inputs, use_cache=False, logits_to_keep=1)
            scattered = model.model.get_image_features(pixel_values, grid)
        close(scattered, rows, 0.0, f"image {name}: get_image_features is tower then projector")
        rgb = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
        resized = resize_like_processor(rgb, gh * 16, gw * 16)
        close(patchify(resized), pixel_values, 0.0, f"image {name}: bicubic u8 resize, scale, merge-block patches")
        tensors[f"{name}.resized_rgb"] = resized.permute(1, 2, 0).contiguous()
        tensors[f"{name}.png_rgb"] = rgb
        tensors[f"{name}.pixel_values"] = pixel_values.float()
        tensors[f"{name}.grid_thw"] = grid.to(torch.int64)
        tensors[f"{name}.patches"] = patches
        tensors[f"{name}.tower"] = vision.last_hidden_state
        tensors[f"{name}.rows"] = rows
        tensors[f"{name}.ids"] = ids.to(torch.int64)
        media_entries.append(top_entry(f"image_{name}", "synthetic PNG of this tool, with a question", text,
                                       ids.tolist(), output.logits[0, -1]))
        record[name] = {
            "source_size": [height, width],
            "resized_size": [gh * 16, gw * 16],
            "grid_thw": [t, gh, gw],
            "patches": gh * gw,
            "positions": positions,
            "png_sha256": sha256(path),
        }
        log(f"image {name}: {height}x{width} -> {gh * 16}x{gw * 16}, {gh * gw} patches, {positions} positions")
    smart_resize = sys.modules[type(processor.image_processor).__module__].smart_resize
    size = processor.image_processor.size
    table = []
    for height, width in ((96, 160), (300, 420), (256, 256), (64, 64), (480, 640), (1080, 1920), (3000, 4000),
                          (100, 5000), (2048, 2048)):
        h, w = smart_resize(height, width, 32, size["shortest_edge"], size["longest_edge"])
        table.append({"source": [height, width], "resized": [h, w], "positions": h * w // 1024})
    record["smart_resize"] = {"factor": 32, "min_pixels": size["shortest_edge"], "max_pixels": size["longest_edge"],
                              "table": table}
    meta = {"images": record, "versions": versions()}
    save(tensors, Path(out) / "vision.safetensors", meta)
    facts["vision"] = record


def resize_like_processor(rgb, height, width):
    """The processor's resize: torchvision bicubic with antialias, on the u8 image, to u8. Returns [3, H, W]."""
    from torchvision.transforms.v2 import functional as tvF

    return tvF.resize(rgb.permute(2, 0, 1), [height, width], interpolation=tvF.InterpolationMode.BICUBIC,
                      antialias=True)


def patchify(resized):
    """The processor's patches: (x - 127.5) / 127.5 in f32, then one row per 16 by 16 patch in merge-block order.

    A row is channel-major, then the 2 temporal copies, then 16 rows and 16 columns of pixels. The 2 by 2 patches of
    a merge block are 4 rows in raster order, and the blocks go in raster order too.
    """
    # The processor fuses the rescale by 1/255 into the mean and the std of 0.5: both become 127.5.
    x = (resized.float() - 127.5) / 127.5
    c, height, width = x.shape
    gh, gw = height // 16, width // 16
    x = x.reshape(c, gh // 2, 2, 16, gw // 2, 2, 16).permute(1, 4, 2, 5, 0, 3, 6)
    x = x.unsqueeze(5).expand(-1, -1, -1, -1, -1, 2, -1, -1)
    return x.reshape(gh * gw, c * 2 * 16 * 16)


def run_audio(model, processor, out, facts, media_entries):
    path = Path(out) / "chirp.wav"
    synthetic_wav(path, 1.5, 16000)
    pcm, samples, rate = read_wav(path)
    extractor = processor.feature_extractor
    text = omni_media_prompt(processor, [{"type": "audio"}, {"type": "text", "text": "What do you hear?"}])
    inputs = processor(text=text, audio=[samples], sampling_rate=rate, return_tensors="pt")
    features, mask = inputs["input_features"], inputs["feature_attention_mask"]
    frames = int(mask.sum())
    positions = -(frames // -8)
    ids = inputs["input_ids"][0]
    check(frames == extractor.num_frames(len(samples)), "mel frame count")
    check(int((ids == model.config.audio_token_id).sum()) == positions, "audio placeholders are ceil(frames / 8)")
    encoder = model.model.audio_encoder
    with torch.no_grad():
        down = encoder.downsample(features.float(), mask.sum(-1)).transpose(1, 2)
        encoded, lengths = encoder(features.float(), mask.sum(-1))
        rows = model.model.audio_projector(encoded[0, : int(lengths[0])])
        scattered = model.model.get_audio_features(features, mask)
        output = model(**inputs, use_cache=False, logits_to_keep=1)
    check(int(lengths[0]) == positions, "encoder length")
    close(scattered, rows, 0.0, "get_audio_features is encoder then projector")
    tensors = {
        "pcm": torch.from_numpy(pcm.astype(np.int16).copy()),
        "samples": torch.from_numpy(samples.copy()),
        "mel": features[0].float(),
        "mel_filters": torch.from_numpy(np.asarray(extractor.mel_filters, dtype=np.float32).copy()),
        "downsample": down[0],
        "encoder": encoded[0, :positions],
        "rows": rows,
        "ids": ids.to(torch.int64),
    }
    media_entries.append(top_entry("audio_chirp", "synthetic WAV of this tool, with a question", text, ids.tolist(),
                                   output.logits[0, -1]))
    record = {
        "samples": len(samples),
        "rate": rate,
        "padded_samples": extractor.pad_length(len(samples)),
        "mel_frames": frames,
        "positions": positions,
        "wav_sha256": sha256(path),
        "rule": "padded = n + (-(n - 400) mod 160), at least 400; frames = 1 + (padded - 400) / 160; "
        "positions = ceil(frames / 8)",
    }
    meta = {"audio": record, "versions": versions()}
    save(tensors, Path(out) / "audio_in.safetensors", meta)
    facts["audio_in"] = record
    log(f"audio: {len(samples)} samples, {frames} mel frames, {positions} positions")


def synthetic_video(path, seconds, rate, size, sound):
    """A video of ffmpeg's test pattern in H.264, with a 440 Hz tone in AAC at 44.1 kHz when `sound`. The bytes depend
    on the ffmpeg build, so the file is kept beside the oracle and its hash is in the metadata."""
    import subprocess

    command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
               f"testsrc2=size={size[1]}x{size[0]}:rate={rate}:duration={seconds}"]
    if sound:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "1"]
    command += ["-c:a", "aac", "-shortest"] if sound else ["-an"]
    subprocess.run(command + [str(path)], check=True)


def video_patchify(frames):
    """Pairs of frames [2k, 3, h, w] in [0, 1] as the processor's patches: [k * h/16 * w/16, 1536] in merge-block
    order, each patch channel, time, row and column (video_processing_lfm3_vl.py, patchify)."""
    n, c, h, w = frames.shape
    gt, gh, gw = n // 2, h // 16, w // 16
    x = frames.view(1, gt, 2, c, gh // 2, 2, 16, gw // 2, 2, 16)
    x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return x.reshape(gt * gh * gw, c * 2 * 16 * 16)


def run_video(model, processor, out, facts):
    """The video gate of Omni (spec 11.11, "The video of a request"): two synthetic videos, one with sound. Its top
    logits go to lfm3-omni.video.json, so a run of this stage alone keeps lfm3-omni.media.json."""
    from torchcodec.decoders import AudioDecoder, VideoDecoder

    tensors = {}
    record = {}
    media_entries = []
    video_token, audio_token = model.config.video_token_id, model.config.audio_token_id
    vp = processor.video_processor
    for name, seconds, rate, size, sound in (("av", 2.5, 10, (120, 160), True), ("v", 3.0, 12, (240, 320), False)):
        path = Path(out) / f"video_{name}.mp4"
        synthetic_video(path, seconds, rate, size, sound)
        conv = [{"role": "user", "content": [{"type": "video", "path": str(path)},
                                             {"type": "text", "text": "Describe the video."}]}]
        inputs = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", load_audio_from_video=sound,
                                               processor_kwargs={"use_audio_in_video": sound})
        ids = inputs["input_ids"][0]
        pixel_values, grid = inputs["pixel_values_videos"], inputs["video_grid_thw"]
        gt, gh, gw = (int(v) for v in grid[0])
        decoder = VideoDecoder(str(path))
        total, fps = decoder.metadata.num_frames, round(float(decoder.metadata.average_fps), 6)
        count = min(max(int(total / fps * vp.fps), vp.min_frames), vp.max_frames, total)
        indices = np.linspace(0, total - 1, count).round().astype(int)
        check(-(-count // 2) == gt, f"video {name}: {count} frames give {gt} pairs")
        decoded = decoder.get_frames_at([int(i) for i in indices]).data  # [n, 3, H, W] u8
        frames = decoded if count % 2 == 0 else torch.cat([decoded, decoded[-1:]])
        resized = torch.stack([resize_like_processor(f.permute(1, 2, 0).contiguous(), gh * 16, gw * 16)
                               for f in frames])
        # The processor fuses the rescale by 1/255 into the mean and the std of 0.5, as for an image (patchify).
        scaled = (resized.float() - 127.5) / 127.5
        close(video_patchify(scaled), pixel_values, 0.0, f"video {name}: frames, u8 resize, scale, patches")
        check(int((ids == video_token).sum()) == gt * gh * gw // 4, f"video {name}: video tokens")
        with torch.no_grad():
            rows = model.model.get_video_features(pixel_values, grid)
            rows = torch.cat(list(rows)) if isinstance(rows, (list, tuple)) else rows
            output = model(**inputs, use_cache=False, logits_to_keep=1)
        # Each pair is an image of one frame in time: the tower of a pair alone gives its rows.
        per = gh * gw // 4
        for k in range(gt):
            one = pixel_values[k * gh * gw:(k + 1) * gh * gw]
            with torch.no_grad():
                alone = model.model.get_image_features(one, torch.tensor([[1, gh, gw]]))
            alone = torch.cat(list(alone)) if isinstance(alone, (list, tuple)) else alone
            close(alone, rows[k * per:(k + 1) * per], 1e-5, f"video {name}: pair {k} is an image")
        tensors[f"{name}.ids"] = ids.to(torch.int64)
        tensors[f"{name}.indices"] = torch.from_numpy(indices.astype(np.int64))
        tensors[f"{name}.decoded_rgb"] = decoded.permute(0, 2, 3, 1).contiguous()
        tensors[f"{name}.resized_rgb"] = resized.permute(0, 2, 3, 1).contiguous()
        tensors[f"{name}.pixel_values"] = pixel_values.float()
        tensors[f"{name}.grid_thw"] = grid.to(torch.int64)
        tensors[f"{name}.rows"] = rows.float()
        entry = {"seconds": seconds, "rate": rate, "size": list(size), "sound": sound, "stream_frames": total,
                 "fps": fps, "frames": count, "grid_thw": [gt, gh, gw], "mp4_sha256": sha256(path)}
        if sound:
            features, mask = inputs["input_features"], inputs["feature_attention_mask"]
            samples = AudioDecoder(str(path), sample_rate=16000, num_channels=1).get_all_samples().data[0]
            check(int(mask.sum()) == processor.feature_extractor.num_frames(len(samples)), f"video {name}: mel")
            with torch.no_grad():
                audio_rows = model.model.get_audio_features(features, mask)
            audio_rows = torch.cat(list(audio_rows)) if isinstance(audio_rows, (list, tuple)) else audio_rows
            check(int((ids == audio_token).sum()) == audio_rows.shape[0], f"video {name}: sound tokens")
            tensors[f"{name}.samples"] = samples.float()
            tensors[f"{name}.mel"] = features[0].float()
            tensors[f"{name}.audio_rows"] = audio_rows.float()
            entry["audio_positions"] = int(audio_rows.shape[0])
        media_entries.append(top_entry(f"video_{name}", "synthetic H.264 video of this tool, with a question",
                                       processor.tokenizer.decode(ids), ids.tolist(), output.logits[0, -1]))
        record[name] = entry
        log(f"video {name}: {total} frames at {fps} fps -> {count} frames, grid {gt}x{gh}x{gw}, {len(ids)} ids")
    meta = {"videos": record, "versions": versions()}
    save(tensors, Path(out) / "video.safetensors", meta)
    document = {"format": 1, "model": "lfm3-omni", "vocab": model.config.text_config.vocab_size,
                "versions": versions(), "prompts": media_entries}
    (Path(out) / "lfm3-omni.video.json").write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")
    facts["video"] = record


def run_audio_out(model, processor, out, facts):
    """The gate of an assistant audio part (spec 11.11, "Read it back"): a conversation whose assistant turn holds the
    text and the speech of the TTS run of the speech stage, through the model with its codes. The processor would
    encode a waveform with the codec; the nagaya holds the codes, so this stage gives the model the codes."""
    from safetensors.torch import load_file

    speech = load_file(str(Path(out) / "speech.safetensors"))
    codes = speech["tts.codes"].to(torch.int64)
    eos = int(processor.audio_eos_code)
    if bool((codes[-1] == eos).all()):
        codes = codes[:-1]
    # The prompt and the targets carry the end frame after the speech, as generate(return_audio_eos=True) does.
    with_eos = torch.cat([codes, torch.full((1, codes.shape[1]), eos, dtype=codes.dtype)])
    tok = processor.tokenizer
    messages = [
        {"role": "system", "content": "Perform TTS in American English using a female voice."},
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello."}, {"type": "audio_out"}]},
        {"role": "user", "content": "Say it again, slower."},
    ]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    check(text.count(processor.audio_out_token) == 1, "the template writes one audio_out placeholder")
    text = text.replace(processor.audio_out_token, processor.audio_out_token * with_eos.shape[0])
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    check(int((ids == model.config.audio_out_token_id).sum()) == with_eos.shape[0], "a position for each frame")
    with torch.no_grad():
        output = model(input_ids=ids, audio_out_codes=with_eos, use_cache=False, logits_to_keep=1)
        rows = model.model.get_audio_out_embeds(with_eos)
    tensors = {"ids": ids[0].to(torch.int64), "codes": codes, "rows": rows.float()}
    record = {"frames": int(codes.shape[0]), "positions": int(with_eos.shape[0]), "eos": eos}
    save(tensors, Path(out) / "audio_out.safetensors", {"audio_out": record, "versions": versions()})
    entry = top_entry("audio_out", "the TTS run of the speech stage as an assistant audio part", text,
                      ids[0].tolist(), output.logits[0, -1])
    document = {"format": 1, "model": "lfm3-omni", "vocab": model.config.text_config.vocab_size,
                "versions": versions(), "prompts": [entry]}
    (Path(out) / "lfm3-omni.audio_out.json").write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")
    facts["audio_out"] = record
    log(f"audio_out: {record['frames']} frames, {record['positions']} positions, {ids.shape[1]} ids")


def teacher_forced(model, ids, codes):
    """One forward over a whole run, with the codes of each audio placeholder: the hidden state of each position."""
    with torch.no_grad():
        out = model.model(
            input_ids=torch.tensor([ids]),
            audio_out_codes=codes if codes is not None and codes.shape[0] else None,
            use_cache=False,
        )
    return out.last_hidden_state[0]


def run_speech(model, processor, codec, out, facts):
    config = model.config
    tok = processor.tokenizer
    audio_out = config.audio_out_token_id
    tensors = {}
    record = {}

    def generate(system, text, mode, max_new):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = tok.encode(prompt, add_special_tokens=False)
        torch.manual_seed(SEED)
        with torch.no_grad():
            result = model.generate(
                input_ids=torch.tensor([ids]),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                max_new_tokens=max_new,
                generation_mode=mode,
                return_audio_eos=True,
                return_dict_in_generate=True,
            )
        length = int(result.sequence_lengths[0])
        sequence = result.sequences[0, :length].tolist()
        frames = int(result.audio_lengths[0])
        codes = result.audio_codes[0, :, :frames].T.contiguous()
        return prompt, ids, sequence, codes

    # The TTS run: speech mode, the default audio sampler of generation_config.json, seed 1234.
    log("speech: tts run")
    prompt, ids, sequence, codes = generate(TTS_SYSTEM, TTS_TEXT, "audio", 64)
    marker = config.audio_start_token_id
    check(sequence[len(ids)] == marker, "the audio mode appends <|audio_start|>")
    prompt_len = len(ids) + 1  # with the marker
    ended = bool(codes.shape[0]) and bool((codes[-1] == config.audio_eos_code).all())
    hidden = teacher_forced(model, sequence, codes)
    # hidden[p] predicts the slot at p + 1. The marker at prompt_len - 1 predicts frame 0.
    frame_hidden = hidden[prompt_len - 1 : prompt_len - 1 + codes.shape[0] + 1]
    with torch.no_grad():
        end_logits = model.lm_head(hidden[-1])
    tensors["tts.prompt_ids"] = torch.tensor(sequence[:prompt_len], dtype=torch.int64)
    tensors["tts.sequence"] = torch.tensor(sequence, dtype=torch.int64)
    tensors["tts.codes"] = codes
    tensors["tts.hidden"] = frame_hidden.contiguous()
    tensors["tts.end_logits"] = end_logits
    speech_codes = codes[:-1] if ended else codes
    record["tts"] = {
        "system": TTS_SYSTEM,
        "text": TTS_TEXT,
        "prompt": prompt,
        "prompt_positions": prompt_len,
        "frames_with_end": int(codes.shape[0]),
        "ended": ended,
        "end_frame_fed": "yes: generate feeds the all-2016 frame back as one position (modeling_lfm3_omni.py "
        "_generate, next_embeds after is_eos), and a prompt carries it after each audio_out clip "
        "(processing_lfm3_omni.py _expand_audio_out)",
    }
    log(f"speech: tts {codes.shape[0]} frames, ended {ended}")

    # The depth decoder with forced codes: 4 frames, 4 reveal masks each.
    count = min(4, int(speech_codes.shape[0]))
    depth_hidden = frame_hidden[:count]
    depth_codes = speech_codes[:count]
    rng = np.random.default_rng(SEED)
    orders = np.stack([rng.permutation(config.num_codebooks) for _ in range(count)])
    masks = []
    for revealed in (0, 4, 8, 11):
        m = np.ones((count, config.num_codebooks), dtype=bool)
        for row in range(count):
            m[row, orders[row, :revealed]] = False
        masks.append(m)
    decoder = model.model.depth_decoder
    with torch.no_grad():
        for index, m in enumerate(masks):
            masked = torch.from_numpy(m)
            logits = decoder(depth_hidden, depth_codes, masked)
            tensors[f"depth.mask{index}"] = masked.to(torch.uint8)
            tensors[f"depth.logits{index}"] = logits.float()
            if index == 0:
                depth_in = decoder.depth_linear(depth_hidden).unflatten(-1, (config.num_codebooks, -1))
                depth_in = depth_in + torch.where(masked[..., None], decoder.mask_embedding,
                                                  decoder.embed_codes(depth_codes))
                tensors["depth.in0"] = depth_in
                tensors["depth.trunk0"] = decoder.trunk(depth_in)
        summed = model.model.get_audio_out_embeds(depth_codes)
        table = model.model.codebook_embedding.weight
        manual = sum(table[depth_codes[:, g] + g * config.codebook_size] for g in range(config.num_codebooks))
    close(summed, manual, 1e-5, "get_audio_out_embeds is the sum of 12 rows of one table")
    tensors["depth.hidden"] = depth_hidden.contiguous()
    tensors["depth.codes"] = depth_codes.contiguous()
    tensors["depth.order"] = torch.from_numpy(orders.astype(np.int64))
    tensors["depth.audio_out_embeds"] = summed
    record["depth"] = {"frames": count, "revealed": [0, 4, 8, 11], "maskgit_schedule": config.maskgit_schedule}

    # The codec: the first 20 speech frames, all at once, frame by frame, and in chunks of 4.
    count = min(20, int(speech_codes.shape[0]))
    fixed = speech_codes[:count]
    with torch.no_grad():
        full = codec.decode(fixed.T[None], return_dict=True).audio_values[0]
        check(full.shape[0] == FRAME_SAMPLES * count, "the codec gives 1920 samples for each frame")
        stream = []
        for chunk_size, label in ((1, "frames"), (4, "chunks4")):
            cache = codec_cache(codec)
            pieces = []
            for start in range(0, count, chunk_size):
                last = start + chunk_size >= count
                piece = codec.decode(fixed[start : start + chunk_size].T[None], streaming_cache=cache,
                                     finalize=last, return_dict=True).audio_values[0]
                pieces.append(piece)
            joined = torch.cat(pieces)
            tensors[f"codec.audio_{label}"] = joined
            stream.append((label, joined, [int(p.shape[0]) for p in pieces]))
        whole = codec.decode(speech_codes.T[None], return_dict=True).audio_values[0]
    tensors["codec.codes"] = fixed.contiguous()
    tensors["codec.audio_full"] = full
    tensors["tts.audio"] = whole
    record["codec"] = {"frames": count, "samples": int(full.shape[0])}
    for label, joined, sizes in stream:
        same_shape = joined.shape == full.shape
        record["codec"][f"{label}_max_abs_vs_full"] = max_abs(joined, full) if same_shape else None
        record["codec"][f"{label}_piece_samples"] = sizes
        log(f"codec: {label} vs full: max abs {record['codec'][f'{label}_max_abs_vs_full']}")

    # The interleaved run: mixed mode, greedy text, the default audio sampler, seed 1234.
    log("speech: interleaved run")
    prompt, ids, sequence, codes = generate(MIXED_SYSTEM, MIXED_TEXT, "interleaved", 96)
    check(sequence[len(ids)] == config.mixed_start_token_id, "the interleaved mode appends <|mixed_start|>")
    prompt_len = len(ids) + 1
    hidden = teacher_forced(model, sequence, codes)
    slots = sequence[prompt_len:]
    with torch.no_grad():
        text_logits = model.lm_head(hidden[prompt_len - 1 : len(sequence) - 1])
    greedy = text_logits.argmax(-1)
    text_slots = [i for i, token in enumerate(slots) if token != audio_out]
    for i in text_slots:
        check(int(greedy[i]) == slots[i], f"interleaved slot {i}: the forced pass gives the greedy text token")
    kinds = torch.tensor([0 if token != audio_out else 1 for token in slots], dtype=torch.uint8)
    tensors["mixed.prompt_ids"] = torch.tensor(sequence[:prompt_len], dtype=torch.int64)
    tensors["mixed.sequence"] = torch.tensor(sequence, dtype=torch.int64)
    tensors["mixed.kinds"] = kinds
    tensors["mixed.codes"] = codes.contiguous()
    tensors["mixed.hidden"] = hidden[prompt_len - 1 :].contiguous()
    tensors["mixed.text_top_ids"] = torch.topk(text_logits, 8).indices
    ended = bool(codes.shape[0]) and bool((codes[-1] == config.audio_eos_code).all())
    mixed_codes = codes[:-1] if ended else codes
    with torch.no_grad():
        tensors["mixed.audio"] = (
            codec.decode(mixed_codes.T[None], return_dict=True).audio_values[0]
            if mixed_codes.shape[0]
            else torch.zeros(0)
        )
    record["mixed"] = {
        "system": MIXED_SYSTEM,
        "text": MIXED_TEXT,
        "prompt": prompt,
        "prompt_positions": prompt_len,
        "slots": len(slots),
        "text_slots": len(text_slots),
        "frames_with_end": int(codes.shape[0]),
        "ended": ended,
        "pattern": "".join("t" if k == 0 else "a" for k in kinds.tolist()),
        "reply_text": tok.decode([t for t in slots if t != audio_out]),
    }
    log(f"speech: mixed {len(slots)} slots, pattern {record['mixed']['pattern']}")
    meta = {"speech": record, "versions": versions(), "seed": SEED,
            "audio_sampler": {k: getattr(model.generation_config, k, None) for k in
                              ("audio_temperature", "audio_top_k", "audio_repetition_penalty",
                               "audio_repetition_window")}}
    save(tensors, Path(out) / "speech.safetensors", meta)
    facts["speech"] = record
    facts["speech"]["audio_sampler"] = meta["audio_sampler"]


def codec_cache(codec):
    module = sys.modules[type(codec).__module__]
    return module.LiquidCodecStreamingCache(codec.config)


LATENT_PROMPT = "Plan a short story about a lighthouse keeper who finds a map. Think before you write."
LATENT_STEPS = 16
LATENT_THEN = 8


def run_latent(model, backbone, lm_head, tokenizer, out, facts):
    """The latent gate of LFM3-1.7B (docs/lab.md, "`latent` and `merge`"). After the prompt, each latent step feeds
    the input of the head (the residual after the last layer, through the final norm), scaled to the mean norm of the
    embedding rows, as the input row of the next position. Then greedy tokens go on from the latent context.

    The stage checks that the final norm of the last layer's residual is the hidden state of the model, and that the
    head is the embedding table, so the map of LatentMAS is a scale only.
    """
    _, ids = chat_ids(tokenizer, LATENT_PROMPT)
    table = backbone.embed_tokens.weight.detach().float()
    check(bool(torch.equal(lm_head.weight, backbone.embed_tokens.weight)), "the head is the embedding table")
    scale = float(table.norm(dim=1).mean())
    norm = backbone.embedding_norm
    inputs, normed, top_ids, top_values = [], [], [], []
    with torch.no_grad():
        tap = LayerTap(backbone)
        out_ = model(input_ids=torch.tensor([ids]), use_cache=True)
        last = backbone.config.num_hidden_layers - 1
        h = tap.outputs["final_norm"][0, -1]
        close(norm(tap.outputs[f"layer{last}"][0, -1:])[0], h, 1e-5, "the final norm of the last residual")
        for _ in range(LATENT_STEPS):
            row = h * (scale / h.norm())
            inputs.append(row)
            out_ = model(inputs_embeds=row.view(1, 1, -1), past_key_values=out_.past_key_values, use_cache=True)
            h = tap.outputs["final_norm"][0, -1]
            normed.append(h)
            logits = out_.logits[0, -1].float()
            top = torch.topk(logits, TOP)
            top_ids.append(top.indices)
            top_values.append(top.values)
        tap.remove()
        then, then_ids, then_values = [], [], []
        for _ in range(LATENT_THEN):
            logits = out_.logits[0, -1].float()
            token = int(torch.argmax(logits))
            top = torch.topk(logits, TOP)
            then.append(token)
            then_ids.append(top.indices)
            then_values.append(top.values)
            out_ = model(input_ids=torch.tensor([[token]]), past_key_values=out_.past_key_values, use_cache=True)
    tensors = {
        "ids": torch.tensor(ids, dtype=torch.int64),
        "scale": torch.tensor([scale]),
        "inputs": torch.stack(inputs),
        "final_norm": torch.stack(normed),
        "mean": torch.stack(normed).mean(0),
        "top_ids": torch.stack(top_ids),
        "top_values": torch.stack(top_values),
        "then_tokens": torch.tensor(then, dtype=torch.int64),
        "then_top_ids": torch.stack(then_ids),
        "then_top_values": torch.stack(then_values),
    }
    read = [tokenizer.decode([int(t[0])]) for t in top_ids]
    meta = {"model": "lfm3", "prompt": LATENT_PROMPT, "steps": LATENT_STEPS, "then": LATENT_THEN,
            "versions": versions()}
    save(tensors, Path(out) / "lfm3.latent.safetensors", meta)
    facts["lfm3_latent"] = {"steps": LATENT_STEPS, "scale": scale, "read": read, "then": tokenizer.decode(then)}
    log(f"latent: scale {scale:.4f}, read {read!r}, then {tokenizer.decode(then)!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni", required=True, type=Path)
    parser.add_argument("--text", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stages", default="text,vision,audio,speech")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("LFM3_REF_THREADS", "16")))
    args = parser.parse_args()
    stages = set(args.stages.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)

    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

    facts_path = args.out / "facts.json"
    facts = json.loads(facts_path.read_text()) if facts_path.exists() else {}
    facts["versions"] = versions()

    if stages & {"text", "control", "project", "latent"}:
        log("load LFM3-1.7B")
        model = AutoModelForCausalLM.from_pretrained(
            args.text, dtype=torch.float32, attn_implementation="eager", trust_remote_code=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.text, trust_remote_code=True)
        if "text" in stages:
            run_text(model, model.model, model.lm_head, tokenizer, "lfm3", args.text,
                     model.config.num_hidden_layers, args.out, facts)
        if "control" in stages:
            run_control(model, model.model, model.lm_head, tokenizer, model.config.num_hidden_layers, args.out,
                        facts)
        if "project" in stages:
            run_project(model.model, model.lm_head, tokenizer, args.out, facts)
        if "latent" in stages:
            run_latent(model, model.model, model.lm_head, tokenizer, args.out, facts)
        del model

    if stages & {"text", "vision", "audio", "speech", "control-omni", "video", "audio-out"}:
        log("load LFM3-Omni-2B")
        model = AutoModelForMultimodalLM.from_pretrained(
            args.omni, dtype=torch.float32, attn_implementation="eager", trust_remote_code=True
        ).eval()
        processor = AutoProcessor.from_pretrained(args.omni, trust_remote_code=True)
        if "text" in stages:
            run_text(model, model.model.language_model, model.lm_head, processor.tokenizer, "lfm3-omni", args.omni,
                     model.config.text_config.num_hidden_layers, args.out, facts)
        if "control-omni" in stages:
            run_control(model, model.model.language_model, model.lm_head, processor.tokenizer,
                        model.config.text_config.num_hidden_layers, args.out, facts, label="lfm3-omni")
        media_entries = []
        if "vision" in stages:
            run_vision(model, processor, args.out, facts, media_entries)
        if "audio" in stages:
            run_audio(model, processor, args.out, facts, media_entries)
        if "video" in stages:
            run_video(model, processor, args.out, facts)
        if media_entries:
            document = {"format": 1, "model": "lfm3-omni", "vocab": model.config.text_config.vocab_size,
                        "versions": versions(), "prompts": media_entries}
            (args.out / "lfm3-omni.media.json").write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")
        if "speech" in stages:
            codec = processor.audio_tokenizer.float().eval()
            run_speech(model, processor, codec, args.out, facts)
        if "audio-out" in stages:
            run_audio_out(model, processor, args.out, facts)

    facts["seconds"] = round(time.time() - STARTED)
    facts_path.write_text(json.dumps(facts, indent=1, ensure_ascii=False) + "\n")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
