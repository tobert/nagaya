# SPDX-License-Identifier: MIT
"""f32 CPU reference for Inkling 2 Mini and the variable-loop research model: the Inkling 2 oracle (spec 8.3).

Run it through tools/inkling2_golden.sh, which names each file and tensor. The direct command is:

  PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 nice -n 19 ionice -c3 <reference venv>/bin/python -I \
      tools/inkling2_reference.py --mini "$INKLING2_MODEL_DIR" --varloop "$INKLING2_VARLOOP_MODEL_DIR" \
      --out target/inkling2-golden

The reference is the transformers model `inkling2_loop` at commit 046bdb8cc2, on the CPU. transformers expands the
NVFP4 and MXFP8 experts to bf16 at load. This tool then moves every other parameter to f32, and each expert
expands to f32 in its call. Thus every sum is f32, and the weights hold their exact values. A run in bf16 differs
from it by 0.2 on average in each logit (measured on 2026-11-21), so the oracle is f32.

The tool reads the model folders and never writes into them. It writes only into --out. It loads one model at a
time; each one holds about 74 GiB of RAM.

The stages:

  text     the logits of 5 prompts (top-64 of the last position and the logsumexp), for each model
  layers   the prompt "plain" through Mini: the row after each run, each message and each gate, and the logits
  cycles   the prompts "plain" and "chat" at 1 to 5 cycles, for each model
  decode   8 greedy tokens of "chat" with the cache of the reference, and the logits after each, for each model
  vision   two synthetic images through the processor, the vision tower and the projector, and a chat prompt that
           holds both, through Mini

The tool stops with an AssertionError when a self check fails. The checks:

  - the patched expert call runs in every layer, so no expert sum is in bf16;
  - each prompt fits in MAX_TOKENS tokens;
  - the hooks see 24 + 2 * 12 = 48 runs of the decoder layers at 3 cycles;
  - the final norm of the hooks, the head and the softcap give the logits of the model;
  - a decode with the cache gives the logits of a forward over the whole sequence with no cache;
  - the count of placeholders of an image equals its count of projector rows;
  - the rows that the model scatters into the placeholders are the projector rows, and the other rows are the
    normed embeddings of their tokens;
  - the hooks of the tower give its output, and a tower run of both images gives the rows of two runs of one.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
TOP = 64
MAX_TOKENS = 640
DECODE_STEPS = 8
STAGES = ("text", "layers", "cycles", "decode", "vision")
# The synthetic images of the stage "vision": (width, height, seed). No side is a multiple of 40, so the processor
# pads each image on the right and at the bottom.
VISION_IMAGES = ((213, 118, 35_0), (637, 479, 35_1))
VISION_PROMPT = "Describe the two images."
STARTED = time.time()


def log(message):
    print(f"[{time.time() - STARTED:7.1f}s] {message}", file=sys.stderr, flush=True)


def check(ok, what):
    """Stop with the name of the self check that failed."""
    if not ok:
        raise AssertionError(f"self check failed: {what}")


def max_abs(a, b):
    return (a.double() - b.double()).abs().max().item()


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
    import tokenizers
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
    }


def source_record(folder):
    """The files of a model folder that the reference reads, with their SHA-256. The weights are named by size."""
    folder = Path(folder)
    names = sorted(p.name for p in folder.iterdir() if p.suffix in (".json", ".jinja"))
    record = {name: sha256(folder / name) for name in names}
    weights = folder / "model-00001-of-00001.safetensors"
    return {
        "folder": str(folder),
        "revision": folder.name,
        "files": record,
        "weights_bytes": weights.stat().st_size,
    }


class ExpertCalls:
    """The expert call of the reference, with every sum in f32. It counts its calls for a self check."""

    count = 0

    @staticmethod
    def forward(self, hidden_states, top_k_index, top_k_weights):
        ExpertCalls.count += 1
        final = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert in hit:
            expert = expert[0]
            if expert == self.num_experts:
                continue
            position, token = torch.where(mask[expert])
            state = hidden_states[token].float()
            state = self._apply_gate(F.linear(state, self.gate_up_proj[expert].float()))
            state = F.linear(state, self.down_proj[expert].float()) * top_k_weights[token, position, None]
            final.index_add_(0, token, state.float())
        return final.to(hidden_states.dtype)


def load(folder):
    """Loads one model in bf16, then moves every parameter but the experts to f32."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from transformers.models.inkling2_loop import modeling_inkling2_loop as modeling

    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForImageTextToText.from_pretrained(folder, dtype=torch.bfloat16).eval()
    for name, parameter in model.named_parameters():
        if ".mlp.experts." not in name:
            parameter.data = parameter.data.float()
    for _name, buffer in model.named_buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.float()
    modeling.Inkling2LoopExperts.forward = ExpertCalls.forward
    text = model.model.language_model
    check(len(text.layers) == 24, f"the model has 24 layers, not {len(text.layers)}")
    log(f"loaded {folder}, {model.config.text_config.loop_num_passes} cycles")
    return model, text, tokenizer


def set_cycles(text, count):
    """The reference reads the count of cycles from its config at each forward."""
    text.config.loop_num_passes = count


def forward(model, ids):
    """The logits of every position of one sequence, with no cache, in f32."""
    before = ExpertCalls.count
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
    check(ExpertCalls.count > before, "the f32 expert call ran")
    return logits


def chat_text(tokenizer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": "Name three colors."}],
        tokenize=False,
        reasoning_effort="none",
    )


def text_prompts(tokenizer):
    long, longer, author = prose()
    prompts = [
        ("plain", "The capital of France is", "fixed in this tool"),
        ("code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return", "fixed in this tool"),
        ("chat", chat_text(tokenizer), "the checkpoint's chat_template.jinja, effort none, rendered by transformers"),
        ("long", long, "prose " + author),
        ("longer", longer, "prose " + author),
    ]
    out = []
    for name, text, source in prompts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        check(0 < len(ids) <= MAX_TOKENS, f"prompt {name} has {len(ids)} tokens")
        out.append((name, text, source, ids))
    return out


def run_text(model, tokenizer, label, folder, out, facts):
    entries = []
    for name, text, source, ids in text_prompts(tokenizer):
        started = time.time()
        logits = forward(model, ids)
        entries.append(top_entry(name, source, text, ids, logits[-1]))
        log(f"{label} {name}: {len(ids)} tokens, {time.time() - started:.1f}s")
    fixture = {
        "family": "inkling2",
        "model": label,
        "reference": "transformers inkling2_loop 046bdb8cc2, f32 sums with bf16 weights, CPU",
        "source": source_record(folder),
        "versions": versions(),
        "prompts": entries,
    }
    path = out / f"{label}.oracle.json"
    path.write_text(json.dumps(fixture, indent=1) + "\n")
    facts[f"{label}.prompt_tokens"] = {e["name"]: len(e["ids"]) for e in entries}
    log(f"wrote {path}")


class RunTap:
    """Keeps the output of each run of each decoder layer, each message and each gate, for one forward."""

    def __init__(self, text):
        self.runs = []
        self.messages = []
        self.gates = []
        self.handles = [layer.register_forward_hook(self.keep_run(i)) for i, layer in enumerate(text.layers)]
        self.handles.append(text.message_input.register_forward_hook(self.keep_message))
        for index, layer in enumerate(text.layers):
            for gate in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
                if hasattr(layer, gate):
                    self.handles.append(getattr(layer, gate).register_forward_hook(self.keep_gate(index, gate)))
        self.handles.append(text.norm.register_forward_hook(self.keep_norm))
        self.final_norm = None

    def keep_run(self, layer):
        def hook(_module, _args, output):
            row = output[0] if isinstance(output, tuple) else output
            self.runs.append((layer, row[0].detach().float().clone()))

        return hook

    def keep_message(self, _module, _args, output):
        self.messages.append(output[0].detach().float().clone())

    def keep_gate(self, layer, gate):
        def hook(_module, _args, output):
            self.gates.append((layer, gate, output[0].detach().float().clone()))

        return hook

    def keep_norm(self, _module, _args, output):
        self.final_norm = output[0].detach().float().clone()

    def remove(self):
        for handle in self.handles:
            handle.remove()


def run_layers(model, text, tokenizer, out, facts):
    """The prompt "plain" through Mini at 3 cycles: each run in order, each message and each gate."""
    ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    tap = RunTap(text)
    logits = forward(model, ids)
    tap.remove()
    check(len(tap.runs) == 48, f"48 runs at 3 cycles, not {len(tap.runs)}")
    order = [layer for layer, _row in tap.runs]
    expected = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
    check(order == expected, f"the order of the runs is {order}")
    check(len(tap.messages) == 24, f"24 messages at 3 cycles, not {len(tap.messages)}")
    check(len(tap.gates) == 96, f"96 gates at 3 cycles, not {len(tap.gates)}")
    head = model.lm_head(tap.final_norm.to(model.lm_head.weight.dtype)).float()
    head = model.logit_softcap(head)
    check(max_abs(head[-1], logits[-1]) == 0.0, "the final norm, the head and the softcap give the logits")
    tensors = {"ids": torch.tensor(ids), "final_norm": tap.final_norm, "logits_last": logits[-1]}
    for run, (layer, row) in enumerate(tap.runs):
        tensors[f"run{run:02}"] = row
    for index, message in enumerate(tap.messages):
        tensors[f"message{index:02}"] = message
    for index, (layer, gate, row) in enumerate(tap.gates):
        tensors[f"gate{index:02}.{gate}"] = row
    meta = {"prompt": "The capital of France is", "run_layers": order, "versions": versions()}
    save(tensors, out / "inkling2-mini.layers.safetensors", meta)
    facts["layers.positions"] = len(ids)
    log(f"wrote layers: {len(tap.runs)} runs, {len(tap.messages)} messages, {len(tap.gates)} gates")


def run_cycles(model, text, tokenizer, label, out, facts):
    """The prompts "plain" and "chat" at 1 to 5 cycles."""
    prompts = {"plain": "The capital of France is", "chat": chat_text(tokenizer)}
    entries = []
    for count in range(1, 6):
        set_cycles(text, count)
        for name, prompt in prompts.items():
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            logits = forward(model, ids)
            entry = top_entry(f"{name}.cycles{count}", "fixed in this tool", prompt, ids, logits[-1])
            entry["cycles"] = count
            entries.append(entry)
            log(f"{label} {name} at {count} cycles: top {tokenizer.decode([entry['top_ids'][0]])!r}")
    set_cycles(text, 3)
    path = out / f"{label}.cycles.json"
    path.write_text(json.dumps({"model": label, "versions": versions(), "prompts": entries}, indent=1) + "\n")
    facts[f"{label}.cycles"] = [1, 2, 3, 4, 5]
    log(f"wrote {path}")


def run_decode(model, tokenizer, label, out, facts):
    """8 greedy tokens of "chat" with the cache of the reference. The logits match a forward with no cache."""
    ids = tokenizer.encode(chat_text(tokenizer), add_special_tokens=False)
    with torch.no_grad():
        generated = model.generate(
            input_ids=torch.tensor([ids]),
            max_new_tokens=DECODE_STEPS,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
    tokens = generated.sequences[0, len(ids) :].tolist()
    check(len(tokens) == DECODE_STEPS, f"{DECODE_STEPS} greedy tokens, not {len(tokens)}")
    cached = torch.stack([row[0].float() for row in generated.logits])
    whole = forward(model, ids + tokens)[len(ids) - 1 : len(ids) - 1 + DECODE_STEPS]
    gap = max_abs(cached, whole)
    log(f"{label} decode: the cache against no cache, max abs {gap:.3g}")
    check(gap <= 1e-3, f"the cache gives the logits of no cache: max abs {gap:.3g}")
    check(whole.argmax(-1).tolist() == tokens, "the greedy tokens are the argmax of the logits with no cache")
    tensors = {"ids": torch.tensor(ids), "tokens": torch.tensor(tokens), "logits": whole}
    meta = {"prompt": chat_text(tokenizer), "cache_gap": gap, "versions": versions()}
    save(tensors, out / f"{label}.decode.safetensors", meta)
    facts[f"{label}.decode.cache_gap"] = gap
    facts[f"{label}.decode.text"] = tokenizer.decode(tokens)


def synthetic_image(width, height, seed):
    """An RGB image of `height x width x 3` bytes: smooth gradients with noise, from a fixed seed."""
    g = torch.Generator().manual_seed(seed)
    y = torch.linspace(0, 1, height)[:, None, None]
    x = torch.linspace(0, 1, width)[None, :, None]
    phase = torch.rand(3, generator=g)
    base = 127.5 + 100 * torch.sin(6.2832 * (x * (1 + phase) + y * (2 - phase) + phase))
    noise = torch.randint(-27, 28, (height, width, 3), generator=g).float()
    return (base + noise).round().clamp(0, 255).to(torch.uint8)


class VisionTap:
    """Keeps the outputs of the parts of the vision tower for each call of the tower."""

    def __init__(self, tower):
        self.rows = []
        self.handles = [
            tower.embeddings.patch_embedding.register_forward_hook(self.keep("patch_embed")),
            tower.embeddings.register_forward_hook(self.keep("embeddings")),
            tower.pre_layernorm.register_forward_hook(self.keep("pre_layernorm")),
            tower.merger.register_forward_hook(self.keep("projector")),
        ]
        for index, layer in enumerate(tower.layers):
            self.handles.append(layer.register_forward_hook(self.keep(f"layer{index:02}")))
        self.handles.append(tower.register_forward_hook(self.keep_tower))

    def keep(self, name):
        def hook(_module, _args, output):
            row = output[0] if isinstance(output, tuple) else output
            self.rows.append((name, row.detach().float().clone()))

        return hook

    def keep_tower(self, _module, _args, output):
        self.rows.append(("tower", output.last_hidden_state.detach().float().clone()))

    def named(self):
        return dict(self.rows)

    def remove(self):
        for handle in self.handles:
            handle.remove()


class EmbedsTap:
    """Keeps the input rows of the text model: the rows after the scatter of the image features."""

    def __init__(self, text):
        self.rows = None
        self.handle = text.register_forward_pre_hook(self.keep, with_kwargs=True)

    def keep(self, _module, _args, kwargs):
        self.rows = kwargs["inputs_embeds"][0].detach().float().clone()

    def remove(self):
        self.handle.remove()


def run_vision(model, text, folder, out, facts):
    """Two images through the processor, the tower and the projector, and a chat prompt with both through Mini."""
    import numpy
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(folder)
    tower = model.model.vision_tower
    vcfg = model.config.vision_config
    merge = vcfg.spatial_merge_size
    image_token = model.config.image_token_id
    check(processor.image_token_id == image_token, "the processor and the model name one image token")
    check(tower.embeddings.patch_embedding.weight.dtype == torch.float32, "the tower runs in f32")
    tensors, images, records = {}, [], []
    for index, (width, height, seed) in enumerate(VISION_IMAGES):
        rgb = synthetic_image(width, height, seed)
        png = out / f"inkling2-vision-image{index}.png"
        Image.fromarray(rgb.numpy()).save(png)
        image = Image.open(png)
        image.load()
        check(torch.equal(torch.from_numpy(numpy.array(image)), rgb), "the PNG holds the bytes")
        images.append(image)
        features = processor.image_processor(images=[image], return_tensors="pt")
        pixels, grid = features["pixel_values"], features["image_grid_thw"]
        tap = VisionTap(tower)
        with torch.no_grad():
            result = tower(pixel_values=pixels, image_grid_thw=grid)
        tap.remove()
        rows = tap.named()
        check(torch.equal(rows["tower"], result.last_hidden_state.float()), "the tap holds the tower output")
        check(torch.equal(rows["projector"], result.pooler_output.float()), "the tap holds the projector rows")
        tokens = int(grid.prod())
        check(rows["projector"].shape == (tokens, vcfg.text_hidden_size), "one projector row for each token")
        check(rows["tower"].shape[0] == tokens * merge * merge + 1, "one class row and the patches")
        prefix = f"image{index}."
        tensors[prefix + "rgb"] = rgb.to(torch.int64)
        tensors[prefix + "pixel_values"] = pixels.float()
        tensors[prefix + "grid"] = grid[0].to(torch.int64)
        for name in ("patch_embed", "embeddings", "pre_layernorm", "tower", "projector"):
            tensors[prefix + name] = rows[name]
        if index == 0:
            for layer in range(vcfg.num_hidden_layers):
                tensors[prefix + f"layer{layer:02}"] = rows[f"layer{layer:02}"]
        records.append({"width": width, "height": height, "seed": seed, "grid": grid[0].tolist(), "png": png.name})
        log(f"vision image {index}: {width}x{height}, grid {grid[0].tolist()}, {tokens} placeholders")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "text", "text": VISION_PROMPT},
            ],
        }
    ]
    rendered = processor.apply_chat_template(messages, tokenize=False, reasoning_effort="none")
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt", reasoning_effort="none"
    )
    ids = inputs["input_ids"][0]
    counts = [int(g.prod()) for g in inputs["image_grid_thw"]]
    check(int((ids == image_token).sum()) == sum(counts), "the count of placeholders equals the projector rows")
    for index, count in enumerate(counts):
        check(count == tensors[f"image{index}.projector"].shape[0], f"image {index} has {count} placeholders")
    pixels = torch.cat([tensors[f"image{i}.pixel_values"] for i in (0, 1)])
    check(torch.equal(inputs["pixel_values"].float(), pixels), "the processor gives the same pixels for the chat")
    embeds = EmbedsTap(text)
    tap = VisionTap(tower)
    before = ExpertCalls.count
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            use_cache=False,
        ).logits[0].float()
    check(ExpertCalls.count > before, "the f32 expert call ran")
    embeds.remove()
    tap.remove()
    joint = tap.named()["projector"]
    separate = torch.cat([tensors["image0.projector"], tensors["image1.projector"]])
    joint_gap = max_abs(joint, separate)
    log(f"vision: one tower run of both images against two runs, max abs {joint_gap:.3g}")
    check(joint_gap <= 1e-4, f"a tower run of both images gives the rows of two runs: {joint_gap:.3g}")
    mask = ids == image_token
    check(torch.equal(embeds.rows[mask], joint), "the scattered rows are the projector rows")
    with torch.no_grad():
        plain = text.embed_tokens(ids[None])[0].float()
    check(torch.equal(embeds.rows[~mask], plain[~mask]), "the other rows are the normed embeddings")
    tensors["prompt.ids"] = ids.to(torch.int64)
    tensors["prompt.logits_last"] = logits[-1]
    meta = {
        "images": records,
        "prompt": rendered,
        "rope_parameters": vcfg.rope_parameters,
        "hidden_act": vcfg.hidden_act,
        "attn_implementation": vcfg._attn_implementation,
        "joint_gap": joint_gap,
        "versions": versions(),
    }
    save(tensors, out / "inkling2-mini.vision.safetensors", meta)
    source = "two synthetic images of this tool and a fixed question"
    entry = top_entry("vision", source, rendered, ids.tolist(), logits[-1])
    entry["images"] = records
    fixture = {
        "family": "inkling2",
        "model": "inkling2-mini",
        "reference": "transformers inkling2_loop 046bdb8cc2, f32 sums with bf16 weights, CPU",
        "source": source_record(folder),
        "versions": versions(),
        "prompts": [entry],
    }
    (out / "inkling2-mini.vision.json").write_text(json.dumps(fixture, indent=1) + "\n")
    facts["vision.placeholders"] = counts
    facts["vision.prompt_tokens"] = len(ids)
    facts["vision.joint_gap"] = joint_gap
    facts["vision.top"] = processor.tokenizer.decode([entry["top_ids"][0]])
    log(f"wrote vision: {len(ids)} prompt tokens, top {facts['vision.top']!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mini", required=True, help="the snapshot folder of Inkling 2 Mini")
    parser.add_argument("--varloop", required=True, help="the snapshot folder of the variable-loop model")
    parser.add_argument("--out", required=True, help="the folder of the reference files")
    parser.add_argument("--stages", default=",".join(STAGES), help="the stages, with commas between them")
    parser.add_argument("--threads", type=int, default=24, help="the CPU threads of torch")
    args = parser.parse_args()
    stages = args.stages.split(",")
    for stage in stages:
        check(stage in STAGES, f"the stage {stage} is one of {STAGES}")
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # The stage "vision" keeps its facts in a file of its own, so that it writes no file of the other stages.
    facts_name = "vision.facts.json" if stages == ["vision"] else "facts.json"
    facts_path = out / facts_name
    facts = json.loads(facts_path.read_text()) if facts_path.exists() else {}
    for label, folder in (("inkling2-mini", args.mini), ("inkling2-varloop", args.varloop)):
        wanted = [s for s in stages if s not in ("layers", "vision") or label == "inkling2-mini"]
        if not wanted:
            continue
        model, text, tokenizer = load(folder)
        if "text" in wanted:
            run_text(model, tokenizer, label, folder, out, facts)
        if "layers" in wanted:
            run_layers(model, text, tokenizer, out, facts)
        if "cycles" in wanted:
            run_cycles(model, text, tokenizer, label, out, facts)
        if "decode" in wanted:
            run_decode(model, tokenizer, label, out, facts)
        if "vision" in wanted:
            run_vision(model, text, folder, out, facts)
        del model, text
        facts_path.write_text(json.dumps(facts, indent=1, sort_keys=True) + "\n")
    log("done")


if __name__ == "__main__":
    main()
