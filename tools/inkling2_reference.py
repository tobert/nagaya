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

The option --quant runs the quantized mode: each expert quantizes its input rows as the checkpoint requires, and the
files get the infix ".quant" (`inkling2-mini.quant.oracle.json`, `inkling2-mini.quant.layers.safetensors`, and so on),
so they never replace the f32 files. transformers quantizes no activation, so `tools/inkling2_quant.py` holds the rules
of vLLM commit 186882d9e8 (the same rules as `crates/nagaya-mk-inkling2-sm121/src/quant.rs`). The layer split comes from
`hf_quant_config.json` of the model folder: NVFP4 (block 16, `fouroversix_mae`) for the layers of Mini 0 to 19, and
MXFP8 (block 32, ue8m0) for the layers 20 to 23. Two rows are quantized in each expert call, as vLLM does them:

  - the input of the gate and up projection: the row of each token, once for all its experts (the router sees the
    unquantized row), with K = the hidden size 2304 (`ops/moe.py` `_quantize_x`, `route`);
  - the input of the down projection: the row of each token and expert slot, after the softcapped SwiGLU AND after the
    routing weight (`ops/activation.py` `_situ_quantize_kernel`), with K = 1152. Thus the quantized mode multiplies
    the routing weight before the down projection. The f32 mode multiplies after it, as transformers does.

The reference keeps f32: a quantized row is expanded to f32 (NVFP4: `e2m1 * block_scale * (amax * (1/1536))`; MXFP8:
`e4m3 * 2^(byte - 127)`), and the f32 linear runs with the expanded weights. The weights hold their exact values, so the
product equals the quantized dot product up to the order of the f32 sums. vLLM also rounds the gate and up output and
the SwiGLU result to bf16 before it quantizes. This tool does not, because the trunk of the engine is f32.

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
  - with --quant, the quantizer runs in every expert call of every layer, on the input row and on the down input
    (`ExpertCalls.verify`);
  - each prompt fits in MAX_TOKENS tokens;
  - the hooks see 24 + 2 * 12 = 48 runs of the decoder layers at 3 cycles;
  - the final norm of the hooks, the head and the softcap give the logits of the model;
  - a decode with the cache gives the logits of a forward over the whole sequence with no cache. With --quant the
    forward replays the quantized rows of the decode (`QuantTape`): a quantizer is a step function, so two orders of
    the f32 sums can flip a code, and a free comparison would measure the flips and not the cache;
  - the count of placeholders of an image equals its count of projector rows;
  - the rows that the model scatters into the placeholders are the projector rows, and the other rows are the
    normed embeddings of their tokens;
  - the hooks of the tower give its output, and a tower run of both images gives the rows of two runs of one.
"""

import argparse
import collections
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
REFERENCE = "transformers inkling2_loop 046bdb8cc2, f32 sums with bf16 weights, CPU"
STARTED = time.time()
# The infix of every file name that this run writes: ".quant" in the quantized mode, so that the f32 files stay.
TAG = ""


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
        "activation_formats": "".join(f[0] for f in ExpertCalls.formats) if ExpertCalls.formats else "none",
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


def quant_module():
    """Loads tools/inkling2_quant.py by its path, because the tool runs with `-I` and has no script folder on its path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("inkling2_quant", ROOT / "tools/inkling2_quant.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuantTape:
    """The quantized rows of the cache run, which the run with no cache replays.

    A quantizer is a step function: two f32 sums that differ in the last bit can fall on two sides of an edge and
    flip a code. The cache run and the run with no cache sum in another order, so a free comparison of their logits
    measures the flips, not the cache. The tape fixes the codes: the second run takes each quantized row from the
    first run, and the remaining gap is the order of the f32 sums only.

    A row has the key (kind, visit, position, slot). A visit is the place of a call in one forward pass, from 0 to
    `visits - 1`, and the same visit has the same layer in each pass. The position of a row is the count of rows that
    the earlier calls of that visit had, plus its place in the call. The input row and the route of a token have the slot -1.
    A route holds the expert ids and the routing weights of the token.
    """

    def __init__(self, visits):
        self.visits = visits
        self.rows = {}
        self.replay = False
        self.flips = 0
        self.start_run()

    def start_run(self):
        self.calls = 0
        self.seen = collections.Counter()

    def begin_call(self, count):
        """Returns the visit and the position of the first row of this call, for a call of `count` rows."""
        visit = self.calls % self.visits
        self.calls += 1
        first = self.seen[visit]
        self.seen[visit] += count
        return visit, first

    def rows_of(self, kind, visit, first, token, slot, fresh):
        """Records the quantized rows `fresh` (one for each entry of `token` and `slot`), or replays the recorded rows.

        A replay counts in `flips` the rows that the free quantizer would have changed. The outer scale of NVFP4 moves
        by a few ulp with the order of the sums, so a row counts only when a value moves by more than 1e-5 of its amax.
        """
        keys = [(kind, visit, first + int(t), int(s)) for t, s in zip(token, slot)]
        if not self.replay:
            for key, row in zip(keys, fresh):
                check(key not in self.rows, f"the tape has the row {key} twice")
                self.rows[key] = row.clone()
            return fresh
        missing = [key for key in keys if key not in self.rows]
        check(not missing, f"the run with no cache has the rows {missing[:3]}, which the cache run did not have")
        taped = torch.stack([self.rows[key] for key in keys])
        moved = (taped - fresh).abs().amax(dim=1) > 1e-5 * taped.abs().amax(dim=1)
        self.flips += int(moved.sum())
        return taped


class ExpertCalls:
    """The expert call of the reference, with every sum in f32. It counts its calls for a self check.

    An expert module with `activation_format` "nvfp4" or "mxfp8" quantizes its input rows (see the docstring of the
    tool). A module with no format runs the f32 call of transformers. `formats` holds the format of each layer in the
    quantized mode, and None in the f32 mode.
    """

    count = 0
    formats = None
    quant = None
    tape = None
    calls = collections.Counter()
    inputs = collections.Counter()
    downs = collections.Counter()

    @staticmethod
    def forward_f32(self, hidden_states, top_k_index, top_k_weights):
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

    @staticmethod
    def forward_quantized(self, hidden_states, top_k_index, top_k_weights):
        """The quantized call. The input row is quantized once for each token. The down input is quantized for each
        token and slot, after the routing weight, as vLLM does it."""
        fmt, layer = self.activation_format, self.layer_index
        final = torch.zeros_like(hidden_states, dtype=torch.float32)
        if ExpertCalls.quant is None:
            ExpertCalls.quant = quant_module()
        tape = ExpertCalls.tape
        if tape:
            visit, first = tape.begin_call(hidden_states.shape[0])
            every = torch.arange(hidden_states.shape[0])
            # The router is a step function too: a near tie between two experts flips with the order of the sums.
            route = torch.cat([top_k_index.float(), top_k_weights.float()], dim=1)
            route = tape.rows_of("route", visit, first, every, [-1] * len(every), route)
            top_k_index = route[:, : top_k_index.shape[1]].long()
            top_k_weights = route[:, top_k_index.shape[1] :].to(top_k_weights.dtype)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        rows = ExpertCalls.quant.fake_quantize(hidden_states.float(), fmt)
        if tape:
            rows = tape.rows_of("input", visit, first, every, [-1] * len(every), rows)
        ExpertCalls.inputs[layer] += 1
        for expert in hit:
            expert = expert[0]
            if expert == self.num_experts:
                continue
            position, token = torch.where(mask[expert])
            state = self._apply_gate(F.linear(rows[token], self.gate_up_proj[expert].float()))
            state = state.float() * top_k_weights[token, position, None]
            state = ExpertCalls.quant.fake_quantize(state, fmt)
            if tape:
                state = tape.rows_of("down", visit, first, token.tolist(), position.tolist(), state)
            ExpertCalls.downs[layer] += 1
            state = F.linear(state, self.down_proj[expert].float())
            final.index_add_(0, token, state.float())
        return final.to(hidden_states.dtype)

    @staticmethod
    def forward(self, hidden_states, top_k_index, top_k_weights):
        ExpertCalls.count += 1
        ExpertCalls.calls[getattr(self, "layer_index", -1)] += 1
        if getattr(self, "activation_format", None) is None:
            return ExpertCalls.forward_f32(self, hidden_states, top_k_index, top_k_weights)
        return ExpertCalls.forward_quantized(self, hidden_states, top_k_index, top_k_weights)

    @staticmethod
    def snapshot():
        return {
            "count": ExpertCalls.count,
            "calls": dict(ExpertCalls.calls),
            "inputs": dict(ExpertCalls.inputs),
            "downs": dict(ExpertCalls.downs),
        }

    @staticmethod
    def verify(before):
        """The self check after a forward: the expert call ran, and in the quantized mode each layer ran it and the
        quantizer ran in each call, on the input row and on the down input."""
        check(ExpertCalls.count > before["count"], "the f32 expert call ran")
        if ExpertCalls.formats is None:
            return
        for layer in range(len(ExpertCalls.formats)):
            calls = ExpertCalls.calls[layer] - before["calls"].get(layer, 0)
            inputs = ExpertCalls.inputs[layer] - before["inputs"].get(layer, 0)
            downs = ExpertCalls.downs[layer] - before["downs"].get(layer, 0)
            check(calls > 0, f"the expert call ran in layer {layer}")
            check(inputs == calls, f"layer {layer} quantized its input row in each of {calls} calls, not {inputs}")
            check(downs >= calls, f"layer {layer} quantized its down input in each of {calls} calls, not {downs}")


def load(folder, quantize):
    """Loads one model in bf16, then moves every parameter but the experts to f32.

    With `quantize`, each expert module gets its layer and its activation format from `hf_quant_config.json`.
    """
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
    ExpertCalls.formats = None
    if quantize:
        ExpertCalls.quant = ExpertCalls.quant or quant_module()
        formats = ExpertCalls.quant.layer_formats(folder, len(text.layers))
        for layer, fmt in enumerate(formats):
            experts = text.layers[layer].mlp.experts
            check(isinstance(experts, modeling.Inkling2LoopExperts), f"layer {layer} has the expert module")
            experts.layer_index, experts.activation_format = layer, fmt
        ExpertCalls.formats = formats
        log(f"quantized mode: activation formats {''.join(f[0] for f in formats)} (n = nvfp4, m = mxfp8)")
    log(f"loaded {folder}, {model.config.text_config.loop_num_passes} cycles")
    return model, text, tokenizer


def set_cycles(text, count):
    """The reference reads the count of cycles from its config at each forward."""
    text.config.loop_num_passes = count


def forward(model, ids):
    """The logits of every position of one sequence, with no cache, in f32."""
    before = ExpertCalls.snapshot()
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
    ExpertCalls.verify(before)
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
        "reference": REFERENCE + (" + quantized expert activations" if TAG else ""),
        "source": source_record(folder),
        "versions": versions(),
        "prompts": entries,
    }
    path = out / f"{label}{TAG}.oracle.json"
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
    save(tensors, out / f"inkling2-mini{TAG}.layers.safetensors", meta)
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
    path = out / f"{label}{TAG}.cycles.json"
    path.write_text(json.dumps({"model": label, "versions": versions(), "prompts": entries}, indent=1) + "\n")
    facts[f"{label}.cycles"] = [1, 2, 3, 4, 5]
    log(f"wrote {path}")


def decode_compare(model, ids, steps, visits):
    """Greedy decode with the cache, and the same positions in a forward with no cache.

    Returns the tokens, the cached logits `[steps, vocab]`, the logits with no cache, and the count of quantized rows
    that the second run changed.
    """
    before = ExpertCalls.snapshot()
    tape = ExpertCalls.tape = QuantTape(visits) if ExpertCalls.formats else None
    with torch.no_grad():
        generated = model.generate(
            input_ids=torch.tensor([ids]),
            max_new_tokens=steps,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
    ExpertCalls.verify(before)
    tokens = generated.sequences[0, len(ids) :].tolist()
    cached = torch.stack([row[0].float() for row in generated.logits])
    if tape is None:
        whole = forward(model, ids + tokens)[len(ids) - 1 : len(ids) - 1 + steps]
        return tokens, cached, whole, 0
    # The cache run fed the ids and the first steps - 1 tokens. The last token has no taped row.
    check(tape.calls == visits * steps, f"the cache run made {tape.calls} expert calls, not {visits} for each of {steps} forwards")
    tape.replay = True
    tape.start_run()
    try:
        whole = forward(model, ids + tokens[:-1])[len(ids) - 1 :]
    finally:
        ExpertCalls.tape = None
    check(tape.calls == visits, f"the run with no cache made {tape.calls} expert calls, not {visits}")
    return tokens, cached, whole, tape.flips


def run_decode(model, tokenizer, label, out, facts):
    """8 greedy tokens of "chat" with the cache of the reference. The logits match a forward with no cache."""
    ids = tokenizer.encode(chat_text(tokenizer), add_special_tokens=False)
    text_config = model.config.text_config
    tokens, cached, whole, flips = decode_compare(model, ids, DECODE_STEPS, text_config.total_num_hidden_layers)
    check(len(tokens) == DECODE_STEPS, f"{DECODE_STEPS} greedy tokens, not {len(tokens)}")
    gap = max_abs(cached, whole)
    log(f"{label} decode: the cache against no cache, max abs {gap:.3g}, {flips} quantized rows replayed that differ")
    check(gap <= 1e-3, f"the cache gives the logits of no cache: max abs {gap:.3g}, limit 0.001")
    check(whole.argmax(-1).tolist() == tokens, "the greedy tokens are the argmax of the logits with no cache")
    tensors = {"ids": torch.tensor(ids), "tokens": torch.tensor(tokens), "logits": whole}
    meta = {"prompt": chat_text(tokenizer), "cache_gap": gap, "replayed_flips": flips, "versions": versions()}
    save(tensors, out / f"{label}{TAG}.decode.safetensors", meta)
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
        png = out / f"inkling2-vision-image{index}{TAG}.png"
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
    before = ExpertCalls.snapshot()
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            use_cache=False,
        ).logits[0].float()
    ExpertCalls.verify(before)
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
    save(tensors, out / f"inkling2-mini{TAG}.vision.safetensors", meta)
    source = "two synthetic images of this tool and a fixed question"
    entry = top_entry("vision", source, rendered, ids.tolist(), logits[-1])
    entry["images"] = records
    fixture = {
        "family": "inkling2",
        "model": "inkling2-mini",
        "reference": REFERENCE + (" + quantized expert activations" if TAG else ""),
        "source": source_record(folder),
        "versions": versions(),
        "prompts": [entry],
    }
    (out / f"inkling2-mini{TAG}.vision.json").write_text(json.dumps(fixture, indent=1) + "\n")
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
    parser.add_argument("--quant", action="store_true", help="quantize the expert activations; write files with .quant")
    args = parser.parse_args()
    global TAG
    TAG = ".quant" if args.quant else ""
    stages = args.stages.split(",")
    for stage in stages:
        check(stage in STAGES, f"the stage {stage} is one of {STAGES}")
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # The stage "vision" keeps its facts in a file of its own, so that it writes no file of the other stages.
    facts_name = f"{TAG.lstrip('.')}{'.' if TAG else ''}{'vision.' if stages == ['vision'] else ''}facts.json"
    facts_path = out / facts_name
    facts = json.loads(facts_path.read_text()) if facts_path.exists() else {}
    for label, folder in (("inkling2-mini", args.mini), ("inkling2-varloop", args.varloop)):
        wanted = [s for s in stages if s not in ("layers", "vision") or label == "inkling2-mini"]
        if not wanted:
            continue
        model, text, tokenizer = load(folder, args.quant)
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
