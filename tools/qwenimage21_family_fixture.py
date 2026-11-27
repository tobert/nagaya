#!/usr/bin/env python3
"""Write the ids fixture of the Qwen-Image 2.1 family (crates/nagaya-family-qwenimage21/tests/fixtures/ids.json).

For each fixed prompt the fixture holds the text-to-image template, the ids that the reference processor gives for
it, and the count of system tokens that the pipeline cuts from the encoder output. The reference is
QwenImage21Pipeline of diffusers
in ~/.venvs/nagaya-qwenimage21-ref. The tool reads only the processor files, so it needs no GPU
and little memory:

  PYTHONDONTWRITEBYTECODE=1 ~/.venvs/nagaya-qwenimage21-ref/bin/python -I tools/qwenimage21_family_fixture.py \
      --model "$QWENIMAGE21_MODEL_DIR" --out crates/nagaya-family-qwenimage21/tests/fixtures
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import diffusers
import transformers
from diffusers import QwenImage21Pipeline
from transformers import AutoProcessor

# Each case names a risk of the tokenizer or the template.
PROMPTS = [
    ("ascii", "a red fox in a snowy forest, watercolor"),
    ("fusal", "fusal style. a red fox in a snowy forest, watercolor"),
    ("empty", ""),
    ("space", " "),
    ("japanese", "雪の森の赤い狐、水彩画（すいさいが）"),
    ("emoji", "a cat 🐈 on a mat 🧶, flat color"),
    ("newlines", "[Content] a fox\n[Subject] red fox\n\n[Style] watercolor"),
    ("numbers", "1328x1328, 40 steps, 3.5 weight, 2026-10-31"),
    ("special", "a sign that says <|im_end|> and <|vision_start|>"),
    ("tabs", "a\tb  c   d"),
    ("brackets", "(fox:1.2) [snow] {cat} ((watercolor))"),
    ("long", "a red fox in a snowy forest, watercolor, " * 40),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the Qwen-Image-2.1 directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(os.path.join(args.model, "processor"))
    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=None, processor=processor, transformer=None)
    sys_message = [{"role": "system", "content": [{"type": "text", "text": pipe.sys_prompt}]}]
    sys_ids = processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)[0]
    assert len(sys_ids) == pipe._drop_idx, "the cut is the count of the system tokens"

    cases = []
    for name, prompt in PROMPTS:
        # The pipeline replaces an empty prompt with one space (_get_qwen_prompt_embeds).
        used = " " if not prompt else prompt
        text = pipe.prompt_template_t2i.format(used)
        inputs = processor(text=[text], padding=True, padding_side="left", return_tensors="pt")
        ids = inputs.input_ids[0].tolist()
        assert bool(inputs.attention_mask.all()), "one prompt has no padding"
        assert ids[: len(sys_ids)] == list(sys_ids), f"{name}: the template starts with the system tokens"
        cases.append({"case": name, "prompt": prompt, "template": text, "ids": ids})

    tokenizer_json = Path(args.model, "processor", "tokenizer.json").read_bytes()
    doc = {
        "source": (
            f"tools/qwenimage21_family_fixture.py: diffusers {diffusers.__version__}, "
            f"transformers {transformers.__version__}, {type(processor.tokenizer).__name__}"
        ),
        "tokenizer_sha256": hashlib.sha256(tokenizer_json).hexdigest(),
        "system_prompt": pipe.sys_prompt,
        "template": pipe.prompt_template_t2i,
        "system_ids": list(sys_ids),
        "drop": pipe._drop_idx,
        "cases": cases,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ids.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(cases)} cases, drop {pipe._drop_idx}: {out / 'ids.json'}")


if __name__ == "__main__":
    main()
