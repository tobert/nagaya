# SPDX-License-Identifier: MIT
"""Make the tokenizer fixture of the SDXL family library (spec 11.9a; milestone 18 step 6).

The tool writes crates/nagaya-text-sdxl/tests/fixtures/ids.json. For each fixed string it holds the ids of both
tokenizers of the checkpoint, with no start, end or pad token. It also holds one string padded to 77 ids by each
tokenizer. The reference is the `CLIPTokenizer` that diffusers loads for the pipeline (transformers 5.21, backed by
`tokenizers`). The two tokenizers have one vocabulary, but `tokenizer_2` has the pad token `!`, so it reads a `!` in
the text as id 0.

The tool reads only the tokenizer files. It uses no GPU and little RAM. Run it with the venv of the reference:

    HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ~/.venvs/sdxl-ref/bin/python -I tools/sdxl_family_fixture.py \
        --model DIR [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "crates/nagaya-text-sdxl/tests/fixtures"

STRINGS = [
    "",
    "a red fox in a snowy forest, watercolor",
    "A Fox, café!",
    "masterpiece, best quality, 1girl, solo, looking at viewer",
    "score_9, score_8_up, score_7_up, source_anime",
    "blurry, low quality, worst quality, jpeg artifacts",
    "photo of a cat   sitting\non a\ttable",
    "UPPER CASE and MiXeD case",
    "don't stop, it's the cat's toy, we'll see, they've gone, I'm here, you'd know, we're done",
    "numbers 1234 and 3.14 and 1,000,000",
    "punctuation!!! ?? ... --- ;; :: (( )) [[ ]] {{ }}",
    "hyphen-ated words and under_scored_words",
    "emoji 🦊🌲❄️ and symbols ©®™ and ½ ¼",
    "naïve résumé façade Ångström",
    "東京の夜景, ネオン",
    "Ελληνικά and русский текст",
    "<|startoftext|> written in text <|endoftext|>",
    "a!b!c",
    "!",
    "trailing space ",
    " leading space",
    "8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
    "(masterpiece:1.2), [blurry], ((detailed))",
    "a photo of an astronaut riding a horse on mars",
    "supercalifragilisticexpialidocious",
    "https://example.com/path?q=1",
    " non-breaking spaces　here",
    "Ｆｕｌｌｗｉｄｔｈ letters",
    "é decomposed accent",
    "tab\tand\r\nnewline",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="the SDXL base 1.0 checkpoint directory")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    from transformers import CLIPTokenizer
    import transformers

    tok1 = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer_2")
    entries = []
    for text in STRINGS:
        ids1 = tok1(text, add_special_tokens=False).input_ids
        ids2 = tok2(text, add_special_tokens=False).input_ids
        entries.append({"text": text, "ids1": ids1, "ids2": ids2})
    padded_text = "a red fox"
    padded = {
        "text": padded_text,
        "ids1": tok1(padded_text, padding="max_length", max_length=77, truncation=True).input_ids,
        "ids2": tok2(padded_text, padding="max_length", max_length=77, truncation=True).input_ids,
    }
    special = {
        "bos": tok1.bos_token_id,
        "eos": tok1.eos_token_id,
        "pad1": tok1.pad_token_id,
        "pad2": tok2.pad_token_id,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "reference": f"transformers {transformers.__version__} CLIPTokenizer",
        "special": special,
        "strings": entries,
        "padded": padded,
    }
    (out / "ids.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    differ = sum(1 for e in entries if e["ids1"] != e["ids2"])
    print(f"wrote {out / 'ids.json'}: {len(entries)} strings, {differ} where the tokenizers differ")


if __name__ == "__main__":
    main()
