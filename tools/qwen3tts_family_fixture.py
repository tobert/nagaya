# SPDX-License-Identifier: MIT
"""Make the fixtures of the Qwen3-TTS family library (spec 11.3, 11.8; milestone 17 step 3).

The tool writes two files into crates/nagaya-text-qwen3tts/tests/fixtures:

- ids.json: for each fixed string, the token ids of the reference processor. The processor is the one that
  `Qwen3TTSModel.from_pretrained` loads, with `fix_mistral_regex=True`. Each entry also says if the Qwen2 split
  pattern gives other ids.
- prompts.json: for each fixed request, the prompt positions as pairs of a text id and a codec id. Each position
  also holds a digest of the reference `talker_input_embeds` row: the sum and the first four values, in f32.

The tool builds the pairs with the rule of the Rust library. It then checks that each pair gives the reference
row: text_projection(text_embedding(text id)) + codec_embedding(codec id). A row that does not agree stops the tool.

The tool loads the VoiceDesign model on the CPU in f32 (about 8 GB of RAM). It uses no GPU. Run it with the venv
of the reference:

    PYTHONDONTWRITEBYTECODE=1 nice -n 10 <reference venv>/bin/python -I tools/qwen3tts_family_fixture.py \
        --model DIR [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "crates/nagaya-text-qwen3tts/tests/fixtures"

# The split pattern of the Qwen2 tokenizer, before transformers replaces it.
QWEN2_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

STRINGS = [
    "",
    "hello",
    " hello",
    "Hello, world!",
    "  two  spaces  ",
    "line one\nline two\n\n\nthree",
    "\t tab and space\r\nCRLF",
    "trailing spaces   ",
    "numbers 1234567 and 3.14159, -42",
    "punctuation: !?;:...--()[]{}<>\"'`~@#$%^&*_+=|\\/",
    "don't you'll we've I'M THEY'RE",
    "CamelCase HTTPServer iPhone XMLHttpRequest",
    "a/b/c path/to/file.txt\n/",
    "naïve café – “quotes” — ellipsis…",
    "decomposed: café naïve",
    "emoji 🦀🔥 and a flag 🇯🇵 and 👩‍👩‍👧",
    "我叫通义千问，是阿里云的开源大模型。",
    "日本語のテキスト、そしてカタカナ。",
    "안녕하세요, 세계!",
    "नमस्ते दुनिया",
    "Здравствуй, мир!",
    "<|im_start|>assistant\nHi<|im_end|>\n",
    "<tts_pad><tts_text_bos><tts_text_eod><tts_text_bos_single>",
    "<think>x</think> <tool_call>y</tool_call> <|endoftext|>",
    "<|im_start|><|im_start|>",
    "<|im_startx|> <tts_pad",
    "\u0000\u0001\u001f control bytes",
    "  　 unicode spaces   sep",
]

# The requests of the prompt fixture: (name, text, instructions, language).
REQUESTS = [
    ("english_auto", "Hello there, how are you today?", "A calm male voice, low pitch, slow pace.", "auto"),
    ("english_named", "The quick brown fox jumps over the lazy dog.", "A bright young female voice.", "english"),
    ("chinese", "我叫通义千问，是阿里云的开源大模型。", "一个温柔的女声，语速适中。", "chinese"),
    ("japanese_no_instructions", "今日はいい天気ですね。", "", "Japanese"),
    ("leading_newline", "\nTwo lines\nof text.", "Whisper.", "auto"),
]


def pairs_for(tts, cfg, input_id, instruct_id, language):
    """The pair rule of the Rust library, written again in Python. It follows `generate` in non-streaming mode."""
    tc = cfg.talker_config
    out = []
    if instruct_id is not None:
        out += [[t, None] for t in instruct_id]
    out += [[t, None] for t in input_id[:3]]
    if language.lower() == "auto":
        prefill = [tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]
    else:
        lang = tc.codec_language_id[language.lower()]
        prefill = [tc.codec_think_id, tc.codec_think_bos_id, lang, tc.codec_think_eos_id]
    codec = prefill + [tc.codec_pad_id, tc.codec_bos_id]
    out += [[cfg.tts_pad_token_id, c] for c in codec[:-2]]
    out += [[cfg.tts_bos_token_id, codec[-2]]]
    out += [[t, tc.codec_pad_id] for t in input_id[3:-5]]
    out += [[cfg.tts_eos_token_id, tc.codec_pad_id]]
    out += [[cfg.tts_pad_token_id, tc.codec_bos_id]]
    return out


def digest(row):
    return {"sum": float(row.sum().item()), "first": [float(x) for x in row[:4].tolist()]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="the checkpoint directory of Qwen3-TTS 12Hz 1.7B VoiceDesign")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    import tokenizers
    import torch
    import transformers
    from qwen_tts import Qwen3TTSModel

    torch.manual_seed(0)
    model = Qwen3TTSModel.from_pretrained(args.model, device_map="cpu", dtype=torch.float32)
    proc = model.processor
    tok = proc.tokenizer
    cfg = model.model.config
    talker = model.model.talker

    # A second tokenizer with the Qwen2 pattern, to show where the replaced pattern changes the ids.
    qwen2 = tokenizers.Tokenizer.from_str(tok.backend_tokenizer.to_str())
    qwen2.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [
            tokenizers.pre_tokenizers.Split(tokenizers.Regex(QWEN2_PATTERN), behavior="isolated"),
            tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )

    def ids_of(text):
        return proc(text=text, return_tensors="pt")["input_ids"][0].tolist()

    entries = []
    for text in STRINGS:
        ids = ids_of(text)
        entries.append({"text": text, "ids": ids, "qwen2_pattern_differs": qwen2.encode(text).ids != ids})

    captured = {}

    class Captured(Exception):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise Captured()

    talker.generate = capture

    def pair_row(text_id, codec_id):
        row = talker.text_projection(talker.get_text_embeddings()(torch.tensor([[text_id]])))[0, 0]
        if codec_id is not None:
            row = row + talker.get_input_embeddings()(torch.tensor([[codec_id]]))[0, 0]
        return row

    prompts = []
    worst = 0.0
    with torch.no_grad():
        for name, text, instruct, language in REQUESTS:
            captured.clear()
            try:
                model.generate_voice_design(text=text, instruct=instruct, language=language)
            except Captured:
                pass
            embeds = captured["inputs_embeds"][0]
            trailing = captured["trailing_text_hidden"][0]
            input_id = ids_of(model._build_assistant_text(text))
            instruct_id = ids_of(model._build_instruct_text(instruct)) if instruct else None
            pairs = pairs_for(model, cfg, input_id, instruct_id, language)
            if len(pairs) != embeds.shape[0]:
                sys.exit(f"{name}: the rule gives {len(pairs)} positions, the reference {embeds.shape[0]}")
            positions = []
            for k, (text_id, codec_id) in enumerate(pairs):
                mine = pair_row(text_id, codec_id)
                diff = (mine - embeds[k]).abs().max().item()
                worst = max(worst, diff)
                if not torch.allclose(mine, embeds[k], rtol=1e-5, atol=1e-5):
                    sys.exit(f"{name}: position {k} ({text_id}, {codec_id}) differs by {diff}")
                positions.append({"text": text_id, "codec": codec_id, "embed": digest(embeds[k])})
            pad_row = pair_row(cfg.tts_pad_token_id, None)
            if trailing.shape[0] != 1 or not torch.allclose(trailing[0], pad_row, rtol=1e-5, atol=1e-5):
                sys.exit(f"{name}: the trailing text is not one tts_pad row")
            prompts.append(
                {
                    "name": name,
                    "text": text,
                    "instructions": instruct,
                    "language": language,
                    "positions": positions,
                }
            )

    tc = cfg.talker_config
    constants = {
        "codec_eos": tc.codec_eos_token_id,
        "trailing_text": cfg.tts_pad_token_id,
        "suppress": [i for i in range(tc.vocab_size - 1024, tc.vocab_size) if i != tc.codec_eos_token_id],
        "languages": sorted(model.get_supported_languages()),
    }

    versions = {
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "torch": torch.__version__,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ids.json", "w", encoding="utf-8") as f:
        json.dump({"versions": versions, "strings": entries}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(out / "prompts.json", "w", encoding="utf-8") as f:
        json.dump(
            {"versions": versions, "max_abs_diff": worst, "constants": constants, "requests": prompts},
            f,
            ensure_ascii=False,
            indent=1,
        )
        f.write("\n")
    print(f"wrote {len(entries)} strings and {len(prompts)} requests; the largest row difference is {worst}")


if __name__ == "__main__":
    main()
