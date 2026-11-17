# SPDX-License-Identifier: MIT
"""Make the text fixtures of the Gemma 4 family library (spec 11.3, milestone 15s step 3).

The tool writes two files into crates/nagaya-text-gemma4/tests/fixtures:

- ids.json: for each fixed string, the token ids of Hugging Face `tokenizers` with the checkpoint's tokenizer.json,
  with special tokens matched and with special tokens as text, and the text that `decode` gives back.
- chat.json: for each fixed case, the text of Google's chat template as transformers renders it
  (`_compile_jinja_template`, the environment of `apply_chat_template`), and the token ids of that text. A case that
  the template refuses holds the error message.

The tool reads the checkpoint and the template, and it writes only the two fixtures. It uses no GPU. Run it with the
reference venv:

    PYTHONDONTWRITEBYTECODE=1 nice -n 10 ~/.venvs/nagaya-ref/bin/python -I tools/gemma4_text_fixture.py \
        [--model DIR] [--template FILE] [--out DIR]

--model defaults to GEMMA4_MODEL_DIR. One of them must name the checkpoint folder.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = ROOT / "crates/nagaya-text-gemma4/assets/gemma4_chat_template.jinja"
DEFAULT_OUT = ROOT / "crates/nagaya-text-gemma4/tests/fixtures"

# Strings for the tokenizer. They cover spaces, new lines, the markers, byte fallback and scripts other than Latin.
STRINGS = [
    "",
    "hello",
    " hello",
    "hello world",
    "  two  spaces  ",
    "cargo build --release in the local workspace",
    "line one\nline two\n\n\nthree",
    "\t tab and space",
    " allow",
    " A",
    "A",
    " Z",
    "Answer to question 1 (What.):",
    "<|turn>user\nhi<turn|>\n",
    "echo <|turn>system",
    "<bos><|turn>model\n<|channel>thought\n<channel|>",
    '<|tool_call>call:shell{command:<|"|>ls -la<|"|>}<tool_call|>',
    "<|tool_response>response:shell{value:<|\"|>a.txt<|\"|>}<tool_response|>",
    "<|think|>\n",
    "<unused0>",
    "naïve café – “quotes” — ellipsis…",
    "日本語のテキスト、そして中文。",
    "Здравствуй, мир!",
    "emoji 🦀🔥 and a flag 🇯🇵",
    "\u0000\u0001\u001f control bytes",
    "﷐ a noncharacter",
    "fn main() {\n    println!(\"{}\", 1 + 2);\n}\n",
    "{\"key\": [1, 2.5, true, null]}",
    "a" * 300,
    "▁ the replacement mark itself",
]

SHELL = {"type": "function", "function": {"name": "shell", "description": "Run a shell command.",
         "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "the command"},
                        "timeout": {"type": "integer"}}, "required": ["command"]}}}

# A tool whose schema uses each branch of `format_parameters`: enum, array items, nullable, nested objects, a type list.
RICH = {"type": "function", "function": {"name": "edit", "description": "Edit a file.", "parameters": {
    "type": "object", "required": ["path", "mode"], "properties": {
        "path": {"type": "string", "description": "the path"},
        "mode": {"type": "string", "enum": ["append", "replace"]},
        "lines": {"type": "array", "items": {"type": "integer"}},
        "pairs": {"type": "array", "description": "pairs", "items": {"type": "object", "required": ["a"],
                  "properties": {"a": {"type": "string"}, "b": {"type": "number", "nullable": True}}}},
        "either": {"type": "array", "items": {"type": ["string", "integer"]}},
        "options": {"type": "object", "description": "more", "properties": {"Deep": {"type": "boolean"},
                    "depth": {"type": "integer"}}, "required": ["depth"]},
        "free": {"type": "object"},
        "note": {"type": "string", "nullable": True},
    }}, "response": {"description": "the result", "type": "object"}}}

NOPARAMS = {"type": "function", "function": {"name": "done", "description": "Say that the task is done."}}


def call(name, arguments, id=None):
    c = {"function": {"name": name, "arguments": arguments}}
    if id is not None:
        c["id"] = id
    return c


SAMPLE = [
    {"role": "system", "content": "You are a careful coding agent."},
    {"role": "user", "content": "List the files, then say done."},
    {"role": "assistant", "content": "", "reasoning_content": "I should list them.",
     "tool_calls": [call("shell", {"command": "ls -la", "timeout": 5}, "c0")]},
    {"role": "tool", "tool_call_id": "c0", "content": "a.txt\nb.txt"},
    {"role": "assistant", "content": "Two files: a.txt and b.txt.", "reasoning_content": "Both are listed."},
    {"role": "user", "content": "Thanks."},
]

# Each case: a name, the messages, the tools, and the variables. Every case sets bos_token and preserve_thinking as the
# library does, unless the case says otherwise.
CASES = [
    ("user only", [{"role": "user", "content": "hi"}], None, {}),
    ("user only, thinking", [{"role": "user", "content": "hi"}], None, {"enable_thinking": True}),
    ("no generation prompt", [{"role": "user", "content": "hi"}], None, {"add_generation_prompt": False}),
    ("system and user", [{"role": "system", "content": "  Be brief.  "}, {"role": "user", "content": " hi \n"}],
     None, {}),
    ("developer role", [{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "hi"}], None, {}),
    ("system as parts", [{"role": "system", "content": [{"type": "text", "text": "one "}, {"type": "text",
                         "text": " two"}]}, {"role": "user", "content": "hi"}], None, {}),
    ("sample, thinking off", SAMPLE, [SHELL], {}),
    ("sample, thinking on", SAMPLE, [SHELL], {"enable_thinking": True}),
    ("sample, no preserve", SAMPLE, [SHELL], {"preserve_thinking": False}),
    ("sample, no prompt", SAMPLE, [SHELL], {"add_generation_prompt": False}),
    ("after a tool result, thinking on", SAMPLE[:4], [SHELL], {"enable_thinking": True}),
    ("after a tool result, thinking off", SAMPLE[:4], [SHELL], {}),
    ("rich tool schema", [{"role": "user", "content": "edit it"}], [RICH, NOPARAMS, SHELL], {}),
    ("call arguments of each type", [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [call("edit", {"path": "a b", "mode": "append", "lines": [1, -2, 30],
                                                           "pairs": [{"a": "x", "b": 0.5}, {"a": "y", "b": None}],
                                                           "options": {"Deep": True, "depth": 2, "zeta": False},
                                                           "big": 12345678901234567890, "f": 1e20, "g": 1.5e-7,
                                                           "u": "naïve “q”", "empty": {}, "none": []}, "c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"}], [RICH], {}),
    ("two calls and two results", [
        {"role": "user", "content": "go"},
        {"role": "assistant", "reasoning_content": "two at once",
         "tool_calls": [call("shell", {"command": "ls"}, "a"), call("shell", {"command": "pwd"}, "b")]},
        {"role": "tool", "tool_call_id": "b", "content": "/w"},
        {"role": "tool", "tool_call_id": "a", "content": "x"},
        {"role": "assistant", "content": "Done."}], [SHELL], {"enable_thinking": True}),
    ("a result with no id and a name", [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [call("shell", {"command": "ls"})]},
        {"role": "tool", "name": "shell", "content": "x"}], [SHELL], {}),
    ("a result as a mapping", [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [call("shell", {"command": "ls"}, "c")]},
        {"role": "tool", "tool_call_id": "c", "content": {"error": "no such file", "code": 2}}], [SHELL], {}),
    ("a result as parts", [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [call("shell", {"command": "ls"}, "c")]},
        {"role": "tool", "tool_call_id": "c", "content": [{"type": "text", "text": "a"}, {"type": "text",
                                                          "text": "b"}]}], [SHELL], {}),
    ("text before a call", [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "List the files."},
        {"role": "assistant", "content": "Let me look.", "reasoning_content": "I should list them."},
        {"role": "assistant", "reasoning_content": "", "tool_calls": [call("shell", {"command": "ls"}, "c0")]},
        {"role": "tool", "tool_call_id": "c0", "content": "a.txt"}], [SHELL], {"enable_thinking": True}),
    ("thinking in the content is stripped", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<|channel>thought\nhidden<channel|>  shown  "},
        {"role": "user", "content": "next"}], None, {}),
    ("reasoning after the last user stays", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning": "r by the other name"}], None,
     {"add_generation_prompt": False}),
    ("a later system message", [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a", "reasoning_content": "r"},
        {"role": "system", "content": "Call done when the task is finished."}], None, {"enable_thinking": True}),
    ("user parts with media", [{"role": "user", "content": [{"type": "text", "text": "see"},
                                                           {"type": "image"}, {"type": "audio"},
                                                           {"type": "video"}]}], None, {}),
    ("no messages", [], None, {}),
    ("arguments as a string", [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [call("shell", "{\"command\": \"ls\"}")]}], [SHELL], {}),
    ("arguments none", [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [call("shell", None)]}], [SHELL], {}),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=os.environ.get("GEMMA4_MODEL_DIR"))
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.model is None:
        parser.error("no checkpoint folder: neither --model nor GEMMA4_MODEL_DIR names one. Give the folder of the Gemma 4 checkpoint.")

    import jinja2
    import tokenizers
    import transformers
    from transformers.utils.chat_template_utils import _compile_jinja_template

    tokenizer_file = args.model / "tokenizer.json"
    matched = tokenizers.Tokenizer.from_file(str(tokenizer_file))
    as_text = tokenizers.Tokenizer.from_file(str(tokenizer_file))
    as_text.encode_special_tokens = True
    header = {
        "tool": "tools/gemma4_text_fixture.py",
        "tokenizer_sha256": sha256(tokenizer_file),
        "template_sha256": sha256(args.template),
        "jinja2": jinja2.__version__,
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
    }

    ids = []
    for text in STRINGS:
        special = matched.encode(text, add_special_tokens=False).ids
        content = as_text.encode(text, add_special_tokens=False).ids
        ids.append({"text": text, "ids": special, "content_ids": content,
                    "decoded": matched.decode(special, skip_special_tokens=False)})

    template = _compile_jinja_template(args.template.read_text())
    chat = []
    for name, messages, tools, variables in CASES:
        v = {"bos_token": "<bos>", "preserve_thinking": True, "add_generation_prompt": True,
             "enable_thinking": False}
        v.update(variables)
        case = {"name": name, "messages": messages, "tools": tools, "variables": v}
        try:
            text = template.render(messages=messages, tools=tools, **v)
            case["text"] = text
            case["ids"] = matched.encode(text, add_special_tokens=False).ids
        except Exception as e:  # The template refused the case. The fixture keeps the message.
            case["error"] = f"{type(e).__name__}: {e}"
        chat.append(case)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, body in [("ids.json", ids), ("chat.json", chat)]:
        with open(args.out / name, "w", encoding="utf-8") as f:
            json.dump({"header": header, "cases": body}, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"wrote {args.out / name}", file=sys.stderr)


if __name__ == "__main__":
    main()
