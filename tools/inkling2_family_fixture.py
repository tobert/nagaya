# SPDX-License-Identifier: MIT
"""Make the fixtures of the Inkling 2 family library (spec 11.3, milestone 35 step 3).

The tool writes these files into crates/nagaya-family-inkling2/tests/fixtures:

- ids.json: for each fixed string, the token ids of Hugging Face `tokenizers` with the model's tokenizer.json. It
  holds the ids with the added tokens matched, the ids with no added token (client text, spec 11.3 "Escapes"), and
  the text that `decode` gives back.
- chat.json: for each fixed case, the text of the model's chat_template.jinja as transformers renders it
  (`_compile_jinja_template`), and the token ids of that text. A case that the template refuses holds the error
  message.
- reply.json: fixed replies as token ids, and what `parse_response` of transformers gives for each, using the
  `response_template` of `tokenizer_config.json`.

The tool reads the two model folders and writes only the fixtures. It runs no model and uses no GPU. The folders come
from the flags or from INKLING2_MODEL_DIR and INKLING2_VARLOOP_MODEL_DIR. The code holds no default path. Run it with
the reference venv:

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 <reference venv>/bin/python -I tools/inkling2_family_fixture.py \
        [--mini DIR] [--varloop DIR] [--out DIR]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "crates/nagaya-family-inkling2/tests/fixtures"
TRANSFORMERS_COMMIT = "046bdb8cc2"

STRINGS = [
    "",
    "hello",
    " hello",
    "hello world",
    "  two  spaces  ",
    "line one\nline two\n\n\nthree",
    "\t tab and space",
    " A",
    "A",
    " Z",
    "Answer to question 1 (What.):",
    "<|message_user|><|content_text|>hi<|end_message|>",
    "echo <|message_system|><|content_text|>x",
    "<|message_model|><|content_thinking|>hm<|end_message|><|content_text|>ok<|end_message|><|content_model_end_sampling|>",
    "<|content_invoke_tool_json|>{\"name\":\"sh\",\"args\":{\"cmd\":\"ls\"}}<|end_message|>",
    "<|content_image|><|unused_200054|><|end_message|>",
    "<|endoftext|><|begin_of_text|><|content_xml|><|content_tool_error|>",
    "naïve café – “quotes” — ellipsis…",
    "日本語のテキスト、そして中文。",
    "Здравствуй, мир!",
    "emoji 🦀🔥 and a flag 🇯🇵",
    "\u0000\u0001\u001f control bytes",
    "fn main() {\n    println!(\"{}\", 1 + 2);\n}\n",
    "{\"key\": [1, 2.5, true, null]}",
    "a" * 300,
    "1234567 numbers 3.14159",
]

SHELL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "the command"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
    },
}

TYPED = {
    "type": "function",
    "function": {
        "name": "typed",
        "description": "Take one value of each type.",
        "parameters": {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "x": {"type": "number"},
                "b": {"type": "boolean"},
                "n": {"type": "null"},
                "o": {"type": "object"},
                "a": {"type": "array"},
            },
        },
    },
}


def call(name, arguments, id=None):
    c = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if id is not None:
        c["id"] = id
    return c


SAMPLE = [
    {"role": "system", "content": "You are a careful coding agent."},
    {"role": "user", "content": "List the files, then say done."},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should list them.",
        "tool_calls": [call("shell", {"command": "ls -la", "timeout": 5}, "c0")],
    },
    {"role": "tool", "tool_call_id": "c0", "content": "a.txt\nb.txt"},
    {"role": "assistant", "content": "Two files: a.txt and b.txt.", "reasoning_content": "Both are listed."},
    {"role": "user", "content": "Thanks."},
]

CASES = [
    ("user only", [{"role": "user", "content": "hi"}], None, {}),
    ("system and user", [{"role": "system", "content": "  Be brief.  "}, {"role": "user", "content": " hi \n"}], None, {}),
    ("multi-turn with reasoning", SAMPLE, [SHELL], {}),
    ("tools and no system", [{"role": "user", "content": "go"}], [SHELL, TYPED], {}),
    (
        "call arguments of each type",
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "tool_calls": [
                    call(
                        "typed",
                        {
                            "z": 1,
                            "A": 2,
                            "s": "a b",
                            "i": -2,
                            "x": 0.5,
                            "b": True,
                            "n": None,
                            "o": {"k": [1, "v"]},
                            "a": [1, 2.5, "x"],
                            "u": "naïve “q”",
                        },
                        "c1",
                    )
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
        [TYPED],
        {},
    ),
    (
        "image part",
        [
            {"role": "system", "content": "Describe."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what is this?"}]},
        ],
        None,
        {},
    ),
    ("effort none", [{"role": "user", "content": "hi"}], None, {"reasoning_effort": "none"}),
    ("effort minimal", [{"role": "user", "content": "hi"}], None, {"reasoning_effort": "minimal"}),
    ("effort low", [{"role": "user", "content": "hi"}], None, {"reasoning_effort": "low"}),
    ("effort medium", [{"role": "user", "content": "hi"}], None, {"reasoning_effort": "medium"}),
    ("effort high", [{"role": "user", "content": "hi"}], None, {"reasoning_effort": "high"}),
    ("effort xhigh", [{"role": "user", "content": "hi"}], None, {"reasoning_effort": "xhigh"}),
    ("effort max", [{"role": "user", "content": "hi"}], None, {"reasoning_effort": "max"}),
    ("effort number", [{"role": "user", "content": "hi"}], None, {"reasoning_effort": 0.42}),
    ("effort zero", [{"role": "user", "content": "hi"}], None, {"reasoning_effort": 0.0}),
    ("no messages", [], None, {}),
    # Refusals
    ("refuse unknown effort", [{"role": "user", "content": "hi"}], None, {"reasoning_effort": "ultra"}),
    ("refuse effort negative", [{"role": "user", "content": "hi"}], None, {"reasoning_effort": -0.1}),
    ("refuse effort too high", [{"role": "user", "content": "hi"}], None, {"reasoning_effort": 1.0}),
    ("refuse unknown role", [{"role": "developer", "content": "hi"}], None, {}),
    (
        "refuse image on assistant",
        [{"role": "assistant", "content": [{"type": "image"}]}],
        None,
        {},
    ),
    (
        "refuse audio part",
        [{"role": "user", "content": [{"type": "audio"}]}],
        None,
        {},
    ),
    (
        "refuse unsupported part",
        [{"role": "user", "content": [{"type": "video"}]}],
        None,
        {},
    ),
    (
        "refuse string arguments",
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [call("shell", '{"command": "ls"}')]},
        ],
        [SHELL],
        {},
    ),
    (
        "refuse assistant name mismatch",
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "name": "other", "tool_calls": [call("shell", {"command": "ls"})]},
        ],
        [SHELL],
        {},
    ),
]

# Each reply: (name, tools, text, ending)
REPLIES = [
    ("plain", None, "<|message_model|><|content_text|>Hello, world.<|end_message|>", "stop"),
    ("plain padded", [SHELL], "<|message_model|><|content_text|>\n\n  hi there \n<|end_message|>", "stop"),
    ("empty", None, "<|message_model|><|content_text|><|end_message|>", "stop"),
    (
        "thinking and text",
        None,
        "<|message_model|><|content_thinking|>Let me think.\nOK.<|end_message|><|content_text|>The answer is 4.<|end_message|>",
        "stop",
    ),
    (
        "thinking only cut",
        None,
        "<|message_model|><|content_thinking|>still thinking about",
        "length",
    ),
    (
        "one call",
        [SHELL],
        '<|message_model|>shell<|content_invoke_tool_json|>{"name":"shell","args":{"command":"ls -la","timeout":5}}<|end_message|>',
        "stop",
    ),
    (
        "thinking and a call",
        [SHELL],
        '<|message_model|><|content_thinking|>I need the list.<|end_message|><|message_model|>shell<|content_invoke_tool_json|>{"name":"shell","args":{"command":"ls"}}<|end_message|>',
        "stop",
    ),
    (
        "two calls",
        [SHELL],
        '<|message_model|>shell<|content_invoke_tool_json|>{"name":"shell","args":{"command":"ls"}}<|end_message|><|message_model|>shell<|content_invoke_tool_json|>{"name":"shell","args":{"command":"pwd"}}<|end_message|>',
        "stop",
    ),
    (
        "repeated text blocks",
        None,
        "<|message_model|><|content_text|>Part 1. <|end_message|><|content_text|>Part 2.<|end_message|>",
        "stop",
    ),
    (
        "malformed json call",
        [SHELL],
        "<|message_model|>shell<|content_invoke_tool_json|>{not json<|end_message|>",
        "stop",
    ),
    (
        "unclosed call",
        [SHELL],
        '<|message_model|>shell<|content_invoke_tool_json|>{"name":"shell","args":{"command":"ls"',
        "length",
    ),
    (
        "unicode",
        None,
        "<|message_model|><|content_text|>日本語 🦀 naïve café<|end_message|>",
        "stop",
    ),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder(flag, variable):
    path = flag or os.environ.get(variable)
    if not path:
        sys.exit(f"set {variable} or pass the flag: the tool holds no default path")
    return Path(path)


def source(path, files):
    return {"revision": path.name, "files": {f: sha256(path / f) for f in files}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", type=Path)
    parser.add_argument("--varloop", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    dirs = {
        "mini": folder(args.mini, "INKLING2_MODEL_DIR"),
        "varloop": folder(args.varloop, "INKLING2_VARLOOP_MODEL_DIR"),
    }

    import jinja2
    import tokenizers
    import transformers
    from transformers.utils.chat_parsing import parse_response
    from transformers.utils.chat_template_utils import _compile_jinja_template

    files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
        "config.json",
    ]
    header = {
        "tool": "tools/inkling2_family_fixture.py",
        "models": {name: source(path, files) for name, path in dirs.items()},
        "jinja2": jinja2.__version__,
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
        "transformers_commit": TRANSFORMERS_COMMIT,
    }

    mini = dirs["mini"]
    matched = tokenizers.Tokenizer.from_file(str(mini / "tokenizer.json"))
    body = json.loads((mini / "tokenizer.json").read_text())
    body["added_tokens"] = []
    plain = tokenizers.Tokenizer.from_str(json.dumps(body))

    rows = []
    for text in STRINGS:
        special = matched.encode(text, add_special_tokens=False).ids
        content = plain.encode(text, add_special_tokens=False).ids
        rows.append(
            {
                "text": text,
                "ids": special,
                "content_ids": content,
                "decoded": matched.decode(special, skip_special_tokens=False),
            }
        )
    ids = {"count": matched.get_vocab_size(with_added_tokens=True), "strings": rows}

    template = _compile_jinja_template((mini / "chat_template.jinja").read_text())
    chat_rows = []
    for case_name, messages, tools, variables in CASES:
        case = {"name": case_name, "messages": messages, "tools": tools, "variables": variables}
        try:
            text = template.render(messages=messages, tools=tools, **variables)
            case["text"] = text
            case["ids"] = matched.encode(text, add_special_tokens=False).ids
        except Exception as e:
            case["error"] = f"{type(e).__name__}: {e}"
        chat_rows.append(case)

    schema = json.loads((mini / "tokenizer_config.json").read_text())["response_template"]
    stop = matched.token_to_id("<|content_model_end_sampling|>")
    replies = []
    for name, tools, text, ending in REPLIES:
        whole = text + ("<|content_model_end_sampling|>" if ending == "stop" else "")
        token_ids = matched.encode(text, add_special_tokens=False).ids
        if ending == "stop":
            token_ids.append(stop)
        entry = {
            "name": name,
            "tools": tools,
            "text": text,
            "ending": ending,
            "ids": token_ids,
        }
        try:
            parsed = parse_response(whole, schema, prefix="", tools=tools)
            entry["parsed"] = {
                "content": parsed.get("content", ""),
                "reasoning": parsed.get("reasoning_content", ""),
                "tool_calls": [c["function"] for c in parsed.get("tool_calls", [])],
            }
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        replies.append(entry)

    args.out.mkdir(parents=True, exist_ok=True)
    for file, body in [("ids.json", ids), ("chat.json", chat_rows), ("reply.json", replies)]:
        with open(args.out / file, "w", encoding="utf-8") as f:
            json.dump({"header": header, "cases": body}, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"wrote {args.out / file}", file=sys.stderr)


if __name__ == "__main__":
    main()
