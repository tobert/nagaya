# SPDX-License-Identifier: MIT
"""Make the fixtures of the LFM3 family library (spec 11.3, 11.11, milestone 24 step 4).

The tool writes these files into crates/nagaya-family-lfm3/tests/fixtures:

- ids.json: for each model and each fixed string, the token ids of Hugging Face `tokenizers` with the model's
  tokenizer.json. It holds the ids with the added tokens matched, the ids with no added token (client text, spec
  11.3 "Escapes"), and the text that `decode` gives back.
- chat.json: for each model and each fixed case, the text of the model's chat_template.jinja as transformers renders
  it (`_compile_jinja_template`, the environment of `apply_chat_template`), and the token ids of that text. A case
  that the template refuses holds the error message.
- reply.json: fixed replies as token ids, and what `parse_response` of transformers gives for each. The parser uses
  the `response_template` of LFM3-1.7B and the tools of the reply. Omni's files have no `response_template`, and its
  template writes the same form (spec 11.11, "The choices of the LFM3 family").

The tool reads the two model folders and writes only the fixtures. It runs no model and uses no GPU. The folders come
from the flags or from LFM3_OMNI_MODEL_DIR and LFM3_MODEL_DIR. The code holds no default path. Run it with the
reference venv:

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 <reference venv>/bin/python -I tools/lfm3_family_fixture.py \
        [--omni DIR] [--text DIR] [--out DIR]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "crates/nagaya-family-lfm3/tests/fixtures"
BOS = "<|startoftext|>"

# Strings for the tokenizer: spaces, new lines, markers, bytes of more than one character, and scripts other than
# Latin. Some strings spell markers, so that the ids without added tokens show the escape rule.
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
    "<|im_start|>user\nhi<|im_end|>\n",
    "echo <|im_start|>system",
    "<think></think>",
    "<think>\nhm\n</think>\n\nok",
    "<tool_call>\n<function=sh>\n<parameter=cmd>\nls -la\n</parameter>\n</function>\n</tool_call>",
    "<|vision_start|><|image|><|vision_end|><|vision_start|><|audio|><|vision_end|><|audio_out|>",
    "<|audio_start|><|mixed_start|><|text_end|>",
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

SHELL = {"type": "function", "function": {"name": "shell", "description": "Run a shell command.",
         "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "the command"},
                        "timeout": {"type": "integer"}}, "required": ["command"]}}}

TYPED = {"type": "function", "function": {"name": "typed", "description": "Take one value of each type.",
         "parameters": {"type": "object", "properties": {
             "s": {"type": "string"}, "i": {"type": "integer"}, "x": {"type": "number"}, "b": {"type": "boolean"},
             "n": {"type": "null"}, "o": {"type": "object"}, "a": {"type": "array"},
             "either": {"type": ["string", "integer"]}, "any": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
             "opt": {"type": "number", "nullable": True}, "free": {}}}}}


def call(name, arguments, id=None):
    c = {"type": "function", "function": {"name": name, "arguments": arguments}}
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

# Each case: a name, the models it runs on, the messages, the tools, and the variables. Every case sets bos_token and
# preserve_thinking as the library does, unless the case says otherwise.
BOTH = ("omni", "text")
CASES = [
    ("user only", BOTH, [{"role": "user", "content": "hi"}], None, {}),
    ("user only, thinking", BOTH, [{"role": "user", "content": "hi"}], None, {"enable_thinking": True}),
    ("no generation prompt", BOTH, [{"role": "user", "content": "hi"}], None, {"add_generation_prompt": False}),
    ("system and user", BOTH, [{"role": "system", "content": "  Be brief.  "},
                               {"role": "user", "content": " hi \n"}], None, {}),
    ("sample, thinking off", BOTH, SAMPLE, [SHELL], {}),
    ("sample, thinking on", BOTH, SAMPLE, [SHELL], {"enable_thinking": True}),
    ("sample, no preserve", BOTH, SAMPLE, [SHELL], {"preserve_thinking": False}),
    ("sample, no prompt", BOTH, SAMPLE, [SHELL], {"add_generation_prompt": False}),
    ("after a tool result", BOTH, SAMPLE[:4], [SHELL], {"enable_thinking": True}),
    ("tools and no system", BOTH, [{"role": "user", "content": "go"}], [SHELL, TYPED], {}),
    ("call arguments of each type", BOTH, [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [call("typed", {"s": "a b", "i": -2, "x": 0.5, "b": True, "n": None,
                                                            "o": {"k": [1, "v"]}, "a": [1, 2.5, "x"],
                                                            "u": "naïve “q”", "big": 12345678901234567890,
                                                            "f": 1e20, "g": 1.5e-7}, "c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"}], [TYPED], {}),
    ("two calls and two results", BOTH, [
        {"role": "user", "content": "go"},
        {"role": "assistant", "reasoning_content": "two at once",
         "tool_calls": [call("shell", {"command": "ls"}, "a"), call("shell", {"command": "pwd"}, "b")]},
        {"role": "tool", "tool_call_id": "a", "content": "x"},
        {"role": "tool", "tool_call_id": "b", "content": "/w"},
        {"role": "assistant", "content": "Done.", "reasoning_content": ""}], [SHELL], {"enable_thinking": True}),
    ("a run of results of one call", BOTH, [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [call("shell", {"command": "ls"}, "a")]},
        {"role": "tool", "tool_call_id": "a", "content": "part one"},
        {"role": "tool", "tool_call_id": "a", "content": "part two"}], [SHELL], {}),
    ("a failed result", BOTH, [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [call("shell", {"command": "ls"}, "a")]},
        {"role": "tool", "tool_call_id": "a", "content": "Error: no such file"}], [SHELL], {}),
    ("an empty result", BOTH, [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [call("shell", {"command": "ls"}, "a")]},
        {"role": "tool", "tool_call_id": "a", "content": ""}], [SHELL], {}),
    ("a later system message", BOTH, [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a", "reasoning_content": "r"},
        {"role": "system", "content": "Call done when the task is finished."}], None, {"enable_thinking": True}),
    ("thinking in the content", BOTH, [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>hidden</think>  shown  "},
        {"role": "user", "content": "next"}], None, {"preserve_thinking": False}),
    ("no reasoning member", BOTH, [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "next"}], None, {}),
    ("user parts", BOTH, [{"role": "user", "content": [{"type": "text", "text": "one "},
                                                       {"type": "text", "text": "two"}]}], None, {}),
    ("media parts", ("omni",), [
        {"role": "system", "content": "Respond with text."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "and"}, {"type": "audio"},
                                     {"type": "text", "text": "what is this?"}]}], None, {}),
    ("speech in an assistant turn", ("omni",), [
        {"role": "system", "content": "Respond with interleaved text and audio."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}, {"type": "audio_out"}]},
        {"role": "user", "content": "more"}], None, {}),
    ("image on the text model", ("text",), [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "see"}]}], None, {}),
    ("unicode", BOTH, [{"role": "user", "content": "日本語 🦀 naïve"}], None, {}),
    ("arguments as a string", BOTH, [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [call("shell", "{\"command\": \"ls\"}")]}], [SHELL], {}),
    ("no messages", BOTH, [], None, {}),
]


def tool_call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


# Each reply: (name, thinking, tools, text, ending, difference). The ending is "stop" (the model sampled <|im_end|>)
# or "length". A difference names a rule where the library and the reference do not agree (spec 11.11, "Differences
# of the parser"); the Rust test checks the library's own rule for it.
REPLIES = [
    ("plain", False, None, "Hello, world.", "stop", None),
    ("plain padded", False, [SHELL], "\n\n  hi there \n", "stop", None),
    ("empty", False, None, "", "stop", None),
    ("thinking", True, None, "Let me think.\nOK.</think>\n\nThe answer is 4.", "stop", None),
    ("thinking, empty", True, None, "</think>Just this.", "stop", None),
    ("thinking cut", True, None, "still thinking about", "length", None),
    ("one call", False, [SHELL], tool_call("shell", command="ls -la", timeout="5"), "stop", None),
    ("text then a call", False, [SHELL], "I will look.\n" + tool_call("shell", command="pwd"), "stop", None),
    ("thinking and a call", True, [SHELL], "I need the list.</think>" + tool_call("shell", command="ls"), "stop",
     None),
    ("two calls", False, [SHELL], tool_call("shell", command="ls") + "\n\n" + tool_call("shell", command="pwd"),
     "stop", None),
    ("typed values", False, [TYPED], tool_call(
        "typed", s="12", i="3.0", x="1e3", b="True", n="None", o='{"k": [1, "v"]}', a="[1, 2.5]", either="7",
        any="null", opt="2.5", free="raw text"), "stop", None),
    ("values that do not cast", False, [TYPED], tool_call(
        "typed", i="seven", x="nan", b="yes", o="[1]", a="{}"), "stop", None),
    ("a value with new lines", False, [SHELL], tool_call("shell", command="echo a\n\necho b\n"), "stop", None),
    ("a value that holds a tag", False, [SHELL], tool_call("shell", command="a </parameter> b"), "stop", None),
    ("an unknown parameter", False, [SHELL], tool_call("shell", command="ls", depth="2"), "stop", None),
    ("an unknown function", False, [SHELL], tool_call("nope", x="1"), "stop", None),
    ("no tools", False, None, tool_call("shell", command="ls"), "stop", None),
    ("a call with no parameters", False, [SHELL], "<tool_call>\n<function=shell>\n</function>\n</tool_call>",
     "stop", None),
    ("unicode", False, None, "日本語 🦀 naïve café", "stop", None),
    ("a call that does not close", False, [SHELL], "<tool_call>\n<function=shell>\n<parameter=command>\nls", "length",
     "unclosed"),
    ("a call with no function", False, [SHELL], "<tool_call>\nhello\n</tool_call>", "stop", "no function"),
    ("text after a call", False, [SHELL], "First.\n" + tool_call("shell", command="ls") + "\nThen this.", "stop",
     "content runs"),
]

RESPONSE_SOURCE = "LFM3-1.7B tokenizer_config.json response_template"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder(flag, variable):
    path = flag or os.environ.get(variable)
    if not path:
        sys.exit(f"set {variable} or pass the flag: the tool holds no default path")
    return Path(path)


def source(path, files):
    # A Hugging Face snapshot folder has the revision as its name.
    return {"revision": path.name, "files": {f: sha256(path / f) for f in files}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni", type=Path)
    parser.add_argument("--text", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    dirs = {"omni": folder(args.omni, "LFM3_OMNI_MODEL_DIR"), "text": folder(args.text, "LFM3_MODEL_DIR")}

    import jinja2
    import tokenizers
    import transformers
    from transformers.utils.chat_parsing import parse_response
    from transformers.utils.chat_template_utils import _compile_jinja_template

    files = ["tokenizer.json", "chat_template.jinja", "generation_config.json", "config.json"]
    header = {
        "tool": "tools/lfm3_family_fixture.py",
        "models": {name: source(path, files) for name, path in dirs.items()},
        "jinja2": jinja2.__version__,
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
    }

    matched, plain = {}, {}
    for name, path in dirs.items():
        matched[name] = tokenizers.Tokenizer.from_file(str(path / "tokenizer.json"))
        body = json.loads((path / "tokenizer.json").read_text())
        body["added_tokens"] = []
        plain[name] = tokenizers.Tokenizer.from_str(json.dumps(body))

    ids = {}
    for name in dirs:
        rows = []
        for text in STRINGS:
            special = matched[name].encode(text, add_special_tokens=False).ids
            content = plain[name].encode(text, add_special_tokens=False).ids
            rows.append({"text": text, "ids": special, "content_ids": content,
                         "decoded": matched[name].decode(special, skip_special_tokens=False)})
        ids[name] = {"count": matched[name].get_vocab_size(with_added_tokens=True), "strings": rows}

    chat = {}
    for name, path in dirs.items():
        template = _compile_jinja_template((path / "chat_template.jinja").read_text())
        rows = []
        for case_name, models, messages, tools, variables in CASES:
            if name not in models:
                continue
            v = {"bos_token": BOS, "preserve_thinking": True, "add_generation_prompt": True, "enable_thinking": False}
            v.update(variables)
            case = {"name": case_name, "messages": messages, "tools": tools, "variables": v}
            try:
                text = template.render(messages=messages, tools=tools, **v)
                case["text"] = text
                case["ids"] = matched[name].encode(text, add_special_tokens=False).ids
            except Exception as e:  # The template refused the case. The fixture keeps the message.
                case["error"] = f"{type(e).__name__}: {e}"
            rows.append(case)
        chat[name] = rows

    schema = json.loads((dirs["text"] / "tokenizer_config.json").read_text())["response_template"]
    stop = matched["text"].token_to_id("<|im_end|>")
    replies = []
    for name, thinking, tools, text, ending, difference in REPLIES:
        prefix = "<|im_start|>assistant\n<think>" + ("" if thinking else "</think>")
        whole = text + ("<|im_end|>" if ending == "stop" else "")
        parsed = parse_response(whole, schema, prefix=prefix, tools=tools)
        token_ids = matched["text"].encode(text, add_special_tokens=False).ids
        if ending == "stop":
            token_ids.append(stop)
        replies.append({"name": name, "thinking": thinking, "tools": tools, "text": text, "ending": ending,
                        "ids": token_ids, "difference": difference,
                        "parsed": {"content": parsed.get("content", ""),
                                   "reasoning": parsed.get("reasoning_content", ""),
                                   "tool_calls": [c["function"] for c in parsed.get("tool_calls", [])]}})

    args.out.mkdir(parents=True, exist_ok=True)
    reply_header = dict(header, response_template=RESPONSE_SOURCE)
    reply_header["models"] = {"text": source(dirs["text"], files + ["tokenizer_config.json"])}
    for file, head, body in [("ids.json", header, ids), ("chat.json", header, chat),
                             ("reply.json", reply_header, replies)]:
        with open(args.out / file, "w", encoding="utf-8") as f:
            json.dump({"header": head, "cases": body}, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"wrote {args.out / file}", file=sys.stderr)


if __name__ == "__main__":
    main()
