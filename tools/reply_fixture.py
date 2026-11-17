#!/usr/bin/env python3
"""Write the reply-parser fixture of nagaya-text-qwen38 from the reference's Python.

The fixture holds replies as token ids. For each reply it holds what the token loop of the first Qwen3.8 server gives
for each token (the piece of text and the phase) and what `ReplyTracker.finish()` gives at the end. The tool runs no
server and uses no GPU. It reads `tools/qwen38_reference`, and the first GGUF shard for the tokenizer: `--gguf`, or the
first path of `QWEN38_SHARDS`. Run it with the venv of `tools/qwen38_reference/requirements.txt`:

    PYTHONDONTWRITEBYTECODE=1 <venv>/bin/python -I tools/reply_fixture.py [--gguf GGUF] [--out OUT]

The loop below copies the token loop of the first Qwen3.8 server (`Generation._loop`): `Utf8Stream.push` of the
token's bytes, then `ReplyTracker.feed`. The phase of a token is the phase of the last released segment, or the phase
of the tracker when nothing was released. At the end the loop flushes the UTF-8 stream into the tracker.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True

TOOLS = Path(__file__).resolve().parent
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "crates/nagaya-text-qwen38/tests/fixtures/reply.json"

# The tool sets. "reference" is the set of the tests of the first Qwen3.8 server: one tool with every
# type, a tool with no parameters, and a tool with a required parameter.
TOOLSETS = {
    "none": None,
    "reference": [
        {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {
            "n": {"type": "integer"}, "x": {"type": "number"}, "flag": {"type": "boolean"}, "obj": {"type": "object"},
            "arr": {"type": "array"}, "s": {"type": "string"}, "any": {}, "either": {"type": ["string", "integer"]},
            "nul": {"type": "null"}, "one": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}}},
        {"type": "function", "function": {"name": "g", "description": "no parameters"}},
        {"type": "function", "function": {"name": "h", "parameters": {"type": "object", "properties": {
            "a": {"type": "string"}}, "required": ["a"]}}},
    ],
    # A tool without the "function" wrapper, and an empty list: the empty list has no tools.
    "bare": [{"name": "w", "parameters": {"properties": {"q": {"type": "string"}, "k": {"type": "integer"}}}}],
    "empty": [],
}


def call(name, **args):
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                   for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


# Each reply: (name, thinking, toolset, text, ending). The text is the reply without the stop token.
# The ending is "stop" (the model sampled a stop token) or "length" (max_tokens or a full context).
def replies() -> list[tuple]:
    r = []

    def add(name, thinking, tools, text, ending="stop"):
        r.append((name, thinking, tools, text, ending))

    # content only
    add("plain", False, "none", "Hello, world.")
    add("plain_padded", False, "reference", "\n\n  hi there \n")
    add("plain_empty", False, "none", "")
    add("plain_whitespace_only", False, "none", " \n\t \x1c ")
    add("plain_think_end_is_text", False, "reference", "a </think> b")
    add("plain_tool_marker_without_tools", False, "none", "a <tool_call> b")
    add("plain_tool_marker_with_empty_tools", False, "empty", "a <tool_call> b")
    add("plain_false_marker", False, "reference", "x<tool_cal y<tool_c")
    add("plain_ends_in_marker_prefix", False, "reference", "abc<tool_c")
    add("plain_length", False, "reference", "the answer is cut off, so", "length")

    # thinking
    add("think_then_content", True, "reference", "let me think\n</think>\n\nThe answer is 4.")
    add("think_empty", True, "reference", "\n</think>\n\nanswer")
    add("think_empty_tight", True, "reference", "</think>answer")
    add("think_only_marker", True, "reference", "</think>")
    add("think_no_content", True, "reference", "just thinking\n</think>\n\n")
    add("think_padded", True, "none", "  \nsome plan  \n\n</think>\n\n  answer here \n")
    add("think_second_marker_is_content", True, "none", "a</think>b</think>c")
    add("think_cut", True, "reference", "hmm, let me think about", "length")
    add("think_cut_in_marker", True, "reference", "plan</thi", "length")
    add("think_cut_with_call_text", True, "reference", "hmm <tool_call> still thinking", "length")
    add("think_cut_empty", True, "reference", "", "length")
    add("think_false_marker", True, "none", "x</thi")
    add("think_false_marker_continued", True, "none", "x</thinkx>y</th")
    add("think_without_tools_tool_text", True, "none", "plan</think>ok <tool_call> not a call")

    # calls
    g, f, h = call("g"), call("f", n=3, s="x"), call("h", a="v")
    add("call_one", False, "reference", g)
    add("call_one_thinking", True, "reference", "plan\n</think>\n\n" + g)
    add("call_content_then_call", True, "reference", "plan\n</think>\n\nSure, calling\n\n" + f)
    add("call_several", True, "reference", "plan\n</think>\n\nI will call three tools.\n\n" + f + "\n" + g + "\n" + h)
    add("call_several_tight", False, "reference", g + g + h)
    add("call_types", False, "reference", call(
        "f", n=3, x=1.5, flag=True, obj={"z": 1, "a": [1, 2, {"m": None, "b": 2}]}, arr=[3, "x", None], s="multi\nline",
        **{"any": "text", "either": "5", "nul": None, "one": 7}))
    add("call_order_kept", False, "reference", call("f", s="1", n=2, flag=False, x=-0.5, obj={"k": 1, "j": 2}))
    add("call_numbers", False, "reference", call("f", n=-12, x=2, **{"any": "1e3"}))
    add("call_either_json_wins", False, "reference", call("f", either="42") + call("f", either="forty") + call("f", either="[1]"))
    add("call_any_raw_or_json", False, "reference", call("f", **{"any": "plain words"}) + call("f", **{"any": '{"a": 1}'}) +
        call("f", **{"any": "  7  "}))
    add("call_string_stays_string", False, "reference", call("f", s="123") + call("f", s='{"a": 1}') + call("f", s="null"))
    add("call_one_null", False, "reference", call("f", one="null") + call("f", one="5"))
    add("call_value_has_close_tag", False, "reference", call("h", a="x\n</parameter>\ny") + call("h", a="q\n</parameter>"))
    add("call_value_unicode", False, "reference", call("h", a="héllo 世界 \U0001f600") + call("g"))
    add("call_value_empty", False, "reference", call("h", a=""))
    add("call_value_spaces", False, "reference", call("h", a="  padded  \n\n"))
    add("call_obj_duplicate_key", False, "reference",
        '<tool_call>\n<function=f>\n<parameter=obj>\n{"a":1,"b":2,"a":3}\n</parameter>\n</function>\n</tool_call>')
    add("call_bare_tool", False, "bare", call("w", q="a b", k=3))
    add("call_extra_whitespace", False, "reference",
        "<tool_call>\n  <function=h>\n\n<parameter=a>\nv\n</parameter>\n\n  </function>\n  </tool_call>  \n\n" + g)
    add("call_with_text_around", True, "reference", "t</think>before <tool_call> middle " + g + " after text")
    add("call_name_with_space", False, "reference", "<tool_call>\n<function=my tool>\n</function>\n</tool_call>")

    # malformed calls
    add("bad_no_function", False, "reference", "<tool_call>\nhello\n</tool_call>")
    add("bad_function_no_newline", False, "reference", "<tool_call>\n<function=g></function>\n</tool_call>")
    add("bad_unclosed_call", False, "reference", "<tool_call>\n<function=g>\n</function>\n")
    add("bad_unclosed_function", False, "reference", "<tool_call>\n<function=g>\n</tool_call>")
    add("bad_param_unclosed", False, "reference", "<tool_call>\n<function=h>\n<parameter=a>\nv\n</function>\n</tool_call>")
    add("bad_param_syntax", False, "reference", "<tool_call>\n<function=h>\njunk\n</function>\n</tool_call>")
    add("bad_unknown_function", False, "reference", call("nope", a="1"))
    add("bad_undeclared_param", False, "reference", call("h", a="1", b="2"))
    add("bad_repeated_param", False, "reference", call("h", a="1") .replace("</function>", "<parameter=a>\n2\n</parameter>\n</function>"))
    add("bad_missing_required", False, "reference", call("h"))
    add("bad_type_integer", False, "reference", call("f", n="abc"))
    add("bad_type_integer_got_float", False, "reference", call("f", n="1.5"))
    add("bad_type_integer_got_bool", False, "reference", call("f", n="true"))
    add("bad_type_boolean", False, "reference", call("f", flag="yes"))
    add("bad_type_number_nan", False, "reference", call("f", x="NaN"))
    add("bad_type_number_big", False, "reference", call("f", x="1e999"))
    add("bad_type_object_got_array", False, "reference", call("f", obj="[1]"))
    add("bad_type_either_array", False, "reference", call("f", nul="0"))
    add("bad_prose_between", False, "reference", g + "\nsome words\n" + g)
    add("bad_prose_after", True, "reference", "t</think>c" + g + " trailing")
    add("bad_then_good", False, "reference", "<tool_call>\n<function=zz\n" + g + call("h", a="ok"))
    add("bad_good_between_bad", False, "reference", call("nope") + g + call("f", n="x"))
    add("bad_open_twice", False, "reference", "<tool_call>\n<tool_call>\n<function=g>\n</function>\n</tool_call>")
    add("bad_close_only", False, "reference", "text </tool_call> more")
    add("bad_function_name_quote", False, "reference", "<tool_call>\n<function=it's>\n</function>\n</tool_call>")
    add("bad_unknown_quoted_name", False, "reference", call("it's \"x\"é"))
    add("bad_type_value_quoted", False, "reference", call("f", n="it's\n\"q\" é\t\\"))

    # cut calls
    add("cut_in_call_open", False, "reference", "ok<tool_ca", "length")
    add("cut_after_call_open", False, "reference", "ok<tool_call>", "length")
    add("cut_in_function", False, "reference", "ok<tool_call>\n<function=f>\n<parameter=n>\n3", "length")
    add("cut_after_good_call", False, "reference", g + "\n<tool_call>\n<function=h>", "length")
    add("cut_whole_call_no_close", False, "reference", g[:-len("</tool_call>")], "length")
    add("cut_thinking_call", True, "reference", "plan\n</think>\n\nx" + f[:20], "length")

    # multi-byte characters
    add("utf8_two_byte", False, "none", "café naïve")
    add("utf8_three_byte", False, "none", "今日はいい天気")
    add("utf8_four_byte", False, "none", "ok \U0001f600\U0001f680 done")
    add("utf8_in_thinking", True, "reference", "考える \U0001f914\n</think>\n\n答え \U0001f600")
    add("utf8_in_call", False, "reference", call("h", a="世界 \U0001f600") + "é")
    add("utf8_cut_two", False, "none", "café", "length")
    add("utf8_cut_four", False, "none", "ok \U0001f600", "length")
    add("utf8_cut_in_thinking", True, "none", "x 世", "length")
    add("utf8_whitespace_unicode", False, "none", "   mid dle 　")
    return r


def make_token_ids(tok, text: str, mode: str, rnd: random.Random) -> list[int]:
    """Token ids for `text`. whole: the tokenizer's own. bytes: one byte-token for each byte of the UTF-8 text.
    chars: each character on its own. cuts: random cut points, each part encoded on its own."""
    if mode == "whole":
        return tok.encode(text)
    if mode == "bytes":
        return [tok.token_to_id[tok._byte_encoder[b]] for b in text.encode("utf-8")]
    if mode == "chars":
        return [i for ch in text for i in tok.encode(ch)]
    assert mode == "cuts"
    points = sorted({rnd.randrange(0, len(text) + 1) for _ in range(rnd.randint(1, 6))} | {0, len(text)})
    return [i for a, b in zip(points, points[1:]) for i in tok.encode(text[a:b])]


def run_reference(chatfmt, tok, thinking: bool, tools, ids: list[int], ending: str) -> dict:
    """The token loop of the first Qwen3.8 server, with the ids given. `released` is what `feed` gave for the token."""
    tracker = chatfmt.ReplyTracker(thinking, tools)
    utf8 = chatfmt.Utf8Stream()
    events = []
    for i in ids:
        text = utf8.push(tok.token_piece(i))
        segs = tracker.feed(text)
        # [phase, piece, released]: a list, not an object, so that the fixture stays small.
        events.append([segs[-1][0] if segs else tracker.phase, text, [list(s) for s in segs]])
    flush = utf8.flush()
    flushed = tracker.feed(flush) if flush else []
    result = tracker.finish()
    # `tail` is what the reply releases after its last token: what the flush released, and the held text.
    return {"events": events, "flush": flush, "tail": [list(t) for t in flushed + tracker.tail],
            "finish": ending, "result": result}


def check_rule(case: dict) -> dict:
    """gogi's rule for pieces, as a diagnostic: the pieces of a phase, joined, equal the part up to strip().
    Returns the parts that break it, for the token pieces and for the released segments."""
    out = {"piece": [], "released": []}
    for phase, key in (("think", "reasoning_content"), ("answer", "content")):
        want = case["result"][key].strip()
        pieces = "".join(e[1] for e in case["events"] if e[0] == phase)
        if pieces.strip() != want:
            out["piece"].append(key)
        segs = [seg for e in case["events"] for seg in e[2]] + case["tail"]
        joined = "".join(t for p, t in segs if p == phase)
        if phase == "think" and joined.endswith("</think>"):
            joined = joined[:-len("</think>")]
        if joined.strip() != want:
            out["released"].append(key)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=None, help="the first GGUF shard; else the first of QWEN38_SHARDS")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report-rule", action="store_true", help="print the cases that break gogi's piece rule")
    args = ap.parse_args()
    sys.path.insert(0, str(TOOLS))
    import qwen38_reference
    from qwen38_reference import chatfmt
    from qwen38_reference.tokenizer import Tokenizer

    if args.gguf is None:
        args.gguf = Path(qwen38_reference.shards_from_env()[0])

    tok = Tokenizer.from_gguf(args.gguf)
    rnd = random.Random(20261020)
    cases = []
    for name, thinking, toolset, text, ending in replies():
        tools = TOOLSETS[toolset]
        # One token for each byte is the hardest split, and it makes the biggest case. A long text gets it only for
        # the first of its kind, so that the fixtures stay small (the test the_fixtures_stay_small).
        size = len(text.encode())
        modes = ["whole"] + (["cuts"] if size <= 160 else []) + (["bytes"] if size <= 40 else [])
        for n, mode in enumerate(modes):
            ids = make_token_ids(tok, text, mode, rnd)
            assert all(i not in (248046, 248044) for i in ids), name
            case = {"name": f"{name}:{mode}{n}" if mode == "cuts" else f"{name}:{mode}", "thinking": thinking,
                    "tools": toolset, "ids": ids}
            case.update(run_reference(chatfmt, tok, thinking, tools, ids, ending))
            cases.append(case)
    # A cut in the middle of a character: drop the last byte token of a UTF-8 reply.
    for name, thinking, toolset, text, ending in replies():
        if ending == "length" and name.startswith("utf8_cut"):
            ids = make_token_ids(tok, text, "bytes", rnd)[:-1]
            case = {"name": f"{name}:half", "thinking": thinking, "tools": toolset, "ids": ids}
            case.update(run_reference(chatfmt, tok, thinking, TOOLSETS[toolset], ids, ending))
            cases.append(case)
    # Bytes that are not UTF-8: a stray continuation byte, a bad lead byte, a surrogate.
    for name, raw in (("bad_continuation", b"a\x80b"), ("bad_lead", b"a\xffb"), ("bad_surrogate", b"a\xed\xa0\x80b"),
                      ("bad_overlong", b"\xc0\xafx"), ("bad_truncated_then_ascii", b"\xe4\xb8z")):
        ids = [tok.token_to_id[tok._byte_encoder[b]] for b in raw]
        case = {"name": f"utf8_{name}:bytes", "thinking": False, "tools": "none", "ids": ids}
        case.update(run_reference(chatfmt, tok, False, None, ids, "stop"))
        cases.append(case)

    header = {
        "reference_sha": qwen38_reference.digest(),
        "source_files": [qwen38_reference.source(name) for name in ("chatfmt.py", "tokenizer.py")] + ["tools/reply_fixture.py"],
        "gguf": args.gguf.name,
        "python": platform.python_version(),
        "generator": "tools/reply_fixture.py",
        "author": "written by Claude Sonnet 5.5 for this tool, 2026-10-20",
        "stop_ids": [248046, 248044],
    }
    lines = ["{", '"header": ' + json.dumps(header, sort_keys=True) + ",",
             '"toolsets": ' + json.dumps(TOOLSETS, separators=(",", ":")) + ",", '"cases": [']
    lines.append(",\n".join(json.dumps(c, separators=(",", ":")) for c in cases))
    lines += ["]", "}"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"{args.out.name}: {args.out.stat().st_size} bytes, {len(cases)} cases")
    if args.report_rule:
        for kind in ("piece", "released"):
            bad = [(c["name"], check_rule(c)[kind]) for c in cases if check_rule(c)[kind]]
            print(f"{kind}: {len(bad)} of {len(cases)} cases break gogi's rule")
            for name, keys in bad[:12]:
                print("  ", name, keys)


main()
