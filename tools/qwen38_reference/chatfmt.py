# SPDX-License-Identifier: MIT
"""The structure of a reply: thinking, content and Qwen3-Coder XML tool calls, streamed and final.

The model's template (tokenizer.chat_template in the GGUF) fixes the format:
  - thinking on: the generation prompt ends "<think>\\n", so generated text starts inside thinking and "</think>"
    ends it; thinking off: the prompt already holds "<think>\\n\\n</think>\\n\\n" and generated text is the answer;
  - a tool call is "<tool_call>\\n<function=NAME>\\n" + "<parameter=P>\\nVALUE\\n</parameter>\\n" ... +
    "</function>\\n</tool_call>", strings raw and everything else as JSON; content, then a blank line, then the calls.

`ReplyTracker(thinking, tools)`: `feed(text)` gives [(phase, text)] with phase think | answer | tool, holding back only
as much text as could still become a marker ("</think>" while thinking, "<tool_call>" in the answer; with no tools
"<tool_call>" is plain text). Each emitted piece belongs to the phase it was written in: "</think>" is the last of the
thinking, "<tool_call>" the first of the tool text; the fed text is exactly the concatenation of what comes out, plus
`tail`, the held text `finish()` releases. `finish()` returns {reasoning_content, content, tool_calls}: reasoning
and content trimmed the way the template trims them when it renders them back; tool_calls each {name, arguments}
with arguments a parsed JSON object. A call that doesn't parse (unclosed tags, a value that fails its declared
type, an unknown function, an undeclared, repeated or missing parameter, prose outside a call) comes back as
{name (or None), raw: the exact text, error: why}: never dropped, never repaired. Thinking that never closes (a
length stop) is all reasoning_content and content "": the finish reason tells the caller.

Argument semantics follow llama.cpp common/parsers/qwen3-coder.cpp (5cf3a3528): the value is exactly the text between
"<parameter=NAME>\\n" and "\\n</parameter>" (lines 106-107: arg_open ends in a newline; arg_close is
"\\n</parameter>\\n"), with no other trimming; a parameter whose only type is string stays a string (lines 113-114); a
parameter with no string type must parse as JSON of a declared type (lines 111-112); one with string among other
types is JSON when the text parses as one of the others and the raw text otherwise (lines 116-134). A parameter
schema with no type means any type. llama.cpp's grammar admits declared parameters only, and so do we.

`Utf8Stream` turns token bytes (Tokenizer.token_piece) into text: a character split across tokens comes out whole
with the token that completes it, and bad bytes become U+FFFD; nothing raises mid-stream.
"""
from __future__ import annotations

import codecs
import json
import math
import re
from dataclasses import dataclass

THINK_END = "</think>"
CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"
FUNC_CLOSE = "</function>"
PARAM_CLOSE = "\n</parameter>"
ALL_TYPES = frozenset({"string", "integer", "number", "boolean", "null", "object", "array"})

_WS = re.compile(r"\s*")
_FUNC = re.compile(r"\s*<function=([^>\n]+)>\n")
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n")


# ---------------------------------------------------------------- tool schemas
@dataclass(frozen=True)
class _Spec:
    props: dict  # parameter name -> frozenset of declared types
    required: tuple


def _types_of(schema) -> frozenset:
    if not isinstance(schema, dict):
        return ALL_TYPES
    found: set = set()
    t = schema.get("type")
    if isinstance(t, str):
        found.add(t)
    elif isinstance(t, list):
        found.update(t)
    for key in ("anyOf", "oneOf"):
        for sub in schema.get(key) or []:
            found |= _types_of(sub)
    if not found:
        return ALL_TYPES
    unknown = found - ALL_TYPES
    if unknown:
        raise ValueError(f"unsupported JSON schema type(s) {sorted(unknown)}")
    return frozenset(found)


def _specs(tools: list) -> dict:
    out = {}
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            raise ValueError(f"a tool needs a function with a name, got {tool!r}")
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        if not isinstance(props, dict):
            raise ValueError(f"tool {fn['name']!r}: parameters.properties must be an object")
        out[fn["name"]] = _Spec({k: _types_of(v) for k, v in props.items()}, tuple(params.get("required") or ()))
    return out


# ---------------------------------------------------------------- values
def _no_constant(name: str):
    raise ValueError(f"{name} is not JSON")


def _finite_float(text: str) -> float:
    v = float(text)
    if not math.isfinite(v):
        raise ValueError(f"{text} is not a finite number")
    return v


def _is(v, t: str) -> bool:
    if t == "integer":
        return type(v) is int
    if t == "number":
        return type(v) in (int, float)
    if t == "boolean":
        return type(v) is bool
    if t == "null":
        return v is None
    if t == "object":
        return isinstance(v, dict)
    if t == "array":
        return isinstance(v, list)
    return False


def _convert(value: str, types: frozenset):
    if types == {"string"}:
        return value
    try:
        v = json.loads(value, parse_constant=_no_constant, parse_float=_finite_float)
    except ValueError as e:
        if "string" in types:
            return value
        raise ValueError(f"not JSON ({e})") from None
    if any(_is(v, t) for t in types if t != "string"):
        return v
    if "string" in types:
        return value
    raise ValueError(f"JSON {type(v).__name__} {value!r} is not one of {sorted(types)}")


# ---------------------------------------------------------------- tool calls
class _Malformed(Exception):
    def __init__(self, why: str, name: str | None):
        super().__init__(why)
        self.why, self.name = why, name


def _parse_one(text: str, pos: int, specs: dict) -> tuple[int, dict]:
    """The call starting at text[pos] (CALL_OPEN). Returns (end, entry); a call that can't be read runs to the next
    CALL_OPEN (or the end) so the rest of the reply is still read."""
    name = None
    try:
        m = _FUNC.match(text, pos + len(CALL_OPEN))
        if not m:
            raise _Malformed("expected <function=NAME> after <tool_call>", None)
        name = m.group(1)
        i = m.end()
        params: list[tuple[str, str]] = []
        while True:
            i = _WS.match(text, i).end()
            if text.startswith(FUNC_CLOSE, i):
                i += len(FUNC_CLOSE)
                break
            m = _PARAM.match(text, i)
            if not m:
                raise _Malformed("expected <parameter=NAME> or </function>", name)
            pname, vstart, j = m.group(1), m.end(), m.end()
            while True:
                vend = text.find(PARAM_CLOSE, j)
                if vend < 0:
                    raise _Malformed(f"parameter {pname!r} is not closed (no </parameter>)", name)
                after = _WS.match(text, vend + len(PARAM_CLOSE)).end()
                if text.startswith("<parameter=", after) or text.startswith(FUNC_CLOSE, after):
                    break
                j = vend + 1  # the "</parameter>" was part of the value
            value = text[vstart:vend]
            params.append((pname, value))
            i = after
        i = _WS.match(text, i).end()
        if not text.startswith(CALL_CLOSE, i):
            raise _Malformed("tool call is not closed (no </tool_call>)", name)
        end = i + len(CALL_CLOSE)
    except _Malformed as e:
        nxt = text.find(CALL_OPEN, pos + 1)
        end = len(text) if nxt < 0 else nxt
        return end, {"name": e.name, "raw": text[pos:end], "error": e.why}
    raw = text[pos:end]
    try:
        spec = specs.get(name)
        if spec is None:
            raise ValueError(f"unknown function {name!r} (tools: {sorted(specs)})")
        arguments: dict = {}
        for pname, value in params:
            if pname in arguments:
                raise ValueError(f"parameter {pname!r} given twice")
            if pname not in spec.props:
                raise ValueError(f"parameter {pname!r} is not declared by {name!r} (declared: {sorted(spec.props)})")
            try:
                arguments[pname] = _convert(value, spec.props[pname])
            except ValueError as e:
                raise ValueError(f"parameter {pname!r} ({'|'.join(sorted(spec.props[pname]))}): {e}") from None
        missing = [r for r in spec.required if r not in arguments]
        if missing:
            raise ValueError(f"missing required parameter(s) {missing} of {name!r}")
    except ValueError as e:
        return end, {"name": name, "raw": raw, "error": str(e)}
    return end, {"name": name, "arguments": arguments}


def _parse_calls(text: str, specs: dict) -> list[dict]:
    calls: list[dict] = []
    pos = 0
    while True:
        pos = _WS.match(text, pos).end()
        if pos >= len(text):
            return calls
        if text.startswith(CALL_OPEN, pos):
            end, entry = _parse_one(text, pos, specs)
            calls.append(entry)
        else:
            nxt = text.find(CALL_OPEN, pos)
            end = len(text) if nxt < 0 else nxt
            bad = {"name": None, "raw": text[pos:end], "error": "text outside a tool call"}
            calls.append(bad)
        pos = end


# ---------------------------------------------------------------- streaming
def _marker_prefix_len(buf: str, marker: str) -> int:
    """The longest proper prefix of `marker` that `buf` ends with."""
    for k in range(min(len(marker) - 1, len(buf)), 0, -1):
        if buf.endswith(marker[:k]):
            return k
    return 0


def _coalesce(pieces: list) -> list:
    out: list = []
    for ph, tx in pieces:
        if out and out[-1][0] == ph:
            out[-1] = (ph, out[-1][1] + tx)
        else:
            out.append((ph, tx))
    return out


class ReplyTracker:
    def __init__(self, thinking: bool, tools: list | None):
        if not isinstance(thinking, bool):
            raise TypeError(f"thinking must be a bool, got {thinking!r}")
        self.thinking = thinking
        self.specs = _specs(tools) if tools else None
        self.tail: list = []  # the held text finish() released
        self._phase = "think" if thinking else "answer"
        self._buf = ""
        self._text: list[str] = []
        self._pre = 0  # characters of a prefill at the head of the text: literal, never searched for a marker
        self._result = None

    @property
    def phase(self) -> str:
        """The phase the reply is in now (think | answer | tool), after everything fed so far; text held back as a
        possible marker still counts as this phase until the marker completes."""
        return self._phase

    def _marker(self):
        if self._phase == "think":
            return THINK_END
        if self._phase == "answer" and self.specs is not None:
            return CALL_OPEN
        return None

    def prefill(self, text: str) -> list:
        """Text the client wrote at the start of the reply, before the model's first token. It belongs to the reply
        (reasoning_content / content cover it) and to the current phase, but it is literal: the service encoded it as
        content, so a "</think>" or "<tool_call>" in it is plain text the model never saw as a control token and
        ends no phase here or in finish(). Only before anything else is fed, once."""
        if self._result is not None or self._text:
            raise RuntimeError("prefill comes first and only once: text was already fed")
        self._text.append(text)
        self._pre = len(text)
        return [(self._phase, text)] if text else []

    def feed(self, text: str) -> list:
        if self._result is not None:
            raise RuntimeError("feed after finish")
        self._text.append(text)
        self._buf += text
        out: list = []
        while self._buf:
            marker = self._marker()
            if marker is None:
                out.append((self._phase, self._buf))
                self._buf = ""
                break
            i = self._buf.find(marker)
            if i >= 0:
                if i:
                    out.append((self._phase, self._buf[:i]))
                if self._phase == "think":
                    out.append(("think", marker))  # the end marker is the last of the thinking
                    self._buf = self._buf[i + len(marker):]
                    self._phase = "answer"
                else:
                    self._buf = self._buf[i:]  # the start marker is the first of the tool text
                    self._phase = "tool"
                continue
            hold = _marker_prefix_len(self._buf, marker)
            emit = self._buf[:len(self._buf) - hold]
            if emit:
                out.append((self._phase, emit))
            self._buf = self._buf[len(self._buf) - hold:]
            break
        return _coalesce(out)

    def _flush(self) -> list:
        out = [(self._phase, self._buf)] if self._buf else []
        self._buf = ""
        return out

    def finish(self) -> dict:
        if self._result is not None:
            return self._result
        self.tail = self._flush()
        text = "".join(self._text)
        reasoning, rest = "", text
        if self.thinking:
            i = text.find(THINK_END, self._pre)
            if i < 0:
                self._result = {"reasoning_content": text.strip(), "content": "", "tool_calls": []}
                return self._result
            reasoning, rest = text[:i], text[i + len(THINK_END):]
            reasoning = reasoning.strip()
        content, calls = rest, []
        if self.specs is not None:
            c = rest.find(CALL_OPEN, 0 if self.thinking else self._pre)  # (thinking: rest starts after the prefill)
            if c >= 0:
                content = rest[:c]
                calls = _parse_calls(rest[c:], self.specs)
        content = content.strip()
        self._result = {"reasoning_content": reasoning, "content": content, "tool_calls": calls}
        return self._result


class Utf8Stream:
    def __init__(self):
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, raw: bytes) -> str:
        return self._dec.decode(raw)

    def flush(self) -> str:
        return self._dec.decode(b"", final=True)
