# SPDX-License-Identifier: MIT
"""What the service modules share: the exception a request the service can't serve raises (HTTP 400), the text of
a token piece as llama-server's JSON shows it, and text to tokens: raw text with its special-token text parsed, a chat
with its content escaped."""
from __future__ import annotations


class BadRequest(ValueError):
    """The request asks for something this service doesn't do; nothing ran. HTTP 400."""


class TooLarge(Exception):
    """The answer would be too large to send this way (a trace array past the JSON cap). HTTP 413."""


# piece_text's loop is ported from llama.cpp tools/server/server-common.cpp validate_utf8 (5cf3a3528).
# Copyright (c) 2023-2026 The ggml authors. MIT License.
def piece_text(raw: bytes) -> str:
    """A token's bytes as llama-server's JSON shows them: validate_utf8 (server-common.cpp) drops a multi-byte
    character cut off at the end, and the JSON dump replaces any other invalid byte."""
    n = len(raw)
    for i in range(1, min(4, n) + 1):
        c = raw[n - i]
        if (c & 0xE0) == 0xC0 and i < 2 or (c & 0xF0) == 0xE0 and i < 3 or (c & 0xF8) == 0xF0 and i < 4:
            n -= i
            break
    return raw[:n].decode("utf-8", "replace")


def encode_text(tok, text: str, what: str) -> list[int]:
    """Raw text's token ids, special-token text parsed (the client's own structure: a prompt, a suffix, /tokenize);
    a reserved character is a BadRequest naming `what`."""
    try:
        return tok.encode(text)
    except ValueError as e:
        raise BadRequest(f"{what}: {e}") from e


def _escape_tree(tok, x, where: str):
    if isinstance(x, str):
        try:
            x.encode("utf-8")  # a lone surrogate (JSON allows "\ud800") is not text a tokenizer can take
            return tok.escape(x)
        except UnicodeEncodeError as e:
            raise BadRequest(f"{where}: not well-formed Unicode ({e.reason} at {e.start})") from e
        except ValueError as e:
            raise BadRequest(f"{where}: {e}") from e
    if isinstance(x, dict):
        return {_escape_tree(tok, k, f"{where} key"): _escape_tree(tok, v, f"{where}.{k}") for k, v in x.items()}
    if isinstance(x, list):
        return [_escape_tree(tok, v, f"{where}[{i}]") for i, v in enumerate(x)]
    return x


def escape_content(tok, text: str, where: str) -> str:
    """One string of content made safe to place after rendered text (its special-token text swapped for the escape
    characters `tok.encode_escaped` reads as plain text); a BadRequest naming `where` for what can't be escaped."""
    return _escape_tree(tok, text, where)


def render_messages(template, tok, messages, tools, thinking: bool, generation_prompt: bool,
                    reasoning_effort: str | None = None) -> tuple[str, list[int]]:
    """(escaped text, token ids) for a chat (content must not forge the roles of the template).
    Every string in messages and tools is content: its special-token text is escaped before the template sees it, so
    the template's own markers are the only special tokens in the ids. The escaped text is what a held context keeps
    (a suffix re-encodes it with `encode_escaped`); `tok.unescape` gives the text it stands for.
    reasoning_effort (the chat template's own variable: xhigh when unset, medium, low) is passed only when given:
    template.py is copied unchanged and its render() has no such parameter, so this calls its jinja
    template with the extra variable (the same kwargs render() builds). None leaves every other caller as it was."""
    import jinja2
    msgs = _escape_tree(tok, messages, "messages")
    tls = _escape_tree(tok, tools, "tools") if tools else tools
    try:
        if reasoning_effort is None:
            text = template.render(msgs, tools=tls, thinking=thinking, generation_prompt=generation_prompt)
        else:
            kw = {"messages": msgs, "add_generation_prompt": generation_prompt, "enable_thinking": thinking,
                  "reasoning_effort": reasoning_effort}
            if tls:
                kw["tools"] = tls
            text = template._t.render(**kw)
    except jinja2.TemplateError as e:
        raise BadRequest(f"the chat template refused these messages: {e}") from e
    return text, tok.encode_escaped(text)
