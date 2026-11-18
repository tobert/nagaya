# SPDX-License-Identifier: MIT
"""Render a conversation through a GGUF's own chat template, in-process.

llama-server renders with minja; we render with jinja2, set up the way
Hugging Face transformers sets it up (trim_blocks, lstrip_blocks, a tojson
that keeps non-ASCII and HTML characters as they are). Parity with the server
is checked against its recorded renders (tests/test_template.py), because a
render that drifts from the one the model primed under is a silent confound.
"""

from __future__ import annotations

import json
from functools import cache

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


def _raise(msg: str):
    raise jinja2.TemplateError(msg)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


@cache
def gguf_template(path: str) -> str:
    """The `tokenizer.chat_template` string from a GGUF's metadata (no weights are read)."""
    from gguf import GGUFReader

    field = GGUFReader(path).fields["tokenizer.chat_template"]
    return bytes(field.parts[field.data[0]]).decode()


class Template:
    def __init__(self, source: str):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        self._t = env.from_string(source)

    @classmethod
    def from_gguf(cls, path: str) -> Template:
        return cls(gguf_template(path))

    def render(self, messages: list[dict], tools: list[dict] | None = None, thinking: bool = True,
               generation_prompt: bool = True) -> str:
        """The prompt text for `messages`; with generation_prompt, it ends at the
        start of the next assistant turn (inside an open <think> when thinking)."""
        kw = {"messages": messages, "add_generation_prompt": generation_prompt, "enable_thinking": thinking}
        if tools:
            kw["tools"] = tools
        return self._t.render(**kw)
