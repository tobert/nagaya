# SPDX-License-Identifier: MIT
"""The reference of Qwen3.8-Flash-Next on the CPU, and the parts of its first server that the fixture tools read.

The fixture tools of the Qwen3.8 crates import this package. It holds:

- `reference`, `glue`: the model as transformers' own `qwen4_exp` code (pinned, not copied), run one layer at a time
  on the weights of the GGUF. gguf-py reads and dequantizes the GGUF.
- `tokenizer`: the byte-level BPE tokenizer, from the GGUF's metadata.
- `template`, `common`, `chatfmt`, `sampling`: the chat template, the escape of content, the reply parser and the
  host sampler of the first Qwen3.8 server.
- `attn_mask`: the float64 reference of the attention cell mask.
- `config.json`: the model's config, from its Hugging Face repository (Apache-2.0), unchanged.

The code comes from the repository of the first Qwen3.8 engine, under MIT. `common.piece_text` is a port of llama.cpp
code and keeps its notice. Install `requirements.txt` into a venv, and set `QWEN38_SHARDS` to the three GGUF shards.
"""

import hashlib
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def shards_from_env() -> list[str]:
    """The three GGUF shards of the model, in order, from `QWEN38_SHARDS` (paths with ':' between them)."""
    value = os.environ.get("QWEN38_SHARDS", "")
    paths = [p for p in value.split(":") if p]
    if len(paths) != 3:
        raise SystemExit(f"QWEN38_SHARDS names {len(paths)} paths, and the model has 3 shards. Set it to the three "
                         "GGUF shards, in the order 00001, 00002, 00003, with ':' between them.")
    return paths


def digest() -> str:
    """The first 16 hex digits of the SHA-256 of the package: the name and the bytes of each file but the tests, in
    name order. A fixture records it as `reference_sha`, so that it names the exact reference that made it."""
    h = hashlib.sha256()
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.suffix in (".py", ".json", ".txt") and not path.name.startswith("test_"):
            h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def source(name: str) -> str:
    """The path of a module of this package, from the root of the repository, for a fixture's `source_files`."""
    path = HERE / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not a file of the reference")
    return str(path.relative_to(HERE.parent.parent))
