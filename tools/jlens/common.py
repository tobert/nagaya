# SPDX-License-Identifier: MIT
"""Shared utilities for J-Lens fitting, merging, and validation.

Implements the canonical safetensors format of docs/engine-abi.md (D10), deterministic WikiText
detokenization and prompt selection matching tools/lfm3_jlens_fit.py, and shard partitioning.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import uuid
from pathlib import Path
from typing import Any

import numpy as np

SEED = 20261011
ALIGN = 4096
WEIGHT_HASH_RE = re.compile(r"(?:blake3|spot-sha256):[0-9a-f]{64}")

DTYPE_MAP = {
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
}


def detok(text: str) -> str:
    """Restore WikiText spacing back to prose: ' , ' -> ', ', '( x )' -> '(x)', '1 @,@ 000' -> '1,000'."""
    text = re.sub(r" @(.)@ ", r"\1", text)
    text = re.sub(r" ([,.;:!?)\]}]|'s|'t|'re|'ve|'ll|'d|n't)", r"\1", text)
    text = re.sub(r"([(\[{]) ", r"\1", text)
    return re.sub(r'" (.*?) "', r'"\1"', text)


def choose(paragraphs: list[str], n: int, seed: int = SEED) -> list[str]:
    """Return the first n paragraphs of one fixed permutation so larger n extends smaller n."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n > len(paragraphs):
        raise ValueError(f"requested {n} prompts, but corpus only has {len(paragraphs)} paragraphs")
    order = np.random.default_rng(seed).permutation(len(paragraphs))
    return [paragraphs[int(i)] for i in order[:n]]


def shard_indices(total: int, shard_index: int, num_shards: int) -> list[int]:
    """Deterministic contiguous partition of range(total) across num_shards workers."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"shard_index {shard_index} out of range [0, {num_shards})")
    base, rem = divmod(total, num_shards)
    start = shard_index * base + min(shard_index, rem)
    count = base + (1 if shard_index < rem else 0)
    return list(range(start, start + count))


def validate_weight_hash(weight_hash: str) -> None:
    """Validate that weight_hash has a supported nagaya identity digest format (blake3: or spot-sha256:)."""
    if not WEIGHT_HASH_RE.fullmatch(weight_hash):
        raise ValueError(
            f"weight_hash {weight_hash!r} is not a valid identity digest; "
            "copy weight_hash from GET /council/v2/identity (format: blake3:<64 hex> or spot-sha256:<64 hex>)"
        )


def canonical_safetensors(
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
    metadata: dict[str, Any],
) -> bytes:
    """Serialize tensors and metadata into canonical safetensors (engine-abi D10).

    Rules:
      - Keys sorted in UTF-8 byte order.
      - Compact JSON header padded with ASCII spaces so (8 + header_len) % 4096 == 0.
      - Tensor payloads concatenated in key order with no padding between tensors.
    """
    header: dict[str, Any] = {}
    offset = 0
    names = sorted(tensors, key=lambda name: name.encode("utf-8"))
    for name in names:
        dtype, shape, data = tensors[name]
        header[name] = {
            "data_offsets": [offset, offset + len(data)],
            "dtype": dtype,
            "shape": list(shape),
        }
        offset += len(data)
    header["__metadata__"] = {str(key): str(value) for key, value in metadata.items()}
    text = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    padding = (ALIGN - (8 + len(text)) % ALIGN) % ALIGN
    out = bytearray(struct.pack("<Q", len(text) + padding))
    out += text + b" " * padding
    for name in names:
        out += tensors[name][2]
    return bytes(out)


def parse_canonical_safetensors(data: bytes) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Parse and strictly validate a canonical D10 safetensors buffer.

    Raises ValueError if alignment, header canonicalization, offsets, lengths, or finiteness fail.
    """
    if len(data) < 8:
        raise ValueError(f"safetensors buffer too short ({len(data)} bytes)")
    header_len = struct.unpack("<Q", data[:8])[0]
    if 8 + header_len > len(data):
        raise ValueError(f"safetensors header length {header_len} exceeds file length {len(data)}")
    if (8 + header_len) % ALIGN != 0:
        raise ValueError(
            f"safetensors header is not D10 4 KiB aligned: 8 + {header_len} = {8 + header_len}"
        )

    raw_header = data[8 : 8 + header_len]
    stripped = raw_header.rstrip(b" ")
    pad = raw_header[len(stripped) :]
    if pad != b" " * len(pad):
        raise ValueError("safetensors header padding contains non-space bytes")
    if len(pad) >= ALIGN:
        raise ValueError(f"safetensors header has excessive padding ({len(pad)} bytes)")

    try:
        header = json.loads(stripped.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"safetensors header is not valid UTF-8 JSON: {exc}") from exc

    canonical_json = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if stripped != canonical_json:
        raise ValueError("safetensors header JSON is not canonical (unsorted keys or extra whitespace)")

    meta_raw = header.get("__metadata__", {})
    if not isinstance(meta_raw, dict):
        raise ValueError("__metadata__ must be a JSON object")
    metadata = {str(k): str(v) for k, v in meta_raw.items()}

    tensor_names = [k for k in header if k != "__metadata__"]
    sorted_names = sorted(tensor_names, key=lambda name: name.encode("utf-8"))
    if tensor_names != sorted_names:
        raise ValueError("safetensors tensor keys are not in UTF-8 byte order")

    payload = data[8 + header_len :]
    expected_offset = 0
    tensors: dict[str, np.ndarray] = {}
    for name in sorted_names:
        entry = header[name]
        dtype_name = entry.get("dtype")
        if dtype_name not in DTYPE_MAP:
            raise ValueError(f"tensor {name!r} has unsupported dtype {dtype_name!r}")
        np_dtype = DTYPE_MAP[dtype_name]
        shape = tuple(int(x) for x in entry["shape"])
        start, end = entry["data_offsets"]
        if start != expected_offset:
            raise ValueError(
                f"tensor {name!r} starts at offset {start}, expected {expected_offset} (no inter-tensor padding allowed)"
            )
        n_elems = int(np.prod(shape)) if shape else 1
        n_bytes = n_elems * np_dtype.itemsize
        if end - start != n_bytes:
            raise ValueError(
                f"tensor {name!r} byte length {end - start} does not match shape {shape} * {np_dtype.itemsize}"
            )
        if end > len(payload):
            raise ValueError(f"tensor {name!r} offset {end} exceeds payload length {len(payload)}")
        arr = np.frombuffer(payload[start:end], dtype=np_dtype).reshape(shape).copy()
        if not np.all(np.isfinite(arr)):
            bad = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
            raise ValueError(f"tensor {name!r} contains {bad} non-finite (NaN/Inf) values")
        tensors[name] = arr
        expected_offset = end

    if expected_offset != len(payload):
        raise ValueError(f"trailing bytes after last tensor: {len(payload) - expected_offset} bytes")

    return tensors, metadata


def load_corpus(
    corpus: str,
    n: int,
    *,
    min_chars: int = 700,
    seed: int = SEED,
    cache_json: Path | None = None,
) -> tuple[list[str], str, str]:
    """Load and deterministically select n prompts from a local JSON file or Hugging Face WikiText-103.

    Returns (prompts, corpus_sha256, corpus_description).
    """
    corpus_path = Path(corpus)
    if corpus_path.is_file():
        raw = corpus_path.read_bytes()
        paragraphs = [detok(p) for p in json.loads(raw)]
        return choose(paragraphs, n, seed=seed), hashlib.sha256(raw).hexdigest(), str(corpus_path.name)

    if corpus != "wikitext-103":
        raise FileNotFoundError(
            f"corpus {corpus!r} is neither an existing JSON file nor 'wikitext-103'"
        )

    if cache_json is not None and cache_json.is_file():
        raw = cache_json.read_bytes()
        paragraphs = json.loads(raw)
        return choose(paragraphs, n, seed=seed), hashlib.sha256(raw).hexdigest(), "Salesforce/wikitext:wikitext-103-raw-v1"

    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    paragraphs = []
    for row in ds:
        text = row["text"].strip()
        if len(text) > min_chars and not (text.startswith("=") and text.endswith("=")):
            paragraphs.append(detok(text))
    canonical_bytes = json.dumps(paragraphs, ensure_ascii=False).encode("utf-8")
    if cache_json is not None:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_json.with_name(f"{cache_json.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_bytes(canonical_bytes)
        tmp.replace(cache_json)
    return (
        choose(paragraphs, n, seed=seed),
        hashlib.sha256(canonical_bytes).hexdigest(),
        "Salesforce/wikitext:wikitext-103-raw-v1",
    )
