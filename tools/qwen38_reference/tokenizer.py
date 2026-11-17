"""A pure-Python GPT-2-style byte-level BPE tokenizer for Qwen3.8-Flash-Next,
built entirely from one GGUF shard's own metadata: `tokenizer.ggml.tokens`,
`tokenizer.ggml.merges`, `tokenizer.ggml.token_type`, and the `qwen35`
pre-tokenizer (`tokenizer.ggml.pre`).

The engine takes token ids and never runs this module. The oracle tools use it
to turn prompts into ids, with no llama.cpp at run time.

It follows these files of llama.cpp, which it reads as a reference only:
  - src/llama-vocab.cpp: the `qwen35` pre-tokenizer regex (LLAMA_VOCAB_PRE_TYPE_QWEN35)
    and `tokenizer_st_partition` (how special/control/user-defined tokens are
    matched as whole tokens before BPE, longest-text-first, when parse_special
    is on).
  - src/unicode.cpp: `unicode_regex_split_custom_qwen35`, a hand-rolled
    matcher for that exact regex (std::regex has no \\p{L} support), and the
    GPT-2 byte<->unicode map (`unicode_byte_to_utf8_map`) confirmed against
    the vocab itself (id 220 is 'Ġ', the classic GPT-2 space token).

A llama.cpp client tokenizes with
`llama_tokenize(vocab, text, add_special=True, parse_special=True)`. This
GGUF has `tokenizer.ggml.add_bos_token = False` and no `add_eos_token` key
(so llama.cpp defaults it to false too), so `add_special` adds nothing for
this vocab -- there is no BOS/EOS to insert. `parse_special` is the flag
that matters here: it is what makes control/user-defined tokens such as
`<|im_start|>` and `<think>` match as whole tokens instead of falling into
BPE. That is what this module reproduces.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

try:
    import regex as _re_unicode  # supports \p{L}, \p{M}, \p{N} -- stdlib re does not
except ImportError as exc:  # pragma: no cover - environment problem, not a fallback
    raise ImportError(
        "the tokenizer needs the third-party `regex` module (for \\p{L}/\\p{M}/\\p{N} "
        "Unicode-property matching in the qwen35 pre-tokenizer pattern); stdlib `re` cannot "
        "express it. Install tools/qwen38_reference/requirements.txt."
    ) from exc

# tokenizer.ggml.token_type values (llama.cpp's LLAMA_TOKEN_ATTR_* / gguf-py's TokenType,
# read as the plain ints the GGUF stores).
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER_DEFINED = 4
TOKEN_TYPE_UNUSED = 5
TOKEN_TYPE_BYTE = 6

# Special/control tokens are matched as whole tokens before BPE (llama.cpp's
# tokenizer_st_partition with parse_special=true, cache_special_tokens built
# from exactly these two attrs -- UNKNOWN doesn't occur in this vocab).
SPECIAL_TOKEN_TYPES = (TOKEN_TYPE_CONTROL, TOKEN_TYPE_USER_DEFINED)

# Content escaping (message content must not forge the roles of the chat template). Unicode's 66
# noncharacters are reserved for a program's internal use and never interchanged, so they can stand for special-token
# text inside content: `escape` swaps each special token's text for its own noncharacter (in special-id order),
# `encode_escaped` encodes each noncharacter as its token's text through BPE (specials off), and raw text holding one
# is refused, never reinterpreted.
RESERVED = [chr(c) for c in range(0xFDD0, 0xFDF0)] + [
    chr((plane << 16) | low) for plane in range(17) for low in (0xFFFE, 0xFFFF)
]
_RESERVED_RE = re.compile("[" + "".join(re.escape(c) for c in RESERVED) + "]")

# The `qwen35` pre-tokenizer pattern, verbatim from llama-vocab.cpp's
# LLAMA_VOCAB_PRE_TYPE_QWEN35 case (the comment there marks it as "original
# regex from tokenizer.json"; unicode.cpp's unicode_regex_split_custom_qwen35
# is a hand-rolled matcher for this same pattern, written because std::regex
# can't do \p{L}/\p{M}/\p{N}). The `regex` module can compile this directly.
QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

def _refuse_reserved(text: str, allowed) -> None:
    for m in _RESERVED_RE.finditer(text):
        if m.group() not in allowed:
            raise ValueError(
                f"U+{ord(m.group()):04X} at offset {m.start()} is a Unicode noncharacter, reserved here for escaping "
                f"special-token text; refused"
            )


def _bytes_to_unicode() -> dict[int, str]:
    """The standard GPT-2 byte->unicode map (OpenAI's `bytes_to_unicode`).

    Printable Latin-1-ish bytes map to themselves; the rest (control chars,
    etc.) map to codepoints starting at 256, so every byte value gets a
    distinct, single-codepoint, printable stand-in. This is a fixed table,
    not something the GGUF stores -- but it is the table the GGUF's own
    vocab was built with (checked below), matching unicode.cpp's
    `unicode_byte_to_utf8_map`.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(
        range(ord("\xae"), ord("\xff") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class Tokenizer:
    """GPT-2-style byte-level BPE over Qwen3.8-Flash-Next's GGUF vocab.

    Special/control/user-defined tokens (e.g. `<|im_start|>`, `<think>`) are
    matched as whole tokens before BPE runs, longest-text-first -- the same
    order llama.cpp's cache_special_tokens uses, and the only order that can
    disambiguate one special token being a text-prefix of another.
    """

    def __init__(
        self,
        id_to_token: list[str],
        token_type: list[int],
        merges: list[tuple[str, str]],
    ) -> None:
        if len(id_to_token) != len(token_type):
            raise ValueError(
                f"tokens ({len(id_to_token)}) and token_type ({len(token_type)}) length mismatch"
            )

        self.id_to_token = id_to_token
        self.token_type = token_type
        self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(id_to_token)}
        if len(self.token_to_id) != len(id_to_token):
            raise ValueError("duplicate token text in vocab: BPE lookup would be ambiguous")

        self.merge_rank: dict[tuple[str, str], int] = {pair: i for i, pair in enumerate(merges)}

        self.special_ids = {
            i for i, t in enumerate(token_type) if t in SPECIAL_TOKEN_TYPES
        }
        # Longest text first, matching llama.cpp's cache_special_tokens sort --
        # required so e.g. a token that is a text-prefix of another special
        # token never steals its match.
        special_texts = sorted(
            (id_to_token[i] for i in self.special_ids), key=len, reverse=True
        )
        if special_texts:
            self._special_re = re.compile(
                "(" + "|".join(re.escape(t) for t in special_texts) + ")"
            )
        else:
            self._special_re = None

        ordered = [id_to_token[i] for i in sorted(self.special_ids)]
        if len(ordered) > len(RESERVED):
            raise ValueError(
                f"{len(ordered)} special tokens but only {len(RESERVED)} reserved characters to escape them with"
            )
        self._escape_of = {t: RESERVED[k] for k, t in enumerate(ordered)}
        self._text_of = {RESERVED[k]: t for k, t in enumerate(ordered)}
        self._escaped_re = re.compile(
            "[" + "".join(re.escape(c) for c in self._text_of) + "]"
        ) if ordered else None

        self._byte_encoder = _bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

        self._pretok_re = _re_unicode.compile(QWEN35_PATTERN)

        self._bpe_cache: dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_gguf(cls, path: str | Path) -> "Tokenizer":
        """Build a tokenizer from one GGUF shard's metadata (tensors are
        never touched -- only `tokenizer.ggml.*` fields are read)."""
        import gguf

        reader = gguf.GGUFReader(str(path))
        fields = reader.fields

        pre = cls._read_str_field(fields, "tokenizer.ggml.pre")
        if pre != "qwen35":
            raise ValueError(
                f"this tokenizer only implements the qwen35 pre-tokenizer; "
                f"GGUF says tokenizer.ggml.pre={pre!r}. This tokenizer is for one model."
            )
        model = cls._read_str_field(fields, "tokenizer.ggml.model")
        if model != "gpt2":
            raise ValueError(
                f"expected a GPT-2-style BPE vocab (tokenizer.ggml.model=gpt2), got {model!r}"
            )

        tokens_field = fields["tokenizer.ggml.tokens"]
        type_field = fields["tokenizer.ggml.token_type"]
        merges_field = fields["tokenizer.ggml.merges"]

        n_tokens = len(tokens_field.data)
        if len(type_field.data) != n_tokens:
            raise ValueError(
                f"tokenizer.ggml.tokens ({n_tokens}) and tokenizer.ggml.token_type "
                f"({len(type_field.data)}) length mismatch in the GGUF"
            )

        id_to_token = [cls._array_str(tokens_field, i) for i in range(n_tokens)]
        token_type = [int(type_field.parts[type_field.data[i]][0]) for i in range(n_tokens)]

        merges = []
        for i in range(len(merges_field.data)):
            raw = cls._array_str(merges_field, i)
            parts = raw.split(" ")
            if len(parts) != 2:
                raise ValueError(f"merge {i} is not exactly two space-separated pieces: {raw!r}")
            merges.append((parts[0], parts[1]))

        return cls(id_to_token, token_type, merges)

    @staticmethod
    def _read_str_field(fields, key: str) -> str:
        field = fields.get(key)
        if field is None:
            raise KeyError(f"GGUF is missing metadata key {key!r}")
        part_idx = field.data[0]
        return bytes(field.parts[part_idx]).decode("utf-8")

    @staticmethod
    def _array_str(field, idx: int) -> str:
        part_idx = field.data[idx]
        return bytes(field.parts[part_idx]).decode("utf-8")

    # -- pickle cache (in-process only unless a path is given) --------------

    def save_cache(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump((self.id_to_token, self.token_type, list(self.merge_rank)), f)

    @classmethod
    def from_cache(cls, path: str | Path) -> "Tokenizer":
        with Path(path).open("rb") as f:
            id_to_token, token_type, merges = pickle.load(f)
        return cls(id_to_token, token_type, merges)

    # -- BPE ------------------------------------------------------------

    def _bpe(self, word: str) -> tuple[str, ...]:
        """Merge one byte-encoded word into vocab pieces, greedily applying
        the lowest-rank (highest-priority) merge present, same as the
        classic GPT-2 BPE algorithm the GGUF's merge list encodes."""
        cached = self._bpe_cache.get(word)
        if cached is not None:
            return cached

        symbols = list(word)
        while len(symbols) > 1:
            best_rank = None
            best_i = None
            for i in range(len(symbols) - 1):
                rank = self.merge_rank.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i is None:
                break
            symbols[best_i : best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]

        result = tuple(symbols)
        self._bpe_cache[word] = result
        return result

    # -- public API -------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Token ids for raw text, special-token text parsed as its token (parse_special). Refuses a reserved
        character (ValueError)."""
        _refuse_reserved(text, set())
        return self._encode(text)

    def encode_escaped(self, text: str) -> list[int]:
        """Token ids for text that may hold `escape`d content: special-token text parsed as its token, as in
        `encode`, and each escape character encoded as its special token's text through BPE (specials off). Refuses a
        reserved character that escapes nothing (ValueError)."""
        _refuse_reserved(text, self._text_of.keys())
        return self._encode(text)

    def escape(self, text: str) -> str:
        """Content made safe to place in a rendered template: each special token's text (longest first, as `encode`
        matches them) swapped for its escape character. Refuses a reserved character already in the text."""
        _refuse_reserved(text, set())
        if self._special_re is None:
            return text
        return self._special_re.sub(lambda m: self._escape_of[m.group()], text)

    def unescape(self, text: str) -> str:
        """The text `escape` (or a render of escaped content) stands for: each escape character back to its token's
        text. Refuses a reserved character that escapes nothing."""
        _refuse_reserved(text, self._text_of.keys())
        if self._escaped_re is None:
            return text
        return self._escaped_re.sub(lambda m: self._text_of[m.group()], text)

    def _encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk, is_special in self._split_special(text):
            if is_special:
                ids.append(self.token_to_id[chunk])
                continue
            if self._escaped_re is not None:
                chunk = self._escaped_re.sub(lambda m: self._text_of[m.group()], chunk)
            for word in self._pretok_re.findall(chunk):
                if not word:
                    continue
                byte_str = "".join(self._byte_encoder[b] for b in word.encode("utf-8"))
                for piece in self._bpe(byte_str):
                    token_id = self.token_to_id.get(piece)
                    if token_id is None:
                        raise RuntimeError(
                            f"BPE produced a piece not in the vocab: {piece!r} "
                            f"(from word {word!r}); this is a bug in the merge order or the "
                            f"vocab build, not something to fall back silently on."
                        )
                    ids.append(token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        for token_id in ids:
            if token_id in self.special_ids:
                buf += self.id_to_token[token_id].encode("utf-8")
                continue
            piece = self.id_to_token[token_id]
            try:
                buf += bytes(self._byte_decoder[ch] for ch in piece)
            except KeyError as exc:
                raise RuntimeError(
                    f"token {token_id} ({piece!r}) has a character outside the GPT-2 "
                    f"byte-to-unicode table -- decode() cannot map it back to bytes."
                ) from exc
        return buf.decode("utf-8")

    def token_piece(self, token_id: int, special: bool = False) -> bytes:
        """One token's bytes, as llama.cpp's token_to_piece gives them (src/llama-vocab.cpp, the BPE branch):
        control and unknown tokens are empty unless `special`; user-defined tokens are their text; normal
        tokens are their byte-decoded text; any other type (unused, byte) is empty, as llama.cpp suppresses it."""
        t = self.token_type[token_id]
        text = self.id_to_token[token_id]
        if t in (TOKEN_TYPE_CONTROL, TOKEN_TYPE_UNKNOWN):
            return text.encode("utf-8") if special else b""
        if t == TOKEN_TYPE_USER_DEFINED:
            return text.encode("utf-8")
        if t == TOKEN_TYPE_NORMAL:
            try:
                return bytes(self._byte_decoder[ch] for ch in text)
            except KeyError as exc:
                raise RuntimeError(f"token {token_id} ({text!r}) has a character outside the GPT-2 byte table") from exc
        return b""

    def _split_special(self, text: str):
        """Yield (chunk, is_special) pairs: text split around whole-token
        matches of special/control/user-defined tokens (longest-first), the
        rest left as plain text for the pre-tokenizer + BPE."""
        if self._special_re is None or not text:
            if text:
                yield text, False
            return
        # re.split with one capturing group alternates plain text (even
        # indices) and matches of that group (odd indices); since the
        # pattern is scanned over the whole string, a plain-text chunk can
        # never coincidentally equal a special token's text -- it would
        # already have been split out as its own (odd-index) match.
        parts = self._special_re.split(text)
        for i, part in enumerate(parts):
            if not part:
                continue
            yield part, (i % 2 == 1)
