#!/usr/bin/env python3
"""Rewrite git history before publication to remove private data and fix attribution.

Reads a `git fast-export` stream from a source repository, transforms blobs,
paths, commit metadata and commit messages, and writes the result into a fresh
target repository with `git fast-import`.

Pass private IP and hostname mappings in a local untracked JSON file via
`--map`. Do not commit that file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

EDIT_NOTICE = (
    "Commit edited before publication to remove private data and/or correct attribution."
)

# Canonical Co-Authored-By replacements (matched case-insensitively on the value).
DEFAULT_COAUTHOR_MAP: dict[str, str] = {
    "deepseek-flash": "DeepSeek 4.1 Flash <noreply@deepseek.com>",
    "deepseek-flash <noreply@deepseek.com>": "DeepSeek 4.1 Flash <noreply@deepseek.com>",
    "deepseek deepseek-flash via kaibo <noreply@deepseek.com>": "DeepSeek 4.1 Flash <noreply@deepseek.com>",
    "deepseek flash (kaibo review) <noreply@deepseek.com>": "DeepSeek 4.1 Flash <noreply@deepseek.com>",
    "deepseek v4 flash (via kaibo) <noreply@deepseek.com>": "DeepSeek 4.1 Flash <noreply@deepseek.com>",
    "deepseek flash (via kaibo) <noreply@deepseek.com>": "DeepSeek 4.1 Flash <noreply@deepseek.com>",
    "deepseek (deepseek-flash, via kaibo) <noreply@deepseek.com>": "DeepSeek 4.1 Flash <noreply@deepseek.com>",
    "gemini 4 neon": "Gemini 4 Neon <noreply@google.com>",
    "glm-5.3 (kaibo review) <noreply@z.ai>": "GLM-5.3 <noreply@z.ai>",
}

DEFAULT_STRIP_TRAILERS = {"claude-session"}

IDENT_RE = re.compile(rb"^(author|committer) (.+) <([^>]+)> (\d+ [+-]\d{4})$")
TRAILER_RE = re.compile(r"^([A-Za-z0-9-]+):\s*(.+)$")
FILE_MOD_RE = re.compile(rb"^M ([0-7]+) (:\d+|[0-9a-f]{40,64}|inline) (.+)$")
FILE_DEL_RE = re.compile(rb"^D (.+)$")
FILE_RENAME_RE = re.compile(rb"^([RC]) (.+)$")


@dataclass
class Rules:
    """Replacement and attribution rules for one rewrite run."""

    author_name: str
    author_email: str
    ip_patterns: list[tuple[re.Pattern[str], str]] = field(default_factory=list)
    host_patterns: list[tuple[re.Pattern[str], str]] = field(default_factory=list)
    coauthor_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_COAUTHOR_MAP))
    strip_trailers: set[str] = field(default_factory=lambda: set(DEFAULT_STRIP_TRAILERS))
    add_signoff: bool = False
    rewrite_paths: bool = True


@dataclass
class Stats:
    """Counters reported at the end of a rewrite run."""

    blobs_total: int = 0
    blobs_modified: int = 0
    commits_total: int = 0
    commits_edited: int = 0
    paths_modified: int = 0
    trailers_stripped: int = 0
    coauthors_normalized: int = 0
    signoffs_added: int = 0


def load_rules(args: argparse.Namespace) -> Rules:
    """Build rewrite rules from CLI flags and the local JSON mapping file."""
    rules = Rules(
        author_name=args.author_name,
        author_email=args.author_email,
        add_signoff=args.signoff,
        rewrite_paths=not args.no_rewrite_paths,
    )
    if args.map:
        raw = json.loads(Path(args.map).read_text(encoding="utf-8"))
        for src, dst in sorted(raw.get("ips", {}).items(), key=lambda kv: -len(kv[0])):
            pat = re.compile(r"(?<![0-9])" + re.escape(src) + r"(?![0-9])")
            rules.ip_patterns.append((pat, dst))
        for src, dst in sorted(raw.get("hosts", {}).items(), key=lambda kv: -len(kv[0])):
            pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(src) + r"(?![A-Za-z0-9])", re.IGNORECASE)
            rules.host_patterns.append((pat, dst))
        for src, dst in raw.get("coauthors", {}).items():
            rules.coauthor_map[src.lower()] = dst
        for key in raw.get("strip_trailers", []):
            rules.strip_trailers.add(key.lower())
    return rules


def scrub_text(text: str, rules: Rules) -> str:
    """Replace private IPs and hostnames in UTF-8 text."""
    out = text
    for pat, dst in rules.ip_patterns:
        out = pat.sub(dst, out)
    for pat, dst in rules.host_patterns:
        def _repl(match: re.Match[str], target: str = dst) -> str:
            token = match.group(0)
            if token.isupper():
                return target.upper()
            return target
        out = pat.sub(_repl, out)
    return out


def unquote_fast_export_path(raw: bytes) -> tuple[str, bool]:
    """Decode a git fast-export path token, returning (path, was_quoted)."""
    if len(raw) >= 2 and raw[0:1] == b'"' and raw[-1:] == b'"':
        inner = raw[1:-1]
        out = bytearray()
        i = 0
        while i < len(inner):
            ch = inner[i]
            if ch == ord("\\") and i + 1 < len(inner):
                nxt = inner[i + 1]
                escapes = {
                    ord("a"): 7,
                    ord("b"): 8,
                    ord("f"): 12,
                    ord("n"): 10,
                    ord("r"): 13,
                    ord("t"): 9,
                    ord("v"): 11,
                    ord("\\"): ord("\\"),
                    ord('"'): ord('"'),
                }
                if nxt in escapes:
                    out.append(escapes[nxt])
                    i += 2
                elif ord("0") <= nxt <= ord("7") and i + 3 < len(inner):
                    out.append(int(inner[i + 1 : i + 4], 8))
                    i += 4
                else:
                    out.append(nxt)
                    i += 2
            else:
                out.append(ch)
                i += 1
        return out.decode("utf-8", errors="surrogateescape"), True
    return raw.decode("utf-8", errors="surrogateescape"), False


def quote_fast_export_path(path: str, force_quote: bool = False) -> bytes:
    """Encode a path for git fast-import."""
    raw = path.encode("utf-8", errors="surrogateescape")
    needs_quote = force_quote or any(b < 0x20 or b >= 0x7F or b in (ord('"'), ord("\\")) for b in raw)
    if not needs_quote:
        return raw
    buf = bytearray(b'"')
    for b in raw:
        if b == ord("\\"):
            buf.extend(b"\\\\")
        elif b == ord('"'):
            buf.extend(b'\\"')
        elif b == ord("\n"):
            buf.extend(b"\\n")
        elif b == ord("\r"):
            buf.extend(b"\\r")
        elif b == ord("\t"):
            buf.extend(b"\\t")
        elif b < 0x20 or b >= 0x7F:
            buf.extend(f"\\{b:03o}".encode("ascii"))
        else:
            buf.append(b)
    buf.append(ord('"'))
    return bytes(buf)


def split_two_paths(raw: bytes) -> tuple[bytes, bytes]:
    """Split a fast-export R/C line payload into source and destination path tokens."""
    if raw.startswith(b'"'):
        i = 1
        while i < len(raw):
            if raw[i] == ord("\\"):
                i += 2
            elif raw[i] == ord('"'):
                return raw[: i + 1], raw[i + 2 :]
            else:
                i += 1
        raise ValueError(f"unterminated quoted path in fast-export line: {raw!r}")
    idx = raw.index(b" ")
    return raw[:idx], raw[idx + 1 :]


def rewrite_path_token(raw: bytes, rules: Rules, stats: Stats) -> tuple[bytes, bool]:
    """Rewrite private hostnames inside a fast-export path token."""
    if not rules.rewrite_paths:
        return raw, False
    path, was_quoted = unquote_fast_export_path(raw)
    new_path = scrub_text(path, rules)
    if new_path == path:
        return raw, False
    stats.paths_modified += 1
    return quote_fast_export_path(new_path, force_quote=was_quoted), True


def rewrite_commit_message(
    msg: str,
    rules: Rules,
    stats: Stats,
) -> tuple[str, bool]:
    """Scrub private data, normalize trailers, and append the edit notice if text was scrubbed."""
    lines = msg.rstrip("\n").splitlines()
    message_scrubbed = False

    # Identify trailing trailer block after a blank line. Line 0 is always the subject.
    trailer_start = len(lines)
    idx = len(lines) - 1
    while idx > 0 and TRAILER_RE.match(lines[idx]):
        idx -= 1
    if idx < len(lines) - 1 and idx >= 1 and lines[idx].strip() == "":
        trailer_start = idx + 1

    body_lines = lines[:trailer_start]
    trailer_lines = lines[trailer_start:]

    # Trim trailing blank lines from body.
    while body_lines and body_lines[-1].strip() == "":
        body_lines.pop()

    # Scrub body text.
    new_body_lines: list[str] = []
    for line in body_lines:
        scrubbed = scrub_text(line, rules)
        if scrubbed != line:
            message_scrubbed = True
        new_body_lines.append(scrubbed)

    # Filter and normalize trailers.
    new_trailers: list[str] = []
    seen_coauthors: set[str] = set()
    has_signoff = False
    target_signoff = f"Signed-off-by: {rules.author_name} <{rules.author_email}>"

    for line in trailer_lines:
        match = TRAILER_RE.match(line)
        if not match:
            scrubbed = scrub_text(line, rules)
            if scrubbed != line:
                message_scrubbed = True
            new_trailers.append(scrubbed)
            continue
        key, val = match.group(1), match.group(2).strip()
        key_lower = key.lower()
        if key_lower in rules.strip_trailers:
            stats.trailers_stripped += 1
            continue
        if key_lower == "co-authored-by":
            norm = rules.coauthor_map.get(val.lower(), val)
            scrubbed_norm = scrub_text(norm, rules)
            if scrubbed_norm != norm:
                message_scrubbed = True
            if scrubbed_norm != val:
                stats.coauthors_normalized += 1
            canon_key = scrubbed_norm.lower()
            if canon_key in seen_coauthors:
                continue
            seen_coauthors.add(canon_key)
            new_trailers.append(f"Co-Authored-By: {scrubbed_norm}")
            continue
        if key_lower == "signed-off-by":
            scrubbed_val = scrub_text(val, rules)
            if scrubbed_val != val:
                message_scrubbed = True
            rendered = f"Signed-off-by: {scrubbed_val}"
            if rendered == target_signoff:
                has_signoff = True
            new_trailers.append(rendered)
            continue
        scrubbed_line = scrub_text(line, rules)
        if scrubbed_line != line:
            message_scrubbed = True
        new_trailers.append(scrubbed_line)

    if rules.add_signoff and not has_signoff:
        new_trailers.insert(0, target_signoff)
        stats.signoffs_added += 1

    if message_scrubbed and EDIT_NOTICE not in new_body_lines:
        if new_body_lines:
            new_body_lines.append("")
        new_body_lines.append(EDIT_NOTICE)

    out_lines = list(new_body_lines)
    if new_trailers:
        if out_lines:
            out_lines.append("")
        out_lines.extend(new_trailers)

    result = "\n".join(out_lines) + "\n"
    return result, message_scrubbed


def read_data_block(stream: BinaryIO, header: bytes) -> bytes:
    """Read a fast-export `data <size>` payload and consume its trailing newline."""
    size = int(header.split()[1])
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError(f"expected {size} bytes in data block, got {len(payload)}")
    # git fast-export emits an optional LF after data payloads.
    pos_byte = stream.peek(1)[:1]
    if pos_byte == b"\n":
        stream.read(1)
    return payload


def write_data_block(stream: BinaryIO, payload: bytes) -> None:
    """Write a fast-import `data <size>` block."""
    stream.write(f"data {len(payload)}\n".encode("ascii"))
    stream.write(payload)
    stream.write(b"\n")


def rewrite_ident_line(line: bytes, rules: Rules) -> tuple[bytes, bool]:
    """Normalize an author or committer line while keeping its timestamp."""
    match = IDENT_RE.match(line.rstrip(b"\r\n"))
    if not match:
        return line, False
    role, name_b, email_b, ts_b = match.groups()
    target_name = rules.author_name.encode("utf-8")
    target_email = rules.author_email.encode("utf-8")
    if name_b == target_name and email_b == target_email:
        return line, False
    return role + b" " + target_name + b" <" + target_email + b"> " + ts_b + b"\n", True


def process_stream(
    src_stream: BinaryIO,
    dst_stream: BinaryIO,
    rules: Rules,
) -> Stats:
    """Filter a fast-export stream into a fast-import stream."""
    stats = Stats()
    modified_blobs: set[bytes] = set()
    author_tag = f"{rules.author_name} <{rules.author_email}>".encode("utf-8")

    while True:
        line = src_stream.readline()
        if not line:
            break

        if line == b"blob\n":
            stats.blobs_total += 1
            dst_stream.write(line)
            mark_id: bytes | None = None
            while True:
                sub = src_stream.readline()
                if not sub:
                    raise EOFError("unexpected EOF inside blob")
                if sub.startswith(b"mark "):
                    mark_id = sub.strip().split()[1]
                    dst_stream.write(sub)
                elif sub.startswith(b"original-oid "):
                    continue
                elif sub.startswith(b"data "):
                    payload = read_data_block(src_stream, sub)
                    if b"\x00" not in payload[:8192]:
                        try:
                            text = payload.decode("utf-8")
                        except UnicodeDecodeError:
                            new_payload = payload
                        else:
                            new_text = scrub_text(text, rules)
                            new_payload = new_text.encode("utf-8") if new_text != text else payload
                    else:
                        new_payload = payload
                    if new_payload != payload:
                        stats.blobs_modified += 1
                        if mark_id is not None:
                            modified_blobs.add(mark_id)
                    write_data_block(dst_stream, new_payload)
                    break
                else:
                    dst_stream.write(sub)
            continue

        if line.startswith(b"commit "):
            stats.commits_total += 1
            commit_header: list[bytes] = [line]
            commit_edited = False
            msg_bytes = b""
            commit_tail: list[bytes] = []

            # Read headers up to `data <N>`.
            while True:
                sub = src_stream.readline()
                if not sub:
                    raise EOFError("unexpected EOF inside commit header")
                if sub.startswith(b"original-oid "):
                    continue
                if sub.startswith(b"author ") or sub.startswith(b"committer "):
                    new_sub, changed = rewrite_ident_line(sub, rules)
                    if changed:
                        commit_edited = True
                    commit_header.append(new_sub)
                elif sub.startswith(b"data "):
                    msg_bytes = read_data_block(src_stream, sub)
                    break
                else:
                    commit_header.append(sub)

            # Read from/merge/file operations until blank line or next command.
            while True:
                sub = src_stream.readline()
                if not sub or sub == b"\n":
                    if sub:
                        commit_tail.append(sub)
                    break
                if sub.startswith(b"M "):
                    m = FILE_MOD_RE.match(sub.rstrip(b"\n"))
                    if m:
                        mode_b, ref_b, path_b = m.groups()
                        if ref_b in modified_blobs:
                            commit_edited = True
                        new_path_b, p_changed = rewrite_path_token(path_b, rules, stats)
                        if p_changed:
                            commit_edited = True
                        commit_tail.append(b"M " + mode_b + b" " + ref_b + b" " + new_path_b + b"\n")
                        continue
                elif sub.startswith(b"D "):
                    m = FILE_DEL_RE.match(sub.rstrip(b"\n"))
                    if m:
                        new_path_b, p_changed = rewrite_path_token(m.group(1), rules, stats)
                        if p_changed:
                            commit_edited = True
                        commit_tail.append(b"D " + new_path_b + b"\n")
                        continue
                elif sub.startswith(b"R ") or sub.startswith(b"C "):
                    m = FILE_RENAME_RE.match(sub.rstrip(b"\n"))
                    if m:
                        op_b, rest_b = m.groups()
                        src_p, dst_p = split_two_paths(rest_b)
                        new_src_p, c1 = rewrite_path_token(src_p, rules, stats)
                        new_dst_p, c2 = rewrite_path_token(dst_p, rules, stats)
                        if c1 or c2:
                            commit_edited = True
                        commit_tail.append(op_b + b" " + new_src_p + b" " + new_dst_p + b"\n")
                        continue
                commit_tail.append(sub)

            msg_text = msg_bytes.decode("utf-8", errors="replace")
            new_msg_text, msg_scrubbed = rewrite_commit_message(msg_text, rules, stats)
            if msg_scrubbed:
                stats.commits_edited += 1

            for hline in commit_header:
                dst_stream.write(hline)
            write_data_block(dst_stream, new_msg_text.encode("utf-8"))
            for tline in commit_tail:
                dst_stream.write(tline)
            continue

        if line.startswith(b"tag "):
            dst_stream.write(line)
            while True:
                sub = src_stream.readline()
                if not sub:
                    raise EOFError("unexpected EOF inside tag")
                if sub.startswith(b"original-oid "):
                    continue
                if sub.startswith(b"tagger "):
                    parts = sub.rstrip(b"\n").rsplit(b" ", 2)
                    if len(parts) == 3:
                        dst_stream.write(b"tagger " + author_tag + b" " + parts[1] + b" " + parts[2] + b"\n")
                    else:
                        dst_stream.write(sub)
                elif sub.startswith(b"data "):
                    payload = read_data_block(src_stream, sub)
                    text = scrub_text(payload.decode("utf-8", errors="replace"), rules)
                    write_data_block(dst_stream, text.encode("utf-8"))
                    break
                else:
                    dst_stream.write(sub)
            continue

        if line.startswith(b"original-oid "):
            continue

        dst_stream.write(line)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Rewrite git history for publication.")
    parser.add_argument("--source", required=True, help="Path to the source git repository.")
    parser.add_argument("--dest", required=True, help="Path to the new target git repository (must not exist).")
    parser.add_argument("--map", required=True, help="Path to the untracked JSON replacement map.")
    parser.add_argument("--author-name", required=True, help="Normalized author and committer name.")
    parser.add_argument("--author-email", required=True, help="Normalized author and committer email.")
    parser.add_argument("--ref", default="main", help="Git ref or rev-list spec to export (default: main).")
    parser.add_argument("--signoff", action="store_true", help="Add a Signed-off-by trailer for the author.")
    parser.add_argument("--no-rewrite-paths", action="store_true", help="Leave historical file paths unchanged.")
    args = parser.parse_args()

    src_dir = Path(args.source).resolve()
    dst_dir = Path(args.dest).resolve()
    if dst_dir.exists():
        print(f"error: destination already exists: {dst_dir}", file=sys.stderr)
        return 1

    rules = load_rules(args)
    dst_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=dst_dir, check=True, stdout=subprocess.DEVNULL)

    exporter = subprocess.Popen(
        [
            "git",
            "fast-export",
            "--signed-tags=strip",
            "--tag-of-filtered-object=rewrite",
            args.ref,
        ],
        cwd=src_dir,
        stdout=subprocess.PIPE,
    )
    importer = subprocess.Popen(
        ["git", "fast-import", "--quiet"],
        cwd=dst_dir,
        stdin=subprocess.PIPE,
    )
    assert exporter.stdout is not None and importer.stdin is not None

    try:
        stats = process_stream(exporter.stdout, importer.stdin, rules)
    finally:
        importer.stdin.close()
        exporter.stdout.close()

    exp_rc = exporter.wait()
    imp_rc = importer.wait()
    if exp_rc != 0 or imp_rc != 0:
        print(f"error: fast-export exited {exp_rc}, fast-import exited {imp_rc}", file=sys.stderr)
        return 1

    # Check out HEAD in the target repo so callers can inspect or test it immediately.
    subprocess.run(["git", "checkout", "--quiet", args.ref], cwd=dst_dir, check=True)
    subprocess.run(["git", "gc", "--quiet", "--prune=now"], cwd=dst_dir, check=True)

    print(
        f"Rewrote {stats.commits_total} commits ({stats.commits_edited} edited), "
        f"{stats.blobs_total} blobs ({stats.blobs_modified} modified), "
        f"{stats.paths_modified} paths modified, "
        f"{stats.trailers_stripped} trailers stripped, "
        f"{stats.coauthors_normalized} co-authors normalized, "
        f"{stats.signoffs_added} signoffs added."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
