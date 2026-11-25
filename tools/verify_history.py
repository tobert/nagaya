#!/usr/bin/env python3
"""Verify that a rewritten git repository has clean attribution and no private data.

Checks every commit, commit message, path and text blob reachable from `--ref`.
Exits 0 only when zero violations are found.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

EDIT_NOTICE = (
    "Commit edited before publication to remove private data and/or correct attribution."
)
TRAILER_RE = re.compile(r"^([A-Za-z0-9-]+):\s*(.+)$")
COAUTHOR_FMT_RE = re.compile(r"^.+ <[^@\s]+@[^@\s]+>$")
CGNAT_IP_RE = re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+\b")
PRIV_192_RE = re.compile(r"\b192\.168\.\d+\.\d+\b")
INTERNAL_DOMAIN_RE = re.compile(r"\bkaijutsu\.local\b", re.IGNORECASE)


def load_patterns(map_path: str) -> list[tuple[str, re.Pattern[str]]]:
    """Load private IP and hostname patterns from the local mapping file."""
    raw = json.loads(Path(map_path).read_text(encoding="utf-8"))
    patterns: list[tuple[str, re.Pattern[str]]] = [
        ("cgnat_ip", CGNAT_IP_RE),
        ("priv_192_ip", PRIV_192_RE),
        ("internal_domain", INTERNAL_DOMAIN_RE),
    ]
    for src in raw.get("ips", {}):
        patterns.append((f"ip:{src}", re.compile(r"(?<![0-9])" + re.escape(src) + r"(?![0-9])")))
    for src in raw.get("hosts", {}):
        patterns.append(
            (f"host:{src}", re.compile(r"(?<![A-Za-z0-9])" + re.escape(src) + r"(?![A-Za-z0-9])", re.IGNORECASE))
        )
    return patterns


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a rewritten git repository.")
    parser.add_argument("--repo", required=True, help="Path to the rewritten git repository.")
    parser.add_argument("--map", required=True, help="Path to the untracked JSON replacement map.")
    parser.add_argument("--author-name", required=True, help="Expected author and committer name.")
    parser.add_argument("--author-email", required=True, help="Expected author and committer email.")
    parser.add_argument("--ref", default="main", help="Git ref to verify (default: main).")
    parser.add_argument("--require-signoff", action="store_true", help="Require Signed-off-by on every commit.")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    patterns = load_patterns(args.map)
    violations: list[str] = []

    # 1. Audit commits, authors, committers, trailers and messages.
    raw_log = subprocess.check_output(
        ["git", "log", args.ref, "--format=%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%s%x1f%b%x1e"],
        cwd=repo,
        text=True,
        errors="replace",
    )
    commits = [c for c in raw_log.split("\x1e") if c.strip()]
    coauthors: Counter[str] = Counter()
    notice_count = 0
    bad_coauthors: Counter[str] = Counter()
    strip_keys = {"claude-session"}
    expected_signoff = f"Signed-off-by: {args.author_name} <{args.author_email}>"

    for entry in commits:
        parts = entry.strip().split("\x1f")
        if len(parts) < 7:
            continue
        h, an, ae, cn, ce, subj, body = parts
        short = h[:8]
        if an != args.author_name or ae != args.author_email:
            violations.append(f"commit {short}: wrong author {an} <{ae}>")
        if cn != args.author_name or ce != args.author_email:
            violations.append(f"commit {short}: wrong committer {cn} <{ce}>")

        full_msg = subj + "\n" + body
        if EDIT_NOTICE in body:
            notice_count += 1
        for label, pat in patterns:
            m = pat.search(full_msg)
            if m:
                violations.append(f"commit {short} message matched {label}: {m.group(0)!r}")

        has_signoff = False
        for line in body.splitlines():
            tm = TRAILER_RE.match(line.strip())
            if not tm:
                continue
            key, val = tm.group(1).lower(), tm.group(2).strip()
            if key in strip_keys:
                violations.append(f"commit {short}: stripped trailer still present: {line}")
            elif key == "co-authored-by":
                coauthors[val] += 1
                if not COAUTHOR_FMT_RE.match(val):
                    bad_coauthors[val] += 1
            elif key == "signed-off-by" and line.strip() == expected_signoff:
                has_signoff = True
        if args.require_signoff and not has_signoff:
            violations.append(f"commit {short}: missing {expected_signoff}")

    for val, cnt in bad_coauthors.items():
        violations.append(f"malformed Co-Authored-By ({cnt} commits): {val!r}")

    # 2. Audit all paths and text blobs reachable from ref.
    objs = subprocess.check_output(
        ["git", "rev-list", "--objects", args.ref],
        cwd=repo,
    ).splitlines()
    blobs: dict[bytes, str] = {}
    for line in objs:
        parts = line.split(b" ", 1)
        if len(parts) == 2:
            oid = parts[0]
            path = parts[1].decode("utf-8", errors="replace")
            blobs[oid] = path
            for label, pat in patterns:
                m = pat.search(path)
                if m:
                    violations.append(f"path matched {label}: {path}")

    blob_count = 0
    with tempfile.TemporaryFile() as tf:
        tf.write(b"\n".join(blobs.keys()) + b"\n")
        tf.seek(0)
        proc = subprocess.Popen(
            ["git", "cat-file", "--batch"],
            cwd=repo,
            stdin=tf,
            stdout=subprocess.PIPE,
        )
        assert proc.stdout is not None
        while True:
            header = proc.stdout.readline()
            if not header:
                break
            hparts = header.split()
            if len(hparts) < 3:
                continue
            oid, typ, size_s = hparts[0], hparts[1], hparts[2]
            size = int(size_s)
            content = proc.stdout.read(size)
            proc.stdout.read(1)
            if typ != b"blob":
                continue
            blob_count += 1
            if b"\x00" in content[:8192]:
                continue
            text = content.decode("utf-8", errors="replace")
            path = blobs.get(oid, "?")
            for label, pat in patterns:
                m = pat.search(text)
                if m:
                    violations.append(f"blob {oid.decode()[:8]} ({path}) matched {label}: {m.group(0)!r}")
        proc.wait()

    print(f"Verified {len(commits)} commits ({notice_count} with edit notice) and {blob_count} unique blobs.")
    print("Co-Authored-By breakdown:")
    for val, cnt in coauthors.most_common():
        print(f"  {cnt:5d}  {val}")

    if violations:
        print(f"\nFAILED: {len(violations)} violations found (showing up to 30):", file=sys.stderr)
        for v in violations[:30]:
            print(f"  - {v}", file=sys.stderr)
        return 1

    print("\nPASSED: 0 violations found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
