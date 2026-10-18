#!/usr/bin/env python3
"""Check Markdown files against the STE limits in AGENTS.md.

A description sentence has at most 25 words. A paragraph has at most 6 sentences. Code blocks, tables, headings,
quoted text, and a bold label at the start of a paragraph are not checked. A code span counts as one word. The exit
status is 1 when a file breaks a limit. With no files, it checks nothing and says so.
"""
import re
import sys

MAX_WORDS = 25
MAX_SENTENCES = 6

FENCE = re.compile(r"^\s*```")
CODE_SPAN = re.compile(r"`[^`]*`")
QUOTE = re.compile(r"\"[^\"]*\"|“[^”]*”")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
ITEM = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
LABEL = re.compile(r"^\*\*[^*]+\*\*\s*")
# A sentence ends at . ? or ! (or one of them and a closing parenthesis) before a space and a letter, a digit, a
# code span, or the end of the block.
END = re.compile(r"(?:(?<=[.?!])|(?<=[.?!]\)))\s+(?=[A-Za-z0-9`(*])")


def blocks(lines):
    """Yield (first line number, text) for each paragraph and each list item."""
    in_fence = False
    start, buf = 0, []

    def flush():
        if buf:
            yield start, " ".join(s.strip() for s in buf)

    for n, line in enumerate(lines, 1):
        if FENCE.match(line):
            yield from flush()
            buf = []
            in_fence = not in_fence
            continue
        if in_fence or line.lstrip().startswith(("|", "#", ">")) or not line.strip():
            yield from flush()
            buf = []
            continue
        if ITEM.match(line):
            yield from flush()
            start, buf = n, [ITEM.sub("", line)]
            continue
        if not buf:
            start = n
        buf.append(line)
    yield from flush()


def words(sentence):
    return re.findall(r"[\w'’/.:+-]+", sentence)


def check(path):
    problems = []
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for n, text in blocks(lines):
        text = LABEL.sub("", text)
        text = LINK.sub(r"\1", text)
        text = CODE_SPAN.sub("CODE", text)
        text = QUOTE.sub("QUOTE", text)
        sentences = [s for s in END.split(text) if s.strip()]
        if len(sentences) > MAX_SENTENCES:
            problems.append(f"{path}:{n}: paragraph has {len(sentences)} sentences (limit {MAX_SENTENCES})")
        for s in sentences:
            count = len(words(s))
            if count > MAX_WORDS:
                problems.append(f"{path}:{n}: sentence has {count} words (limit {MAX_WORDS}): {s[:70]}...")
    return problems


def main(paths):
    if not paths:
        print("no files to check: give the Markdown files as arguments", file=sys.stderr)
        return 2
    problems = [p for path in paths for p in check(path)]
    for p in problems:
        print(p)
    print(f"{len(problems)} problems in {len(paths)} files", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
