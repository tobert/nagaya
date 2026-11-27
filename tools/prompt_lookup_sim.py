"""Prompt-lookup drafts, simulated on texts that the model wrote (docs/gb10-research.md, D1 step 1).

A draft is the K tokens that followed the last earlier occurrence of the newest n tokens (n from `max_n` down to
`min_n`, the longest match first). A block of 1 + len(draft) rows is checked in one pass; the longest prefix of the
draft that equals the true greedy continuation is accepted, and the block always yields one more token (the model's
own). Greedy text makes the check exact: it is the token the model writes anyway.

Input: a JSON list of {"kind", "ids", "start"} (the prompt is ids[:start], the model's text ids[start:]). Run:

  python3 prompt_lookup_sim.py texts.json            # the tables for K in 2, 4, 8, 16

`cost(rows)` is the time of a block of `rows` rows in decode steps of 1 row. It is an ESTIMATE from the three points
of the research doc (1 row: 1.0, 8 rows: 1.8, 16 rows: 2.5), linear between them; it is not measured.
"""
import json
import sys
from collections import defaultdict

COST_POINTS = [(1, 1.0), (8, 1.8), (16, 2.5)]


def parse_points(text):
    """Parse a cost table `rows:steps,rows:steps,...` (at least 2 points, rows ascending), e.g. a measured one."""
    try:
        pts = [(int(r), float(c)) for r, c in (p.split(":") for p in text.split(","))]
        if len(pts) < 2 or any(a[0] >= b[0] for a, b in zip(pts, pts[1:])):
            raise ValueError
    except ValueError:
        raise ValueError(
            f"bad cost table {text!r}: write at least two points as rows:steps with ascending rows, e.g. 1:1.0,8:1.8,16:2.5"
        ) from None
    return pts


def cost(rows, pts=COST_POINTS):
    """Estimated steps of a block of `rows` rows; beyond the last point the last slope goes on."""
    for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
        if rows <= r1:
            return c0 + (c1 - c0) * (rows - r0) / (r1 - r0)
    (r0, c0), (r1, c1) = pts[-2:]
    return c1 + (c1 - c0) * (rows - r1) / (r1 - r0)


def draft(ids, pos, k, min_n, max_n):
    """The draft after ids[:pos]: up to k tokens, or [] with no match of at least min_n tokens."""
    for n in range(min(max_n, pos - 1), min_n - 1, -1):
        gram = ids[pos - n:pos]
        for j in range(pos - n - 1, -1, -1):
            if ids[j:j + n] == gram:
                return ids[j + n:min(j + n + k, pos)]
    return []


def run(ids, start, k, min_n, max_n):
    """Blocks of the lookup over ids[start:]: a list of (drafted, accepted), one per block."""
    steps, pos = [], start
    while pos < len(ids):
        d = draft(ids, pos, k, min_n, max_n)
        accepted = 0
        # The block yields 1 + accepted tokens: never more than the text that is left.
        while accepted < len(d) and accepted < len(ids) - pos - 1 and d[accepted] == ids[pos + accepted]:
            accepted += 1
        steps.append((len(d), accepted))
        pos += accepted + 1
    return steps


def summarize(steps, pts=COST_POINTS):
    """The measures of a list of blocks. `speed` is tokens per cost, relative to plain decode (1 row per step).
    `accepted_per_drafted_block` is None when no block had a draft."""
    if not steps:
        raise ValueError("no blocks: the text has no generated tokens after `start`; check the `start` of each text")
    tokens = sum(a + 1 for _, a in steps)
    spent = sum(cost(d + 1, pts) for d, _ in steps)
    drafted = [(d, a) for d, a in steps if d]
    return {
        "blocks": len(steps),
        "tokens": tokens,
        "accepted_per_block": sum(a for _, a in steps) / len(steps),
        "draft_share": len(drafted) / len(steps),
        "accepted_per_drafted_block": sum(a for _, a in drafted) / len(drafted) if drafted else None,
        "speed": tokens / spent,
    }


def table(texts, ks=(2, 4, 8, 16), min_n=1, max_n=4, pts=COST_POINTS):
    """{kind: {k: summary}} with an "all" kind; each kind pools the blocks of its texts."""
    out = {}
    groups = defaultdict(list)
    for t in texts:
        groups[t["kind"]].append(t)
        groups["all"].append(t)
    for kind, group in groups.items():
        out[kind] = {k: summarize([s for t in group for s in run(t["ids"], t["start"], k, min_n, max_n)], pts) for k in ks}
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Simulate prompt-lookup drafts on texts the model wrote.")
    ap.add_argument("texts", help="JSON list of {kind, ids, start}")
    ap.add_argument("--cost", default=None, help="cost table rows:steps,... (default: the doc's estimate 1:1.0,8:1.8,16:2.5)")
    args = ap.parse_args()
    try:
        pts = parse_points(args.cost) if args.cost else COST_POINTS
        texts = json.load(open(args.texts))
    except (ValueError, OSError) as e:
        sys.exit(f"prompt_lookup_sim: {e}")
    label = "measured" if args.cost else "ESTIMATE from the doc"
    for min_n in (1, 2, 3):
        print(f"\nmin_n={min_n} max_n=4   (speed vs plain decode; cost table: {label})")
        print(f"{'kind':8} {'K':>3} {'acc/block':>10} {'draft share':>12} {'acc/drafted':>12} {'speed':>7}")
        try:
            res = table(texts, min_n=min_n, pts=pts)
        except ValueError as e:
            sys.exit(f"prompt_lookup_sim: {e}")
        for kind in sorted(res, key=lambda x: (x == "all", x)):
            for k, s in res[kind].items():
                drafted = "none" if s["accepted_per_drafted_block"] is None else f"{s['accepted_per_drafted_block']:.2f}"
                print(f"{kind:8} {k:>3} {s['accepted_per_block']:>10.2f} {s['draft_share']:>12.2f} "
                      f"{drafted:>12} {s['speed']:>7.2f}")


if __name__ == "__main__":
    main()
