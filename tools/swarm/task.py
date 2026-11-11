"""The prompts of the bin packing task, and the parse of a program from a reply. The text of each prompt is the text of
the first trials (docs/swarm-ideas.md), so a run of a recipe compares with them."""
import ast
import re

PROBLEM = """We solve online bin packing. Items arrive one at a time with sizes from 1 to 100; every bin has capacity 100.
For each item, a Python function `priority(item, bins)` gets the item size (a float) and a numpy array `bins` of the
remaining capacities of the bins that can hold the item (each is at least the item size; an unused bin has 100).
It returns a numpy array of the same shape with one score per bin. The item goes into the bin with the highest score.
The goal is to use as few bins as possible. The score is the percent of excess bins over the lower bound; lower is
better. Only numpy (as np) and math may be imported. The function must be fast: it runs 25,000 times."""

FOCI = [
    "Make the fit tighter: favor bins that this item leaves nearly full.",
    "Avoid leaving small gaps that no future item can fill.",
    "Treat large items and small items differently, with a threshold.",
    "Use a nonlinear function of the remaining capacity, such as a power or an exponential.",
    "Consider what remains after placing the item, relative to the item size.",
    "Think about keeping some bins with room for the common item sizes (about 30 to 60).",
    "Add a small secondary term that breaks ties between equally good bins.",
    "Penalize opening a new empty bin unless it is really needed.",
    "Use the distribution of the items: sizes follow a Weibull shape with a mean near 40.",
    "Think about the leftover space modulo typical item sizes.",
    "Combine best fit with a bonus for bins that become exactly full.",
    "Change only the constants of the parent, in a careful direction.",
    "Simplify the parent and remove any part that does not help.",
    "Prefer bins whose leftover would match a likely future item.",
    "Use a smooth score instead of a hard threshold, or a hard one instead of a smooth one.",
    "Try something unusual that the parent does not do at all.",
]

EXPR_HEAD = "import numpy as np\nimport math\n\n\ndef priority(item, bins):\n    leftover = bins - item\n    return "
CODE_PREFILL = "```python\n" + EXPR_HEAD
FORM = ("Reply with one python code block. The body is a single return of one numpy expression in item, bins and "
        "leftover (leftover = bins - item); np.where, np.exp, np.log and comparisons are allowed.")
IDEA_PREFILL = "Idea:"
PLAN_PREFILL = "Plan:"
VOTE_PREFILL = "Best idea: "
THINK_ASK = "In at most two sentences, propose one concrete change to priority. Do not write code."
IMPROVE = "Write an improved version. "
PRIME_ASK = "Think about where this program wastes bins, and what kinds of change could lower the excess."


def show(parent):
    """The problem, then the parent program and its score."""
    return f"{PROBLEM}\n\nThe current program scores {parent[1]:.3f}% excess:\n```python\n{parent[0].strip()}\n```\n"


def show_last(parent, middle):
    """The problem, then the middle, then the parent program last. Thus the program stays inside the sliding window
    of 256 positions of LFM3 while the model writes."""
    return (f"{PROBLEM}\n\n{middle}\n\nThe current program scores {parent[1]:.3f}% excess:\n```python\n"
            f"{parent[0].strip()}\n```\n")


def listing(ideas):
    """The ideas as a numbered list, from 1."""
    return "\n".join(f"{j + 1}. {x.strip()}" for j, x in enumerate(ideas))


def split(head, prompt):
    """The messages of a prompt that starts with the head: the head with a boundary, then the rest. Each prompt of a
    round starts with the same head, so the nagaya feeds it once and holds its snapshot."""
    if not prompt.startswith(head):
        raise ValueError("The prompt does not start with the head of its round.")
    return [{"role": "user", "content": head, "snap": True},
            {"role": "user", "content": prompt[len(head):].lstrip("\n")}]


def think_prompt(head, focus):
    return head + f"\nFocus: {focus}\n" + THINK_ASK


def prime_prompt(head):
    """The user message of the prime, and the first user message of each think agent after a prime."""
    return head + "\n" + PRIME_ASK


def think_turns(head, prime, focus, snapshots=False):
    """The messages of a think agent after a prime: the head, the prime's text as the assistant's reply, then the
    focus. Every agent shares the messages before the focus. With `snapshots`, the head is its own message, and the
    prime's text ends with a boundary, so the nagaya holds the shared part. Without it, each agent feeds it again."""
    asked = split(head, prime_prompt(head)) if snapshots else [{"role": "user", "content": prime_prompt(head)}]
    reply = {"role": "assistant", "content": prime}
    if snapshots:
        reply["snap"] = True
    return asked + [reply, {"role": "user", "content": f"Focus: {focus}\n" + THINK_ASK}]


def vote_prompt(head, ideas):
    return (head + f"\nIdeas from {len(ideas)} agents:\n{listing(ideas)}\n\nWhich one idea is most likely to lower "
            "the excess? Reply with its number only.")


def combine_prompt(parent, ideas):
    return (show_last(parent, f"Ideas from {len(ideas)} agents:\n{listing(ideas)}") +
            "\nCombine the best of these ideas into one concrete change to priority, in at most two sentences. "
            "Do not write code.")


def joined_distill_prompt(head, idea):
    return (head + f"\nAn agent proposed: {idea.strip()}\nState one concrete change to priority, in at most two "
            "sentences. Do not write code.")


def act_prompt(head, plan):
    return head + f"\nApply this idea: {plan}\n" + IMPROVE + FORM


def act_plain_prompt(head):
    """The act prompt with no plan: context 1 of a contrast."""
    return head + "\n" + IMPROVE + FORM


def after_prefill(text, prefill):
    """The text of a reply with its prefill first. A generation gives the prefill, and an ensemble does not."""
    return text if text.startswith(prefill) else prefill + text


def program(text):
    """The program of a reply that went on from CODE_PREFILL. Its first line is the expression, or None."""
    if not text.startswith(CODE_PREFILL):
        return None
    line = text[len(CODE_PREFILL):].split("\n")[0].split("#")[0].strip().rstrip("`").strip()
    line = re.sub(r"\bmath\.", "np.", line)
    return EXPR_HEAD + line + "\n" if line else None


def vote_index(text, count):
    """The 0-based index of the idea that a vote reply names, or None when the reply names no idea of the list."""
    body = text[len(VOTE_PREFILL):] if text.startswith(VOTE_PREFILL) else text
    m = re.search(r"\d+", body)
    j = int(m.group()) - 1 if m else -1
    return j if 0 <= j < count else None


def is_constant(src):
    """True when the last return of priority gives one number, such as `return 0.0`. A program that does not parse
    is not a constant."""
    if not src:
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "priority":
            returns = sorted((n for n in ast.walk(node) if isinstance(n, ast.Return)), key=lambda n: n.lineno)
            if not returns or returns[-1].value is None:
                return False
            try:
                value = ast.literal_eval(returns[-1].value)
            except (ValueError, TypeError, SyntaxError, RecursionError):
                return False
            return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False
