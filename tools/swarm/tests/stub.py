"""A stub transport with canned replies, and a fake evaluator. No test calls a live nagaya."""
import threading

from swarm import binpack, client, task

# The expressions that the stub codes, in turn, and their fake scores. "0.0" is a constant, and "oops(" fails.
CODE_LINES = ["-leftover * 0.9", "0.0", "oops(", "-leftover"]
SCORES = {"-(bins - item)": 4.08, "-leftover * 0.9": 3.5, "-leftover": 4.08}


def fake_evaluate(src, seed):
    """A score from the table of SCORES, by the expression of the program; a constant fails as in the sandbox."""
    if src == binpack.BEST_FIT:
        return {"excess": SCORES["-(bins - item)"] + (0.06 if seed == 2 else 0.0)}
    line = src.split("return ", 1)[1].strip()
    if line in SCORES:
        return {"excess": SCORES[line] + (0.06 if seed == 2 else 0.0)}
    return {"fail": f"fake: {line}"}


class StubTransport:
    """It records each POST, and it answers from the prefill of the first context. A generation gives its prefill
    in the content, and an ensemble does not, as on the nagaya. A generation whose last message is a user message
    is a prime: it gives `prime`, a reply with reasoning and content."""

    def __init__(self, vote="Best idea: 2", prime=None):
        self.calls = []
        self.vote = vote
        self.prime = prime if prime is not None else {"reasoning": " Small gaps waste bins. ",
                                                      "content": "Try a tighter fit.\n"}
        self._code = 0
        self._lock = threading.Lock()

    def _reply(self, prefill):
        if prefill == task.IDEA_PREFILL:
            return task.IDEA_PREFILL + " Favor bins that end nearly full."
        if prefill == task.VOTE_PREFILL:
            return self.vote
        if prefill == task.PLAN_PREFILL:
            return task.PLAN_PREFILL + " Scale the leftover by 0.9."
        if prefill == task.CODE_PREFILL:
            with self._lock:
                line = CODE_LINES[self._code % len(CODE_LINES)]
                self._code += 1
            return task.CODE_PREFILL + line + "\n```\n"
        raise AssertionError(f"the stub has no reply for the prefill {prefill!r}")

    def post(self, path, body):
        with self._lock:
            self.calls.append((path, body))
        usage = {"prompt": 100, "fed": 10, "completion": 5}
        if path == client.GENERATIONS and ("contexts" in body or "forks" in body):
            # Several replies (gogi "Several replies"): one for each context, or one for each fork of one context.
            contexts = body.get("contexts") or [body["context"]] * (
                body["forks"] if isinstance(body["forks"], int) else len(body["forks"]))
            replies = [{"index": i, "content": self._reply(c["messages"][-1]["content"]),
                        "usage": {"completion": 5}} for i, c in enumerate(contexts)]
            return {"usage": {"prompt": 100 * (len(contexts) if "contexts" in body else 1), "fed": 10,
                              "completion": 5 * len(contexts)}, "replies": replies}
        if path == client.GENERATIONS and body["context"]["messages"][-1]["role"] == "user":
            return {**self.prime, "usage": usage}
        if path == client.GENERATIONS:
            return {"content": self._reply(body["context"]["messages"][-1]["content"]), "usage": usage}
        if path == client.ENSEMBLE:
            prefill = body["contexts"][0]["messages"][-1]["content"]
            return {"content": self._reply(prefill)[len(prefill):], "usage": usage}
        if path == client.EMBEDDINGS:
            n = len(body["inputs"])
            rows = [{"index": i, "vector": [1.0 if j == i % 4 else 0.1 for j in range(4)]} for i in range(n)]
            return {"embeddings": list(reversed(rows))}
        raise AssertionError(f"the stub has no route {path}")

    def paths(self):
        return [p for p, _ in self.calls]
