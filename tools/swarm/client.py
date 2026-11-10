"""The calls of the driver to the nagaya: a generation, joined forks (`ensemble`) and an embedding. Each call gives
its text and its usage, so the driver counts the tokens of each stage."""
import numpy as np

GENERATIONS = "/council/v2/generations"
ENSEMBLE = "/nagaya/v1/lab/ensemble"
EMBEDDINGS = "/council/v2/embeddings"


def usage_of(reply):
    """The token counts of a reply: prompt, fed and completion. An absent count is 0."""
    u = reply.get("usage") or {}
    return {k: int(u.get(k, 0)) for k in ("prompt", "fed", "completion")}


def _context(user, prefill):
    messages = [{"role": "user", "content": user}]
    if prefill is not None:
        messages.append({"role": "assistant", "content": prefill, "continue": True})
    return {"thinking": False, "messages": messages}


class Client:
    """The calls over one transport. A transport has `post(path, body)`, which gives the JSON reply."""

    def __init__(self, transport):
        self.transport = transport

    def generate(self, model, user, max_tokens, temperature, seed, prefill=None, latent=0):
        """One reply to one user message. With latent above 0, the call is an `ensemble` of one context with
        latent steps (docs/lab.md, "Latent steps in a request")."""
        if latent > 0:
            return self.ensemble(model, [user], "logprob_sum", max_tokens, temperature, seed, prefill, latent=latent)
        reply = self.transport.post(GENERATIONS, {
            "model": model, "max_tokens": max_tokens, "sample": {"temperature": temperature, "seed": seed},
            "context": _context(user, prefill)})
        return reply.get("content", ""), usage_of(reply)

    def ensemble(self, model, users, combine, max_tokens, temperature, seed, prefill=None, strength=None, latent=0):
        """Joined forks: one reply from the combined rows of one context for each user message. The member
        `latent` goes into the body only when it is above 0, so a nagaya with no latent steps serves the rest."""
        body = {"model": model, "contexts": [_context(u, prefill) for u in users], "combine": combine,
                "max_tokens": max_tokens, "sample": {"temperature": temperature, "seed": seed}}
        if strength is not None:
            body["strength"] = strength
        if latent > 0:
            body["latent"] = {"steps": latent}
        reply = self.transport.post(ENSEMBLE, body)
        return reply.get("content", ""), usage_of(reply)

    def embed(self, model, texts, dimensions):
        """One unit vector for each text, in the order of the texts."""
        reply = self.transport.post(EMBEDDINGS, {"model": model, "task": "clustering", "dimensions": dimensions,
                                                 "inputs": texts})
        rows = sorted(reply["embeddings"], key=lambda e: e["index"])
        if len(rows) != len(texts):
            raise RuntimeError(f"{EMBEDDINGS} gave {len(rows)} vectors for {len(texts)} texts.")
        return np.array([e["vector"] for e in rows], dtype=float), usage_of(reply)


def farthest(vectors, k):
    """Greedy farthest-point selection from the centroid outward: k indices whose vectors are far apart."""
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    first = int(np.argmax(np.linalg.norm(v - v.mean(0), axis=1)))
    chosen = [first]
    dist = 1 - v @ v[first]
    dist[first] = -np.inf
    while len(chosen) < min(k, len(v)):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, 1 - v @ v[nxt])
        dist[chosen] = -np.inf
    return chosen
