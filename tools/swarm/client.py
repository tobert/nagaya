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


def _context(user, prefill, system=None, thinking=False):
    """A council v2 context. `user` is one user message, or a list of messages in order. A system prompt goes
    first. A prefill is an open assistant message with a `content` member. With thinking on, that member closes the
    thinking (gogi "Continuations"), so the reply after a prefill has no reasoning."""
    messages = [{"role": "user", "content": user}] if isinstance(user, str) else [dict(m) for m in user]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    if prefill is not None:
        messages.append({"role": "assistant", "content": prefill, "continue": True})
    return {"thinking": thinking, "messages": messages}


def _sample(temperature, seed, sample=None):
    """The member `sample` of a body: the temperature, the other members of the recipe, then the seed."""
    return {"temperature": temperature, **(sample or {}), "seed": seed}


def thought(reply):
    """The text of a reply with thinking: its reasoning, then its content, with a blank line between them. A reply
    that `max_tokens` cut inside the thinking has its text in the reasoning only."""
    parts = [(reply.get(k) or "").strip() for k in ("reasoning", "content")]
    return "\n\n".join(p for p in parts if p)


class Client:
    """The calls over one transport. A transport has `post(path, body)`, which gives the JSON reply."""

    def __init__(self, transport):
        self.transport = transport

    def generate(self, model, user, max_tokens, temperature, seed, prefill=None, latent=0, system=None, sample=None,
                 thinking=False):
        """One reply to one user message, or to a list of messages. With latent above 0, the call is an `ensemble`
        of one context with latent steps (docs/lab.md, "Latent steps in a request"). The text is the `content` of
        the reply; the reasoning is not in it."""
        if latent > 0:
            return self.ensemble(model, [user], "logprob_sum", max_tokens, temperature, seed, prefill, latent=latent,
                                 system=system, sample=sample, thinking=thinking)
        reply = self.transport.post(GENERATIONS, {
            "model": model, "max_tokens": max_tokens, "sample": _sample(temperature, seed, sample),
            "context": _context(user, prefill, system, thinking)})
        return reply.get("content", ""), usage_of(reply)

    def prime(self, model, user, max_tokens, temperature, seed, system=None, sample=None):
        """One generation with thinking on and no prefill. The text is the reasoning, then the content
        (`thought`)."""
        reply = self.transport.post(GENERATIONS, {
            "model": model, "max_tokens": max_tokens, "sample": _sample(temperature, seed, sample),
            "context": _context(user, None, system, True)})
        return thought(reply), usage_of(reply)

    def ensemble(self, model, users, combine, max_tokens, temperature, seed, prefill=None, strength=None, latent=0,
                 system=None, sample=None, thinking=False):
        """Joined forks: one reply from the combined rows of one context for each user message. The member
        `latent` goes into the body only when it is above 0, so a nagaya with no latent steps serves the rest."""
        body = {"model": model, "contexts": [_context(u, prefill, system, thinking) for u in users],
                "combine": combine, "max_tokens": max_tokens, "sample": _sample(temperature, seed, sample)}
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
