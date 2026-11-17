# SPDX-License-Identifier: MIT
"""The host sampler of the first Qwen3.8 server: seeded, and bit-reproducible.

`SampleParams.from_request` reads the request's `sample` object {temperature, top_p, top_k, min_p, seed}; any other
key, a wrong type (a bool is not a number) or an out-of-range value raises BadRequest. A missing field takes its
default: temperature 0.6, top_p 0.95, top_k 20, min_p 0.0. Where those come from: the model ships no
generation_config.json, so these are Qwen's thinking-mode settings. The GGUF's own `general.sampling.*` says
top_k 20, top_p 0.95 and temp 1.0: it agrees except on temperature; the server kept 0.6 from its first runs. A missing or null seed draws
a fresh one from the OS (secrets); the params carry the seed used, so any run replays.

`Sampler(params).pick(logits)` returns (token_id, p, h). The chain is llama.cpp's default order (common/sampling.cpp
in 5cf3a3528: top_k, top_p, min_p, then temperature, then the draw), with these boundary semantics:
  - top_k: keep the k largest logits; k <= 0 is off (llama-sampler.cpp llama_sampler_top_k_impl, line 321). Ties are
    ordered by ascending id, so the cut is a total order (llama.cpp leaves tie order to its sort).
  - top_p: softmax over the candidates top_k left, keep the smallest prefix whose running mass is >= p, never
    fewer than one (llama_sampler_top_p_apply, line 1549: `cum_sum >= ctx->p`; p >= 1 is off).
  - min_p: keep logit >= max + ln(min_p), i.e. p_i >= min_p * p_max, on the untempered candidates
    (llama_sampler_min_p_apply, line 1749; min_p <= 0 is off).
  - temperature divides the survivors' logits, then softmax. Temperature 0 is greedy argmax (lowest id on a tie),
    with no filter and no RNG draw.
The draw is numpy Generator(PCG64(seed)), exactly one uniform double per sampled step, inverse CDF over the
survivors ordered (logit desc, id asc) in float64: the same seed and the same logits give the same tokens, bit
for bit. Non-finite logits raise RuntimeError: a NaN logit is a bug upstream, and sampling it would hide that.

p and h are the chosen token's probability, and the entropy (nats), of the untempered
full-vocabulary softmax in float64, whatever the filters kept.
"""
from __future__ import annotations

import math
import secrets
from dataclasses import dataclass

import numpy as np

from .common import BadRequest

DEFAULTS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
KEYS = ("temperature", "top_p", "top_k", "min_p", "seed")


def _number(v, key: str) -> float:
    if type(v) is bool or not isinstance(v, (int, float)):
        raise BadRequest(f"sample.{key} must be a number, got {v!r}")
    return float(v)


def _integer(v, key: str) -> int:
    if type(v) is not int:
        raise BadRequest(f"sample.{key} must be an integer, got {v!r}")
    return v


@dataclass(frozen=True)
class SampleParams:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    seed: int

    @classmethod
    def from_request(cls, sample: dict | None) -> "SampleParams":
        if sample is None:
            sample = {}
        if not isinstance(sample, dict):
            raise BadRequest(f"sample must be an object, got {sample!r}")
        extra = sorted(set(sample) - set(KEYS))
        if extra:
            raise BadRequest(f"unsupported sample field(s): {', '.join(map(str, extra))} (this service has {list(KEYS)})")
        temperature = _number(sample.get("temperature", DEFAULTS["temperature"]), "temperature")
        if not (math.isfinite(temperature) and temperature >= 0.0):
            raise BadRequest(f"sample.temperature must be finite and >= 0, got {temperature!r}")
        top_p = _number(sample.get("top_p", DEFAULTS["top_p"]), "top_p")
        if not 0.0 < top_p <= 1.0:
            raise BadRequest(f"sample.top_p must be in (0, 1], got {top_p!r}")
        top_k = _integer(sample.get("top_k", DEFAULTS["top_k"]), "top_k")
        if top_k < 0:
            raise BadRequest(f"sample.top_k must be an integer >= 0 (0 = off), got {top_k!r}")
        min_p = _number(sample.get("min_p", DEFAULTS["min_p"]), "min_p")
        if not 0.0 <= min_p < 1.0:
            raise BadRequest(f"sample.min_p must be in [0, 1), got {min_p!r}")
        seed = sample.get("seed")
        if seed is None:
            seed = secrets.randbits(64)
        else:
            seed = _integer(seed, "seed")
            if not 0 <= seed < 2**64:
                raise BadRequest(f"sample.seed must be in [0, 2**64), got {seed!r}")
        return cls(temperature, top_p, top_k, min_p, seed)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in KEYS}


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


class Sampler:
    def __init__(self, params: SampleParams):
        self.params = params
        self.rng = np.random.Generator(np.random.PCG64(params.seed))

    def _logits(self, logits) -> np.ndarray:
        l = np.asarray(logits)
        if l.ndim != 1 or l.size == 0:
            raise RuntimeError(f"logits must be a non-empty 1-D array, got shape {l.shape}")
        l = l.astype(np.float64)
        if not np.isfinite(l).all():
            bad = np.flatnonzero(~np.isfinite(l))
            raise RuntimeError(f"non-finite logits at {len(bad)} position(s), first at id {int(bad[0])}: {l[bad[0]]}")
        return l

    def candidates(self, logits) -> tuple[np.ndarray, np.ndarray]:
        """The survivors of top_k, top_p and min_p, ordered (logit desc, id asc), and the probabilities the draw
        uses (tempered softmax over them). Greedy: the argmax alone, probability 1."""
        l = self._logits(logits)
        return self._candidates(l)

    def _candidates(self, l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.params
        n = l.size
        if p.temperature == 0.0:
            i = int(np.argmax(l))  # the first maximum: the lowest id
            return np.array([i]), np.array([1.0])
        if 0 < p.top_k < n:
            thr = np.partition(l, n - p.top_k)[n - p.top_k]  # the k-th largest; every tie of it is a candidate
            cand = np.flatnonzero(l >= thr)
        else:
            cand = np.arange(n)
        order = np.lexsort((cand, -l[cand]))  # primary key last: -logit, then id
        ids = cand[order][:p.top_k] if p.top_k > 0 else cand[order]
        z = l[ids]
        if p.top_p < 1.0:
            pr = _softmax(z)  # top_p
            keep = min(int(np.searchsorted(np.cumsum(pr), p.top_p, side="left")) + 1, len(ids))
            ids, z = ids[:keep], z[:keep]
        if p.min_p > 0.0:
            keep = int(np.count_nonzero(z >= z[0] + math.log(p.min_p)))  # z is sorted: a prefix, and z[0] passes
            ids, z = ids[:keep], z[:keep]
        return ids, _softmax(z / p.temperature)

    def pick(self, logits) -> tuple[int, float, float]:
        l = self._logits(logits)
        ids, pr = self._candidates(l)
        if self.params.temperature == 0.0:
            tok = int(ids[0])  # greedy: no draw
        else:
            u = self.rng.random()
            cdf = np.cumsum(pr)  # draw
            tok = int(ids[min(int(np.searchsorted(cdf, u * cdf[-1], side="right")), len(ids) - 1)])
        p, h = raw_stats(l, tok)
        return tok, p, h


def raw_stats(l: np.ndarray, token: int) -> tuple[float, float]:
    """kaiseki chat.py raw_stats, in float64: the token's probability and the entropy (nats) of the untempered
    softmax over the whole vocabulary."""
    z = l - l.max()  # stats
    e = np.exp(z)
    s = e.sum()
    # H = -sum p (z - log s) = log s - sum(e * z) / s: one exp pass (the same value to float64 summation error,
    # ~V * 2^-53). The sum is numpy's pairwise sum,
    # not np.dot: a BLAS dot's order follows its thread count, and OpenBLAS's 32 workers then spin through the next GPU
    # pass, slowing it.
    return float(e[token] / s), float(math.log(s) - float((e * z).sum()) / s)
