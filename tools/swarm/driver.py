"""The driver of a recipe. It runs rounds of think, select, distill and act, evaluates each program, and logs each
program, each round and a summary as JSON lines."""
import itertools
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import binpack, client, task

TRAIN_SEED = 1
TEST_SEED = 2
STAGE_NAMES = ("think", "select", "distill", "act", "evaluate")
CALL = ("system", "sample", "thinking")
TOKENS = ("prompt", "fed", "completion")


class Spread:
    """The round-robin spread of independent calls over the served models of a recipe."""

    def __init__(self, models):
        self._cycle = itertools.cycle(models)
        self._lock = threading.Lock()

    def next(self):
        with self._lock:
            return next(self._cycle)


class Stage:
    """The counts of one stage of one round: calls, tokens and wall time. Calls of one stage run on many threads."""

    def __init__(self):
        self.calls = 0
        self.tokens = dict.fromkeys(TOKENS, 0)
        self.seconds = 0.0
        self.models = {}
        self._lock = threading.Lock()

    def count(self, model, usage):
        with self._lock:
            self.calls += 1
            self.models[model] = self.models.get(model, 0) + 1
            for k in TOKENS:
                self.tokens[k] += usage.get(k, 0)

    def record(self):
        return {"calls": self.calls, **self.tokens, "seconds": round(self.seconds, 3), "models": dict(self.models)}


def shares(programs):
    """The counts and shares of a list of program records: valid, constant and improved."""
    n = len(programs)
    out = {"programs": n}
    for k in ("valid", "constant", "improved"):
        c = sum(1 for p in programs if p[k])
        out[k] = c
        out[f"{k}_share"] = round(c / n, 4) if n else None
    return out


def totals(rounds):
    """The sums of the stage records of many rounds, and the counts of their programs."""
    stages = {}
    for r in rounds:
        for name, s in r["stages"].items():
            t = stages.setdefault(name, {"calls": 0, **dict.fromkeys(TOKENS, 0), "seconds": 0.0, "models": {}})
            t["calls"] += s["calls"]
            t["seconds"] = round(t["seconds"] + s["seconds"], 3)
            for k in TOKENS:
                t[k] += s[k]
            for m, c in s["models"].items():
                t["models"][m] = t["models"].get(m, 0) + c
    counts = {k: sum(r[k] for r in rounds) for k in ("programs", "valid", "constant", "improved")}
    n = counts["programs"]
    for k in ("valid", "constant", "improved"):
        counts[f"{k}_share"] = round(counts[k] / n, 4) if n else None
    return {"rounds": len(rounds), **counts, "stages": stages}


class Log:
    """A JSONL file that many threads write."""

    def __init__(self, f):
        self.f = f
        self._lock = threading.Lock()

    def write(self, obj):
        with self._lock:
            self.f.write(json.dumps(obj) + "\n")
            self.f.flush()


class Archive:
    """The evaluated programs that scored. The parent policy is a tournament of two among the top 5."""

    def __init__(self, evaluate, log, clock=time.time):
        self.evaluate = evaluate
        self.log = log
        self.clock = clock
        self.t0 = clock()
        first = evaluate(binpack.BEST_FIT, TRAIN_SEED)
        if "excess" not in first:
            raise RuntimeError(f"The evaluator failed on best fit: {first}. Check bwrap, prlimit and numpy.")
        self.items = [(binpack.BEST_FIT, first["excess"])]
        self._lock = threading.Lock()

    def parent(self, rng):
        with self._lock:
            pool = sorted(self.items, key=lambda x: x[1])[:5]
        return pool[min(rng.randrange(len(pool)), rng.randrange(len(pool)))]

    def best(self):
        with self._lock:
            return min(self.items, key=lambda x: x[1])

    def add(self, src, parent_excess, extra):
        """Evaluate one program, keep it when it scores, and log it. The record has the flags of the round stats."""
        result = self.evaluate(src, TRAIN_SEED) if src else {"fail": "no code"}
        valid = "excess" in result
        rec = {**result, "valid": valid, "constant": task.is_constant(src),
               "improved": valid and result["excess"] < parent_excess, "parent": parent_excess, **extra}
        with self._lock:
            if valid and src not in [s for s, _ in self.items]:
                self.items.append((src, result["excess"]))
            best = min(x for _, x in self.items)
        self.log.write({"t": round(self.clock() - self.t0, 1), **rec, "best": best, "src": src})
        return rec


def options(stage):
    """The keyword arguments of a call from the members system, sample and thinking of its stage."""
    return {k: stage[k] for k in CALL if k in stage}


def temperature(stage, member):
    """The temperature of one member of a stage: its step of the ladder, else the temperature of the stage."""
    ladder = stage.get("ladder")
    return ladder[member % len(ladder)] if ladder else stage["temperature"]


def by_temperature(programs):
    """One row for each act temperature of a list of program records, in the order of the temperature: the counts
    programs, valid and improved, and the best excess of the valid programs (None when none is valid)."""
    rows = {}
    for p in programs:
        t = p["sample"]["temperature"]
        r = rows.setdefault(t, {"temperature": t, "programs": 0, "valid": 0, "improved": 0, "best_excess": None})
        r["programs"] += 1
        r["valid"] += bool(p["valid"])
        r["improved"] += bool(p["improved"])
        if p["valid"] and (r["best_excess"] is None or p["excess"] < r["best_excess"]):
            r["best_excess"] = p["excess"]
    return [rows[t] for t in sorted(rows)]


class Driver:
    """One run of a recipe against one client."""

    def __init__(self, recipe, client_, evaluate, log, seed=0, clock=time.time):
        self.recipe = recipe
        self.client = client_
        self.log = log
        self.clock = clock
        self.spread = Spread(recipe["models"])
        self.rng = random.Random(seed)
        self._rng_lock = threading.Lock()
        self.archive = Archive(evaluate, log, clock)
        self.evaluate = evaluate
        self.rounds = []
        self.programs = []
        self._rounds_lock = threading.Lock()
        self._counter = itertools.count(1)

    def _rand(self, fn, *args):
        with self._rng_lock:
            return fn(*args)

    def _call(self, stage, model, fn, *args, **kwargs):
        text, usage = fn(model, *args, **kwargs)
        stage.count(model, usage)
        return text

    def _prompt(self, head, prompt):
        """The user part of a call: the text, or with `snapshots` the head as its own message, then the rest."""
        return task.split(head, prompt) if self.recipe["snapshots"] else prompt

    def prime(self, stage, head, seed):
        """The text of the prime: one generation with thinking over the shared head."""
        p = self.recipe["prime"]
        return self._call(stage, self.spread.next(), self.client.prime,
                          self._prompt(head, task.prime_prompt(head)), p["max_tokens"],
                          p["temperature"], seed, system=p.get("system"), sample=p.get("sample"))

    def think(self, stage, head, seed, prime=None):
        t = self.recipe["think"]
        n, foci = t["agents"], t["foci"]
        chosen = self._rand(lambda: self.rng.sample(foci, min(n, len(foci))) +
                            [self.rng.choice(foci) for _ in range(n - len(foci))])
        models = [self.spread.next() for _ in range(n)]
        temperatures = [temperature(t, i) for i in range(n)]
        shares = self.recipe["snapshots"]

        def one(i):
            user = (self._prompt(head, task.think_prompt(head, chosen[i])) if prime is None else
                    task.think_turns(head, prime, chosen[i], shares))
            text = self._call(stage, models[i], self.client.generate, user, t["max_tokens"], temperatures[i],
                              seed * 1000 + i, task.IDEA_PREFILL, latent=t["latent"], **options(t))
            return task.after_prefill(text, task.IDEA_PREFILL)
        with ThreadPoolExecutor(n) as ex:
            return list(ex.map(one, range(n))), temperatures

    def select(self, stage, head, ideas, seed):
        """The kept ideas and the record of the selection."""
        s = self.recipe["select"]
        if s["by"] == "all":
            return ideas, {}
        if s["by"] == "vote":
            text = self._call(stage, self.spread.next(), self.client.generate,
                              self._prompt(head, task.vote_prompt(head, ideas)), 4, 0.0,
                              seed, task.VOTE_PREFILL, **options(s))
            j = task.vote_index(task.after_prefill(text, task.VOTE_PREFILL), len(ideas))
            bad = j is None
            if bad:
                j = self._rand(self.rng.randrange, len(ideas))
            return [ideas[j]], {"vote": j, "bad_vote": bad}
        vectors = self._call(stage, s["model"], self.client.embed, ideas, s["dimensions"])
        kept = client.farthest(vectors, s["keep"])
        return [ideas[i] for i in kept], {"kept": kept}

    def distill(self, stage, parent, head, kept, seed):
        d = self.recipe["distill"]
        if d["by"] == "pick":
            return kept[0].strip()
        if d["by"] == "combine":
            text = self._call(stage, self.spread.next(), self.client.generate, task.combine_prompt(parent, kept),
                              d["max_tokens"], d["temperature"], seed, task.PLAN_PREFILL, latent=d["latent"],
                              **options(d))
        else:
            text = self._call(stage, self.spread.next(), self.client.ensemble,
                              [self._prompt(head, task.joined_distill_prompt(head, x)) for x in kept], "prob_mean",
                              d["max_tokens"],
                              d["temperature"], seed, task.PLAN_PREFILL, latent=d["latent"], **options(d))
        return task.after_prefill(text, task.PLAN_PREFILL).strip()

    def act(self, stage, head, plan, seed):
        """The replies of the act stage, each with its model and its temperature."""
        a = self.recipe["act"]
        prompt = self._prompt(head, task.act_prompt(head, plan))
        plain = self._prompt(head, task.act_plain_prompt(head))
        pre = task.CODE_PREFILL
        if a["by"] == "joined":
            model = self.spread.next()
            text = self._call(stage, model, self.client.ensemble, [prompt] * a["forks"], a["combine"],
                              a["max_tokens"], a["temperature"], seed * 100, pre, latent=a["latent"], **options(a))
            return [(model, a["temperature"], text)]
        models = [self.spread.next() for _ in range(a["forks"])]
        temperatures = [temperature(a, f) for f in range(a["forks"])]
        if self.recipe["snapshots"]:
            # Each fork has the same prompt, so its end is a boundary. The nagaya feeds the prompt once for the forks
            # in flight (spec 11.14), and a fork feeds only its prefill.
            prompt[-1]["snap"] = True

        def one(f):
            if a["by"] == "contrast":
                return self._call(stage, models[f], self.client.ensemble, [prompt, plain],
                                  "contrast", a["max_tokens"], temperatures[f], seed * 100 + f, pre,
                                  strength=a.get("strength"), latent=a["latent"], **options(a))
            return self._call(stage, models[f], self.client.generate, prompt, a["max_tokens"], temperatures[f],
                              seed * 100 + f, pre, latent=a["latent"], **options(a))
        with ThreadPoolExecutor(a["forks"]) as ex:
            return list(zip(models, temperatures, ex.map(one, range(a["forks"]))))

    def round(self, seed):
        """One round: a parent, the four stages and the evaluation. It logs and returns the round record."""
        names = (("prime",) if "prime" in self.recipe else ()) + STAGE_NAMES
        stages = {name: Stage() for name in names}
        parent = self._rand(self.archive.parent, self.rng)
        head = task.show(parent)

        def timed(name, fn, *args):
            t = self.clock()
            out = fn(stages[name], *args)
            stages[name].seconds = self.clock() - t
            return out

        primed = timed("prime", self.prime, head, seed) if "prime" in stages else None
        ideas, think_temperatures = timed("think", self.think, head, seed, primed)
        kept, pick = timed("select", self.select, head, ideas, seed)
        plan = timed("distill", self.distill, parent, head, kept, seed)
        replies = timed("act", self.act, head, plan, seed)
        kind = "-".join(self.recipe[s]["by"] for s in ("select", "distill", "act"))

        def evaluate(stage, replies):
            def one(fr):
                f, (model, temp, text) = fr
                src = task.program(task.after_prefill(text, task.CODE_PREFILL))
                sample = {"temperature": temp, **self.recipe["act"].get("sample", {})}
                return self.archive.add(src, parent[1], {"kind": kind, "round": seed, "fork": f, "model": model,
                                                         "sample": sample, "plan": plan, "ideas": ideas, **pick})
            with ThreadPoolExecutor(len(replies)) as ex:
                return list(ex.map(one, enumerate(replies)))
        programs = timed("evaluate", evaluate, replies)
        extra = {} if primed is None else {"prime": primed}
        rec = {"round": seed, "t": round(self.clock() - self.archive.t0, 1), "parent": parent[1], **pick, **extra,
               **shares(programs), "think_temperatures": think_temperatures,
               "stages": {n: s.record() for n, s in stages.items()}}
        self.log.write({"round_stats": rec})
        with self._rounds_lock:
            self.rounds.append(rec)
            self.programs.extend(programs)
        return rec

    def run(self):
        """Run rounds, loop.in_flight at once, until loop.minutes end or loop.rounds rounds start. Then log the
        summary with the best program on the train seed and on the test seed."""
        lp = self.recipe["loop"]
        end = self.clock() + lp["minutes"] * 60
        limit = lp.get("rounds")

        def worker():
            while self.clock() < end:
                n = next(self._counter)
                if limit is not None and n > limit:
                    return
                self.round(n)
        with ThreadPoolExecutor(lp["in_flight"]) as ex:
            for f in [ex.submit(worker) for _ in range(lp["in_flight"])]:
                f.result()
        best = self.archive.best()
        test = self.evaluate(best[0], TEST_SEED)
        summary = {"recipe": self.recipe, "archive": len(self.archive.items), "best_train": best[1],
                   "best_test": test.get("excess"), "best_src": best[0], **totals(self.rounds),
                   "by_temperature": by_temperature(self.programs)}
        self.log.write({"summary": summary})
        return summary
