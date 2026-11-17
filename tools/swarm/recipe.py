"""The recipe of a round (tools/swarm/README.md, "An agent's interface to a swarm"). A recipe is a JSON object with the
stages prime, think, select, distill, act and loop. The parse fills the defaults and refuses a recipe that it cannot
run. A member that is absent leaves the parse, and so the requests, as they were before the member existed."""
import json
import math

from . import task

PARENT_POLICY = "tournament of the top 5"
LATENT_MAX = 256
ENSEMBLE_MAX = 8
EMBED_MODEL = "embeddinggemma-2"

STAGES = ("models", "system", "snapshots", "batch", "prime", "think", "select", "distill", "act", "loop")
CALL = ("system", "sample", "thinking")
KEYS = {
    "prime": ("max_tokens", "temperature", "sample", "system"),
    "think": ("agents", "foci", "max_tokens", "temperature", "latent") + CALL,
    "select": ("by", "keep", "model", "dimensions", "system", "thinking"),
    "distill": ("by", "latent", "max_tokens", "temperature") + CALL,
    "act": ("forks", "by", "max_tokens", "temperature", "latent", "strength", "combine") + CALL,
    "loop": ("minutes", "in_flight", "parent", "rounds"),
}
# The members of `sample` that the driver sends, in the order of the body. The driver sets the seed.
SAMPLE = ("temperature", "top_p", "top_k", "min_p")
# `ladder` gives a temperature to each member of a stage in turn. The driver sends the temperature, not the ladder.
LADDER = "ladder"
NO_PENALTY = ("repetition_penalty is not a member: the sampler of the nagaya (spec 11.2) has no repetition penalty. "
              "Remove it.")


class RecipeError(ValueError):
    """A recipe that the driver cannot run. The message names the member."""


def _int(obj, key, name, default, lo, hi=None):
    v = obj.get(key, default)
    if v is None or isinstance(v, bool) or not isinstance(v, int) or v < lo or (hi is not None and v > hi):
        top = f" to {hi}" if hi is not None else " or more"
        raise RecipeError(f"{name} must be a whole number from {lo}{top}; it is {v!r}.")
    return v


def _num(obj, key, name, default, lo):
    v = obj.get(key, default)
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < lo:
        raise RecipeError(f"{name} must be a finite number of {lo} or more; it is {v!r}.")
    return float(v)


def _choice(obj, key, name, default, choices):
    v = obj.get(key, default)
    if v not in choices:
        raise RecipeError(f"{name} must be one of {', '.join(map(str, choices))}; it is {v!r}.")
    return v


def _stage(obj, name):
    v = obj.get(name, {})
    if not isinstance(v, dict):
        raise RecipeError(f"{name} must be an object; it is {v!r}.")
    for k in v:
        if k == "repetition_penalty":
            raise RecipeError(f"{name}.{NO_PENALTY}")
        if k not in KEYS[name]:
            raise RecipeError(f"{name}.{k} is not a member. The members are {', '.join(KEYS[name])}.")
    return v


def _sample(st, name, default_temperature):
    """The temperature of a stage, and the other members of its `sample` in the order of SAMPLE. The member
    `temperature` of the stage and `sample.temperature` are the same setting, so a stage sets at most one."""
    v = st.get("sample", {})
    if not isinstance(v, dict):
        raise RecipeError(f"{name}.sample must be an object; it is {v!r}.")
    for k in v:
        if k == "repetition_penalty":
            raise RecipeError(f"{name}.sample.{NO_PENALTY}")
        if k not in SAMPLE + (LADDER,):
            raise RecipeError(f"{name}.sample.{k} is not a member. The members are {', '.join(SAMPLE + (LADDER,))}; "
                              "the driver sets the seed.")
    if "temperature" in st and "temperature" in v:
        raise RecipeError(f"{name}.temperature and {name}.sample.temperature are the same setting. Remove one.")
    ladder = None
    if LADDER in v:
        if "temperature" in st or "temperature" in v:
            where = f"{name}.temperature" if "temperature" in st else f"{name}.sample.temperature"
            raise RecipeError(f"{name}.sample.ladder and {where} both set the temperature. Remove one.")
        ladder = v[LADDER]
        if not isinstance(ladder, list) or not ladder:
            raise RecipeError(f"{name}.sample.ladder must be a list of one or more temperatures; it is {ladder!r}.")
        ladder = [_num({"t": t}, "t", f"{name}.sample.ladder[{i}]", None, 0.0) for i, t in enumerate(ladder)]
        temperature = None
    elif "temperature" in v:
        temperature = _num(v, "temperature", f"{name}.sample.temperature", None, 0.0)
    else:
        temperature = _num(st, "temperature", f"{name}.temperature", default_temperature, 0.0)
    extra = {}
    if "top_p" in v:
        p = v["top_p"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not (0.0 < p <= 1.0):
            raise RecipeError(f"{name}.sample.top_p must be above 0 and at most 1; it is {p!r}.")
        extra["top_p"] = float(p)
    if "top_k" in v:
        extra["top_k"] = _int(v, "top_k", f"{name}.sample.top_k", None, 0)
    if "min_p" in v:
        p = v["min_p"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not (0.0 <= p < 1.0):
            raise RecipeError(f"{name}.sample.min_p must be at least 0 and below 1; it is {p!r}.")
        extra["min_p"] = float(p)
    return temperature, extra, ladder


def _system(st, name, top):
    """The system prompt of a stage: its own member, else the top-level `system`, else None."""
    v = st.get("system", top)
    if v is not None and (not isinstance(v, str) or not v.strip()):
        raise RecipeError(f"{name}.system must be a line of text; it is {v!r}.")
    return v


def _call_options(st, name, top, out, sample=True, default_temperature=None, one_call=None):
    """Add the members system, sample and thinking of a stage that makes calls to `out`. Each absent member stays
    out of `out`, so the parse of an old recipe does not change. `one_call` says why the stage makes one call, so
    it refuses a ladder."""
    if sample:
        out["temperature"], extra, ladder = _sample(st, name, default_temperature)
        if extra:
            out["sample"] = extra
        if ladder is not None:
            if one_call is not None:
                raise RecipeError(f"{name}.sample.ladder gives a temperature to each member of a stage, and "
                                  f"{one_call}. Use {name}.sample.temperature.")
            out[LADDER] = ladder
    system = _system(st, name, top)
    if system is not None:
        out["system"] = system
    if "thinking" in st:
        if not isinstance(st["thinking"], bool):
            raise RecipeError(f"{name}.thinking must be true or false; it is {st['thinking']!r}.")
        out["thinking"] = st["thinking"]
    return out


def _no_call(st, name, why):
    """Refuse the call members of a stage that makes no call."""
    for k in CALL:
        if k in st:
            raise RecipeError(f"{name}.{k} acts only on a call, and {why}. Remove it.")


def parse(obj):
    """The recipe with each default filled, or a RecipeError."""
    if not isinstance(obj, dict):
        raise RecipeError(f"A recipe must be a JSON object; it is {type(obj).__name__}.")
    for k in obj:
        if k not in STAGES:
            raise RecipeError(f"{k} is not a member of a recipe. The members are {', '.join(STAGES)}.")

    models = obj.get("models")
    if not isinstance(models, list) or not models or not all(isinstance(m, str) and m for m in models):
        raise RecipeError(f"models must be a list of one or more served model names; it is {models!r}.")
    if len(set(models)) != len(models):
        raise RecipeError(f"models names a model twice: {models!r}.")
    top = obj.get("system")
    if top is not None and (not isinstance(top, str) or not top.strip()):
        raise RecipeError(f"system must be a line of text; it is {top!r}.")

    snapshots = obj.get("snapshots", False)
    if not isinstance(snapshots, bool):
        raise RecipeError(f"snapshots must be true or false; it is {snapshots!r}.")

    batch = obj.get("batch", False)
    if not isinstance(batch, bool):
        raise RecipeError(f"batch must be true or false; it is {batch!r}.")

    prime = None
    if "prime" in obj:
        pr = _stage(obj, "prime")
        if "max_tokens" not in pr:
            raise RecipeError("prime.max_tokens is missing. A prime thinks, so give it a token budget, such as 384.")
        prime = _call_options(pr, "prime", top, {"max_tokens": _int(pr, "max_tokens", "prime.max_tokens", None, 1)},
                              default_temperature=0.6, one_call="a prime makes one call")

    t = _stage(obj, "think")
    agents = _int(t, "agents", "think.agents", None, 1, 64)
    foci = t.get("foci", task.FOCI)
    if not isinstance(foci, list) or not foci or not all(isinstance(f, str) and f.strip() for f in foci):
        raise RecipeError(f"think.foci must be a list of one or more lines of text; it is {foci!r}.")
    think = {"agents": agents, "foci": list(foci),
             "max_tokens": _int(t, "max_tokens", "think.max_tokens", 48, 1),
             "temperature": None,
             "latent": _int(t, "latent", "think.latent", 0, 0, LATENT_MAX)}
    _call_options(t, "think", top, think, default_temperature=1.0)

    s = _stage(obj, "select")
    by = _choice(s, "by", "select.by", None, ("vote", "novelty", "all"))
    keep_default = {"vote": 1, "all": agents, "novelty": None}[by]
    keep = _int(s, "keep", "select.keep", keep_default, 1, agents)
    if by == "vote" and keep != 1:
        raise RecipeError(f"select.keep must be 1 with select.by vote, because a vote names one idea; it is {keep}.")
    if by == "all" and keep != agents:
        raise RecipeError(f"select.keep must equal think.agents ({agents}) with select.by all; it is {keep}.")
    select = {"by": by, "keep": keep, "model": s.get("model", EMBED_MODEL),
              "dimensions": _choice(s, "dimensions", "select.dimensions", 256, (768, 512, 256, 128))}
    if not isinstance(select["model"], str) or not select["model"]:
        raise RecipeError(f"select.model must name the embedding model; it is {select['model']!r}.")
    if by == "vote":
        _call_options(s, "select", top, select, sample=False)
    else:
        _no_call(s, "select", f"select.by {by} asks no model for text")

    d = _stage(obj, "distill")
    distill = {"by": _choice(d, "by", "distill.by", None, ("pick", "combine", "joined")),
               "latent": _int(d, "latent", "distill.latent", 0, 0, LATENT_MAX),
               "max_tokens": _int(d, "max_tokens", "distill.max_tokens", 64, 1),
               "temperature": None}
    if distill["by"] == "pick":
        _no_call(d, "distill", "distill.by pick makes no call")
        distill["temperature"] = _num(d, "temperature", "distill.temperature", 0.7, 0.0)
    else:
        _call_options(d, "distill", top, distill, default_temperature=0.7,
                      one_call=f"distill.by {distill['by']} makes one call")
    if distill["by"] == "pick":
        if keep != 1:
            raise RecipeError(f"distill.by pick needs select.keep 1, because it codes one idea; select.keep is {keep}.")
        if distill["latent"]:
            raise RecipeError("distill.latent must be 0 with distill.by pick, because a pick makes no call.")
    if distill["by"] == "joined" and keep > ENSEMBLE_MAX:
        raise RecipeError(f"distill.by joined holds one context for each idea, at most {ENSEMBLE_MAX}; "
                          f"select.keep is {keep}.")

    a = _stage(obj, "act")
    act = {"forks": _int(a, "forks", "act.forks", 1, 1, 64),
           "by": _choice(a, "by", "act.by", "plain", ("plain", "contrast", "joined")),
           "max_tokens": _int(a, "max_tokens", "act.max_tokens", 96, 1),
           "temperature": None,
           "latent": _int(a, "latent", "act.latent", 0, 0, LATENT_MAX)}
    joined = a.get("by") == "joined"
    _call_options(a, "act", top, act, default_temperature=1.0,
                  one_call="act.by joined makes one call" if joined else None)
    if "strength" in a:
        if act["by"] != "contrast":
            raise RecipeError("act.strength acts only with act.by contrast. Remove it, or set act.by to contrast.")
        v = a["strength"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise RecipeError(f"act.strength must be a finite number; it is {v!r}.")
        act["strength"] = float(v)
    if "combine" in a and act["by"] != "joined":
        raise RecipeError("act.combine acts only with act.by joined. Remove it, or set act.by to joined.")
    if act["by"] == "joined":
        act["combine"] = _choice(a, "combine", "act.combine", "logprob_sum", ("logprob_sum", "prob_mean"))
        if act["forks"] > ENSEMBLE_MAX:
            raise RecipeError(f"act.forks must be {ENSEMBLE_MAX} or fewer with act.by joined; it is {act['forks']}.")

    lp = _stage(obj, "loop")
    loop = {"minutes": _num(lp, "minutes", "loop.minutes", None, 0.0),
            "in_flight": _int(lp, "in_flight", "loop.in_flight", 1, 1, 64),
            "parent": _choice(lp, "parent", "loop.parent", PARENT_POLICY, (PARENT_POLICY,))}
    if loop["minutes"] <= 0:
        raise RecipeError("loop.minutes must be above 0.")
    if "rounds" in lp:
        loop["rounds"] = _int(lp, "rounds", "loop.rounds", None, 1)

    out = {"models": list(models), "snapshots": snapshots, "batch": batch, "think": think, "select": select,
           "distill": distill, "act": act, "loop": loop}
    if top is not None:
        out["system"] = top
    if prime is not None:
        out["prime"] = prime
    return out


def load(path):
    """The parsed recipe of a JSON file."""
    with open(path) as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError as e:
            raise RecipeError(f"{path} is not JSON: {e}.") from e
    return parse(obj)
