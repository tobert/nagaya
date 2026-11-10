"""The recipe of a round (docs/swarm-ideas.md, "An agent's interface to a swarm"). A recipe is a JSON object with the
stages think, select, distill, act and loop. The parse fills the defaults and refuses a recipe that it cannot run."""
import json
import math

from . import task

PARENT_POLICY = "tournament of the top 5"
LATENT_MAX = 256
ENSEMBLE_MAX = 8
EMBED_MODEL = "embeddinggemma-2"

STAGES = ("models", "think", "select", "distill", "act", "loop")
KEYS = {
    "think": ("agents", "foci", "max_tokens", "temperature", "latent"),
    "select": ("by", "keep", "model", "dimensions"),
    "distill": ("by", "latent", "max_tokens", "temperature"),
    "act": ("forks", "by", "max_tokens", "temperature", "latent", "strength", "combine"),
    "loop": ("minutes", "in_flight", "parent", "rounds"),
}


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
        if k not in KEYS[name]:
            raise RecipeError(f"{name}.{k} is not a member. The members are {', '.join(KEYS[name])}.")
    return v


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

    t = _stage(obj, "think")
    agents = _int(t, "agents", "think.agents", None, 1, 64)
    foci = t.get("foci", task.FOCI)
    if not isinstance(foci, list) or not foci or not all(isinstance(f, str) and f.strip() for f in foci):
        raise RecipeError(f"think.foci must be a list of one or more lines of text; it is {foci!r}.")
    think = {"agents": agents, "foci": list(foci),
             "max_tokens": _int(t, "max_tokens", "think.max_tokens", 48, 1),
             "temperature": _num(t, "temperature", "think.temperature", 1.0, 0.0),
             "latent": _int(t, "latent", "think.latent", 0, 0, LATENT_MAX)}

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

    d = _stage(obj, "distill")
    distill = {"by": _choice(d, "by", "distill.by", None, ("pick", "combine", "joined")),
               "latent": _int(d, "latent", "distill.latent", 0, 0, LATENT_MAX),
               "max_tokens": _int(d, "max_tokens", "distill.max_tokens", 64, 1),
               "temperature": _num(d, "temperature", "distill.temperature", 0.7, 0.0)}
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
           "temperature": _num(a, "temperature", "act.temperature", 1.0, 0.0),
           "latent": _int(a, "latent", "act.latent", 0, 0, LATENT_MAX)}
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

    return {"models": list(models), "think": think, "select": select, "distill": distill, "act": act, "loop": loop}


def load(path):
    """The parsed recipe of a JSON file."""
    with open(path) as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError as e:
            raise RecipeError(f"{path} is not JSON: {e}.") from e
    return parse(obj)
