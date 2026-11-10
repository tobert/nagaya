"""Run a recipe: python3 -m swarm RECIPE --out LOG, in tools/."""
import argparse
import json
import sys

from . import client, driver, recipe, sandbox, transport


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m swarm", description="Run a swarm recipe against the nagaya.")
    ap.add_argument("recipe", help="the recipe, a JSON file")
    ap.add_argument("--out", required=True, help="the JSONL log; the driver writes it again from the start")
    ap.add_argument("--url", help=f"the base URL of the nagaya; else NAGAYA_URL, else {transport.DEFAULT_URL}")
    ap.add_argument("--seed", type=int, default=0, help="the seed of the parents, the foci and the call seeds")
    a = ap.parse_args(argv)
    try:
        r = recipe.load(a.recipe)
    except recipe.RecipeError as e:
        print(f"swarm: the recipe {a.recipe} is not valid. {e}", file=sys.stderr)
        return 2
    c = client.Client(transport.HttpTransport(transport.base_url(a.url)))
    with open(a.out, "w") as f:
        summary = driver.Driver(r, c, sandbox.evaluate, driver.Log(f), seed=a.seed).run()
    summary.pop("best_src")
    summary.pop("recipe")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
