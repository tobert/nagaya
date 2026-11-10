"""Tests of the recipe parse and its errors."""
import copy
import glob
import os
import unittest

from swarm import recipe, task

DOC_EXAMPLE = {
    "models": ["lfm3", "lfm3-1"],
    "think": {"agents": 8, "foci": ["…"], "max_tokens": 48, "temperature": 1.0, "latent": 0},
    "select": {"by": "vote", "keep": 1},
    "distill": {"by": "combine", "latent": 16, "max_tokens": 64, "temperature": 0.7},
    "act": {"forks": 4, "by": "plain", "max_tokens": 96, "temperature": 1.0},
    "loop": {"minutes": 20, "in_flight": 4, "parent": "tournament of the top 5"},
}

MINIMAL = {"models": ["lfm3"], "think": {"agents": 8}, "select": {"by": "all"}, "distill": {"by": "combine"},
           "loop": {"minutes": 1}}


def changed(base, path, value):
    """A copy of base with one member set; the value None removes the member."""
    out = copy.deepcopy(base)
    *head, last = path.split(".")
    obj = out
    for k in head:
        obj = obj.setdefault(k, {})
    if value is None:
        obj.pop(last, None)
    else:
        obj[last] = value
    return out


class RecipeTest(unittest.TestCase):
    def test_the_example_of_the_doc_parses(self):
        r = recipe.parse(DOC_EXAMPLE)
        self.assertEqual(r["models"], ["lfm3", "lfm3-1"])
        self.assertEqual(r["think"]["foci"], ["…"])
        self.assertEqual(r["distill"], {"by": "combine", "latent": 16, "max_tokens": 64, "temperature": 0.7})
        self.assertEqual(r["act"]["forks"], 4)
        self.assertEqual(r["loop"]["in_flight"], 4)

    def test_the_defaults(self):
        r = recipe.parse(MINIMAL)
        self.assertEqual(r["think"], {"agents": 8, "foci": task.FOCI, "max_tokens": 48, "temperature": 1.0,
                                      "latent": 0})
        self.assertEqual(r["select"]["keep"], 8)
        self.assertEqual(r["select"]["model"], "embeddinggemma-2")
        self.assertEqual(r["distill"], {"by": "combine", "latent": 0, "max_tokens": 64, "temperature": 0.7})
        self.assertEqual(r["act"], {"forks": 1, "by": "plain", "max_tokens": 96, "temperature": 1.0, "latent": 0})
        self.assertEqual(r["loop"], {"minutes": 1.0, "in_flight": 1, "parent": recipe.PARENT_POLICY})

    def test_a_vote_keeps_one_by_default(self):
        r = recipe.parse(changed(changed(MINIMAL, "select.by", "vote"), "distill.by", "pick"))
        self.assertEqual(r["select"]["keep"], 1)

    def test_joined_act_takes_its_combine(self):
        r = recipe.parse(changed(MINIMAL, "act", {"by": "joined", "forks": 4}))
        self.assertEqual(r["act"]["combine"], "logprob_sum")

    def test_each_recipe_file_parses(self):
        here = os.path.join(os.path.dirname(__file__), "..", "recipes")
        files = sorted(glob.glob(os.path.join(here, "*.json")))
        self.assertEqual([os.path.basename(f) for f in files],
                         ["pick-vote.json", "s2s-combine-card-prime.json", "s2s-combine-card.json",
                          "s2s-combine-latent.json", "s2s-combine.json", "s2s-vote.json"])
        for f in files:
            with self.subTest(f=f):
                recipe.load(f)

    def test_errors_name_the_member(self):
        cases = [
            (changed(MINIMAL, "models", None), "models"),
            (changed(MINIMAL, "models", []), "models"),
            (changed(MINIMAL, "models", ["lfm3", "lfm3"]), "models"),
            (changed(MINIMAL, "extra", 1), "extra"),
            (changed(MINIMAL, "think.agents", None), "think.agents"),
            (changed(MINIMAL, "think.agents", 0), "think.agents"),
            (changed(MINIMAL, "think.agents", True), "think.agents"),
            (changed(MINIMAL, "think.agent", 8), "think.agent"),
            (changed(MINIMAL, "think.foci", []), "think.foci"),
            (changed(MINIMAL, "think.latent", 257), "think.latent"),
            (changed(MINIMAL, "think.temperature", float("nan")), "think.temperature"),
            (changed(MINIMAL, "select.by", "best"), "select.by"),
            (changed(changed(MINIMAL, "select.by", "vote"), "select.keep", 2), "select.keep"),
            (changed(MINIMAL, "select.keep", 3), "select.keep"),
            (changed(MINIMAL, "select.by", "novelty"), "select.keep"),
            (changed(changed(MINIMAL, "select.by", "novelty"), "select.keep", 9), "select.keep"),
            (changed(MINIMAL, "select.dimensions", 100), "select.dimensions"),
            (changed(MINIMAL, "distill.by", None), "distill.by"),
            (changed(MINIMAL, "distill.by", "pick"), "select.keep"),
            (changed(changed(changed(MINIMAL, "select.by", "vote"), "distill.by", "pick"), "distill.latent", 4),
             "distill.latent"),
            (changed(changed(MINIMAL, "think.agents", 9), "distill.by", "joined"), "select.keep"),
            (changed(MINIMAL, "act.by", "both"), "act.by"),
            (changed(MINIMAL, "act.strength", 0.5), "act.strength"),
            (changed(changed(MINIMAL, "act.by", "contrast"), "act.strength", float("inf")), "act.strength"),
            (changed(MINIMAL, "act.combine", "prob_mean"), "act.combine"),
            (changed(changed(MINIMAL, "act.by", "joined"), "act.forks", 9), "act.forks"),
            (changed(changed(MINIMAL, "act.by", "joined"), "act.combine", "contrast"), "act.combine"),
            (changed(MINIMAL, "loop.minutes", None), "loop.minutes"),
            (changed(MINIMAL, "loop.minutes", 0), "loop.minutes"),
            (changed(MINIMAL, "loop.parent", "the best"), "loop.parent"),
            (changed(MINIMAL, "loop.rounds", 0), "loop.rounds"),
            (changed(MINIMAL, "loop", []), "loop"),
        ]
        for obj, member in cases:
            with self.subTest(member=member, obj=obj):
                with self.assertRaises(recipe.RecipeError) as cm:
                    recipe.parse(obj)
                self.assertIn(member, str(cm.exception))

    def test_a_recipe_must_be_an_object(self):
        with self.assertRaises(recipe.RecipeError):
            recipe.parse([MINIMAL])


if __name__ == "__main__":
    unittest.main()
