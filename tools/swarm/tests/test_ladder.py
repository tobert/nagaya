"""Tests of the temperature ladder: each member of a stage takes its step of `sample.ladder`, the log names the
sample of each program, and the summary counts the programs of each act temperature."""
import os
import unittest

from swarm import client, driver, recipe, task
from swarm.tests.stub import StubTransport
from swarm.tests.test_driver import BASE, run, with_
from swarm.tests.test_options import by_prefill, gens
from swarm.tests.test_recipe import MINIMAL, changed

RECIPES = os.path.join(os.path.dirname(__file__), "..", "recipes")

LADDER = with_(BASE,
               think={"agents": 5, "sample": {"ladder": [0.5, 1.5], "top_p": 0.9}},
               act={"forks": 5, "by": "plain", "sample": {"ladder": [0.3, 0.6, 0.9], "top_k": 40}})


def temperatures_by_seed(bodies):
    return {b["sample"]["seed"]: b["sample"]["temperature"] for b in bodies}


class CodeByTemperature(StubTransport):
    """A stub whose code depends on the temperature of the call, so the counts of each temperature are fixed."""

    LINES = {0.3: "-leftover * 0.9", 0.6: "oops(", 0.9: "-leftover"}

    def _code_reply(self, body):
        return task.CODE_PREFILL + self.LINES[body["sample"]["temperature"]] + "\n```\n"

    def post(self, path, body):
        reply = super().post(path, body)
        if path == client.GENERATIONS and body["context"]["messages"][-1].get("content") == task.CODE_PREFILL:
            reply = {**reply, "content": self._code_reply(body)}
        return reply


class LadderRequestTest(unittest.TestCase):
    def test_each_member_takes_its_step_and_the_ladder_wraps(self):
        stub, _, _ = run(LADDER)
        bodies = gens(stub)
        think = by_prefill(bodies, task.IDEA_PREFILL)
        act = by_prefill(bodies, task.CODE_PREFILL)
        # Round 1: agent i has the seed 1000 + i, and fork f has the seed 100 + f.
        self.assertEqual(temperatures_by_seed(think), {1000: 0.5, 1001: 1.5, 1002: 0.5, 1003: 1.5, 1004: 0.5})
        self.assertEqual(temperatures_by_seed(act), {100: 0.3, 101: 0.6, 102: 0.9, 103: 0.3, 104: 0.6})
        # The other members of `sample` reach every member, and the ladder itself is not in a body.
        for b in think:
            self.assertEqual(list(b["sample"]), ["temperature", "top_p", "seed"])
            self.assertEqual(b["sample"]["top_p"], 0.9)
        for b in act:
            self.assertEqual(list(b["sample"]), ["temperature", "top_k", "seed"])
            self.assertEqual(b["sample"]["top_k"], 40)

    def test_contrast_forks_take_the_ladder(self):
        obj = with_(LADDER, act={"forks": 4, "by": "contrast", "sample": {"ladder": [0.2, 0.4]}})
        stub, _, _ = run(obj)
        ens = [b for p, b in stub.calls if p == client.ENSEMBLE]
        self.assertEqual(temperatures_by_seed(ens), {100: 0.2, 101: 0.4, 102: 0.2, 103: 0.4})

    def test_latent_think_agents_take_the_ladder(self):
        obj = with_(LADDER, think={"agents": 3, "latent": 4, "sample": {"ladder": [0.5, 1.5]}})
        stub, _, _ = run(obj)
        ens = [b for p, b in stub.calls if p == client.ENSEMBLE]
        self.assertEqual(temperatures_by_seed(ens), {1000: 0.5, 1001: 1.5, 1002: 0.5})


class LadderLogTest(unittest.TestCase):
    def test_programs_rounds_and_summary_name_the_temperatures(self):
        obj = with_(LADDER, act={"forks": 6, "by": "plain", "sample": {"ladder": [0.3, 0.6, 0.9], "top_p": 0.95}})
        _, lines, summary = run(obj, stub=CodeByTemperature())
        programs = [x for x in lines if "src" in x and "fork" in x]
        self.assertEqual(len(programs), 6)
        for p in programs:
            self.assertEqual(p["sample"], {"temperature": [0.3, 0.6, 0.9][p["fork"] % 3], "top_p": 0.95})
        r = [x["round_stats"] for x in lines if "round_stats" in x][0]
        self.assertEqual(r["think_temperatures"], [0.5, 1.5, 0.5, 1.5, 0.5])
        # The parent of round 1 is best fit, 4.08. At 0.3 the stub codes 3.5, at 0.6 a failure, at 0.9 4.08.
        self.assertEqual(summary["by_temperature"], [
            {"temperature": 0.3, "programs": 2, "valid": 2, "improved": 2, "best_excess": 3.5},
            {"temperature": 0.6, "programs": 2, "valid": 0, "improved": 0, "best_excess": None},
            {"temperature": 0.9, "programs": 2, "valid": 2, "improved": 0, "best_excess": 4.08}])

    def test_without_a_ladder_the_log_names_the_stage_temperature(self):
        _, lines, summary = run(BASE)
        programs = [x for x in lines if "src" in x and "fork" in x]
        self.assertEqual({tuple(p["sample"].items()) for p in programs}, {(("temperature", 1.0),)})
        r = [x["round_stats"] for x in lines if "round_stats" in x][0]
        self.assertEqual(r["think_temperatures"], [1.0] * 4)
        self.assertEqual([row["temperature"] for row in summary["by_temperature"]], [1.0])
        self.assertEqual(summary["by_temperature"][0]["programs"], 4)

    def test_joined_act_logs_the_stage_temperature(self):
        obj = with_(BASE, act={"forks": 2, "by": "joined", "temperature": 0.4})
        _, lines, _ = run(obj)
        programs = [x for x in lines if "src" in x and "fork" in x]
        self.assertEqual([p["sample"] for p in programs], [{"temperature": 0.4}])

    def test_by_temperature_counts(self):
        recs = [{"sample": {"temperature": 1.2}, "valid": True, "improved": False, "excess": 4.0},
                {"sample": {"temperature": 0.3}, "valid": False, "improved": False},
                {"sample": {"temperature": 1.2}, "valid": True, "improved": True, "excess": 3.0},
                {"sample": {"temperature": 0.3}, "valid": True, "improved": True, "excess": 3.9}]
        self.assertEqual(driver.by_temperature(recs), [
            {"temperature": 0.3, "programs": 2, "valid": 1, "improved": 1, "best_excess": 3.9},
            {"temperature": 1.2, "programs": 2, "valid": 2, "improved": 1, "best_excess": 3.0}])
        self.assertEqual(driver.by_temperature([]), [])


class LadderRecipeTest(unittest.TestCase):
    def test_the_ladder_parses(self):
        r = recipe.parse(LADDER)
        self.assertEqual(r["think"]["ladder"], [0.5, 1.5])
        self.assertIsNone(r["think"]["temperature"])
        self.assertEqual(r["think"]["sample"], {"top_p": 0.9})
        self.assertEqual(r["act"]["ladder"], [0.3, 0.6, 0.9])

    def test_the_ladder_recipe(self):
        r = recipe.load(os.path.join(RECIPES, "s2s-combine-warm-ladder.json"))
        warm = recipe.load(os.path.join(RECIPES, "s2s-combine-warm.json"))
        self.assertEqual((r["act"]["ladder"], r["act"]["sample"]), ([0.3, 0.6, 0.9, 1.2], {"top_p": 0.95}))
        self.assertEqual((r["think"]["ladder"], r["think"]["sample"]),
                         ([0.7, 0.8, 0.9, 1.0, 1.0, 1.1, 1.2, 1.2], {"top_p": 0.95, "top_k": 20}))
        for k in ("models", "select", "distill", "loop"):
            self.assertEqual(r[k], warm[k])

    def test_the_refusals_name_the_member(self):
        ladder = {"ladder": [0.5]}
        cases = [
            (changed(MINIMAL, "think.sample", {"ladder": []}), "think.sample.ladder must be a list"),
            (changed(MINIMAL, "think.sample", {"ladder": 0.5}), "think.sample.ladder must be a list"),
            (changed(MINIMAL, "act.sample", {"ladder": [0.5, -0.1]}), "act.sample.ladder[1]"),
            (changed(MINIMAL, "act.sample", {"ladder": [float("nan")]}), "act.sample.ladder[0]"),
            (changed(MINIMAL, "act.sample", {"ladder": [True]}), "act.sample.ladder[0]"),
            (changed(changed(MINIMAL, "act.sample", ladder), "act.temperature", 0.5),
             "act.sample.ladder and act.temperature"),
            (changed(MINIMAL, "act.sample", {"ladder": [0.5], "temperature": 0.5}),
             "act.sample.ladder and act.sample.temperature"),
            (changed(MINIMAL, "prime", {"max_tokens": 9, "sample": ladder}), "a prime makes one call"),
            (changed(MINIMAL, "distill.sample", ladder), "distill.by combine makes one call"),
            (changed(changed(MINIMAL, "distill.by", "joined"), "distill.sample", ladder),
             "distill.by joined makes one call"),
            (changed(MINIMAL, "act", {"by": "joined", "forks": 2, "sample": ladder}), "act.by joined makes one call"),
            (changed(changed(changed(MINIMAL, "select.by", "vote"), "distill.by", "pick"), "distill.sample", ladder),
             "distill.sample"),
        ]
        for obj, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(recipe.RecipeError) as cm:
                    recipe.parse(obj)
                self.assertIn(message, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
