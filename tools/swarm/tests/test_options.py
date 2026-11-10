"""Tests of the call options of a recipe: system, sample, thinking and prime. Each option reaches the body, and a
recipe without them sends the bytes of the driver before them (fixtures/requests-a36b117.json)."""
import copy
import hashlib
import io
import json
import os
import unittest

from swarm import client, driver, recipe, task
from swarm.tests.stub import StubTransport, fake_evaluate
from swarm.tests.test_driver import BASE, run, with_
from swarm.tests.test_recipe import MINIMAL, changed

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "requests-a36b117.json")
RECIPES = os.path.join(os.path.dirname(__file__), "..", "recipes")


def gens(stub):
    return [b for p, b in stub.calls if p == client.GENERATIONS]


def by_prefill(bodies, prefill):
    return [b for b in bodies if b["context"]["messages"][-1].get("content") == prefill]


class DefaultsTest(unittest.TestCase):
    def test_a_recipe_without_options_sends_the_bytes_of_before(self):
        # The fixture holds the SHA-256 of each body that the driver of a36b117 sent, for 7 recipes of 2 rounds.
        with open(FIXTURE) as f:
            cases = json.load(f)
        self.assertEqual(len(cases), 7)
        for case in cases:
            with self.subTest(case=case["name"]):
                stub = StubTransport()
                d = driver.Driver(recipe.parse(case["recipe"]), client.Client(stub), fake_evaluate,
                                  driver.Log(io.StringIO()), seed=3)
                d.run()
                got = sorted(hashlib.sha256(json.dumps([p, b]).encode()).hexdigest() for p, b in stub.calls)
                self.assertEqual(got, case["calls"])

    def test_an_old_recipe_parses_with_no_new_member(self):
        r = recipe.parse(with_(BASE, distill={"by": "combine"}, select={"by": "all"}))
        for stage in ("think", "distill", "act"):
            self.assertFalse({"system", "sample", "thinking"} & r[stage].keys(), stage)
        self.assertNotIn("prime", r)
        self.assertNotIn("system", r)


class ClientOptionsTest(unittest.TestCase):
    def setUp(self):
        self.stub = StubTransport()
        self.c = client.Client(self.stub)

    def test_system_sample_and_thinking_reach_the_body_exactly(self):
        self.c.generate("lfm3", "U", 48, 1.0, 7, task.IDEA_PREFILL, system="S",
                        sample={"top_p": 0.95, "top_k": 20, "min_p": 0.05}, thinking=True)
        path, body = self.stub.calls[0]
        self.assertEqual(path, client.GENERATIONS)
        self.assertEqual(json.dumps(body), json.dumps({
            "model": "lfm3", "max_tokens": 48,
            "sample": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "seed": 7},
            "context": {"thinking": True, "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": "Idea:", "continue": True}]}}))

    def test_each_context_of_an_ensemble_takes_the_options(self):
        self.c.ensemble("lfm3", ["A", "B"], "contrast", 96, 0.1, 5, task.CODE_PREFILL, system="S",
                        sample={"top_k": 50}, thinking=True)
        body = self.stub.calls[0][1]
        self.assertEqual(body["sample"], {"temperature": 0.1, "top_k": 50, "seed": 5})
        for c in body["contexts"]:
            self.assertTrue(c["thinking"])
            self.assertEqual(c["messages"][0], {"role": "system", "content": "S"})
            self.assertEqual(len(c["messages"]), 3)

    def test_a_prime_thinks_with_no_prefill_and_gives_reasoning_then_content(self):
        text, usage = self.c.prime("lfm3", "H", 384, 0.6, 9, system="S", sample={"top_p": 0.95})
        body = self.stub.calls[0][1]
        self.assertEqual(body["max_tokens"], 384)
        self.assertEqual(body["sample"], {"temperature": 0.6, "top_p": 0.95, "seed": 9})
        self.assertEqual(body["context"], {"thinking": True, "messages": [{"role": "system", "content": "S"},
                                                                          {"role": "user", "content": "H"}]})
        self.assertEqual(text, "Small gaps waste bins.\n\nTry a tighter fit.")
        self.assertEqual(usage["completion"], 5)

    def test_a_cut_prime_gives_its_reasoning(self):
        c = client.Client(StubTransport(prime={"reasoning": "half a thou", "content": ""}))
        self.assertEqual(c.prime("lfm3", "H", 8, 0.6, 1)[0], "half a thou")


CARD = {"models": ["lfm3", "lfm3-1"],
        "system": "SYS",
        "think": {"agents": 4, "system": "THINK", "sample": {"temperature": 0.9, "top_p": 0.95, "top_k": 20}},
        "select": {"by": "vote", "thinking": True},
        "distill": {"by": "pick"},
        "act": {"forks": 2, "by": "plain", "temperature": 0.1, "sample": {"top_k": 50}},
        "loop": {"minutes": 5, "in_flight": 1, "rounds": 1}}


class DriverOptionsTest(unittest.TestCase):
    def test_each_stage_sends_its_own_options(self):
        stub, _, _ = run(CARD)
        bodies = gens(stub)
        think = by_prefill(bodies, task.IDEA_PREFILL)
        vote = by_prefill(bodies, task.VOTE_PREFILL)
        act = by_prefill(bodies, task.CODE_PREFILL)
        self.assertEqual((len(think), len(vote), len(act)), (4, 1, 2))
        for b in think:
            self.assertEqual(b["context"]["messages"][0], {"role": "system", "content": "THINK"})
            self.assertEqual(list(b["sample"]), ["temperature", "top_p", "top_k", "seed"])
            self.assertEqual((b["sample"]["temperature"], b["sample"]["top_p"], b["sample"]["top_k"]),
                             (0.9, 0.95, 20))
            self.assertFalse(b["context"]["thinking"])
        # The vote and act set no system, so the top-level one reaches them.
        self.assertEqual(vote[0]["context"]["messages"][0], {"role": "system", "content": "SYS"})
        self.assertTrue(vote[0]["context"]["thinking"])
        self.assertEqual(vote[0]["sample"]["temperature"], 0.0)
        self.assertEqual(set(vote[0]["sample"]), {"temperature", "seed"})
        for b in act:
            self.assertEqual(b["context"]["messages"][0], {"role": "system", "content": "SYS"})
            self.assertEqual(b["sample"], {"temperature": 0.1, "top_k": 50, "seed": b["sample"]["seed"]})
            self.assertFalse(b["context"]["thinking"])

    def test_combine_and_contrast_take_the_options(self):
        obj = with_(CARD, select={"by": "all"}, distill={"by": "combine", "system": "D", "thinking": True},
                    act={"forks": 2, "by": "contrast", "system": "A"})
        stub, _, _ = run(obj)
        plan = by_prefill(gens(stub), task.PLAN_PREFILL)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["context"]["messages"][0], {"role": "system", "content": "D"})
        self.assertTrue(plan[0]["context"]["thinking"])
        ens = [b for p, b in stub.calls if p == client.ENSEMBLE]
        self.assertEqual(len(ens), 2)
        for b in ens:
            self.assertEqual([c["messages"][0]["content"] for c in b["contexts"]], ["A", "A"])


PRIMED = with_(CARD, prime={"max_tokens": 384, "sample": {"temperature": 0.6, "top_p": 0.95}})


class PrimeTest(unittest.TestCase):
    def test_the_prime_reply_sits_between_the_head_and_the_focus(self):
        stub, lines, _ = run(PRIMED)
        bodies = gens(stub)
        primes = [b for b in bodies if b["context"]["messages"][-1]["role"] == "user"]
        self.assertEqual(len(primes), 1)
        p = primes[0]
        self.assertEqual(p["max_tokens"], 384)
        self.assertEqual(p["sample"], {"temperature": 0.6, "top_p": 0.95, "seed": 1})
        self.assertTrue(p["context"]["thinking"])
        system, head = p["context"]["messages"]
        self.assertEqual(system, {"role": "system", "content": "SYS"})
        self.assertTrue(head["content"].endswith(task.PRIME_ASK))

        think = by_prefill(bodies, task.IDEA_PREFILL)
        self.assertEqual(len(think), 4)
        for b in think:
            m = b["context"]["messages"]
            self.assertEqual([x["role"] for x in m], ["system", "user", "assistant", "user", "assistant"])
            self.assertEqual(m[0], {"role": "system", "content": "THINK"})
            self.assertEqual(m[1], head)
            self.assertEqual(m[2], {"role": "assistant", "content": "Small gaps waste bins.\n\nTry a tighter fit."})
            self.assertTrue(m[3]["content"].startswith("Focus: "))
            self.assertTrue(m[3]["content"].endswith(task.THINK_ASK))
            self.assertEqual(m[4], {"role": "assistant", "content": task.IDEA_PREFILL, "continue": True})
        # The agents differ only in the focus: every message before it is the same.
        self.assertEqual(len({json.dumps(b["context"]["messages"][:3]) for b in think}), 1)

        r = [x["round_stats"] for x in lines if "round_stats" in x][0]
        self.assertEqual(list(r["stages"])[0], "prime")
        self.assertEqual(r["stages"]["prime"]["calls"], 1)
        self.assertEqual(r["stages"]["prime"]["completion"], 5)
        self.assertEqual(r["prime"], "Small gaps waste bins.\n\nTry a tighter fit.")

    def test_no_prime_stage_without_prime(self):
        _, lines, summary = run(CARD)
        r = [x["round_stats"] for x in lines if "round_stats" in x][0]
        self.assertNotIn("prime", r["stages"])
        self.assertNotIn("prime", r)
        self.assertNotIn("prime", summary["stages"])

    def test_latent_think_agents_take_the_primed_messages(self):
        obj = copy.deepcopy(PRIMED)
        obj["think"]["latent"] = 8
        stub, _, _ = run(obj)
        ens = [b for p, b in stub.calls if p == client.ENSEMBLE]
        self.assertEqual(len(ens), 4)
        for b in ens:
            self.assertEqual(len(b["contexts"]), 1)
            self.assertEqual(b["contexts"][0]["messages"][2]["role"], "assistant")


class RecipeOptionsTest(unittest.TestCase):
    def test_the_options_parse(self):
        r = recipe.parse(PRIMED)
        self.assertEqual(r["system"], "SYS")
        self.assertEqual(r["think"]["system"], "THINK")
        self.assertEqual(r["think"]["temperature"], 0.9)
        self.assertEqual(r["think"]["sample"], {"top_p": 0.95, "top_k": 20})
        self.assertEqual(r["act"]["system"], "SYS")
        self.assertEqual(r["select"]["system"], "SYS")
        self.assertTrue(r["select"]["thinking"])
        self.assertNotIn("system", r["distill"])
        self.assertEqual(r["prime"], {"max_tokens": 384, "temperature": 0.6, "sample": {"top_p": 0.95},
                                      "system": "SYS"})

    def test_the_card_recipes(self):
        plain = recipe.load(os.path.join(RECIPES, "s2s-combine-card.json"))
        primed = recipe.load(os.path.join(RECIPES, "s2s-combine-card-prime.json"))
        for r in (plain, primed):
            self.assertEqual((r["think"]["temperature"], r["think"]["sample"]), (1.0, {"top_p": 0.95, "top_k": 20}))
            self.assertEqual((r["distill"]["temperature"], r["distill"]["sample"]), (0.6, {"top_p": 0.95}))
            self.assertEqual((r["act"]["temperature"], r["act"]["sample"]), (0.1, {"top_k": 50}))
            for stage in ("think", "distill", "act"):
                self.assertIn("system", r[stage])
        self.assertNotIn("prime", plain)
        self.assertEqual(primed["prime"]["max_tokens"], 384)
        self.assertEqual((primed["prime"]["temperature"], primed["prime"]["sample"]), (0.6, {"top_p": 0.95}))
        base = recipe.load(os.path.join(RECIPES, "s2s-combine.json"))
        for k in ("models", "select", "loop"):
            self.assertEqual(plain[k], base[k])

    def test_errors_name_the_member(self):
        cases = [
            (changed(changed(MINIMAL, "think.temperature", 1.0), "think.sample", {"temperature": 0.5}),
             "think.temperature and think.sample.temperature"),
            (changed(MINIMAL, "act.sample", {"repetition_penalty": 1.1}), "act.sample.repetition_penalty"),
            (changed(MINIMAL, "act.repetition_penalty", 1.1), "act.repetition_penalty"),
            (changed(MINIMAL, "think.sample", {"top_q": 0.5}), "think.sample.top_q"),
            (changed(MINIMAL, "think.sample", {"seed": 1}), "think.sample.seed"),
            (changed(MINIMAL, "think.sample", []), "think.sample"),
            (changed(MINIMAL, "think.sample", {"top_p": 0.0}), "think.sample.top_p"),
            (changed(MINIMAL, "think.sample", {"top_k": -1}), "think.sample.top_k"),
            (changed(MINIMAL, "think.sample", {"min_p": 1.0}), "think.sample.min_p"),
            (changed(MINIMAL, "think.thinking", "yes"), "think.thinking"),
            (changed(MINIMAL, "think.system", ""), "think.system"),
            (changed(MINIMAL, "system", 3), "system"),
            (changed(MINIMAL, "prime", {"sample": {"temperature": 0.6}}), "prime.max_tokens"),
            (changed(MINIMAL, "prime", {"max_tokens": 0}), "prime.max_tokens"),
            (changed(MINIMAL, "prime", {"max_tokens": 9, "thinking": False}), "prime.thinking"),
            (changed(MINIMAL, "select.system", "S"), "select.system"),
            (changed(changed(changed(MINIMAL, "select.by", "vote"), "distill.by", "pick"), "distill.sample",
                     {"top_k": 5}), "distill.sample"),
            (changed(changed(MINIMAL, "select.by", "vote"), "select.sample", {"top_k": 5}), "select.sample"),
        ]
        for obj, member in cases:
            with self.subTest(member=member):
                with self.assertRaises(recipe.RecipeError) as cm:
                    recipe.parse(obj)
                self.assertIn(member, str(cm.exception))

    def test_repetition_penalty_says_why(self):
        with self.assertRaises(recipe.RecipeError) as cm:
            recipe.parse(changed(MINIMAL, "act.sample", {"repetition_penalty": 1.1}))
        self.assertIn("no repetition penalty", str(cm.exception))

    def test_a_prime_without_a_budget_says_why(self):
        with self.assertRaises(recipe.RecipeError) as cm:
            recipe.parse(changed(MINIMAL, "prime", {"system": "S"}))
        self.assertIn("prime.max_tokens is missing", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
