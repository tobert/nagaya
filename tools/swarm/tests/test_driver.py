"""Tests of the driver: the spread over models, the stats, and whole rounds with the stub transport."""
import copy
import io
import itertools
import json
import unittest

from swarm import client, driver, recipe
from swarm.tests.stub import StubTransport, fake_evaluate

BASE = {"models": ["lfm3", "lfm3-1"],
        "think": {"agents": 4},
        "select": {"by": "vote"},
        "distill": {"by": "pick"},
        "act": {"forks": 4, "by": "plain"},
        "loop": {"minutes": 5, "in_flight": 1, "rounds": 1}}


def run(obj, stub=None, seed=0):
    stub = stub or StubTransport()
    out = io.StringIO()
    d = driver.Driver(recipe.parse(obj), client.Client(stub), fake_evaluate, driver.Log(out), seed=seed)
    summary = d.run()
    lines = [json.loads(x) for x in out.getvalue().splitlines()]
    return stub, lines, summary


def with_(base, **stages):
    out = copy.deepcopy(base)
    for k, v in stages.items():
        out[k] = v
    return out


class SpreadTest(unittest.TestCase):
    def test_round_robin(self):
        s = driver.Spread(["a", "b", "c"])
        self.assertEqual([s.next() for _ in range(7)], ["a", "b", "c", "a", "b", "c", "a"])


class StatsTest(unittest.TestCase):
    def test_shares(self):
        progs = [{"valid": True, "constant": False, "improved": True},
                 {"valid": False, "constant": True, "improved": False},
                 {"valid": True, "constant": False, "improved": False},
                 {"valid": False, "constant": False, "improved": False}]
        self.assertEqual(driver.shares(progs), {"programs": 4, "valid": 2, "valid_share": 0.5, "constant": 1,
                                                "constant_share": 0.25, "improved": 1, "improved_share": 0.25})

    def test_shares_of_nothing(self):
        self.assertIsNone(driver.shares([])["valid_share"])

    def test_totals_sum_the_stages(self):
        def rnd(calls, prompt, seconds, model):
            return {"programs": 2, "valid": 1, "constant": 1, "improved": 0,
                    "stages": {"think": {"calls": calls, "prompt": prompt, "fed": 1, "completion": 2,
                                         "seconds": seconds, "models": {model: calls}}}}
        t = driver.totals([rnd(4, 100, 1.5, "a"), rnd(2, 50, 0.25, "b"), rnd(1, 1, 0.25, "a")])
        self.assertEqual(t["rounds"], 3)
        self.assertEqual(t["programs"], 6)
        self.assertEqual(t["valid_share"], 0.5)
        self.assertEqual(t["stages"]["think"], {"calls": 7, "prompt": 151, "fed": 3, "completion": 6,
                                                "seconds": 2.0, "models": {"a": 5, "b": 2}})


class RoundTest(unittest.TestCase):
    def test_one_round_of_vote_pick_and_plain_forks(self):
        stub, lines, summary = run(BASE)
        gens = [b for p, b in stub.calls if p == client.GENERATIONS]
        self.assertEqual(stub.paths(), [client.GENERATIONS] * 9)
        # The calls run on threads, so the seed of each call gives its place in the spread. Four think agents
        # take lfm3, lfm3-1, lfm3, lfm3-1; the vote takes lfm3; the four forks take lfm3-1, lfm3, lfm3-1, lfm3.
        think = sorted((b for b in gens if b["sample"]["seed"] >= 1000), key=lambda b: b["sample"]["seed"])
        forks = sorted((b for b in gens if 100 <= b["sample"]["seed"] < 1000), key=lambda b: b["sample"]["seed"])
        votes = [b for b in gens if b["sample"]["seed"] == 1]
        self.assertEqual([b["model"] for b in think], ["lfm3", "lfm3-1", "lfm3", "lfm3-1"])
        self.assertEqual([b["model"] for b in votes], ["lfm3"])
        self.assertEqual([b["model"] for b in forks], ["lfm3-1", "lfm3", "lfm3-1", "lfm3"])
        vote = votes[0]
        self.assertEqual(vote["max_tokens"], 4)
        self.assertEqual(vote["sample"]["temperature"], 0.0)
        # The vote names idea 2, so each fork codes the second idea.
        for b in forks:
            self.assertIn("Apply this idea: Idea: Favor bins", b["context"]["messages"][0]["content"])

        programs = [x for x in lines if "src" in x]
        rounds = [x["round_stats"] for x in lines if "round_stats" in x]
        self.assertEqual(len(programs), 4)
        self.assertEqual(len(rounds), 1)
        self.assertEqual(lines[-1].keys(), {"summary"})
        r = rounds[0]
        self.assertEqual((r["programs"], r["valid"], r["constant"], r["improved"]), (4, 2, 1, 1))
        self.assertEqual(r["vote"], 1)
        self.assertFalse(r["bad_vote"])
        self.assertEqual(r["stages"]["think"]["calls"], 4)
        self.assertEqual(r["stages"]["think"]["prompt"], 400)
        self.assertEqual(r["stages"]["think"]["models"], {"lfm3": 2, "lfm3-1": 2})
        self.assertEqual(r["stages"]["select"]["calls"], 1)
        self.assertEqual(r["stages"]["distill"]["calls"], 0)
        self.assertEqual(r["stages"]["act"]["calls"], 4)
        self.assertEqual(r["stages"]["act"]["completion"], 20)
        self.assertEqual(summary["best_train"], 3.5)
        self.assertAlmostEqual(summary["best_test"], 3.56)
        self.assertEqual(summary["rounds"], 1)
        self.assertEqual(summary["constant_share"], 0.25)
        self.assertEqual(summary["stages"]["act"]["calls"], 4)
        kinds = {p["kind"] for p in programs}
        self.assertEqual(kinds, {"vote-pick-plain"})

    def test_a_bad_vote_is_logged(self):
        _, lines, _ = run(BASE, StubTransport(vote="Best idea: none"))
        r = [x["round_stats"] for x in lines if "round_stats" in x][0]
        self.assertTrue(r["bad_vote"])

    def test_combine_with_latent_steps(self):
        obj = with_(BASE, select={"by": "all"}, distill={"by": "combine", "latent": 16})
        stub, _, _ = run(obj)
        ens = [b for p, b in stub.calls if p == client.ENSEMBLE]
        self.assertEqual(len(ens), 1)
        self.assertEqual(ens[0]["latent"], {"steps": 16})
        user = ens[0]["contexts"][0]["messages"][0]["content"]
        self.assertTrue(user.index("4. Idea:") < user.index("def priority"))
        for p, b in stub.calls:
            if p == client.GENERATIONS:
                self.assertNotIn("latent", b)

    def test_novelty_keeps_far_ideas_and_joined_distill(self):
        obj = with_(BASE, select={"by": "novelty", "keep": 3}, distill={"by": "joined"})
        stub, lines, _ = run(obj)
        self.assertEqual(stub.paths().count(client.EMBEDDINGS), 1)
        ens = [b for p, b in stub.calls if p == client.ENSEMBLE]
        self.assertEqual(len(ens), 1)
        self.assertEqual(ens[0]["combine"], "prob_mean")
        self.assertEqual(len(ens[0]["contexts"]), 3)
        self.assertNotIn("latent", ens[0])
        r = [x["round_stats"] for x in lines if "round_stats" in x][0]
        self.assertEqual(len(set(r["kept"])), 3)
        self.assertEqual(r["stages"]["select"]["models"], {"embeddinggemma-2": 1})

    def test_contrast_forks_spread_and_joined_act_is_one_call(self):
        obj = with_(BASE, act={"forks": 2, "by": "contrast", "strength": 0.25})
        stub, _, _ = run(obj)
        ens = sorted((b for p, b in stub.calls if p == client.ENSEMBLE), key=lambda b: b["sample"]["seed"])
        self.assertEqual([b["model"] for b in ens], ["lfm3-1", "lfm3"])
        for b in ens:
            self.assertEqual(b["combine"], "contrast")
            self.assertEqual(b["strength"], 0.25)
            self.assertNotIn("Apply this idea", b["contexts"][1]["messages"][0]["content"])

        obj = with_(BASE, act={"forks": 4, "by": "joined"})
        stub, lines, _ = run(obj)
        ens = [b for p, b in stub.calls if p == client.ENSEMBLE]
        self.assertEqual(len(ens), 1)
        self.assertEqual(len(ens[0]["contexts"]), 4)
        self.assertEqual(len([x for x in lines if "src" in x]), 1)

    def test_rounds_in_flight_stop_at_the_limit(self):
        obj = with_(BASE, loop={"minutes": 5, "in_flight": 3, "rounds": 5})
        stub, lines, summary = run(obj)
        self.assertEqual(summary["rounds"], 5)
        self.assertEqual(sorted(x["round_stats"]["round"] for x in lines if "round_stats" in x), [1, 2, 3, 4, 5])
        self.assertEqual(len(stub.calls), 5 * 9)

    def test_the_minutes_end_the_run(self):
        ticks = itertools.count(0, 30)
        obj = with_(BASE, loop={"minutes": 1, "in_flight": 1})
        out = io.StringIO()
        d = driver.Driver(recipe.parse(obj), client.Client(StubTransport()), fake_evaluate, driver.Log(out),
                          clock=lambda: next(ticks))
        summary = d.run()
        self.assertGreaterEqual(summary["rounds"], 1)
        self.assertLess(summary["rounds"], 3)


if __name__ == "__main__":
    unittest.main()
