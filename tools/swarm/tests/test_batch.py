"""Tests of the member `batch` of a recipe: a stage sends one request for each model, with `contexts` for the think
agents and `forks` for the act forks (gogi "Several replies"). Thus a round makes few round trips."""
import unittest

from swarm import client, recipe, task
from swarm.recipe import RecipeError
from swarm.tests.stub import StubTransport
from swarm.tests.test_driver import BASE, run, with_

ON = with_(BASE, snapshots=True, batch=True)


def gens(stub):
    return [b for p, b in stub.calls if p == client.GENERATIONS]


class RecipeTest(unittest.TestCase):
    def test_batch_is_off_by_default_and_must_be_true_or_false(self):
        self.assertIs(recipe.parse(BASE)["batch"], False)
        self.assertIs(recipe.parse(ON)["batch"], True)
        with self.assertRaisesRegex(RecipeError, "batch must be true or false"):
            recipe.parse(with_(BASE, batch="yes"))


class DriverTest(unittest.TestCase):
    def test_a_recipe_with_batch_off_sends_one_request_for_each_reply(self):
        stub, _, _ = run(with_(BASE, snapshots=True))
        self.assertFalse([b for b in gens(stub) if "contexts" in b or "forks" in b])

    def test_the_think_agents_of_one_model_are_the_contexts_of_one_request(self):
        stub, lines, _ = run(ON)
        think = [b for b in gens(stub) if "contexts" in b]
        # BASE has 4 agents on 2 models: one request for each model, with 2 contexts each.
        self.assertEqual(sorted(b["model"] for b in think), ["lfm3", "lfm3-1"])
        for b in think:
            self.assertNotIn("context", b)
            self.assertEqual(len(b["contexts"]), 2)
            for c in b["contexts"]:
                self.assertEqual(c["messages"][0]["snap"], True)
                self.assertTrue(c["messages"][1]["content"].startswith("Focus: "))
                self.assertEqual(c["messages"][-1],
                                 {"role": "assistant", "content": task.IDEA_PREFILL, "continue": True})
            self.assertEqual(b["sample"]["temperature"], 1.0)
        # The 4 ideas reach the round, in the order of the agents.
        self.assertEqual(len(lines[0]["ideas"]), 4)
        stats = lines[-2]["round_stats"]["stages"]["think"]
        self.assertEqual((stats["calls"], stats["completion"]), (2, 4 * 5))

    def test_the_agents_of_a_ladder_go_in_one_request_for_each_temperature(self):
        stub, _, _ = run(with_(ON, models=["lfm3"], think={"agents": 4, "sample": {"ladder": [0.5, 1.0]}}))
        think = [b for b in gens(stub) if "contexts" in b]
        self.assertEqual(sorted((b["sample"]["temperature"], len(b["contexts"])) for b in think), [(0.5, 2), (1.0, 2)])

    def test_the_seed_of_each_request_differs_so_no_two_agents_share_a_seed(self):
        stub, _, _ = run(with_(ON, models=["lfm3"], think={"agents": 4, "sample": {"ladder": [0.5, 1.0]}}))
        seeds = set()
        for b in [b for b in gens(stub) if "contexts" in b]:
            for i in range(len(b["contexts"])):
                seeds.add(b["sample"]["seed"] + i)
        self.assertEqual(len(seeds), 4)

    def test_the_act_forks_of_one_model_are_the_forks_of_one_request(self):
        stub, lines, _ = run(with_(ON, act={"forks": 4, "by": "plain", "sample": {"ladder": [0.3, 0.9]}}))
        act = [b for b in gens(stub) if "forks" in b]
        self.assertEqual(sorted(b["model"] for b in act), ["lfm3", "lfm3-1"])
        for b in act:
            self.assertEqual(b["context"]["messages"][-1],
                             {"role": "assistant", "content": task.CODE_PREFILL, "continue": True})
            self.assertEqual([m.get("snap") for m in b["context"]["messages"][:-1]], [True, True])
            # Each fork has its own temperature, in a list.
            self.assertEqual(len(b["forks"]), 2)
            self.assertTrue(all(set(f) == {"temperature"} for f in b["forks"]))
        temperatures = sorted(f["temperature"] for b in act for f in b["forks"])
        self.assertEqual(temperatures, [0.3, 0.3, 0.9, 0.9])
        programs = [line for line in lines if "fork" in line]
        self.assertEqual(sorted(p["fork"] for p in programs), [0, 1, 2, 3])
        self.assertEqual(sorted(p["sample"]["temperature"] for p in programs), [0.3, 0.3, 0.9, 0.9])
        stats = lines[-2]["round_stats"]["stages"]["act"]
        self.assertEqual(stats["calls"], 2)

    def test_each_fork_of_one_request_has_the_temperature_of_its_place_in_the_ladder(self):
        stub, _, _ = run(with_(ON, models=["lfm3"], act={"forks": 4, "by": "plain", "sample": {"ladder": [0.3, 0.9]}}))
        act = [b for b in gens(stub) if "forks" in b]
        self.assertEqual([b["forks"] for b in act], [[{"temperature": t} for t in (0.3, 0.9, 0.3, 0.9)]])

    def test_the_ideas_keep_the_order_of_the_agents_when_the_models_take_them_in_turn(self):
        # A transport that gives each agent its own focus back. With 2 models, the agents 0 and 2 go to one model and
        # 1 and 3 to the other, so the replies of the two requests must go back to their places.
        class Echo(StubTransport):
            def post(self, path, body):
                reply = super().post(path, body)
                focus = lambda c: task.IDEA_PREFILL + " " + c["messages"][-2]["content"].split("\n")[0]
                if path == client.GENERATIONS and "contexts" in body:
                    for r, c in zip(reply["replies"], body["contexts"]):
                        r["content"] = focus(c)
                elif path == client.GENERATIONS and body["context"]["messages"][-1].get("content") == task.IDEA_PREFILL:
                    reply["content"] = focus(body["context"])
                return reply
        one_by_one = run(with_(BASE, snapshots=True), Echo())[1][0]["ideas"]
        together = run(ON, Echo())[1][0]["ideas"]
        self.assertEqual(len(set(one_by_one)), 4)
        self.assertEqual(together, one_by_one)

    def test_a_stage_with_latent_steps_or_joined_forks_keeps_its_own_calls(self):
        stub, _, _ = run(with_(ON, think={"agents": 4, "latent": 2}, act={"forks": 2, "by": "contrast"}))
        self.assertFalse([b for b in gens(stub) if "contexts" in b or "forks" in b])
        self.assertTrue([p for p in stub.paths() if p == client.ENSEMBLE])


if __name__ == "__main__":
    unittest.main()
