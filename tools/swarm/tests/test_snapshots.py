"""Tests of the member `snapshots` of a recipe: the shared part of the prompts of a stage is its own messages with a
boundary, and the driver builds it once for each model before the calls that share it."""
import json
import unittest

from swarm import client, recipe, task
from swarm.recipe import RecipeError
from swarm.tests.stub import StubTransport
from swarm.tests.test_driver import BASE, run, with_

ON = with_(BASE, snapshots=True)


def builds(stub):
    return [b for p, b in stub.calls if p == client.CONTEXTS]


def gens(stub, prefill):
    return [b for p, b in stub.calls if p == client.GENERATIONS and
            b["context"]["messages"][-1].get("content") == prefill]


def first(stub, path, prefill):
    """The index of the first call of `path` whose last message holds `prefill`."""
    return next(i for i, (p, b) in enumerate(stub.calls) if p == path and
                b["context"]["messages"][-1].get("content") == prefill)


class RecipeTest(unittest.TestCase):
    def test_snapshots_is_off_by_default_and_a_recipe_turns_it_on(self):
        self.assertIs(recipe.parse(BASE)["snapshots"], False)
        self.assertIs(recipe.parse(ON)["snapshots"], True)

    def test_snapshots_must_be_true_or_false(self):
        for bad in (1, "yes", None, []):
            with self.assertRaisesRegex(RecipeError, "snapshots must be true or false"):
                recipe.parse(with_(BASE, snapshots=bad))


class SplitTest(unittest.TestCase):
    def test_the_head_is_its_own_message_with_a_boundary_and_no_text_is_lost(self):
        head = task.show(("def priority(item, bins):\n    return -(bins - item)\n", 4.08))
        prompt = task.think_prompt(head, "Make the fit tighter.")
        got = task.split(head, prompt)
        self.assertEqual(got[0], {"role": "user", "content": head, "snap": True})
        self.assertEqual(got[1]["role"], "user")
        self.assertNotIn("snap", got[1])
        self.assertEqual(head + "\n" + got[1]["content"], prompt)

    def test_a_prompt_that_does_not_start_with_the_head_is_an_error(self):
        with self.assertRaises(ValueError):
            task.split("the head", "another text")


class ClientTest(unittest.TestCase):
    def test_a_build_sends_the_messages_with_a_boundary_at_the_last_one(self):
        stub = StubTransport()
        usage = client.Client(stub).build("lfm3", [{"role": "user", "content": "H", "snap": True},
                                                   {"role": "user", "content": "A"}], system="S")
        self.assertEqual(stub.calls, [(client.CONTEXTS, {"model": "lfm3", "context": {"thinking": False, "messages": [
            {"role": "system", "content": "S"}, {"role": "user", "content": "H", "snap": True},
            {"role": "user", "content": "A", "snap": True}]}})])
        self.assertEqual(usage, {"prompt": 60, "fed": 60, "completion": 0})


class DriverTest(unittest.TestCase):
    def test_a_recipe_with_snapshots_off_builds_nothing_and_sends_one_text(self):
        stub, _, _ = run(BASE)
        self.assertEqual(builds(stub), [])
        for b in gens(stub, task.IDEA_PREFILL):
            self.assertIsInstance(b["context"]["messages"][0]["content"], str)
            self.assertEqual(len(b["context"]["messages"]), 2)

    def test_the_think_agents_share_a_head_that_the_driver_builds_first_for_each_model(self):
        stub, lines, _ = run(ON)
        ideas = gens(stub, task.IDEA_PREFILL)
        self.assertEqual(len(ideas), 4)
        heads = {json.dumps(b["context"]["messages"][0]) for b in ideas}
        self.assertEqual(len(heads), 1)
        head = ideas[0]["context"]["messages"][0]
        self.assertEqual((head["role"], head["snap"]), ("user", True))
        self.assertTrue(head["content"].startswith(task.PROBLEM))
        for b in ideas:
            focus = b["context"]["messages"][1]
            self.assertTrue(focus["content"].startswith("Focus: "), focus)
            self.assertNotIn("snap", focus)
            self.assertEqual(len(b["context"]["messages"]), 3)
        # One build of the head for each model of the agents, before the first agent.
        before = stub.calls[:first(stub, client.GENERATIONS, task.IDEA_PREFILL)]
        self.assertEqual(sorted(b["model"] for p, b in before if p == client.CONTEXTS), ["lfm3", "lfm3-1"])
        for p, b in before:
            self.assertEqual(b["context"], {"thinking": False, "messages": [head]})
        # The record of the stage counts the builds: 4 agents and 2 builds.
        think = lines[-2]["round_stats"]["stages"]["think"]
        self.assertEqual((think["calls"], think["prompt"], think["fed"]), (6, 4 * 100 + 2 * 60, 4 * 10 + 2 * 60))

    def test_the_act_forks_share_their_whole_prompt_that_the_driver_builds_first(self):
        stub, _, _ = run(ON)
        forks = gens(stub, task.CODE_PREFILL)
        self.assertEqual(len(forks), 4)
        shared = forks[0]["context"]["messages"][:-1]
        self.assertEqual([m.get("snap") for m in shared], [True, True])
        self.assertTrue(shared[1]["content"].startswith("Apply this idea: "))
        for b in forks:
            self.assertEqual(b["context"]["messages"][:-1], shared)
        at = first(stub, client.GENERATIONS, task.CODE_PREFILL)
        act_builds = [b for p, b in stub.calls[:at] if p == client.CONTEXTS and len(b["context"]["messages"]) == 2]
        self.assertEqual(sorted(b["model"] for b in act_builds), sorted({b["model"] for b in forks}))
        for b in act_builds:
            self.assertEqual(b["context"]["messages"], shared)

    def test_the_vote_shares_the_head_of_the_round(self):
        stub, _, _ = run(ON)
        vote = gens(stub, task.VOTE_PREFILL)[0]["context"]["messages"]
        self.assertEqual(vote[0], gens(stub, task.IDEA_PREFILL)[0]["context"]["messages"][0])
        self.assertTrue(vote[1]["content"].startswith("Ideas from 4 agents:"))

    def test_after_a_prime_the_agents_share_the_head_the_ask_and_the_prime(self):
        stub, _, _ = run(with_(ON, prime={"max_tokens": 64}))
        ideas = gens(stub, task.IDEA_PREFILL)
        shared = ideas[0]["context"]["messages"][:3]
        self.assertEqual([(m["role"], m.get("snap")) for m in shared],
                         [("user", True), ("user", None), ("assistant", True)])
        self.assertEqual(shared[1]["content"], task.PRIME_ASK)
        for b in ideas:
            self.assertEqual(b["context"]["messages"][:3], shared)
            self.assertTrue(b["context"]["messages"][3]["content"].startswith("Focus: "))
        before = stub.calls[:first(stub, client.GENERATIONS, task.IDEA_PREFILL)]
        long = [b for p, b in before if p == client.CONTEXTS and len(b["context"]["messages"]) == 3]
        self.assertEqual(sorted(b["model"] for b in long), ["lfm3", "lfm3-1"])
        for b in long:
            self.assertEqual(b["context"]["messages"], shared)
        # The prime itself sends the head as its own message, so its model holds the head after it.
        prime = [b for p, b in stub.calls if p == client.GENERATIONS and b["context"]["thinking"]][0]
        self.assertEqual(prime["context"]["messages"], shared[:2])

    def test_both_contexts_of_a_contrast_fork_start_with_the_head(self):
        stub, _, _ = run(with_(ON, act={"forks": 2, "by": "contrast"}))
        head = gens(stub, task.IDEA_PREFILL)[0]["context"]["messages"][0]
        calls = [b for p, b in stub.calls if p == client.ENSEMBLE]
        self.assertEqual(len(calls), 2)
        for b in calls:
            with_plan, plain = (c["messages"] for c in b["contexts"])
            self.assertEqual((with_plan[0], plain[0]), (head, head))
            self.assertTrue(with_plan[1]["content"].startswith("Apply this idea: "))
            self.assertTrue(plain[1]["content"].startswith(task.IMPROVE))

    def test_each_context_of_a_joined_distill_starts_with_the_head(self):
        stub, _, _ = run(with_(ON, select={"by": "all"}, distill={"by": "joined"}))
        head = gens(stub, task.IDEA_PREFILL)[0]["context"]["messages"][0]
        joined = [b for p, b in stub.calls if p == client.ENSEMBLE and b["combine"] == "prob_mean"]
        self.assertEqual(len(joined), 1)
        self.assertEqual(len(joined[0]["contexts"]), 4)
        for c in joined[0]["contexts"]:
            self.assertEqual(c["messages"][0], head)
            self.assertTrue(c["messages"][1]["content"].startswith("An agent proposed: "))

    def test_a_stage_system_prompt_goes_first_in_its_build(self):
        stub, _, _ = run(with_(ON, think={"agents": 4, "system": "Be bold."}))
        before = stub.calls[:first(stub, client.GENERATIONS, task.IDEA_PREFILL)]
        for p, b in before:
            self.assertEqual(b["context"]["messages"][0], {"role": "system", "content": "Be bold."})


if __name__ == "__main__":
    unittest.main()
