"""Tests of the member `snapshots` of a recipe: the shared part of the prompts of a stage is its own messages with a
boundary. The driver makes no call only for a snapshot: the nagaya feeds a prefix once for the requests in flight that
share it (spec 11.14)."""
import json
import unittest

from swarm import client, recipe, task
from swarm.recipe import RecipeError
from swarm.tests.stub import StubTransport
from swarm.tests.test_driver import BASE, run, with_

ON = with_(BASE, snapshots=True)


def builds(stub):
    """The calls of a build route. The driver has none: a build before a fan-out is a call that the nagaya makes
    needless."""
    return [b for p, b in stub.calls if p == "/council/v2/contexts"]


def gens(stub, prefill):
    return [b for p, b in stub.calls if p == client.GENERATIONS and
            b["context"]["messages"][-1].get("content") == prefill]


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


class DriverTest(unittest.TestCase):
    def test_a_recipe_with_snapshots_off_builds_nothing_and_sends_one_text(self):
        stub, _, _ = run(BASE)
        self.assertEqual(builds(stub), [])
        for b in gens(stub, task.IDEA_PREFILL):
            self.assertIsInstance(b["context"]["messages"][0]["content"], str)
            self.assertEqual(len(b["context"]["messages"]), 2)

    def test_a_recipe_with_snapshots_makes_no_call_only_for_a_snapshot(self):
        for extra in ({}, {"prime": {"max_tokens": 64}}, {"act": {"forks": 2, "by": "contrast"}}):
            stub, lines, _ = run(with_(ON, **extra))
            self.assertEqual(builds(stub), [])
            self.assertEqual(set(stub.paths()) - {client.GENERATIONS, client.ENSEMBLE}, set())
        # The record of a stage counts its calls to the model, and no other call.
        self.assertEqual(lines[-2]["round_stats"]["stages"]["think"]["calls"], 4)

    def test_the_think_agents_share_a_head_with_a_boundary(self):
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
        think = lines[-2]["round_stats"]["stages"]["think"]
        self.assertEqual((think["calls"], think["prompt"], think["fed"]), (4, 4 * 100, 4 * 10))

    def test_the_act_forks_share_their_whole_prompt_with_a_boundary_at_its_end(self):
        stub, _, _ = run(ON)
        forks = gens(stub, task.CODE_PREFILL)
        self.assertEqual(len(forks), 4)
        shared = forks[0]["context"]["messages"][:-1]
        self.assertEqual([m.get("snap") for m in shared], [True, True])
        self.assertTrue(shared[1]["content"].startswith("Apply this idea: "))
        for b in forks:
            self.assertEqual(b["context"]["messages"][:-1], shared)

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

    def test_a_stage_system_prompt_goes_before_the_head(self):
        stub, _, _ = run(with_(ON, think={"agents": 4, "system": "Be bold."}))
        for b in gens(stub, task.IDEA_PREFILL):
            self.assertEqual(b["context"]["messages"][0], {"role": "system", "content": "Be bold."})
            self.assertEqual(b["context"]["messages"][1]["snap"], True)


if __name__ == "__main__":
    unittest.main()
