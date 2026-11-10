"""Tests of the bodies of the calls, over the stub transport."""
import unittest

import numpy as np

from swarm import client, task
from swarm.tests.stub import StubTransport


class ClientTest(unittest.TestCase):
    def setUp(self):
        self.stub = StubTransport()
        self.c = client.Client(self.stub)

    def test_a_generation_with_a_prefill(self):
        text, usage = self.c.generate("lfm3", "U", 48, 1.0, 7, task.IDEA_PREFILL)
        path, body = self.stub.calls[0]
        self.assertEqual(path, "/council/v2/generations")
        self.assertEqual(body, {"model": "lfm3", "max_tokens": 48, "sample": {"temperature": 1.0, "seed": 7},
                                "context": {"thinking": False, "messages": [
                                    {"role": "user", "content": "U"},
                                    {"role": "assistant", "content": "Idea:", "continue": True}]}})
        self.assertTrue(text.startswith("Idea:"))
        self.assertEqual(usage, {"prompt": 100, "fed": 10, "completion": 5})

    def test_latent_steps_go_to_ensemble_with_one_context(self):
        self.c.generate("lfm3", "U", 64, 0.7, 3, task.PLAN_PREFILL, latent=16)
        path, body = self.stub.calls[0]
        self.assertEqual(path, "/nagaya/v1/lab/ensemble")
        self.assertEqual(len(body["contexts"]), 1)
        self.assertEqual(body["combine"], "logprob_sum")
        self.assertEqual(body["latent"], {"steps": 16})

    def test_no_latent_member_at_zero(self):
        self.c.ensemble("lfm3", ["A", "B"], "contrast", 96, 1.0, 5, task.CODE_PREFILL, strength=0.5, latent=0)
        path, body = self.stub.calls[0]
        self.assertNotIn("latent", body)
        self.assertEqual(body["strength"], 0.5)
        self.assertEqual([c["messages"][0]["content"] for c in body["contexts"]], ["A", "B"])

    def test_no_strength_member_without_a_strength(self):
        self.c.ensemble("lfm3", ["A"], "prob_mean", 96, 1.0, 5, task.CODE_PREFILL)
        self.assertNotIn("strength", self.stub.calls[0][1])

    def test_embeddings_keep_the_order_of_the_texts(self):
        v, _ = self.c.embed("embeddinggemma-2", ["a", "b", "c"], 256)
        self.assertEqual(v.shape, (3, 4))
        self.assertEqual([int(np.argmax(r)) for r in v], [0, 1, 2])
        self.assertEqual(self.stub.calls[0][1]["dimensions"], 256)


class FarthestTest(unittest.TestCase):
    def test_distinct_and_far(self):
        v = np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.98, 0.12]])
        kept = client.farthest(v, 2)
        self.assertEqual(sorted(kept), [0, 2])

    def test_no_index_twice_with_equal_vectors(self):
        v = np.ones((4, 3))
        self.assertEqual(sorted(client.farthest(v, 4)), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
