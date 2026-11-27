"""Tests of the pure parts of inkling2_dspark.py (tiny shapes, no weights). Run: python3 -m unittest test_inkling2_dspark
(in tools/, with a venv that has torch: ~/venvs/nagaya-inkling2-ref)."""
import math
import unittest

import torch

import inkling2_dspark as d


class NormTest(unittest.TestCase):
    def test_rms_uses_a_plain_weight(self):
        x = torch.tensor([[3.0, 4.0]])
        out = d.rms(x, torch.tensor([1.0, 2.0]), eps=0.0).float()
        s = math.sqrt((9 + 16) / 2)
        self.assertTrue(torch.allclose(out, torch.tensor([[3 / s, 8 / s]]), atol=1e-2))

    def test_mix_weights_the_taps_then_normalizes(self):
        feats = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # [T=1, taps=2, H=2]
        out = d.mix(feats, torch.tensor([0.75, 0.25]), torch.ones(2), eps=0.0).float()
        s = math.sqrt((0.75 ** 2 + 0.25 ** 2) / 2)
        self.assertTrue(torch.allclose(out, torch.tensor([[0.75 / s, 0.25 / s]]), atol=1e-2))


class RopeTest(unittest.TestCase):
    def test_position_zero_is_the_identity_and_norm_is_kept(self):
        x = torch.randn(2, 3, 8).to(torch.bfloat16)
        self.assertTrue(torch.equal(d.rope(x, torch.zeros(2, dtype=torch.long)), x))
        y = d.rope(x, torch.tensor([5, 900]))
        self.assertTrue(torch.allclose(y.float().norm(dim=-1), x.float().norm(dim=-1), rtol=2e-2))

    def test_the_halves_rotate_not_the_pairs(self):
        # x = [1, 0 | 0, 0] at a position with angle a in the first frequency: out = [cos a, 0 | sin a, 0].
        x = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        y = d.rope(x, torch.tensor([1]), theta=1.0).float()  # inv freq 1 for index 0, so a = 1 rad
        self.assertTrue(torch.allclose(y[0, 0], torch.tensor([math.cos(1), 0.0, math.sin(1), 0.0]), atol=1e-2))


class WindowTest(unittest.TestCase):
    def test_local_reads_1024_before_the_block_full_reads_all(self):
        self.assertEqual(d.key_range(5000, True), (5000 - 1024, 5008))
        self.assertEqual(d.key_range(5000, False), (0, 5008))
        self.assertEqual(d.key_range(10, True), (0, 18))


class AttentionTest(unittest.TestCase):
    def test_a_query_head_reads_its_group_kv_head_and_the_block_is_not_masked(self):
        k = torch.zeros(4, 2, 4)  # equal scores: a plain mean over all 4 keys, even the last (future) ones
        v = torch.zeros(4, 2, 4)
        v[:, 0], v[:, 1] = 1.0, 3.0
        v[3, 0] = 5.0  # the last key counts: mean of (1, 1, 1, 5) = 2
        q = torch.zeros(1, 4, 4)  # 4 query heads over 2 kv heads: heads 0, 1 -> kv 0; heads 2, 3 -> kv 1
        out = d.attention(q, k, v).float().reshape(4, 4)
        self.assertTrue(torch.allclose(out[0], torch.full((4,), 2.0)))
        self.assertTrue(torch.allclose(out[1], torch.full((4,), 2.0)))
        self.assertTrue(torch.allclose(out[2], torch.full((4,), 3.0)))
        v[:, 1] = 7.0
        out = d.attention(q, k, v).float().reshape(4, 4)
        self.assertTrue(torch.allclose(out[3], torch.full((4,), 7.0)))
        self.assertTrue(torch.allclose(out[0], torch.full((4,), 2.0)))


    def test_the_softmax_runs_over_the_keys(self):
        q = torch.zeros(1, 1, 4)
        q[0, 0, 0] = 40.0
        k = torch.zeros(2, 1, 4)
        k[1, 0, 0] = 1.0  # a score of 40 * 1 * (1 / sqrt(128)) for key 1 against 0 for key 0
        v = torch.ones(2, 1, 4)
        v[0] = -1.0
        out = d.attention(q, k, v, scale=1.0).float()
        self.assertTrue(torch.allclose(out, torch.ones(1, 4), atol=1e-3))


class MarkovTest(unittest.TestCase):
    def test_zero_input_halves_the_state(self):
        raw = torch.zeros(12)
        state, latent = d.markov_gate(raw, torch.ones(4, dtype=torch.bfloat16))
        self.assertTrue(torch.equal(state.float(), torch.full((4,), 0.5)))  # sigmoid(0) * 1 + (1 - .5) * tanh(0)
        self.assertTrue(torch.equal(latent.float(), torch.zeros(4)))

    def test_the_gate_keeps_or_replaces(self):
        raw = torch.cat([torch.full((2,), 20.0), torch.full((2,), 1.0), torch.zeros(2)])
        state, _ = d.markov_gate(raw, torch.tensor([0.5, -0.5], dtype=torch.bfloat16))
        self.assertTrue(torch.allclose(state.float(), torch.tensor([0.5, -0.5]), atol=1e-2))  # gate 1: keep
        raw = torch.cat([torch.full((2,), -20.0), torch.full((2,), 1.0), torch.zeros(2)])
        state, _ = d.markov_gate(raw, torch.tensor([0.5, -0.5], dtype=torch.bfloat16))
        self.assertTrue(torch.allclose(state.float(), torch.full((2,), math.tanh(1.0)), atol=1e-2))  # gate 0: tanh(cand)

    def test_softcap_bounds_the_logits(self):
        out = d.softcap(torch.tensor([0.0, 1000.0, -1000.0]), torch.zeros(3)).float()
        self.assertTrue(torch.allclose(out, torch.tensor([0.0, 30.0, -30.0])))
        mid = d.softcap(torch.tensor([15.0]), torch.tensor([15.0])).float()  # 30 * tanh(30 / 30)
        self.assertTrue(torch.allclose(mid, torch.tensor([30 * math.tanh(1.0)]), atol=0.2))


class AcceptTest(unittest.TestCase):
    def test_longest_equal_prefix(self):
        self.assertEqual(d.accepted([1, 2, 3], [1, 2, 9]), 2)
        self.assertEqual(d.accepted([1, 2, 3], [1, 2, 3, 4]), 3)
        self.assertEqual(d.accepted([5], [1]), 0)
        self.assertEqual(d.accepted([1, 2, 3], [1]), 1)


if __name__ == "__main__":
    unittest.main()
