"""Tests of the pure parts of inkling2_dspark.py (tiny shapes, no weights). Run, from tools/, in a venv that has torch:

  <reference venv>/bin/python -m unittest test_inkling2_dspark

Without torch the tests skip."""
import math
import unittest

try:
    import torch

    import inkling2_dspark as d
except ImportError:
    torch = d = None

needs_torch = unittest.skipIf(torch is None, "the package torch is absent")


@needs_torch
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


@needs_torch
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


@needs_torch
class WindowTest(unittest.TestCase):
    def test_local_reads_1024_before_the_block_full_reads_all(self):
        self.assertEqual(d.key_range(5000, True), (5000 - 1024, 5008))
        self.assertEqual(d.key_range(5000, False), (0, 5008))
        self.assertEqual(d.key_range(10, True), (0, 18))


@needs_torch
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


@needs_torch
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


@needs_torch
class SoftcapRoundingTest(unittest.TestCase):
    def test_one_rounding_at_the_end_as_vllm(self):
        # vLLM: x = bf16(a + b); then x / cap, tanh, * cap in fp32; one bf16 rounding of the result.
        g = torch.Generator().manual_seed(3)
        a = (torch.randn(20000, generator=g) * 20).to(torch.bfloat16)
        b = (torch.randn(20000, generator=g) * 5).to(torch.bfloat16)
        x = (a.float() + b.float()).to(torch.bfloat16).float()
        want = (torch.tanh(x / 30.0) * 30.0).to(torch.bfloat16)
        self.assertTrue(torch.equal(d.softcap(a, b), want))


@needs_torch
class GreedyTest(unittest.TestCase):
    def test_the_vocab_is_the_unpadded_size_of_the_config(self):
        self.assertEqual(d.VOCAB, 200058)

    def test_the_padding_rows_of_the_vocab_never_win(self):
        logits = torch.zeros(10)
        logits[8] = 5.0  # a padding row (>= vocab 6)
        logits[2] = 1.0
        self.assertEqual(d.greedy(logits, vocab=6), 2)

    def test_nan_counts_as_minus_infinity(self):
        logits = torch.tensor([float("nan"), 1.0, 0.5])
        self.assertEqual(d.greedy(logits, vocab=3), 1)


@needs_torch
class AcceptTest(unittest.TestCase):
    def test_longest_equal_prefix(self):
        self.assertEqual(d.accepted([1, 2, 3], [1, 2, 9]), 2)
        self.assertEqual(d.accepted([1, 2, 3], [1, 2, 3, 4]), 3)
        self.assertEqual(d.accepted([5], [1]), 0)
        self.assertEqual(d.accepted([1, 2, 3], [1]), 1)


@needs_torch
class SpeedTest(unittest.TestCase):
    def test_a_block_costs_its_rows_plus_the_head(self):
        import inkling2_dspark_run as run

        pts = [(1, 1.0), (8, 1.8), (16, 2.5)]
        # Two blocks of 7 drafts, 2 and 0 accepted: 4 tokens for 2 * (1.8 + 0.26) steps.
        self.assertAlmostEqual(run.head_speed([2, 0], [7, 7], pts, 0.26), 4 / (2 * 2.06))


if __name__ == "__main__":
    unittest.main()
