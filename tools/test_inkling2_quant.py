"""Tests of tools/inkling2_quant.py and of the quantized expert call of tools/inkling2_reference.py.

Run them in the reference venv, from tools/ (they need torch, safetensors and transformers, and no model):

  <reference venv>/bin/python -m unittest test_inkling2_quant

The fixture `crates/nagaya-mk-inkling2-sm121/tests/fixtures/quant_kernels.safetensors` holds the bytes that the
Triton kernels of vLLM give for 154 rows (53 + 53 NVFP4 and 48 MXFP8). Each row must give the same bytes here.
"""

import itertools
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

import inkling2_quant as quant
import inkling2_reference as reference

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "crates/nagaya-mk-inkling2-sm121/tests/fixtures/quant_kernels.safetensors"


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tensors = load_file(str(FIXTURE))

    def nvfp4(self, name, rows):
        x = self.tensors[f"{name}.x"].to(torch.float32)
        self.assertEqual(x.shape[0], rows)
        codes, scales, amax = quant.quantize_nvfp4(x)
        want = self.tensors[f"{name}.values"]
        self.assertTrue(torch.equal(quant.pack_nibbles(codes), want), "the packed codes")
        self.assertTrue(torch.equal(scales, self.tensors[f"{name}.scales"]), "the block scale bytes")
        self.assertTrue(torch.equal(amax, self.tensors[f"{name}.amax"]), "the row amax")

    def test_nvfp4_with_k_2304_gives_the_bytes_of_the_kernel(self):
        self.nvfp4("nvfp4_k2304", 53)

    def test_nvfp4_with_k_1152_gives_the_bytes_of_the_kernel(self):
        self.nvfp4("nvfp4_k1152", 53)

    def test_mxfp8_gives_the_bytes_of_the_kernel(self):
        x = self.tensors["mxfp8_k2304.x"].to(torch.float32)
        self.assertEqual(x.shape[0], 48)
        codes, scales = quant.quantize_mxfp8(x)
        self.assertTrue(torch.equal(codes, self.tensors["mxfp8_k2304.values"]), "the e4m3 codes")
        self.assertTrue(torch.equal(scales, self.tensors["mxfp8_k2304.scales"]), "the ue8m0 scale bytes")

    def test_the_fake_quantizer_expands_the_fixture_codes(self):
        # The codes of the kernel, expanded by hand in f64, equal the f32 expansion: every product is exact or
        # rounds once.
        x = self.tensors["nvfp4_k1152.x"].to(torch.float32)
        got = quant.fake_quantize(x, "nvfp4")
        packed = self.tensors["nvfp4_k1152.values"].to(torch.int64)
        codes = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(x.shape)
        scale = self.tensors["nvfp4_k1152.scales"].view(torch.float8_e4m3fn).to(torch.float64)
        values = quant.E2M1.to(torch.float64)[codes].reshape(x.shape[0], -1, 16) * scale[..., None]
        outer = self.tensors["nvfp4_k1152.amax"].to(torch.float64) * (1 / 1536)
        want = (values * outer[:, None, None]).reshape(x.shape)
        self.assertTrue(torch.allclose(got.to(torch.float64), want, rtol=1e-6, atol=0))


class Cast(unittest.TestCase):
    def test_the_e4m3_cast_equals_the_exhaustive_search_of_quant_rs(self):
        grid = torch.arange(0, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
        grid = grid.to(torch.float64)
        mids = (grid[:-1] + grid[1:]) / 2
        generator = torch.Generator().manual_seed(7)
        values = torch.cat([mids, mids.nextafter(torch.tensor(0.0, dtype=torch.float64)), torch.rand(20000, generator=generator) * 500])
        values = values.to(torch.float32)
        got = quant.e4m3_bytes(values)
        for value, byte in zip(values.tolist(), got.tolist()):
            self.assertEqual(byte, search_e4m3(value), value)

    def test_the_ue8m0_byte_rounds_a_nonzero_mantissa_up(self):
        # The fixture has no raw scale one step above a power of two, so this test covers the mask of the carry.
        base = torch.tensor(448.0 * 2.0**-3, dtype=torch.float32)
        amax = torch.stack([base.view(torch.int32).add(delta).view(torch.float32) for delta in range(-3, 4)])
        row = torch.zeros(len(amax), 32)
        row[:, 0] = amax
        _codes, scales = quant.quantize_mxfp8(row)
        raw = (amax * quant.RECIP_448).view(torch.int32)
        for got, bits in zip(scales[:, 0].tolist(), raw.tolist()):
            self.assertEqual(got, (bits >> 23) + (1 if bits & 0x7FFFFF else 0), hex(bits))
        self.assertTrue(any(bits & 0x7FFFFF == 1 for bits in raw.tolist()), "one case has the mantissa 1")

    def test_the_e2m1_ties_go_to_the_even_code(self):
        ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -5.0, 100.0, float("inf"), -0.1])
        self.assertEqual(quant.e2m1_codes(ties).tolist(), [0, 2, 2, 4, 4, 6, 6, 14, 7, 7, 8])

    def test_a_value_that_is_not_finite_is_an_error(self):
        for bad in (float("nan"), float("inf")):
            row = torch.zeros(1, 32)
            row[0, 3] = bad
            with self.assertRaises(ValueError):
                quant.quantize_nvfp4(row)
            with self.assertRaises(ValueError):
                quant.quantize_mxfp8(row)

    def test_a_zero_row_gives_a_zero_row(self):
        zero = torch.zeros(2, 64)
        self.assertTrue(torch.equal(quant.fake_quantize(zero, "nvfp4"), zero))
        self.assertTrue(torch.equal(quant.fake_quantize(zero, "mxfp8"), zero))


def search_e4m3(value):
    """The search of `encode_e4m3` in quant.rs: the nearest of the 126 finite codes, ties to the even code."""
    if value >= 448.0:
        return 0x7E
    best_code, best_error = 0, value
    for code in range(1, 0x7F):
        represented = torch.tensor([code], dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float64).item()
        error = abs(value - represented)
        if error < best_error or (error == best_error and code % 2 == 0 and best_code % 2 == 1):
            best_error, best_code = error, code
    return best_code


class Formats(unittest.TestCase):
    def test_the_split_of_a_config_is_read_from_the_file(self):
        import json
        import tempfile

        def group(rule, layers):
            names = [f"model.llm.layers.{i}.mlp.experts.{w}_weight" for i in layers for w in ("w13", "w2")]
            return {"required_activation_quantization": rule, "tensors": names}

        config = {"model": {"parameter_groups": [group(quant.NVFP4_RULE, range(0, 2)), group(quant.MXFP8_RULE, range(2, 3)), group(None, [])]}}
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "hf_quant_config.json").write_text(json.dumps(config))
            self.assertEqual(quant.layer_formats(folder, 3), ["nvfp4", "nvfp4", "mxfp8"])
            with self.assertRaises(ValueError):
                quant.layer_formats(folder, 4)
            config["model"]["parameter_groups"][1]["required_activation_quantization"] = dict(quant.MXFP8_RULE, block_size=64)
            (Path(folder) / "hf_quant_config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                quant.layer_formats(folder, 3)

    def test_the_checkpoint_of_mini_splits_at_layer_20_when_it_is_present(self):
        import os

        folder = os.environ.get("INKLING2_MODEL_DIR")
        if not folder or not (Path(folder) / "hf_quant_config.json").exists():
            self.skipTest("INKLING2_MODEL_DIR is not set")
        self.assertEqual(quant.layer_formats(folder, 24), ["nvfp4"] * 20 + ["mxfp8"] * 4)


def tiny_experts(hidden=64, inter=32, count=4, top=2):
    """A tiny `Inkling2LoopExperts` with random weights, and a routing of random tokens."""
    from transformers.models.inkling2_loop import configuration_inkling2_loop as cfg
    from transformers.models.inkling2_loop import modeling_inkling2_loop as modeling

    config = cfg.Inkling2LoopTextConfig(
        hidden_size=hidden, moe_intermediate_size=inter, n_routed_experts=count, num_experts_per_tok=top
    )
    experts = modeling.Inkling2LoopExperts(config)
    generator = torch.Generator().manual_seed(3)
    experts.gate_up_proj.data = torch.randn(count, 2 * inter, hidden, generator=generator) * 0.2
    experts.down_proj.data = torch.randn(count, hidden, inter, generator=generator) * 0.2
    return experts, generator


class ExpertCall(unittest.TestCase):
    def setUp(self):
        self.experts, generator = tiny_experts()
        self.tokens = 5
        self.hidden = torch.randn(self.tokens, 64, generator=generator)
        self.index = torch.stack([torch.randperm(4, generator=generator)[:2] for _ in range(self.tokens)])
        self.weights = torch.rand(self.tokens, 2, generator=generator) + 0.1

    def call(self):
        return reference.ExpertCalls.forward(self.experts, self.hidden, self.index, self.weights)

    def test_with_no_format_the_call_is_the_f32_call(self):
        self.experts.activation_format = None
        want = reference.ExpertCalls.forward_f32(self.experts, self.hidden, self.index, self.weights)
        self.assertTrue(torch.equal(self.call(), want))

    def test_a_quantized_call_equals_a_hand_loop_over_the_rows(self):
        for fmt in ("nvfp4", "mxfp8"):
            self.experts.activation_format, self.experts.layer_index = fmt, 0
            got = self.call()
            want = torch.zeros(self.tokens, 64)
            for token, slot in itertools.product(range(self.tokens), range(2)):
                expert = int(self.index[token, slot])
                row = quant.fake_quantize(self.hidden[token : token + 1], fmt)
                gate_up = torch.nn.functional.linear(row, self.experts.gate_up_proj[expert])
                gate, up = gate_up.chunk(2, dim=-1)
                act = self.experts.act_fn(gate, up) * self.weights[token, slot]
                act = quant.fake_quantize(act, fmt)
                want[token] += torch.nn.functional.linear(act, self.experts.down_proj[expert])[0]
            self.assertTrue(torch.allclose(got, want, rtol=1e-5, atol=1e-6), fmt)
            self.assertFalse(torch.allclose(got, self.f32(), rtol=1e-4, atol=1e-6), f"{fmt} changes the result")

    def f32(self):
        return reference.ExpertCalls.forward_f32(self.experts, self.hidden, self.index, self.weights)

    def test_the_tally_counts_the_two_quantizations_of_each_call(self):
        self.experts.activation_format, self.experts.layer_index = "nvfp4", 7
        before = reference.ExpertCalls.snapshot()
        self.call()
        self.call()
        after = reference.ExpertCalls.snapshot()
        self.assertEqual(after["calls"][7] - before["calls"].get(7, 0), 2)
        self.assertEqual(after["inputs"][7] - before["inputs"].get(7, 0), 2)
        self.assertGreaterEqual(after["downs"][7] - before["downs"].get(7, 0), 2)

    def test_the_self_check_stops_a_layer_that_skipped_the_quantizer(self):
        reference.ExpertCalls.formats = ["nvfp4"]
        before = reference.ExpertCalls.snapshot()
        self.experts.activation_format, self.experts.layer_index = None, 0
        self.call()
        with self.assertRaises(AssertionError):
            reference.ExpertCalls.verify(before)
        reference.ExpertCalls.formats = None


if __name__ == "__main__":
    unittest.main()
