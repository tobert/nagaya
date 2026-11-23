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


class Nvfp4Weights(unittest.TestCase):
    """The f32 expansion of an NVFP4 expert weight, as engine-abi "The expert epilogue of NVFP4" states it."""

    def crafted(self):
        # One row of 32 values: block 0 holds the code 2 (1.0) with the scale byte 0x38 (1.0). Block 1 holds the
        # code 7 (6.0) with the scale byte 0x40 (2.0).
        codes = torch.tensor([[0x22] * 8 + [0x77] * 8], dtype=torch.uint8)
        scales = torch.tensor([[0x38, 0x40]], dtype=torch.uint8)
        return codes, scales

    def test_the_factor_is_not_rounded_to_bf16(self):
        codes, scales = self.crafted()
        amax = torch.tensor(1.0 + 2.0**-10)  # not a bf16 value: the bf16 loader makes it 1.0
        denominator = torch.tensor(1.0)
        got = quant.expand_nvfp4_weight(codes, scales, amax, denominator)
        self.assertEqual(got.dtype, torch.float32)
        self.assertEqual(got[0, 0].item(), 1.0 + 2.0**-10)
        self.assertEqual(got[0, 16].item(), 12.0 * (1.0 + 2.0**-10))
        self.assertNotEqual(got[0, 0].to(torch.bfloat16).item(), got[0, 0].item(), "the case must separate f32 from bf16")

    def test_the_factor_is_amax_over_the_denominator_in_f32(self):
        codes, scales = self.crafted()
        amax, denominator = torch.tensor(0.1645), torch.tensor(2688.0)
        got = quant.expand_nvfp4_weight(codes, scales, amax, denominator)
        factor = amax / denominator
        self.assertEqual(got[0, 0].item(), (torch.tensor(1.0) * factor).item())
        self.assertEqual(got[0, 31].item(), (torch.tensor(12.0) * factor).item())

    def test_it_equals_the_loader_of_transformers_when_the_factor_is_a_bf16_value(self):
        from transformers.core_model_loading import Inkling2LoopDequantizeExperts

        generator = torch.Generator().manual_seed(5)
        weight = torch.randint(0, 256, (2, 6, 32), generator=generator, dtype=torch.uint8)
        scale = torch.randint(0x20, 0x50, (2, 6, 4), generator=generator, dtype=torch.uint8).view(torch.float8_e4m3fn)
        amax, denominator = torch.tensor([0.5, 1.5]), torch.tensor([4.0])
        op = Inkling2LoopDequantizeExperts()
        names = ["w", "s", "a", "d"]
        got = op.convert(dict(zip(names, (weight, scale, amax, denominator))), names, ["out"])["out"]
        for e in range(2):
            mine = quant.expand_nvfp4_weight(weight[e], scale[e], amax[e], denominator[0])
            self.assertTrue(torch.equal(mine.to(torch.bfloat16), got[e]), f"expert {e}")


def write_checkpoint(folder, layer, tensors):
    import json

    from safetensors.torch import save_file

    save_file(tensors, str(folder / "model.safetensors"))
    index = {"weight_map": {name: "model.safetensors" for name in tensors}}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index))


class Nvfp4Source(unittest.TestCase):
    """The reader that gives the f32 weights of one expert from the checkpoint, with the order of rows of the loader."""

    def test_it_gives_the_gate_up_rows_as_gate_then_up_and_the_down_rows_as_they_are(self):
        import tempfile

        generator = torch.Generator().manual_seed(9)
        experts, rows, cols = 3, 32, 32
        prefix = "model.llm.layers.4.mlp.experts."
        tensors = {
            prefix + "w13_weight": torch.randint(0, 256, (experts, rows, cols // 2), generator=generator, dtype=torch.uint8),
            prefix + "w13_weight_scale": torch.randint(0x20, 0x50, (experts, rows, cols // 16), generator=generator, dtype=torch.uint8).view(torch.float8_e4m3fn),
            prefix + "w13_weight_amax": torch.tensor([0.25, 0.3, 0.7]),
            prefix + "w13_weight_outer_scale_denominator": torch.tensor([3.0]),
            prefix + "w2_weight": torch.randint(0, 256, (experts, cols, rows // 2), generator=generator, dtype=torch.uint8),
            prefix + "w2_weight_scale": torch.randint(0x20, 0x50, (experts, cols, rows // 32), generator=generator, dtype=torch.uint8).view(torch.float8_e4m3fn),
            prefix + "w2_weight_amax": torch.tensor([0.5, 0.6, 0.9]),
            prefix + "w2_weight_outer_scale_denominator": torch.tensor([5.0]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            write_checkpoint(Path(tmp), 4, tensors)
            source = reference.Nvfp4Source(tmp, 4)
            for e in range(experts):
                plain = quant.expand_nvfp4_weight(
                    tensors[prefix + "w13_weight"][e],
                    tensors[prefix + "w13_weight_scale"][e],
                    tensors[prefix + "w13_weight_amax"][e],
                    tensors[prefix + "w13_weight_outer_scale_denominator"][0],
                )
                got = source.weight("gate_up_proj", e)
                self.assertTrue(torch.equal(got[: rows // 2], plain[0::2]), f"expert {e}: the gate rows are the even rows")
                self.assertTrue(torch.equal(got[rows // 2 :], plain[1::2]), f"expert {e}: the up rows are the odd rows")
                down = quant.expand_nvfp4_weight(
                    tensors[prefix + "w2_weight"][e],
                    tensors[prefix + "w2_weight_scale"][e],
                    tensors[prefix + "w2_weight_amax"][e],
                    tensors[prefix + "w2_weight_outer_scale_denominator"][0],
                )
                self.assertTrue(torch.equal(source.weight("down_proj", e), down), f"expert {e}: down")


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

    def test_an_nvfp4_layer_takes_its_weights_from_its_source_and_not_from_its_parameters(self):
        class Source:
            def __init__(self, experts):
                self.gate_up, self.down = experts.gate_up_proj.detach() * 1.25, experts.down_proj.detach() * 0.5

            def weight(self, name, expert):
                return (self.gate_up if name == "gate_up_proj" else self.down)[expert].clone()

        self.experts.activation_format, self.experts.layer_index = "nvfp4", 0
        plain = self.call()
        source = Source(self.experts)
        self.experts.nvfp4_source = source
        self.experts.gate_up_proj.data = torch.empty(0)
        self.experts.down_proj.data = torch.empty(0)
        got = self.call()
        self.assertFalse(torch.allclose(got, plain, rtol=1e-3, atol=1e-6), "the call must use the source")
        want = torch.zeros(self.tokens, 64)
        for token, slot in itertools.product(range(self.tokens), range(2)):
            expert = int(self.index[token, slot])
            row = quant.fake_quantize(self.hidden[token : token + 1], "nvfp4")
            gate, up = torch.nn.functional.linear(row, source.gate_up[expert]).chunk(2, dim=-1)
            act = quant.fake_quantize(self.experts.act_fn(gate, up) * self.weights[token, slot], "nvfp4")
            want[token] += torch.nn.functional.linear(act, source.down[expert])[0]
        self.assertTrue(torch.allclose(got, want, rtol=1e-5, atol=1e-6))

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


def tiny_model(seed, quantized, near_tie=None):
    """A tiny Inkling 2 text model: 6 layers, 3 cycles of the layers 2 and 3, 8 experts, hidden size 64.

    With `quantized`, the layers 0 to 3 use NVFP4 and the layers 4 and 5 use MXFP8, as the layers of Mini do in two
    groups. The expert weights are large on purpose: a flipped code then moves a logit by a visible amount.
    """
    from transformers.models.inkling2_loop import configuration_inkling2_loop as cfg
    from transformers.models.inkling2_loop import modeling_inkling2_loop as modeling

    torch.manual_seed(seed)
    config = cfg.Inkling2LoopTextConfig(
        vocab_size=96,
        unpadded_vocab_size=90,
        eos_token_id=None,
        hidden_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=16,
        sliding_window_size=8,
        d_rel=4,
        rel_extent=16,
        n_layers_per_attn_res_block=2,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        loop_layer_index_start=2,
        loop_layer_index_end=4,
        loop_num_passes=3,
        loop_gate_rank=8,
        layer_types=["hybrid_sliding"] * 5 + ["hybrid"],
    )
    model = modeling.Inkling2LoopForCausalLM(config).eval()
    for name, parameter in model.named_parameters():
        if parameter.dim() > 1:
            parameter.data = torch.randn_like(parameter) * (0.5 if ".experts." in name else 0.3)
    if near_tie:
        with torch.no_grad():
            for layer in model.model.layers:
                router = layer.mlp.gate.weight
                for first in range(0, 8, 2):
                    router[first + 1] = router[first] + near_tie * torch.randn_like(router[first])
    modeling.Inkling2LoopExperts.forward = reference.ExpertCalls.forward
    formats = ["nvfp4"] * 4 + ["mxfp8"] * 2
    for index, layer in enumerate(model.model.layers):
        layer.mlp.experts.layer_index = index
        layer.mlp.experts.activation_format = formats[index] if quantized else None
    reference.ExpertCalls.formats = formats if quantized else None
    return model, config


class DecodeWithCache(unittest.TestCase):
    """The cache run and the run with no cache of the quantized model.

    A code of the quantizer flips when two f32 sums that differ in their last bit fall on two sides of an edge. The
    cache run (one row in each matrix product) and the run with no cache (all the rows) sum in another order. A flip in
    one block of one layer moved a logit of the tiny model by 0.3 to 8 in 6 of 60 seeds, so the check of the decode must
    not depend on it.
    """

    def tearDown(self):
        reference.ExpertCalls.formats = None
        reference.ExpertCalls.tape = None

    def free_gap(self, seed, near_tie=None):
        """The gap of the free comparison: the run with no cache quantizes and routes by itself."""
        model, config = tiny_model(seed, True, near_tie)
        generator = torch.Generator().manual_seed(100 + seed)
        ids = torch.randint(0, 90, (12,), generator=generator).tolist()
        with torch.no_grad():
            out = model.generate(
                input_ids=torch.tensor([ids]), max_new_tokens=6, do_sample=False, output_logits=True,
                return_dict_in_generate=True,
            )
        cached = torch.stack([row[0].float() for row in out.logits])
        whole = reference.forward(model, ids + out.sequences[0, 12:-1].tolist())[11:]
        return (cached - whole).abs().max().item()

    def compare(self, seed, quantized, near_tie=None):
        model, config = tiny_model(seed, quantized, near_tie)
        generator = torch.Generator().manual_seed(100 + seed)
        ids = torch.randint(0, 90, (12,), generator=generator).tolist()
        tokens, cached, whole, flips = reference.decode_compare(model, ids, 6, config.total_num_hidden_layers)
        self.assertEqual(len(tokens), 6)
        return (cached - whole).abs().max().item(), flips

    def test_the_f32_model_gives_the_same_logits_with_the_cache(self):
        gap, flips = self.compare(0, False)
        self.assertLess(gap, 1e-3)
        self.assertEqual(flips, 0)

    def test_the_quantized_model_gives_the_same_logits_with_the_cache_when_a_code_flips(self):
        # In the seeds 1 and 7 the run with no cache flips a code, and a free comparison gives a large gap.
        for seed in (1, 7):
            self.assertGreater(self.free_gap(seed), 0.1, f"seed {seed}: the test must meet a flip")
            gap, _flips = self.compare(seed, True)
            self.assertLess(gap, 1e-3, f"seed {seed}")

    def test_the_quantized_model_gives_the_same_logits_with_the_cache_when_routes_tie(self):
        # A router with pairs of experts that tie to 1e-7 gives the same experts in another order of slots, or another
        # expert, when the sums change. The research model of Inkling 2 with a variable loop did this: 2247 rows
        # differed and the gap was 33.5. The tape must replay the route of each token, not only the quantized rows.
        for seed in (0, 1, 2):
            gap, flips = self.compare(seed, True, near_tie=1e-7)
            self.assertGreater(flips, 0, f"seed {seed}: the replay must change the route of a row")
            self.assertLess(gap, 1e-3, f"seed {seed}")

    def test_the_quantized_model_gives_the_same_logits_with_the_cache_when_no_code_flips(self):
        self.assertLess(self.free_gap(0), 1e-3, "seed 0 has no flip")
        gap, flips = self.compare(0, True)
        self.assertEqual(flips, 0)
        self.assertLess(gap, 1e-3)


class TapeOfOps(unittest.TestCase):
    """The tape of the ops (`OpTape`): the rows that a device gate reads, and the replay that checks them."""

    ids = list(range(12))

    def tearDown(self):
        reference.ExpertCalls.formats = None
        reference.ExpertCalls.record = None
        reference.ExpertCalls.tape = None

    def record(self, seed=0, tie=False):
        model, _config = tiny_model(seed, True)
        text = model.model
        if tie:
            with torch.no_grad():
                for layer in text.layers:
                    layer.mlp.gate.weight[1:] = layer.mlp.gate.weight[1]
        ops = reference.OpTape(text)
        reference.forward(model, self.ids)
        ops.remove()
        order = [call["layer"] for call in ops.calls]
        tensors, formats = ops.tensors(order)
        return model, text, ops, order, tensors, formats

    def test_the_tape_has_each_key_of_each_run_with_the_shapes_of_the_docstring(self):
        _model, _text, _ops, order, tensors, formats = self.record()
        self.assertEqual(order, [0, 1, 2, 3, 2, 3, 2, 3, 4, 5])
        count = len(self.ids)
        for run, layer in enumerate(order):
            name = f"run{run:02}"
            nvfp4 = formats[run] == "nvfp4"
            self.assertEqual(tensors[f"{name}.h_e"].shape, (count, 64))
            self.assertEqual(tensors[f"{name}.r"].shape, (count, 64))
            self.assertEqual(tensors[f"{name}.moe"].shape, (count, 64))
            self.assertEqual(tensors[f"{name}.attn"].shape, (count, 4 * 16))
            self.assertEqual(tensors[f"{name}.route_ids"].dtype, torch.int32)
            self.assertEqual(tensors[f"{name}.route_ids"].shape, (count, 2))
            self.assertEqual(tensors[f"{name}.route_weights"].shape, (count, 2))
            self.assertEqual(tensors[f"{name}.down_q"].shape, (count, 2, 32))
            self.assertEqual(tensors[f"{name}.x_codes"].shape, (count, 32 if nvfp4 else 64))
            self.assertEqual(tensors[f"{name}.x_scales"].shape, (count, 4 if nvfp4 else 2))
            self.assertEqual(f"{name}.x_amax" in tensors, nvfp4)
            self.assertEqual(f"{name}.down_amax" in tensors, nvfp4)
            self.assertEqual(tensors[f"{name}.down_codes"].shape, (count, 2, 16 if nvfp4 else 32))

    def test_the_picks_go_in_ascending_expert_order_with_their_own_weights(self):
        _model, _text, ops, _order, tensors, _formats = self.record()
        for run, call in enumerate(ops.calls):
            ids = tensors[f"run{run:02}.route_ids"]
            self.assertTrue(bool((ids[:, 1:] > ids[:, :-1]).all()), f"run {run}: ids ascend")
            self.assertTrue(bool((tensors[f"run{run:02}.route_weights"] > 0).all()))

    def test_the_routing_weight_acts_before_the_down_quantizer(self):
        # The down row of a slot is the quantized SwiGLU row times the weight of that slot: the rows differ from the
        # quantized rows of the f32 mode, which multiplies after the down projection.
        _model, text, _ops, _order, tensors, formats = self.record()
        experts = text.layers[0].mlp.experts
        quant = reference.quant_module()
        ids, weights, x_q = tensors["run00.route_ids"].long(), tensors["run00.route_weights"], tensors["run00.x_q"]
        token, slot = 3, 1
        gate_up = torch.nn.functional.linear(x_q[token : token + 1], experts.gate_up_proj[int(ids[token, slot])])
        want = quant.fake_quantize(experts._apply_gate(gate_up).float() * weights[token, slot], formats[0])
        self.assertTrue(torch.equal(tensors["run00.down_q"][token, slot], want[0]))

    def test_the_raw_parts_expand_to_the_fake_quantized_rows(self):
        _model, _text, _ops, order, tensors, formats = self.record()
        quant = reference.quant_module()
        for run, fmt in enumerate(formats):
            name = f"run{run:02}"
            codes, scales = tensors[f"{name}.x_codes"], tensors[f"{name}.x_scales"]
            if fmt == "nvfp4":
                packed = codes.to(torch.int64)
                unpacked = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(codes.shape[0], -1)
                got = quant.dequantize_nvfp4(unpacked, scales, tensors[f"{name}.x_amax"])
            else:
                got = quant.dequantize_mxfp8(codes, scales)
            self.assertTrue(torch.equal(got, tensors[f"{name}.x_q"]), f"run {run} {fmt}")

    def test_the_buffers_hold_codes_then_scales_then_amax(self):
        _model, _text, _ops, _order, tensors, formats = self.record()
        for run, fmt in enumerate(formats):
            name = f"run{run:02}"
            for part, rows in (("x", 12), ("down", 24)):
                buf = tensors[f"{name}.{part}_buf"]
                codes = tensors[f"{name}.{part}_codes"].reshape(-1)
                scales = tensors[f"{name}.{part}_scales"].reshape(-1)
                self.assertTrue(torch.equal(buf[: codes.numel()], codes))
                self.assertTrue(torch.equal(buf[codes.numel() : codes.numel() + scales.numel()], scales))
                tail = buf[codes.numel() + scales.numel() :]
                if fmt == "nvfp4":
                    self.assertTrue(torch.equal(tail.view(torch.float32), tensors[f"{name}.{part}_amax"].reshape(-1)))
                else:
                    self.assertEqual(tail.numel(), 0)

    def test_each_run_replays_to_its_down_rows_and_its_output_bit_for_bit(self):
        _model, text, _ops, order, tensors, formats = self.record()
        reference.check_replay(text, tensors, order, formats)

    def test_the_replay_stops_on_a_changed_row(self):
        _model, text, _ops, order, tensors, formats = self.record()
        tensors["run04.moe"] = tensors["run04.moe"].clone()
        tensors["run04.moe"][0, 0] += 1e-6
        with self.assertRaises(AssertionError):
            reference.check_replay(text, tensors, order, formats)

    def test_the_attention_output_and_the_moe_output_join_the_run_output(self):
        # Layer 0 has no history: its run output is `r + post_mlp_convolution(moe)`, and `r` is the output of the
        # post attention convolution of `wo(attn)`. The test uses the first run of the tape for both.
        model, text, _ops, _order, tensors, _formats = self.record()
        layer = text.layers[0]
        with torch.no_grad():
            first = layer.post_attention_convolution(layer.self_attn.o_proj.linear(tensors["run00.attn"])[None])[0]
            self.assertTrue(torch.equal(first.float(), tensors["run00.r"]), "r follows wo(attn)")
            out = layer.post_mlp_convolution(tensors["run00.moe"][None])[0].float()
            seen = []
            handle = layer.register_forward_hook(lambda _m, _a, output: seen.append(output[0] if isinstance(output, tuple) else output))
            reference.forward(model, self.ids)
            handle.remove()
        self.assertTrue(torch.equal(tensors["run00.r"] + out, seen[0][0].float()), "run 0 is r + conv(moe)")

    def test_a_tie_at_the_last_place_is_marked(self):
        _model, _text, _ops, _order, tensors, _formats = self.record(tie=True)
        self.assertTrue(bool((tensors["run00.route_tie"] == 1).all()))
        _model, _text, _ops, _order, tensors, _formats = self.record()
        self.assertTrue(bool((tensors["run00.route_tie"] == 0).all()))


if __name__ == "__main__":
    unittest.main()
