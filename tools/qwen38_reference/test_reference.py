# SPDX-License-Identifier: MIT
"""Tests of the Qwen3.8 reference. Run them in the venv of `requirements.txt`, from `tools/`:

    python -m unittest qwen38_reference.test_reference

- The installed transformers is the pinned version.
- The layer-streaming driver (build each layer on meta, load a state dict, run, drop) equals
  `Qwen4ExpTextModel.forward` bit for bit on a small random model that exercises every part of
  the architecture: GDN with 3 V heads per K head, QSA with a budget small enough that the
  indexer really selects, PLE, the MoE, hyper-connections. Same ops in the same order on the
  same CPU, so the tolerance is exact equality.
- `QWEN38_SHARDS` names three shards, or the tool stops with a message.
- With the GGUF: the reader gives the rows of each expert, and a Q8_0 row equals its blocks decoded by hand. The PLE
  hash constants that the HF module derives from its seed equal the ones in the GGUF.

A test that needs the GGUF skips when `QWEN38_SHARDS` is unset or a shard is absent. With `NAGAYA_REQUIRE_MODELS=1`,
it fails instead.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from . import glue, reference, shards_from_env


def shards_or_skip(test: unittest.TestCase) -> list[str]:
    """The shards of `QWEN38_SHARDS`. Absent: skip, or fail under `NAGAYA_REQUIRE_MODELS=1`."""
    paths = [p for p in os.environ.get("QWEN38_SHARDS", "").split(":") if p]
    missing = "QWEN38_SHARDS is unset" if not paths else next(
        (f"{p} is absent" for p in paths if not Path(p).exists()), None)
    if missing is None:
        return paths
    if os.environ.get("NAGAYA_REQUIRE_MODELS") == "1":
        test.fail(f"{missing}, and NAGAYA_REQUIRE_MODELS=1 requires the model")
    test.skipTest(f"{missing}: no Qwen3.8 GGUF")


class Pin(unittest.TestCase):
    def test_transformers_is_the_pinned_version(self):
        import transformers

        self.assertEqual(transformers.__version__, "5.17.0")


class Shards(unittest.TestCase):
    def test_three_paths_are_the_shards_in_order(self):
        with mock.patch.dict(os.environ, {"QWEN38_SHARDS": "/a/1.gguf:/a/2.gguf:/a/3.gguf"}):
            self.assertEqual(shards_from_env(), ["/a/1.gguf", "/a/2.gguf", "/a/3.gguf"])

    def test_an_unset_variable_stops_the_tool_with_a_message(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as stop:
                shards_from_env()
        self.assertIn("QWEN38_SHARDS names 0 paths", str(stop.exception))

    def test_two_paths_stop_the_tool(self):
        with mock.patch.dict(os.environ, {"QWEN38_SHARDS": "/a/1.gguf:/a/2.gguf"}):
            with self.assertRaises(SystemExit):
                shards_from_env()


def tiny_config():
    return reference.text_config(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=16, linear_value_head_dim=16,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=24, shared_expert_intermediate_size=24,
        hc_lowrank=8, vocab_size=300, ple_layer_ids=[2], ple_embed_dim=64, heads_per_ngram=2,
        ngram_vocab_size_base=1000, split_ngram_parts=1, indexer_n_heads=2, indexer_head_dim=16,
        indexer_budget=8, indexer_compress_ratio=4, eos_token_id=7, bos_token_id=7,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        max_position_embeddings=512,
    )


class Driver(unittest.TestCase):
    @torch.no_grad()
    def test_stream_forward_equals_model_forward(self):
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel

        torch.manual_seed(0)
        cfg = tiny_config()
        model = Qwen4ExpTextModel(cfg).eval()
        # random non-zero values everywhere (init leaves norms at 0 and the PLE conv at 0)
        for p in model.parameters():
            p.normal_(0, 0.2)
        for layer in model.layers:
            if layer.layer_type == "linear_attention":
                layer.linear_attn.A_log.uniform_(-1, 1)
        g = torch.Generator().manual_seed(1)
        seqs = [torch.randint(0, cfg.vocab_size, (1, n), generator=g) for n in (40, 23)]
        seqs[0][0, 11] = cfg.eos_token_id  # the PLE hash resets at eos
        want = [model(input_ids=s, use_cache=False).last_hidden_state for s in seqs]

        table = model.layers[1].ple.ple_embedding.ngram_embedding.weight
        fetched = []

        def rows(ids: np.ndarray) -> torch.Tensor:
            fetched.append(len(ids))
            return table[torch.from_numpy(ids)]

        def layer_source(il):
            sd = {k: v.clone() for k, v in model.layers[il].state_dict().items()
                  if "ngram_embedding" not in k and not k.startswith("ple.ple_embedding.")}
            return reference.build_layer(cfg, il, sd, rows if il == 1 else None)

        got = reference.stream_forward(cfg, seqs, model.embed_tokens, layer_source, model.hyper_connection_mixer)
        self.assertTrue(fetched, "the PLE row source was never used")
        # the indexer really selected: fewer than all visible blocks at the last query
        self.assertGreater(seqs[0].shape[1] // cfg.indexer_compress_ratio,
                           cfg.indexer_budget // cfg.indexer_compress_ratio)
        for w, o in zip(want, got):
            self.assertEqual(w.shape, o.shape)
            self.assertTrue(torch.equal(w, o), f"max |diff| {(w - o).abs().max().item():.3e}")


def q8_0_rows_by_hand(raw: np.ndarray, width: int) -> np.ndarray:
    """Q8_0 blocks decoded by hand: 32 values for each block of a half-float scale and 32 int8."""
    blocks = raw.reshape(raw.shape[0], width // 32, 34)
    scale = blocks[:, :, :2].copy().view("<f2").astype(np.float32)
    quants = blocks[:, :, 2:].copy().view(np.int8).astype(np.float32)
    return (scale * quants).reshape(raw.shape[0], width)


class Gguf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = None

    def source(self) -> glue.GGUFSource:
        shards = shards_or_skip(self)
        if Gguf.src is None:
            Gguf.src = glue.GGUFSource(shards)
        return Gguf.src

    def test_a_q8_0_row_equals_its_blocks_decoded_by_hand(self):
        src = self.source()
        name = "output_hc_down.weight"
        self.assertEqual(src.qtype(name), "Q8_0")
        width, n_rows = src.ne(name)
        tensor = src._tensors[name]
        raw = np.asarray(tensor.data).reshape(n_rows, -1)
        for row in (0, 1, n_rows - 1):
            got = src.get(name, row).numpy()
            want = q8_0_rows_by_hand(raw[row:row + 1], width)[0]
            self.assertTrue(np.array_equal(got.view(np.uint32), want.view(np.uint32)), row)

    def test_an_expert_is_its_own_rows_of_the_tensor(self):
        src = self.source()
        name = "blk.0.ffn_gate_exps.weight"
        ne = src.ne(name)
        self.assertEqual(len(ne), 3)
        for e in (0, 1, ne[2] - 1):
            whole = src.dequant_rows(name, e * ne[1], (e + 1) * ne[1])
            got = src.get(name, e).numpy()
            self.assertEqual(got.shape, (ne[1], ne[0]))
            self.assertTrue(np.array_equal(got, whole), e)
            # The first row of the expert is not the first row of its neighbour.
            other = src.dequant_rows(name, ((e + 1) % ne[2]) * ne[1], ((e + 1) % ne[2]) * ne[1] + 1)
            self.assertFalse(np.array_equal(got[:1], other), e)

    def test_a_whole_tensor_has_the_numpy_order_of_its_dims(self):
        src = self.source()
        name = "output_hc_down.weight"
        ne = src.ne(name)
        whole = src.get(name)
        self.assertEqual(list(whole.shape), list(reversed(ne)))
        self.assertTrue(torch.equal(whole[5], src.get(name, 5)))

    def test_rows_out_of_the_tensor_are_refused(self):
        src = self.source()
        name = "output_hc_down.weight"
        n_rows = src.ne(name)[1]
        with self.assertRaises(IndexError):
            src.dequant_rows(name, n_rows - 1, n_rows + 1)

    def test_ple_constants_match_gguf(self):
        """The HF module derives multipliers/offsets/sizes from seed and base; the GGUF carries the
        checkpoint's. They must agree or every PLE row is the wrong one."""
        import gguf
        from transformers.models.qwen4_exp import modeling_qwen4_exp as M

        r = gguf.GGUFReader(shards_or_skip(self)[0])
        kv = lambda k: [int(x) for x in r.fields[k].contents()]
        cfg = reference.text_config()
        with torch.device("meta"):
            emb = M.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, layer_idx=1, ple_layer_index=0)
        mult = M._build_layer_multipliers(emb.unigram_vocab_size, emb.ngram_size, 0, emb.seed).tolist()
        self.assertEqual(mult, kv("qwen4exp.ple.layer_multipliers"))
        self.assertEqual(emb.head_offsets, kv("qwen4exp.ple.head_offsets"))
        self.assertEqual(emb.head_vocab_sizes, kv("qwen4exp.ple.head_vocab_sizes"))
        self.assertEqual(int(r.fields["qwen4exp.ple.eos_token_id"].contents()), cfg.eos_token_id)
        self.assertEqual(emb.ngram_embedding.num_embeddings, 320001536)  # == per_layer_token_embd rows


if __name__ == "__main__":
    unittest.main()
