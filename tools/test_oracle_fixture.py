"""Tests of the pure parts of oracle_fixture.py. Run: python3 -m unittest test_oracle_fixture (in tools/).

The reference needs torch and 30 GiB, so these tests do not run it. The constants of the vector are also in
`crates/nagaya-mk-qwen38-gfx1151/src/control.rs`, in the test `the_vector_of_the_oracle_matches_the_python_tool`.
Change both together.
"""
import hashlib
import struct
import sys
import unittest
from unittest import mock

import oracle_fixture as tool

# (layer, j, the numerator of v * 256). The Rust test has the same table.
VECTOR_CASES = [
    (0, 0, -128),
    (0, 1, -97),
    (1, 0, -111),
    (24, 8, 14),
    (40, 10239, 52),
    (47, 5000, -71),
]


class Vector(unittest.TestCase):
    def test_the_numerators_are_the_hand_computed_values(self):
        for layer, j, want in VECTOR_CASES:
            self.assertEqual(tool.vector_numerator(layer, j), want, (layer, j))

    def test_the_values_are_exact_in_f32(self):
        self.assertEqual(tool.vector_value(0, 0), -0.5)
        self.assertEqual(tool.vector_value(0, 1), -0.37890625)
        self.assertEqual(tool.vector_value(40, 10239), 0.203125)
        for layer in (0, 17, 47):
            for j in range(0, tool.WIDTH, 97):
                value = tool.vector_value(layer, j)
                self.assertEqual(struct.unpack("<f", struct.pack("<f", value))[0], value)
                self.assertTrue(-0.5 <= value <= 0.5)

    def test_the_vector_has_the_width_of_the_residual(self):
        self.assertEqual(len(tool.control_vector(3)), 10240)

    def test_a_layer_changes_the_vector(self):
        self.assertNotEqual(tool.control_vector(8), tool.control_vector(24))


class Indices(unittest.TestCase):
    def test_the_residual_indices_are_fixed(self):
        indices = tool.sample_indices(10240)
        self.assertEqual(len(indices), 64)
        self.assertEqual(indices[:3], [3, 163, 323])
        self.assertEqual(indices[-1], 10083)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(all(0 <= i < 10240 for i in indices))

    def test_the_fold_indices_are_fixed(self):
        indices = tool.sample_indices(2560)
        self.assertEqual(indices[:2], [3, 43])
        self.assertEqual(indices[-1], 63 * 40 + 3)
        self.assertTrue(all(0 <= i < 2560 for i in indices))


class Dtype(unittest.TestCase):
    def test_float32_passes(self):
        tool.require_float32("float32")

    def test_another_dtype_stops_the_tool_with_a_clear_message(self):
        with self.assertRaises(SystemExit) as stop:
            tool.require_float32("bfloat16")
        self.assertIn("bfloat16, not float32", str(stop.exception))
        self.assertIn("run the reference in float32", str(stop.exception))


class Bytes(unittest.TestCase):
    def test_f32_bytes_are_little_endian(self):
        self.assertEqual(tool.f32_bytes([1.0]), b"\x00\x00\x80\x3f")
        self.assertEqual(tool.f32_values(b"\x00\x00\x80\xbf"), (-1.0,))

    def test_a_part_of_a_value_is_refused(self):
        with self.assertRaises(ValueError):
            tool.f32_values(b"\x00\x00\x80")


def row(per_stream):
    """A row whose stream k holds `per_stream[k]` in each of its 2,560 values."""
    return tool.f32_bytes([x for x in per_stream for _ in range(tool.STREAM_WIDTH)])


class Entries(unittest.TestCase):
    def test_a_residual_entry_holds_the_norms_the_sample_and_the_hash(self):
        raw = row([1.0, 2.0, 3.0, 4.0])
        entry = tool.residual_entry(5, raw, tool.sample_indices(tool.WIDTH))
        self.assertEqual(entry["layer"], 5)
        self.assertEqual(entry["sha256"], hashlib.sha256(raw).hexdigest())
        for k, norm in enumerate(entry["stream_norms"]):
            self.assertAlmostEqual(norm, (k + 1) * 2560 ** 0.5, places=9)
        self.assertEqual(len(entry["sample"]), 64)
        # Index 3 is in stream 0, and index 10083 is in stream 3 (10083 // 2560).
        self.assertEqual(entry["sample"][0], 1.0)
        self.assertEqual(entry["sample"][-1], 4.0)

    def test_a_residual_row_of_the_wrong_width_is_refused(self):
        with self.assertRaises(ValueError) as error:
            tool.residual_entry(0, tool.f32_bytes([0.0] * 10239), tool.sample_indices(tool.WIDTH))
        self.assertIn("10239 values", str(error.exception))

    def test_a_fold_entry(self):
        raw = tool.f32_bytes([2.0] * 2560)
        entry = tool.fold_entry(raw, tool.sample_indices(2560))
        self.assertAlmostEqual(entry["norm"], 2.0 * 2560 ** 0.5, places=9)
        self.assertEqual(entry["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(len(entry["sample"]), 64)
        with self.assertRaises(ValueError):
            tool.fold_entry(tool.f32_bytes([0.0] * 2559), tool.sample_indices(2560))


class Document(unittest.TestCase):
    def test_the_layout_names_the_formula_the_strength_the_layers_and_the_runs(self):
        document = tool.residual_document([{"name": "plain"}], {"reference_rev": "abc"}, 0.5)
        self.assertEqual(document["dtype"], "float32")
        self.assertEqual(document["layers"], 48)
        self.assertEqual(document["width"], 10240)
        self.assertEqual(document["fold_width"], 2560)
        self.assertEqual(document["reference_rev"], "abc")
        self.assertEqual(document["sample_indices"], tool.sample_indices(10240))
        self.assertEqual(document["fold_sample_indices"], tool.sample_indices(2560))
        self.assertEqual(document["add"]["layers"], [8, 24, 40])
        self.assertEqual(document["add"]["strength"], 0.5)
        self.assertEqual(document["add"]["run"], "plain_add")
        self.assertIn("31*j + 17*l", document["add"]["formula"])
        self.assertEqual(document["runs"], [{"name": "plain"}])

    def test_the_strength_is_also_given_as_the_f32_that_the_engine_reads(self):
        document = tool.residual_document([], {}, 0.1)
        self.assertEqual(document["add"]["strength"], 0.1)
        self.assertEqual(document["add"]["strength_f32"], struct.unpack("<f", struct.pack("<f", 0.1))[0])

    def test_the_runs_and_the_golden_files(self):
        self.assertEqual([name for name, _, _ in tool.RUNS], ["plain", "chat", "plain_add"])
        self.assertEqual([add for _, _, add in tool.RUNS], [False, False, True])
        self.assertEqual(tool.residual_files("plain_add"), ("residual.plain_add.f32", "fold.plain_add.f32"))


class Options(unittest.TestCase):
    def run_main(self, *argv):
        with mock.patch.object(sys, "argv", ["oracle_fixture.py", *argv]):
            with self.assertRaises(SystemExit) as stop:
                tool.main()
        return stop.exception.code

    def test_golden_without_residual_is_refused(self):
        self.assertEqual(self.run_main("out.json", "--golden", "dir"), 2)

    def test_strength_without_residual_is_refused(self):
        self.assertEqual(self.run_main("out.json", "--strength", "1"), 2)

    def test_residual_with_an_out_argument_is_refused(self):
        self.assertEqual(self.run_main("out.json", "--residual", "r.json"), 2)

    def test_no_argument_is_refused(self):
        self.assertEqual(self.run_main(), 2)


if __name__ == "__main__":
    unittest.main()
