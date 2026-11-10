"""Tests of the parse of a reply: the program, the vote and the constant."""
import unittest

from swarm import task


class ProgramTest(unittest.TestCase):
    def test_the_first_line_is_the_expression(self):
        src = task.program(task.CODE_PREFILL + "-leftover * 0.9  # tighter\n```\nmore text")
        self.assertEqual(src, task.EXPR_HEAD + "-leftover * 0.9\n")

    def test_math_becomes_numpy(self):
        src = task.program(task.CODE_PREFILL + "math.exp(-leftover)```")
        self.assertEqual(src, task.EXPR_HEAD + "np.exp(-leftover)\n")

    def test_a_reply_without_the_prefill_has_no_program(self):
        self.assertIsNone(task.program("```python\ndef priority(item, bins):\n    return -bins\n```"))

    def test_an_empty_line_has_no_program(self):
        self.assertIsNone(task.program(task.CODE_PREFILL + "\n    x = 1\n"))

    def test_after_prefill(self):
        self.assertEqual(task.after_prefill(" 3", "Best idea:"), "Best idea: 3")
        self.assertEqual(task.after_prefill("Best idea: 3", "Best idea:"), "Best idea: 3")


class VoteTest(unittest.TestCase):
    def test_the_number_after_the_prefill(self):
        self.assertEqual(task.vote_index("Best idea: 3", 8), 2)
        self.assertEqual(task.vote_index("Best idea: #8.", 8), 7)

    def test_a_number_outside_the_list_is_none(self):
        self.assertIsNone(task.vote_index("Best idea: 9", 8))
        self.assertIsNone(task.vote_index("Best idea: 0", 8))

    def test_no_number_is_none(self):
        self.assertIsNone(task.vote_index("Best idea: the first", 8))


class ConstantTest(unittest.TestCase):
    def test_a_number_is_a_constant(self):
        for expr in ("0.0", "-1", "1e-3", "+2", "(0)"):
            with self.subTest(expr=expr):
                self.assertTrue(task.is_constant(task.EXPR_HEAD + expr + "\n"))

    def test_an_expression_is_not_a_constant(self):
        for expr in ("-leftover", "np.zeros_like(bins)", "0.0 * bins", "True", "'a'"):
            with self.subTest(expr=expr):
                self.assertFalse(task.is_constant(task.EXPR_HEAD + expr + "\n"))

    def test_a_bad_program_is_not_a_constant(self):
        self.assertFalse(task.is_constant(None))
        self.assertFalse(task.is_constant(task.EXPR_HEAD + "oops(\n"))
        self.assertFalse(task.is_constant("x = 0\n"))

    def test_the_last_return_counts(self):
        src = "def priority(item, bins):\n    if item > 50:\n        return -bins\n    return 0\n"
        self.assertTrue(task.is_constant(src))


class PromptTest(unittest.TestCase):
    def test_combine_puts_the_parent_last(self):
        parent = ("PARENT_SOURCE", 4.08)
        p = task.combine_prompt(parent, ["Idea: a", "Idea: b"])
        self.assertGreater(p.index("PARENT_SOURCE"), p.index("2. Idea: b"))
        self.assertLess(len(p) - p.index("PARENT_SOURCE"), 400)

    def test_the_act_prompt_names_the_plan(self):
        p = task.act_prompt(task.show(("SRC", 1.0)), "Plan: X")
        self.assertIn("Apply this idea: Plan: X", p)
        self.assertTrue(p.endswith(task.FORM))


if __name__ == "__main__":
    unittest.main()
