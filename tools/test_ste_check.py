"""Tests for ste_check: each limit must catch a text that breaks it and pass a text that keeps it."""
import os
import tempfile
import unittest

import ste_check


def problems(text):
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(text)
    try:
        return ste_check.check(f.name)
    finally:
        os.unlink(f.name)


class SteCheck(unittest.TestCase):
    def test_a_short_sentence_passes(self):
        self.assertEqual(problems("The ledger admits every call. It refuses what does not fit.\n"), [])

    def test_a_long_sentence_fails(self):
        long = " ".join(["word"] * 26) + ".\n"
        self.assertEqual(len(problems(long)), 1)

    def test_a_long_paragraph_fails(self):
        self.assertEqual(len(problems("One. Two. Three. Four. Five. Six. Seven.\n")), 1)

    def test_a_sentence_that_starts_lowercase_is_split(self):
        text = " ".join(["word"] * 20) + ". kaibo " + " ".join(["word"] * 20) + ".\n"
        self.assertEqual(problems(text), [])

    def test_a_sentence_that_ends_in_a_parenthesis_is_split(self):
        text = "(" + " ".join(["word"] * 20) + ".) " + " ".join(["word"] * 20) + ".\n"
        self.assertEqual(problems(text), [])

    def test_a_bold_label_is_not_a_sentence(self):
        self.assertEqual(problems("**1.1 Label.** One. Two. Three. Four. Five. Six.\n"), [])

    def test_quotes_code_tables_and_fences_are_not_checked(self):
        long = " ".join(["word"] * 30)
        text = f'He said "{long}".\n\n`{long}` is one span.\n\n| {long} |\n\n```\n{long}.\n```\n'
        self.assertEqual(problems(text), [])

    def test_each_list_item_is_its_own_block(self):
        text = "".join(f"- Item {i}. It is short.\n" for i in range(5))
        self.assertEqual(problems(text), [])


if __name__ == "__main__":
    unittest.main()
