"""Tests of prompt_lookup_sim.py. Run: python3 -m unittest test_prompt_lookup_sim (in tools/)."""
import unittest

import prompt_lookup_sim as sim


class DraftTest(unittest.TestCase):
    def test_longest_match_wins_and_the_last_occurrence_is_used(self):
        # Newest 2 tokens [1, 2] occur at 0 (followed by 9) and at 4 (followed by 7).
        ids = [1, 2, 9, 9, 1, 2, 7, 8, 1, 2]
        self.assertEqual(sim.draft(ids, 10, 3, 1, 4), [7, 8, 1])

    def test_a_longer_match_beats_a_later_shorter_one(self):
        # [5, 1, 2] matches at 0 (then 6). The shorter [1, 2] also matches at 4, then 7.
        ids = [5, 1, 2, 6, 3, 1, 2, 7, 5, 1, 2]
        self.assertEqual(sim.draft(ids, 11, 1, 1, 4), [6])

    def test_no_match_gives_no_draft(self):
        self.assertEqual(sim.draft([1, 2, 3, 4], 4, 4, 1, 4), [])

    def test_the_draft_stops_at_the_end_of_the_context(self):
        ids = [1, 2, 3, 1]
        self.assertEqual(sim.draft(ids, 4, 8, 1, 4), [2, 3, 1])

    def test_the_draft_never_reads_the_text_after_the_context(self):
        ids = [1, 2, 3, 1, 9, 9]  # ids[4:] is the future: a draft that reads it is a leak
        self.assertEqual(sim.draft(ids, 4, 8, 1, 4), [2, 3, 1])

    def test_min_n_refuses_short_matches(self):
        ids = [1, 2, 3, 4, 1]
        self.assertEqual(sim.draft(ids, 5, 2, 2, 4), [])


class RunTest(unittest.TestCase):
    def test_a_copy_is_accepted_whole(self):
        # Prompt [1..6]; the model then copies it. Drafts find it after the first token.
        ids = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
        steps = sim.run(ids, 6, 4, 1, 4)
        # step 1: context ends in 6, no earlier 6 followed by anything -> no draft, 1 token (1).
        self.assertEqual(steps[0], (0, 0))
        # step 2: context ... 6, 1 -> draft [2,3,4,5] all accepted, 5 tokens.
        self.assertEqual(steps[1], (4, 4))
        # the whole text is 6 generated tokens: 1 + 5.
        self.assertEqual(sum(a + 1 for _, a in steps), 6)

    def test_a_wrong_draft_accepts_its_correct_prefix_only(self):
        ids = [1, 2, 3, 4, 9, 9, 1, 2, 7, 7]
        steps = sim.run(ids, 7, 3, 1, 4)  # context [1,2,3,4,9,9,1], model writes 2, 7, 7
        self.assertEqual(steps[0], (3, 1))  # draft [2,3,4], 2 right, 3 wrong
        self.assertEqual(sum(a + 1 for _, a in steps), 3)

    def test_the_last_block_does_not_overrun_the_text(self):
        ids = [1, 2, 1, 2]
        steps = sim.run(ids, 3, 8, 1, 4)  # one token left to write
        self.assertEqual(sum(a + 1 for _, a in steps), 1)


class CostTest(unittest.TestCase):
    def test_cost_points_of_the_doc(self):
        self.assertEqual(sim.cost(1), 1.0)
        self.assertAlmostEqual(sim.cost(8), 1.8)
        self.assertAlmostEqual(sim.cost(16), 2.5)
        self.assertAlmostEqual(sim.cost(12), 2.15)

    def test_speed_of_no_drafts_is_one(self):
        self.assertEqual(sim.summarize([(0, 0)] * 5)["speed"], 1.0)

    def test_speed_and_shares(self):
        s = sim.summarize([(7, 7), (0, 0)])
        self.assertAlmostEqual(s["speed"], 9 / (1.8 + 1.0))
        self.assertAlmostEqual(s["draft_share"], 0.5)
        self.assertAlmostEqual(s["accepted_per_block"], 3.5)
        self.assertAlmostEqual(s["accepted_per_drafted_block"], 7.0)


if __name__ == "__main__":
    unittest.main()
