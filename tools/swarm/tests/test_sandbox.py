"""A test of the sandbox on the CPU: best fit gives the score of the first trials (docs/swarm-ideas.md)."""
import shutil
import unittest

from swarm import binpack, sandbox


@unittest.skipUnless(shutil.which("bwrap") and shutil.which("prlimit"), "the sandbox needs bwrap and prlimit")
class SandboxTest(unittest.TestCase):
    def test_best_fit_scores_4_08_on_the_train_seed(self):
        r = sandbox.evaluate(binpack.BEST_FIT, 1)
        self.assertIn("excess", r, r)
        self.assertAlmostEqual(r["excess"], 4.08, places=2)

    def test_the_sandbox_has_no_network(self):
        r = sandbox.evaluate("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=2)\n", 1, timeout=10)
        self.assertIn("fail", r)
        self.assertIn("OSError", r["fail"])


if __name__ == "__main__":
    unittest.main()
