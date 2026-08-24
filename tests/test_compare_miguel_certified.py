import unittest

from compare_miguel_certified import gated_frequency_assignment


class TestGatedFrequencyAssignment(unittest.TestCase):
    def test_invalid_distant_candidate_cannot_steal_nearly_exact_match(self):
        # Regression from the physical k*a/pi=0.125 comparison.  An
        # unconstrained Hungarian assignment can pair 0.724288 with 0.713747
        # and sacrifice the excellent 0.724288 -> 0.724304 edge because the
        # third raw candidate is far from both low certified roots.
        raw = [
            {"omega_norm": 0.724288},
            {"omega_norm": 0.794709},
            {"omega_norm": 1.306578},
        ]
        certified = [
            {"omega_norm": 0.713747},
            {"omega_norm": 0.724304},
            {"omega_norm": 1.306567},
        ]
        matches = gated_frequency_assignment(raw, certified, 0.01)
        pairs = {(i, j) for i, j, _delta in matches}
        self.assertEqual(pairs, {(0, 1), (2, 2)})

    def test_matching_is_one_to_one(self):
        raw = [{"omega_norm": 1.0}, {"omega_norm": 1.001}]
        certified = [{"omega_norm": 1.0002}]
        matches = gated_frequency_assignment(raw, certified, 0.01)
        self.assertEqual(len(matches), 1)


if __name__ == "__main__":
    unittest.main()
