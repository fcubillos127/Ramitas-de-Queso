import unittest

import numpy as np

from targeted_rootfinder import recursive_bounded_minima


class TestRecursiveBoundedMinima(unittest.TestCase):
    def test_finds_two_narrow_minima_in_one_window(self):
        a = 0.4317
        b = 0.5684
        width = 2.0e-3

        def objective(x):
            # Piecewise lower envelope of two narrow quadratic wells.  A single
            # bounded minimisation can return only one of them; recursive side
            # searches must recover both.
            return min(((x - a) / width) ** 2, ((x - b) / width) ** 2)

        minima = recursive_bounded_minima(
            objective,
            0.35,
            0.65,
            accept_value=1e-8,
            xatol=1e-11,
            max_depth=6,
            exclusion_radius=1e-4,
        )
        xs = np.array([item.x for item in minima])
        self.assertTrue(np.any(np.abs(xs - a) < 1e-6), xs)
        self.assertTrue(np.any(np.abs(xs - b) < 1e-6), xs)

    def test_subdivision_recovers_minimum_hidden_from_broad_interval(self):
        root = 0.7234

        def objective(x):
            # A broad, deeper distractor lies outside the acceptance threshold;
            # the accepted narrow well must be found after subdivision.
            narrow = ((x - root) / 5e-4) ** 2
            distractor = 1.0 + ((x - 0.35) / 0.2) ** 2
            return min(narrow, distractor)

        minima = recursive_bounded_minima(
            objective,
            0.2,
            0.9,
            accept_value=1e-7,
            xatol=1e-11,
            max_depth=7,
            exclusion_radius=1e-4,
        )
        xs = np.array([item.x for item in minima])
        self.assertTrue(np.any(np.abs(xs - root) < 1e-6), xs)

    def test_nonfinite_region_is_penalised_not_accepted(self):
        root = 0.81

        def objective(x):
            if 0.49 < x < 0.51:
                return np.inf
            return ((x - root) / 1e-2) ** 2

        minima = recursive_bounded_minima(
            objective,
            0.3,
            0.95,
            accept_value=1e-8,
            xatol=1e-11,
            max_depth=6,
            exclusion_radius=1e-4,
        )
        self.assertGreaterEqual(len(minima), 1)
        self.assertLess(abs(minima[0].x - root), 1e-6)
        self.assertTrue(all(np.isfinite(item.value) for item in minima))


if __name__ == "__main__":
    unittest.main()
