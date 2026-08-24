import unittest

import numpy as np

import g0_certified as g0c


class DummyRed:
    def __init__(self):
        self.a = 1.0
        self.lattice = "sq"
        self.n_suma = 5
        self.vel0 = [295.0, 295.0]

    def k0(self, f, pol):
        return (f[0] + 1j * f[1]) / self.vel0[pol]


class TestCertifiedConvergence(unittest.TestCase):
    def test_converged_result_is_complete_square_truncation(self):
        red = DummyRed()
        k = 1.5 * np.pi
        f = [2 * np.pi * 295.0 * 0.8, 0.2]

        matrix, info = g0c.G0_converged(
            red, f, k, pol=1, cut=2,
            n_suma_ini=5,
            tol=5e-4,
            n_suma_max=30,
            stable_passes=2,
        )

        self.assertTrue(info["converged"])
        n = info["n_suma"]
        direct_at_same_n = g0c.G0_matrix(red, f, k, 1, 2, n)
        np.testing.assert_allclose(matrix, direct_at_same_n, rtol=0, atol=1e-13)

    def test_refinement_changes_decrease_away_from_singularities(self):
        red = DummyRed()
        k = 1.5 * np.pi
        f = [2 * np.pi * 295.0 * 0.8, 0.2]

        mats = [g0c.G0_matrix(red, f, k, 1, 2, n) for n in (5, 8, 12, 20)]
        diffs = [np.max(np.abs(mats[i + 1] - mats[i])) for i in range(len(mats) - 1)]

        self.assertLess(diffs[1], diffs[0])
        self.assertLess(diffs[2], diffs[1])


if __name__ == "__main__":
    unittest.main()
