import unittest

import numpy as np

from rootfinder_certified import equilibrate_matrix


class TestSecularMatrixEquilibration(unittest.TestCase):
    def test_full_rank_dynamic_range_is_not_mistaken_for_root(self):
        A = np.diag([1e8, 1.0, 1e-8]).astype(complex)
        raw_sigma = np.linalg.svd(A, compute_uv=False)[-1]
        balanced_sigma = np.linalg.svd(
            equilibrate_matrix(A, passes=8), compute_uv=False
        )[-1]

        self.assertLess(raw_sigma, 1e-7)
        self.assertGreater(balanced_sigma, 1e-2)

    def test_exact_nullity_is_preserved_by_equilibration(self):
        A = np.diag([1e8, 1.0, 0.0]).astype(complex)
        balanced = equilibrate_matrix(A, passes=8)
        singular = np.linalg.svd(balanced, compute_uv=False)

        self.assertEqual(np.linalg.matrix_rank(A), np.linalg.matrix_rank(balanced))
        self.assertEqual(np.count_nonzero(singular < 1e-12), 1)


if __name__ == "__main__":
    unittest.main()
