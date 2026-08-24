import unittest

import numpy as np

from certified_bands import (
    BandRunConfig,
    matches_baseline_scope,
    path_grid,
    validate_supported_scope,
)


class TestCertifiedBandDriver(unittest.TestCase):
    def test_default_configuration_is_exact_closed_baseline(self):
        self.assertTrue(matches_baseline_scope(BandRunConfig()))

    def test_parameter_change_is_not_silently_called_certified(self):
        changed = BandRunConfig(n_suma=60)
        self.assertFalse(matches_baseline_scope(changed))
        # It remains a supported psi=0 calculation, but its quality label must
        # be COMPLETE_UNCERTIFIED_PARAMETER_SCOPE until separately validated.
        validate_supported_scope(changed)

    def test_psi_nonzero_is_explicitly_blocked(self):
        with self.assertRaises(NotImplementedError):
            validate_supported_scope(BandRunConfig(psi=0.2))

    def test_default_path_contains_closed_loop_and_two_audited_half_steps(self):
        kp = path_grid(9)
        self.assertEqual(len(kp), 27)
        self.assertAlmostEqual(float(kp[0]), 0.0)
        self.assertAlmostEqual(float(kp[-1]), 3.0)
        self.assertTrue(np.any(np.isclose(kp, 0.0625)))
        self.assertTrue(np.any(np.isclose(kp, 2.0625)))
        self.assertEqual(int(np.count_nonzero(np.isclose(kp, 1.0))), 1)
        self.assertEqual(int(np.count_nonzero(np.isclose(kp, 2.0))), 1)

    def test_guard_band_must_extend_above_reported_window(self):
        with self.assertRaises(ValueError):
            validate_supported_scope(BandRunConfig(search_w_max=1.40))


if __name__ == "__main__":
    unittest.main()
