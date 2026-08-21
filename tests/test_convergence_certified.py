import unittest

from convergence_certified import assess_root_sequence, match_root_sets
from rootfinder_certified import RootCandidate


def root(w, residual=1e-9, multiplicity=1, source="test"):
    return RootCandidate(
        omega=w,
        omega_norm=w,
        sigma_min=residual,
        sigma_min_raw=residual,
        sigma_min_balanced=residual,
        multiplicity=multiplicity,
        det_abs=0.0,
        source=source,
    )


class TestConvergenceCertified(unittest.TestCase):
    def test_matching_is_one_to_one(self):
        old = [root(0.50), root(0.70)]
        new = [root(0.50001), root(0.50002), root(0.70001)]

        matches, missing_old, missing_new = match_root_sets(
            old, new, max_match_delta_norm=1e-2
        )

        self.assertEqual(len(matches), 2)
        self.assertEqual(missing_old, ())
        self.assertEqual(len(missing_new), 1)

    def test_stable_spectrum_converges_after_two_refinements(self):
        n = [8, 12, 20]
        spectra = [
            [root(0.80010), root(1.10012)],
            [root(0.80003), root(1.10004)],
            [root(0.80001), root(1.10001)],
        ]
        result = assess_root_sequence(
            n,
            spectra,
            stable_steps=2,
            frequency_atol_norm=1e-4,
            frequency_rtol=0.0,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.recommended_n_suma, 20)

    def test_transient_root_blocks_convergence_until_it_disappears_stably(self):
        n = [8, 12, 20, 30, 40]
        spectra = [
            [root(0.8), root(1.0), root(1.015)],
            [root(0.80003), root(1.00003), root(1.014)],
            [root(0.80001), root(1.00001)],
            [root(0.800005), root(1.000005)],
            [root(0.800003), root(1.000003)],
        ]
        result = assess_root_sequence(
            n,
            spectra,
            stable_steps=2,
            frequency_atol_norm=5e-5,
            frequency_rtol=0.0,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.recommended_n_suma, 40)
        self.assertFalse(result.steps[1].converged)  # 12 -> 20 loses transient root
        self.assertTrue(result.steps[2].converged)
        self.assertTrue(result.steps[3].converged)

    def test_multiplicity_change_prevents_convergence(self):
        n = [8, 12, 20]
        spectra = [
            [root(0.9, multiplicity=1)],
            [root(0.90001, multiplicity=2)],
            [root(0.900005, multiplicity=2)],
        ]
        result = assess_root_sequence(
            n,
            spectra,
            stable_steps=2,
            frequency_atol_norm=5e-5,
            frequency_rtol=0.0,
        )
        self.assertFalse(result.converged)
        self.assertIsNone(result.recommended_n_suma)
        self.assertFalse(result.steps[0].multiplicity_stable)

    def test_bad_residual_prevents_convergence(self):
        n = [8, 12, 20]
        spectra = [
            [root(0.7)],
            [root(0.70001, residual=2e-5)],
            [root(0.700005)],
        ]
        result = assess_root_sequence(
            n,
            spectra,
            stable_steps=2,
            frequency_atol_norm=5e-5,
            frequency_rtol=0.0,
            residual_tol=1e-6,
        )
        self.assertFalse(result.converged)
        self.assertFalse(result.steps[0].residuals_ok)
        self.assertFalse(result.steps[1].residuals_ok)

    def test_frequency_drift_prevents_false_convergence(self):
        n = [8, 12, 20, 30]
        spectra = [
            [root(1.0119)],
            [root(1.0115)],
            [root(1.01138)],
            [root(1.01135)],
        ]
        result = assess_root_sequence(
            n,
            spectra,
            stable_steps=2,
            frequency_atol_norm=2e-5,
            frequency_rtol=0.0,
        )
        self.assertFalse(result.converged)
        self.assertIsNone(result.recommended_n_suma)


if __name__ == "__main__":
    unittest.main()
