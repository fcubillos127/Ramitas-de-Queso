import unittest

from adaptive_nsum import certify_roots_adaptive_nsum
from rootfinder_certified import RootCandidate


def root(w, residual=1e-9, multiplicity=1):
    return RootCandidate(
        omega=w,
        omega_norm=w,
        sigma_min=residual,
        sigma_min_raw=residual,
        sigma_min_balanced=residual,
        multiplicity=multiplicity,
        det_abs=0.0,
        source="test",
    )


class DummyRed:
    def __init__(self):
        self.n_suma = 99


class TestAdaptiveNsum(unittest.TestCase):
    def _finder(self, spectra):
        def finder(red, k, C_l0, **kwargs):
            return spectra[red.n_suma]
        return finder

    def test_candidate_requires_one_extra_confirmation_step(self):
        red = DummyRed()
        spectra = {
            8: [root(0.80003)],
            12: [root(0.80002)],
            20: [root(0.80001)],
            30: [root(0.800005)],
            40: [root(0.800003)],
        }
        result = certify_roots_adaptive_nsum(
            red,
            1.0,
            1.0,
            n_schedule=[8, 12, 20, 30, 40],
            stable_steps=2,
            confirmation_steps=1,
            frequency_atol_norm=5e-5,
            frequency_rtol=0.0,
            finder=self._finder(spectra),
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.recommended_n_suma, 20)
        self.assertEqual(result.certification_n_suma, 30)
        self.assertEqual(result.evaluated_n_values, (8, 12, 20, 30))
        self.assertEqual(red.n_suma, 99)

    def test_failed_confirmation_discards_early_candidate(self):
        red = DummyRed()
        spectra = {
            8: [root(0.80003)],
            12: [root(0.80002)],
            20: [root(0.80001)],
            30: [root(0.80100)],
        }
        result = certify_roots_adaptive_nsum(
            red,
            1.0,
            1.0,
            n_schedule=[8, 12, 20, 30],
            stable_steps=2,
            confirmation_steps=1,
            frequency_atol_norm=5e-5,
            frequency_rtol=0.0,
            finder=self._finder(spectra),
        )
        self.assertFalse(result.converged)
        self.assertIsNone(result.recommended_n_suma)
        self.assertIsNone(result.certification_n_suma)
        self.assertEqual(red.n_suma, 99)

    def test_late_stable_regime_recommends_higher_truncation(self):
        red = DummyRed()
        spectra = {
            8: [root(1.0119)],
            12: [root(1.0115)],
            20: [root(1.01138)],
            30: [root(1.011356)],
            40: [root(1.011350)],
            60: [root(1.011347)],
        }
        result = certify_roots_adaptive_nsum(
            red,
            1.0,
            1.0,
            n_schedule=[8, 12, 20, 30, 40, 60],
            stable_steps=2,
            confirmation_steps=1,
            frequency_atol_norm=5e-5,
            frequency_rtol=0.0,
            finder=self._finder(spectra),
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.recommended_n_suma, 40)
        self.assertEqual(result.certification_n_suma, 60)


if __name__ == "__main__":
    unittest.main()
