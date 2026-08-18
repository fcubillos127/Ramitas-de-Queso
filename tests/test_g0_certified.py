import unittest

import numpy as np
from scipy.special import hankel1, jn_zeros

import suma_de_red as legacy_sr
import g0_certified as g0c


class TestCertifiedLatticeSum(unittest.TestCase):
    @staticmethod
    def direct_square_S0(k_vec, k0, a=1.0, n_real=80):
        idx = np.arange(-n_real, n_real + 1)
        i, j = np.meshgrid(idx, idx, indexing="ij")
        mask = (i != 0) | (j != 0)
        rx = a * i
        ry = a * j
        radius = np.hypot(rx, ry)
        phase = np.exp(1j * (k_vec[0] * rx + k_vec[1] * ry))
        return np.sum(phase[mask] * hankel1(0, k0 * radius[mask]))

    def test_gamma_q0_matches_independent_real_space_sum(self):
        a = 1.0
        k_path_gamma = np.pi / a
        k_vec = legacy_sr.K(a, k_path_gamma, "sq")

        for k0 in (np.pi + 0.5j, 2.3 + 0.4j, 4.0 + 0.5j):
            q_mod, ang = g0c.precompute_Qh(a, k_vec, 45, "sq")
            reciprocal = g0c.S_pre(0, 0, k0, q_mod, ang, a, "sq")
            direct = self.direct_square_S0(k_vec, k0, a=a, n_real=80)
            self.assertLess(abs(reciprocal - direct), 2e-6)

    def test_q0_fix_is_local_to_gamma_channel_N0(self):
        """Away from Q=0 the certified and legacy formulas should agree."""
        a = 1.0
        k_path = 1.5 * np.pi / a  # interior of Gamma-X for square lattice
        k_vec = legacy_sr.K(a, k_path, "sq")
        k0 = 3.1 + 0.2j
        q_mod, ang = g0c.precompute_Qh(a, k_vec, 35, "sq")

        for N in range(0, 5):
            certified = g0c.S_pre(N, 0, k0, q_mod, ang, a, "sq")
            legacy = legacy_sr.S_pre(N, 0, k0, q_mod, ang, a, "sq")
            self.assertLess(abs(certified - legacy), 1e-10)

    def test_bessel_zero_is_removable_and_matches_bilateral_limit(self):
        a = 1.0
        N = 2
        k0_star = complex(jn_zeros(N + 1, 1)[0])
        k_path = 1.5 * np.pi / a
        k_vec = legacy_sr.K(a, k_path, "sq")
        q_mod, ang = g0c.precompute_Qh(a, k_vec, 80, "sq")

        regularised = g0c.S_pre(
            N, 0, k0_star, q_mod, ang, a, "sq", bessel_zero_tol=1e-6
        )

        errors = []
        for delta in (1e-3, 1e-4, 1e-5):
            left = g0c.S_pre(
                N, 0, k0_star - delta, q_mod, ang, a, "sq",
                bessel_zero_tol=0.0,
            )
            right = g0c.S_pre(
                N, 0, k0_star + delta, q_mod, ang, a, "sq",
                bessel_zero_tol=0.0,
            )
            bilateral = 0.5 * (left + right)
            errors.append(abs(regularised - bilateral))

        self.assertLess(errors[-1], 1e-6)
        self.assertLess(errors[1], errors[0] / 20.0)
        self.assertLess(errors[2], errors[1] / 20.0)

    def test_bessel_regularised_value_is_finite(self):
        a = 1.0
        k_path = 1.5 * np.pi / a
        k_vec = legacy_sr.K(a, k_path, "sq")
        q_mod, ang = g0c.precompute_Qh(a, k_vec, 80, "sq")

        for N in (0, 1, 2, 3):
            k0_star = complex(jn_zeros(N + 1, 1)[0])
            value = g0c.S_pre(
                N, 0, k0_star, q_mod, ang, a, "sq", bessel_zero_tol=1e-6
            )
            self.assertTrue(np.isfinite(value.real))
            self.assertTrue(np.isfinite(value.imag))


if __name__ == "__main__":
    unittest.main()
