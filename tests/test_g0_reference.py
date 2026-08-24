import unittest

import numpy as np
from scipy.special import hankel1

import suma_de_red as sr


class TestLatticeSumReference(unittest.TestCase):
    """Reference checks for the reciprocal-space lattice sum.

    These tests deliberately use a complex k0 with Im(k0)>0 so the direct
    real-space lattice sum converges exponentially and can serve as an
    independent numerical reference.
    """

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

    def test_direct_reference_is_converged(self):
        a = 1.0
        k_vec = np.array([0.0, 0.0])
        k0 = np.pi + 0.5j

        s40 = self.direct_square_S0(k_vec, k0, a=a, n_real=40)
        s80 = self.direct_square_S0(k_vec, k0, a=a, n_real=80)

        self.assertLess(abs(s80 - s40), 5e-9)

    def test_legacy_gamma_q0_bug_is_reproducible(self):
        """The legacy Chin implementation drops the finite Q=0 contribution.

        At Gamma and N=0, J1(Qa)/[Q(Q^2-k0^2)] has the finite limit
        -a/(2*k0^2).  The current implementation replaces the zero denominator
        by 1e-12 while keeping J1(0)=0, hence sets this contribution to zero.
        The resulting S0 must therefore disagree with the independently summed
        real-space lattice sum.
        """
        a = 1.0
        k_path_gamma = np.pi / a  # sr.K maps this point to (0, 0) for sq lattice
        k_vec = sr.K(a, k_path_gamma, "sq")
        k0 = np.pi + 0.5j

        q_mod, ang = sr.precompute_Qh(a, k_vec, 30, "sq")
        legacy = sr.S_pre(0, 0, k0, q_mod, ang, a, "sq")
        direct = self.direct_square_S0(k_vec, k0, a=a, n_real=80)

        # This is a regression test for the diagnosed defect, not desired final
        # behaviour.  It should be removed/replaced once the production lattice
        # sum is migrated to the certified implementation.
        self.assertGreater(abs(legacy - direct), 0.5)


if __name__ == "__main__":
    unittest.main()
