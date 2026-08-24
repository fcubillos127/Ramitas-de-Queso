import unittest

import numpy as np

from certified_solver import CertifiedRed
from rootfinder_certified import find_roots_at_k, certify_frequency


class _DoubleDegeneracyModel:
    """3x3 secular matrix with a two-dimensional nullspace at omega=1.

    T = I and G0 = I + diag(d, -d, 2), d=omega-1, hence
    det(TG0-I) = -2 d^2.  The determinant has the SAME sign on both sides
    of the root, so a sign-change-only detector is blind to it.
    """

    a = 2.0 * np.pi
    cut = 1
    n_suma = 1

    @staticmethod
    def _Tn(f, n):
        return 1.0 + 0.0j

    @staticmethod
    def G0(f, k, pol, cut, n_suma=None):
        d = float(f[0]) - 1.0
        return np.eye(3, dtype=complex) + np.diag([d, -d, 2.0])


class TestCertifiedRootFinderSynthetic(unittest.TestCase):
    def test_even_degeneracy_is_found_without_sign_change(self):
        model = _DoubleDegeneracyModel()
        roots = find_roots_at_k(
            model,
            k=0.0,
            C_l0=1.0,
            w_norm_min=0.5,
            w_norm_max=1.5,
            ngrid=101,
            scan_eta_norm=0.0,
            sigma_accept=1e-9,
            multiplicity_tol=1e-7,
        )

        self.assertEqual(len(roots), 1)
        root = roots[0]
        self.assertAlmostEqual(root.omega_norm, 1.0, places=7)
        self.assertLess(root.sigma_min, 1e-9)
        self.assertEqual(root.multiplicity, 2)
        self.assertIn("svd", root.source)
        self.assertNotIn("sign", root.source)


class TestCertifiedRootFinderPhysical(unittest.TestCase):
    @staticmethod
    def build_square_psi0():
        r = CertifiedRed(comp=["matrix", "inclusion"])
        r.dens = [1150.0, 1250.0]
        r.vel0 = [295.0, 295.0]
        r.vels = [894.0, 894.0]
        r.filling = 0.5
        r.cut = 4
        r.nbands = 8
        r.nk = 10
        r.n_suma = 12
        r.lattice = "sq"
        r.psi = 0.0
        r.a = 1.0
        r._set_k_end()
        r.cond_borde = "hollow"
        r.imag_tol = 0.8
        r.sol_tol = 1e-2
        r.asign_param()
        r.r1 = 0.45
        r.r2 = 0.50
        return r

    def test_gamma_contains_expected_certified_roots_and_doublet(self):
        r = self.build_square_psi0()
        gamma = np.pi / r.a
        roots = find_roots_at_k(
            r,
            gamma,
            C_l0=295.0,
            w_norm_min=0.75,
            w_norm_max=1.25,
            ngrid=280,
            scan_eta_norm=1e-6,
            sigma_accept=3e-5,
            multiplicity_tol=3e-5,
        )

        freqs = np.array([x.omega_norm for x in roots])
        self.assertGreaterEqual(len(freqs), 3)

        expected = [0.8520, 1.0646, 1.2173]
        for target in expected:
            self.assertLess(np.min(np.abs(freqs - target)), 3e-3)

        doublet = min(roots, key=lambda x: abs(x.omega_norm - 1.0646))
        self.assertGreaterEqual(doublet.multiplicity, 2)
        self.assertLess(doublet.sigma_min, 3e-5)

    def test_known_near_pole_false_candidate_is_not_certified(self):
        r = self.build_square_psi0()
        k = 1.5 * np.pi / r.a
        scale = 2.0 * np.pi * 295.0 / r.a
        diagnostic = certify_frequency(
            r,
            k,
            omega=0.25127 * scale,
            C_l0=295.0,
            sigma_accept=1e-4,
            multiplicity_tol=1e-4,
        )
        self.assertFalse(diagnostic["accepted"])


if __name__ == "__main__":
    unittest.main()
