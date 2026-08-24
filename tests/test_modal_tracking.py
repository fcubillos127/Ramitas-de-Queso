import unittest

import numpy as np

from certified_solver import CertifiedRed
from modal_tracking import (
    ModalSubspace,
    _equilibrate_with_scaling,
    build_modal_spectrum,
    candidate_modal_links,
    subspace_metrics,
)
from rootfinder_certified import RootCandidate, find_roots_at_k


CT0 = 295.0


def fake_root(w, multiplicity=1):
    return RootCandidate(
        omega=float(w),
        omega_norm=float(w),
        sigma_min=1e-10,
        sigma_min_raw=1e-10,
        sigma_min_balanced=1e-10,
        multiplicity=int(multiplicity),
        det_abs=0.0,
        source="test",
    )


def fake_mode(basis, w=1.0):
    basis = np.asarray(basis, dtype=complex)
    if basis.ndim == 1:
        basis = basis[:, None]
    q, _ = np.linalg.qr(basis, mode="reduced")
    m = q.shape[1]
    return ModalSubspace(
        k=0.0,
        omega=float(w),
        omega_norm=float(w),
        multiplicity=m,
        basis=q,
        angular_weights=np.sum(np.abs(q) ** 2, axis=1) / m,
        raw_subspace_residual=0.0,
        relative_subspace_residual=0.0,
        balanced_singular_values=np.zeros(q.shape[0]),
        root=fake_root(w, m),
    )


def build_reference_red(n_suma=40):
    r = CertifiedRed(comp=["matrix", "inclusion"])
    r.dens = [1150.0, 1250.0]
    r.vel0 = [CT0, CT0]
    r.vels = [894.0, 894.0]
    r.filling = 0.5
    r.cut = 4
    r.nbands = 10
    r.nk = 13
    r.n_suma = int(n_suma)
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


class TestModalSubspaceMetrics(unittest.TestCase):
    def test_simple_mode_is_phase_invariant(self):
        q = np.array([1.0, 2.0j, -0.5], dtype=complex)
        first = fake_mode(q)
        second = fake_mode(np.exp(1.234j) * q)
        metrics = subspace_metrics(first, second)
        self.assertAlmostEqual(metrics.affinity, 1.0, places=12)
        self.assertAlmostEqual(metrics.coverage_first, 1.0, places=12)
        self.assertAlmostEqual(metrics.coverage_second, 1.0, places=12)

    def test_degenerate_subspace_is_basis_rotation_invariant(self):
        Q = np.eye(4, dtype=complex)[:, :2]
        U = np.array(
            [
                [1.0, 1.0j],
                [1.0j, 1.0],
            ],
            dtype=complex,
        ) / np.sqrt(2.0)
        first = fake_mode(Q)
        second = fake_mode(Q @ U)
        metrics = subspace_metrics(first, second)
        np.testing.assert_allclose(metrics.principal_cosines, [1.0, 1.0], atol=1e-12)
        self.assertAlmostEqual(metrics.affinity, 1.0, places=12)

    def test_degenerate_event_can_link_to_two_simple_children(self):
        parent = fake_mode(np.eye(3, dtype=complex)[:, :2], w=1.0)
        child_a = fake_mode(np.array([1.0, 0.0, 0.0]), w=0.98)
        child_b = fake_mode(np.array([0.0, 1.0, 0.0]), w=1.02)
        unrelated = fake_mode(np.array([0.0, 0.0, 1.0]), w=1.01)

        links = candidate_modal_links(
            [parent],
            [child_a, child_b, unrelated],
            max_delta_omega_norm=0.05,
            min_affinity=0.9,
        )
        self.assertEqual({link.current_index for link in links}, {0, 1})
        for link in links:
            self.assertAlmostEqual(link.metrics.affinity, 1.0, places=12)
            self.assertAlmostEqual(link.metrics.coverage_first, 0.5, places=12)
            self.assertAlmostEqual(link.metrics.coverage_second, 1.0, places=12)

    def test_accumulated_scalings_reproduce_balanced_matrix(self):
        A = np.array(
            [
                [1e8, 2.0, 0.0],
                [3.0, 4e-6, 5.0],
                [0.0, 6.0, 7e3],
            ],
            dtype=complex,
        )
        B, left, right = _equilibrate_with_scaling(A, passes=5)
        rebuilt = left[:, None] * A * right[None, :]
        np.testing.assert_allclose(B, rebuilt, rtol=1e-12, atol=1e-12)


class TestModalSubspacePhysical(unittest.TestCase):
    def test_gamma_doublet_has_two_dimensional_raw_nullspace(self):
        r = build_reference_red(n_suma=40)
        k_gamma = np.pi / r.a
        roots = find_roots_at_k(
            r,
            k_gamma,
            CT0,
            w_norm_min=1e-3,
            w_norm_max=1.25,
            ngrid=120,
            scan_eta_norm=1e-6,
            sigma_accept=1e-6,
            multiplicity_tol=1e-5,
        )
        doublets = [root for root in roots if root.multiplicity == 2]
        self.assertEqual(len(doublets), 1)
        root = doublets[0]
        self.assertLess(abs(root.omega_norm - 1.0645835), 5e-4)

        modal = build_modal_spectrum(r, k_gamma, [root])[0]
        self.assertEqual(modal.basis.shape, (2 * r.cut + 1, 2))
        np.testing.assert_allclose(
            modal.basis.conj().T @ modal.basis,
            np.eye(2),
            atol=1e-10,
        )
        self.assertLess(modal.relative_subspace_residual, 1e-6)
        self.assertAlmostEqual(float(np.sum(modal.angular_weights)), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
