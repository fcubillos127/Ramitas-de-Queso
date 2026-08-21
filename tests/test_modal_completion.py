import unittest

import numpy as np

from modal_completion import diagnose_descendant_coverage, orthonormal_union
from modal_tracking import ModalSubspace
from rootfinder_certified import RootCandidate


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


class TestModalCompletionDiagnostics(unittest.TestCase):
    def test_union_removes_double_counting_of_collinear_candidates(self):
        q = np.array([1.0, 0.0, 0.0], dtype=complex)
        a = fake_mode(q, 1.0)
        b = fake_mode(np.exp(0.7j) * q, 1.01)
        union = orthonormal_union([a, b])
        self.assertEqual(union.shape[1], 1)

    def test_doublet_with_only_one_child_requests_refinement(self):
        parent = fake_mode(np.eye(3, dtype=complex)[:, :2], 1.0)
        child = fake_mode(np.array([1.0, 0.0, 0.0]), 1.01)
        diag = diagnose_descendant_coverage(
            [parent],
            [child],
            max_delta_omega_norm=0.05,
            min_pair_affinity=0.1,
            coverage_floor=0.6,
        )[0]
        self.assertEqual(diag.source_multiplicity, 2)
        self.assertEqual(diag.union_dimension, 1)
        self.assertTrue(diag.needs_refinement)
        self.assertAlmostEqual(diag.coverage, 0.5, places=12)

    def test_doublet_with_two_orthogonal_children_is_complete(self):
        parent = fake_mode(np.eye(3, dtype=complex)[:, :2], 1.0)
        child_a = fake_mode(np.array([1.0, 0.0, 0.0]), 0.99)
        child_b = fake_mode(np.array([0.0, 1.0, 0.0]), 1.01)
        diag = diagnose_descendant_coverage(
            [parent],
            [child_a, child_b],
            max_delta_omega_norm=0.05,
            min_pair_affinity=0.1,
            coverage_floor=0.6,
        )[0]
        self.assertEqual(diag.union_dimension, 2)
        self.assertFalse(diag.needs_refinement)
        self.assertAlmostEqual(diag.coverage, 1.0, places=12)

    def test_simple_parent_can_merge_into_doublet_without_false_alarm(self):
        parent = fake_mode(np.array([1.0, 1.0, 0.0]), 1.0)
        doublet = fake_mode(np.eye(3, dtype=complex)[:, :2], 1.01)
        diag = diagnose_descendant_coverage(
            [parent],
            [doublet],
            max_delta_omega_norm=0.05,
            min_pair_affinity=0.1,
            coverage_floor=0.6,
        )[0]
        self.assertGreaterEqual(diag.union_dimension, 1)
        self.assertFalse(diag.needs_refinement)
        self.assertAlmostEqual(diag.coverage, 1.0, places=12)

    def test_unrelated_nearby_root_does_not_satisfy_modal_capacity(self):
        parent = fake_mode(np.eye(3, dtype=complex)[:, :2], 1.0)
        unrelated = fake_mode(np.array([0.0, 0.0, 1.0]), 1.005)
        diag = diagnose_descendant_coverage(
            [parent],
            [unrelated],
            max_delta_omega_norm=0.05,
            min_pair_affinity=0.1,
            coverage_floor=0.6,
        )[0]
        self.assertEqual(diag.candidate_indices, ())
        self.assertTrue(diag.needs_refinement)


if __name__ == "__main__":
    unittest.main()
