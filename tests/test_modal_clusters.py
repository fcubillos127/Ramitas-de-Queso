import unittest

import numpy as np

from modal_clusters import build_modal_cluster, certify_cluster_continuity
from modal_tracking import ModalSubspace
from rootfinder_certified import RootCandidate


def fake_root(w):
    return RootCandidate(
        omega=float(w),
        omega_norm=float(w),
        sigma_min=1e-10,
        sigma_min_raw=1e-10,
        sigma_min_balanced=1e-10,
        multiplicity=1,
        det_abs=0.0,
        source="test",
    )


def fake_mode(vector, w):
    q = np.asarray(vector, dtype=complex).reshape(-1, 1)
    q = q / np.linalg.norm(q)
    return ModalSubspace(
        k=0.0,
        omega=float(w),
        omega_norm=float(w),
        multiplicity=1,
        basis=q,
        angular_weights=np.abs(q[:, 0]) ** 2,
        raw_subspace_residual=0.0,
        relative_subspace_residual=0.0,
        balanced_singular_values=np.zeros(q.shape[0]),
        root=fake_root(w),
    )


class TestModalClusters(unittest.TestCase):
    def test_joint_subspace_survives_internal_rotation(self):
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        theta = np.deg2rad(55.0)
        r1 = np.cos(theta) * e1 + np.sin(theta) * e2
        r2 = -np.sin(theta) * e1 + np.cos(theta) * e2

        source = [fake_mode(e1, 1.00), fake_mode(e2, 1.03)]
        target = [fake_mode(r1, 0.99), fake_mode(r2, 1.05)]

        # One diagonal individual overlap is deliberately below the standard
        # 0.65 cosine threshold, but the two-dimensional joint subspace is exact.
        self.assertLess(abs(np.vdot(e1, r1)), 0.65)
        result = certify_cluster_continuity(
            source,
            target,
            [0, 1],
            [0, 1],
            min_principal_cosine=0.65,
            max_center_shift_norm=0.10,
        )
        self.assertTrue(result.certified)
        np.testing.assert_allclose(result.metrics.principal_cosines, [1.0, 1.0], atol=1e-12)

    def test_cluster_rejects_loss_of_one_modal_direction(self):
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        e3 = np.array([0.0, 0.0, 1.0])
        source = [fake_mode(e1, 1.00), fake_mode(e2, 1.03)]
        target = [fake_mode(e1, 0.99), fake_mode(e3, 1.05)]
        result = certify_cluster_continuity(source, target, [0, 1], [0, 1])
        self.assertFalse(result.certified)
        self.assertLess(result.metrics.min_principal_cosine, 0.65)

    def test_cluster_dimension_is_sum_of_event_multiplicities(self):
        modes = [fake_mode([1, 0, 0], 1.0), fake_mode([0, 1, 0], 1.1)]
        cluster = build_modal_cluster(modes, [0, 1])
        self.assertEqual(cluster.dimension, 2)
        np.testing.assert_allclose(cluster.basis.conj().T @ cluster.basis, np.eye(2), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
