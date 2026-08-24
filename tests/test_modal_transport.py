import unittest

import numpy as np

from modal_tracking import ModalSubspace
from modal_transport import assign_modal_transport
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


def fake_mode(basis, w):
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


class TestModalTransport(unittest.TestCase):
    def test_modal_identity_beats_frequency_order_at_crossing(self):
        e1 = np.array([1.0, 0.0], dtype=complex)
        e2 = np.array([0.0, 1.0], dtype=complex)
        source = [fake_mode(e1, 0.90), fake_mode(e2, 1.10)]
        # Frequency order is reversed relative to modal identity.
        target = [fake_mode(e2, 0.96), fake_mode(e1, 1.04)]

        result = assign_modal_transport(
            source,
            target,
            max_delta_omega_norm=0.20,
            min_principal_cosine=0.7,
        )
        pairs = {(edge.source_index, edge.target_index) for edge in result.edges}
        self.assertEqual(pairs, {(0, 1), (1, 0)})
        self.assertEqual(result.matched_dimensions, 2)

    def test_doublet_splits_into_two_simple_children(self):
        parent = fake_mode(np.eye(3, dtype=complex)[:, :2], 1.0)
        children = [
            fake_mode(np.array([1.0, 0.0, 0.0]), 0.98),
            fake_mode(np.array([0.0, 1.0, 0.0]), 1.02),
        ]
        result = assign_modal_transport(
            [parent],
            children,
            max_delta_omega_norm=0.05,
            min_principal_cosine=0.7,
        )
        self.assertEqual(result.matched_dimensions, 2)
        self.assertEqual(result.source_used, (2,))
        self.assertEqual(result.target_used, (1, 1))
        self.assertEqual({edge.target_index for edge in result.edges}, {0, 1})

    def test_identical_doublets_transport_two_dimensions_without_basis_choice(self):
        Q = np.eye(4, dtype=complex)[:, :2]
        U = np.array([[1.0, 1.0j], [1.0j, 1.0]], dtype=complex) / np.sqrt(2.0)
        source = fake_mode(Q, 1.0)
        target = fake_mode(Q @ U, 1.01)
        result = assign_modal_transport(
            [source], [target], max_delta_omega_norm=0.05, min_principal_cosine=0.7
        )
        self.assertEqual(len(result.edges), 1)
        self.assertEqual(result.edges[0].dimensions, 2)
        self.assertEqual(result.matched_dimensions, 2)

    def test_partial_intersection_limits_edge_capacity(self):
        source = fake_mode(np.eye(3, dtype=complex)[:, :2], 1.0)
        target = fake_mode(np.eye(3, dtype=complex)[:, [0, 2]], 1.01)
        result = assign_modal_transport(
            [source], [target], max_delta_omega_norm=0.05, min_principal_cosine=0.7
        )
        self.assertEqual(result.matched_dimensions, 1)
        self.assertEqual(result.source_unmatched, (1,))
        self.assertEqual(result.target_unmatched, (1,))

    def test_unrelated_nearby_root_remains_unmatched(self):
        source = fake_mode(np.array([1.0, 0.0]), 1.0)
        target = fake_mode(np.array([0.0, 1.0]), 1.001)
        result = assign_modal_transport(
            [source], [target], max_delta_omega_norm=0.05, min_principal_cosine=0.7
        )
        self.assertEqual(result.matched_dimensions, 0)
        self.assertEqual(result.source_unmatched, (1,))
        self.assertEqual(result.target_unmatched, (1,))


if __name__ == "__main__":
    unittest.main()
