import unittest

import numpy as np

from modal_path import assemble_modal_path_graph
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


def fake_mode(k, basis, w):
    basis = np.asarray(basis, dtype=complex)
    if basis.ndim == 1:
        basis = basis[:, None]
    q, _ = np.linalg.qr(basis, mode="reduced")
    m = q.shape[1]
    return ModalSubspace(
        k=float(k),
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


class FakeRed:
    """Only the modal-basis constructor is bypassed in these topology tests."""


class TestModalPathTopology(unittest.TestCase):
    def setUp(self):
        # assemble_modal_path_graph normally builds modes from the real secular
        # matrix. These tests exercise only graph semantics by monkeypatching the
        # imported builder locally.
        import modal_path
        self.modal_path = modal_path
        self.original_builder = modal_path.build_modal_spectrum

    def tearDown(self):
        self.modal_path.build_modal_spectrum = self.original_builder

    def _assemble_from_modes(self, modes_by_layer):
        mapping = {}
        roots = []
        ks = []
        for i, modes in enumerate(modes_by_layer):
            k = float(i)
            ks.append(k)
            local_roots = tuple(mode.root for mode in modes)
            roots.append(local_roots)
            mapping[(k, tuple(id(root) for root in local_roots))] = tuple(modes)

        def builder(_red, k, local_roots):
            key = (float(k), tuple(id(root) for root in local_roots))
            return mapping[key]

        self.modal_path.build_modal_spectrum = builder
        return assemble_modal_path_graph(
            FakeRed(),
            ks,
            roots,
            search_half_width_norm=0.20,
            transport_max_delta_omega_norm=0.20,
            min_pair_affinity=0.10,
            coverage_floor=0.50,
            min_principal_cosine=0.70,
        )

    def test_doublet_is_graph_node_with_two_transport_dimensions(self):
        e1 = np.array([1.0, 0.0, 0.0], dtype=complex)
        e2 = np.array([0.0, 1.0, 0.0], dtype=complex)
        left = [fake_mode(0, e1, 0.98), fake_mode(0, e2, 1.02)]
        middle = [fake_mode(1, np.column_stack([e1, e2]), 1.00)]
        right = [fake_mode(2, e1, 1.03), fake_mode(2, e2, 0.97)]

        graph = self._assemble_from_modes([left, middle, right])
        self.assertTrue(graph.complete_under_policy)
        self.assertEqual(graph.layers[1].modes[0].multiplicity, 2)

        into_middle = [edge for edge in graph.edges if edge.right_layer == 1]
        out_middle = [edge for edge in graph.edges if edge.left_layer == 1]
        self.assertEqual(sum(edge.dimensions for edge in into_middle), 2)
        self.assertEqual(sum(edge.dimensions for edge in out_middle), 2)
        self.assertEqual(len(out_middle), 2)

    def test_modal_crossing_preserves_identity_not_frequency_order(self):
        e1 = np.array([1.0, 0.0], dtype=complex)
        e2 = np.array([0.0, 1.0], dtype=complex)
        left = [fake_mode(0, e1, 0.90), fake_mode(0, e2, 1.10)]
        right = [fake_mode(1, e2, 0.96), fake_mode(1, e1, 1.04)]

        graph = self._assemble_from_modes([left, right])
        pairs = {(edge.left_event, edge.right_event) for edge in graph.edges}
        self.assertEqual(pairs, {(0, 1), (1, 0)})

    def test_missing_modal_direction_marks_pair_incomplete(self):
        e1 = np.array([1.0, 0.0, 0.0], dtype=complex)
        e2 = np.array([0.0, 1.0, 0.0], dtype=complex)
        left = [fake_mode(0, np.column_stack([e1, e2]), 1.0)]
        right = [fake_mode(1, e1, 1.01)]

        graph = self._assemble_from_modes([left, right])
        self.assertFalse(graph.complete_under_policy)
        self.assertFalse(graph.pairs[0].complete_bidirectionally)
        self.assertEqual(graph.pairs[0].transport.source_unmatched, (1,))


if __name__ == "__main__":
    unittest.main()
