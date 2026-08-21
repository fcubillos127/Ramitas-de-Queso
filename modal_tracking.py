"""Modal fingerprints and subspace comparisons for certified MST roots.

The local root finder answers where A(w,k)=T(w)G0(w,k)-I loses rank.  This
module adds the information needed for band identity: the right null subspace
of A in the *original angular-harmonic coordinates*.

A diagonal equilibration is used only as a numerically stable route to the
nullspace.  If B = D_r A D_c and B y = 0, then x = D_c y belongs to the raw
nullspace of A.  We therefore map balanced right-singular vectors back through
D_c and orthonormalise them before any comparison.

For a simple root, modal identity can be measured by |q1^H q2|^2.  For a
multiple root the individual singular vectors are not unique, so comparisons
must be made between whole subspaces.  The singular values of Q1^H Q2 are the
cosines of the principal angles and are invariant under phase changes and
unitary rotations within either degenerate subspace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from rootfinder_certified import RootCandidate, secular_matrix


@dataclass(frozen=True)
class ModalSubspace:
    k: float
    omega: float
    omega_norm: float
    multiplicity: int
    basis: np.ndarray
    angular_weights: np.ndarray
    raw_subspace_residual: float
    relative_subspace_residual: float
    balanced_singular_values: np.ndarray
    root: RootCandidate


@dataclass(frozen=True)
class SubspaceMetrics:
    principal_cosines: np.ndarray
    overlap_mass: float
    affinity: float
    coverage_first: float
    coverage_second: float
    min_principal_cosine: float


@dataclass(frozen=True)
class ModalLink:
    previous_index: int
    current_index: int
    delta_omega_norm: float
    metrics: SubspaceMetrics


def _equilibrate_with_scaling(A, passes: int = 8):
    """Return D_r A D_c together with the accumulated diagonal scalings."""
    B = np.asarray(A, dtype=complex).copy()
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("A must be a square matrix")

    n = B.shape[0]
    left = np.ones(n, dtype=float)
    right = np.ones(n, dtype=float)
    tiny = np.finfo(float).tiny

    for _ in range(max(0, int(passes))):
        row_norm = np.linalg.norm(B, axis=1)
        row_scale = np.ones_like(row_norm)
        good_rows = np.isfinite(row_norm) & (row_norm > tiny)
        row_scale[good_rows] = 1.0 / np.sqrt(row_norm[good_rows])
        B = row_scale[:, None] * B
        left *= row_scale

        col_norm = np.linalg.norm(B, axis=0)
        col_scale = np.ones_like(col_norm)
        good_cols = np.isfinite(col_norm) & (col_norm > tiny)
        col_scale[good_cols] = 1.0 / np.sqrt(col_norm[good_cols])
        B = B * col_scale[None, :]
        right *= col_scale

    return B, left, right


def _orthonormal_columns(X):
    X = np.asarray(X, dtype=complex)
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError("X must contain at least one column")
    Q, R = np.linalg.qr(X, mode="reduced")
    diag = np.abs(np.diag(R))
    if np.any(~np.isfinite(diag)) or np.any(diag <= np.finfo(float).tiny):
        raise np.linalg.LinAlgError("mapped nullspace lost numerical rank")
    return Q


def extract_modal_subspace(
    red,
    k: float,
    root: RootCandidate,
    *,
    balance_passes: int = 8,
) -> ModalSubspace:
    """Extract a certified root's right nullspace in raw angular coordinates.

    The dimension is taken from ``root.multiplicity``, which has already been
    certified by the dual-SVD root finder.  The balanced SVD is used to obtain a
    stable basis, but the basis is transformed back to the original variables
    before it is returned.
    """
    multiplicity = max(1, int(root.multiplicity))
    A = np.asarray(secular_matrix(red, root.omega, float(k), imag=0.0), dtype=complex)
    if not np.all(np.isfinite(A)):
        raise ValueError("cannot extract a modal subspace from a non-finite secular matrix")
    if multiplicity > A.shape[1]:
        raise ValueError("root multiplicity exceeds secular-matrix dimension")

    B, _left_scale, right_scale = _equilibrate_with_scaling(A, passes=balance_passes)
    _U, singular_values, Vh = np.linalg.svd(B, full_matrices=False)

    # SVD is sorted from largest to smallest singular value.  The last m right
    # singular vectors span the numerical nullspace in balanced coordinates.
    Y = Vh.conj().T[:, -multiplicity:]

    # B = D_r A D_c, hence B y=0 => A(D_c y)=0.
    X = right_scale[:, None] * Y
    Q = _orthonormal_columns(X)

    AQ = A @ Q
    raw_residual = float(np.linalg.norm(AQ, ord=2))
    A_norm = float(np.linalg.norm(A, ord=2))
    relative_residual = raw_residual / max(A_norm, np.finfo(float).tiny)

    # Average angular-harmonic weight over the whole subspace.  This is basis
    # invariant within a degenerate subspace because it is the diagonal of
    # Q Q^H divided by the subspace dimension.
    angular_weights = np.sum(np.abs(Q) ** 2, axis=1) / float(multiplicity)

    return ModalSubspace(
        k=float(k),
        omega=float(root.omega),
        omega_norm=float(root.omega_norm),
        multiplicity=multiplicity,
        basis=Q,
        angular_weights=np.asarray(angular_weights, dtype=float),
        raw_subspace_residual=raw_residual,
        relative_subspace_residual=float(relative_residual),
        balanced_singular_values=np.asarray(singular_values, dtype=float),
        root=root,
    )


def subspace_metrics(first, second) -> SubspaceMetrics:
    """Gauge-invariant similarity metrics between two orthonormal subspaces.

    ``affinity`` is ||Q1^H Q2||_F^2 / min(m1,m2).  It equals one when the
    smaller subspace lies entirely inside the larger one.  ``coverage_first``
    and ``coverage_second`` retain the directional information needed to
    recognise splitting/merging of a degenerate event.
    """
    Q1 = np.asarray(first.basis if isinstance(first, ModalSubspace) else first, dtype=complex)
    Q2 = np.asarray(second.basis if isinstance(second, ModalSubspace) else second, dtype=complex)
    if Q1.ndim != 2 or Q2.ndim != 2:
        raise ValueError("subspace bases must be two-dimensional arrays")
    if Q1.shape[0] != Q2.shape[0]:
        raise ValueError("subspaces must use the same angular-harmonic basis")
    if Q1.shape[1] < 1 or Q2.shape[1] < 1:
        raise ValueError("subspaces must have positive dimension")

    overlap = Q1.conj().T @ Q2
    cosines = np.linalg.svd(overlap, compute_uv=False)
    cosines = np.clip(np.real_if_close(cosines).astype(float), 0.0, 1.0)
    mass = float(np.sum(cosines ** 2))
    m1 = int(Q1.shape[1])
    m2 = int(Q2.shape[1])

    return SubspaceMetrics(
        principal_cosines=cosines,
        overlap_mass=mass,
        affinity=mass / float(min(m1, m2)),
        coverage_first=mass / float(m1),
        coverage_second=mass / float(m2),
        min_principal_cosine=float(np.min(cosines)),
    )


def build_modal_spectrum(
    red,
    k: float,
    roots: Sequence[RootCandidate],
    *,
    balance_passes: int = 8,
) -> tuple[ModalSubspace, ...]:
    """Attach modal subspaces to a fixed-k certified root spectrum."""
    return tuple(
        extract_modal_subspace(red, float(k), root, balance_passes=balance_passes)
        for root in roots
    )


def candidate_modal_links(
    previous: Sequence[ModalSubspace],
    current: Sequence[ModalSubspace],
    *,
    max_delta_omega_norm: float = np.inf,
    min_affinity: float = 0.0,
) -> tuple[ModalLink, ...]:
    """Return all admissible modal links between adjacent local spectra.

    This function intentionally does not force a one-to-one assignment.  A
    multiplicity-two event is allowed to link strongly to two simple roots at
    the neighbouring k point; forcing a bijection at that stage would invent an
    arbitrary basis inside the degenerate subspace.
    """
    links: list[ModalLink] = []
    for i, old in enumerate(previous):
        for j, new in enumerate(current):
            delta = abs(float(new.omega_norm) - float(old.omega_norm))
            if delta > float(max_delta_omega_norm):
                continue
            metrics = subspace_metrics(old, new)
            if metrics.affinity < float(min_affinity):
                continue
            links.append(
                ModalLink(
                    previous_index=int(i),
                    current_index=int(j),
                    delta_omega_norm=float(delta),
                    metrics=metrics,
                )
            )

    links.sort(
        key=lambda link: (
            link.previous_index,
            -link.metrics.affinity,
            link.delta_omega_norm,
            link.current_index,
        )
    )
    return tuple(links)


def strongest_links_by_event(
    links: Iterable[ModalLink],
) -> dict[int, tuple[ModalLink, ...]]:
    """Group already-scored links by their source event, strongest first."""
    grouped: dict[int, list[ModalLink]] = {}
    for link in links:
        grouped.setdefault(int(link.previous_index), []).append(link)
    return {
        index: tuple(
            sorted(
                values,
                key=lambda item: (-item.metrics.affinity, item.delta_omega_norm),
            )
        )
        for index, values in grouped.items()
    }
