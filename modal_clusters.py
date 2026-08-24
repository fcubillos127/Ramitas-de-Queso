"""Gauge-invariant continuity diagnostics for groups of nearby modal events.

Individual eigenvectors are not a stable identity when two nearby simple roots
rotate rapidly into one another.  In that situation the physically meaningful
object can be the *joint invariant subspace* spanned by the nearby events.

This module is deliberately diagnostic: it does not alter certified roots and
it does not invent a branch ordering through a cluster.  It only asks whether
two selected groups span the same modal subspace to within principal-angle
criteria.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from modal_tracking import ModalSubspace, SubspaceMetrics, subspace_metrics


@dataclass(frozen=True)
class ModalCluster:
    event_indices: tuple[int, ...]
    omega_min: float
    omega_max: float
    omega_center: float
    dimension: int
    basis: np.ndarray


@dataclass(frozen=True)
class ClusterContinuity:
    source: ModalCluster
    target: ModalCluster
    metrics: SubspaceMetrics
    center_shift: float
    certified: bool


def build_modal_cluster(
    modes: Sequence[ModalSubspace],
    event_indices: Sequence[int],
    *,
    rank_rtol: float = 1e-10,
) -> ModalCluster:
    """Build an orthonormal basis for the union of selected spectral events."""
    modes = tuple(modes)
    indices = tuple(int(i) for i in event_indices)
    if not indices:
        raise ValueError("a modal cluster requires at least one event")
    if len(set(indices)) != len(indices):
        raise ValueError("event_indices must be unique")
    if min(indices) < 0 or max(indices) >= len(modes):
        raise IndexError("cluster event index out of range")

    selected = [modes[i] for i in indices]
    ambient = selected[0].basis.shape[0]
    if any(mode.basis.shape[0] != ambient for mode in selected):
        raise ValueError("cluster events must use the same angular basis")

    expected_dimension = int(sum(mode.multiplicity for mode in selected))
    X = np.concatenate([np.asarray(mode.basis, dtype=complex) for mode in selected], axis=1)
    U, singular, _Vh = np.linalg.svd(X, full_matrices=False)
    if singular.size == 0 or singular[0] <= np.finfo(float).tiny:
        raise np.linalg.LinAlgError("modal cluster is numerically empty")
    rank = int(np.count_nonzero(singular > float(rank_rtol) * singular[0]))
    if rank != expected_dimension:
        raise np.linalg.LinAlgError(
            f"cluster lost modal rank: expected {expected_dimension}, obtained {rank}"
        )

    frequencies = np.asarray([mode.omega_norm for mode in selected], dtype=float)
    weights = np.asarray([mode.multiplicity for mode in selected], dtype=float)
    center = float(np.average(frequencies, weights=weights))
    return ModalCluster(
        event_indices=indices,
        omega_min=float(np.min(frequencies)),
        omega_max=float(np.max(frequencies)),
        omega_center=center,
        dimension=expected_dimension,
        basis=np.asarray(U[:, :rank], dtype=complex),
    )


def certify_cluster_continuity(
    source_modes: Sequence[ModalSubspace],
    target_modes: Sequence[ModalSubspace],
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    *,
    min_principal_cosine: float = 0.65,
    max_center_shift_norm: float = 0.10,
) -> ClusterContinuity:
    """Certify continuity of two equal-dimensional joint modal subspaces.

    This certificate deliberately says nothing about a unique one-to-one branch
    assignment *inside* the cluster.  It only establishes that the same modal
    subspace is transported across the k step.
    """
    source = build_modal_cluster(source_modes, source_indices)
    target = build_modal_cluster(target_modes, target_indices)
    if source.dimension != target.dimension:
        raise ValueError("source and target clusters must have equal modal dimension")

    metrics = subspace_metrics(source.basis, target.basis)
    center_shift = abs(float(target.omega_center) - float(source.omega_center))
    certified = bool(
        metrics.min_principal_cosine >= float(min_principal_cosine)
        and center_shift <= float(max_center_shift_norm)
    )
    return ClusterContinuity(
        source=source,
        target=target,
        metrics=metrics,
        center_shift=float(center_shift),
        certified=certified,
    )
