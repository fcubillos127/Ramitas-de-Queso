"""Capacity-aware modal transport between adjacent certified spectra.

A spectral event has capacity equal to its geometric multiplicity.  Simple
roots therefore carry one modal dimension, while an m-fold degeneracy carries
m.  Adjacent events are connected only when principal-angle overlap shows that
they share at least one modal direction.

The resulting assignment is a small integer transportation problem.  It is
solved globally for each pair of neighbouring k points, rather than greedily,
so competing near-crossing candidates do not depend on iteration order.

This layer assigns *segments* between adjacent k points.  It deliberately does
not split a degenerate node into arbitrary basis vectors and therefore does not
claim a unique branch identity exactly at a degeneracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from modal_tracking import ModalSubspace, SubspaceMetrics, subspace_metrics


@dataclass(frozen=True)
class TransportCandidate:
    source_index: int
    target_index: int
    capacity: int
    delta_omega_norm: float
    metrics: SubspaceMetrics
    modal_score: float
    score_per_dimension: float


@dataclass(frozen=True)
class TransportEdge:
    source_index: int
    target_index: int
    dimensions: int
    delta_omega_norm: float
    principal_cosines: np.ndarray
    modal_score: float


@dataclass(frozen=True)
class AdjacentTransport:
    edges: tuple[TransportEdge, ...]
    source_used: tuple[int, ...]
    target_used: tuple[int, ...]
    source_unmatched: tuple[int, ...]
    target_unmatched: tuple[int, ...]
    matched_dimensions: int
    total_source_dimensions: int
    total_target_dimensions: int


def transport_candidates(
    source: Sequence[ModalSubspace],
    target: Sequence[ModalSubspace],
    *,
    max_delta_omega_norm: float = 0.08,
    min_principal_cosine: float = 0.65,
    frequency_weight: float = 0.05,
) -> tuple[TransportCandidate, ...]:
    """Build event-to-event transport candidates from principal-angle overlap.

    Edge capacity is the number of principal directions whose cosine exceeds
    ``min_principal_cosine``.  Hence two doublets with only one common modal
    direction cannot transport two dimensions through the same edge.
    """
    candidates: list[TransportCandidate] = []
    width = max(float(max_delta_omega_norm), np.finfo(float).eps)

    for i, left in enumerate(source):
        for j, right in enumerate(target):
            delta = abs(float(right.omega_norm) - float(left.omega_norm))
            if delta > float(max_delta_omega_norm):
                continue

            metrics = subspace_metrics(left, right)
            accepted = metrics.principal_cosines[
                metrics.principal_cosines >= float(min_principal_cosine)
            ]
            capacity = int(len(accepted))
            if capacity < 1:
                continue

            modal_score = float(np.mean(accepted ** 2))
            frequency_penalty = float(frequency_weight) * (delta / width) ** 2
            score = modal_score - frequency_penalty
            if score <= 0.0:
                continue

            candidates.append(
                TransportCandidate(
                    source_index=int(i),
                    target_index=int(j),
                    capacity=capacity,
                    delta_omega_norm=float(delta),
                    metrics=metrics,
                    modal_score=modal_score,
                    score_per_dimension=float(score),
                )
            )

    return tuple(candidates)


def assign_modal_transport(
    source: Sequence[ModalSubspace],
    target: Sequence[ModalSubspace],
    *,
    max_delta_omega_norm: float = 0.08,
    min_principal_cosine: float = 0.65,
    frequency_weight: float = 0.05,
    cardinality_bonus: float = 2.0,
) -> AdjacentTransport:
    """Globally assign modal dimensions between two neighbouring spectra.

    The optimization first favours transporting as many admissible modal
    dimensions as possible (``cardinality_bonus``), then uses overlap and a
    weak frequency-continuity penalty to resolve competition.  Unmatched
    dimensions are retained explicitly instead of being forced through a weak
    edge.
    """
    source = tuple(source)
    target = tuple(target)
    candidates = transport_candidates(
        source,
        target,
        max_delta_omega_norm=max_delta_omega_norm,
        min_principal_cosine=min_principal_cosine,
        frequency_weight=frequency_weight,
    )

    source_capacity = np.asarray([mode.multiplicity for mode in source], dtype=int)
    target_capacity = np.asarray([mode.multiplicity for mode in target], dtype=int)
    total_source = int(np.sum(source_capacity))
    total_target = int(np.sum(target_capacity))

    if not candidates:
        return AdjacentTransport(
            edges=(),
            source_used=tuple(0 for _ in source),
            target_used=tuple(0 for _ in target),
            source_unmatched=tuple(int(v) for v in source_capacity),
            target_unmatched=tuple(int(v) for v in target_capacity),
            matched_dimensions=0,
            total_source_dimensions=total_source,
            total_target_dimensions=total_target,
        )

    nvar = len(candidates)
    c = -np.asarray(
        [float(cardinality_bonus) + edge.score_per_dimension for edge in candidates],
        dtype=float,
    )

    rows = []
    ub = []
    for i, cap in enumerate(source_capacity):
        row = np.zeros(nvar, dtype=float)
        for e, candidate in enumerate(candidates):
            if candidate.source_index == i:
                row[e] = 1.0
        rows.append(row)
        ub.append(float(cap))

    for j, cap in enumerate(target_capacity):
        row = np.zeros(nvar, dtype=float)
        for e, candidate in enumerate(candidates):
            if candidate.target_index == j:
                row[e] = 1.0
        rows.append(row)
        ub.append(float(cap))

    constraints = LinearConstraint(
        np.vstack(rows),
        lb=np.zeros(len(rows), dtype=float),
        ub=np.asarray(ub, dtype=float),
    )
    bounds = Bounds(
        lb=np.zeros(nvar, dtype=float),
        ub=np.asarray([edge.capacity for edge in candidates], dtype=float),
    )

    result = milp(
        c=c,
        integrality=np.ones(nvar, dtype=int),
        bounds=bounds,
        constraints=constraints,
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"modal transport MILP failed: {result.message}")

    flow = np.rint(result.x).astype(int)
    edges: list[TransportEdge] = []
    source_used = np.zeros(len(source), dtype=int)
    target_used = np.zeros(len(target), dtype=int)

    for candidate, dimensions in zip(candidates, flow):
        dimensions = int(dimensions)
        if dimensions <= 0:
            continue
        source_used[candidate.source_index] += dimensions
        target_used[candidate.target_index] += dimensions
        edges.append(
            TransportEdge(
                source_index=candidate.source_index,
                target_index=candidate.target_index,
                dimensions=dimensions,
                delta_omega_norm=candidate.delta_omega_norm,
                principal_cosines=np.asarray(candidate.metrics.principal_cosines, dtype=float),
                modal_score=candidate.modal_score,
            )
        )

    edges.sort(key=lambda edge: (edge.source_index, edge.target_index))
    source_unmatched = source_capacity - source_used
    target_unmatched = target_capacity - target_used

    return AdjacentTransport(
        edges=tuple(edges),
        source_used=tuple(int(v) for v in source_used),
        target_used=tuple(int(v) for v in target_used),
        source_unmatched=tuple(int(v) for v in source_unmatched),
        target_unmatched=tuple(int(v) for v in target_unmatched),
        matched_dimensions=int(np.sum(flow)),
        total_source_dimensions=total_source,
        total_target_dimensions=total_target,
    )
