"""Bidirectionally completed modal event graph along a Bloch path.

The local certified solver returns spectral *events* at each k.  An event may
have geometric multiplicity larger than one, so it must not be expanded into
arbitrary frequency-sorted columns at a degeneracy.

This module builds a layered graph instead:

* each layer is one Bloch point;
* each node is one certified spectral event with its modal null subspace;
* each edge carries the number of modal dimensions transported between two
  adjacent events;
* local spectra are completed in both path directions before edges are frozen.

The bidirectional completion is important.  A purely forward continuation can
recover a missed descendant but cannot diagnose that the source layer itself
was incomplete.  Alternating forward/backward sweeps lets either neighbour
request a targeted certified search until the root sets stabilise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from modal_completion import (
    DescendantCoverage,
    complete_spectrum_from_previous,
    diagnose_descendant_coverage,
)
from modal_tracking import ModalSubspace, build_modal_spectrum
from modal_transport import AdjacentTransport, assign_modal_transport
from rootfinder_certified import RootCandidate, find_roots_at_k


@dataclass(frozen=True)
class PathLayer:
    index: int
    k: float
    roots: tuple[RootCandidate, ...]
    modes: tuple[ModalSubspace, ...]


@dataclass(frozen=True)
class PairContinuity:
    left_index: int
    right_index: int
    forward_diagnostics: tuple[DescendantCoverage, ...]
    backward_diagnostics: tuple[DescendantCoverage, ...]
    transport: AdjacentTransport
    complete_bidirectionally: bool


@dataclass(frozen=True)
class EventEdge:
    left_layer: int
    left_event: int
    right_layer: int
    right_event: int
    dimensions: int
    delta_omega_norm: float
    modal_score: float


@dataclass(frozen=True)
class ModalPathGraph:
    layers: tuple[PathLayer, ...]
    pairs: tuple[PairContinuity, ...]
    edges: tuple[EventEdge, ...]
    completion_sweeps: int
    roots_added: int
    stabilised: bool
    complete_under_policy: bool


def _sorted_roots(roots: Sequence[RootCandidate]) -> tuple[RootCandidate, ...]:
    return tuple(sorted(roots, key=lambda root: float(root.omega_norm)))


def _count_new_roots(before: Sequence[RootCandidate], after: Sequence[RootCandidate]) -> int:
    return max(0, len(after) - len(before))


def complete_root_sets_bidirectionally(
    red,
    k_values: Sequence[float],
    C_l0: float,
    root_sets: Sequence[Sequence[RootCandidate]],
    *,
    max_sweeps: int = 3,
    search_half_width_norm: float = 0.06,
    targeted_max_depth: int = 5,
    completion_max_rounds: int = 2,
    min_pair_affinity: float = 0.15,
    coverage_floor: float = 0.60,
    w_norm_min_global: float = 1e-3,
    w_norm_max_global: float = 1.25,
    dedup_tol_norm: float = 5e-6,
    finder_kwargs: dict | None = None,
) -> tuple[tuple[tuple[RootCandidate, ...], ...], int, int, bool]:
    """Alternate forward/backward modal-completion sweeps until stabilisation."""
    k_values = tuple(float(k) for k in k_values)
    roots = [_sorted_roots(items) for items in root_sets]
    if len(k_values) != len(roots):
        raise ValueError("k_values and root_sets must have the same length")
    if len(k_values) < 2:
        raise ValueError("at least two k points are required")

    total_added = 0
    sweeps_done = 0
    stabilised = False

    completion_kwargs = dict(
        search_half_width_norm=float(search_half_width_norm),
        targeted_max_depth=int(targeted_max_depth),
        max_rounds=int(completion_max_rounds),
        min_pair_affinity=float(min_pair_affinity),
        coverage_floor=float(coverage_floor),
        w_norm_min_global=float(w_norm_min_global),
        w_norm_max_global=float(w_norm_max_global),
        dedup_tol_norm=float(dedup_tol_norm),
        finder_kwargs=dict(finder_kwargs or {}),
    )

    for sweep in range(max(1, int(max_sweeps))):
        added_this_sweep = 0

        # Forward: layer i requests descendants in i+1.
        for i in range(len(k_values) - 1):
            source_modes = build_modal_spectrum(red, k_values[i], roots[i])
            before = roots[i + 1]
            result = complete_spectrum_from_previous(
                red,
                k_values[i + 1],
                C_l0,
                source_modes,
                before,
                **completion_kwargs,
            )
            roots[i + 1] = _sorted_roots(result.roots)
            added_this_sweep += _count_new_roots(before, roots[i + 1])

        # Backward: layer i requests ancestors in i-1 using exactly the same
        # subspace-coverage logic, only with reversed path orientation.
        for i in range(len(k_values) - 1, 0, -1):
            source_modes = build_modal_spectrum(red, k_values[i], roots[i])
            before = roots[i - 1]
            result = complete_spectrum_from_previous(
                red,
                k_values[i - 1],
                C_l0,
                source_modes,
                before,
                **completion_kwargs,
            )
            roots[i - 1] = _sorted_roots(result.roots)
            added_this_sweep += _count_new_roots(before, roots[i - 1])

        sweeps_done = sweep + 1
        total_added += added_this_sweep
        if added_this_sweep == 0:
            stabilised = True
            break

    return tuple(roots), sweeps_done, total_added, stabilised


def assemble_modal_path_graph(
    red,
    k_values: Sequence[float],
    root_sets: Sequence[Sequence[RootCandidate]],
    *,
    search_half_width_norm: float = 0.06,
    min_pair_affinity: float = 0.15,
    coverage_floor: float = 0.60,
    transport_max_delta_omega_norm: float = 0.08,
    min_principal_cosine: float = 0.65,
    frequency_weight: float = 0.05,
    completion_sweeps: int = 0,
    roots_added: int = 0,
    stabilised: bool = True,
) -> ModalPathGraph:
    """Freeze completed local spectra into a layered modal-event graph."""
    k_values = tuple(float(k) for k in k_values)
    roots = tuple(_sorted_roots(items) for items in root_sets)
    if len(k_values) != len(roots):
        raise ValueError("k_values and root_sets must have the same length")

    layers: list[PathLayer] = []
    for i, (k, local_roots) in enumerate(zip(k_values, roots)):
        modes = build_modal_spectrum(red, k, local_roots)
        layers.append(PathLayer(index=i, k=k, roots=local_roots, modes=modes))

    pairs: list[PairContinuity] = []
    edges: list[EventEdge] = []
    all_complete = True

    for i in range(len(layers) - 1):
        left = layers[i]
        right = layers[i + 1]
        forward = diagnose_descendant_coverage(
            left.modes,
            right.modes,
            max_delta_omega_norm=float(search_half_width_norm),
            min_pair_affinity=float(min_pair_affinity),
            coverage_floor=float(coverage_floor),
        )
        backward = diagnose_descendant_coverage(
            right.modes,
            left.modes,
            max_delta_omega_norm=float(search_half_width_norm),
            min_pair_affinity=float(min_pair_affinity),
            coverage_floor=float(coverage_floor),
        )
        pair_complete = (
            all(not item.needs_refinement for item in forward)
            and all(not item.needs_refinement for item in backward)
        )
        all_complete = all_complete and pair_complete

        transport = assign_modal_transport(
            left.modes,
            right.modes,
            max_delta_omega_norm=float(transport_max_delta_omega_norm),
            min_principal_cosine=float(min_principal_cosine),
            frequency_weight=float(frequency_weight),
        )
        pairs.append(
            PairContinuity(
                left_index=i,
                right_index=i + 1,
                forward_diagnostics=forward,
                backward_diagnostics=backward,
                transport=transport,
                complete_bidirectionally=bool(pair_complete),
            )
        )
        for edge in transport.edges:
            edges.append(
                EventEdge(
                    left_layer=i,
                    left_event=edge.source_index,
                    right_layer=i + 1,
                    right_event=edge.target_index,
                    dimensions=edge.dimensions,
                    delta_omega_norm=edge.delta_omega_norm,
                    modal_score=edge.modal_score,
                )
            )

    return ModalPathGraph(
        layers=tuple(layers),
        pairs=tuple(pairs),
        edges=tuple(edges),
        completion_sweeps=int(completion_sweeps),
        roots_added=int(roots_added),
        stabilised=bool(stabilised),
        complete_under_policy=bool(stabilised and all_complete),
    )


def build_modal_path_graph(
    red,
    k_values: Sequence[float],
    C_l0: float,
    *,
    w_norm_min: float = 1e-3,
    w_norm_max: float = 1.25,
    ngrid: int = 140,
    max_sweeps: int = 3,
    search_half_width_norm: float = 0.06,
    targeted_max_depth: int = 5,
    completion_max_rounds: int = 2,
    min_pair_affinity: float = 0.15,
    coverage_floor: float = 0.60,
    transport_max_delta_omega_norm: float = 0.08,
    min_principal_cosine: float = 0.65,
    frequency_weight: float = 0.05,
    finder_kwargs: dict | None = None,
) -> ModalPathGraph:
    """Discover, bidirectionally complete, and connect a Bloch-path spectrum."""
    k_values = tuple(float(k) for k in k_values)
    kwargs = dict(finder_kwargs or {})
    kwargs.setdefault("scan_eta_norm", 1e-6)
    kwargs.setdefault("sigma_accept", 1e-6)
    kwargs.setdefault("multiplicity_tol", 1e-5)

    initial = []
    for k in k_values:
        initial.append(
            tuple(
                find_roots_at_k(
                    red,
                    k,
                    C_l0,
                    w_norm_min=float(w_norm_min),
                    w_norm_max=float(w_norm_max),
                    ngrid=int(ngrid),
                    **kwargs,
                )
            )
        )

    completed, sweeps, added, stabilised = complete_root_sets_bidirectionally(
        red,
        k_values,
        C_l0,
        initial,
        max_sweeps=max_sweeps,
        search_half_width_norm=search_half_width_norm,
        targeted_max_depth=targeted_max_depth,
        completion_max_rounds=completion_max_rounds,
        min_pair_affinity=min_pair_affinity,
        coverage_floor=coverage_floor,
        w_norm_min_global=w_norm_min,
        w_norm_max_global=w_norm_max,
        finder_kwargs=kwargs,
    )

    return assemble_modal_path_graph(
        red,
        k_values,
        completed,
        search_half_width_norm=search_half_width_norm,
        min_pair_affinity=min_pair_affinity,
        coverage_floor=coverage_floor,
        transport_max_delta_omega_norm=transport_max_delta_omega_norm,
        min_principal_cosine=min_principal_cosine,
        frequency_weight=frequency_weight,
        completion_sweeps=sweeps,
        roots_added=added,
        stabilised=stabilised,
    )
