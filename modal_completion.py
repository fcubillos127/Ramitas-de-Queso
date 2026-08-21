"""Use modal continuity to detect and repair incomplete fixed-k spectra.

A certified root can still be *missed* by a coarse global frequency scan. This
is particularly likely when an m-fold degeneracy at k_i splits into several
nearby simple roots at k_{i+1}. The local roots that are found remain valid;
the problem is completeness, not certification.

This module treats the modal subspace at k_i as a dimension-counting diagnostic.
If the neighbouring roots within a continuation window do not span enough of
that subspace, a targeted recursive SVD search is triggered. Every newly found
candidate is still subjected to the same dual-SVD certificate used by
``rootfinder_certified``.

The algorithm is deliberately a reference completion layer. It does not yet
assign global band labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from modal_tracking import ModalSubspace, build_modal_spectrum, subspace_metrics
from rootfinder_certified import RootCandidate
from targeted_rootfinder import find_roots_targeted


@dataclass(frozen=True)
class DescendantCoverage:
    source_index: int
    source_multiplicity: int
    candidate_indices: tuple[int, ...]
    candidate_capacity: int
    union_dimension: int
    coverage: float
    principal_cosines: np.ndarray
    needs_refinement: bool


@dataclass(frozen=True)
class CompletionRound:
    round_index: int
    diagnostics_before: tuple[DescendantCoverage, ...]
    searched_windows: tuple[tuple[float, float], ...]
    roots_added: int


@dataclass(frozen=True)
class SpectrumCompletion:
    roots: tuple[RootCandidate, ...]
    modes: tuple[ModalSubspace, ...]
    diagnostics: tuple[DescendantCoverage, ...]
    rounds: tuple[CompletionRound, ...]
    complete_under_policy: bool


def orthonormal_union(
    modes: Sequence[ModalSubspace],
    *,
    rank_rtol: float = 1e-10,
) -> np.ndarray:
    """Return an orthonormal basis for the union of several modal subspaces."""
    modes = tuple(modes)
    if not modes:
        raise ValueError("at least one modal subspace is required")
    ambient = modes[0].basis.shape[0]
    if any(mode.basis.shape[0] != ambient for mode in modes):
        raise ValueError("all modal subspaces must use the same ambient basis")

    X = np.concatenate([np.asarray(mode.basis, dtype=complex) for mode in modes], axis=1)
    U, singular, _Vh = np.linalg.svd(X, full_matrices=False)
    if singular.size == 0 or singular[0] <= np.finfo(float).tiny:
        raise np.linalg.LinAlgError("candidate modal union is numerically empty")
    rank = int(np.count_nonzero(singular > float(rank_rtol) * singular[0]))
    rank = max(1, rank)
    return U[:, :rank]


def diagnose_descendant_coverage(
    previous: Sequence[ModalSubspace],
    current: Sequence[ModalSubspace],
    *,
    max_delta_omega_norm: float = 0.06,
    min_pair_affinity: float = 0.15,
    coverage_floor: float = 0.60,
    rank_rtol: float = 1e-10,
) -> tuple[DescendantCoverage, ...]:
    """Check whether each previous modal event is represented at the next k.

    Candidate events are first gated by frequency and a deliberately weak modal
    affinity. Their *union subspace* is then compared with the source event,
    which avoids double counting nearly collinear candidate roots.

    Refinement is requested if either

    * the candidate union has dimension smaller than the source multiplicity,
      or
    * the union captures less than ``coverage_floor`` of the source subspace.
    """
    diagnostics: list[DescendantCoverage] = []
    current = tuple(current)

    for i, source in enumerate(previous):
        candidates: list[int] = []
        for j, target in enumerate(current):
            delta = abs(float(target.omega_norm) - float(source.omega_norm))
            if delta > float(max_delta_omega_norm):
                continue
            metrics = subspace_metrics(source, target)
            if metrics.affinity >= float(min_pair_affinity):
                candidates.append(j)

        capacity = int(sum(current[j].multiplicity for j in candidates))
        if not candidates:
            diagnostics.append(
                DescendantCoverage(
                    source_index=int(i),
                    source_multiplicity=int(source.multiplicity),
                    candidate_indices=(),
                    candidate_capacity=0,
                    union_dimension=0,
                    coverage=0.0,
                    principal_cosines=np.array([], dtype=float),
                    needs_refinement=True,
                )
            )
            continue

        union = orthonormal_union([current[j] for j in candidates], rank_rtol=rank_rtol)
        metrics = subspace_metrics(source.basis, union)
        union_dim = int(union.shape[1])
        needs = (
            union_dim < int(source.multiplicity)
            or float(metrics.coverage_first) < float(coverage_floor)
        )
        diagnostics.append(
            DescendantCoverage(
                source_index=int(i),
                source_multiplicity=int(source.multiplicity),
                candidate_indices=tuple(candidates),
                candidate_capacity=capacity,
                union_dimension=union_dim,
                coverage=float(metrics.coverage_first),
                principal_cosines=np.asarray(metrics.principal_cosines, dtype=float),
                needs_refinement=bool(needs),
            )
        )

    return tuple(diagnostics)


def merge_root_candidates(
    existing: Sequence[RootCandidate],
    additions: Sequence[RootCandidate],
    *,
    tol_norm: float = 5e-6,
) -> tuple[RootCandidate, ...]:
    """Merge certified root sets without turning a close doublet into one root."""
    ordered = sorted(tuple(existing) + tuple(additions), key=lambda root: root.omega_norm)
    merged: list[RootCandidate] = []
    for root in ordered:
        if not merged or abs(root.omega_norm - merged[-1].omega_norm) > float(tol_norm):
            merged.append(root)
            continue
        # Same numerical root found by two discovery routes: keep the stronger
        # certificate while preserving the fact that this is one spectral event.
        if root.sigma_min < merged[-1].sigma_min:
            merged[-1] = root
    return tuple(merged)


def complete_spectrum_from_previous(
    red,
    k_current: float,
    C_l0: float,
    previous_modes: Sequence[ModalSubspace],
    current_roots: Sequence[RootCandidate],
    *,
    search_half_width_norm: float = 0.06,
    targeted_max_depth: int = 5,
    max_rounds: int = 2,
    min_pair_affinity: float = 0.15,
    coverage_floor: float = 0.60,
    w_norm_min_global: float = 1e-3,
    w_norm_max_global: float = 1.25,
    dedup_tol_norm: float = 5e-6,
    finder_kwargs: dict | None = None,
) -> SpectrumCompletion:
    """Repair an incomplete current spectrum using targeted certified searches.

    A flagged source event opens a narrow frequency window centred on its
    previous certified frequency. The window is searched recursively with the
    dual-SVD residual rather than sampled by another uniform grid. This makes
    the completion step sensitive to narrow split roots without paying the cost
    of a globally dense frequency mesh.
    """
    roots = tuple(sorted(current_roots, key=lambda root: root.omega_norm))
    rounds: list[CompletionRound] = []
    kwargs = dict(finder_kwargs or {})
    kwargs.setdefault("sigma_accept", 1e-6)
    kwargs.setdefault("multiplicity_tol", 1e-5)
    kwargs.setdefault("dedup_tol_norm", dedup_tol_norm)
    kwargs.setdefault("max_depth", int(targeted_max_depth))

    diagnostics: tuple[DescendantCoverage, ...] = ()
    modes: tuple[ModalSubspace, ...] = ()

    for round_index in range(max(0, int(max_rounds)) + 1):
        modes = build_modal_spectrum(red, float(k_current), roots)
        diagnostics = diagnose_descendant_coverage(
            previous_modes,
            modes,
            max_delta_omega_norm=search_half_width_norm,
            min_pair_affinity=min_pair_affinity,
            coverage_floor=coverage_floor,
        )
        flagged = [diag for diag in diagnostics if diag.needs_refinement]
        if not flagged or round_index >= int(max_rounds):
            break

        windows = []
        additions: list[RootCandidate] = []
        for diag in flagged:
            source = previous_modes[diag.source_index]
            lo = max(float(w_norm_min_global), source.omega_norm - float(search_half_width_norm))
            hi = min(float(w_norm_max_global), source.omega_norm + float(search_half_width_norm))
            if hi <= lo:
                continue
            windows.append((float(lo), float(hi)))
            found = find_roots_targeted(
                red,
                float(k_current),
                C_l0,
                w_norm_min=lo,
                w_norm_max=hi,
                **kwargs,
            )
            additions.extend(found)

        before = len(roots)
        roots = merge_root_candidates(roots, additions, tol_norm=dedup_tol_norm)
        rounds.append(
            CompletionRound(
                round_index=int(round_index + 1),
                diagnostics_before=diagnostics,
                searched_windows=tuple(windows),
                roots_added=max(0, len(roots) - before),
            )
        )

        if len(roots) == before:
            # Repeating the same targeted windows cannot add information unless
            # the continuation window or recursive depth is deliberately changed.
            break

    modes = build_modal_spectrum(red, float(k_current), roots)
    diagnostics = diagnose_descendant_coverage(
        previous_modes,
        modes,
        max_delta_omega_norm=search_half_width_norm,
        min_pair_affinity=min_pair_affinity,
        coverage_floor=coverage_floor,
    )
    complete = all(not diag.needs_refinement for diag in diagnostics)

    return SpectrumCompletion(
        roots=roots,
        modes=modes,
        diagnostics=diagnostics,
        rounds=tuple(rounds),
        complete_under_policy=bool(complete),
    )
