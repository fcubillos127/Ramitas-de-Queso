"""Convergence certification of local MST roots versus reciprocal truncation.

This module answers one question at a fixed Bloch point k:

    Has the *set of certified roots* stabilised as n_suma increases?

It deliberately does not track physical bands between different k points.  A
root here is a local spectral event returned by ``rootfinder_certified`` and
carries its geometric multiplicity.

Convergence requires, over several consecutive reciprocal truncations:

* a one-to-one match of all roots;
* stable root multiplicities;
* small normalized frequency drift;
* certified singular residuals at every compared truncation.

Transient roots at low truncation are allowed, but a spectrum is not declared
converged until they have disappeared (or stabilised) for the requested number
of consecutive refinement steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from rootfinder_certified import RootCandidate, find_roots_at_k


@dataclass(frozen=True)
class RootMatch:
    previous_index: int
    current_index: int
    delta_norm: float
    multiplicity_previous: int
    multiplicity_current: int


@dataclass(frozen=True)
class StepConvergence:
    n_previous: int
    n_current: int
    matched: tuple[RootMatch, ...]
    unmatched_previous: tuple[int, ...]
    unmatched_current: tuple[int, ...]
    root_count_previous: int
    root_count_current: int
    mode_count_previous: int
    mode_count_current: int
    max_delta_norm: float
    multiplicity_stable: bool
    residuals_ok: bool
    converged: bool


@dataclass(frozen=True)
class SpectrumConvergence:
    k: float
    n_values: tuple[int, ...]
    root_sets: tuple[tuple[RootCandidate, ...], ...]
    steps: tuple[StepConvergence, ...]
    stable_steps_required: int
    converged: bool
    recommended_n_suma: int | None
    final_n_suma: int
    final_roots: tuple[RootCandidate, ...]


def _mode_count(roots: Sequence[RootCandidate]) -> int:
    return int(sum(max(1, int(root.multiplicity)) for root in roots))


def match_root_sets(
    previous: Sequence[RootCandidate],
    current: Sequence[RootCandidate],
    *,
    max_match_delta_norm: float = 2e-2,
    multiplicity_penalty_norm: float = 5e-3,
) -> tuple[tuple[RootMatch, ...], tuple[int, ...], tuple[int, ...]]:
    """One-to-one spectral matching between two reciprocal truncations.

    Frequency proximity is the primary matching variable.  A multiplicity
    mismatch is penalised but never hidden: convergence assessment separately
    requires multiplicity equality.  Pairs farther apart than
    ``max_match_delta_norm`` are left unmatched.
    """
    previous = tuple(previous)
    current = tuple(current)
    if not previous or not current:
        return (), tuple(range(len(previous))), tuple(range(len(current)))

    large = 1e9
    cost = np.full((len(previous), len(current)), large, dtype=float)
    delta = np.full_like(cost, np.inf)

    for i, old in enumerate(previous):
        for j, new in enumerate(current):
            d = abs(float(new.omega_norm) - float(old.omega_norm))
            delta[i, j] = d
            if d <= float(max_match_delta_norm):
                dm = abs(int(new.multiplicity) - int(old.multiplicity))
                cost[i, j] = d + float(multiplicity_penalty_norm) * dm

    rows, cols = linear_sum_assignment(cost)
    accepted: list[RootMatch] = []
    used_old: set[int] = set()
    used_new: set[int] = set()

    for i, j in zip(rows, cols):
        if cost[i, j] >= large / 2:
            continue
        accepted.append(
            RootMatch(
                previous_index=int(i),
                current_index=int(j),
                delta_norm=float(delta[i, j]),
                multiplicity_previous=int(previous[i].multiplicity),
                multiplicity_current=int(current[j].multiplicity),
            )
        )
        used_old.add(int(i))
        used_new.add(int(j))

    accepted.sort(key=lambda match: previous[match.previous_index].omega_norm)
    unmatched_old = tuple(i for i in range(len(previous)) if i not in used_old)
    unmatched_new = tuple(j for j in range(len(current)) if j not in used_new)
    return tuple(accepted), unmatched_old, unmatched_new


def compare_consecutive_spectra(
    n_previous: int,
    previous: Sequence[RootCandidate],
    n_current: int,
    current: Sequence[RootCandidate],
    *,
    frequency_atol_norm: float = 5e-5,
    frequency_rtol: float = 5e-5,
    residual_tol: float = 1e-6,
    max_match_delta_norm: float = 2e-2,
    multiplicity_penalty_norm: float = 5e-3,
) -> StepConvergence:
    matches, unmatched_old, unmatched_new = match_root_sets(
        previous,
        current,
        max_match_delta_norm=max_match_delta_norm,
        multiplicity_penalty_norm=multiplicity_penalty_norm,
    )

    complete_bijection = (
        len(matches) == len(previous) == len(current)
        and not unmatched_old
        and not unmatched_new
    )

    multiplicity_stable = complete_bijection and all(
        match.multiplicity_previous == match.multiplicity_current
        for match in matches
    )

    frequency_stable = complete_bijection
    for match in matches:
        old = previous[match.previous_index]
        new = current[match.current_index]
        allowed = max(
            float(frequency_atol_norm),
            float(frequency_rtol) * max(abs(float(old.omega_norm)), abs(float(new.omega_norm))),
        )
        if match.delta_norm > allowed:
            frequency_stable = False
            break

    residuals_ok = all(
        float(root.sigma_min) <= float(residual_tol)
        for root in tuple(previous) + tuple(current)
    )

    max_delta = max((match.delta_norm for match in matches), default=np.inf)
    return StepConvergence(
        n_previous=int(n_previous),
        n_current=int(n_current),
        matched=matches,
        unmatched_previous=unmatched_old,
        unmatched_current=unmatched_new,
        root_count_previous=len(previous),
        root_count_current=len(current),
        mode_count_previous=_mode_count(previous),
        mode_count_current=_mode_count(current),
        max_delta_norm=float(max_delta),
        multiplicity_stable=bool(multiplicity_stable),
        residuals_ok=bool(residuals_ok),
        converged=bool(
            complete_bijection
            and multiplicity_stable
            and frequency_stable
            and residuals_ok
        ),
    )


def assess_root_sequence(
    n_values: Sequence[int],
    root_sets: Sequence[Sequence[RootCandidate]],
    *,
    stable_steps: int = 2,
    frequency_atol_norm: float = 5e-5,
    frequency_rtol: float = 5e-5,
    residual_tol: float = 1e-6,
    max_match_delta_norm: float = 2e-2,
    multiplicity_penalty_norm: float = 5e-3,
    k: float = np.nan,
) -> SpectrumConvergence:
    """Assess convergence of already-computed root sets.

    ``stable_steps=2`` means that two *consecutive refinement transitions*
    must pass, hence at least three reciprocal truncations are required before
    convergence can be declared.
    """
    n_values = tuple(int(n) for n in n_values)
    roots = tuple(tuple(root_set) for root_set in root_sets)
    if len(n_values) != len(roots):
        raise ValueError("n_values and root_sets must have the same length")
    if len(n_values) < 2:
        raise ValueError("at least two reciprocal truncations are required")
    if any(b <= a for a, b in zip(n_values[:-1], n_values[1:])):
        raise ValueError("n_values must be strictly increasing")
    stable_steps = max(1, int(stable_steps))

    steps: list[StepConvergence] = []
    for i in range(1, len(n_values)):
        steps.append(
            compare_consecutive_spectra(
                n_values[i - 1],
                roots[i - 1],
                n_values[i],
                roots[i],
                frequency_atol_norm=frequency_atol_norm,
                frequency_rtol=frequency_rtol,
                residual_tol=residual_tol,
                max_match_delta_norm=max_match_delta_norm,
                multiplicity_penalty_norm=multiplicity_penalty_norm,
            )
        )

    recommended = None
    run = 0
    for step in steps:
        run = run + 1 if step.converged else 0
        if run >= stable_steps and recommended is None:
            recommended = int(step.n_current)

    final_converged = (
        len(steps) >= stable_steps
        and all(step.converged for step in steps[-stable_steps:])
    )

    return SpectrumConvergence(
        k=float(k),
        n_values=n_values,
        root_sets=roots,
        steps=tuple(steps),
        stable_steps_required=stable_steps,
        converged=bool(final_converged),
        recommended_n_suma=recommended,
        final_n_suma=int(n_values[-1]),
        final_roots=roots[-1],
    )


def certify_roots_vs_nsum(
    red,
    k: float,
    C_l0: float,
    *,
    n_values: Iterable[int] = (8, 12, 16, 20, 30, 40, 60),
    stable_steps: int = 2,
    frequency_atol_norm: float = 5e-5,
    frequency_rtol: float = 5e-5,
    residual_tol: float = 1e-6,
    max_match_delta_norm: float = 2e-2,
    multiplicity_penalty_norm: float = 5e-3,
    finder_kwargs: dict | None = None,
) -> SpectrumConvergence:
    """Compute and certify the local root spectrum over increasing n_suma.

    ``red.n_suma`` is restored even if root discovery raises.  No band-tracking
    state is modified.
    """
    values = tuple(sorted({int(n) for n in n_values}))
    if len(values) < 2 or values[0] < 1:
        raise ValueError("n_values must contain at least two positive integers")

    kwargs = dict(finder_kwargs or {})
    kwargs.setdefault("sigma_accept", residual_tol)

    original_n = int(getattr(red, "n_suma", values[0]))
    root_sets: list[tuple[RootCandidate, ...]] = []
    try:
        for n in values:
            red.n_suma = int(n)
            found = find_roots_at_k(red, float(k), C_l0, **kwargs)
            root_sets.append(tuple(found))
    finally:
        red.n_suma = original_n

    return assess_root_sequence(
        values,
        root_sets,
        stable_steps=stable_steps,
        frequency_atol_norm=frequency_atol_norm,
        frequency_rtol=frequency_rtol,
        residual_tol=residual_tol,
        max_match_delta_norm=max_match_delta_norm,
        multiplicity_penalty_norm=multiplicity_penalty_norm,
        k=float(k),
    )


def convergence_table(result: SpectrumConvergence) -> list[dict]:
    """Flatten a convergence result into rows convenient for CSV/DataFrame use."""
    rows: list[dict] = []
    for n, roots in zip(result.n_values, result.root_sets):
        for index, root in enumerate(roots):
            rows.append(
                {
                    "k": float(result.k),
                    "n_suma": int(n),
                    "root_index_local": int(index),
                    "omega_norm": float(root.omega_norm),
                    "residual": float(root.sigma_min),
                    "sigma_raw": float(root.sigma_min_raw),
                    "sigma_balanced": float(root.sigma_min_balanced),
                    "multiplicity": int(root.multiplicity),
                    "source": str(root.source),
                }
            )
    return rows
