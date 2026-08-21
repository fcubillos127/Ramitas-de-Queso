"""Adaptive reciprocal-truncation certification for local MST spectra.

The routine evaluates an increasing schedule of n_suma values and stops only
after a stable regime has been observed *and* confirmed by additional
refinement steps.  This is more conservative than stopping at the first pair
of small frequency differences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from convergence_certified import StepConvergence, compare_consecutive_spectra
from rootfinder_certified import RootCandidate, find_roots_at_k


@dataclass(frozen=True)
class AdaptiveNsumResult:
    k: float
    evaluated_n_values: tuple[int, ...]
    root_sets: tuple[tuple[RootCandidate, ...], ...]
    steps: tuple[StepConvergence, ...]
    converged: bool
    recommended_n_suma: int | None
    certification_n_suma: int | None
    stable_steps_required: int
    confirmation_steps_required: int
    final_roots: tuple[RootCandidate, ...]


def certify_roots_adaptive_nsum(
    red,
    k: float,
    C_l0: float,
    *,
    n_schedule: Iterable[int] = (8, 12, 20, 30, 40, 60, 80),
    stable_steps: int = 2,
    confirmation_steps: int = 1,
    frequency_atol_norm: float = 5e-5,
    frequency_rtol: float = 5e-5,
    residual_tol: float = 1e-6,
    max_match_delta_norm: float = 2e-2,
    finder_kwargs: dict | None = None,
    finder: Callable | None = None,
) -> AdaptiveNsumResult:
    """Adaptively certify a local spectrum against reciprocal truncation.

    Example with ``stable_steps=2`` and ``confirmation_steps=1``:

    * two consecutive converged refinement transitions establish a candidate
      minimum n_suma;
    * one further converged transition confirms that candidate;
    * any failed transition resets the stable run and discards the candidate.

    ``recommended_n_suma`` is the minimum truncation at which the final stable
    regime first satisfied ``stable_steps``.  ``certification_n_suma`` is the
    higher truncation actually evaluated to confirm it.  Frequencies in
    ``final_roots`` come from the certification truncation and are therefore
    the most accurate values computed in the adaptive run.
    """
    schedule = tuple(sorted({int(n) for n in n_schedule}))
    if len(schedule) < 2 or schedule[0] < 1:
        raise ValueError("n_schedule must contain at least two positive integers")
    stable_steps = max(1, int(stable_steps))
    confirmation_steps = max(0, int(confirmation_steps))
    required_run = stable_steps + confirmation_steps

    kwargs = dict(finder_kwargs or {})
    kwargs.setdefault("sigma_accept", residual_tol)
    finder_fn = finder or find_roots_at_k

    original_n = int(getattr(red, "n_suma", schedule[0]))
    evaluated: list[int] = []
    spectra: list[tuple[RootCandidate, ...]] = []
    steps: list[StepConvergence] = []

    run = 0
    candidate_n: int | None = None
    converged = False
    certification_n: int | None = None

    try:
        for n in schedule:
            red.n_suma = int(n)
            roots = tuple(finder_fn(red, float(k), C_l0, **kwargs))
            evaluated.append(int(n))
            spectra.append(roots)

            if len(spectra) == 1:
                continue

            step = compare_consecutive_spectra(
                evaluated[-2],
                spectra[-2],
                evaluated[-1],
                spectra[-1],
                frequency_atol_norm=frequency_atol_norm,
                frequency_rtol=frequency_rtol,
                residual_tol=residual_tol,
                max_match_delta_norm=max_match_delta_norm,
                multiplicity_penalty_norm=0.0,
            )
            steps.append(step)

            if step.converged:
                run += 1
                if run == stable_steps:
                    candidate_n = int(step.n_current)
                if run >= required_run:
                    converged = True
                    certification_n = int(step.n_current)
                    break
            else:
                run = 0
                candidate_n = None
    finally:
        red.n_suma = original_n

    if not converged:
        candidate_n = None
        certification_n = None

    return AdaptiveNsumResult(
        k=float(k),
        evaluated_n_values=tuple(evaluated),
        root_sets=tuple(spectra),
        steps=tuple(steps),
        converged=bool(converged),
        recommended_n_suma=candidate_n,
        certification_n_suma=certification_n,
        stable_steps_required=stable_steps,
        confirmation_steps_required=confirmation_steps,
        final_roots=spectra[-1] if spectra else (),
    )
