"""Targeted root discovery without relying on uniform-grid hits.

This module is used only after an independent diagnostic has identified a
frequency window that should contain missing modal descendants.  The window is
searched recursively with bounded minimisation of the certified singular-value
residual.  Rejected broad intervals are subdivided; certified minima are
recorded and the remaining left/right intervals are searched as well.

Every returned MST root is still validated by the dual-SVD criterion from
``rootfinder_certified``.  The purpose here is discovery completeness inside a
small trusted continuation window, not a replacement for the cheap global
scan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

from rootfinder_certified import RootCandidate, secular_diagnostics


@dataclass(frozen=True)
class RecursiveMinimum:
    x: float
    value: float
    depth: int
    interval: tuple[float, float]


def recursive_bounded_minima(
    objective: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    accept_value: float,
    xatol: float = 1e-10,
    max_depth: int = 5,
    exclusion_radius: float = 1e-5,
) -> tuple[RecursiveMinimum, ...]:
    """Find narrow accepted minima by recursively subdividing rejected windows.

    ``minimize_scalar(method='bounded')`` is not a global optimiser on a
    multimodal interval.  Therefore a rejected interval is never trusted: it
    is split in two until ``max_depth``.  When an accepted minimum is found,
    its small exclusion neighbourhood is removed and both remaining sides are
    searched.  Duplicate minima from overlapping recursion paths are collapsed.
    """
    lo = float(lo)
    hi = float(hi)
    if not hi > lo:
        raise ValueError("hi must exceed lo")
    max_depth = max(0, int(max_depth))
    exclusion_radius = max(float(exclusion_radius), 4.0 * float(xatol))

    found: list[RecursiveMinimum] = []
    stack: list[tuple[float, float, int]] = [(lo, hi, 0)]

    def finite_objective(x):
        value = float(objective(float(x)))
        return value if np.isfinite(value) else 1e100

    while stack:
        left, right, depth = stack.pop()
        if not right > left:
            continue
        try:
            result = minimize_scalar(
                finite_objective,
                bounds=(left, right),
                method="bounded",
                options={"xatol": float(xatol), "maxiter": 200},
            )
        except (ValueError, FloatingPointError):
            result = None

        accepted = (
            result is not None
            and result.success
            and np.isfinite(result.fun)
            and float(result.fun) <= float(accept_value)
        )

        if accepted:
            x = float(result.x)
            found.append(
                RecursiveMinimum(
                    x=x,
                    value=float(result.fun),
                    depth=int(depth),
                    interval=(float(left), float(right)),
                )
            )
            if depth < max_depth:
                l_hi = x - exclusion_radius
                r_lo = x + exclusion_radius
                if l_hi > left:
                    stack.append((left, l_hi, depth + 1))
                if right > r_lo:
                    stack.append((r_lo, right, depth + 1))
            continue

        if depth < max_depth:
            mid = 0.5 * (left + right)
            stack.append((left, mid, depth + 1))
            stack.append((mid, right, depth + 1))

    found.sort(key=lambda item: item.x)
    deduped: list[RecursiveMinimum] = []
    for item in found:
        if not deduped or abs(item.x - deduped[-1].x) > 2.0 * exclusion_radius:
            deduped.append(item)
        elif item.value < deduped[-1].value:
            deduped[-1] = item
    return tuple(deduped)


def find_roots_targeted(
    red,
    k: float,
    C_l0: float,
    *,
    w_norm_min: float,
    w_norm_max: float,
    sigma_accept: float = 1e-6,
    multiplicity_tol: float = 1e-5,
    balance_passes: int = 8,
    refine_xatol_norm: float = 1e-10,
    dedup_tol_norm: float = 5e-6,
    max_depth: int = 5,
) -> tuple[RootCandidate, ...]:
    """Recursively discover and certify roots inside one normalized window."""
    scale = 2.0 * np.pi * float(C_l0) / float(red.a)

    def objective(x_norm):
        residual = secular_diagnostics(
            red,
            float(x_norm) * scale,
            float(k),
            imag=0.0,
            multiplicity_tol=multiplicity_tol,
            balance_passes=balance_passes,
        )[0]
        return float(residual) if np.isfinite(residual) else 1e100

    minima = recursive_bounded_minima(
        objective,
        float(w_norm_min),
        float(w_norm_max),
        accept_value=float(sigma_accept),
        xatol=float(refine_xatol_norm),
        max_depth=int(max_depth),
        exclusion_radius=max(2.5 * float(dedup_tol_norm), 1e-7),
    )

    roots: list[RootCandidate] = []
    for minimum in minima:
        root_norm = float(minimum.x)
        root_omega = root_norm * scale
        residual, multiplicity, det_abs, diag = secular_diagnostics(
            red,
            root_omega,
            float(k),
            imag=0.0,
            multiplicity_tol=multiplicity_tol,
            balance_passes=balance_passes,
        )
        if residual > float(sigma_accept):
            continue
        roots.append(
            RootCandidate(
                omega=float(root_omega),
                omega_norm=root_norm,
                sigma_min=float(residual),
                sigma_min_raw=float(diag["raw_sigma_min"]),
                sigma_min_balanced=float(diag["balanced_sigma_min"]),
                multiplicity=max(1, int(multiplicity)),
                det_abs=float(det_abs),
                source="targeted_recursive_svd",
            )
        )

    roots.sort(key=lambda root: root.omega_norm)
    deduped: list[RootCandidate] = []
    for root in roots:
        if not deduped or abs(root.omega_norm - deduped[-1].omega_norm) > float(dedup_tol_norm):
            deduped.append(root)
        elif root.sigma_min < deduped[-1].sigma_min:
            deduped[-1] = root
    return tuple(deduped)
