"""Reference root discovery for the certified MST secular matrix.

This module deliberately does *not* track bands between different k points.
Its only responsibility is to answer, for one fixed k:

    Which real frequencies make A(w,k) = T(w) G0(w,k) - I singular?

The historical sign-change detector is retained as one source of candidate
intervals, but it is supplemented by minima of a singular-value residual,
which can detect even-multiplicity roots that do not change the sign of det(A).

A robust certificate must survive two opposite numerical pathologies:

* raw sigma_min(A) can become tiny merely because A has huge internal dynamic
  range (notably in the low-frequency limit);
* after diagonal equilibration, a reciprocal-space pole can be dominated by a
  low-rank divergent contribution and *also* look nearly singular.

Therefore the accepted residual is deliberately conservative:

    R = max(sigma_min(A), sigma_min(D_r A D_c)).

An authentic rank loss makes both quantities vanish. Pure scaling artifacts
and reciprocal-space poles make only one of them small. Diagonal scaling is
invertible away from exactly zero rows/columns and does not alter exact rank.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class RootCandidate:
    omega: float
    omega_norm: float
    sigma_min: float          # conservative residual R=max(raw, balanced)
    sigma_min_raw: float
    sigma_min_balanced: float
    multiplicity: int
    det_abs: float
    source: str


def _frequency_scale(red, C_l0: float) -> float:
    """omega corresponding to one unit of omega*a/(2*pi*C_l0)."""
    return 2.0 * np.pi * float(C_l0) / float(red.a)


def secular_matrix(red, omega: float, k: float, *, imag: float = 0.0):
    """Return A = T G0 - I for the current Red/CertifiedRed configuration."""
    cut = int(red.cut)
    f = [float(omega), float(imag)]
    T_diag = np.asarray([red._Tn(f, n) for n in range(-cut, cut + 1)], dtype=complex)
    G = np.asarray(red.G0(f, float(k), 1, cut), dtype=complex)
    return np.diag(T_diag) @ G - np.eye(2 * cut + 1, dtype=complex)


def equilibrate_matrix(A, passes: int = 8):
    """Diagonally equilibrate rows and columns without changing matrix rank."""
    B = np.asarray(A, dtype=complex).copy()
    tiny = np.finfo(float).tiny
    for _ in range(max(0, int(passes))):
        row_norm = np.linalg.norm(B, axis=1)
        row_scale = np.ones_like(row_norm)
        good_rows = row_norm > tiny
        row_scale[good_rows] = 1.0 / np.sqrt(row_norm[good_rows])
        B = row_scale[:, None] * B

        col_norm = np.linalg.norm(B, axis=0)
        col_scale = np.ones_like(col_norm)
        good_cols = col_norm > tiny
        col_scale[good_cols] = 1.0 / np.sqrt(col_norm[good_cols])
        B = B * col_scale[None, :]
    return B


def secular_diagnostics(
    red,
    omega: float,
    k: float,
    *,
    imag: float = 0.0,
    multiplicity_tol: float = 1e-5,
    balance_passes: int = 8,
):
    """Return conservative residual, nullity estimate, |det A| and diagnostics.

    The first returned value is

        max(raw_sigma_min, balanced_sigma_min).

    Multiplicity is estimated from the equilibrated singular spectrum, but is
    meaningful only for candidates that also pass the raw singular-value test.
    """
    try:
        A = secular_matrix(red, omega, k, imag=imag)
        if not np.all(np.isfinite(A)):
            return np.inf, 0, np.inf, {
                "raw_sigma_min": np.inf,
                "balanced_sigma_min": np.inf,
                "raw_singular_values": np.array([], dtype=float),
                "balanced_singular_values": np.array([], dtype=float),
            }

        raw_singular = np.linalg.svd(A, compute_uv=False)
        raw_sigma = float(raw_singular[-1])

        balanced = equilibrate_matrix(A, passes=balance_passes)
        balanced_singular = np.linalg.svd(balanced, compute_uv=False)
        balanced_sigma = float(balanced_singular[-1])

        residual = max(raw_sigma, balanced_sigma)
        multiplicity = int(
            np.count_nonzero(balanced_singular <= float(multiplicity_tol))
        )
        try:
            det_abs = float(abs(np.linalg.det(A)))
        except (FloatingPointError, OverflowError):
            det_abs = np.inf

        return residual, multiplicity, det_abs, {
            "raw_sigma_min": raw_sigma,
            "balanced_sigma_min": balanced_sigma,
            "raw_singular_values": raw_singular,
            "balanced_singular_values": balanced_singular,
        }
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return np.inf, 0, np.inf, {
            "raw_sigma_min": np.inf,
            "balanced_sigma_min": np.inf,
            "raw_singular_values": np.array([], dtype=float),
            "balanced_singular_values": np.array([], dtype=float),
        }


def _det_real(red, omega: float, k: float, imag: float) -> float:
    try:
        A = secular_matrix(red, omega, k, imag=imag)
        value = np.linalg.det(A)
        return float(np.real(value)) if np.isfinite(value) else np.nan
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return np.nan


def _residual(red, omega: float, k: float, imag: float, balance_passes: int) -> float:
    return secular_diagnostics(
        red, omega, k, imag=imag, balance_passes=balance_passes
    )[0]


def _finite_minimizer_residual(
    red,
    omega: float,
    k: float,
    imag: float,
    balance_passes: int,
) -> float:
    """Finite objective value for Brent/bounded minimisation.

    Exact reciprocal-space poles are represented by ``+inf`` in the physical
    diagnostic.  Passing infinities directly into SciPy's parabolic step can
    trigger invalid ``inf-inf`` arithmetic.  For optimisation only, map every
    non-finite residual to a very large finite penalty.  Final certification
    still calls ``secular_diagnostics`` and therefore retains the exact pole
    rejection semantics.
    """
    value = _residual(red, omega, k, imag, balance_passes)
    return float(value) if np.isfinite(value) else 1.0e100


def _deduplicate(candidates: Iterable[RootCandidate], tol_norm: float):
    ordered = sorted(candidates, key=lambda c: c.omega_norm)
    out: list[RootCandidate] = []
    for cand in ordered:
        if not out or abs(cand.omega_norm - out[-1].omega_norm) > tol_norm:
            out.append(cand)
            continue

        old = out[-1]
        best = cand if cand.sigma_min < old.sigma_min else old
        sources = "+".join(sorted(set(old.source.split("+") + cand.source.split("+"))))
        out[-1] = replace(
            best,
            multiplicity=max(old.multiplicity, cand.multiplicity),
            source=sources,
        )
    return out


def find_roots_at_k(
    red,
    k: float,
    C_l0: float,
    *,
    w_norm_min: float = 1e-3,
    w_norm_max: float = 1.25,
    ngrid: int = 500,
    scan_eta_norm: float = 1e-6,
    sigma_accept: float = 1e-6,
    multiplicity_tol: float = 1e-5,
    refine_xatol_norm: float = 1e-10,
    dedup_tol_norm: float = 5e-6,
    max_minima: int | None = None,
    balance_passes: int = 8,
):
    """Discover and certify all resolvable roots at one fixed k.

    Candidate sources
    -----------------
    sign:
        Intervals where Re(det A) changes sign on a slightly complex scan.

    svd:
        Local minima of the conservative singular residual R. This discovers
        roots of even multiplicity without turning reciprocal-space poles into
        accepted roots.

    A root is accepted only if R <= sigma_accept on the real frequency axis.
    """
    if ngrid < 5:
        raise ValueError("ngrid must be >= 5")
    if not (0.0 <= w_norm_min < w_norm_max):
        raise ValueError("invalid normalized frequency interval")

    scale = _frequency_scale(red, C_l0)
    wn = np.linspace(float(w_norm_min), float(w_norm_max), int(ngrid))
    omega = wn * scale
    scan_eta = float(scan_eta_norm) * scale

    det_re = np.array([_det_real(red, w, k, scan_eta) for w in omega], dtype=float)
    residual_scan = np.array(
        [_residual(red, w, k, scan_eta, balance_passes) for w in omega], dtype=float
    )

    brackets: list[tuple[float, float, str]] = []

    for i in range(len(omega) - 1):
        va, vb = det_re[i], det_re[i + 1]
        if np.isfinite(va) and np.isfinite(vb) and va * vb < 0.0:
            brackets.append((wn[i], wn[i + 1], "sign"))

    minima = []
    for i in range(1, len(wn) - 1):
        s0, s1, s2 = residual_scan[i - 1], residual_scan[i], residual_scan[i + 1]
        if np.isfinite(s1) and s1 <= s0 and s1 <= s2 and (s1 < s0 or s1 < s2):
            minima.append((s1, i))

    if max_minima is not None and len(minima) > int(max_minima):
        minima = sorted(minima, key=lambda item: item[0])[: int(max_minima)]

    for _, i in minima:
        brackets.append((wn[i - 1], wn[i + 1], "svd"))

    candidates: list[RootCandidate] = []
    for lo, hi, source in brackets:
        if not (hi > lo):
            continue

        def objective(x_norm):
            return _finite_minimizer_residual(
                red, float(x_norm) * scale, k, 0.0, balance_passes
            )

        try:
            result = minimize_scalar(
                objective,
                bounds=(float(lo), float(hi)),
                method="bounded",
                options={"xatol": float(refine_xatol_norm), "maxiter": 200},
            )
        except (ValueError, FloatingPointError):
            continue

        if not result.success or not np.isfinite(result.fun):
            continue

        root_norm = float(result.x)
        root_omega = root_norm * scale
        residual, multiplicity, det_abs, diag = secular_diagnostics(
            red,
            root_omega,
            k,
            imag=0.0,
            multiplicity_tol=multiplicity_tol,
            balance_passes=balance_passes,
        )

        if residual <= float(sigma_accept):
            candidates.append(
                RootCandidate(
                    omega=root_omega,
                    omega_norm=root_norm,
                    sigma_min=residual,
                    sigma_min_raw=float(diag["raw_sigma_min"]),
                    sigma_min_balanced=float(diag["balanced_sigma_min"]),
                    multiplicity=max(1, multiplicity),
                    det_abs=det_abs,
                    source=source,
                )
            )

    return _deduplicate(candidates, float(dedup_tol_norm))


def certify_frequency(
    red,
    k: float,
    omega: float,
    C_l0: float,
    *,
    sigma_accept: float = 1e-6,
    multiplicity_tol: float = 1e-5,
    balance_passes: int = 8,
):
    """Certify/reject one externally supplied frequency by the dual SVD test."""
    residual, multiplicity, det_abs, diag = secular_diagnostics(
        red,
        omega,
        k,
        imag=0.0,
        multiplicity_tol=multiplicity_tol,
        balance_passes=balance_passes,
    )
    return {
        "accepted": bool(residual <= sigma_accept),
        "omega": float(omega),
        "omega_norm": float(omega / _frequency_scale(red, C_l0)),
        "sigma_min": float(residual),
        "sigma_min_raw": float(diag["raw_sigma_min"]),
        "sigma_min_balanced": float(diag["balanced_sigma_min"]),
        "multiplicity": int(multiplicity),
        "det_abs": float(det_abs),
        "raw_singular_values": diag["raw_singular_values"],
        "balanced_singular_values": diag["balanced_singular_values"],
    }


def independent_fullgrid(
    red,
    C_l0: float,
    *,
    w_norm_min: float = 1e-3,
    w_norm_max: float = 1.25,
    ngrid: int = 500,
    maxbands: int | None = None,
    expand_multiplicity: bool = True,
    **finder_kwargs,
):
    """Apply find_roots_at_k independently at every k; NO band tracking."""
    k_values = np.asarray(red.k, dtype=float)
    if maxbands is None:
        maxbands = int(red.nbands)
    out = np.full((len(k_values), int(maxbands), 2), np.nan, dtype=float)
    metadata = []

    for ik, k in enumerate(k_values):
        roots = find_roots_at_k(
            red,
            float(k),
            C_l0,
            w_norm_min=w_norm_min,
            w_norm_max=w_norm_max,
            ngrid=ngrid,
            **finder_kwargs,
        )
        local = []
        for root in roots:
            copies = root.multiplicity if expand_multiplicity else 1
            local.extend([root] * copies)
        local.sort(key=lambda r: r.omega)

        for ib, root in enumerate(local[: int(maxbands)]):
            out[ik, ib, 0] = root.omega
            out[ik, ib, 1] = 0.0

        metadata.append(roots)

    return out, metadata
