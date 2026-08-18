"""Certified reference implementation of the reciprocal-space lattice sum.

This module is intentionally conservative.  It exists to validate the lattice
sum used by the production band solver before performance optimisations are
considered.

Two removable singularities are handled analytically:

1. Q = |k + G| -> 0 for N = 0:

       J_1(Q a) / [Q (Q^2-k0^2)] -> -a/(2 k0^2).

   For N>0 the same contribution tends to zero.

2. J_{N+1}(k0 a) -> 0 in the Chin representation.  The physical lattice
   sum is finite unless the same frequency is also a genuine reciprocal-space
   pole Q^2=k0^2.  We evaluate the removable 0/0 form with l'Hopital's rule.

Genuine reciprocal-space poles Q^2=k0^2 are NOT regularised away.  With the
complex-frequency continuation used by the band solver they are normally
avoided automatically; an exact pole raises FloatingPointError.
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp

import suma_de_red as legacy_sr


_Q_ZERO_TOL = 1e-13
_POLE_TOL = 1e-13
_BESSEL_ZERO_TOL = 1e-8


def precompute_Qh(a, k_vec, n, lattice="hx"):
    """Reuse the repository geometry generator; no physics is changed here."""
    return legacy_sr.precompute_Qh(a, k_vec, n, lattice)


def _cell_area(a: float, lattice: str) -> float:
    if lattice == "sq":
        return a**2
    if lattice == "hx":
        return np.sqrt(3.0) * a**2 / 2.0
    raise ValueError(f"Lattice no reconocida: {lattice!r}")


def _raw_reciprocal_sum_and_derivative(
    N: int,
    k0: complex,
    Qh_mod,
    ang,
    a: float,
    *,
    q_zero_tol: float = _Q_ZERO_TOL,
    pole_tol: float = _POLE_TOL,
):
    """Return R_N(k0) and dR_N/dk0 before the Chin prefactor.

    R_N = sum_G J_{N+1}(Qa) exp(i N theta_Q) /
                [Q (Q^2-k0^2)].

    The Q=0 contribution is evaluated from its analytic limit.  Genuine
    Q^2=k0^2 poles are left as poles and therefore raise when hit exactly.
    """
    Q = np.asarray(Qh_mod, dtype=float)
    theta = np.asarray(ang, dtype=float)
    numerator = jv(N + 1, Q * a) * np.exp(1j * N * theta)

    zero_mask = Q <= q_zero_tol
    reg_mask = ~zero_mask

    q = Q[reg_mask]
    num = numerator[reg_mask]
    pole_den = q**2 - k0**2
    scale = np.maximum(1.0, q**2 + abs(k0) ** 2)
    if np.any(np.abs(pole_den) <= pole_tol * scale):
        raise FloatingPointError(
            "Genuine reciprocal-space pole Q^2=k0^2 encountered in lattice sum"
        )

    raw = np.sum(num / (q * pole_den), dtype=complex)
    draw = np.sum(num * (2.0 * k0) / (q * pole_den**2), dtype=complex)

    if np.any(zero_mask) and N == 0:
        multiplicity = int(np.count_nonzero(zero_mask))
        raw += multiplicity * (-a / (2.0 * k0**2))
        draw += multiplicity * (a / k0**3)

    return raw, draw


def S1_pre_with_derivative(N, k0, Qh_mod, ang, a, lattice="hx"):
    """Return the Chin reciprocal term and its derivative with respect to k0."""
    raw, draw = _raw_reciprocal_sum_and_derivative(
        int(N), complex(k0), Qh_mod, ang, float(a)
    )
    pref = 4.0 * (1j) ** (int(N) + 1) / _cell_area(float(a), lattice)
    value = pref * k0 * raw
    derivative = pref * (raw + k0 * draw)
    return value, derivative


def S1_pre(N, k0, Qh_mod, ang, a, lattice="hx"):
    return S1_pre_with_derivative(N, k0, Qh_mod, ang, a, lattice)[0]


def _diagonal_term_and_derivative(N: int, k0: complex, a: float):
    if N != 0:
        return 0.0j, 0.0j

    # Algebraically equivalent to
    # (2j + k0*a*pi*H1(k0*a))/(k0*pi*a), but the separated form has a
    # transparent derivative.
    value = 2j / (k0 * np.pi * a) + hankel1(1, k0 * a)
    derivative = -2j / (k0**2 * np.pi * a) + a * h1vp(1, k0 * a)
    return value, derivative


def S_pre(
    M,
    m,
    k0,
    Qh_mod,
    ang,
    a,
    lattice="hx",
    *,
    bessel_zero_tol: float = _BESSEL_ZERO_TOL,
):
    """Regularised Chin lattice sum for one matrix element."""
    N0 = int(M) - int(m)
    N = abs(N0)
    k0 = complex(k0)
    a = float(a)

    term1, dterm1 = _diagonal_term_and_derivative(N, k0, a)
    term2, dterm2 = S1_pre_with_derivative(N, k0, Qh_mod, ang, a, lattice)

    denominator = jv(N + 1, k0 * a)
    if abs(denominator) > bessel_zero_tol:
        value = -(term1 + term2) / denominator
    else:
        # Removable singularity of the Chin representation.  If the numerator
        # did not tend to zero, l'Hopital would not be valid; guard against
        # silently masking such a case.
        numerator = term1 + term2
        derivative_denominator = a * jvp(N + 1, k0 * a)
        if abs(derivative_denominator) < 1e-12:
            raise FloatingPointError("Multiple/ill-conditioned Bessel zero")

        # At finite reciprocal truncation the cancellation is not exact.  The
        # derivative form is nevertheless the well-defined limit approached as
        # n_suma converges.
        value = -(dterm1 + dterm2) / derivative_denominator

    if N0 < 0:
        value = -np.conj(value)
    return value


def G0_matrix(red, f, k, pol, cut, n_suma):
    """Construct G0 using the certified scalar lattice-sum elements."""
    a = float(red.a)
    lattice = red.lattice
    k0 = red.k0(f, pol)
    k_vec = legacy_sr.K(a, k, lattice)
    Qh_mod, ang = precompute_Qh(a, k_vec, int(n_suma), lattice)

    size = 2 * int(cut) + 1
    values = {}
    for d in range(0, 2 * int(cut) + 1):
        values[d] = S_pre(d, 0, k0, Qh_mod, ang, a, lattice)
    for d in range(1, 2 * int(cut) + 1):
        values[-d] = -np.conj(values[d])

    out = np.empty((size, size), dtype=complex)
    for i in range(-int(cut), int(cut) + 1):
        for j in range(-int(cut), int(cut) + 1):
            out[i + int(cut), j + int(cut)] = values[i - j]
    return out


def G0_converged(
    red,
    f,
    k,
    pol,
    cut,
    *,
    n_suma_ini=None,
    tol=1e-6,
    n_suma_max=200,
    stable_passes=2,
    norm="max",
):
    """Robust convergence reference by recomputing the COMPLETE reciprocal sum.

    This is intentionally slower than the incremental production routine.  It
    cannot skip inner reciprocal shells: G0(n) always contains every pair
    h1,h2 in [-n,n].  It is therefore suitable as a validation oracle.
    """
    if n_suma_ini is None:
        n_suma_ini = max(1, int(getattr(red, "n_suma", 1)))
    n_suma_ini = max(1, int(n_suma_ini))
    n_suma_max = int(n_suma_max)
    stable_passes = max(1, int(stable_passes))

    previous = G0_matrix(red, f, k, pol, cut, n_suma_ini)
    ok = 0
    err_rel = np.inf

    for n in range(n_suma_ini + 1, n_suma_max + 1):
        current = G0_matrix(red, f, k, pol, cut, n)
        delta = current - previous
        if norm == "fro":
            numerator = np.linalg.norm(delta)
            denominator = max(1.0, np.linalg.norm(current))
        elif norm == "max":
            numerator = np.max(np.abs(delta))
            denominator = max(1.0, np.max(np.abs(current)))
        else:
            raise ValueError("norm must be 'max' or 'fro'")

        err_rel = float(numerator / denominator)
        ok = ok + 1 if err_rel < tol else 0
        if ok >= stable_passes:
            return current, {
                "n_suma": n,
                "err_rel": err_rel,
                "converged": True,
                "stable_passes": ok,
            }
        previous = current

    return previous, {
        "n_suma": n_suma_max,
        "err_rel": err_rel,
        "converged": False,
        "stable_passes": ok,
    }
