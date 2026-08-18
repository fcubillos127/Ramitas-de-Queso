"""Reproducible root-finder audit on the certified G0 layer.

The script compares, at representative k points of the square path:

1. historical sign-change + fsolve logic (but using CertifiedRed, so G0 is fixed),
2. dual-SVD discovery/certification from rootfinder_certified.

This isolates root-finder defects from lattice-sum defects.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve, linear_sum_assignment

from certified_solver import CertifiedRed
from rootfinder_certified import certify_frequency, find_roots_at_k


CT0 = 295.0


def build_reference_red(n_suma=12):
    r = CertifiedRed(comp=["matrix", "inclusion"])
    r.dens = [1150.0, 1250.0]
    r.vel0 = [CT0, CT0]
    r.vels = [894.0, 894.0]
    r.filling = 0.5
    r.cut = 4
    r.nbands = 8
    r.nk = 13
    r.n_suma = int(n_suma)
    r.lattice = "sq"
    r.psi = 0.0
    r.a = 1.0
    r._set_k_end()
    r.cond_borde = "hollow"
    r.imag_tol = 0.8
    r.sol_tol = 1e-2
    r.asign_param()
    r.r1 = 0.45
    r.r2 = 0.50
    return r


def _det_vec(red, w_real, w_imag, k):
    return np.asarray(red.determinant_longitudinal([w_real, w_imag], k, red.cut), float)


def historical_roots_at_k(
    red,
    k,
    *,
    w_norm_min=1e-3,
    w_norm_max=1.25,
    ventanas_por_unidad=100,
    eta=1e-2,
    max_nivel=2,
):
    """Minimal reproduction of zeros_longitudinal_fullgrid discovery/refinement."""
    scale = 2.0 * np.pi * CT0 / red.a
    w_min = w_norm_min * scale
    w_max = w_norm_max * scale
    npoints = max(5, int(ventanas_por_unidad * (w_norm_max - w_norm_min)))
    grid = np.linspace(w_min, w_max, npoints)

    def eval_re(w):
        try:
            return float(_det_vec(red, w, eta, k)[0])
        except Exception:
            return np.nan

    vals = np.asarray([eval_re(w) for w in grid], float)

    def subdivide(a, b, level, fa, fb):
        c = 0.5 * (a + b)
        fc = eval_re(c)
        if level >= max_nivel:
            return [(a, b)]
        out = []
        if np.isfinite(fa) and np.isfinite(fc) and fa * fc < 0:
            out += subdivide(a, c, level + 1, fa, fc)
        if np.isfinite(fc) and np.isfinite(fb) and fc * fb < 0:
            out += subdivide(c, b, level + 1, fc, fb)
        return out or [(a, b)]

    intervals = []
    for i in range(len(grid) - 1):
        fa, fb = vals[i], vals[i + 1]
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            intervals += subdivide(grid[i], grid[i + 1], 0, fa, fb)

    roots = []
    for lo, hi in intervals:
        seed = 0.5 * (lo + hi)
        try:
            sol, _, ier, _ = fsolve(
                lambda x: _det_vec(red, x[0], x[1], k),
                [seed, eta],
                xtol=red.sol_tol,
                epsfcn=red.epsfcn,
                full_output=True,
            )
        except Exception:
            continue

        wr, wi = map(float, sol)
        wn = wr / scale
        if ier != 1 or abs(wi) >= red.imag_tol or not (0.0 < wn < w_norm_max):
            continue
        if any(np.isclose(wr, old[0], rtol=1e-4) for old in roots):
            continue
        roots.append((wr, wi))

    return sorted(roots)


def one_to_one_match(old_freq, new_freq, tolerance):
    """Return accepted Hungarian matches and unmatched indices."""
    old_freq = np.asarray(old_freq, dtype=float)
    new_freq = np.asarray(new_freq, dtype=float)
    if old_freq.size == 0 or new_freq.size == 0:
        return [], list(range(old_freq.size)), list(range(new_freq.size))

    cost = np.abs(old_freq[:, None] - new_freq[None, :])
    old_idx, new_idx = linear_sum_assignment(cost)

    matches = []
    used_old = set()
    used_new = set()
    for i, j in zip(old_idx, new_idx):
        if cost[i, j] < tolerance:
            matches.append((int(i), int(j), float(cost[i, j])))
            used_old.add(int(i))
            used_new.add(int(j))

    unmatched_old = [i for i in range(old_freq.size) if i not in used_old]
    unmatched_new = [j for j in range(new_freq.size) if j not in used_new]
    return matches, unmatched_old, unmatched_new


def main():
    r = build_reference_red(n_suma=12)
    k_values = np.linspace(0.0, 3.0 * np.pi / r.a, 7)
    match_tol_norm = 2e-3

    total_old = matched_old = false_old = total_new = missed_new = 0
    frequency_errors = []

    print("k/pi | historical candidate -> certified residual | SVD roots")
    print("-" * 100)

    for k in k_values:
        old = historical_roots_at_k(r, float(k))
        new = find_roots_at_k(
            r,
            float(k),
            CT0,
            w_norm_min=1e-3,
            w_norm_max=1.25,
            ngrid=120,
            sigma_accept=1e-6,
            multiplicity_tol=1e-5,
        )

        old_diag = []
        for wr, wi in old:
            diag = certify_frequency(r, float(k), wr, CT0, sigma_accept=1e-6)
            old_diag.append((diag["omega_norm"], diag["sigma_min"], diag["accepted"]))

        old_freq = np.asarray([x[0] for x in old_diag], float)
        new_freq = np.asarray([x.omega_norm for x in new], float)
        matches, unmatched_old, unmatched_new = one_to_one_match(
            old_freq, new_freq, match_tol_norm
        )

        total_old += len(old_freq)
        total_new += len(new_freq)
        matched_old += len(matches)
        false_old += len(unmatched_old)
        missed_new += len(unmatched_new)
        frequency_errors.extend(error for _, _, error in matches)

        old_text = ", ".join(
            f"{w:.6f} (R={s:.1e}, {'ok' if ok else 'REJECT'})"
            for w, s, ok in old_diag
        ) or "--"
        new_text = ", ".join(
            f"{x.omega_norm:.6f} [m={x.multiplicity}, {x.source}]" for x in new
        ) or "--"
        print(f"{k/np.pi:4.1f} | {old_text}\n     | SVD: {new_text}")
        if unmatched_old:
            print("     | unmatched historical:", ", ".join(f"{old_freq[i]:.6f}" for i in unmatched_old))
        if unmatched_new:
            print("     | missed by historical :", ", ".join(f"{new_freq[j]:.6f}" for j in unmatched_new))
        print()

    print("Summary")
    print("-------")
    print(f"historical candidates : {total_old}")
    print(f"one-to-one matches     : {matched_old}")
    print(f"extra historical      : {false_old}")
    print(f"certified SVD roots    : {total_new}")
    print(f"missed by historical   : {missed_new}")
    if frequency_errors:
        print(f"median |Delta w_norm|  : {np.median(frequency_errors):.3e}")
        print(f"max    |Delta w_norm|  : {np.max(frequency_errors):.3e}")


if __name__ == "__main__":
    main()
