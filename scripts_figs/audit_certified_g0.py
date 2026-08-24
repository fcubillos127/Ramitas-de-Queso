"""Reproducible audit of the certified G0 layer and the production root finder.

Run from the repository root:

    python scripts_figs/audit_certified_g0.py

The script does not modify data files.  It compares legacy and certified G0 at
Gamma and then scans the corrected secular matrix using both |det| and the
smallest singular values.  The latter expose even-multiplicity roots that a
sign-change detector cannot see.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from Bandas_Tools import Red
from certified_solver import CertifiedRed


CT0 = 295.0


def build(cls, *, nk=30, n_suma=20):
    r = cls(comp=["matriz", "inclusion"])
    r.dens = [1150.0, 1250.0]
    r.vel0 = [CT0, CT0]
    r.vels = [894.0, 894.0]
    r.filling = 0.5
    r.cut = 4
    r.nbands = 8
    r.nk = nk
    r.n_suma = n_suma
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


def matrix_A(red, w_norm, k, eta=1e-2):
    omega = w_norm * 2.0 * np.pi * CT0 / red.a
    f = [omega, eta]
    cutoff = red.cut
    T = np.diag([red._Tn(f, n) for n in range(-cutoff, cutoff + 1)])
    G = red.G0(f, k, 1, cutoff)
    return T @ G - np.eye(2 * cutoff + 1)


def scan_gamma(red, wmin=0.05, wmax=1.35, ngrid=2600):
    k_gamma = np.pi / red.a
    grid = np.linspace(wmin, wmax, ngrid)
    dets = np.empty(ngrid, dtype=complex)
    smin = np.empty(ngrid)

    for i, w in enumerate(grid):
        A = matrix_A(red, w, k_gamma)
        dets[i] = np.linalg.det(A)
        smin[i] = np.linalg.svd(A, compute_uv=False)[-1]

    # Candidate physical roots from deep minima of sigma_min.
    score = -np.log10(np.maximum(smin, 1e-300))
    idx = find_peaks(score, prominence=0.8)[0]
    roots = []
    for i in idx:
        lo = grid[max(0, i - 2)]
        hi = grid[min(ngrid - 1, i + 2)]
        res = minimize_scalar(
            lambda x: np.linalg.svd(matrix_A(red, x, k_gamma), compute_uv=False)[-1],
            bounds=(lo, hi), method="bounded", options={"xatol": 1e-11}
        )
        if res.fun < 5e-4:
            sv = np.linalg.svd(matrix_A(red, res.x, k_gamma), compute_uv=False)
            roots.append((res.x, res.fun, sv[-3:]))

    sign_intervals = np.nonzero(np.real(dets[:-1]) * np.real(dets[1:]) < 0)[0]
    return grid, dets, smin, roots, sign_intervals


def main():
    legacy = build(Red, n_suma=20)
    certified = build(CertifiedRed, n_suma=20)
    k_gamma = np.pi
    w_probe = 0.5
    omega = w_probe * 2 * np.pi * CT0
    f = [omega, 0.2]

    gl = legacy.G0(f, k_gamma, 1, legacy.cut)
    gc = certified.G0(f, k_gamma, 1, certified.cut)
    rel = np.max(np.abs(gl - gc)) / max(1.0, np.max(np.abs(gc)))
    print(f"Gamma G0 legacy-vs-certified max relative difference: {rel:.6e}")

    print("\nCertified Gamma secular roots from sigma_min:")
    _, dets, _, roots, sign_intervals = scan_gamma(certified)
    for w, sm, tail in roots:
        eps = 1e-4
        dm = np.linalg.det(matrix_A(certified, w - eps, k_gamma))
        dp = np.linalg.det(matrix_A(certified, w + eps, k_gamma))
        changes_sign = np.real(dm) * np.real(dp) < 0
        multiplicity = int(np.count_nonzero(tail < 5e-4))
        print(
            f"  w_norm={w:.9f}  sigma_min={sm:.3e}  "
            f"small-SV multiplicity~{multiplicity}  Re(det) sign change={changes_sign}"
        )

    print(f"\nNumber of Re(det) sign-change intervals on the scan: {len(sign_intervals)}")
    print("A sigma_min root with sign-change=False is invisible to the current discovery rule.")


if __name__ == "__main__":
    main()
