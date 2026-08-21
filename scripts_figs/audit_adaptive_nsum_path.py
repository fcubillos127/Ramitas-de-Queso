"""Adaptive n_suma audit on a denser set of Bloch points.

The default 13-point square-path scan is intended to validate that no narrow
k-region between the earlier seven audit points requires a larger reciprocal
truncation under the frozen psi=0/cut=4 baseline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from adaptive_nsum import certify_roots_adaptive_nsum
from certified_solver import CertifiedRed


CT0 = 295.0


def build_reference_red():
    r = CertifiedRed(comp=["matrix", "inclusion"])
    r.dens = [1150.0, 1250.0]
    r.vel0 = [CT0, CT0]
    r.vels = [894.0, 894.0]
    r.filling = 0.5
    r.cut = 4
    r.nbands = 10
    r.n_suma = 8
    r.nk = 13
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk", type=int, default=13)
    parser.add_argument("--ngrid", type=int, default=120)
    parser.add_argument(
        "--n-schedule",
        nargs="+",
        type=int,
        default=[8, 12, 20, 30, 40, 60, 80],
    )
    parser.add_argument("--stable-steps", type=int, default=2)
    parser.add_argument("--confirmation-steps", type=int, default=1)
    parser.add_argument("--freq-atol", type=float, default=5e-5)
    parser.add_argument("--freq-rtol", type=float, default=5e-5)
    parser.add_argument("--residual-tol", type=float, default=1e-6)
    args = parser.parse_args()

    if args.nk < 2:
        raise ValueError("nk must be >= 2")

    r = build_reference_red()
    k_over_pi = np.linspace(0.0, 3.0, int(args.nk))
    rows = []

    print(
        "k/pi | converged | recommended | certified_at | evaluated Ns | "
        "roots | modes | max final R"
    )
    print("-" * 112)

    for kp in k_over_pi:
        k = float(kp) * np.pi / r.a
        result = certify_roots_adaptive_nsum(
            r,
            k,
            CT0,
            n_schedule=args.n_schedule,
            stable_steps=args.stable_steps,
            confirmation_steps=args.confirmation_steps,
            frequency_atol_norm=args.freq_atol,
            frequency_rtol=args.freq_rtol,
            residual_tol=args.residual_tol,
            finder_kwargs={
                "w_norm_min": 1e-3,
                "w_norm_max": 1.25,
                "ngrid": args.ngrid,
                "scan_eta_norm": 1e-6,
                "sigma_accept": args.residual_tol,
                "multiplicity_tol": 1e-5,
            },
        )

        modes = sum(root.multiplicity for root in result.final_roots)
        max_r = max((root.sigma_min for root in result.final_roots), default=np.nan)
        evaluated = ",".join(str(n) for n in result.evaluated_n_values)
        print(
            f"{kp:4.2f} | {str(result.converged):9s} | "
            f"{str(result.recommended_n_suma):11s} | "
            f"{str(result.certification_n_suma):12s} | "
            f"{evaluated:20s} | {len(result.final_roots):5d} | "
            f"{modes:5d} | {max_r:.2e}"
        )
        rows.append(result)

    unconverged = [
        float(kp) for kp, result in zip(k_over_pi, rows) if not result.converged
    ]
    recommended = [
        int(result.recommended_n_suma)
        for result in rows
        if result.recommended_n_suma is not None
    ]
    certified = [
        int(result.certification_n_suma)
        for result in rows
        if result.certification_n_suma is not None
    ]

    print("\nSummary")
    print("-------")
    print(f"points: {len(rows)}")
    print(f"unconverged k/pi: {unconverged or 'none'}")
    print(f"max recommended n_suma: {max(recommended) if recommended else None}")
    print(f"max certification n_suma: {max(certified) if certified else None}")


if __name__ == "__main__":
    main()
