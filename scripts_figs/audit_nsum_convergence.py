"""Audit convergence of certified MST roots versus n_suma.

This script is intentionally separate from production plotting. It computes
independent local spectra at representative Bloch points and asks when the
*entire root set* becomes stable under reciprocal-space refinement.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Running ``python scripts_figs/audit_nsum_convergence.py`` puts scripts_figs/
# rather than the repository root on sys.path.  Add the root explicitly so the
# audit modules can be imported in CI and from a normal checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from certified_solver import CertifiedRed
from convergence_certified import certify_roots_vs_nsum, convergence_table


CT0 = 295.0


def build_reference_red():
    r = CertifiedRed(comp=["matrix", "inclusion"])
    r.dens = [1150.0, 1250.0]
    r.vel0 = [CT0, CT0]
    r.vels = [894.0, 894.0]
    r.filling = 0.5
    r.cut = 4
    r.nbands = 10
    r.nk = 13
    r.n_suma = 8
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


def print_result(result):
    print("\n" + "=" * 88)
    print(f"k/pi = {result.k / np.pi:.6f}")
    print(
        f"final converged={result.converged}  "
        f"recommended_n_suma={result.recommended_n_suma}"
    )
    print("n_suma | roots (omega_norm; multiplicity; residual)")
    print("-" * 88)
    for n, roots in zip(result.n_values, result.root_sets):
        text = ", ".join(
            f"{root.omega_norm:.9f};m={root.multiplicity};R={root.sigma_min:.1e}"
            for root in roots
        )
        print(f"{n:6d} | {text or '--'}")

    print("\nrefinement steps")
    for step in result.steps:
        print(
            f"{step.n_previous:3d}->{step.n_current:3d}: "
            f"conv={step.converged}  "
            f"roots={step.root_count_previous}->{step.root_count_current}  "
            f"modes={step.mode_count_previous}->{step.mode_count_current}  "
            f"max_domega={step.max_delta_norm:.3e}  "
            f"mult={step.multiplicity_stable}  residual={step.residuals_ok}  "
            f"unmatched=({len(step.unmatched_previous)},{len(step.unmatched_current)})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=[8, 12, 16, 20, 30, 40, 60],
    )
    parser.add_argument("--ngrid", type=int, default=180)
    parser.add_argument("--stable-steps", type=int, default=2)
    parser.add_argument("--freq-atol", type=float, default=5e-5)
    parser.add_argument("--freq-rtol", type=float, default=5e-5)
    parser.add_argument("--residual-tol", type=float, default=1e-6)
    parser.add_argument(
        "--k-over-pi",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    )
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    r = build_reference_red()
    all_rows = []
    summary = []

    for kp in args.k_over_pi:
        k = float(kp) * np.pi / r.a
        result = certify_roots_vs_nsum(
            r,
            k,
            CT0,
            n_values=args.n_values,
            stable_steps=args.stable_steps,
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
        print_result(result)
        all_rows.extend(convergence_table(result))
        summary.append(
            (
                kp,
                result.converged,
                result.recommended_n_suma,
                len(result.final_roots),
                sum(root.multiplicity for root in result.final_roots),
            )
        )

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("k/pi | converged | recommended n_suma | final roots | final modes")
    for kp, conv, nrec, nr, nm in summary:
        print(f"{kp:4.1f} | {str(conv):9s} | {str(nrec):18s} | {nr:11d} | {nm:11d}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        if all_rows:
            with args.csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)
        print(f"\nSaved detailed convergence rows to {args.csv}")


if __name__ == "__main__":
    main()
