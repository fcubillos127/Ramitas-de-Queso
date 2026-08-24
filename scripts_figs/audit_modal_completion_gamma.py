"""Demonstrate modal completion and transport through the Gamma doublet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from certified_solver import CertifiedRed
from modal_completion import complete_spectrum_from_previous, diagnose_descendant_coverage
from modal_tracking import build_modal_spectrum
from modal_transport import assign_modal_transport
from rootfinder_certified import find_roots_at_k


CT0 = 295.0


def build_reference_red(n_suma=40):
    r = CertifiedRed(comp=["matrix", "inclusion"])
    r.dens = [1150.0, 1250.0]
    r.vel0 = [CT0, CT0]
    r.vels = [894.0, 894.0]
    r.filling = 0.5
    r.cut = 4
    r.nbands = 10
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


def roots_at(r, kp, ngrid):
    k = float(kp) * np.pi / r.a
    roots = find_roots_at_k(
        r,
        k,
        CT0,
        w_norm_min=1e-3,
        w_norm_max=1.25,
        ngrid=int(ngrid),
        scan_eta_norm=1e-6,
        sigma_accept=1e-6,
        multiplicity_tol=1e-5,
    )
    return k, roots


def print_roots(title, roots):
    print("\n" + title)
    print("-" * len(title))
    for i, root in enumerate(roots):
        print(
            f"{i:2d}: omega_norm={root.omega_norm:.9f} "
            f"m={root.multiplicity} R={root.sigma_min:.3e} source={root.source}"
        )


def print_diagnostics(title, diagnostics):
    print("\n" + title)
    print("-" * len(title))
    for diag in diagnostics:
        cosines = ",".join(f"{value:.5f}" for value in diag.principal_cosines)
        print(
            f"source={diag.source_index} m={diag.source_multiplicity} "
            f"candidates={diag.candidate_indices} capacity={diag.candidate_capacity} "
            f"union_dim={diag.union_dimension} coverage={diag.coverage:.5f} "
            f"refine={diag.needs_refinement} cos=[{cosines}]"
        )


def print_transport(transport):
    print("\nmodal transport Gamma -> Gamma-X")
    print("--------------------------------")
    for edge in transport.edges:
        cosines = ",".join(f"{value:.5f}" for value in edge.principal_cosines)
        print(
            f"{edge.source_index}->{edge.target_index}: dimensions={edge.dimensions} "
            f"domega={edge.delta_omega_norm:.6f} modal_score={edge.modal_score:.5f} "
            f"cos=[{cosines}]"
        )
    print(f"source_used={transport.source_used}")
    print(f"source_unmatched={transport.source_unmatched}")
    print(f"target_unmatched={transport.target_unmatched}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-suma", type=int, default=40)
    parser.add_argument("--coarse-ngrid", type=int, default=140)
    parser.add_argument("--targeted-max-depth", type=int, default=5)
    parser.add_argument("--half-width", type=float, default=0.06)
    args = parser.parse_args()

    r = build_reference_red(args.n_suma)
    k_prev, roots_prev = roots_at(r, 1.0, args.coarse_ngrid)
    k_curr, roots_curr = roots_at(r, 1.125, args.coarse_ngrid)

    modes_prev = build_modal_spectrum(r, k_prev, roots_prev)
    modes_curr = build_modal_spectrum(r, k_curr, roots_curr)
    before = diagnose_descendant_coverage(
        modes_prev,
        modes_curr,
        max_delta_omega_norm=args.half_width,
        min_pair_affinity=0.15,
        coverage_floor=0.60,
    )

    print_roots("Gamma roots", roots_prev)
    print_roots("Gamma-X roots before completion", roots_curr)
    print_diagnostics("Coverage before completion", before)

    result = complete_spectrum_from_previous(
        r,
        k_curr,
        CT0,
        modes_prev,
        roots_curr,
        search_half_width_norm=args.half_width,
        targeted_max_depth=args.targeted_max_depth,
        max_rounds=2,
        min_pair_affinity=0.15,
        coverage_floor=0.60,
        finder_kwargs={
            "sigma_accept": 1e-6,
            "multiplicity_tol": 1e-5,
        },
    )

    print_roots("Gamma-X roots after completion", result.roots)
    print_diagnostics("Coverage after completion", result.diagnostics)
    print("\ncompletion rounds")
    for item in result.rounds:
        print(
            f"round={item.round_index} roots_added={item.roots_added} "
            f"windows={item.searched_windows}"
        )
    print(f"complete_under_policy={result.complete_under_policy}")

    recovered = [root for root in result.roots if abs(root.omega_norm - 1.06347) < 5e-4]
    if not recovered:
        raise SystemExit("modal completion failed to recover the known Gamma-X descendant")
    if not result.complete_under_policy:
        raise SystemExit("modal completion did not restore descendant coverage")

    transport = assign_modal_transport(
        modes_prev,
        result.modes,
        max_delta_omega_norm=0.08,
        min_principal_cosine=0.65,
        frequency_weight=0.05,
    )
    print_transport(transport)

    # Gamma contains 1 + 2 + 1 = four modal dimensions in the tested frequency
    # window.  After completion all four must continue to Gamma-X.  In
    # particular source index 1 is the doublet and must transport two dimensions.
    if transport.matched_dimensions != 4:
        raise SystemExit(
            f"expected four transported Gamma dimensions, got {transport.matched_dimensions}"
        )
    if transport.source_used != (1, 2, 1):
        raise SystemExit(f"unexpected Gamma transport capacities: {transport.source_used}")


if __name__ == "__main__":
    main()
