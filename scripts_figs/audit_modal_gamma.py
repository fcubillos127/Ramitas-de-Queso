"""Audit modal-subspace continuity through the Gamma-point doublet.

The purpose is diagnostic, not production band labelling.  Certified local
roots are computed at several neighbouring k points with a fixed converged
reciprocal truncation.  Their right null subspaces are compared through
principal angles in the original angular-harmonic basis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from certified_solver import CertifiedRed
from modal_tracking import build_modal_spectrum, candidate_modal_links, subspace_metrics
from rootfinder_certified import find_roots_at_k


CT0 = 295.0


def build_reference_red(n_suma: int = 60):
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


def local_spectrum(r, k_over_pi: float, ngrid: int):
    k = float(k_over_pi) * np.pi / r.a
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
    modes = build_modal_spectrum(r, k, roots)
    return k, roots, modes


def print_spectrum(kp, modes):
    print("\n" + "=" * 96)
    print(f"k/pi = {kp:.6f}")
    print("idx | omega_norm   | mult | root residual | modal rel. residual | dominant harmonics")
    print("-" * 96)
    for i, mode in enumerate(modes):
        cut = (len(mode.angular_weights) - 1) // 2
        order = np.argsort(mode.angular_weights)[::-1][:3]
        dominant = ", ".join(
            f"n={int(j-cut):+d}:{mode.angular_weights[j]:.3f}" for j in order
        )
        print(
            f"{i:3d} | {mode.omega_norm:12.9f} | {mode.multiplicity:4d} | "
            f"{mode.root.sigma_min:13.3e} | {mode.relative_subspace_residual:18.3e} | "
            f"{dominant}"
        )


def print_adjacent_links(kp0, modes0, kp1, modes1, max_delta):
    print("\n" + "-" * 96)
    print(f"modal links: k/pi {kp0:.6f} -> {kp1:.6f}")
    links = candidate_modal_links(
        modes0,
        modes1,
        max_delta_omega_norm=float(max_delta),
        min_affinity=0.05,
    )
    print("old->new | domega | affinity | coverage old | coverage new | principal cosines")
    for link in links:
        cosines = ",".join(f"{c:.4f}" for c in link.metrics.principal_cosines)
        print(
            f"{link.previous_index:3d}->{link.current_index:<3d} | "
            f"{link.delta_omega_norm:7.4f} | {link.metrics.affinity:8.5f} | "
            f"{link.metrics.coverage_first:12.5f} | "
            f"{link.metrics.coverage_second:12.5f} | [{cosines}]"
        )


def gamma_nsum_subspace_check(ngrid: int):
    spectra = []
    for n in (40, 60):
        r = build_reference_red(n)
        _k, _roots, modes = local_spectrum(r, 1.0, ngrid)
        doublets = [mode for mode in modes if mode.multiplicity == 2]
        if len(doublets) != 1:
            raise RuntimeError(f"expected one Gamma doublet at n_suma={n}, found {len(doublets)}")
        spectra.append(doublets[0])

    metrics = subspace_metrics(spectra[0], spectra[1])
    print("\n" + "=" * 96)
    print("Gamma doublet: modal convergence n_suma=40 -> 60")
    print(f"omega_norm: {spectra[0].omega_norm:.9f} -> {spectra[1].omega_norm:.9f}")
    print(
        "principal cosines: "
        + ", ".join(f"{value:.12f}" for value in metrics.principal_cosines)
    )
    print(f"subspace affinity: {metrics.affinity:.12f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k-over-pi",
        nargs="+",
        type=float,
        default=[0.75, 0.875, 1.0, 1.125, 1.25],
    )
    parser.add_argument("--n-suma", type=int, default=60)
    parser.add_argument("--ngrid", type=int, default=140)
    parser.add_argument("--max-delta", type=float, default=0.25)
    parser.add_argument("--skip-nsum-check", action="store_true")
    args = parser.parse_args()

    r = build_reference_red(args.n_suma)
    spectra = []
    for kp in args.k_over_pi:
        _k, _roots, modes = local_spectrum(r, kp, args.ngrid)
        spectra.append((float(kp), modes))
        print_spectrum(float(kp), modes)

    for (kp0, modes0), (kp1, modes1) in zip(spectra[:-1], spectra[1:]):
        print_adjacent_links(kp0, modes0, kp1, modes1, args.max_delta)

    if not args.skip_nsum_check:
        gamma_nsum_subspace_check(args.ngrid)


if __name__ == "__main__":
    main()
