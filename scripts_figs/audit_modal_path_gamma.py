"""Audit the bidirectionally completed modal event graph around Gamma."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from certified_solver import CertifiedRed
from modal_path import build_modal_path_graph


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-suma", type=int, default=40)
    parser.add_argument("--ngrid", type=int, default=120)
    parser.add_argument(
        "--k-over-pi", nargs="+", type=float,
        default=[0.75, 0.875, 1.0, 1.125, 1.25],
    )
    parser.add_argument("--w-min", type=float, default=0.70)
    parser.add_argument("--w-max", type=float, default=1.40)
    parser.add_argument("--half-width", type=float, default=0.08)
    parser.add_argument("--max-sweeps", type=int, default=3)
    args = parser.parse_args()

    r = build_reference_red(args.n_suma)
    k_values = [float(kp) * np.pi / r.a for kp in args.k_over_pi]
    graph = build_modal_path_graph(
        r,
        k_values,
        CT0,
        w_norm_min=args.w_min,
        w_norm_max=args.w_max,
        ngrid=args.ngrid,
        max_sweeps=args.max_sweeps,
        search_half_width_norm=args.half_width,
        targeted_max_depth=5,
        completion_max_rounds=2,
        min_pair_affinity=0.15,
        # coverage_floor is derived from min_principal_cosine**2 so the
        # completeness and transport gates are mathematically consistent.
        transport_max_delta_omega_norm=0.08,
        min_principal_cosine=0.65,
        frequency_weight=0.05,
    )

    print("modal path audit around Gamma")
    print("=" * 100)
    print(f"frequency window=[{args.w_min:.3f},{args.w_max:.3f}]")
    print(
        f"completion_sweeps={graph.completion_sweeps} roots_added={graph.roots_added} "
        f"stabilised={graph.stabilised} complete_under_policy={graph.complete_under_policy}"
    )

    for kp, layer in zip(args.k_over_pi, graph.layers):
        print("\n" + f"k/pi={kp:.6f}")
        print("idx | omega_norm   | mult | R")
        for j, mode in enumerate(layer.modes):
            print(
                f"{j:3d} | {mode.omega_norm:12.9f} | {mode.multiplicity:4d} | "
                f"{mode.root.sigma_min:.3e}"
            )

    print("\nadjacent transport")
    print("=" * 100)
    for pair in graph.pairs:
        kp0 = args.k_over_pi[pair.left_index]
        kp1 = args.k_over_pi[pair.right_index]
        tr = pair.transport
        print(
            f"{kp0:.6f}->{kp1:.6f}: complete={pair.complete_bidirectionally} "
            f"matched={tr.matched_dimensions}/{tr.total_source_dimensions}->{tr.total_target_dimensions} "
            f"source_unmatched={tr.source_unmatched} target_unmatched={tr.target_unmatched}"
        )
        for edge in tr.edges:
            print(
                f"    {edge.source_index}->{edge.target_index} dim={edge.dimensions} "
                f"dw={edge.delta_omega_norm:.6f} score={edge.modal_score:.5f}"
            )

    internal_unmatched = []
    for pair in graph.pairs:
        if any(pair.transport.source_unmatched) or any(pair.transport.target_unmatched):
            internal_unmatched.append(pair.left_index)

    print("\nsummary")
    print("-" * 100)
    print(f"event_edges={len(graph.edges)}")
    print(f"pairs_with_transport_unmatched={internal_unmatched or 'none'}")

    if not graph.stabilised:
        raise SystemExit("bidirectional completion did not stabilise")
    if not graph.complete_under_policy:
        raise SystemExit("modal path is not complete under the current bidirectional policy")


if __name__ == "__main__":
    main()
