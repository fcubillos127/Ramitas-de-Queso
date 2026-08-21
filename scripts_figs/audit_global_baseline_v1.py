"""First integrated global baseline for the certified square-lattice solver.

Scope is deliberately finite: psi=0, cut=4, n_suma=40 and the already audited
frequency sector 0.70 <= omega*a/(2*pi*Ct0) <= 1.40.  The purpose is to close a
usable v1 baseline, not to claim the acoustic limit or deformed-scatterer model
are finished.

The scalar path parameter used by suma_de_red.K is
    0 -> M, 1*pi/a -> Gamma, 2*pi/a -> X, 3*pi/a -> M.
This script therefore audits M-Gamma-X-M explicitly and verifies closure at M.
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
from modal_path import build_modal_path_graph

CT0 = 295.0


def build_reference_red(n_suma: int = 40) -> CertifiedRed:
    r = CertifiedRed(comp=["matrix", "inclusion"])
    r.dens = [1150.0, 1250.0]
    r.vel0 = [CT0, CT0]
    r.vels = [894.0, 894.0]
    r.filling = 0.5
    r.cut = 4
    r.nbands = 12
    r.nk = 37
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


def path_grid(points_per_segment: int):
    p = int(points_per_segment)
    if p < 3:
        raise ValueError("points_per_segment must be >= 3")
    # Three equal segments, shared endpoints only once.
    mg = np.linspace(0.0, 1.0, p)
    gx = np.linspace(1.0, 2.0, p)[1:]
    xm = np.linspace(2.0, 3.0, p)[1:]
    return np.concatenate([mg, gx, xm])


def event_signature(layer):
    return tuple((float(m.omega_norm), int(m.multiplicity)) for m in layer.modes)


def assert_same_M_endpoints(graph, tol: float = 5e-5):
    left = event_signature(graph.layers[0])
    right = event_signature(graph.layers[-1])
    if len(left) != len(right):
        raise AssertionError(f"M endpoint event counts differ: {len(left)} vs {len(right)}")
    max_drift = 0.0
    for j, ((wl, ml), (wr, mr)) in enumerate(zip(left, right)):
        if ml != mr:
            raise AssertionError(f"M endpoint multiplicity mismatch at event {j}: {ml} vs {mr}")
        drift = abs(wl - wr)
        max_drift = max(max_drift, drift)
        if drift > tol:
            raise AssertionError(
                f"M endpoint frequency mismatch at event {j}: {wl:.9f} vs {wr:.9f}"
            )
    return max_drift


def high_symmetry_index(points_per_segment: int, label: str) -> int:
    p = int(points_per_segment)
    return {"M0": 0, "G": p - 1, "X": 2 * (p - 1), "M1": 3 * (p - 1)}[label]


def segment_metrics(graph, start: int, stop: int):
    edges = [e for e in graph.edges if start <= e.left_layer < stop]
    if not edges:
        return np.nan, np.nan, 0
    scores = np.asarray([e.modal_score for e in edges], dtype=float)
    jumps = np.asarray([e.delta_omega_norm for e in edges], dtype=float)
    return float(np.min(scores)), float(np.max(jumps)), len(edges)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-suma", type=int, default=40)
    parser.add_argument("--ngrid", type=int, default=110)
    parser.add_argument("--points-per-segment", type=int, default=13)
    parser.add_argument("--w-min", type=float, default=0.70)
    parser.add_argument("--w-max", type=float, default=1.40)
    args = parser.parse_args()

    red = build_reference_red(args.n_suma)
    kp = path_grid(args.points_per_segment)
    graph = build_modal_path_graph(
        red,
        kp * np.pi / red.a,
        CT0,
        w_norm_min=args.w_min,
        w_norm_max=args.w_max,
        ngrid=args.ngrid,
        max_sweeps=3,
        search_half_width_norm=0.10,
        targeted_max_depth=5,
        completion_max_rounds=2,
        min_pair_affinity=0.15,
        transport_max_delta_omega_norm=0.10,
        min_principal_cosine=0.65,
        frequency_weight=0.05,
    )

    print("global certified baseline v1: M-Gamma-X-M")
    print("=" * 112)
    print(
        f"points={len(kp)} points_per_segment={args.points_per_segment} "
        f"n_suma={args.n_suma} cut={red.cut} window=[{args.w_min:.2f},{args.w_max:.2f}]"
    )
    print(
        f"stabilised={graph.stabilised} complete={graph.complete_under_policy} "
        f"completion_sweeps={graph.completion_sweeps} roots_added={graph.roots_added}"
    )

    total_unmatched = 0
    for pair in graph.pairs:
        total_unmatched += int(sum(pair.transport.source_unmatched))
        total_unmatched += int(sum(pair.transport.target_unmatched))
    print(f"total_unmatched_modal_dimensions={total_unmatched}")

    if not graph.stabilised or not graph.complete_under_policy:
        raise SystemExit("global modal graph did not certify")
    if total_unmatched != 0:
        raise SystemExit(f"global graph contains {total_unmatched} unmatched modal dimensions")

    m_closure = assert_same_M_endpoints(graph)
    print(f"M_endpoint_max_frequency_drift={m_closure:.3e}")

    p = args.points_per_segment
    special = {
        "M(start)": high_symmetry_index(p, "M0"),
        "Gamma": high_symmetry_index(p, "G"),
        "X": high_symmetry_index(p, "X"),
        "M(end)": high_symmetry_index(p, "M1"),
    }
    print("\nhigh-symmetry spectra")
    print("-" * 112)
    for label, idx in special.items():
        layer = graph.layers[idx]
        spectrum = ", ".join(
            f"{mode.omega_norm:.9f}(m={mode.multiplicity})" for mode in layer.modes
        )
        print(f"{label:8s} k/pi={kp[idx]:.6f}: {spectrum}")

    print("\nsegment transport metrics")
    print("-" * 112)
    segments = (
        ("M->Gamma", 0, p - 1),
        ("Gamma->X", p - 1, 2 * (p - 1)),
        ("X->M", 2 * (p - 1), 3 * (p - 1)),
    )
    for name, start, stop in segments:
        min_score, max_jump, nedges = segment_metrics(graph, start, stop)
        print(
            f"{name:10s}: edges={nedges:3d} min_modal_score={min_score:.6f} "
            f"max_adjacent_dw={max_jump:.6f}"
        )

    # The Gamma doublet is a key regression target of the new pipeline.
    gamma = graph.layers[special["Gamma"]]
    doublets = [m for m in gamma.modes if m.multiplicity == 2 and abs(m.omega_norm - 1.06458) < 5e-3]
    if len(doublets) != 1:
        raise SystemExit(f"expected one certified Gamma doublet, found {len(doublets)}")

    print("\nPASS: first integrated certified baseline closes on M-Gamma-X-M")


if __name__ == "__main__":
    main()
