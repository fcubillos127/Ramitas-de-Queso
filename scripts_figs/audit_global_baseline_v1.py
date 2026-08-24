"""Integrated certified baseline for the square-lattice psi=0 reference problem.

The displayed scientific window is 0.70 <= omega*a/(2*pi*Ct0) <= 1.40, but
roots are discovered in a slightly wider upper guard band.  This prevents a
physical branch that merely crosses the plotting boundary from being mistaken
for a discontinuity.

Two local facts established independently are built into this *audit policy*:

* the M doublet needs one half-step (k/pi=0.0625) to resolve its two outgoing
  modal directions with the event-level overlap threshold;
* immediately after X the two upper simple roots rotate strongly inside an
  almost invariant two-dimensional subspace, so continuity is certified for
  the joint cluster without claiming a unique branch identity inside it.

This script still does not modify the production solver.  It closes a finite,
reproducible v1 baseline for psi=0, cut=4, n_suma=40.
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
from modal_clusters import certify_cluster_continuity
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
    mg = np.linspace(0.0, 1.0, p)
    gx = np.linspace(1.0, 2.0, p)[1:]
    xm = np.linspace(2.0, 3.0, p)[1:]
    base = np.concatenate([mg, gx, xm])
    # Local half-steps justified by independent modal audits, not plotting hacks.
    critical = np.asarray([0.0625, 2.0625], dtype=float)
    return np.unique(np.concatenate([base, critical]))


def index_at(kp, value, tol=1e-12):
    matches = np.flatnonzero(np.isclose(kp, float(value), atol=tol, rtol=0.0))
    if len(matches) != 1:
        raise ValueError(f"expected one path point at k/pi={value}, found {matches}")
    return int(matches[0])


def in_core(mode, w_min, w_max):
    return float(w_min) <= float(mode.omega_norm) <= float(w_max)


def core_signature(layer, w_min, w_max):
    return tuple(
        (float(mode.omega_norm), int(mode.multiplicity))
        for mode in layer.modes
        if in_core(mode, w_min, w_max)
    )


def compare_M_endpoints(graph, w_min, w_max, tol: float = 5e-5):
    left = core_signature(graph.layers[0], w_min, w_max)
    right = core_signature(graph.layers[-1], w_min, w_max)
    if len(left) != len(right):
        return False, np.inf, f"event counts differ: {len(left)} vs {len(right)}"
    max_drift = 0.0
    for j, ((wl, ml), (wr, mr)) in enumerate(zip(left, right)):
        if ml != mr:
            return False, np.inf, f"multiplicity mismatch at event {j}: {ml} vs {mr}"
        drift = abs(wl - wr)
        max_drift = max(max_drift, drift)
        if drift > tol:
            return False, max_drift, f"frequency mismatch at event {j}: {wl:.9f} vs {wr:.9f}"
    return True, max_drift, "ok"


def core_unmatched_indices(layer, unmatched, w_min, w_max):
    return tuple(
        i
        for i, (mode, missing) in enumerate(zip(layer.modes, unmatched))
        if int(missing) > 0 and in_core(mode, w_min, w_max)
    )


def upper_x_cluster_indices(layer):
    indices = [
        i for i, mode in enumerate(layer.modes)
        if 0.95 <= float(mode.omega_norm) <= 1.10
    ]
    if len(indices) != 2:
        raise ValueError(f"expected two upper X-cluster events, found {indices}")
    return tuple(indices)


def certify_core_continuity(kp, graph, w_min, w_max):
    """Return unresolved core pairs after the independently certified X rescue."""
    unresolved = []
    x_cluster_result = None

    for pair in graph.pairs:
        left = graph.layers[pair.left_index]
        right = graph.layers[pair.right_index]
        source_bad = core_unmatched_indices(
            left, pair.transport.source_unmatched, w_min, w_max
        )
        target_bad = core_unmatched_indices(
            right, pair.transport.target_unmatched, w_min, w_max
        )
        if not source_bad and not target_bad:
            continue

        # The only allowed event-level exception is the independently diagnosed
        # two-mode rotation immediately after X.  The whole 2D cluster must pass
        # a strong principal-angle certificate; no individual edge is invented.
        if np.isclose(kp[pair.left_index], 2.0) and np.isclose(kp[pair.right_index], 2.0625):
            source_cluster = upper_x_cluster_indices(left)
            target_cluster = upper_x_cluster_indices(right)
            result = certify_cluster_continuity(
                left.modes,
                right.modes,
                source_cluster,
                target_cluster,
                min_principal_cosine=0.65,
                max_center_shift_norm=0.10,
            )
            if (
                result.certified
                and set(source_bad).issubset(source_cluster)
                and set(target_bad).issubset(target_cluster)
            ):
                x_cluster_result = result
                continue

        unresolved.append(
            (pair.left_index, pair.right_index, source_bad, target_bad)
        )

    return tuple(unresolved), x_cluster_result


def print_spectrum(kp, graph, w_min, w_max):
    print("\nlayer spectrum (CORE marks the reported window; GUARD is auxiliary)")
    print("-" * 118)
    for i, layer in enumerate(graph.layers):
        for j, mode in enumerate(layer.modes):
            region = "CORE " if in_core(mode, w_min, w_max) else "GUARD"
            print(
                f"LAYER {i:02d} k/pi={kp[i]:.7f} event={j:02d} {region} "
                f"omega={mode.omega_norm:.12f} m={mode.multiplicity:d} "
                f"R={mode.root.sigma_min:.3e}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-suma", type=int, default=40)
    parser.add_argument("--ngrid", type=int, default=110)
    parser.add_argument("--points-per-segment", type=int, default=9)
    parser.add_argument("--w-min", type=float, default=0.70)
    parser.add_argument("--w-max", type=float, default=1.40)
    parser.add_argument("--search-w-max", type=float, default=1.50)
    args = parser.parse_args()
    if args.search_w_max <= args.w_max:
        raise SystemExit("search-w-max must exceed the reported w-max guard boundary")

    red = build_reference_red(args.n_suma)
    kp = path_grid(args.points_per_segment)
    graph = build_modal_path_graph(
        red,
        kp * np.pi / red.a,
        CT0,
        w_norm_min=args.w_min,
        w_norm_max=args.search_w_max,
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
    print("=" * 118)
    print(
        f"path_points={len(kp)} base_points_per_segment={args.points_per_segment} "
        f"n_suma={args.n_suma} cut={red.cut} core=[{args.w_min:.2f},{args.w_max:.2f}] "
        f"search_guard_max={args.search_w_max:.2f}"
    )
    print(
        f"stabilised={graph.stabilised} raw_graph_complete={graph.complete_under_policy} "
        f"completion_sweeps={graph.completion_sweeps} roots_added={graph.roots_added}"
    )

    print_spectrum(kp, graph, args.w_min, args.w_max)

    unresolved, x_cluster = certify_core_continuity(
        kp, graph, args.w_min, args.w_max
    )
    print("\ncore continuity")
    print("-" * 118)
    if unresolved:
        for left, right, source_bad, target_bad in unresolved:
            print(
                f"UNRESOLVED {left:02d}->{right:02d} "
                f"k/pi={kp[left]:.7f}->{kp[right]:.7f} "
                f"source_bad={source_bad} target_bad={target_bad}"
            )
    else:
        print("all core modal dimensions accounted for")

    if x_cluster is not None:
        print(
            "X_CLUSTER_CERTIFIED=True "
            f"principal_cosines={np.array2string(x_cluster.metrics.principal_cosines, precision=8)} "
            f"affinity={x_cluster.metrics.affinity:.8f} "
            f"center_shift={x_cluster.center_shift:.3e}"
        )
    else:
        print("X_CLUSTER_CERTIFIED=False_or_not_needed")

    m_ok, m_drift, m_message = compare_M_endpoints(
        graph, args.w_min, args.w_max
    )
    print(f"M_ENDPOINT_CLOSURE={m_ok} max_drift={m_drift:.3e} detail={m_message}")

    gamma_index = index_at(kp, 1.0)
    gamma_core = [
        mode for mode in graph.layers[gamma_index].modes
        if in_core(mode, args.w_min, args.w_max)
    ]
    gamma_doublets = [
        mode for mode in gamma_core
        if mode.multiplicity == 2 and abs(mode.omega_norm - 1.06458) < 5e-3
    ]
    gamma_ok = len(gamma_doublets) == 1
    print(f"GAMMA_DOUBLET={gamma_ok} count={len(gamma_doublets)}")

    print("\nhigh-symmetry core spectra")
    print("-" * 118)
    for label, kval in (("M(start)", 0.0), ("Gamma", 1.0), ("X", 2.0), ("M(end)", 3.0)):
        idx = index_at(kp, kval)
        modes = [m for m in graph.layers[idx].modes if in_core(m, args.w_min, args.w_max)]
        text = ", ".join(f"{m.omega_norm:.9f}(m={m.multiplicity})" for m in modes)
        print(f"{label:8s}: {text}")

    certified = bool(
        graph.stabilised
        and not unresolved
        and x_cluster is not None
        and x_cluster.certified
        and m_ok
        and gamma_ok
    )
    print(f"\nBASELINE_CERTIFIED={certified}")
    if not certified:
        raise SystemExit("baseline v1 still has unresolved core continuity")
    print("PASS: baseline v1 is closed in the reported spectral window")


if __name__ == "__main__":
    main()
