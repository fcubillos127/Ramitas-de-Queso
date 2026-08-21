"""Audit convergence of modal tracking with respect to Bloch-path spacing.

The test is deliberately local to the already-certified spectral sector around
Gamma. It does not attempt to validate the acoustic limit or psi != 0.

To isolate *only* the effect of dk, certified roots are discovered once on the
finest 41-point grid. The 21- and 11-point graphs are exact nested subsets of
that same local spectral data. This prevents root-discovery noise and repeated
expensive solves from being confused with a tracking-discretisation effect.

For nested grids with 11, 21 and 41 points on k/pi in [0.75, 1.25], we require:
  * every graph is complete under the same modal policy;
  * multiplicities at the common nodes are identical by construction;
  * every coarse-grid modal edge is reachable through the finer graph;
  * the Gamma doublet transports exactly two dimensions on both sides;
  * the maximum adjacent frequency jump decreases as dk is refined.
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
from modal_path import assemble_modal_path_graph, build_modal_path_graph

CT0 = 295.0


def build_reference_red(n_suma: int = 40) -> CertifiedRed:
    r = CertifiedRed(comp=["matrix", "inclusion"])
    r.dens = [1150.0, 1250.0]
    r.vel0 = [CT0, CT0]
    r.vels = [894.0, 894.0]
    r.filling = 0.5
    r.cut = 4
    r.nbands = 10
    r.nk = 41
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


def make_fine_graph(red, ngrid: int):
    kp = np.linspace(0.75, 1.25, 41)
    graph = build_modal_path_graph(
        red,
        kp * np.pi / red.a,
        CT0,
        w_norm_min=0.70,
        w_norm_max=1.40,
        ngrid=int(ngrid),
        max_sweeps=3,
        search_half_width_norm=0.08,
        targeted_max_depth=5,
        completion_max_rounds=2,
        min_pair_affinity=0.15,
        transport_max_delta_omega_norm=0.08,
        min_principal_cosine=0.65,
        frequency_weight=0.05,
    )
    return kp, graph


def nested_graph(red, fine_kp, fine_graph, nk: int):
    if nk not in (11, 21, 41):
        raise ValueError("this audit expects nk in {11,21,41}")
    stride = (41 - 1) // (int(nk) - 1)
    indices = list(range(0, 41, stride))
    if len(indices) != nk or indices[-1] != 40:
        raise AssertionError("nested-grid construction failed")
    kp = np.asarray([fine_kp[i] for i in indices], dtype=float)
    root_sets = [fine_graph.layers[i].roots for i in indices]
    graph = assemble_modal_path_graph(
        red,
        kp * np.pi / red.a,
        root_sets,
        search_half_width_norm=0.08,
        min_pair_affinity=0.15,
        transport_max_delta_omega_norm=0.08,
        min_principal_cosine=0.65,
        frequency_weight=0.05,
        stabilised=True,
    )
    return kp, graph


def reachable_pairs(graph, left_layer: int, right_layer: int):
    """Return endpoint pairs connected by at least one directed modal path."""
    if right_layer <= left_layer:
        raise ValueError("right_layer must be larger than left_layer")
    current = {(event, event) for event in range(len(graph.layers[left_layer].modes))}
    for layer_idx in range(left_layer, right_layer):
        adjacency = {}
        for edge in graph.edges:
            if edge.left_layer == layer_idx and edge.right_layer == layer_idx + 1:
                adjacency.setdefault(edge.left_event, set()).add(edge.right_event)
        nxt = set()
        for origin, event in current:
            for target in adjacency.get(event, ()):
                nxt.add((origin, target))
        current = nxt
    return current


def compare_coarse_edges_to_fine(coarse, fine, ratio: int):
    missing = []
    checked = 0
    for ic in range(len(coarse.layers) - 1):
        fine_left = ic * ratio
        fine_right = (ic + 1) * ratio
        reach = reachable_pairs(fine, fine_left, fine_right)
        coarse_pairs = {
            (edge.left_event, edge.right_event)
            for edge in coarse.edges
            if edge.left_layer == ic and edge.right_layer == ic + 1
        }
        checked += len(coarse_pairs)
        for pair in coarse_pairs:
            if pair not in reach:
                missing.append((ic, pair))
    if missing:
        raise AssertionError(f"coarse modal links not reproducible on finer grid: {missing}")
    return checked


def graph_metrics(graph):
    scores = np.asarray([edge.modal_score for edge in graph.edges], dtype=float)
    jumps = np.asarray([edge.delta_omega_norm for edge in graph.edges], dtype=float)
    unmatched = 0
    for pair in graph.pairs:
        unmatched += int(sum(pair.transport.source_unmatched))
        unmatched += int(sum(pair.transport.target_unmatched))
    return {
        "min_score": float(np.min(scores)) if scores.size else np.nan,
        "median_score": float(np.median(scores)) if scores.size else np.nan,
        "max_jump": float(np.max(jumps)) if jumps.size else np.nan,
        "unmatched": int(unmatched),
        "edges": int(len(graph.edges)),
    }


def assert_gamma_doublet(graph):
    mid = len(graph.layers) // 2
    layer = graph.layers[mid]
    doublets = [
        (j, mode)
        for j, mode in enumerate(layer.modes)
        if mode.multiplicity == 2 and abs(mode.omega_norm - 1.06458) < 5e-3
    ]
    if len(doublets) != 1:
        raise AssertionError(f"expected exactly one Gamma doublet, found {len(doublets)}")
    j, _ = doublets[0]
    incoming = [e for e in graph.edges if e.right_layer == mid and e.right_event == j]
    outgoing = [e for e in graph.edges if e.left_layer == mid and e.left_event == j]
    if sum(e.dimensions for e in incoming) != 2:
        raise AssertionError("Gamma doublet does not receive two modal dimensions")
    if sum(e.dimensions for e in outgoing) != 2:
        raise AssertionError("Gamma doublet does not emit two modal dimensions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-suma", type=int, default=40)
    parser.add_argument("--ngrid", type=int, default=100)
    args = parser.parse_args()

    red = build_reference_red(args.n_suma)
    fine_kp, fine_graph = make_fine_graph(red, args.ngrid)
    if not fine_graph.stabilised or not fine_graph.complete_under_policy:
        raise SystemExit("the 41-point reference graph is not complete")

    results = {}
    print("dk convergence audit around Gamma")
    print("=" * 108)
    print("local roots are discovered once on Nk=41; Nk=21 and 11 reuse exact nested subsets")
    for nk in (11, 21, 41):
        kp, graph = nested_graph(red, fine_kp, fine_graph, nk)
        results[nk] = (kp, graph)
        metrics = graph_metrics(graph)
        print(
            f"Nk={nk:2d} dk/pi={(kp[1]-kp[0]):.6f} "
            f"complete={graph.complete_under_policy} edges={metrics['edges']} "
            f"unmatched={metrics['unmatched']} min_score={metrics['min_score']:.6f} "
            f"median_score={metrics['median_score']:.6f} max_dw={metrics['max_jump']:.6f}"
        )
        if not graph.complete_under_policy:
            raise SystemExit(f"Nk={nk} modal graph is not complete")
        if metrics["unmatched"] != 0:
            raise SystemExit(f"Nk={nk} contains unmatched modal dimensions")
        assert_gamma_doublet(graph)

    print("\nnested-grid topology")
    print("-" * 108)
    for nc, nf in ((11, 21), (21, 41)):
        ratio = (nf - 1) // (nc - 1)
        checked = compare_coarse_edges_to_fine(results[nc][1], results[nf][1], ratio)
        print(f"Nk={nc}->{nf}: refinement_ratio={ratio} coarse_links_reproduced={checked}")

    max_jumps = [graph_metrics(results[n][1])["max_jump"] for n in (11, 21, 41)]
    if not (max_jumps[1] < max_jumps[0] and max_jumps[2] < max_jumps[1]):
        raise SystemExit(f"maximum adjacent frequency jump did not decrease: {max_jumps}")

    min_scores = [graph_metrics(results[n][1])["min_score"] for n in (11, 21, 41)]
    print(f"\nmin modal scores Nk=11,21,41: {[round(v, 6) for v in min_scores]}")
    print("PASS: modal topology is stable under dk refinement in the certified Gamma sector")


if __name__ == "__main__":
    main()
