"""Audit convergence of modal tracking with respect to Bloch-path spacing.

The test is deliberately local to the already-certified spectral sector around
Gamma.  It does not attempt to validate the acoustic limit or psi != 0.

For nested grids with 11, 21 and 41 points on k/pi in [0.75, 1.25], we require:
  * every graph is complete under the same modal policy;
  * local certified spectra agree at all common k points;
  * multiplicities agree at common k points;
  * every coarse-grid modal edge is reachable through the finer graph;
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
from modal_path import build_modal_path_graph

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


def make_graph(red, nk: int, ngrid: int):
    kp = np.linspace(0.75, 1.25, int(nk))
    k_values = kp * np.pi / red.a
    graph = build_modal_path_graph(
        red,
        k_values,
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


def layer_signature(layer):
    return tuple((float(mode.omega_norm), int(mode.multiplicity)) for mode in layer.modes)


def compare_common_layers(coarse, fine, *, freq_tol: float = 5e-5):
    nc = len(coarse.layers)
    nf = len(fine.layers)
    ratio_num = nf - 1
    ratio_den = nc - 1
    if ratio_num % ratio_den != 0:
        raise AssertionError(f"grids are not nested: {nc} vs {nf}")
    ratio = ratio_num // ratio_den

    max_drift = 0.0
    for ic, layer_c in enumerate(coarse.layers):
        layer_f = fine.layers[ic * ratio]
        sig_c = layer_signature(layer_c)
        sig_f = layer_signature(layer_f)
        if len(sig_c) != len(sig_f):
            raise AssertionError(
                f"event-count mismatch at common layer {ic}: {len(sig_c)} vs {len(sig_f)}"
            )
        for j, ((wc, mc), (wf, mf)) in enumerate(zip(sig_c, sig_f)):
            if mc != mf:
                raise AssertionError(
                    f"multiplicity mismatch at common layer {ic}, event {j}: {mc} vs {mf}"
                )
            drift = abs(wc - wf)
            max_drift = max(max_drift, drift)
            if drift > freq_tol:
                raise AssertionError(
                    f"frequency drift {drift:.3e} exceeds {freq_tol:.3e} "
                    f"at common layer {ic}, event {j}"
                )
    return ratio, max_drift


def reachable_pairs(graph, left_layer: int, right_layer: int):
    """Return endpoint pairs connected by at least one directed modal path."""
    if right_layer <= left_layer:
        raise ValueError("right_layer must be larger than left_layer")

    current = {(event, event) for event in range(len(graph.layers[left_layer].modes))}
    # Store (origin_event, current_event).
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
        "roots_added": int(graph.roots_added),
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
    grids = (11, 21, 41)
    results = {}

    print("dk convergence audit around Gamma")
    print("=" * 108)
    for nk in grids:
        kp, graph = make_graph(red, nk, args.ngrid)
        results[nk] = (kp, graph)
        metrics = graph_metrics(graph)
        print(
            f"Nk={nk:2d} dk/pi={(kp[1]-kp[0]):.6f} "
            f"complete={graph.complete_under_policy} stabilised={graph.stabilised} "
            f"edges={metrics['edges']} roots_added={metrics['roots_added']} "
            f"unmatched={metrics['unmatched']} min_score={metrics['min_score']:.6f} "
            f"median_score={metrics['median_score']:.6f} max_dw={metrics['max_jump']:.6f}"
        )
        if not graph.stabilised or not graph.complete_under_policy:
            raise SystemExit(f"Nk={nk} modal graph is not complete")
        if metrics["unmatched"] != 0:
            raise SystemExit(f"Nk={nk} contains unmatched modal dimensions")
        assert_gamma_doublet(graph)

    comparisons = ((11, 21), (21, 41))
    print("\nnested-grid consistency")
    print("-" * 108)
    for nc, nf in comparisons:
        ratio, max_drift = compare_common_layers(results[nc][1], results[nf][1])
        checked = compare_coarse_edges_to_fine(results[nc][1], results[nf][1], ratio)
        print(
            f"Nk={nc}->{nf}: refinement_ratio={ratio} "
            f"max_common_frequency_drift={max_drift:.3e} "
            f"coarse_links_reproduced={checked}"
        )

    max_jumps = [graph_metrics(results[n][1])["max_jump"] for n in grids]
    if not (max_jumps[1] < max_jumps[0] and max_jumps[2] < max_jumps[1]):
        raise SystemExit(f"maximum adjacent frequency jump did not decrease: {max_jumps}")

    print("\nPASS: modal topology is stable under dk refinement in the certified Gamma sector")


if __name__ == "__main__":
    main()
