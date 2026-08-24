"""User-facing driver for the audited psi=0 square-lattice band solver.

The numerical audit established a finite baseline rather than a universal
claim: square lattice, psi=0, hollow circular inclusion, and the reported
frequency window 0.70 <= omega*a/(2*pi*Ct0) <= 1.40.  This module turns that
audit pipeline into a reproducible command-line tool while preserving the
scientific distinctions that motivated the audit:

* roots are spectral events with multiplicity, not frequency-sorted columns;
* local roots must pass the dual-SVD certificate;
* band identity is transported by modal subspaces;
* a branch crossing the displayed frequency boundary is followed in a guard
  band instead of being labelled discontinuous;
* the rapidly rotating two-mode group immediately after X is represented as a
  joint cluster, without inventing a unique branch identity inside it.

psi != 0 is intentionally rejected here.  Its deformed-scatterer T-matrix has
not yet been re-derived/certified and therefore must not inherit the psi=0
quality label by accident.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from certified_solver import CertifiedRed
from modal_clusters import ClusterContinuity, certify_cluster_continuity
from modal_path import ModalPathGraph, build_modal_path_graph


@dataclass(frozen=True)
class BandRunConfig:
    lattice: str = "sq"
    psi: float = 0.0
    cut: int = 4
    n_suma: int = 40
    points_per_segment: int = 9
    ngrid: int = 110
    w_min: float = 0.70
    w_max: float = 1.40
    search_w_max: float = 1.50
    a: float = 1.0
    filling: float = 0.5
    r1_over_a: float = 0.45
    r2_over_a: float = 0.50
    dens_matrix: float = 1150.0
    dens_inclusion: float = 1250.0
    cl_matrix: float = 295.0
    ct_matrix: float = 295.0
    cl_inclusion: float = 894.0
    ct_inclusion: float = 894.0
    cond_borde: str = "hollow"
    sigma_accept: float = 1e-6
    multiplicity_tol: float = 1e-5
    min_principal_cosine: float = 0.65
    max_delta_omega_norm: float = 0.10

    @property
    def Ct0(self) -> float:
        return float(self.ct_matrix)


@dataclass(frozen=True)
class BandRunResult:
    config: BandRunConfig
    red: CertifiedRed
    k_over_pi: np.ndarray
    graph: ModalPathGraph
    unresolved_core_pairs: tuple
    x_cluster: ClusterContinuity | None
    m_endpoint_ok: bool
    m_endpoint_max_drift: float
    gamma_reference_ok: bool | None
    baseline_scope: bool
    status: str


BASELINE_REFERENCE = BandRunConfig()


def _close(a: float, b: float, atol: float = 1e-12) -> bool:
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=0.0))


def matches_baseline_scope(config: BandRunConfig) -> bool:
    """Whether the run matches the parameter set actually closed in the audit."""
    ref = BASELINE_REFERENCE
    scalar_names = (
        "psi", "cut", "n_suma", "points_per_segment", "ngrid", "w_min", "w_max",
        "search_w_max", "a", "filling", "r1_over_a", "r2_over_a",
        "dens_matrix", "dens_inclusion", "cl_matrix", "ct_matrix",
        "cl_inclusion", "ct_inclusion", "sigma_accept", "multiplicity_tol",
        "min_principal_cosine", "max_delta_omega_norm",
    )
    return bool(
        config.lattice == ref.lattice
        and config.cond_borde == ref.cond_borde
        and all(_close(getattr(config, name), getattr(ref, name)) for name in scalar_names)
    )


def validate_supported_scope(config: BandRunConfig) -> None:
    if config.lattice != "sq":
        raise NotImplementedError(
            "certified_bands currently exposes only the audited square-lattice path"
        )
    if not _close(config.psi, 0.0):
        raise NotImplementedError(
            "psi != 0 is intentionally blocked until the deformed-scatterer T-matrix is re-derived"
        )
    if config.cond_borde != "hollow":
        raise NotImplementedError(
            "the current user-facing certified driver is scoped to the audited hollow psi=0 problem"
        )
    if config.search_w_max <= config.w_max:
        raise ValueError("search_w_max must exceed w_max to provide a spectral guard band")
    if config.points_per_segment < 3:
        raise ValueError("points_per_segment must be >= 3")
    if config.ngrid < 5:
        raise ValueError("ngrid must be >= 5")


def build_red(config: BandRunConfig) -> CertifiedRed:
    """Construct CertifiedRed using the same parameter semantics as Miguel's Red."""
    validate_supported_scope(config)
    r = CertifiedRed(comp=["matrix", "inclusion"])
    r.dens = [float(config.dens_matrix), float(config.dens_inclusion)]
    r.vel0 = [float(config.cl_matrix), float(config.ct_matrix)]
    r.vels = [float(config.cl_inclusion), float(config.ct_inclusion)]
    r.filling = float(config.filling)
    r.cut = int(config.cut)
    r.nbands = 20
    r.nk = 3 * (int(config.points_per_segment) - 1) + 1
    r.n_suma = int(config.n_suma)
    r.lattice = config.lattice
    r.psi = float(config.psi)
    r.a = float(config.a)
    r._set_k_end()
    r.cond_borde = config.cond_borde
    r.imag_tol = 0.8
    r.sol_tol = 1e-2
    r.asign_param()
    # asign_param recomputes r1 from filling, so manual geometry comes last.
    r.r1 = float(config.r1_over_a) * r.a
    r.r2 = float(config.r2_over_a) * r.a
    return r


def path_grid(points_per_segment: int) -> np.ndarray:
    """M-Gamma-X-M in k*a/pi, plus locally justified half steps."""
    p = int(points_per_segment)
    if p < 3:
        raise ValueError("points_per_segment must be >= 3")
    mg = np.linspace(0.0, 1.0, p)
    gx = np.linspace(1.0, 2.0, p)[1:]
    xm = np.linspace(2.0, 3.0, p)[1:]
    base = np.concatenate([mg, gx, xm])
    # 0.0625 resolves the M doublet; 2.0625 resolves/diagnoses the X cluster.
    return np.unique(np.concatenate([base, np.asarray([0.0625, 2.0625])]))


def _index_at(kp: np.ndarray, value: float, tol: float = 1e-12) -> int:
    idx = np.flatnonzero(np.isclose(kp, float(value), atol=tol, rtol=0.0))
    if len(idx) != 1:
        raise ValueError(f"expected one path point at k*a/pi={value}, found {idx}")
    return int(idx[0])


def _in_core(mode, config: BandRunConfig) -> bool:
    return float(config.w_min) <= float(mode.omega_norm) <= float(config.w_max)


def _core_signature(layer, config: BandRunConfig):
    return tuple(
        (float(mode.omega_norm), int(mode.multiplicity))
        for mode in layer.modes
        if _in_core(mode, config)
    )


def compare_M_endpoints(graph: ModalPathGraph, config: BandRunConfig, tol: float = 5e-5):
    left = _core_signature(graph.layers[0], config)
    right = _core_signature(graph.layers[-1], config)
    if len(left) != len(right):
        return False, np.inf, f"event counts differ: {len(left)} vs {len(right)}"
    drift = 0.0
    for j, ((wl, ml), (wr, mr)) in enumerate(zip(left, right)):
        if ml != mr:
            return False, np.inf, f"multiplicity mismatch at event {j}: {ml} vs {mr}"
        delta = abs(wl - wr)
        drift = max(drift, delta)
        if delta > tol:
            return False, drift, f"frequency mismatch at event {j}: {wl} vs {wr}"
    return True, drift, "ok"


def _core_unmatched_indices(layer, unmatched, config: BandRunConfig):
    return tuple(
        i
        for i, (mode, missing) in enumerate(zip(layer.modes, unmatched))
        if int(missing) > 0 and _in_core(mode, config)
    )


def _upper_x_cluster_indices(layer):
    idx = [i for i, mode in enumerate(layer.modes) if 0.95 <= mode.omega_norm <= 1.10]
    if len(idx) != 2:
        raise ValueError(f"expected two upper X-cluster events, found {idx}")
    return tuple(idx)


def certify_core_continuity(kp, graph, config: BandRunConfig):
    """Classify unresolved core transport, allowing only the audited X cluster rescue."""
    unresolved = []
    x_cluster = None
    for pair in graph.pairs:
        left = graph.layers[pair.left_index]
        right = graph.layers[pair.right_index]
        source_bad = _core_unmatched_indices(left, pair.transport.source_unmatched, config)
        target_bad = _core_unmatched_indices(right, pair.transport.target_unmatched, config)
        if not source_bad and not target_bad:
            continue

        if np.isclose(kp[pair.left_index], 2.0) and np.isclose(kp[pair.right_index], 2.0625):
            source_cluster = _upper_x_cluster_indices(left)
            target_cluster = _upper_x_cluster_indices(right)
            candidate = certify_cluster_continuity(
                left.modes,
                right.modes,
                source_cluster,
                target_cluster,
                min_principal_cosine=config.min_principal_cosine,
                max_center_shift_norm=config.max_delta_omega_norm,
            )
            if (
                candidate.certified
                and set(source_bad).issubset(source_cluster)
                and set(target_bad).issubset(target_cluster)
            ):
                x_cluster = candidate
                continue

        unresolved.append((pair.left_index, pair.right_index, source_bad, target_bad))
    return tuple(unresolved), x_cluster


def _gamma_reference_ok(kp, graph, config: BandRunConfig) -> bool | None:
    if not matches_baseline_scope(config):
        return None
    gamma = graph.layers[_index_at(kp, 1.0)]
    expected = [
        mode for mode in gamma.modes
        if _in_core(mode, config)
        and mode.multiplicity == 2
        and abs(mode.omega_norm - 1.0645835) < 5e-3
    ]
    return len(expected) == 1


def run_certified_bands(config: BandRunConfig = BandRunConfig()) -> BandRunResult:
    """Run the audited event/root/transport pipeline and return structured results."""
    red = build_red(config)
    kp = path_grid(config.points_per_segment)
    graph = build_modal_path_graph(
        red,
        kp * np.pi / red.a,
        config.Ct0,
        w_norm_min=config.w_min,
        w_norm_max=config.search_w_max,
        ngrid=config.ngrid,
        max_sweeps=3,
        search_half_width_norm=config.max_delta_omega_norm,
        targeted_max_depth=5,
        completion_max_rounds=2,
        min_pair_affinity=0.15,
        transport_max_delta_omega_norm=config.max_delta_omega_norm,
        min_principal_cosine=config.min_principal_cosine,
        frequency_weight=0.05,
        finder_kwargs={
            "sigma_accept": config.sigma_accept,
            "multiplicity_tol": config.multiplicity_tol,
        },
    )
    unresolved, x_cluster = certify_core_continuity(kp, graph, config)
    m_ok, m_drift, _ = compare_M_endpoints(graph, config)
    gamma_ok = _gamma_reference_ok(kp, graph, config)
    baseline_scope = matches_baseline_scope(config)

    current_policy_complete = bool(
        graph.stabilised
        and not unresolved
        and m_ok
        and (x_cluster is None or x_cluster.certified)
    )
    baseline_certified = bool(
        baseline_scope
        and current_policy_complete
        and x_cluster is not None
        and x_cluster.certified
        and gamma_ok is True
    )
    if baseline_certified:
        status = "CERTIFIED_BASELINE_V1"
    elif current_policy_complete:
        status = "COMPLETE_UNCERTIFIED_PARAMETER_SCOPE"
    else:
        status = "INCOMPLETE_UNDER_CURRENT_POLICY"

    return BandRunResult(
        config=config,
        red=red,
        k_over_pi=kp,
        graph=graph,
        unresolved_core_pairs=unresolved,
        x_cluster=x_cluster,
        m_endpoint_ok=bool(m_ok),
        m_endpoint_max_drift=float(m_drift),
        gamma_reference_ok=gamma_ok,
        baseline_scope=baseline_scope,
        status=status,
    )


def _high_symmetry(result: BandRunResult) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for label, value in (("M_start", 0.0), ("Gamma", 1.0), ("X", 2.0), ("M_end", 3.0)):
        layer = result.graph.layers[_index_at(result.k_over_pi, value)]
        out[label] = [
            {"omega_norm": float(m.omega_norm), "multiplicity": int(m.multiplicity)}
            for m in layer.modes if _in_core(m, result.config)
        ]
    return out


def diagnostics_dict(result: BandRunResult) -> dict[str, Any]:
    cluster = None
    if result.x_cluster is not None:
        cluster = {
            "certified": bool(result.x_cluster.certified),
            "principal_cosines": [float(v) for v in result.x_cluster.metrics.principal_cosines],
            "affinity": float(result.x_cluster.metrics.affinity),
            "center_shift": float(result.x_cluster.center_shift),
        }
    return {
        "status": result.status,
        "baseline_scope": bool(result.baseline_scope),
        "config": asdict(result.config),
        "path_points": int(len(result.k_over_pi)),
        "stabilised": bool(result.graph.stabilised),
        "raw_graph_complete": bool(result.graph.complete_under_policy),
        "completion_sweeps": int(result.graph.completion_sweeps),
        "roots_added_by_completion": int(result.graph.roots_added),
        "unresolved_core_pairs": [
            {
                "left_layer": int(a), "right_layer": int(b),
                "source_events": [int(v) for v in s],
                "target_events": [int(v) for v in t],
            }
            for a, b, s, t in result.unresolved_core_pairs
        ],
        "x_cluster": cluster,
        "M_endpoint_closure": {
            "ok": bool(result.m_endpoint_ok),
            "max_frequency_drift": float(result.m_endpoint_max_drift),
        },
        "gamma_reference_ok": result.gamma_reference_ok,
        "high_symmetry": _high_symmetry(result),
        "scope_note": (
            "CERTIFIED_BASELINE_V1 applies only to the exact baseline parameter profile. "
            "Other psi=0 parameter choices may run, but are labelled uncertified scope."
        ),
    }


def export_events_csv(result: BandRunResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layer", "event", "k_over_pi", "k_value", "omega_norm", "multiplicity",
        "sigma_min", "sigma_min_raw", "sigma_min_balanced", "det_abs", "source", "region",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, layer in enumerate(result.graph.layers):
            for j, mode in enumerate(layer.modes):
                root = mode.root
                writer.writerow({
                    "layer": i,
                    "event": j,
                    "k_over_pi": f"{result.k_over_pi[i]:.12g}",
                    "k_value": f"{layer.k:.16g}",
                    "omega_norm": f"{mode.omega_norm:.16g}",
                    "multiplicity": int(mode.multiplicity),
                    "sigma_min": f"{root.sigma_min:.16g}",
                    "sigma_min_raw": f"{root.sigma_min_raw:.16g}",
                    "sigma_min_balanced": f"{root.sigma_min_balanced:.16g}",
                    "det_abs": f"{root.det_abs:.16g}",
                    "source": root.source,
                    "region": "core" if _in_core(mode, result.config) else "guard",
                })
    return path


def export_connections_csv(result: BandRunResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kind", "left_layer", "left_event", "right_layer", "right_event",
        "dimensions", "delta_omega_norm", "modal_score", "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for edge in result.graph.edges:
            left = result.graph.layers[edge.left_layer].modes[edge.left_event]
            right = result.graph.layers[edge.right_layer].modes[edge.right_event]
            if not (_in_core(left, result.config) and _in_core(right, result.config)):
                continue
            writer.writerow({
                "kind": "individual_modal_transport",
                "left_layer": edge.left_layer,
                "left_event": edge.left_event,
                "right_layer": edge.right_layer,
                "right_event": edge.right_event,
                "dimensions": edge.dimensions,
                "delta_omega_norm": f"{edge.delta_omega_norm:.16g}",
                "modal_score": f"{edge.modal_score:.16g}",
                "note": "unique event-level transport",
            })
        if result.x_cluster is not None:
            left_layer = _index_at(result.k_over_pi, 2.0)
            right_layer = _index_at(result.k_over_pi, 2.0625)
            writer.writerow({
                "kind": "joint_modal_cluster",
                "left_layer": left_layer,
                "left_event": "",
                "right_layer": right_layer,
                "right_event": "",
                "dimensions": result.x_cluster.source.dimension,
                "delta_omega_norm": f"{result.x_cluster.center_shift:.16g}",
                "modal_score": f"{result.x_cluster.metrics.affinity:.16g}",
                "note": "joint subspace continuous; branch identity inside cluster intentionally unresolved",
            })
    return path


def plot_result(result: BandRunResult, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    # Draw only individually certified modal-transport edges.
    for edge in result.graph.edges:
        left = result.graph.layers[edge.left_layer].modes[edge.left_event]
        right = result.graph.layers[edge.right_layer].modes[edge.right_event]
        if not (_in_core(left, result.config) and _in_core(right, result.config)):
            continue
        ax.plot(
            [result.k_over_pi[edge.left_layer], result.k_over_pi[edge.right_layer]],
            [left.omega_norm, right.omega_norm],
            linewidth=1.0,
        )

    for i, layer in enumerate(result.graph.layers):
        for mode in layer.modes:
            if _in_core(mode, result.config):
                size = 18.0 + 20.0 * (int(mode.multiplicity) - 1)
                ax.scatter(result.k_over_pi[i], mode.omega_norm, s=size, zorder=3)

    # Do not invent a one-to-one edge through the X cluster.  Show the four
    # possible connections as a dotted visual envelope and state what it means.
    if result.x_cluster is not None and result.x_cluster.certified:
        il = _index_at(result.k_over_pi, 2.0)
        ir = _index_at(result.k_over_pi, 2.0625)
        source_idx = result.x_cluster.source.event_indices
        target_idx = result.x_cluster.target.event_indices
        first = True
        for a in source_idx:
            for b in target_idx:
                ax.plot(
                    [result.k_over_pi[il], result.k_over_pi[ir]],
                    [result.graph.layers[il].modes[a].omega_norm,
                     result.graph.layers[ir].modes[b].omega_norm],
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.45,
                    label="joint 2D cluster; internal identity unresolved" if first else None,
                )
                first = False

    ax.axvline(1.0, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(2.0, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(result.config.w_min, result.config.w_max)
    ax.set_xticks([0.0, 1.0, 2.0, 3.0], ["M", r"$\Gamma$", "X", "M"])
    ax.set_xlabel("Bloch path")
    ax.set_ylabel(r"$\omega a/(2\pi C_{t0})$")
    ax.set_title(f"Certified spectral events — {result.status}")
    ax.grid(alpha=0.2)
    if result.x_cluster is not None:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def export_run(result: BandRunResult, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "bands_csv": export_events_csv(result, out / "bandas.csv"),
        "connections_csv": export_connections_csv(result, out / "conexiones.csv"),
        "diagnostics_json": out / "diagnostico.json",
        "figure_png": plot_result(result, out / "bandas.png"),
    }
    files["diagnostics_json"].write_text(
        json.dumps(diagnostics_dict(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return files


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audited psi=0 square-lattice band solver")
    p.add_argument("--output-dir", default="data/certified_baseline_v1")
    p.add_argument("--cut", type=int, default=4)
    p.add_argument("--n-suma", type=int, default=40)
    p.add_argument("--points-per-segment", type=int, default=9)
    p.add_argument("--ngrid", type=int, default=110)
    p.add_argument("--w-min", type=float, default=0.70)
    p.add_argument("--w-max", type=float, default=1.40)
    p.add_argument("--search-w-max", type=float, default=1.50)
    p.add_argument("--psi", type=float, default=0.0)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--filling", type=float, default=0.5)
    p.add_argument("--r1-over-a", type=float, default=0.45)
    p.add_argument("--r2-over-a", type=float, default=0.50)
    return p


def main() -> None:
    args = _parser().parse_args()
    config = BandRunConfig(
        psi=args.psi,
        cut=args.cut,
        n_suma=args.n_suma,
        points_per_segment=args.points_per_segment,
        ngrid=args.ngrid,
        w_min=args.w_min,
        w_max=args.w_max,
        search_w_max=args.search_w_max,
        a=args.a,
        filling=args.filling,
        r1_over_a=args.r1_over_a,
        r2_over_a=args.r2_over_a,
    )
    result = run_certified_bands(config)
    files = export_run(result, args.output_dir)
    print(f"status={result.status}")
    print(f"baseline_scope={result.baseline_scope}")
    print(f"path_points={len(result.k_over_pi)} roots_added={result.graph.roots_added}")
    print(f"unresolved_core_pairs={len(result.unresolved_core_pairs)}")
    for name, path in files.items():
        print(f"{name}={path}")
    if result.status == "INCOMPLETE_UNDER_CURRENT_POLICY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
