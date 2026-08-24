"""Compare Miguel's raw historical full-grid solver with the certified psi=0 baseline.

The comparison is deliberately made *before* postprocess_miguel.  The question
is not whether heuristics can make the historical output look cleaner, but what
zeros_longitudinal_fullgrid itself returns on the same physical problem.

Inputs
------
A directory produced by certified_bands.py containing bandas.csv and
conexiones.csv.

Outputs
-------
original_raw.csv
    Every finite point returned by zeros_longitudinal_fullgrid in the reported
    spectral window, including its residual when re-evaluated with the
    certified secular matrix.
comparison.csv
    One row per original point with nearest/matched certified event information.
missing_certified.csv
    Certified modal dimensions for which the raw historical solver supplied no
    frequency within the matching tolerance at the same k.
comparacion_overlay.png
    Raw historical points and certified spectral events on the same axes.
comparacion_resumen.json
    Machine-readable counts and parameters.

This script does not run the historical post-processing layer and does not
silently repair its output.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from Bandas_Tools import Red
from certified_bands import BandRunConfig, build_red as build_certified_red
from rootfinder_certified import certify_frequency


def build_legacy_red(config: BandRunConfig, nk: int, nbands: int, output_dir: Path) -> Red:
    r = Red(comp=["matrix", "inclusion"])
    r.dens = [config.dens_matrix, config.dens_inclusion]
    r.vel0 = [config.cl_matrix, config.ct_matrix]
    r.vels = [config.cl_inclusion, config.ct_inclusion]
    r.filling = config.filling
    r.cut = config.cut
    r.nbands = int(nbands)
    r.nk = int(nk)
    r.n_suma = config.n_suma
    r.lattice = config.lattice
    r.psi = config.psi
    r.a = config.a
    r._set_k_end()
    r.cond_borde = config.cond_borde
    r.imag_tol = 0.8
    r.sol_tol = 1e-2
    r.asign_param()
    r.r1 = config.r1_over_a * r.a
    r.r2 = config.r2_over_a * r.a

    # Prevent the historical method from writing into a user Documents tree.
    solver_dir = output_dir / "legacy_solver_files"
    solver_dir.mkdir(parents=True, exist_ok=True)
    r.foldername = str(solver_dir)
    r.frecfolder = str(solver_dir / "psi_0")
    os.makedirs(r.frecfolder, exist_ok=True)
    return r


def read_certified_events(path: Path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["region"] != "core":
                continue
            rows.append(
                {
                    "layer": int(row["layer"]),
                    "event": int(row["event"]),
                    "k_over_pi": float(row["k_over_pi"]),
                    "omega_norm": float(row["omega_norm"]),
                    "multiplicity": int(row["multiplicity"]),
                    "sigma_min": float(row["sigma_min"]),
                }
            )
    return rows


def expanded_certified_at_k(rows, k_over_pi: float, atol: float = 2e-10):
    expanded = []
    for row in rows:
        if not np.isclose(row["k_over_pi"], k_over_pi, atol=atol, rtol=0.0):
            continue
        for copy in range(row["multiplicity"]):
            item = dict(row)
            item["multiplicity_copy"] = copy
            expanded.append(item)
    return expanded


def extract_original_points(red: Red, config: BandRunConfig):
    if red.omega_longitudinal is None:
        raise RuntimeError("zeros_longitudinal_fullgrid did not populate omega_longitudinal")
    scale = red.a / (2.0 * np.pi * config.Ct0)
    rows = []
    for ik, k in enumerate(np.asarray(red.k, dtype=float)):
        kpi = float(k * red.a / np.pi)
        for ib in range(red.omega_longitudinal.shape[1]):
            wr = float(red.omega_longitudinal[ik, ib, 0])
            wi = float(red.omega_longitudinal[ik, ib, 1])
            if not np.isfinite(wr):
                continue
            wn = wr * scale
            if not (config.w_min <= wn <= config.w_max):
                continue
            rows.append(
                {
                    "k_index": ik,
                    "band_column": ib,
                    "k_value": float(k),
                    "k_over_pi": kpi,
                    "omega": wr,
                    "omega_norm": wn,
                    "omega_imag": wi,
                    "omega_imag_norm": wi * scale,
                }
            )
    return rows


def compare_at_common_k(original, certified, match_tol: float):
    comparisons = []
    missing = []
    original_by_ik = {}
    for row in original:
        original_by_ik.setdefault(row["k_index"], []).append(row)

    for ik, local_original in sorted(original_by_ik.items()):
        kpi = local_original[0]["k_over_pi"]
        local_cert = expanded_certified_at_k(certified, kpi)
        no = len(local_original)
        nc = len(local_cert)
        matched_o = set()
        matched_c = set()
        assignment = {}
        if no and nc:
            cost = np.abs(
                np.asarray([r["omega_norm"] for r in local_original])[:, None]
                - np.asarray([r["omega_norm"] for r in local_cert])[None, :]
            )
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                delta = float(cost[i, j])
                if delta <= float(match_tol):
                    matched_o.add(int(i))
                    matched_c.add(int(j))
                    assignment[int(i)] = (int(j), delta)

        for i, row in enumerate(local_original):
            nearest_delta = np.inf
            nearest_w = np.nan
            if local_cert:
                deltas = np.abs(
                    np.asarray([c["omega_norm"] for c in local_cert]) - row["omega_norm"]
                )
                jn = int(np.argmin(deltas))
                nearest_delta = float(deltas[jn])
                nearest_w = float(local_cert[jn]["omega_norm"])
            match = assignment.get(i)
            comparisons.append(
                {
                    **row,
                    "matched_to_certified": match is not None,
                    "matched_certified_omega_norm": (
                        float(local_cert[match[0]]["omega_norm"]) if match else np.nan
                    ),
                    "match_delta": float(match[1]) if match else np.nan,
                    "nearest_certified_omega_norm": nearest_w,
                    "nearest_certified_delta": nearest_delta,
                }
            )

        for j, row in enumerate(local_cert):
            if j not in matched_c:
                missing.append(
                    {
                        "k_index": ik,
                        "k_over_pi": kpi,
                        "event": row["event"],
                        "multiplicity_copy": row["multiplicity_copy"],
                        "omega_norm": row["omega_norm"],
                        "sigma_min": row["sigma_min"],
                    }
                )

    return comparisons, missing


def attach_certified_residuals(comparisons, certified_red, config: BandRunConfig):
    for row in comparisons:
        diag = certify_frequency(
            certified_red,
            row["k_value"],
            row["omega"],
            config.Ct0,
            sigma_accept=config.sigma_accept,
            multiplicity_tol=config.multiplicity_tol,
        )
        row["certified_svd_accepted_at_original_frequency"] = bool(diag["accepted"])
        row["certified_sigma_min"] = float(diag["sigma_min"])
        row["certified_sigma_min_raw"] = float(diag["sigma_min_raw"])
        row["certified_sigma_min_balanced"] = float(diag["sigma_min_balanced"])
        row["certified_multiplicity_at_original_frequency"] = int(diag["multiplicity"])
    return comparisons


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_overlay(path: Path, original, certified, config: BandRunConfig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    if original:
        ax.scatter(
            [r["k_over_pi"] for r in original],
            [r["omega_norm"] for r in original],
            marker="x",
            s=24,
            label="Miguel raw zeros_longitudinal_fullgrid",
        )
    if certified:
        ax.scatter(
            [r["k_over_pi"] for r in certified],
            [r["omega_norm"] for r in certified],
            marker="o",
            facecolors="none",
            s=34,
            label="certified spectral events",
        )
    ax.axvline(1.0, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(2.0, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(config.w_min, config.w_max)
    ax.set_xticks([0.0, 1.0, 2.0, 3.0], ["M", r"$\Gamma$", "X", "M"])
    ax.set_xlabel("Bloch path")
    ax.set_ylabel(r"$\omega a/(2\pi C_{t0})$")
    ax.set_title("Raw historical solver vs certified reference")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--certified-dir", default="data/certified_baseline_v1")
    p.add_argument("--output-dir", default="data/comparison_miguel_vs_certified")
    p.add_argument("--nk", type=int, default=25)
    p.add_argument("--nbands", type=int, default=8)
    p.add_argument("--ventanas-por-unidad", type=int, default=100)
    p.add_argument("--match-tol", type=float, default=0.01)
    args = p.parse_args()

    config = BandRunConfig()
    certified_dir = Path(args.certified_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    certified = read_certified_events(certified_dir / "bandas.csv")
    if not certified:
        raise SystemExit("certified bandas.csv contains no core events")

    legacy = build_legacy_red(config, args.nk, args.nbands, output_dir)
    print(
        "running raw zeros_longitudinal_fullgrid: "
        f"nk={args.nk} cut={config.cut} n_suma={config.n_suma} "
        f"window=[{config.w_min},{config.w_max}]"
    )
    legacy.zeros_longitudinal_fullgrid(
        C_l0=config.Ct0,
        ventanas_por_unidad=args.ventanas_por_unidad,
        w_norm_min=config.w_min,
        w_norm_max=config.w_max,
    )

    original = extract_original_points(legacy, config)
    certified_red = build_certified_red(config)
    comparison, missing = compare_at_common_k(original, certified, args.match_tol)
    attach_certified_residuals(comparison, certified_red, config)

    write_rows(output_dir / "original_raw.csv", original)
    write_rows(output_dir / "comparison.csv", comparison)
    write_rows(output_dir / "missing_certified.csv", missing)
    plot_overlay(output_dir / "comparacion_overlay.png", original, certified, config)

    matched = sum(bool(r["matched_to_certified"]) for r in comparison)
    svd_ok = sum(bool(r["certified_svd_accepted_at_original_frequency"]) for r in comparison)
    summary = {
        "comparison_scope": "raw zeros_longitudinal_fullgrid before postprocess_miguel",
        "parameters": {
            "nk": args.nk,
            "nbands": args.nbands,
            "ventanas_por_unidad": args.ventanas_por_unidad,
            "match_tol_norm": args.match_tol,
            "cut": config.cut,
            "n_suma": config.n_suma,
            "w_min": config.w_min,
            "w_max": config.w_max,
        },
        "original_raw_points_in_window": len(original),
        "original_points_matched_to_certified_within_tolerance": matched,
        "original_points_unmatched_to_certified": len(comparison) - matched,
        "original_points_passing_dual_svd_at_their_returned_frequency": svd_ok,
        "certified_modal_dimensions_missing_from_original": len(missing),
        "note": (
            "SVD rejection at the exact historical frequency can reflect either a false candidate "
            "or inadequate root refinement; nearest certified delta in comparison.csv separates these cases."
        ),
    }
    (output_dir / "comparacion_resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
