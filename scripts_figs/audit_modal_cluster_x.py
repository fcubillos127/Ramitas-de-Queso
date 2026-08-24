"""Test whether the unresolved pair immediately after X is a rotating 2D cluster.

The event-level tracker leaves one of the two upper X modes unmatched between
k/pi=2.0 and 2.0625.  This audit asks a narrower question: do the *two upper
modes together* span a continuous two-dimensional modal subspace?  A positive
answer validates cluster continuity without claiming a unique branch identity
inside that cluster.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from certified_solver import CertifiedRed
from modal_clusters import certify_cluster_continuity
from modal_tracking import build_modal_spectrum, subspace_metrics
from rootfinder_certified import find_roots_at_k

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
    r.n_suma = 40
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


def spectrum(red, k_over_pi):
    k = float(k_over_pi) * np.pi / red.a
    roots = find_roots_at_k(
        red,
        k,
        CT0,
        w_norm_min=0.70,
        w_norm_max=1.40,
        ngrid=110,
        scan_eta_norm=1e-6,
        sigma_accept=1e-6,
        multiplicity_tol=1e-5,
    )
    return build_modal_spectrum(red, k, roots)


def upper_pair_indices(modes):
    indices = [i for i, mode in enumerate(modes) if 0.95 <= mode.omega_norm <= 1.10]
    if len(indices) != 2:
        raise SystemExit(f"expected exactly two upper X-cluster events, found {indices}")
    return indices


def main():
    red = build_reference_red()
    left = spectrum(red, 2.0)
    right = spectrum(red, 2.0625)
    il = upper_pair_indices(left)
    ir = upper_pair_indices(right)

    print("X-cluster continuity audit")
    print("=" * 88)
    print("left frequencies :", [round(left[i].omega_norm, 9) for i in il])
    print("right frequencies:", [round(right[i].omega_norm, 9) for i in ir])

    pairwise = np.zeros((2, 2), dtype=float)
    for a, i in enumerate(il):
        for b, j in enumerate(ir):
            pairwise[a, b] = subspace_metrics(left[i], right[j]).affinity
    print("pairwise simple-mode affinity matrix")
    print(np.array2string(pairwise, precision=6, suppress_small=False))

    result = certify_cluster_continuity(
        left,
        right,
        il,
        ir,
        min_principal_cosine=0.65,
        max_center_shift_norm=0.10,
    )
    print("cluster principal cosines:", np.array2string(result.metrics.principal_cosines, precision=8))
    print(f"cluster min cosine={result.metrics.min_principal_cosine:.8f}")
    print(f"cluster affinity={result.metrics.affinity:.8f}")
    print(f"cluster center shift={result.center_shift:.8f}")
    print(f"CLUSTER_CERTIFIED={result.certified}")

    if not result.certified:
        raise SystemExit("upper X pair is not continuous as a two-dimensional cluster")


if __name__ == "__main__":
    main()
