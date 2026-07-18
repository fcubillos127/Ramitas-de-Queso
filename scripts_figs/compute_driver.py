"""Calcula bandas (det-min) para 5 psi y guarda npz. Uso: python compute_driver.py <lattice> <out.npz>"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bandcalc import build_red, compute_bands_eig, CT0

LAT = sys.argv[1] if len(sys.argv) > 1 else "sq"
OUT = sys.argv[2] if len(sys.argv) > 2 else "bands_%s.npz" % LAT
PSIS = [0.0, 0.2, 0.4, 0.6, 0.8]
NK, WMAX, NGRID, CUT = 70, 1.4, 1100, 7

data = {"lattice": LAT, "psis": np.array(PSIS), "a": 1.0, "Ct0": CT0, "wmax": WMAX}
t_all = time.perf_counter()
for i, psi in enumerate(PSIS):
    r = build_red(LAT, psi, cut=CUT, nk=NK)
    t0 = time.perf_counter()
    k_arr, wn, im = compute_bands_eig(r, nk=NK, wmax=WMAX, ngrid=NGRID)
    data["k_%d" % i] = k_arr
    data["wn_%d" % i] = wn
    data["im_%d" % i] = im
    np.savez(OUT, **data)
    print("[%s] psi=%.1f  %.1f s  (%d pts)" % (LAT, psi, time.perf_counter()-t0,
                                               np.sum(np.isfinite(wn))), flush=True)
print("[%s] TOTAL %.1f s -> %s" % (LAT, time.perf_counter()-t_all, OUT), flush=True)
