"""
Extractor de bandas por minimos de |det(T G0 - I)| sobre una grilla fija de omega.
Como la grilla de omega es la MISMA para todo k, el coeficiente T_n(omega) se cachea
y se reutiliza entre todos los k (T no depende de k), haciendo el calculo rapido
incluso con pre-deformacion (psi != 0). Fisicamente: las bandas son los ceros del
determinante, i.e. minimos profundos de |det|.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.signal import argrelmin
from Bandas_Tools import Red

CT0 = 295.0


def build_red(lattice, psi, cut=2, nk=90):
    r = Red(comp=["matriz", "inclusion"])
    r.dens = [1150, 1250]; r.vel0 = [295, 295]; r.vels = [894, 894]
    r.filling = 0.5; r.cut = cut; r.nbands = 12; r.nk = nk; r.n_suma = 5
    r.lattice = lattice; r.psi = psi; r.a = 1.0; r._set_k_end()
    r.cond_borde = "hollow"; r.imag_tol = 0.8; r.sol_tol = 1e-2
    r.asign_param(); r.r1 = 0.45; r.r2 = 0.5
    return r


def compute_bands(r, nk=90, wmax=1.4, ngrid=800, eta=1e-3, thr=0.15, maxbands=12):
    """Devuelve (k_arr, wn) con wn de forma (nk, maxbands), NaN-padded, en unidades ωa/2πCt0."""
    a = r.a
    k_arr = np.linspace(0.0, r.k_end, nk)
    wgrid = np.linspace(1e-3 * 2*np.pi*CT0/a, wmax * 2*np.pi*CT0/a, ngrid)
    dw = wgrid[1] - wgrid[0]
    wn = np.full((nk, maxbands), np.nan)
    for ik, k in enumerate(k_arr):
        mags = np.empty(ngrid)
        for j, w in enumerate(wgrid):
            re, im = r.determinant_longitudinal([w, eta], k, r.cut)
            mags[j] = abs(re + 1j*im)
        idx = argrelmin(mags, order=2)[0]
        idx = idx[mags[idx] < thr]
        ws = []
        for i in idx:
            if 1 <= i < ngrid-1:
                y0, y1, y2 = mags[i-1], mags[i], mags[i+1]
                den = (y0 - 2*y1 + y2)
                shift = 0.5*(y0 - y2)/den if abs(den) > 1e-30 else 0.0
                ws.append(wgrid[i] + shift*dw)
        ws = sorted(ws)[:maxbands]
        wn[ik, :len(ws)] = np.array(ws) * a / (2*np.pi*CT0)
    return k_arr, wn


if __name__ == "__main__":
    import time
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lat = sys.argv[1] if len(sys.argv) > 1 else "sq"
    for psi in [0.0, 0.8]:
        r = build_red(lat, psi)
        t0 = time.perf_counter()
        k_arr, wn = compute_bands(r)
        print("%s psi=%.1f: %.1f s, %d pts" % (lat, psi, time.perf_counter()-t0,
                                               np.sum(np.isfinite(wn))))
