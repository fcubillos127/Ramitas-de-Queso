"""
Tarea 4: gap vs psi. El gap es el desdoblamiento del modo degenerado en el punto
de alta simetria bajo la pre-deformacion angular:
  - red cuadrada  : punto M (k=0), ventana ~1.01-1.08
  - red triangular: punto K (Dirac, k=2pi/3a), ventana ~0.44-0.52
Se calcula |det|(w) en el punto exacto con alta resolucion y se toman los dos
minimos mas bajos de la ventana; gap = w_up - w_lo.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import argrelmin
from bandcalc import build_red, CT0

CONF = {
    "sq": dict(kpt=0.0,               win=(1.005, 1.085), label="M"),
    "hx": dict(kpt=2*np.pi/3.0,       win=(0.435, 0.520), label="K"),
}
PSIS = np.linspace(0.0, 0.8, 17)
NGRID = 1200
EexT = 5e-4  # eta pequeno para minimos agudos


def two_lowest_minima(r, kpt, win, ngrid=NGRID, eta=EexT):
    a = r.a
    wlo, whi = win
    wg = np.linspace(wlo*2*np.pi*CT0/a, whi*2*np.pi*CT0/a, ngrid)
    mags = np.array([abs(complex(*r.determinant_longitudinal([w, eta], kpt, r.cut))) for w in wg])
    idx = argrelmin(mags, order=3)[0]
    if len(idx) == 0:
        return None
    # ordenar por frecuencia, refinar parabolicamente
    wns = []
    for i in idx:
        y0, y1, y2 = mags[i-1], mags[i], mags[i+1]
        den = (y0 - 2*y1 + y2)
        sh = 0.5*(y0 - y2)/den if abs(den) > 1e-30 else 0.0
        wns.append((wg[i] + sh*(wg[1]-wg[0])) * a/(2*np.pi*CT0))
    wns = sorted(wns)
    return wns


def gap_curve(lattice):
    c = CONF[lattice]
    lows, ups, gaps = [], [], []
    for psi in PSIS:
        r = build_red(lattice, psi, cut=2)
        wns = two_lowest_minima(r, c["kpt"], c["win"])
        if not wns:
            lows.append(np.nan); ups.append(np.nan); gaps.append(np.nan); continue
        lo = wns[0]
        up = wns[1] if len(wns) > 1 else wns[0]
        lows.append(lo); ups.append(up); gaps.append(up - lo)
    return np.array(lows), np.array(ups), np.array(gaps)


if __name__ == "__main__":
    S = sys.argv[1]  # dir de salida
    res = {}
    for lat in ["sq", "hx"]:
        t0 = time.perf_counter()
        lo, up, gap = gap_curve(lat)
        res[lat] = (lo, up, gap)
        print("[%s] %s  gap(psi): %s  (%.1f s)" % (lat, CONF[lat]["label"],
              np.round(gap, 4), time.perf_counter()-t0), flush=True)
    np.savez(os.path.join(S, "gap_vs_psi.npz"),
             psis=PSIS, sq=np.array(res["sq"]), hx=np.array(res["hx"]))

    # figura
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    styl = {"sq": ("o-", "#1f77b4", "cuadrada (punto M)"),
            "hx": ("s-", "#d62728", "triangular (punto K)")}
    for lat in ["sq", "hx"]:
        m, col, lab = styl[lat]
        ax.plot(PSIS, res[lat][2], m, color=col, label=lab, ms=5)
    ax.set_xlabel(r"$\psi$ (pre-deformación angular)", fontsize=12)
    ax.set_ylabel(r"gap  $\Delta(\omega a/2\pi C_{t0})$", fontsize=12)
    ax.set_title("Apertura del gap vs. pre-deformación", fontsize=12)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(S, "gap_vs_psi.png"), dpi=160, bbox_inches="tight")
    print("-> gap_vs_psi.png")
