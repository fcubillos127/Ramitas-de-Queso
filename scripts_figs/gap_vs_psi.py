"""
Tarea 4: gap vs psi.
  - red triangular: apertura del cono de Dirac en K, medida del gap directo
    entre las dos bandas mas bajas (banda1 - banda0) leido del .npz de bandas
    (metodo de autovalores, limpio).
  - red cuadrada: desdoblamiento de la resonancia plana en M, medido con los
    dos minimos mas bajos de |det| en la ventana (las resonancias de M son de
    'fuga' y no aparecen como cruces Re(mu)=1, por eso aqui se usa |det|).

Uso: python gap_vs_psi.py <dir_salida> <bands_hx.npz>
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import argrelmin
from bandcalc import build_red, CT0

PSIS = np.linspace(0.0, 0.8, 17)


# ---- red cuadrada: desdoblamiento en M via |det| ---------------------------
def sq_gap_detmin(kpt=0.0, win=(1.005, 1.085), ngrid=900, eta=5e-4):
    gaps = []
    for psi in PSIS:
        r = build_red("sq", psi, cut=2)
        a = r.a
        wg = np.linspace(win[0]*2*np.pi*CT0/a, win[1]*2*np.pi*CT0/a, ngrid)
        mags = np.array([abs(complex(*r.determinant_longitudinal([w, eta], kpt, 2))) for w in wg])
        idx = argrelmin(mags, order=3)[0]
        wns = sorted(wg[i]*a/(2*np.pi*CT0) for i in idx)
        gaps.append(wns[1]-wns[0] if len(wns) >= 2 else 0.0)
    return np.array(gaps)


# ---- red triangular: gap de Dirac en K leido de los datos de banda ---------
def hx_diracK_from_bands(npz, kK=2*np.pi/3.0, win=(0.40, 0.56)):
    """Frecuencia del punto de Dirac en K (centro del par de bandas) vs psi, leida
    del .npz de bandas. Exactamente en K los dos modos siguen degenerados por
    simetria aun con psi; lo observable limpio es el CORRIMIENTO del punto, no un
    gap (que solo abre, y ruidosamente, ligeramente fuera de K). Devuelve (psis, wK)."""
    d = np.load(npz)
    k = d["k_0"]; ik = int(np.argmin(np.abs(k - kK)))
    psis = np.array(d["psis"]); wK = []
    for i in range(len(psis)):
        b = np.sort(d["wn_%d" % i][ik][np.isfinite(d["wn_%d" % i][ik])])
        b = b[(b > win[0]) & (b < win[1])]
        wK.append(np.mean(b) if len(b) else np.nan)
    return psis, np.array(wK)


if __name__ == "__main__":
    S = sys.argv[1]
    hx_npz = sys.argv[2]
    t0 = time.perf_counter()
    gsq = sq_gap_detmin()
    print("sq M gap:", np.round(gsq, 4), flush=True)
    psis_hx, wK = hx_diracK_from_bands(hx_npz)
    print("hx wK:", np.round(wK, 4), flush=True)
    np.savez(os.path.join(S, "gap_vs_psi.npz"), psis=PSIS, sq=gsq, psis_hx=psis_hx, hx_wK=wK)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(PSIS, gsq, "o-", color="#1f77b4", ms=5)
    a1.set_xlabel(r"$\psi$ (pre-deformación angular)", fontsize=12)
    a1.set_ylabel(r"gap  $\Delta(\omega a/2\pi C_{t0})$", fontsize=12)
    a1.set_title("Red cuadrada — desdoblamiento en M", fontsize=12); a1.grid(alpha=.3)
    a2.plot(psis_hx, wK, "s-", color="#d62728", ms=6)
    a2.set_xlabel(r"$\psi$ (pre-deformación angular)", fontsize=12)
    a2.set_ylabel(r"$\omega_K a/2\pi C_{t0}$", fontsize=12)
    a2.set_title("Red triangular — corrimiento del punto de Dirac (K)", fontsize=12); a2.grid(alpha=.3)
    fig.suptitle("Tarea 4: efecto de la pre-deformación en el espectro", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(S, "gap_vs_psi.png"), dpi=160, bbox_inches="tight")
    print("total %.1fs -> gap_vs_psi.png" % (time.perf_counter()-t0))
