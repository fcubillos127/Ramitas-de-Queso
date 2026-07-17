"""
Tarea 3: forma de los modos en el punto que se gapea (M de la red cuadrada),
sin deformacion (psi=0, modo degenerado) y con deformacion (psi=0.8, dos ramas
desdobladas). El campo antiplano se reconstruye alrededor del cilindro como
    u_z(r,theta) = sum_m a_m [ J_m(k0 r) + T_m H_m(k0 r) ] e^{i m theta},
con a_m el vector nulo de (T G0 - I) en la solucion de banda.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import svd
from scipy.special import jv, hankel1
from scipy.signal import argrelmin
from scipy.optimize import fsolve
from bandcalc import build_red, CT0


def band_near(r, kpt, wn_target, half=0.03, ngrid=600, eta=1e-3):
    """Cero COMPLEJO del determinante f=[Re w, Im w] de la banda mas cercana a wn_target."""
    a = r.a
    lo = (wn_target-half)*2*np.pi*CT0/a; hi = (wn_target+half)*2*np.pi*CT0/a
    wg = np.linspace(lo, hi, ngrid)
    mags = np.array([abs(complex(*r.determinant_longitudinal([w, eta], kpt, r.cut))) for w in wg])
    idx = argrelmin(mags, order=2)[0]
    if len(idx) == 0:
        idx = [int(np.argmin(mags))]
    wns = np.array([wg[i] for i in idx])
    best = idx[int(np.argmin(np.abs(wns*a/(2*np.pi*CT0) - wn_target)))]
    return [float(wg[best]), eta]


def null_vector(r, f, kpt):
    cut = r.cut
    Tdiag = np.array([r._Tn(f, n) for n in range(-cut, cut+1)], dtype=complex)
    G0 = r.G0(f, kpt, pol=1, cut=cut)
    A = np.diag(Tdiag) @ G0 - np.eye(2*cut+1)
    U, S, Vh = svd(A)
    return np.conj(Vh[-1]), Tdiag, S[-1]


def field(r, a_vec, Tdiag, f, L=0.95, npix=300, rmask=0.5):
    cut = r.cut
    k0 = r.k0(f, 1)
    xs = np.linspace(-L, L, npix)
    X, Y = np.meshgrid(xs, xs)
    R = np.sqrt(X**2 + Y**2); TH = np.arctan2(Y, X)
    U = np.zeros_like(X, dtype=complex)
    for idx, m in enumerate(range(-cut, cut+1)):
        U += a_vec[idx] * (jv(m, k0*R) + Tdiag[idx]*hankel1(m, k0*R)) * np.exp(1j*m*TH)
    U[R < rmask] = np.nan
    ph = np.exp(-1j*np.angle(U[np.isfinite(U)][np.argmax(np.abs(U[np.isfinite(U)]))]))
    return X, Y, U*ph   # fija la fase global para que Re sea representativa


def main(out):
    kpt = 0.0  # M en la red cuadrada
    targets = [
        (0.0, 1.016, r"$\psi=0$  (M, modo degenerado)"),
        (0.8, 1.016, r"$\psi=0.8$  (rama inferior)"),
        (0.8, 1.044, r"$\psi=0.8$  (rama superior)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (psi, wt, title) in zip(axes, targets):
        r = build_red("sq", psi, cut=2)
        f = band_near(r, kpt, wt)
        a_vec, Tdiag, smin = null_vector(r, f, kpt)
        X, Y, U = field(r, a_vec, Tdiag, f)
        F = np.real(U); vmax = np.nanmax(np.abs(F)) or 1.0
        ax.pcolormesh(X, Y, F/vmax, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(0.5*np.cos(th), 0.5*np.sin(th), "k-", lw=1.2)
        ax.plot(0.45*np.cos(th), 0.45*np.sin(th), "k:", lw=0.9)
        wn = f[0]*r.a/(2*np.pi*CT0)
        ax.set_title(title + "\n" + r"$\omega a/2\pi C_{t0}=%.3f$" % wn, fontsize=10)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        print("psi=%.1f target=%.3f -> wn=%.4f  sigma_min=%.1e" % (psi, wt, wn, smin), flush=True)
    fig.suptitle(r"Modo antiplano $\mathrm{Re}(u_z)$ en M — red cuadrada ($r_1{=}0.45a, r_2{=}0.5a$)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1])
