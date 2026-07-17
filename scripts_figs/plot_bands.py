"""
Figuras 'bonitas' de estructura de bandas a partir de los .npz de compute_bands.py.
Reordena el camino nativo al convencional y usa distancia geométrica real en k
para el eje x. Estilo tipo tesis (Figs. 3-4): puntos negros, líneas de alta simetría.

Uso:
    python plot_bands.py <bands.npz> <out_prefix> [ylim_lo ylim_hi]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from suma_de_red import K as Kvec

NPZ    = sys.argv[1]
PREFIX = sys.argv[2]
YLO    = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
YHI    = float(sys.argv[4]) if len(sys.argv) > 4 else 1.4


def path_order(lattice, k, a):
    """Devuelve (orden_idx, xcoord, ticks, labels) para el camino convencional.
    sq: X-Γ-M-X   |   hx: Γ-M-K-Γ.  x = distancia euclidiana real en k."""
    k = np.asarray(k)
    if lattice == "sq":
        segB = np.where((k > np.pi/a - 1e-9) & (k <= 2*np.pi/a + 1e-9))[0]  # Γ->X
        segA = np.where(k <= np.pi/a + 1e-9)[0]                             # M->Γ
        segC = np.where(k > 2*np.pi/a - 1e-9)[0]                            # X->M
        order = np.concatenate([segB[::-1], segA[::-1], segC[::-1]])        # X-Γ-M-X
        labels = ["X", "Γ", "M", "X"]
        bounds = [len(segB), len(segA), len(segC)]
    else:  # hx : nativo M->K->Γ->M ; queremos Γ-M-K-Γ
        thirds = 2*np.pi/(3*a)
        segA = np.where(k <= thirds + 1e-9)[0]                              # M->K
        segB = np.where((k > thirds - 1e-9) & (k <= 2*np.pi/a + 1e-9))[0]   # K->Γ
        segC = np.where(k > 2*np.pi/a - 1e-9)[0]                            # Γ->M
        order = np.concatenate([segC, segA, segB])                         # Γ-M-K-Γ
        labels = ["Γ", "M", "K", "Γ"]
        bounds = [len(segC), len(segA), len(segB)]
    # distancia geométrica real en el espacio k a lo largo del orden
    vecs = np.array([Kvec(a, kk, lattice) for kk in k[order]])
    d = np.zeros(len(order))
    d[1:] = np.cumsum(np.linalg.norm(np.diff(vecs, axis=0), axis=1))
    ticks = [0.0]
    c = 0
    for b in bounds:
        c += b
        ticks.append(d[min(c, len(d)-1)])
    return order, d, ticks, labels


def load(npz):
    d = np.load(npz, allow_pickle=True)
    lattice = str(d["lattice"]); a = float(d["a"]); Ct0 = float(d["Ct0"])
    psis = d["psis"]
    out = []
    for i in range(len(psis)):
        if ("wn_%d" % i) not in d.files:
            continue
        out.append((float(psis[i]), np.array(d["k_%d" % i]), np.array(d["wn_%d" % i])))
    return lattice, a, Ct0, out


def clean_isolated(xs, ys, dx, dy, min_neigh=1):
    """Elimina puntos espurios: exige >=min_neigh vecinos dentro de la caja (dx,dy).
    Normaliza x,y por (dx,dy) y cuenta vecinos por distancia Chebyshev < 1."""
    if len(xs) == 0:
        return np.array([], dtype=bool)
    xn = xs/dx; yn = ys/dy
    keep = np.zeros(len(xs), dtype=bool)
    for i in range(len(xs)):
        d = np.maximum(np.abs(xn - xn[i]), np.abs(yn - yn[i]))
        keep[i] = (np.sum((d > 0) & (d <= 1.0)) >= min_neigh)
    return keep


def panel(ax, lattice, a, Ct0, k, wn_raw, ylo, yhi, title):
    order, x, ticks, labels = path_order(lattice, k, a)
    wn = wn_raw[order, :]   # (npath, nbands) ya normalizado
    # Nube de puntos (x, y) de todas las bandas -> filtrar espurios por continuidad
    xx, yy = [], []
    for b in range(wn.shape[1]):
        y = wn[:, b]
        m = np.isfinite(y)
        xx.append(x[m]); yy.append(y[m])
    xx = np.concatenate(xx) if xx else np.array([])
    yy = np.concatenate(yy) if yy else np.array([])
    span = ticks[-1] - ticks[0]
    keep = clean_isolated(xx, yy, dx=2.2*span/max(len(order), 1), dy=0.035, min_neigh=2)
    vis = keep & (yy >= ylo-0.05) & (yy <= yhi+0.05)
    ax.plot(xx[vis], yy[vis], ".", color="k", ms=3.2)
    for t in ticks:
        ax.axvline(t, color="0.6", lw=0.6, zorder=0)
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_xlim(ticks[0], ticks[-1]); ax.set_ylim(ylo, yhi)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=10)


def main():
    lattice, a, Ct0, series = load(NPZ)
    latname = {"sq": "cuadrada", "hx": "triangular"}.get(lattice, lattice)
    n = len(series)

    # (1) panel comparativo 1xN, rango completo
    fig, axes = plt.subplots(1, n, figsize=(3.0*n, 3.8), sharey=True)
    if n == 1: axes = [axes]
    for ax, (psi, k, frec) in zip(axes, series):
        panel(ax, lattice, a, Ct0, k, frec, YLO, YHI, r"$\psi=%.1f$" % psi)
    axes[0].set_ylabel(r"$\omega a/2\pi C_{t0}$", fontsize=13)
    fig.suptitle("Estructura de bandas — red %s  ($r_1{=}0.45a,\\ r_2{=}0.5a$)" % latname,
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PREFIX + "_full.png", dpi=160, bbox_inches="tight")
    print("->", PREFIX + "_full.png")

    # (2) zoom tipo Fig.4  (0.7 - 1.2)
    fig, axes = plt.subplots(1, n, figsize=(3.0*n, 3.8), sharey=True)
    if n == 1: axes = [axes]
    for ax, (psi, k, frec) in zip(axes, series):
        panel(ax, lattice, a, Ct0, k, frec, 0.7, 1.2, r"$\psi=%.1f$" % psi)
    axes[0].set_ylabel(r"$\omega a/2\pi C_{t0}$", fontsize=13)
    fig.suptitle("Zoom (0.7–1.2) — red %s" % latname, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PREFIX + "_zoom.png", dpi=160, bbox_inches="tight")
    print("->", PREFIX + "_zoom.png")


if __name__ == "__main__":
    main()
