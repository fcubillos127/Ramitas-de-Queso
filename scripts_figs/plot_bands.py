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
    sq: X-Γ-M-X   |   hx: Γ-M-K-Γ.
    Cada tramo de alta simetría ocupa el MISMO ancho (segmentos equiespaciados,
    como en las figuras del artículo): dentro de cada tramo la coordenada x va
    de i a i+1 según la posición fraccional en k."""
    k = np.asarray(k)
    pi = np.pi
    if lattice == "sq":
        # (indices del tramo, k en x=inicio, k en x=fin)  en orden de display X-Γ-M-X
        segs = [
            (np.where((k > pi/a - 1e-9) & (k <= 2*pi/a + 1e-9))[0], 2*pi/a, pi/a),   # X->Γ
            (np.where(k <= pi/a + 1e-9)[0],                          pi/a,   0.0),    # Γ->M
            (np.where(k > 2*pi/a - 1e-9)[0],                         3*pi/a, 2*pi/a), # M->X
        ]
        labels = ["X", "Γ", "M", "X"]
    else:  # hx : Γ-M-K-Γ
        t = 2*pi/(3*a); kend = 2*pi*(1 + 1/np.sqrt(3))/a
        segs = [
            (np.where(k > 2*pi/a - 1e-9)[0],                      2*pi/a, kend),   # Γ->M
            (np.where(k <= t + 1e-9)[0],                          0.0,    t),       # M->K
            (np.where((k > t - 1e-9) & (k <= 2*pi/a + 1e-9))[0],  t,      2*pi/a),  # K->Γ
        ]
        labels = ["Γ", "M", "K", "Γ"]
    order_parts, x_parts = [], []
    for i, (idx, k0, k1) in enumerate(segs):
        frac = (k[idx] - k0) / (k1 - k0)          # 0 en x=i, 1 en x=i+1
        srt = np.argsort(frac)
        order_parts.append(idx[srt])
        x_parts.append(i + frac[srt])
    order = np.concatenate(order_parts)
    x = np.concatenate(x_parts)
    ticks = [0, 1, 2, 3]
    return order, x, ticks, labels


IMTOL = 0.12   # corte de 'fuga' |Im(mu)| para mostrar una banda

def load(npz):
    d = np.load(npz, allow_pickle=True)
    lattice = str(d["lattice"]); a = float(d["a"]); Ct0 = float(d["Ct0"])
    psis = d["psis"]
    out = []
    for i in range(len(psis)):
        if ("wn_%d" % i) not in d.files:
            continue
        wn = np.array(d["wn_%d" % i]).copy()
        if ("im_%d" % i) in d.files:            # enmascarar modos demasiado 'de fuga'
            im = np.array(d["im_%d" % i])
            wn[im > IMTOL] = np.nan
        out.append((float(psis[i]), np.array(d["k_%d" % i]), wn))
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
    keep = clean_isolated(xx, yy, dx=2.4*span/max(len(order), 1), dy=0.028, min_neigh=2)
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
