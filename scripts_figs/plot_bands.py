"""
Figuras 'bonitas' de estructura de bandas a partir de los .npz de compute_driver.py.
Camino convencional con tramos de alta simetría EQUIespaciados (X-Γ, Γ-M, M-X del
mismo ancho), estilo de las figuras del artículo. Puntos negros.

Como CLI:
    python plot_bands.py <bands.npz> <out_prefix> [ylim_lo ylim_hi]
Como función (p.ej. en VSCode):
    from plot_bands import make_figures
    make_figures("data/bands_sq.npz", "graphs/bandas_sq", imtol=0.12, show=True)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

IMTOL = 0.12   # corte de 'fuga' |Im(mu)| por defecto para mostrar una banda


def path_order(lattice, k, a):
    """(orden_idx, x, ticks, labels) con tramos de alta simetría equiespaciados.

    El `k` ESCALAR recorre físicamente, según suma_de_red.K (verificado):
        sq: M-Γ-X-M   (k=0→M, π/a→Γ, 2π/a→X, 3π/a→M)
        hx: M-K-Γ-M   (k=0→M, 2π/3a→K, 2π/a→Γ, k_end→M)
    Se respeta ESE orden natural, en el sentido de k CRECIENTE, sin invertir
    ningún tramo. Cada tramo se reescala a ancho 1 (ticks en 0,1,2,3) porque en
    hx los tramos no tienen igual ancho en k escalar.

    ⚠️ La versión anterior INVERTÍA cada tramo para rotularlo X-Γ-M-X (sq) /
    Γ-M-K-Γ (hx). Invertir un tramo voltea el signo aparente de la velocidad
    de grupo -> las bandas salían con la dispersión al revés y NO calzaban con
    las figuras de Miguel (lo detectó el usuario por la banda acústica en X-M,
    que debe tener pendiente positiva). Miguel grafica ω vs k escalar directo
    (graficar_bandas_grid, sin invertir), así que el orden natural reproduce
    la forma de sus curvas."""
    k = np.asarray(k); pi = np.pi
    if lattice == "sq":
        bounds = [0.0, pi/a, 2*pi/a, 3*pi/a]
        labels = ["M", "Γ", "X", "M"]
    else:  # hx
        kend = 2*pi*(1 + 1/np.sqrt(3))/a
        bounds = [0.0, 2*pi/(3*a), 2*pi/a, kend]
        labels = ["M", "K", "Γ", "M"]
    order_parts, x_parts = [], []
    for i in range(len(bounds) - 1):
        k0, k1 = bounds[i], bounds[i + 1]
        idx = np.where((k >= k0 - 1e-9) & (k <= k1 + 1e-9))[0]
        if i > 0:                       # el punto de frontera va al tramo previo
            idx = idx[k[idx] > k0 + 1e-9]
        frac = (k[idx] - k0) / (k1 - k0)
        srt = np.argsort(frac)
        order_parts.append(idx[srt]); x_parts.append(i + frac[srt])
    return np.concatenate(order_parts), np.concatenate(x_parts), [0, 1, 2, 3], labels


def load(npz, imtol=IMTOL):
    d = np.load(npz, allow_pickle=True)
    lattice = str(d["lattice"]); a = float(d["a"]); Ct0 = float(d["Ct0"])
    psis = d["psis"]; out = []
    for i in range(len(psis)):
        if ("wn_%d" % i) not in d.files:
            continue
        wn = np.array(d["wn_%d" % i]).copy()
        if ("im_%d" % i) in d.files:
            wn[np.array(d["im_%d" % i]) > imtol] = np.nan
        out.append((float(psis[i]), np.array(d["k_%d" % i]), wn))
    return lattice, a, Ct0, out


def clean_isolated(xs, ys, dx, dy, min_neigh=2):
    """Quita puntos espurios: exige >=min_neigh vecinos en la caja (dx,dy)."""
    if len(xs) == 0:
        return np.array([], dtype=bool)
    xn = xs/dx; yn = ys/dy
    keep = np.zeros(len(xs), dtype=bool)
    for i in range(len(xs)):
        d = np.maximum(np.abs(xn - xn[i]), np.abs(yn - yn[i]))
        keep[i] = (np.sum((d > 0) & (d <= 1.0)) >= min_neigh)
    return keep


def panel(ax, lattice, a, Ct0, k, wn_raw, ylo, yhi, title, clean=True):
    """clean=True aplica el filtro clean_isolated (quita puntos sin >=2
    vecinos): apropiado para la salida CRUDA del metodo por autovalores.
    clean=False lo SALTA: usalo con datos ya curados (p.ej. tras
    postprocess_miguel.post_process, o bandas editadas a mano) -- si no, el
    filtro vuelve a borrar los puntos de banda plana insertados, que en la
    grilla equiespaciada quedan mas separados y parecen 'aislados'."""
    order, x, ticks, labels = path_order(lattice, k, a)
    wn = wn_raw[order, :]
    xx, yy = [], []
    for b in range(wn.shape[1]):
        m = np.isfinite(wn[:, b]); xx.append(x[m]); yy.append(wn[m, b])
    xx = np.concatenate(xx) if xx else np.array([])
    yy = np.concatenate(yy) if yy else np.array([])
    span = ticks[-1] - ticks[0]
    if clean:
        keep = clean_isolated(xx, yy, dx=2.4*span/max(len(order), 1), dy=0.028, min_neigh=2)
    else:
        keep = np.ones(len(xx), dtype=bool)
    vis = keep & (yy >= ylo-0.05) & (yy <= yhi+0.05)
    ax.plot(xx[vis], yy[vis], ".", color="k", ms=3.2)
    for t in ticks:
        ax.axvline(t, color="0.6", lw=0.6, zorder=0)
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_xlim(ticks[0], ticks[-1]); ax.set_ylim(ylo, yhi)
    ax.set_title(title, fontsize=12); ax.tick_params(labelsize=10)


def _grid_fig(lattice, a, Ct0, series, ylo, yhi, suptitle, clean=True):
    n = len(series)
    fig, axes = plt.subplots(1, n, figsize=(3.0*n, 3.8), sharey=True)
    if n == 1: axes = [axes]
    for ax, (psi, k, wn) in zip(axes, series):
        panel(ax, lattice, a, Ct0, k, wn, ylo, yhi, r"$\psi=%.1f$" % psi, clean=clean)
    axes[0].set_ylabel(r"$\omega a/2\pi C_{t0}$", fontsize=13)
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def make_figures(npz, prefix, ylo=0.0, yhi=1.4, imtol=IMTOL, show=False, clean=True):
    """Genera <prefix>_full.png (rango completo) y <prefix>_zoom.png (0.7-1.2).
    Devuelve (fig_full, fig_zoom). show=True los muestra (VSCode/Jupyter).

    clean=True: filtro clean_isolated ON (salida cruda del metodo por
    autovalores). clean=False: OFF, para datos YA curados (exportados tras
    postprocess_miguel.post_process o edicion a mano) -- si no, se re-borran
    los puntos de banda plana insertados."""
    lattice, a, Ct0, series = load(npz, imtol=imtol)
    latname = {"sq": "cuadrada", "hx": "triangular"}.get(lattice, lattice)
    for p in {os.path.dirname(prefix)} - {""}:
        os.makedirs(p, exist_ok=True)
    ttl = "Estructura de bandas — red %s  ($r_1{=}0.45a,\\ r_2{=}0.5a$)" % latname
    fig_full = _grid_fig(lattice, a, Ct0, series, ylo, yhi, ttl, clean=clean)
    fig_full.savefig(prefix + "_full.png", dpi=160, bbox_inches="tight")
    fig_zoom = _grid_fig(lattice, a, Ct0, series, 0.7, 1.2, "Zoom (0.7–1.2) — red %s" % latname, clean=clean)
    fig_zoom.savefig(prefix + "_zoom.png", dpi=160, bbox_inches="tight")
    print("->", prefix + "_full.png", "/", prefix + "_zoom.png")
    if show:
        plt.show()
    return fig_full, fig_zoom


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    NPZ = sys.argv[1]; PREFIX = sys.argv[2]
    YLO = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    YHI = float(sys.argv[4]) if len(sys.argv) > 4 else 1.4
    make_figures(NPZ, PREFIX, YLO, YHI)
