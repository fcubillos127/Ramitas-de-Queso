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


def imtol_auto(im, wn, fallback=IMTOL):
    """Umbral ADAPTATIVO de |Im(mu)| para UNA serie (un psi): el mayor salto
    de log10|Im| en el tramo central de la distribucion (percentiles 30-98).

    Por que dinamico y no fijo (idea de Miguel, verificada con datos reales de
    bands_sq_c7.npz, nk=70, cut=7): el umbral natural que separa las bandas
    propagantes de la fuga SE MUEVE con psi -- 0.006 (psi=0), 0.047 (0.2),
    0.057 (0.4), 0.117 (0.6), 0.123 (0.8), un factor ~20 -- porque la
    pre-deformacion acopla modos y hace mas 'fugaz' todo el espectro. Un IMTOL
    fijo o corta bandas reales en psi alto o deja pasar fuga en psi bajo.

    Limite honesto: la distribucion no siempre es nitidamente bimodal (los
    saltos son modestos), asi que el umbral elegido puede fluctuar entre
    datasets; por eso se imprime SIEMPRE el valor usado. Si falla, pasar un
    imtol numerico."""
    m = np.isfinite(wn) & np.isfinite(im) & (im > 1e-12)
    v = np.sort(im[m])
    if v.size < 20:
        return float(fallback)
    lv = np.log10(v)
    j0, j1 = int(0.30 * len(lv)), int(0.98 * len(lv))
    if j1 - j0 < 2:
        return float(fallback)
    dif = np.diff(lv[j0:j1])
    j = int(np.argmax(dif)) + j0
    return float(np.sqrt(v[j] * v[j + 1]))


def filtro_im_por_banda(k, wn, im, dw_step=0.05, ventana=3, factor=20.0,
                        min_cadena=4, imtol_solitarios=None, imtol_max=1.0,
                        piso=0.01):
    """Filtro de fuga POR BANDA en vez de por figura (prototipo, idea de
    Miguel llevada un paso mas alla del umbral escalar adaptativo):

    un punto se acepta si su |Im(mu)| es CONSISTENTE con el de sus vecinos de
    la misma banda a lo largo de k. La fuga es una propiedad suave de cada
    banda (una resonancia plana puede tener |Im|~0.5 uniforme en toda la zona
    y ser perfectamente fisica); los espurios saltan respecto de su contexto.
    Un umbral escalar -- fijo o 'auto' -- no puede distinguir esos dos casos:
    o mata la banda fugaz coherente o deja pasar el punto suelto.

    Algoritmo:
      1. ENCADENAR: se enlazan puntos de columnas de k adyacentes por cercania
         en omega (greedy, tolerancia dw_step) formando cadenas ~ bandas.
      2. CONSISTENCIA LOCAL: dentro de cada cadena de largo >= min_cadena, un
         punto se rechaza si su log10|Im| excede la mediana de sus `ventana`
         vecinos de cadena en mas de log10(factor) (solo por EXCESO: fuga
         anomalamente chica no es sospechosa).
      3. Cadenas cortas (< min_cadena) no tienen contexto de banda: se les
         aplica el umbral escalar `imtol_solitarios` (default: imtol_auto).
      4. Techo absoluto `imtol_max`: |Im(mu)| mayor se rechaza SIEMPRE, aunque
         la cadena entera sea consistente -- una cadena de basura coherente
         (p.ej. cruces pegados a un polo) se auto-validaria sin esto.

    `piso`: se compara log10(max(|Im|, piso)). IMPRESCINDIBLE -- las bandas
    propagantes tienen |Im| ~ 1e-16 y sin piso las razones estallan sin
    significado fisico (medido: max 41000x sin piso vs 56x con piso=0.01).
    Calibracion de `factor` con piso=0.01 sobre datos reales (nk=70, cut=7):
    el salto dentro de una cadena esta en p95 ~ 4-12x y p99 ~ 19-30x, de ahi
    el default 20 (rechaza aprox. el 1% mas anomalo de cada cadena).

    Devuelve una mascara booleana keep (nk, nb). Solo aplica al metodo por
    autovalores (los datos del solver de Miguel no traen |Im(mu)|).

    ⚠️ RESULTADO DEL PROTOTIPO (medido, no teorico): comparado con imtol='auto'
    sobre bands_sq_c7.npz, este filtro RESCATA 154-257 puntos por psi y no quita
    ninguno -- o sea es mas PERMISIVO, no mas limpio, y la figura sale mas
    ruidosa. Rescata correctamente bandas de fuga coherentes (p.ej. la plana en
    w~1.207 con |Im|~0.58 sostenido en muchos k, que 'auto' cortaba), pero
    ~50% de lo que rescata cae sobre curvas de red vacia |k+G|: son cadenas de
    fantasmas COHERENTES, y la coherencia sola no implica que sean fisicas.
    Conclusion: util como criterio de RESCATE cuando sabes que hay bandas
    fugaces reales, pero NO sustituye al filtro geometrico de red vacia
    (detectar_fantasmas en postprocess_miguel). Para figuras limpias de una
    sola pasada sigue ganando imtol='auto'."""
    nk, nb = wn.shape
    keep = np.zeros((nk, nb), dtype=bool)
    fin = np.isfinite(wn) & np.isfinite(im)

    # 1) encadenar por continuidad en omega
    chains = {}
    nxt = 0
    prev_pts = []                                # [(n, w, label)] de la columna anterior
    label = -np.ones((nk, nb), dtype=int)
    for i in range(nk):
        cur = [(n, wn[i, n]) for n in range(nb) if fin[i, n]]
        pairs = []
        for (n2, w2) in cur:
            for (n1, w1, lab1) in prev_pts:
                d = abs(w2 - w1)
                if d <= dw_step:
                    pairs.append((d, n1, lab1, n2))
        pairs.sort(key=lambda t: t[0])
        used_prev, used_cur, asig = set(), set(), {}
        for d, n1, lab1, n2 in pairs:
            if n1 in used_prev or n2 in used_cur:
                continue
            used_prev.add(n1); used_cur.add(n2)
            asig[n2] = lab1
        prev_pts = []
        for (n2, w2) in cur:
            lab = asig.get(n2)
            if lab is None:
                lab = nxt; nxt += 1; chains[lab] = []
            label[i, n2] = lab
            chains[lab].append((i, n2))
            prev_pts.append((n2, w2, lab))

    # 2-4) filtrar
    if imtol_solitarios is None:
        imtol_solitarios = imtol_auto(im, wn)
    for lab, pts in chains.items():
        if len(pts) < min_cadena:
            for (i, n) in pts:
                keep[i, n] = (im[i, n] <= imtol_solitarios) and (im[i, n] <= imtol_max)
            continue
        logs = np.log10(np.maximum(np.array([im[i, n] for (i, n) in pts]), piso))
        for j, (i, n) in enumerate(pts):
            lo, hi = max(0, j - ventana), min(len(pts), j + ventana + 1)
            vecinos = np.delete(logs[lo:hi], j - lo)
            ref = np.median(vecinos)
            keep[i, n] = (logs[j] <= ref + np.log10(factor)) and (im[i, n] <= imtol_max)
    return keep


def load(npz, imtol=IMTOL):
    """imtol: numero (corte fijo de |Im(mu)|), 'auto' (umbral escalar
    adaptativo POR PSI via imtol_auto) o 'banda' (consistencia de |Im| dentro
    de cada banda via filtro_im_por_banda; el mas fiel al curado a mano).
    En los modos no numericos se imprime lo decidido por serie."""
    d = np.load(npz, allow_pickle=True)
    lattice = str(d["lattice"]); a = float(d["a"]); Ct0 = float(d["Ct0"])
    psis = d["psis"]; out = []
    for i in range(len(psis)):
        if ("wn_%d" % i) not in d.files:
            continue
        wn = np.array(d["wn_%d" % i]).copy()
        kk = np.array(d["k_%d" % i])
        if ("im_%d" % i) in d.files:
            im = np.array(d["im_%d" % i])
            if isinstance(imtol, str) and imtol == "banda":
                keep = filtro_im_por_banda(kk, wn, im)
                n0 = int(np.isfinite(wn).sum())
                wn[~keep] = np.nan
                print("[plot_bands] psi=%.1f: filtro por banda: %d -> %d puntos"
                      % (float(psis[i]), n0, int(np.isfinite(wn).sum())))
            else:
                if isinstance(imtol, str) and imtol == "auto":
                    thr = imtol_auto(im, wn)
                    print("[plot_bands] psi=%.1f: imtol auto = %.4f" % (float(psis[i]), thr))
                else:
                    thr = float(imtol)
                wn[im > thr] = np.nan
        out.append((float(psis[i]), kk, wn))
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


def make_figures(npz, prefix, ylo=0.0, yhi=1.4, imtol=IMTOL, show=False, clean=True,
                 zoom_ylo=0.7, zoom_yhi=1.2):
    """Genera <prefix>_full.png (rango ylo-yhi) y <prefix>_zoom.png (rango
    zoom_ylo-zoom_yhi). Devuelve (fig_full, fig_zoom). show=True los muestra
    (VSCode/Jupyter).

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
    zoom_ttl = "Zoom (%.1f–%.1f) — red %s" % (zoom_ylo, zoom_yhi, latname)
    fig_zoom = _grid_fig(lattice, a, Ct0, series, zoom_ylo, zoom_yhi, zoom_ttl, clean=clean)
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
