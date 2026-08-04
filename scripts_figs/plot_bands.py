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


def _encadenar(wn, dw_step):
    """Agrupa los puntos finitos en cadenas ~ bandas, enlazando columnas de k
    adyacentes por cercania en omega (greedy, tolerancia dw_step).
    Devuelve dict {label: [(i, n), ...]} ordenado por k creciente."""
    nk, nb = wn.shape
    fin = np.isfinite(wn)
    chains, nxt, prev = {}, 0, []
    for i in range(nk):
        cur = [(n, wn[i, n]) for n in range(nb) if fin[i, n]]
        pares = []
        for (n2, w2) in cur:
            for (n1, w1, l1) in prev:
                dd = abs(w2 - w1)
                if dd <= dw_step:
                    pares.append((dd, n1, l1, n2))
        pares.sort(key=lambda t: t[0])
        up, uc, asig = set(), set(), {}
        for dd, n1, l1, n2 in pares:
            if n1 in up or n2 in uc:
                continue
            up.add(n1); uc.add(n2); asig[n2] = l1
        prev = []
        for (n2, w2) in cur:
            l = asig.get(n2)
            if l is None:
                l = nxt; nxt += 1; chains[l] = []
            chains[l].append((i, n2))
            prev.append((n2, w2, l))
    return chains


def _curvas_red_vacia(k, lattice, a, shells=3, wmax=2.0):
    """Bandas de red vacia w_norm(k) = |k_vec(k)+G|*a/2pi sobre la malla k."""
    import suma_de_red as sr
    kv = np.array([sr.K(a, x, lattice) for x in k])
    out = []
    for k1 in range(-shells, shells + 1):
        for k2 in range(-shells, shells + 1):
            G = sr.Kh(a, k1, k2, lattice)
            w = np.linalg.norm(kv + G[None, :], axis=1) * a / (2 * np.pi)
            if w.min() <= wmax:
                out.append(w)
    return np.array(out)


def filtro_consenso(k, wn, im, lattice="sq", a=1.0, dw_step=0.05,
                    el_tol=0.018, el_persist=2, el_min_vecinos=2, el_floor=0.10,
                    shells=3, min_cadena=4, ventana=3, factor=20.0,
                    piso=0.01, imtol_max=1.0, verbose=False):
    """CONSENSO entre criterios INDEPENDIENTES (lo que pidio el usuario, pero
    sobre evidencia distinta en cada uno). Un punto se conserva si pasa los
    tres:

      1. GEOMETRIA  no es un fantasma de red vacia: no esta a menos de `el_tol`
                    de una curva |k+G| que ademas este poblada en columnas de k
                    vecinas (tubo + persistencia, el criterio de
                    postprocess_miguel.detectar_fantasmas). Proteccion el_floor
                    cerca de Gamma, donde G=0 y la acustica convergen.
      2. FUGA       su |Im(mu)| es consistente con el de su banda
                    (filtro_im_por_banda); las cadenas cortas caen al umbral
                    escalar imtol_auto.
      3. CONTINUIDAD pertenece a una cadena de al menos `min_cadena` puntos, o
                    bien sobrevive el umbral escalar si es corta.

    Por que asi y no votando entre imtol fijo/'auto'/'banda': esos tres son
    umbrales sobre la MISMA cantidad y estan anidados (verificado: auto ⊆ fijo
    ⊆ banda en los 5 psi), de modo que su interseccion es identica a 'auto' y
    no aporta informacion. El consenso solo sirve entre evidencias distintas.

    Devuelve (keep, info) con info = conteos por criterio."""
    nk, nb = wn.shape
    fin = np.isfinite(wn) & np.isfinite(im)
    chains = _encadenar(wn, dw_step)

    # --- 1) geometria: fantasmas de red vacia -------------------------------
    curvas = _curvas_red_vacia(k, lattice, a, shells=shells)
    en_tubo = np.array([np.any(np.abs(wn - c[:, None]) < el_tol, axis=1) for c in curvas])
    es_fantasma = np.zeros((nk, nb), dtype=bool)
    for ci, c in enumerate(curvas):
        cerca = fin & (np.abs(wn - c[:, None]) < el_tol)
        if not cerca.any():
            continue
        for i, n in np.argwhere(cerca):
            if wn[i, n] < el_floor and abs(c[i]) < el_floor:
                continue                       # proteccion acustica cerca de Gamma
            j0, j1 = max(0, i - el_persist), min(nk, i + el_persist + 1)
            vec = sum(1 for j in range(j0, j1) if j != i and en_tubo[ci][j])
            if vec >= el_min_vecinos:
                es_fantasma[i, n] = True

    # --- 2) fuga por banda + 3) continuidad ---------------------------------
    keep_fuga = filtro_im_por_banda(k, wn, im, dw_step=dw_step, ventana=ventana,
                                    factor=factor, min_cadena=min_cadena,
                                    imtol_max=imtol_max, piso=piso)
    en_cadena_larga = np.zeros((nk, nb), dtype=bool)
    for pts in chains.values():
        if len(pts) >= min_cadena:
            for (i, n) in pts:
                en_cadena_larga[i, n] = True
    thr = imtol_auto(im, wn)
    keep_cont = en_cadena_larga | (fin & (im <= thr))

    keep = fin & ~es_fantasma & keep_fuga & keep_cont
    info = {"crudo": int(fin.sum()), "fantasmas": int((fin & es_fantasma).sum()),
            "rechaza_fuga": int((fin & ~keep_fuga).sum()),
            "rechaza_cont": int((fin & ~keep_cont).sum()),
            "final": int(keep.sum()), "imtol_solitarios": thr}
    if verbose:
        print("   [consenso] crudo=%d  fantasmas=-%d  fuga=-%d  continuidad=-%d  -> %d"
              % (info["crudo"], info["fantasmas"], info["rechaza_fuga"],
                 info["rechaza_cont"], info["final"]))
    return keep, info


def filtro_fusion(k, wn, im, lattice="sq", a=1.0, imtol_base=0.12,
                  dw_step=0.05, max_hueco=3, min_apoyo=3, imtol_rescate=1.0,
                  evitar_fantasmas=True, el_tol=0.018, el_persist=2,
                  el_min_vecinos=2, el_floor=0.10, shells=3, verbose=False):
    """FUSION DIRIGIDA: `imtol_base` (0.12) manda, y solo se RESCATAN los
    puntos que COMPLETAN una banda ya presente en esa base.

    Motivacion (observacion del usuario sobre las figuras): con IMTOL=0.12
    faltan tramos de las bandas 2 y 3 cerca de M que el criterio por banda si
    encuentra. Pero la union simple no sirve -- 'banda' contiene a 'fijo'
    (verificado: auto ⊆ fijo ⊆ banda en los 5 psi), asi que unir daria la
    version permisiva completa, con toda su maraña. Lo que hace falta es
    rescatar solo lo que CONTINUA algo que ya existe.

    Un punto fuera de la base se acepta si cumple TODO:
      a) |Im(mu)| <= imtol_rescate  (techo del rescate; nunca basura extrema)
      b) pertenece a una cadena (continuidad en omega, tolerancia dw_step) que
         ya tiene >= `min_apoyo` puntos DENTRO de la base  -> esta completando
         una banda existente, no inventando una nueva
      c) esta a <= `max_hueco` pasos de k de un punto de la base de SU MISMA
         cadena -> rellena un hueco, no extiende la banda indefinidamente
      d) si evitar_fantasmas: no esta sobre una curva de red vacia |k+G|
         poblada en los k vecinos

    Devuelve (keep, info)."""
    nk, nb = wn.shape
    fin = np.isfinite(wn) & np.isfinite(im)
    base = fin & (im <= imtol_base)
    chains = _encadenar(wn, dw_step)

    # (d) geometria
    es_fantasma = np.zeros((nk, nb), dtype=bool)
    if evitar_fantasmas:
        curvas = _curvas_red_vacia(k, lattice, a, shells=shells)
        en_tubo = np.array([np.any(np.abs(wn - c[:, None]) < el_tol, axis=1)
                            for c in curvas])
        for ci, c in enumerate(curvas):
            for i, n in np.argwhere(fin & (np.abs(wn - c[:, None]) < el_tol)):
                if wn[i, n] < el_floor and abs(c[i]) < el_floor:
                    continue
                j0, j1 = max(0, i - el_persist), min(nk, i + el_persist + 1)
                if sum(1 for j in range(j0, j1) if j != i and en_tubo[ci][j]) >= el_min_vecinos:
                    es_fantasma[i, n] = True

    keep = base.copy()
    n_resc = 0
    for pts in chains.values():
        idx_base = [j for j, (i, n) in enumerate(pts) if base[i, n]]
        if len(idx_base) < min_apoyo:          # (b) sin apoyo suficiente
            continue
        ks_base = [pts[j][0] for j in idx_base]
        for j, (i, n) in enumerate(pts):
            if base[i, n] or not fin[i, n]:
                continue
            if im[i, n] > imtol_rescate:       # (a)
                continue
            if min(abs(i - kb) for kb in ks_base) > max_hueco:   # (c)
                continue
            if es_fantasma[i, n]:              # (d)
                continue
            keep[i, n] = True
            n_resc += 1
    info = {"crudo": int(fin.sum()), "base": int(base.sum()),
            "rescatados": n_resc, "final": int(keep.sum())}
    if verbose:
        print("   [fusion] crudo=%d  base(imtol<=%.2f)=%d  +rescatados=%d  -> %d"
              % (info["crudo"], imtol_base, info["base"], n_resc, info["final"]))
    return keep, info


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
    adaptativo POR PSI via imtol_auto), 'banda' (consistencia de |Im| dentro de
    cada banda; PERMISIVO, ver su docstring) o 'consenso' (geometria |k+G| +
    fuga por banda + continuidad; el mas limpio de una pasada).
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
            if isinstance(imtol, str) and imtol == "fusion":
                print("[plot_bands] psi=%.1f:" % float(psis[i]))
                keep, _ = filtro_fusion(kk, wn, im, lattice=lattice, a=a,
                                        verbose=True)
                wn[~keep] = np.nan
            elif isinstance(imtol, str) and imtol == "consenso":
                print("[plot_bands] psi=%.1f:" % float(psis[i]))
                keep, _ = filtro_consenso(kk, wn, im, lattice=lattice, a=a,
                                          verbose=True)
                wn[~keep] = np.nan
            elif isinstance(imtol, str) and imtol == "banda":
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
