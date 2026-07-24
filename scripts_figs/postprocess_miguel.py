"""
Post-procesamiento AUTOMATICO para correr DESPUES de Red.zeros_longitudinal_fullgrid
(el solver original de Miguel). No modifica Bandas_Tools.py ni el solver: opera
sobre el objeto Red ya calculado (self.omega_longitudinal).

Uso:
    from Bandas_Tools import Red
    import sys; sys.path.insert(0, "scripts_figs")
    from postprocess_miguel import post_process

    red = Red(...)
    ...
    red.zeros_longitudinal_fullgrid(C_l0=295.0, ...)
    post_process(red)   # limpia + completa in-place y re-grafica

Qué hace, en orden:
  1) FANTASMAS DE RED VACIA (lo que hay que QUITAR, cadenas en X dispersivas):
     el determinante det(T*G0 - I) tiene POLOS de la suma de red en las bandas
     de red vacia  w = C_t0*|k+G|  (denominadores |k+G|^2 - k0^2 en S1). Pegado
     a cada polo el determinante cambia de signo, y el buscador de cambios de
     signo + fsolve del solver encuentra ahi "raices" que siguen EXACTAMENTE
     las curvas |k+G| (verificado sobre datos reales: 30% de los puntos con
     psi=0 y 49% con psi=0.8 caen a <0.005 de una curva de red vacia, mientras
     la mediana del resto queda a ~0.04). No son bandas fisicas del cristal
     (cruzan otras bandas sin hibridizar) y son las que Miguel borraba a mano.
     Se detectan por TUBO + PERSISTENCIA: un punto es fantasma si esta a menos
     de `el_tol` de una curva |k+G| Y en columnas de k vecinas hay mas puntos
     siguiendo la MISMA curva (un cruce accidental de una banda real con la
     curva solo la toca en 1-2 columnas; el fantasma la sigue en muchas).
  2) AISLADOS: puntos sin enlace de paso con ningun vecino de k (detector
     original de este modulo). Ambos borrados via red.delete_point(...).
  3) COMPLETAR (lo que hay que AGREGAR, los "vacios" de las bandas planas
     ~1.1-1.4): el buscador de cambios de signo del solver muestrea Re(det) en
     ~ventanas_por_unidad puntos por unidad de w_norm; las resonancias planas
     son pares polo-cero con cambios de signo ANGOSTISIMOS que ese muestreo se
     salta en muchas columnas de k (por eso las bandas planas salen a pedazos).
     Aqui se rastrean los autovalores mu_i de T*G0 en una grilla fina SOLO en
     las ventanas pedidas y se toman los cruces Re(mu)=1 (condicion equivalente
     a det=0 pero de cruce, no de minimo); cada cruce se refina con fsolve
     sobre el MISMO Det_longitudinal del solver (mismos parametros que
     resolver_con_fsolve) y se inserta en omega_longitudinal. No se fabrica
     nada: todos los puntos insertados son soluciones calculadas de det=0.
  4) Reordena cada columna de k por frecuencia ascendente (la convencion del
     solver) y rellena huecos internos de a lo mas `max_gap` pasos de k, solo
     si los flancos son parecidos (interpolacion propia, no la de la clase).
  5) Re-grafica con red.graficar_bandas_grid() (de Miguel).

Antes de tocar nada se guarda una copia en red._omega_backup_postprocess;
para deshacer TODO:  red.omega_longitudinal = red._omega_backup_postprocess.copy()
(red.restore_deleted() deshace borrados individuales, pero tras el reordenado
del paso 4 sus indices (i,n) ya no corresponden -- usa el backup completo.)

HISTORIAL DE ITERACIONES (tres rondas, cada una corrigio un error real):
  a) Detector de espurios por "ventana absoluta de frecuencia": borraba hasta
     47% de los puntos incluida la banda acustica completa (pendiente real
     excede cualquier ventana fija) -> se cambio a enlace POR PASO.
  b) red.smooth_interpolate_longitudinal() rellena huecos sin limite de
     tamano y fabrico decenas de puntos a traves de huecos de 8-10 pasos
     (279->330); y aun limitando el tamano, el solver ordena por frecuencia
     por-k sin rastrear ramas, asi que un hueco de 1 paso puede estar
     flanqueado por ramas fisicas distintas (caso real: 0.05 -> 0.98) ->
     interpolacion propia con max_gap Y flancos parecidos (max_edge_diff).
  c) El enlace de paso NO basta (feedback del usuario con pantallazo marcado):
     lo que hay que quitar son CADENAS CONECTADAS (pasan el test de enlace) y
     lo que falta NO se puede interpolar (no hay puntos que interpolar). La
     fisica de ambas cosas esta arriba: fantasmas de polos de red vacia
     (quitar) y cambios de signo angostos de resonancias planas (recalcular
     dirigido). Esta version agrega los pasos 1 y 3 para eso.

QUE NO HACE Y POR QUE: no usa red.order_bands_by_continuity_global().
Verificado sobre datos reales (psi=0.8, nk=20, cut=2): con sus parametros por
defecto BORRA TODOS LOS DATOS (67 puntos finitos -> 0; "assigned=0/160" en su
propio log) y ademas escribe en un atributo aparte
(self.omega_longitudinal_ordered), no en self.omega_longitudinal.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # raiz del repo
sys.path.insert(0, _HERE)                     # scripts_figs (para bandcalc)
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import fsolve

import suma_de_red as sr


# ---------------------------------------------------------------------------
# 1) Fantasmas de red vacia
# ---------------------------------------------------------------------------

def curvas_red_vacia(red, shells=3, wmax=2.0):
    """Curvas de red vacia w_norm(k) = |k_vec(k)+G|*a/2pi evaluadas sobre
    red.k, para todos los G = k1*b1 + k2*b2 con |k1|,|k2| <= shells cuya curva
    entra bajo `wmax`. Devuelve dict {(k1,k2): array(nk)}. Vale para 'sq' y
    'hx' (usa el mismo mapeo escalar->vector sr.K que usa G0)."""
    a = float(red.a)
    kk = np.asarray(red.k, dtype=float)
    kvecs = np.array([sr.K(a, k, red.lattice) for k in kk])
    curvas = {}
    for k1 in range(-shells, shells + 1):
        for k2 in range(-shells, shells + 1):
            G = sr.Kh(a, k1, k2, red.lattice)
            w = np.linalg.norm(kvecs + G[None, :], axis=1) * a / (2 * np.pi)
            if w.min() <= wmax:
                curvas[(k1, k2)] = w
    return curvas


def detectar_fantasmas(red, el_tol=0.018, el_persist=2, el_min_vecinos=2,
                       el_floor=0.10, shells=3):
    """Puntos "fantasma" pegados a una banda de red vacia w = C_t0*|k+G|
    (polos de la suma de red; ver docstring del modulo). Criterio:

      punto (i,n) es fantasma si existe una curva c=|k+G| tal que
        |w_norm(i,n) - c(i)| < el_tol                            (tubo)
      y al menos `el_min_vecinos` de las columnas j en {i-p..i+p, j!=i}
      (p = el_persist) tienen algun punto tambien a < el_tol de c(j)
                                                                 (persistencia).

    La persistencia distingue el fantasma (sigue la curva por muchas columnas)
    del cruce accidental de una banda real con la curva (la toca en 1-2
    columnas, porque sus pendientes difieren). Verificado sobre datos reales:
    sin persistencia el tubo marcaba tambien los cruces de la banda optica
    plana ~0.8 con las curvas plegadas.

    el_floor: los puntos con w_norm < el_floor NUNCA se marcan para la curva
    G=0 -- cerca de Gamma la banda acustica real y la recta libre w=C_t*k
    convergen (ambas van a 0) y ahi no se pueden distinguir; preferimos
    conservar 1-2 puntos ambiguos antes que comernos la punta de la banda
    acustica.

    Devuelve (lista [(i,n)], dict diagnostico {(i,n): (k1,k2)}).
    """
    a = float(red.a)
    Ct0 = float(red.vel0[1])
    wn = red.omega_longitudinal[:, :, 0] * a / (2 * np.pi * Ct0)
    nk, nb = wn.shape
    finite = np.isfinite(wn)
    curvas = curvas_red_vacia(red, shells=shells)

    # por curva: que columnas tienen ALGUN punto dentro de su tubo
    en_tubo = {}
    for c, w in curvas.items():
        en_tubo[c] = np.any(np.abs(wn - w[:, None]) < el_tol, axis=1) & np.any(finite, axis=1)

    fantasmas, quien = [], {}
    for i in range(nk):
        for n in range(nb):
            if not finite[i, n]:
                continue
            for c, w in curvas.items():
                if abs(wn[i, n] - w[i]) >= el_tol:
                    continue
                if c == (0, 0) and wn[i, n] < el_floor:
                    continue        # proteccion banda acustica cerca de Gamma
                j0, j1 = max(0, i - el_persist), min(nk, i + el_persist + 1)
                vecinos = sum(1 for j in range(j0, j1) if j != i and en_tubo[c][j])
                if vecinos >= el_min_vecinos:
                    fantasmas.append((i, n))
                    quien[(i, n)] = c
                    break
    return fantasmas, quien


# ---------------------------------------------------------------------------
# 2) Aislados (enlace de paso) -- detector de la version anterior, sin cambios
# ---------------------------------------------------------------------------

def _greedy_links(wn, dw_step):
    """
    Empareja puntos entre columnas de k ADYACENTES (i, i+1) por cercania en
    frecuencia (greedy: par mas cercano primero, sin reusar indices), con
    tolerancia POR PASO `dw_step`. Devuelve una mascara (nk, nb) que indica
    si cada punto finito tiene un enlace valido hacia i-1 o i+1.

    Por que por paso y no por ventana absoluta: una banda real con pendiente
    (p.ej. la acustica) puede cambiar mas que una tolerancia fija a lo largo
    de varios pasos de k, aunque el cambio DE UN PASO A OTRO sea chico. Medir
    contra el vecino inmediato evita marcar bandas fisicas continuas como
    espurias solo por tener pendiente.
    """
    nk, nb = wn.shape
    has_link = np.zeros((nk, nb), dtype=bool)
    for i in range(nk - 1):
        cur = [(n, wn[i, n]) for n in range(nb) if np.isfinite(wn[i, n])]
        nxt = [(n, wn[i + 1, n]) for n in range(nb) if np.isfinite(wn[i + 1, n])]
        pairs = []
        for (n1, w1) in cur:
            for (n2, w2) in nxt:
                d = abs(w1 - w2)
                if d <= dw_step:
                    pairs.append((d, n1, n2))
        pairs.sort()
        used1, used2 = set(), set()
        for (d, n1, n2) in pairs:
            if n1 in used1 or n2 in used2:
                continue
            has_link[i, n1] = True
            has_link[i + 1, n2] = True
            used1.add(n1); used2.add(n2)
    return has_link


def detectar_espurios(red, dw_step=0.06):
    """
    Devuelve una lista de indices (i, n) a borrar: puntos finitos que NO
    tienen un enlace valido (diferencia de frecuencia normalizada <= dw_step)
    con ningun punto en la columna de k inmediatamente anterior o siguiente.
    OJO: esto solo atrapa puntos totalmente sueltos; las cadenas conectadas
    que siguen la red vacia las atrapa detectar_fantasmas().
    """
    a = float(red.a)
    Ct0 = float(red.vel0[1])
    om = red.omega_longitudinal
    wn = om[:, :, 0] * a / (2 * np.pi * Ct0)

    link = _greedy_links(wn, dw_step)
    finite = np.isfinite(wn)
    espurio = finite & ~link
    idx = np.argwhere(espurio)
    return [(int(i), int(n)) for i, n in idx]


# ---------------------------------------------------------------------------
# 3) Completar bandas (cruces de autovalores + refinado fsolve del solver)
# ---------------------------------------------------------------------------

def completar_bandas_eig(red, windows=None, ngrid_por_unidad=800,
                         eta=1e-3, imtol_eig=0.8, dedup=0.012,
                         refinar=True, snap=0.02, eta_det=1e-2,
                         evitar_fantasmas=True, el_tol=0.018, el_floor=0.10,
                         shells=3, chunk=0.15, solape=0.01, verbose=True):
    """Busca soluciones de det(T*G0-I)=0 que el barrido de cambios de signo
    del solver se salto (tipicamente las bandas planas de resonancia, cuyos
    cambios de signo son angostisimos) y las INSERTA en red.omega_longitudinal.

    Metodo: en cada ventana (w_lo, w_hi) de `windows` (en w_norm) se rastrean
    los autovalores mu_i(w) de T*G0 sobre una grilla fina y se interpolan los
    cruces Re(mu_i)=1 con |Im(mu_i)| < imtol_eig (los polos de red vacia no
    generan cruces aqui: con w complejo (eta>0) el autovalor asociado a un
    polo de residuo chico queda acotado lejos de 1). Cada cruce que no
    duplique un punto ya existente en esa columna (|dw_norm| < dedup) se
    refina con fsolve sobre red.Det_longitudinal -- MISMOS parametros que usa
    resolver_con_fsolve dentro de zeros_longitudinal_fullgrid: semilla
    [w, eta_det], xtol=red.sol_tol, epsfcn=red.epsfcn, aceptacion ier==1 y
    |w_imag| < red.imag_tol. Si fsolve no converge o se corre mas de `snap`
    (en w_norm) de la semilla, se conserva el valor del cruce de autovalores
    (que es una solucion calculada del mismo determinante, no un invento).

    windows=None (default): una sola ventana que cubre TODO el rango que el
    solver exploro, (0.02, max_encontrado + 0.02) -- no solo las planas: con
    psi grande tambien la banda ACUSTICA se vuelve fuertemente resonante y el
    barrido se salta tramos enteros de ella (verificado con psi=0.8: la
    acustica salia a pedazos). Pasa ventanas explicitas si solo quieres
    completar una zona (mas rapido: el costo escala con el ancho total).

    chunk/solape: cada ventana se procesa en SUB-ventanas de ancho <= `chunk`
    con `solape` de traslape. Por que: los autovalores de T*G0 tienen POLOS
    (los de red vacia) y al atravesar uno el rastreo de ramas por asignacion
    se descalabra; el emparejamiento corrupto se propaga hacia ARRIBA en la
    grilla y enmascara cruces legitimos lejos del polo (verificado: con una
    sola ventana 0.02-1.41 desaparecian los cruces de la banda plana de 1.115
    en M-Gamma que una ventana angosta 1.02-1.42 si encontraba). Trocear
    confina el dano al trozo que contiene el polo; el solape + dedup cubren
    los cruces que caen justo en un borde.

    evitar_fantasmas: no inserta candidatos a menos de `el_tol` de una curva
    de red vacia (mismas curvas y proteccion el_floor que detectar_fantasmas).
    Por que NO dejarselo a la segunda pasada de detectar_fantasmas: los polos
    fuertes tambien generan cruces Re(mu)=1, y si se insertan re-poblan la
    cadena sobre la curva -- con lo que la persistencia condenaria despues
    tambien a los puntos REALES de bandas que cruzan esa curva (verificado:
    aparecian mordiscos en la banda optica justo donde la red vacia la
    cruza). El precio es un hueco de 1-2 columnas donde una banda real cruza
    una curva de red vacia; rellenar_huecos_chicos cubre la mayoria.

    OJO Ct0: usa float(red.vel0[1]), NO la constante CT0 de bandcalc.
    Devuelve la lista de (i, n_col, w_norm) insertados."""
    from bandcalc import _eig_branches

    a = float(red.a)
    Ct0 = float(red.vel0[1])
    om = red.omega_longitudinal
    nk = om.shape[0]
    kk = np.asarray(red.k, dtype=float)
    if windows is None:
        wn_all = om[:, :, 0] * a / (2 * np.pi * Ct0)
        if not np.isfinite(wn_all).any():
            return []
        windows = ((0.02, float(np.nanmax(wn_all)) + 0.02),)
    curvas = curvas_red_vacia(red, shells=shells) if evitar_fantasmas else {}
    insertados = []
    n_refinados = 0
    n_saltados = 0

    # trocear cada ventana en sub-ventanas (ver docstring: confina el
    # descalabro del rastreo de ramas que provocan los polos)
    sub_windows = []
    for (w_lo, w_hi) in windows:
        nch = max(1, int(np.ceil((w_hi - w_lo) / chunk)))
        edges = np.linspace(w_lo, w_hi, nch + 1)
        for j in range(nch):
            lo = edges[j] - (solape if j > 0 else 0.0)
            sub_windows.append((lo, edges[j + 1]))

    for (w_lo, w_hi) in sub_windows:
        ng = max(120, int(round((w_hi - w_lo) * ngrid_por_unidad)))
        wg = np.linspace(w_lo, w_hi, ng) * 2 * np.pi * Ct0 / a   # rad/s
        for i in range(nk):
            tr = _eig_branches(red, kk[i], wg, eta)
            cand = []
            for b in range(tr.shape[1]):
                re = np.real(tr[:, b]) - 1.0
                im = np.imag(tr[:, b])
                for j in np.nonzero(re[:-1] * re[1:] < 0)[0]:
                    t = re[j] / (re[j] - re[j + 1])
                    imv = abs(im[j] + t * (im[j + 1] - im[j]))
                    if imv < imtol_eig:
                        cand.append((wg[j] + t * (wg[j + 1] - wg[j])) * a / (2 * np.pi * Ct0))
            if not cand:
                continue
            existentes = om[i, :, 0] * a / (2 * np.pi * Ct0)
            existentes = existentes[np.isfinite(existentes)]
            for wn_c in sorted(cand):
                if existentes.size and np.min(np.abs(existentes - wn_c)) < dedup:
                    continue
                if evitar_fantasmas:
                    es_fantasma = False
                    for c, w in curvas.items():
                        if abs(wn_c - w[i]) < el_tol:
                            if not (c == (0, 0) and wn_c < el_floor):
                                es_fantasma = True
                                break
                    if es_fantasma:
                        n_saltados += 1
                        continue
                w_rad = wn_c * 2 * np.pi * Ct0 / a
                w_ins, wi_ins = w_rad, 0.0
                if refinar:
                    try:
                        sol, info, ier, _ = fsolve(
                            red.Det_longitudinal, [w_rad, eta_det],
                            args=(kk[i], red.cut, False, 1e-6, 300, 2, False),
                            xtol=red.sol_tol, epsfcn=red.epsfcn, full_output=True)
                        w_ref, wi_ref = sol
                        drift = abs(w_ref - w_rad) * a / (2 * np.pi * Ct0)
                        if ier == 1 and abs(wi_ref) < red.imag_tol and drift < snap:
                            w_ins, wi_ins = w_ref, wi_ref
                            n_refinados += 1
                    except Exception:
                        pass
                # insertar en el primer hueco NaN de la columna (o crecer)
                fila = om[i, :, 0]
                libres = np.where(~np.isfinite(fila))[0]
                if libres.size:
                    n_col = int(libres[0])
                else:
                    om = np.concatenate([om, np.full((nk, 1, 2), np.nan)], axis=1)
                    red.omega_longitudinal = om
                    red.nbands = om.shape[1]
                    n_col = om.shape[1] - 1
                om[i, n_col, 0] = w_ins
                om[i, n_col, 1] = wi_ins
                existentes = np.append(existentes, w_ins * a / (2 * np.pi * Ct0))
                insertados.append((i, n_col, float(w_ins * a / (2 * np.pi * Ct0))))
    if verbose and (insertados or n_saltados):
        print("[completar_bandas_eig] insertados %d puntos (%d refinados con fsolve; "
              "%d candidatos saltados por caer sobre la red vacia)"
              % (len(insertados), n_refinados, n_saltados))
    return insertados


def resort_por_k(red):
    """Reordena cada columna de k por frecuencia real ascendente, NaN al
    final (la misma convencion con la que el solver llena omega_longitudinal).
    Necesario despues de insertar puntos para que las 'bandas' (columnas)
    vuelvan a ser monotonicas en el indice."""
    om = red.omega_longitudinal
    nk, nb = om.shape[0], om.shape[1]
    for i in range(nk):
        fila = om[i, :, 0]
        orden = np.argsort(np.where(np.isfinite(fila), fila, np.inf))
        om[i, :, :] = om[i, orden, :]
    return red


# ---------------------------------------------------------------------------
# 4) Relleno de huecos chicos -- sin cambios respecto de la version anterior
# ---------------------------------------------------------------------------

def _fill_small_gaps(k, y_real, y_imag, max_gap=1, max_edge_diff=None):
    """
    Rellena SOLO huecos internos de a lo mas `max_gap` puntos consecutivos,
    flanqueados por valores reales a ambos lados, con interpolacion pchip.

    Por que no usar red.smooth_interpolate_longitudinal(): esa funcion (de
    Miguel) rellena TODOS los huecos internos sin limite de tamano -- en
    datos reales de este proyecto eso fabrico decenas de puntos a traves de
    huecos de 8-10 pasos de k en bandas ruidosas de resonancia (279->330 y
    177->218 puntos). Aqui se limita explicitamente el tamano del hueco.

    max_edge_diff: ademas del tamano, exige que los dos valores que flanquean
    el hueco sean parecidos entre si (|v_antes - v_despues| <= max_edge_diff,
    en las mismas unidades que y_real) antes de rellenar. Por que: el solver
    ordena por frecuencia ascendente en CADA k por separado, sin rastrear la
    rama fisica -- un hueco de 1 punto puede estar flanqueado por dos ramas
    DISTINTAS (caso real: 0.05 -> 0.98; pchip inventaba ~0.52 que no
    corresponde a nada calculado). Si es None, no se exige (no recomendado).
    """
    valid = np.isfinite(y_real)
    y_real_f = y_real.copy()
    y_imag_f = y_imag.copy()
    if valid.sum() < 2:
        return y_real_f, y_imag_f

    idx_valid = np.where(valid)[0]
    lo, hi = idx_valid.min(), idx_valid.max()
    to_fill = []
    i = lo
    while i <= hi:
        if not valid[i]:
            j = i
            while j <= hi and not valid[j]:
                j += 1
            if (j - i) <= max_gap:
                v_before, v_after = y_real[i - 1], y_real[j]
                if max_edge_diff is None or abs(v_after - v_before) <= max_edge_diff:
                    to_fill.extend(range(i, j))
            i = j
        else:
            i += 1
    if not to_fill:
        return y_real_f, y_imag_f

    xr = k[valid]
    pr = PchipInterpolator(xr, y_real[valid], extrapolate=False)
    pi = PchipInterpolator(xr, y_imag[valid], extrapolate=False)
    xi = k[to_fill]
    y_real_f[to_fill] = pr(xi)
    y_imag_f[to_fill] = pi(xi)
    return y_real_f, y_imag_f


def rellenar_huecos_chicos(red, max_gap=1, dw_step=0.06):
    """Aplica _fill_small_gaps banda por banda sobre red.omega_longitudinal,
    in-place. `dw_step` (en omega normalizada) se convierte a las unidades
    crudas de red.omega_longitudinal y se usa como max_edge_diff: exige que
    los dos valores que flanquean un hueco sean parecidos (misma tolerancia
    que usa detectar_espurios) antes de rellenarlo -- evita interpolar entre
    dos ramas fisicas distintas que coinciden en el mismo indice de columna.
    Devuelve (n_antes, n_despues)."""
    k = np.asarray(red.k)
    om = red.omega_longitudinal
    nb = om.shape[1]
    a = float(red.a)
    Ct0 = float(red.vel0[1])
    max_edge_diff = dw_step * 2 * np.pi * Ct0 / a
    n_antes = int(np.sum(np.isfinite(om[:, :, 0])))
    for n in range(nb):
        yr, yi = _fill_small_gaps(k, om[:, n, 0], om[:, n, 1],
                                   max_gap=max_gap, max_edge_diff=max_edge_diff)
        om[:, n, 0] = yr
        om[:, n, 1] = yi
    n_despues = int(np.sum(np.isfinite(om[:, :, 0])))
    return n_antes, n_despues


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def post_process(red, dw_step=0.06,
                 fantasmas=True, el_tol=0.018, el_persist=2, el_min_vecinos=2,
                 el_floor=0.10,
                 completar=True, windows=None, imtol_eig=0.5,
                 refinar=True,
                 interpolar=True, max_gap=1, graficar=True, ylim=(0.0, 1.4),
                 verbose=True):
    """Corre el post-procesamiento sobre `red` (ya calculado con
    zeros_longitudinal_fullgrid). Modifica red.omega_longitudinal in-place;
    guarda antes una copia en red._omega_backup_postprocess. Devuelve `red`.

    Pasos (cada uno desactivable): fantasmas de red vacia -> aislados ->
    completar bandas en `windows` (LENTO: recalcula T*G0 en una grilla fina;
    del orden de la corrida original en esas ventanas) -> reordenar por k ->
    rellenar huecos chicos. Ver docstrings individuales para los criterios."""
    # OJO (verificado con datos reales): red.delete_point llama a
    # _ensure_tensor(nk, red.nbands) y, si red.nbands NO coincide con
    # omega_longitudinal.shape[1], REEMPLAZA el tensor entero por NaN
    # (perdida total de datos silenciosa salvo por los AVISOs "ya es NaN").
    # Pasa facil al cargar un omega_longitudinal externo sobre un Red armado
    # con otro nbands. Se sincroniza aqui antes de tocar nada.
    red.nbands = int(red.omega_longitudinal.shape[1])
    red.nk = int(red.omega_longitudinal.shape[0])
    red._omega_backup_postprocess = red.omega_longitudinal.copy()
    n_antes = int(np.sum(np.isfinite(red.omega_longitudinal[:, :, 0])))

    if fantasmas:
        fant, quien = detectar_fantasmas(red, el_tol=el_tol, el_persist=el_persist,
                                         el_min_vecinos=el_min_vecinos,
                                         el_floor=el_floor)
        for (i, n) in fant:
            red.delete_point(i, n, mode="fullgrid", preview=False, sync_disk=False)
        if verbose:
            print("[post_process] fantasmas de red vacia eliminados: %d / %d puntos"
                  % (len(fant), n_antes))

    espurios = detectar_espurios(red, dw_step=dw_step)
    for (i, n) in espurios:
        red.delete_point(i, n, mode="fullgrid", preview=False, sync_disk=False)
    if verbose:
        print("[post_process] puntos aislados eliminados: %d" % len(espurios))

    if completar:
        completar_bandas_eig(red, windows=windows, imtol_eig=imtol_eig,
                             refinar=refinar, verbose=verbose)
        if fantasmas:
            # por si algun cruce insertado cayo sobre una curva de red vacia
            fant2, _ = detectar_fantasmas(red, el_tol=el_tol, el_persist=el_persist,
                                          el_min_vecinos=el_min_vecinos,
                                          el_floor=el_floor)
            for (i, n) in fant2:
                red.delete_point(i, n, mode="fullgrid", preview=False, sync_disk=False)
            if verbose and fant2:
                print("[post_process] fantasmas entre los insertados: %d" % len(fant2))

    resort_por_k(red)

    if interpolar:
        n_pre, n_post = rellenar_huecos_chicos(red, max_gap=max_gap, dw_step=dw_step)
        if verbose:
            print("[post_process] huecos chicos (<= %d pasos) rellenados: %d -> %d puntos"
                  % (max_gap, n_pre, n_post))

    if verbose:
        n_fin = int(np.sum(np.isfinite(red.omega_longitudinal[:, :, 0])))
        print("[post_process] total: %d -> %d puntos  (backup en red._omega_backup_postprocess)"
              % (n_antes, n_fin))

    if graficar:
        red.graficar_bandas_grid(ylim=list(ylim))

    return red
