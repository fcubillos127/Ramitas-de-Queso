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
    post_process(red)   # limpia in-place y re-grafica

Qué hace, en orden:
  1) Detecta puntos aislados/espurios por ENLACE DE PASO (sin match cercano
     con ningún punto en la columna de k inmediatamente anterior/siguiente) y
     los borra con red.delete_point(mode="fullgrid", preview=False), que es
     REVERSIBLE (red.restore_deleted()).
  2) Rellena huecos internos de a lo más `max_gap` pasos de k (default 1,
     es decir: un único punto faltante flanqueado por valores reales a ambos
     lados) con interpolación pchip propia (_fill_small_gaps), Y SOLO SI los
     dos valores que flanquean el hueco son parecidos entre sí (misma
     tolerancia dw_step que usa el detector de espurios). NO usa
     red.smooth_interpolate_longitudinal(): esa función de Miguel rellena
     CUALQUIER hueco interno sin límite de tamaño y, verificado sobre datos
     reales, fabricó decenas de puntos a través de huecos de 8-10 pasos en
     bandas ruidosas de resonancia (279->330 y 177->218 puntos -- muchos más
     de lo justificable como "relleno de huecos chicos").
  3) Opcionalmente re-grafica con red.graficar_bandas_grid() (también de Miguel).

ITERACIÓN IMPORTANTE (dos rondas de verificación, no una): la primera versión
de este post-procesamiento solo limitaba el TAMAÑO del hueco (max_gap), pero
eso no basta -- el solver de Miguel ordena las soluciones por frecuencia
ascendente en CADA k por separado, sin rastrear a qué rama física pertenece
cada "columna" de banda. Se encontró un caso real con un hueco de 1 solo punto
flanqueado por omega_norm=0.05 y 0.98 (dos ramas físicas distintas que
coincidieron en el mismo índice de columna); pchip sin más chequeo inventaba
un punto intermedio (~0.52) que no corresponde a nada calculado. Por eso ahora
también se exige que los valores flanqueantes sean parecidos (max_edge_diff)
antes de rellenar. Verificado visualmente (ver diagnose2_psi*.png del
historial de esta sesión) que ya no aparecen puntos fabricados fuera de
tendencia.

QUÉ NO HACE Y POR QUÉ: no usa red.order_bands_by_continuity_global(). Se probó
sobre datos reales (psi=0.8, nk=20, cut=2) y con sus parámetros por defecto
BORRA TODOS LOS DATOS (67 puntos finitos -> 0; "assigned=0/160" en su propio
log). Además escribe el resultado en un atributo NUEVO
(self.omega_longitudinal_ordered), no en self.omega_longitudinal -- si en el
futuro se quiere usar, hay que asignar el resultado de vuelta explícitamente
Y calibrar sus tolerancias (delta_max_norm, penalty_missing, ...) para que no
vuelva a vaciar los datos. No se investigó más a fondo por tocar un subsistema
("solver mejorado") fuera del alcance de este post-procesamiento.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.interpolate import PchipInterpolator


def _greedy_links(wn, dw_step):
    """
    Empareja puntos entre columnas de k ADYACENTES (i, i+1) por cercanía en
    frecuencia (greedy: par más cercano primero, sin reusar índices), con
    tolerancia POR PASO `dw_step`. Devuelve una máscara (nk, nb) que indica
    si cada punto finito tiene un enlace válido hacia i-1 o i+1.

    Por qué por paso y no por ventana absoluta: una banda real con pendiente
    (p.ej. la acústica) puede cambiar más que una tolerancia fija a lo largo
    de varios pasos de k, aunque el cambio DE UN PASO A OTRO sea chico. Medir
    contra el vecino inmediato evita marcar bandas físicas continuas como
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
    Devuelve una lista de índices (i, n) a borrar: puntos finitos que NO
    tienen un enlace válido (diferencia de frecuencia normalizada <= dw_step)
    con ningún punto en la columna de k inmediatamente anterior o siguiente.
    Verificado sobre datos reales que esto preserva las bandas físicas
    continuas (incluida la acústica, con pendiente) y solo marca puntos
    genuinamente aislados (ver debug_detector_psi*.png).
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


def _fill_small_gaps(k, y_real, y_imag, max_gap=1, max_edge_diff=None):
    """
    Rellena SOLO huecos internos de a lo más `max_gap` puntos consecutivos,
    flanqueados por valores reales a ambos lados, con interpolación pchip.

    Por qué no usar red.smooth_interpolate_longitudinal(): esa función (de
    Miguel) rellena TODOS los huecos internos sin límite de tamaño -- en
    datos reales de este proyecto eso fabricó decenas de puntos a través de
    huecos de 8-10 pasos de k en bandas ruidosas de resonancia, inventando
    curvas suaves donde el solver no encontró nada (verificado: 279->330 y
    177->218 puntos finales, muchos más de lo justificable como "relleno de
    huecos chicos"). Aquí se limita explícitamente el tamaño del hueco.

    max_edge_diff: además del tamaño, exige que los dos valores que flanquean
    el hueco sean parecidos entre sí (|v_antes - v_despues| <= max_edge_diff,
    en las mismas unidades que y_real) antes de rellenar. Por qué: el solver
    de Miguel ordena por frecuencia ascendente en CADA k por separado, sin
    rastrear la rama física -- un hueco de 1 solo punto puede estar
    flanqueado por dos ramas físicas DISTINTAS (verificado con un caso real:
    banda "0" pasaba de omega_norm=0.05 a 0.98 con un solo hueco en medio;
    sin este chequeo, pchip inventaba un punto intermedio ~0.52 que no
    corresponde a nada calculado). Si es None, no se exige (comportamiento
    antiguo, no recomendado).
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
    dos ramas físicas distintas que coinciden en el mismo índice de columna.
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


def post_process(red, dw_step=0.06,
                  interpolar=True, max_gap=1, graficar=True, ylim=(0.0, 1.4),
                  verbose=True):
    """Corre el post-procesamiento sobre `red` (ya calculado con
    zeros_longitudinal_fullgrid). Modifica red.omega_longitudinal in-place.
    Devuelve `red`.

    max_gap: tamaño máximo (en pasos de k) de un hueco interno para rellenarlo
    por interpolación. Huecos más grandes se dejan como NaN (no se inventan
    datos que el solver no encontró)."""
    n_antes = int(np.sum(np.isfinite(red.omega_longitudinal[:, :, 0])))

    espurios = detectar_espurios(red, dw_step=dw_step)
    for (i, n) in espurios:
        red.delete_point(i, n, mode="fullgrid", preview=False, sync_disk=False)

    if verbose:
        print("[post_process] espurios eliminados: %d / %d puntos"
              % (len(espurios), n_antes))

    if interpolar:
        n_pre, n_post = rellenar_huecos_chicos(red, max_gap=max_gap, dw_step=dw_step)
        if verbose:
            print("[post_process] huecos chicos (<= %d pasos) rellenados: %d -> %d puntos"
                  % (max_gap, n_pre, n_post))

    if graficar:
        red.graficar_bandas_grid(ylim=list(ylim))

    return red
