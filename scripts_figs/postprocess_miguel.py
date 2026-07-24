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
  1) Detecta puntos aislados/espurios en (k, omega_norm) -- sin suficientes
     vecinos cercanos -- y los borra con red.delete_point(mode="fullgrid",
     preview=False), que es REVERSIBLE (red.restore_deleted()).
  2) Rellena huecos internos pequeños con red.smooth_interpolate_longitudinal()
     (herramienta propia de Miguel; verificado que preserva exactamente los
     puntos existentes y solo interpola huecos).
  3) Opcionalmente re-grafica con red.graficar_bandas_grid() (también de Miguel).

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


def post_process(red, dw_step=0.06,
                  interpolar=True, graficar=True, ylim=(0.0, 1.4),
                  verbose=True):
    """Corre el post-procesamiento sobre `red` (ya calculado con
    zeros_longitudinal_fullgrid). Modifica red.omega_longitudinal in-place.
    Devuelve `red`."""
    n_antes = int(np.sum(np.isfinite(red.omega_longitudinal[:, :, 0])))

    espurios = detectar_espurios(red, dw_step=dw_step)
    for (i, n) in espurios:
        red.delete_point(i, n, mode="fullgrid", preview=False, sync_disk=False)

    if verbose:
        print("[post_process] espurios eliminados: %d / %d puntos"
              % (len(espurios), n_antes))

    if interpolar:
        n_pre_interp = int(np.sum(np.isfinite(red.omega_longitudinal[:, :, 0])))
        red.smooth_interpolate_longitudinal()
        n_post_interp = int(np.sum(np.isfinite(red.omega_longitudinal[:, :, 0])))
        if verbose:
            print("[post_process] huecos rellenados: %d -> %d puntos"
                  % (n_pre_interp, n_post_interp))

    if graficar:
        red.graficar_bandas_grid(ylim=list(ylim))

    return red
