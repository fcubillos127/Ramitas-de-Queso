"""
Extractor de bandas por minimos de |det(T G0 - I)| sobre una grilla fija de omega.
Como la grilla de omega es la MISMA para todo k, el coeficiente T_n(omega) se cachea
y se reutiliza entre todos los k (T no depende de k), haciendo el calculo rapido
incluso con pre-deformacion (psi != 0). Fisicamente: las bandas son los ceros del
determinante, i.e. minimos profundos de |det|.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.signal import argrelmin
from scipy.optimize import linear_sum_assignment
from Bandas_Tools import Red

CT0 = 295.0


def _eig_branches(r, k, wg, eta):
    """Autovalores de T(w)G0(w,k) rastreados en ramas continuas en w.

    La condicion de banda det(T G0 - I)=0 equivale a que ALGUN autovalor mu_i de
    T G0 valga 1. Rastrear cada rama y buscar el cruce Re(mu_i)=1 es mucho mas
    robusto que buscar minimos de |det| (que es el producto de los factores y
    borra ramas cercanas), y no requiere umbral de magnitud.
    """
    nb = 2*r.cut + 1
    E = np.empty((len(wg), nb), dtype=complex)
    for j, w in enumerate(wg):
        f = [w, eta]
        T = np.array([r._Tn(f, n) for n in range(-r.cut, r.cut+1)], dtype=complex)
        E[j] = np.linalg.eigvals(np.diag(T) @ r.G0(f, k, 1, r.cut))
    tr = np.empty_like(E)
    tr[0] = E[0]
    for j in range(1, len(wg)):
        C = np.abs(E[j][None, :] - tr[j-1][:, None])   # coste (prev x cur)
        ri, ci = linear_sum_assignment(C)
        order = np.empty(nb, dtype=int); order[ri] = ci
        tr[j] = E[j][order]
    return tr


def _frecuencias_red_vacia(r, k, shells=3, wmax=2.0):
    """Frecuencias normalizadas de red vacia en un punto k:
    w_norm = |k_vec(k) + G| * a / 2pi,  con G = k1*b1 + k2*b2.

    Son POLOS de la suma de red (el denominador de S1_pre es Qh(Qh^2-k0^2)) y
    por tanto polos del determinante. NO son bandas: ahi el determinante
    diverge en vez de anularse. Se conocen en forma cerrada, asi que se pueden
    excluir exactamente."""
    import suma_de_red as sr
    a = float(r.a)
    kv = sr.K(a, k, r.lattice)
    out = []
    for k1 in range(-shells, shells + 1):
        for k2 in range(-shells, shells + 1):
            w = float(np.linalg.norm(kv + sr.Kh(a, k1, k2, r.lattice))) * a / (2*np.pi)
            if w <= wmax:
                out.append(w)
    return np.array(sorted(out))


def compute_bands_eig(r, nk=100, wmax=1.4, ngrid=900, eta=1e-3, imtol=0.6, maxbands=18,
                      pole_tol=0.018, pole_floor=0.10, shells=3):
    """Bandas por cruce de autovalores Re(mu)=1. Devuelve (k_arr, wn, im):
    wn (nk, maxbands) frecuencias normalizadas y im la |Im(mu)| en el cruce
    (medida de 'fuga' del modo, para filtrar despues en el graficado). NaN-padded.
    imtol aqui solo descarta cruces absurdamente lejanos; el corte fino se hace al plotear.

    EXCLUSION DE POLOS (pole_tol): cerca de un polo de la suma de red
    (w = |k+G|*a/2pi) un autovalor mu diverge y BARRE el valor 1 de camino,
    produciendo un cruce que no es una banda. Medido con psi=0.6, cut=12 en un
    k tipico: de 13 cruces, 6 caian exactamente sobre polos y solo 2 eran
    raices genuinas. Como las frecuencias de polo se conocen en forma cerrada,
    aqui se descartan los cruces a menos de `pole_tol` de una de ellas, EN LA
    BUSQUEDA -- asi no llegan nunca al .npz y no hay que limpiarlos despues.

    pole_floor: bajo esa frecuencia no se excluye por el polo G=0, porque cerca
    de Gamma la banda acustica real y la recta libre w=C_t*k convergen y no se
    pueden separar; alli se prefiere conservar.

    pole_tol=0 o None desactiva la exclusion (comportamiento anterior)."""
    a = r.a
    k_arr = np.linspace(0.0, r.k_end, nk)
    wg = np.linspace(1e-3 * 2*np.pi*CT0/a, wmax * 2*np.pi*CT0/a, ngrid)
    wn = np.full((nk, maxbands), np.nan)
    im_out = np.full((nk, maxbands), np.nan)
    n_polos = 0
    for ik, k in enumerate(k_arr):
        libres = (_frecuencias_red_vacia(r, k, shells=shells, wmax=wmax + 0.1)
                  if pole_tol else np.array([]))
        tr = _eig_branches(r, k, wg, eta)
        rows = []
        for i in range(tr.shape[1]):
            re = np.real(tr[:, i]) - 1.0
            im = np.imag(tr[:, i])
            for j in np.nonzero(re[:-1] * re[1:] < 0)[0]:
                t = re[j] / (re[j] - re[j+1])
                imv = abs(im[j] + t*(im[j+1] - im[j]))
                if imv >= imtol:
                    continue
                w_c = (wg[j] + t*(wg[j+1]-wg[j])) * a/(2*np.pi*CT0)
                if libres.size:
                    d = np.abs(libres - w_c)
                    cerca = d < pole_tol
                    if cerca.any():
                        # proteccion cerca de Gamma: no excluir por el polo G=0
                        solo_g0 = np.all(libres[cerca] < pole_floor)
                        if not (solo_g0 and w_c < pole_floor):
                            n_polos += 1
                            continue
                rows.append((w_c, imv))
        rows = sorted(rows)[:maxbands]
        for b, (w, iv) in enumerate(rows):
            wn[ik, b] = w; im_out[ik, b] = iv
    if pole_tol:
        compute_bands_eig.ultimo_n_polos = n_polos
    return k_arr, wn, im_out


def build_red(lattice, psi, cut=2, nk=90, n_suma=5, imag_tol=0.8, sol_tol=1e-2,
              cond_borde="hollow", r1=0.45, r2=0.5, filling=0.5, a=1.0,
              dens=(1150, 1250), vel0=(295, 295), vels=(894, 894)):
    """Arma un objeto Red con TODOS los parametros de control expuestos:
    cut (orden multipolar), n_suma (terminos de la suma de red), imag_tol y
    sol_tol (tolerancias del solver), cond_borde, geometria (r1, r2, filling, a)
    y materiales (dens, vel0, vels). Los valores por defecto son los usados en
    los scripts hasta ahora; cambia lo que necesites al llamar la funcion."""
    r = Red(comp=["matriz", "inclusion"])
    r.dens = list(dens); r.vel0 = list(vel0); r.vels = list(vels)
    r.filling = filling; r.cut = cut; r.nbands = 12; r.nk = nk; r.n_suma = n_suma
    r.lattice = lattice; r.psi = psi; r.a = a; r._set_k_end()
    r.cond_borde = cond_borde; r.imag_tol = imag_tol; r.sol_tol = sol_tol
    r.asign_param(); r.r1 = r1; r.r2 = r2
    return r


def compute_bands(r, nk=90, wmax=1.4, ngrid=800, eta=1e-3, thr=0.15, maxbands=12):
    """Devuelve (k_arr, wn) con wn de forma (nk, maxbands), NaN-padded, en unidades ωa/2πCt0."""
    a = r.a
    k_arr = np.linspace(0.0, r.k_end, nk)
    wgrid = np.linspace(1e-3 * 2*np.pi*CT0/a, wmax * 2*np.pi*CT0/a, ngrid)
    dw = wgrid[1] - wgrid[0]
    wn = np.full((nk, maxbands), np.nan)
    for ik, k in enumerate(k_arr):
        mags = np.empty(ngrid)
        for j, w in enumerate(wgrid):
            re, im = r.determinant_longitudinal([w, eta], k, r.cut)
            mags[j] = abs(re + 1j*im)
        idx = argrelmin(mags, order=2)[0]
        idx = idx[mags[idx] < thr]
        ws = []
        for i in idx:
            if 1 <= i < ngrid-1:
                y0, y1, y2 = mags[i-1], mags[i], mags[i+1]
                den = (y0 - 2*y1 + y2)
                shift = 0.5*(y0 - y2)/den if abs(den) > 1e-30 else 0.0
                ws.append(wgrid[i] + shift*dw)
        ws = sorted(ws)[:maxbands]
        wn[ik, :len(ws)] = np.array(ws) * a / (2*np.pi*CT0)
    return k_arr, wn


if __name__ == "__main__":
    import time
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lat = sys.argv[1] if len(sys.argv) > 1 else "sq"
    for psi in [0.0, 0.8]:
        r = build_red(lat, psi)
        t0 = time.perf_counter()
        k_arr, wn = compute_bands(r)
        print("%s psi=%.1f: %.1f s, %d pts" % (lat, psi, time.perf_counter()-t0,
                                               np.sum(np.isfinite(wn))))
