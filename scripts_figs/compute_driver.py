"""
Calcula la estructura de bandas (autovalores Re(mu)=1) para una lista de psi y
guarda un .npz con k, wn (frecuencias norm.) e im (|Im(mu)|, medida de fuga).

Como CLI:
    python compute_driver.py <lattice> <out.npz>
Como función (p.ej. en VSCode), con control total de parámetros:
    from compute_driver import run
    run("sq", "data/bands_sq.npz", nk=70, cut=7, n_suma=8, eta=1e-3, imtol=0.6)

Qué parámetro afecta qué (importante para no confundirse):
  - cut, n_suma, eta, imtol, ngrid, wmax  -> SÍ afectan el método por
    autovalores (compute_bands_eig), que es el que usan estos scripts.
  - imag_tol, sol_tol                     -> NO los usa compute_bands_eig (no
    hay fsolve en este método). Solo importan si además corres el solver
    original de Miguel (Red.zeros_longitudinal_fullgrid) sobre el mismo
    objeto Red, p.ej. vía bridge_to_omega. Se dejan pasar por si acaso.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bandcalc import build_red, compute_bands_eig, CT0

# Parámetros por defecto (edítalos aquí, o pásalos directo a run())
PSIS = [0.0, 0.2, 0.4, 0.6, 0.8]
NK, WMAX, NGRID = 70, 1.4, 1100
CUT = 7
N_SUMA = 5          # términos de la suma de red (convergencia de G0)
ETA = 1e-3          # parte imaginaria de omega usada al evaluar T(w), G0(w)
IMTOL = 0.6         # corte de |Im(mu)| al ACEPTAR un cruce (grueso; el fino se
                    # hace después en plot_bands.IMTOL sin recalcular)
IMAG_TOL = 0.8      # solo relevante si usas zeros_longitudinal_fullgrid
SOL_TOL = 1e-2      # idem
COND_BORDE = "hollow"


def run(lattice, out, psis=PSIS, nk=NK, cut=CUT, ngrid=NGRID, wmax=WMAX,
        n_suma=N_SUMA, eta=ETA, imtol=IMTOL,
        imag_tol=IMAG_TOL, sol_tol=SOL_TOL, cond_borde=COND_BORDE,
        r1=0.45, r2=0.5, filling=0.5, a=1.0):
    """Calcula y guarda las bandas para todos los psi. Devuelve la ruta del .npz."""
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    data = {"lattice": lattice, "psis": np.array(psis), "a": a, "Ct0": CT0, "wmax": wmax,
            "cut": cut, "n_suma": n_suma, "eta": eta, "imtol": imtol}
    t_all = time.perf_counter()
    for i, psi in enumerate(psis):
        r = build_red(lattice, psi, cut=cut, nk=nk, n_suma=n_suma,
                       imag_tol=imag_tol, sol_tol=sol_tol, cond_borde=cond_borde,
                       r1=r1, r2=r2, filling=filling, a=a)
        t0 = time.perf_counter()
        k_arr, wn, im = compute_bands_eig(r, nk=nk, wmax=wmax, ngrid=ngrid,
                                           eta=eta, imtol=imtol)
        data["k_%d" % i] = k_arr
        data["wn_%d" % i] = wn
        data["im_%d" % i] = im
        np.savez(out, **data)   # guardado incremental
        print("[%s] psi=%.1f  %.1f s  (%d pts)" % (lattice, psi, time.perf_counter()-t0,
                                                   int(np.sum(np.isfinite(wn)))), flush=True)
    print("[%s] TOTAL %.1f s -> %s" % (lattice, time.perf_counter()-t_all, out), flush=True)
    return out


if __name__ == "__main__":
    LAT = sys.argv[1] if len(sys.argv) > 1 else "sq"
    OUT = sys.argv[2] if len(sys.argv) > 2 else "bands_%s.npz" % LAT
    run(LAT, OUT)
