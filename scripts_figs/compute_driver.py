"""
Calcula la estructura de bandas (autovalores Re(mu)=1) para una lista de psi y
guarda un .npz con k, wn (frecuencias norm.) e im (|Im(mu)|, medida de fuga).

Como CLI:
    python compute_driver.py <lattice> <out.npz>
Como función (p.ej. en VSCode):
    from compute_driver import run
    run("sq", "data/bands_sq.npz", nk=70, cut=7)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bandcalc import build_red, compute_bands_eig, CT0

# Parámetros por defecto (edítalos o pásalos a run())
PSIS = [0.0, 0.2, 0.4, 0.6, 0.8]
NK, WMAX, NGRID, CUT = 70, 1.4, 1100, 7


def run(lattice, out, psis=PSIS, nk=NK, cut=CUT, ngrid=NGRID, wmax=WMAX):
    """Calcula y guarda las bandas para todos los psi. Devuelve la ruta del .npz."""
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    data = {"lattice": lattice, "psis": np.array(psis), "a": 1.0, "Ct0": CT0, "wmax": wmax}
    t_all = time.perf_counter()
    for i, psi in enumerate(psis):
        r = build_red(lattice, psi, cut=cut, nk=nk)
        t0 = time.perf_counter()
        k_arr, wn, im = compute_bands_eig(r, nk=nk, wmax=wmax, ngrid=ngrid)
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
