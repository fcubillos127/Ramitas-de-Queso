"""
Explorador de UNA estructura (un lattice, un psi) con control total de los
parámetros numéricos, para tantear combinaciones antes de pasar al draft con
varios psi (ejemplo_completo.py / compute_driver.run).

Como función (VSCode / Jupyter):
    from explore_one import compute_one, plot_one, compare

    k, wn, im, t = compute_one("sq", psi=0.0, cut=2, n_suma=5, nk=50, ngrid=900,
                                wmax=1.4, maxbands=14)
    plot_one(k, wn, "sq", im=im, imtol=0.12, title="sq  ψ=0  cut=2  n_suma=5")

    # comparar varias configuraciones lado a lado (p.ej. cut=2 vs cut=7):
    compare([
        dict(lattice="sq", psi=0.0, cut=2, n_suma=5, label="cut=2"),
        dict(lattice="sq", psi=0.0, cut=7, n_suma=5, label="cut=7"),
    ])

Nota sobre materiales: por ahora estos scripts normalizan la frecuencia con un
Ct0 FIJO (295 m/s, el de bandcalc.CT0), desacoplado de lo que le pongas a
r.vel0. Si cambias vel0/dens/vels sin arreglar antes esa normalización, los
resultados quedan mal escalados. Por eso compute_one() NO expone materiales
todavía: solo geometría (r1, r2, filling, a) y parámetros numéricos (cut,
n_suma, nk, ngrid, wmax, eta, imtol, maxbands, imag_tol, sol_tol, cond_borde).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import matplotlib.pyplot as plt
from bandcalc import build_red, compute_bands_eig, CT0
from plot_bands import panel


def compute_one(lattice, psi, cut=2, n_suma=5, nk=50, ngrid=900, wmax=1.4,
                 eta=1e-3, imtol=0.6, maxbands=14, r1=0.45, r2=0.5, filling=0.5,
                 a=1.0, imag_tol=0.8, sol_tol=1e-2, cond_borde="hollow",
                 verbose=True):
    """Calcula UNA estructura (un lattice, un psi) con todos los parámetros de
    control expuestos. Devuelve (k, wn, im, elapsed_s).

    - cut: orden multipolar, modos m ∈ {-cut..cut} (tamaño de matriz 2*cut+1).
    - n_suma: términos de la suma de red (convergencia de G0).
    - nk: puntos de k a lo largo del camino de alta simetría.
    - ngrid: puntos de omega muestreados para detectar cruces Re(mu)=1
      (más = más preciso/lento; muy bajo puede confundir el seguimiento de
      autovalores y ensuciar la banda).
    - wmax: tope superior de omega normalizada (omega*a / 2*pi*Ct0) a explorar.
    - eta: parte imaginaria fija de omega al evaluar T(w), G0(w).
    - imtol: corte GRUESO de |Im(mu)| al aceptar un cruce durante el cálculo
      (el corte fino, sin recalcular, se hace en plot_one/imtol).
    - maxbands: cuántas bandas (cruces) se retienen como máximo por punto de k.
    - r1, r2, filling, a: geometría de la celda.
    - imag_tol, sol_tol, cond_borde: solo relevantes si además usas el solver
      original de Miguel (zeros_longitudinal_fullgrid) sobre este mismo Red.
    """
    r = build_red(lattice, psi, cut=cut, nk=nk, n_suma=n_suma, imag_tol=imag_tol,
                  sol_tol=sol_tol, cond_borde=cond_borde, r1=r1, r2=r2,
                  filling=filling, a=a)
    t0 = time.perf_counter()
    k, wn, im = compute_bands_eig(r, nk=nk, wmax=wmax, ngrid=ngrid, eta=eta,
                                   imtol=imtol, maxbands=maxbands)
    elapsed = time.perf_counter() - t0
    if verbose:
        npts = int(np.sum(np.isfinite(wn)))
        print("[%s] psi=%.2f cut=%d n_suma=%d nk=%d ngrid=%d wmax=%.2f -> "
              "%.1f s, %d puntos" % (lattice, psi, cut, n_suma, nk, ngrid, wmax,
                                     elapsed, npts))
    return k, wn, im, elapsed


def plot_one(k, wn, lattice, a=1.0, Ct0=CT0, im=None, imtol=0.12,
             ylo=0.0, yhi=1.4, title=None, ax=None, figsize=(5, 4.5)):
    """Grafica UNA estructura ya calculada. Aplica el corte fino `imtol` sobre
    `im` SIN recalcular (bueno para iterar rápido el nivel de limpieza)."""
    wn_show = wn.copy()
    if im is not None:
        wn_show[im > imtol] = np.nan
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    panel(ax, lattice, a, Ct0, k, wn_show, ylo, yhi, title or "")
    ax.set_ylabel(r"$\omega a/2\pi C_{t0}$", fontsize=12)
    if standalone:
        fig.tight_layout()
        return fig, ax
    return ax


def compare(configs, ylo=0.0, yhi=1.4, imtol=0.12, figsize_per=3.2):
    """Calcula y grafica varias configuraciones lado a lado, para comparar
    directamente el efecto de un parámetro (p.ej. distintos cut o n_suma para
    el mismo psi/lattice). `configs`: lista de dicts con los kwargs de
    compute_one (más opcional 'label' para el título del panel)."""
    results = []
    for cfg in configs:
        cfg = dict(cfg)
        label = cfg.pop("label", None)
        k, wn, im, t = compute_one(**cfg)
        lat = cfg.get("lattice", "sq")
        auto_label = "%s ψ=%.1f cut=%d n_suma=%d" % (
            lat, cfg.get("psi", 0.0), cfg.get("cut", 2), cfg.get("n_suma", 5))
        results.append((k, wn, im, lat, cfg.get("a", 1.0), label or auto_label))
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per*n, 4.4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (k, wn, im, lat, a, label) in zip(axes, results):
        plot_one(k, wn, lat, a=a, im=im, imtol=imtol, ylo=ylo, yhi=yhi,
                 title=label, ax=ax)
    fig.tight_layout()
    return fig, axes, results
