"""
Explorar UNA estructura a la vez (un lattice, un psi): elegir parámetros,
calcular, mirar, ajustar. Cuando estés conforme, pasas a ejemplo_completo.py
para el draft con varios psi usando los parámetros que hayas encontrado aquí.

Cómo usarlo en VSCode: igual que ejemplo_completo.py — abre el archivo, corre
las celdas '# %%' en orden (botón "Run Cell" o Shift+Enter). Las figuras salen
inline en la Ventana Interactiva.
"""

# %% [1] Imports
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts_figs"))
import numpy as np
import matplotlib.pyplot as plt

from explore_one import compute_one, plot_one, compare
print("OK — listo para explorar")


# %% [2] UNA estructura: elige parámetros, calcula y grafica
LATTICE = "sq"      # "sq" (cuadrada) o "hx" (triangular)
PSI     = 0.0

CUT     = 2         # modos angulares m ∈ {-CUT..CUT}   (la tesis usó 2)
N_SUMA  = 5         # términos de la suma de red
NBANDS  = 14         # cuántos cruces (bandas) retener como máximo por k
WMAX    = 1.4        # tope superior de omega_norm
NK      = 50          # puntos de k a lo largo del camino
NGRID   = 900        # densidad de la malla de omega (precisión del cruce)
ETA     = 1e-3        # parte imaginaria fija al evaluar T, G0

R1, R2  = 0.45, 0.5    # geometría de la celda (radios, en unidades de a)

k, wn, im, t = compute_one(LATTICE, PSI, cut=CUT, n_suma=N_SUMA, nk=NK,
                            ngrid=NGRID, wmax=WMAX, eta=ETA, maxbands=NBANDS,
                            r1=R1, r2=R2)

PLOT_IMTOL = 0.12   # corte fino de |Im(mu)| al graficar (no recalcula)
plot_one(k, wn, LATTICE, im=im, imtol=PLOT_IMTOL, ylo=0.0, yhi=WMAX,
          title="%s  ψ=%.2f  cut=%d  n_suma=%d  (%.1fs)" % (LATTICE, PSI, CUT, N_SUMA, t))


# %% [3] Ajustar el corte de fuga (PLOT_IMTOL) SIN recalcular — iterar rápido
fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
for ax, tol in zip(axes, [0.06, 0.10, 0.15, 0.30]):
    plot_one(k, wn, LATTICE, im=im, imtol=tol, ylo=0.0, yhi=WMAX,
             title="IMTOL=%.2f" % tol, ax=ax)
fig.tight_layout()


# %% [4] Comparar configuraciones lado a lado (p.ej. distintos CUT, mismo psi)
fig, axes, results = compare([
    dict(lattice=LATTICE, psi=PSI, cut=2, n_suma=5, nk=NK, ngrid=NGRID, wmax=WMAX, label="cut=2"),
    dict(lattice=LATTICE, psi=PSI, cut=4, n_suma=5, nk=NK, ngrid=NGRID, wmax=WMAX, label="cut=4"),
    dict(lattice=LATTICE, psi=PSI, cut=7, n_suma=5, nk=NK, ngrid=NGRID, wmax=WMAX, label="cut=7"),
], imtol=PLOT_IMTOL, ylo=0.0, yhi=WMAX)


# %% [5] Comparar N_SUMA (dejando cut fijo en el que hayas elegido)
fig, axes, results = compare([
    dict(lattice=LATTICE, psi=PSI, cut=CUT, n_suma=3, nk=NK, ngrid=NGRID, wmax=WMAX, label="n_suma=3"),
    dict(lattice=LATTICE, psi=PSI, cut=CUT, n_suma=5, nk=NK, ngrid=NGRID, wmax=WMAX, label="n_suma=5"),
    dict(lattice=LATTICE, psi=PSI, cut=CUT, n_suma=9, nk=NK, ngrid=NGRID, wmax=WMAX, label="n_suma=9"),
], imtol=PLOT_IMTOL, ylo=0.0, yhi=WMAX)


# %% [6] Cuando estés conforme: anota (CUT, N_SUMA, NGRID, NK, PLOT_IMTOL, ...)
#         y pásalos a ejemplo_completo.py (celda [1]) para el draft con varios psi.
print("Parámetros elegidos: cut=%d n_suma=%d nk=%d ngrid=%d wmax=%.2f imtol=%.2f"
      % (CUT, N_SUMA, NK, NGRID, WMAX, PLOT_IMTOL))
