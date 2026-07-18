"""
Ejemplo end-to-end (VSCode): calcular bandas -> graficar -> editar a mano -> figura final.

Cómo usarlo en VSCode:
  1) Abre la carpeta del repo como workspace y elige el intérprete de tu .venv.
  2) Abre este archivo. VSCode detecta las celdas '# %%': aparece "Run Cell" arriba
     de cada una (o usa Shift+Enter). Se ejecuta en la Ventana Interactiva y las
     figuras salen inline.
  3) Corre las celdas en orden. La celda de CÁLCULO es lenta (cut=7 ~20 min/red):
     para probar rápido usa NK=30, CUT=2 la primera vez.

Todo se corre desde la raíz del repo. Salidas: data/ (npz) y graphs/ (png).
"""

# %% [1] Configuración e imports
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                          # raíz del repo
sys.path.insert(0, os.path.join(HERE, "scripts_figs"))
import numpy as np
import matplotlib.pyplot as plt

from compute_driver import run                    # calcular bandas
from plot_bands import make_figures               # figura (segmentos equiespaciados)
from bridge_to_omega import eig_to_red, red_to_eig_npz   # editar a mano
import gap_vs_psi as gvp                           # gap vs psi
import compute_mode                                # modos

# --- parámetros que puedes tocar ---
NK   = 70      # puntos de k por camino (usa 30 para pruebas rápidas)
CUT  = 7       # modos angulares m ∈ {-CUT..CUT} (usa 2 para pruebas rápidas)
IMTOL = 0.12   # corte de 'fuga' |Im(mu)| al graficar (0.10-0.12 = limpio)
os.makedirs("data", exist_ok=True); os.makedirs("graphs", exist_ok=True)
print("OK — configurado. NK=%d CUT=%d" % (NK, CUT))


# %% [2] Calcular bandas  (LENTO con cut=7; sáltate esta celda si ya tienes el .npz)
run("sq", "data/bands_sq.npz", nk=NK, cut=CUT)
run("hx", "data/bands_hx.npz", nk=NK, cut=CUT)


# %% [3] Graficar bandas (X-Γ-M-X / Γ-M-K-Γ, tramos equiespaciados) — se ven inline
make_figures("data/bands_sq.npz", "graphs/bandas_sq", imtol=IMTOL, show=True)
make_figures("data/bands_hx.npz", "graphs/bandas_hx", imtol=IMTOL, show=True)


# %% [4] Puente a las herramientas de edición: cargar un psi y ver los índices
#     psi_index: 0->psi=0.0, 1->0.2, 2->0.4, 3->0.6, 4->0.8
red = eig_to_red("data/bands_sq.npz", psi_index=4, imtol=IMTOL, cut=CUT)
red.order_bands_by_continuity_global()     # reordena bandas por continuidad
red.graficar_bandas_grid(ylim=[0, 1.4])    # mira aquí los puntos espurios a borrar
# (índice i = punto de k [0..nk-1];  n = número de banda [0..nbands-1])


# %% [5] Editar a mano: borrar espurios, suavizar y volver a ver
red.delete_point(i=30, n=5, mode="fullgrid")   # <-- ajusta (i, n) a lo que veas
# red.delete_point(i=31, n=5, mode="fullgrid")
# red.restore_deleted()                        # deshacer el último borrado
red.smooth_interpolate_longitudinal()          # rellena huecos internos
red.graficar_bandas_grid(ylim=[0, 1.4])


# %% [6] Exportar lo editado y figura final con segmentos equiespaciados
red_to_eig_npz(red, "data/bands_sq_edit.npz", psi=float(red.psi))
make_figures("data/bands_sq_edit.npz", "graphs/bandas_sq_edit", imtol=1.0, show=True)
#   (imtol=1.0 aquí porque los datos editados ya están filtrados)


# %% [7] Gap vs psi  (cuadrada: desdoblamiento en M ; triangular: corrimiento de K)
gsq = gvp.sq_gap_detmin()
psh, wK = gvp.hx_diracK_from_bands("data/bands_hx.npz")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
a1.plot(gvp.PSIS, gsq, "o-"); a1.set_title("cuadrada — gap en M")
a1.set_xlabel(r"$\psi$"); a1.set_ylabel(r"gap $\Delta(\omega a/2\pi C_{t0})$"); a1.grid(alpha=.3)
a2.plot(psh, wK, "s-", color="crimson"); a2.set_title("triangular — corrimiento de K")
a2.set_xlabel(r"$\psi$"); a2.set_ylabel(r"$\omega_K a/2\pi C_{t0}$"); a2.grid(alpha=.3)
fig.tight_layout(); fig.savefig("graphs/gap_vs_psi.png", dpi=160, bbox_inches="tight")
plt.show()


# %% [8] Modos antiplanos Re(u_z) en M (psi=0 vs psi=0.8)  -> guarda un PNG
compute_mode.main("graphs/modos_sq.png")
print("Modo guardado en graphs/modos_sq.png")
