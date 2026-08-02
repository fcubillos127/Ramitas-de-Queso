"""
Punto de entrada (VSCode): correr la RUTINA ORIGINAL DE MIGUEL de principio a fin
para UNA estructura a la vez.

    solver de Miguel  ->  post-proceso automatico  ->  figura estilo articulo

  1. Red.zeros_longitudinal_fullgrid  : el solver original (barrido de cambios de
     signo + fsolve). Lento. Guarda en red.omega_longitudinal.
  2. postprocess_miguel.post_process  : QUITA los fantasmas de red vacia (cadenas
     en X) y AGREGA lo que el barrido se salto (bandas planas ~1.1-1.4 y tramos
     de la acustica con psi grande). Ver scripts_figs/README.md seccion 4b.
  3. plot_bands.make_figures(clean=False) : figura con el camino de alta simetria
     EQUIespaciado y en el orden natural correcto  M-Gamma-X-M (cuadrada) /
     M-K-Gamma-M (triangular).  clean=False = NO re-borrar lo que agrego el paso 2.

Como usarlo en VSCode:
  - Abre la carpeta del repo como workspace, elige el interprete de tu .venv.
  - VSCode detecta las celdas '# %%': aparece "Run Cell" arriba de cada una
    (o Shift+Enter). Corre las celdas en orden.
  - Para tantear: cambia los parametros de la celda [2] y re-corre [2]-[4].

OJO velocidad: el solver (paso 1) tarda ~15 s (psi=0) a ~75 s (psi=0.8) con
nk=50, cut=2; el completado del paso 2 suma ~40-110 s. Un psi completo son
~1-3 min. Para probar rapido: baja nk (p.ej. 30) o pon COMPLETAR=False.
"""

# %% [1] Imports y rutas
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                 # raiz del repo
sys.path.insert(0, os.path.join(HERE, "scripts_figs"))
import numpy as np
import matplotlib.pyplot as plt

from bandcalc import build_red                            # arma el objeto Red
from postprocess_miguel import post_process               # limpieza + completado
from plot_bands import make_figures                       # figura equiespaciada M-G-X-M
from bridge_to_omega import red_to_eig_npz                # exporta a .npz para plot_bands

os.makedirs("data", exist_ok=True)
os.makedirs("graphs", exist_ok=True)
print("OK - configurado.")


# %% [2] Elegir UNA estructura y sus parametros
LATTICE = "sq"       # "sq" cuadrada  |  "hx" triangular
PSI     = 0.8        # pre-deformacion angular
NK      = 50         # puntos de k sobre el camino (usa 30 para probar rapido)
CUT     = 6          # modos m in {-CUT..CUT}. 6 = el valor del main.py de Miguel y el
                     # que REPRODUCE las Figs. 3-4 (verificado). Con cut=2 NO aparece
                     # la banda plana de ~1.02 que protagoniza la Fig. 4.
NBANDS  = 8          # nº de soluciones que guarda el solver por cada k
N_SUMA  = 5          # terminos de la suma de red (convergencia de G0)
WMIN    = 1e-3       # piso de omega normalizada a EXPLORAR
WMAX    = 1.4        # tope de omega normalizada a EXPLORAR (el solver no busca mas arriba
                     # de esto; si subes YHI/ZOOM_YHI, sube WMAX tambien o no habra datos)
# Para buscar SOLO en una franja (p.ej. la banda plana): WMIN=1.0, WMAX=1.1.
# OJO 1: el nº de muestras es VENTANAS_POR_UNIDAD*(WMAX-WMIN), o sea la DENSIDAD
#        se mantiene. Con una franja angosta quedan pocas muestras en total
#        (100*0.1 = 10), asi que conviene SUBIR VENTANAS_POR_UNIDAD (p.ej. 1000
#        -> 100 muestras en la franja, 10x mas fino que el barrido completo y
#        aun asi mas rapido, porque la franja es 14x mas angosta).
# OJO 2: el filtro de aceptacion del solver es  0 < w_norm < WMAX  (ver
#        resolver_con_fsolve): el limite inferior es 0, NO WMIN. Si fsolve se
#        desplaza hacia abajo puede devolver soluciones por debajo de WMIN;
#        filtralas despues si te estorban.

# Tolerancias del solver de Miguel (fsolve dentro de zeros_longitudinal_fullgrid):
IMAG_TOL = 0.8       # descarta soluciones con |Im(omega)| mayor (fuga)
SOL_TOL  = 1e-2      # xtol de fsolve
VENTANAS_POR_UNIDAD = 100   # densidad del muestreo de Re(det) al buscar cambios de signo

# Post-proceso (cada paso se apaga por separado):
FANTASMAS  = True    # QUITA los puntos pegados a las curvas de red vacia |k+G|
AISLADOS   = True    # QUITA los puntos sin enlace con ningun k vecino
COMPLETAR  = True    # AGREGA lo que el barrido se salto (LENTO: recalcula T*G0)
INTERPOLAR = True    # RELLENA huecos internos de <= 1 paso de k
# Para ver los datos CRUDOS del solver, sin tocar nada: pon los cuatro en False
# (o comenta la llamada a post_process en la celda [3]).

# Figura (celda [4]):
YHI      = 1.4       # tope del eje omega en la figura "full" (<= WMAX, si no, se ve vacio arriba)
ZOOM_YLO = 0.7        # rango del eje omega en la figura "zoom"
ZOOM_YHI = 1.2        # (subelo hasta WMAX si quieres ver mas alto, p.ej. 1.7 -> sube WMAX tambien)

r = build_red(LATTICE, PSI, cut=CUT, nk=NK, n_suma=N_SUMA,
              imag_tol=IMAG_TOL, sol_tol=SOL_TOL, cond_borde="hollow",
              r1=0.45, r2=0.5, filling=0.5, a=1.0)
r.nbands = NBANDS
# guardar los .txt del solver en una carpeta local y predecible (asign_param los
# habia apuntado a ~/Documents/...); asi no ensucia tu carpeta de usuario:
r.foldername = os.path.join(HERE, "data", "miguel_%s" % LATTICE)
r.frecfolder = os.path.join(r.foldername, "psi_%s" % PSI)
os.makedirs(r.frecfolder, exist_ok=True)
print("Red lista: %s  psi=%.2f  nk=%d cut=%d nbands=%d" % (LATTICE, PSI, NK, CUT, NBANDS))


# %% [3] Correr el SOLVER de Miguel  (LENTO) + post-proceso
#   C_l0 = Ct0 = vel0[1] = 295 -> normaliza el eje a omega*a/2pi*Ct0
r.zeros_longitudinal_fullgrid(C_l0=float(r.vel0[1]),
                              ventanas_por_unidad=VENTANAS_POR_UNIDAD,
                              w_norm_min=WMIN, w_norm_max=WMAX)
# post_process modifica r.omega_longitudinal in-place; backup en
# r._omega_backup_postprocess (deshacer: r.omega_longitudinal = r._omega_backup_postprocess.copy())
post_process(r, fantasmas=FANTASMAS, aislados=AISLADOS,
             completar=COMPLETAR, interpolar=INTERPOLAR,
             graficar=False)          # graficar=False: usamos plot_bands abajo
print("Post-proceso listo. Puntos finitos:",
      int(np.sum(np.isfinite(r.omega_longitudinal[:, :, 0]))))


# %% [4] Figura estilo articulo (camino M-Gamma-X-M equiespaciado)
#   Exporta lo post-procesado a .npz y grafica con clean=False (respeta el post-proceso).
npz = os.path.join("data", "miguel_%s_psi%s.npz" % (LATTICE, PSI))
red_to_eig_npz(r, npz, psi=PSI)
make_figures(npz, os.path.join("graphs", "miguel_%s_psi%s" % (LATTICE, PSI)),
             imtol=1.0, clean=False, show=True,   # imtol=1.0: los datos ya estan filtrados
             yhi=YHI, zoom_ylo=ZOOM_YLO, zoom_yhi=ZOOM_YHI)
#   -> graphs/miguel_<lat>_psi<psi>_full.png  y  _zoom.png
#
# NOTA sobre etiquetas: r.graficar_bandas_grid() (de Miguel) rotula X-Gamma-M-X,
# pero sus etiquetas X<->M estan intercambiadas respecto de lo que calcula el
# codigo. plot_bands (esta figura) usa el rotulo correcto M-Gamma-X-M. La forma
# de las curvas es la misma; solo cambian bien puestas las etiquetas de extremos.
