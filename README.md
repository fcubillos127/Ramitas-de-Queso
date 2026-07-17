# Bandas fonónicas — arreglo periódico de inclusiones con recubrimiento

Cálculo de la **estructura de bandas acústica/fonónica** para una red periódica 2D
(cuadrada `sq` o hexagonal `hx`) de inclusiones cilíndricas con recubrimiento
(matriz + inclusión + coating controlado por el parámetro `psi`). El método es de
tipo *multiple-scattering / KKR*: se arma la suma de red `G0`, el `T`-matrix de un
dispersor (coeficientes de dispersión `T_n`) y las bandas salen de las raíces en
frecuencia de `det(T·G0 − I) = 0`.

## Estructura del proyecto

```
.
├── Bandas_Tools.py        # Clase Red: núcleo (G0, T-matrix, determinante, bandas, plots)
├── suma_de_red.py         # Sumas de red / structure constants (K, Kh, S, S_pre, ...)
├── utils.py               # Coeficientes A/B/C de las condiciones de borde elásticas
├── project_io.py          # Helpers de rutas: data/ y graphs/
├── main.py                # Script de ejemplo
├── requirements.txt
└── fnv/                   # Paquete F_n / V_n (integrales del recubrimiento)
    ├── __init__.py
    ├── fnv_core.py        # Integrandos e integración
    ├── fnv_store.py       # Contenedor FNVData + build_fnv_grid + save/load .npz
    ├── fnv_plot.py        # Gráficos de F_n / V_n
    └── fnv_csv.py         # Exportar/leer F_n / V_n en CSV
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso básico

```python
import numpy as np
from Bandas_Tools import Red

red = Red(comp=["matriz", "inclusion"])
red.dens = [1150, 1250]      # densidades [kg/m^3]
red.vel0 = [295, 295]        # [C_l, C_t] en la matriz [m/s]
red.vels = [894, 894]        # [C_l, C_t] en la inclusión [m/s]
red.filling = 0.5            # fracción de llenado
red.cut = 6                  # tamaño del T-matrix: (2*cut+1)
red.nbands = 5
red.nk = 50                  # puntos de k
red.n_suma = 5               # términos de la suma de red
red.lattice = "sq"           # 'sq' o 'hx'
red.psi = 1                  # parámetro del recubrimiento
red.a = 0.1                  # constante de red [m]
red._set_k_end()             # recalcular fin del camino de k (depende de a, lattice)
red.cond_borde = "rigid"     # ver nota abajo
red.imag_tol = 0.8
red.sol_tol = 1e-2

red.asign_param()            # deriva mu, lambda, k, carpetas y r1 (desde filling)
red.r1 = 0.45 * red.a        # OJO: sobrescribir r1 DESPUÉS de asign_param
red.r2 = 0.5 * red.a

red.graficar_dif_determinante(0, 0.8, np.pi, 6)   # (psi, psi2, k, cut)
```

### Orden de llamadas (importa)

- `asign_param()` **recalcula `r1` desde `filling`**, así que cualquier `r1`/`r2`
  manual debe fijarse **después** de `asign_param()`.
- `psi` debe estar seteado **antes** de `asign_param()` (se usa al armar la ruta de
  resultados).
- `_set_k_end()` depende de `a` y `lattice`; vuelve a llamarlo si cambias `a`.

### Sobre `cond_borde`

En `determinant_longitudinal`, `cond_borde='rigid'` usa
`coeficiente_dispersion_elastic`; cualquier otro valor usa
`coeficiente_dispersion_hollow`.

## Ejecutar sin ventana (servidores / CI / lotes)

`main.py` termina en `plt.show()`, que **bloquea** en modo interactivo. Para correr
headless, usa el backend `Agg` y guarda las figuras a disco:

```bash
MPLBACKEND=Agg python main.py
```

o desde Python:

```python
import matplotlib
matplotlib.use("Agg")   # antes de importar pyplot
```

Las figuras y datos se guardan bajo `graphs/` y `data/` (ver `project_io.py`).
