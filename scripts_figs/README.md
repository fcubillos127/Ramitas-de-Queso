# scripts_figs — figuras de bandas, modos y gap

Rutinas para reproducir las figuras (red **cuadrada** y **triangular**, cavidad
recubierta con `r1 = 0.45a`, `r2 = 0.5a`, para `ψ = 0, 0.2, 0.4, 0.6, 0.8`).

> **¿Solo quieres correrlo?** Hay dos puntos de entrada en la raíz del repo, abre
> el que corresponda en VSCode y corre sus celdas `# %%` una por una:
> - **`explorar_estructura.py`** — UNA estructura (un lattice, un psi) a la vez,
>   con todos los parámetros numéricos a mano, para tantear `cut`, `n_suma`,
>   `nk`, `ngrid`, `wmax`, nº de bandas, etc. antes de comprometerte a un draft.
> - **`ejemplo_completo.py`** — el draft completo con varios psi: calcular →
>   graficar → editar a mano → figura final → gap → modos.
>
> Lo de abajo es la referencia de cada script por separado.

Todos los scripts añaden solo la raíz del repo al `sys.path`, así que se corren
**desde la raíz del proyecto** (donde están `Bandas_Tools.py`, `suma_de_red.py`, …).

## Requisitos

```bash
pip install -r requirements.txt      # numpy, scipy, matplotlib, pandas, tqdm
```

En VSCode: abre la carpeta del repo como workspace y usa el intérprete donde
instalaste eso. Las figuras se guardan donde indique el 2º argumento.

## Método (resumen)

Una banda es donde el determinante `det(T·G0 − I) = 0`. En vez de buscar mínimos
de `|det|` (frágil: es el *producto* de los factores, borra ramas cercanas y su
mínimo no siempre cruza un umbral), rastreamos los **autovalores `μᵢ` de `T·G0`**
en ramas continuas en `ω` y tomamos los **cruces `Re(μᵢ) = 1`**. Es una condición
de cruce (sin umbral de magnitud) y separa bandas juntas. Se guarda `|Im(μ)|`
(medida de "fuga" del modo) para filtrar en el graficado. Como `T` no depende de
`k`, se cachea y se reutiliza entre todos los `k`.

## 1) Estructura de bandas

```bash
# calcular (guarda un .npz con k, wn=frecuencias, im=|Im(mu)|)
python scripts_figs/compute_driver.py sq  data/bands_sq.npz
python scripts_figs/compute_driver.py hx  data/bands_hx.npz

# graficar (full 0-1.4 y zoom 0.7-1.2), estilo tesis, camino X-Γ-M-X / Γ-M-K-Γ
python scripts_figs/plot_bands.py data/bands_sq.npz  graphs/bandas_sq
python scripts_figs/plot_bands.py data/bands_hx.npz  graphs/bandas_hx
```

### Parámetros: cuáles afectan qué

`compute_driver.run(...)` expone **todos** los parámetros de control (también
editables como constantes arriba del archivo, o pasándolos directo a `run()`):

```python
run("sq", "data/bands_sq.npz",
    nk=70, cut=7, ngrid=1100, wmax=1.4,      # malla / geometría del cálculo
    n_suma=5, eta=1e-3, imtol=0.6,           # suma de red y solver por autovalores
    imag_tol=0.8, sol_tol=1e-2,              # solo si además usas zeros_longitudinal_fullgrid
    cond_borde="hollow", r1=0.45, r2=0.5, filling=0.5, a=1.0)
```

| Parámetro | Qué controla | ¿Afecta el método por autovalores? |
|---|---|---|
| `cut` | orden multipolar, modos `m ∈ {-cut..cut}` | **Sí** (tamaño de la matriz `T·G0`) |
| `n_suma` | términos de la suma de red (convergencia de `G0`) | **Sí** |
| `nk` | puntos de `k` por camino | **Sí** (densidad de la banda) |
| `ngrid` | puntos de `ω` al buscar cruces `Re(μ)=1` | **Sí** (resolución/precisión) |
| `eta` | parte imaginaria fija de `ω` al evaluar `T(ω)`, `G0(ω)` | **Sí** (picos más/menos agudos) |
| `imtol` (en `compute_bands_eig`/`run`) | corte **grueso** de `\|Im(μ)\|` al aceptar un cruce durante el cálculo | **Sí**, pero grueso |
| `IMTOL` (en `plot_bands.py`) | corte **fino** de `\|Im(μ)\|` al graficar | No recalcula — se aplica sobre datos ya guardados |
| `imag_tol`, `sol_tol` | tolerancias del solver original de Miguel (`fsolve` en `zeros_longitudinal_fullgrid`) | **No** — el método por autovalores no llama a `fsolve` |
| `r1`, `r2`, `filling`, `a`, materiales | geometría/física de la celda | **Sí** |

En resumen: para "tantear" combinaciones, lo que hay que mover es
`cut`, `n_suma`, `ngrid`, `eta` (recalculan) y `IMTOL` de `plot_bands.py`
(no recalcula, es gratis iterar).

## 2) Gap vs. pre-deformación

```bash
python scripts_figs/gap_vs_psi.py graphs/
# -> graphs/gap_vs_psi.png  y  graphs/gap_vs_psi.npz
```
Mide la separación de las dos bandas cerca del punto de alta simetría:
desdoblamiento en **M** (cuadrada) y apertura del cono de **Dirac en K**
(triangular). Ventanas y punto central en el dict `CONF`.

## 3) Modos (campo antiplano `Re(u_z)`)

```bash
python scripts_figs/compute_mode.py graphs/modos_sq.png
```
Reconstruye el modo desde el vector nulo de `(T·G0 − I)` en el punto M:
`u_z(r,θ) = Σ_m a_m [J_m(k0 r) + T_m H_m(k0 r)] e^{imθ}`. Las frecuencias objetivo
están en la lista `targets`.

## 4b) Post-procesamiento automático del solver ORIGINAL de Miguel

`postprocess_miguel.py` corre DESPUÉS de `Red.zeros_longitudinal_fullgrid`
(el solver con ventanas + `fsolve`), sin tocar `Bandas_Tools.py`:

```python
from Bandas_Tools import Red
import sys; sys.path.insert(0, "scripts_figs")
from postprocess_miguel import post_process

red = Red(...)
# ... parámetros, asign_param(), etc. ...
red.zeros_longitudinal_fullgrid(C_l0=295.0, ventanas_por_unidad=100, w_norm_max=1.4)
post_process(red)   # limpia in-place y re-grafica con graficar_bandas_grid
```

Hace, en orden: (1) detecta puntos aislados por **enlace de paso** (compara
cada punto solo contra sus vecinos de `k` inmediatos, no contra una ventana
absoluta — así no confunde una banda con pendiente real, como la acústica,
con ruido) y los borra con `red.delete_point(mode="fullgrid", preview=False)`
(reversible con `red.restore_deleted()`); (2) rellena huecos internos de a lo
más 1 paso de `k` (`max_gap`), **y solo si** los dos valores que flanquean el
hueco son parecidos entre sí — con interpolación propia, no con
`red.smooth_interpolate_longitudinal()`; (3) re-grafica.

⚠️ **Dos rondas de verificación, no una** — ambas encontraron problemas reales
que se corrigieron antes de dar el resultado por bueno:
- Primero, un detector de espurios por "ventana absoluta de frecuencia"
  **borraba hasta 47% de los puntos, incluida la banda acústica completa**
  (tiene pendiente real que excede cualquier ventana fija en pocos pasos) →
  se rediseñó a "enlace de paso" tras comparar visualmente antes/después.
- Después, usar `red.smooth_interpolate_longitudinal()` (sin límite de tamaño
  de hueco) **fabricó decenas de puntos** a través de huecos de 8-10 pasos en
  zonas ruidosas de resonancia. Al limitarlo a huecos de 1 paso, seguía
  fabricando puntos ocasionales: el solver ordena por frecuencia ascendente en
  cada `k` por separado sin rastrear la rama física, así que un hueco de 1
  paso puede estar flanqueado por dos ramas físicas distintas (caso real:
  ω_norm=0.05 saltando a 0.98 con un punto de hueco en medio; sin chequeo
  adicional, la interpolación inventaba un punto intermedio ~0.52 que no
  corresponde a nada calculado). Se agregó la condición de que los valores
  flanqueantes deben ser parecidos antes de rellenar.

Verificado sobre datos reales (`nk=50, cut=2`, ψ=0.0 y ψ=0.8) tras ambas
correcciones: elimina ~9-12% de puntos genuinamente aislados, cambio neto de
puntos ≈0, y ya no aparecen puntos fabricados fuera de tendencia (comparación
visual antes/después/borrado/interpolado). **No** elimina "islas" de 2-3
puntos que casualmente se enlazan entre sí — solo puntos totalmente sueltos.

⚠️ **No usa `red.order_bands_by_continuity_global()`**: verificado que (a) no
modifica `self.omega_longitudinal` (escribe en un atributo aparte,
`self.omega_longitudinal_ordered`) y (b) con sus parámetros por defecto puede
vaciar TODOS los puntos en datos reales (un caso probado: 67 finitos → 0). No
se investigó más por tocar un subsistema aparte fuera de este alcance.

## 4) Editar a mano las bandas (puente a las herramientas de la clase Red)

Las rutinas de edición del código (`delete_point`, `order_bands_by_continuity_global`,
`smooth_interpolate_longitudinal`, `graficar_bandas_grid`) operan sobre
`self.omega_longitudinal`. `bridge_to_omega.py` carga ahí las bandas por
autovalores (ya con el corte de fuga `imtol`), para partir del punto más limpio:

```python
from scripts_figs.bridge_to_omega import eig_to_red, red_to_eig_npz

red = eig_to_red("data/bands_sq.npz", psi_index=4, imtol=0.12)  # psi=0.8, corte de fuga
red.order_bands_by_continuity_global()      # reordena bandas por continuidad
red.delete_point(i=30, n=5, mode="fullgrid")# borra un espurio (undo: red.restore_deleted())
red.smooth_interpolate_longitudinal()       # rellena huecos internos
red.graficar_bandas_grid(ylim=[0, 1.4])     # grafica con el estilo del código

# exportar lo editado y reploteo con segmentos EQUIespaciados:
red_to_eig_npz(red, "data/bands_sq_edit.npz", psi=0.8)
# python scripts_figs/plot_bands.py data/bands_sq_edit.npz graphs/bandas_sq_edit
```

- `IMTOL`/`imtol` (corte `|Im(μ)|`): **≈0.10–0.12** deja las bandas propagantes
  limpias (ver barrido). Súbelo si quieres conservar más bandas planas de resonancia.
- Los ejes salen con **tramos de alta simetría equiespaciados** (X-Γ, Γ-M, M-X del
  mismo ancho), como las figuras del artículo (en `plot_bands.py`).

## Parámetros físicos (en `bandcalc.build_red`)

```python
dens = [1150, 1250]      # densidades [kg/m^3]  (inclusión no se usa: cavidad)
vel0 = [295, 295]        # [C_l, C_t] matriz [m/s]   (Ct0 = 295 normaliza el eje)
vels = [894, 894]        # [C_l, C_t] inclusión
cut  = 2                 # modos m ∈ {-2..2}  (como en la tesis)
cond_borde = 'hollow'    # cavidad recubierta + pre-deformación angular psi
r1 = 0.45, r2 = 0.5, a = 1.0
```

> Nota: las **bandas planas** ~1.0–1.3 son resonancias localizadas de "fuga"
> (ω compleja); por eso su `|Im(μ)|` es mayor y el corte `IMTOL` controla cuánto
> se muestran.
