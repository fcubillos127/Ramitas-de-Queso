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

# graficar (full 0-1.4 y zoom 0.7-1.2), estilo tesis, camino M-Γ-X-M / M-K-Γ-M
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
post_process(red)   # QUITA fantasmas + AGREGA lo que falta, in-place, y re-grafica
# deshacer TODO:  red.omega_longitudinal = red._omega_backup_postprocess.copy()
```

Reproduce las dos cosas que Miguel hacía a mano (verificado contra su
pantallazo marcado: verde = quitar, naranja = agregar), en orden:

1. **QUITAR fantasmas de red vacía** (las cadenas en "X" diagonales). El
   determinante `det(T·G0 − I)` tiene **polos** de la suma de red sobre las
   bandas de red vacía `ω = C_t0·|k+G|`; el buscador de cambios de signo +
   `fsolve` del solver encuentra "raíces" pegadas a esos polos que **siguen
   exactamente** las curvas `|k+G|` pero no son bandas físicas del cristal
   (verificado: 30% de los puntos con ψ=0 y 49% con ψ=0.8 caen a <0.005 de
   una curva de red vacía). Se detectan por **tubo + persistencia**: dentro de
   `el_tol` de una curva `|k+G|` **y** con esa misma curva poblada en varias
   columnas de `k` vecinas (un cruce accidental de una banda real la toca en
   1-2 columnas; el fantasma la sigue en muchas). Cerca de Γ la curva `G=0` y
   la banda acústica real convergen, así que ahí (`ω<el_floor`) no se marca.
2. **QUITAR aislados** por enlace de paso (detector previo, para puntos
   totalmente sueltos que no siguen ninguna curva).
3. **AGREGAR lo que falta** (los "vacíos" de las bandas planas ~1.1–1.4 y, con
   ψ grande, tramos de la acústica). El barrido del solver muestrea `Re(det)`
   en pocos puntos por unidad de `ω` y **se salta** los cambios de signo
   angostísimos de las resonancias planas. Aquí se rastrean los autovalores
   `μᵢ` de `T·G0` en una grilla fina (**troceada** para que los polos no
   descalabren el rastreo de ramas) y se toman los cruces `Re(μᵢ)=1`; cada
   cruce se **refina con `fsolve` sobre el mismo `Det_longitudinal`** del
   solver (mismos `sol_tol`, `imag_tol`, `epsfcn`) y se inserta. **No se
   fabrica nada**: todo lo insertado es solución calculada de `det=0`. Los
   candidatos que caen sobre una curva de red vacía se descartan (no se
   re-siembran fantasmas).
4. Reordena cada columna de `k` por frecuencia y rellena huecos de ≤1 paso.
5. Re-grafica con `red.graficar_bandas_grid()`.

Verificado sobre datos reales (`nk=50, cut=2`): ψ=0.0 279→278 pts (90
fantasmas + 27 aislados fuera, 109 insertados) y ψ=0.8 177→292 pts (79+25
fuera, 213 insertados) — las cadenas en X desaparecen y las bandas planas
quedan continuas, sin renacer fantasmas entre los insertados (`+0 post` en
ambos). El paso 3 es **lento** (recalcula `T·G0` en grilla fina, ~40-110 s por
ψ con estos parámetros): desactívalo con `completar=False` o acótalo con
`windows=[(w_lo, w_hi)]` si solo quieres limpiar.

⚠️ **Historial de tres iteraciones** (cada una corrigió un error real):
- Detector por "ventana absoluta de frecuencia": borraba hasta 47% incl. la
  banda acústica (pendiente real) → **enlace de paso**.
- `red.smooth_interpolate_longitudinal()` (sin límite de hueco) fabricó
  decenas de puntos a través de huecos de 8-10 pasos (279→330); y aun limitado
  a 1 paso, el solver ordena por frecuencia por-`k` sin rastrear ramas, así que
  un hueco puede unir dos ramas distintas (caso real: 0.05→0.98) → relleno
  propio con `max_gap` **y** flancos parecidos.
- El enlace de paso **no bastaba** (feedback con pantallazo): lo que hay que
  quitar son **cadenas conectadas** (pasan el test de enlace) y lo que falta
  **no se puede interpolar** (no hay puntos). De ahí los pasos 1 y 3, basados
  en la física: polos de red vacía (quitar) y cruces de autovalores
  recalculados (agregar).

⚠️ **Gotcha de `delete_point`**: llama a `_ensure_tensor(nk, red.nbands)` y, si
`red.nbands ≠ omega_longitudinal.shape[1]`, **reemplaza el tensor entero por
NaN** (pérdida total silenciosa). `post_process` sincroniza `red.nbands` antes
de tocar nada; si llamas los pasos por separado, hazlo tú también.

⚠️ **No usa `red.order_bands_by_continuity_global()`**: verificado que (a) no
modifica `self.omega_longitudinal` (escribe en un atributo aparte,
`self.omega_longitudinal_ordered`) y (b) con sus parámetros por defecto puede
vaciar TODOS los puntos en datos reales (un caso probado: 67 finitos → 0). No
se investigó más por tocar un subsistema aparte fuera de este alcance.

⚠️ **Al graficar datos ya post-procesados, usa `clean=False`.** `plot_bands`
(y `graficar_bandas_grid` no, pero `plot_bands.panel`/`make_figures` sí) trae
su propio filtro `clean_isolated` (exige ≥2 vecinos en una caja), calibrado
para la salida CRUDA del método por autovalores. Sobre datos que ya pasaron
por `post_process`, ese filtro **vuelve a borrar** los puntos de banda plana
recién insertados (en la grilla equiespaciada quedan más separados y parecen
"aislados") — la figura se ve más pelada que la del post-proceso. Exporta con
`red_to_eig_npz` y grafica con `make_figures(..., clean=False)` (o
`panel(..., clean=False)`) para respetar el post-proceso. Con la salida cruda
del método por autovalores, deja `clean=True` (default).

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
- Los ejes salen con **tramos de alta simetría equiespaciados** (M-Γ, Γ-X, X-M del
  mismo ancho), como las figuras del artículo (en `plot_bands.py`). El camino es
  **M-Γ-X-M** (cuadrada) / **M-K-Γ-M** (triangular) — el orden **natural** que
  recorre el `k` escalar en `suma_de_red.K` (ver ⚠️ abajo), sin invertir tramos.

⚠️ **`graficar_bandas_grid` de Miguel rotula X-Γ-M-X / M-K-Γ-M**, pero el `k`
escalar recorre físicamente **M-Γ-X-M** (sq): las etiquetas X↔M de su versión
cuadrada están intercambiadas respecto de lo que calcula `K()`. La *forma* de
las curvas es correcta (grafica ω vs k directo); solo el rótulo de los extremos
está cambiado. `plot_bands.path_order` usa el rótulo correcto (M-Γ-X-M) y **no
invierte tramos** — una versión anterior sí los invertía y salía la dispersión
con el signo al revés (p. ej. la acústica en X-M con pendiente negativa en vez
de positiva).

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
