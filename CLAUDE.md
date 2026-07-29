# Ramitas-de-Queso — bandas fonónicas con pre-deformación

Cálculo de **estructura de bandas elástica/acústica** para una red periódica 2D
(cuadrada `sq` o hexagonal/triangular `hx`) de cavidades cilíndricas con
recubrimiento pre-deformado (parámetro `ψ`), vía **teoría de scattering
múltiple** (tipo KKR): `det(T·G0 − I) = 0`.

Basado en la tesis de magíster de Miguel Letelier Villegas, *"Metamateriales
Elásticos No-Lineales"* (U. de Chile, prof. guía Claudio Falcón). **La tesis y
el artículo asociado (Anexo 1, Elsevier) no están en el repo** — los subió el
usuario como archivos sueltos. Si necesitas contrastar una fórmula o figura
contra la teoría, pídeselos; no asumas que existen en el árbol de archivos.

## Estructura

```
Bandas_Tools.py      # Clase Red: núcleo original de Miguel (G0, T-matrix, determinante,
                      #   solver por fsolve, herramientas de edición manual de bandas)
suma_de_red.py        # Sumas de red (structure constants), K/Kh/S_pre/precompute_Qh
utils.py               # Coeficientes A/B/C de condiciones de borde (poco usado hoy)
project_io.py           # Helpers de rutas data/ y graphs/
fnv/                     # Paquete F_n/V_n (integrales del recubrimiento)
main.py                  # Script de ejemplo original de Miguel (cut=6, no necesariamente
                          #   los parámetros de las figuras publicadas)

scripts_figs/             # Pipeline NUEVO (mío), independiente del solver original
  bandcalc.py                #   build_red(), compute_bands_eig() — método por autovalores
  compute_driver.py          #   run(): barrido multi-psi, guarda .npz
  explore_one.py              #   compute_one()/plot_one()/compare() — UNA estructura a la vez
  plot_bands.py                #   figuras con segmentos de alta simetría equiespaciados
  bridge_to_omega.py            #   puente hacia las herramientas de edición manual de Red
  gap_vs_psi.py                  #   gap en M (cuadrada) / corrimiento de Dirac en K (triangular)
  compute_mode.py                #   reconstrucción del campo u_z desde el vector nulo

explorar_estructura.py    # Punto de entrada VSCode: UNA estructura, tantear parámetros
ejemplo_completo.py       # Punto de entrada VSCode: draft con varios psi, extremo a extremo
```

## Dos pipelines — no mezclar sin saberlo

Hay **dos formas independientes** de calcular bandas en este repo:

1. **Solver original de Miguel**: `Red.zeros_longitudinal_fullgrid(...)` — búsqueda
   por ventanas + `fsolve`. Usa `imag_tol` y `sol_tol`. Guarda en
   `self.omega_longitudinal`, que es lo que consumen sus herramientas de edición
   manual (`delete_point`, `restore_deleted`, `order_bands_by_continuity_global`,
   `smooth_interpolate_longitudinal`, `graficar_bandas_grid`).
2. **Método por autovalores (mío)**: `bandcalc.compute_bands_eig(...)` — una banda
   es donde algún autovalor `μᵢ` de `T·G0` cruza `Re(μᵢ)=1`. Más robusto que buscar
   mínimos de `|det|` (que es el *producto* de factores: borra ramas cercanas y
   pierde soluciones). **No usa `fsolve`**, así que `imag_tol`/`sol_tol` **no lo
   afectan**. Guarda `|Im(μ)|` por punto para filtrar "fuga" en el graficado sin
   recalcular (`plot_bands.IMTOL` / `explore_one.plot_one(..., imtol=...)`).

`bridge_to_omega.eig_to_red()` conecta ambos: carga resultados del método 2 dentro
de `omega_longitudinal` para poder editarlos a mano con las herramientas del método 1.

## Convenciones físicas / gotchas de nomenclatura

- Eje de frecuencia normalizado: **`ωa/2πC_t0`**. Camino de alta simetría (el
  orden **natural** del `k` escalar en `suma_de_red.K`, verificado):
  **`M-Γ-X-M`** (cuadrada: k=0→M, π/a→Γ, 2π/a→X, 3π/a→M) /
  **`M-K-Γ-M`** (triangular). `plot_bands.py` los dibuja con **tramos
  equiespaciados** (mismo ancho cada segmento), como las figuras del artículo —
  no distancia geométrica real en k.
- ⚠️ **`graficar_bandas_grid` de Miguel rotula `X-Γ-M-X`** (cuadrada), pero el
  `k` escalar recorre `M-Γ-X-M`: **las etiquetas X↔M están intercambiadas** en
  su versión. La *forma* de las curvas es correcta (grafica ω vs k directo). Una
  versión previa de `plot_bands.path_order` **invertía cada tramo** para forzar
  el rótulo X-Γ-M-X, y eso **voltea el signo de la velocidad de grupo** (la
  acústica en X-M salía con pendiente negativa). Corregido: orden natural
  M-Γ-X-M, sin invertir. Lo detectó el usuario por la dispersión de la acústica.
- `cond_borde='rigid'` es un nombre engañoso: internamente selecciona
  `coeficiente_dispersion_elastic` (inclusión con material propio). **Cualquier
  otro valor** (p. ej. `'hollow'`, que es lo que usan `scripts_figs`) selecciona
  `coeficiente_dispersion_hollow` (cavidad + recubrimiento, sin segundo material).
- `coeficiente_dispersion_hollow` **no depende de materiales**: `mu0` se asigna
  pero nunca se usa en el cuerpo de la función. Para ese modelo la física depende
  solo de `ψ`, `r1/a`, `r2/a` — no de densidad ni velocidades.
- ⚠️ **`cut=6` es el valor que reproduce las Figs. 3–4** (`m ∈ {−6..6}`), no
  `cut=2`. Es lo que dice el `main.py` de Miguel y está **verificado**: con
  `cut=6` el espectro calza con la Fig. 3 en todos los puntos de alta simetría
  (X: 0.416/0.702/0.708/1.324 vs sus ~0.42/~0.71/~1.33; Γ: 0.848/1.027/1.218 vs
  sus ~0.85/~1.02/~1.21; M: 0.263/0.709/1.029 vs sus ~0.27/~0.71/~1.02).
  **La banda plana de ~1.02 —la protagonista de la Fig. 4— NO EXISTE con
  `cut=2`**: con cut=2 en Γ salen 0.851 y 1.180, sin nada en 1.02. Una versión
  previa de este archivo afirmaba "la tesis usa cut=2"; era falso y provocó una
  investigación larga en falso (se llegó a sospechar del solver, de
  `cond_borde` y del fix de la parte imaginaria). Si algo no calza con las
  figuras publicadas, **lo primero que hay que revisar es `cut`**.
- Las bandas "planas" cerca de `ωa/2πC_t0 ≈ 1.0–1.3` son **resonancias de fuga**
  (frecuencia compleja). Es normal que salgan más ruidosas/dispersas que las
  bandas propagantes — no es necesariamente un bug.
- **Las figuras publicadas de Miguel están curadas a mano** (por eso existen
  `delete_point`/`restore_deleted`/`order_bands_by_continuity_global`/
  `smooth_interpolate_longitudinal` con *undo*). Ningún cálculo crudo —ni el suyo
  ni el mío— sale idéntico a una figura de paper sin ese paso de limpieza.

## ⚠️ Gotcha verificado: `order_bands_by_continuity_global()` no hace lo que parece

`Red.order_bands_by_continuity_global()` **no modifica `self.omega_longitudinal`**:
escribe el resultado en un atributo aparte (`self.omega_longitudinal_ordered`)
y lo retorna — hay que capturarlo y asignarlo explícitamente para que tenga
efecto. Peor: con sus parámetros por defecto (`delta_max_norm=0.18`, etc.),
verificado sobre datos reales que puede **vaciar todos los puntos** (un caso
probado: 67 finitos → 0, `"assigned=0/160"` en su propio log). No se depuró
más a fondo por ser un subsistema aparte (fuera de alcance salvo que se pida).
Para limpieza automática de puntos espurios del solver original de Miguel, usar
`scripts_figs/postprocess_miguel.py` (detector propio, verificado, no usa esta
función) en vez de asumir que `order_bands_by_continuity_global` sirve tal cual.

## Post-proceso del solver de Miguel: `postprocess_miguel.post_process`

Verificado contra el pantallazo marcado del usuario (verde = quitar, naranja =
agregar). Hace las dos cosas que Miguel curaba a mano:
- **QUITA fantasmas de red vacía**: el determinante tiene *polos* de la suma de
  red sobre `ω = C_t0·|k+G|`; el solver mete "raíces" pegadas a ellos que
  siguen esas curvas pero no son bandas físicas (las cadenas en "X"). Se
  detectan por **tubo + persistencia** (`detectar_fantasmas`), con protección
  `el_floor` cerca de Γ donde `G=0` y la acústica convergen.
- **AGREGA lo que el barrido se saltó** (bandas planas ~1.1–1.4 y tramos de la
  acústica con ψ grande): `completar_bandas_eig` rastrea autovalores `μᵢ` de
  `T·G0` en grilla fina **troceada** (los polos descalabran el rastreo de
  ramas; trocear confina el daño), toma cruces `Re(μᵢ)=1`, los refina con
  `fsolve` sobre el **mismo `Det_longitudinal`** del solver, descarta los que
  caen sobre red vacía. No fabrica: todo insertado resuelve `det=0`. Es lento.
Backup completo en `red._omega_backup_postprocess` (deshacer todo). `+0 post`
= ningún fantasma renació entre los insertados. NO usa
`smooth_interpolate_longitudinal` ni `order_bands_by_continuity_global`.

## ⚠️ Gotcha verificado: `delete_point` puede vaciar TODO el tensor

`Red.delete_point(i, n, mode="fullgrid")` llama a `_ensure_tensor(nk,
self.nbands)`, que **reemplaza `omega_longitudinal` por NaN entero** si
`self.nbands ≠ omega_longitudinal.shape[1]`. Pasa al cargar un
`omega_longitudinal` externo (p. ej. `.npy` crudo del solver, o desde
`bridge_to_omega`) sobre un `Red` armado con otro `nbands`. Único síntoma:
avisos `"ya es NaN en RAM"` y datos vacíos. **Sincronizar `red.nbands =
red.omega_longitudinal.shape[1]` antes de borrar** (ya lo hace `post_process`).

## ⚠️ Gotcha activo: `CT0` fijo, desacoplado de `r.vel0`

`bandcalc.py` define `CT0 = 295.0` como constante de módulo y la usa para
construir/normalizar la malla de `ω` en `compute_bands_eig` — **sin mirar
`r.vel0`**. Si alguien cambia `vel0`/`dens`/`vels` en `build_red()` sin arreglar
antes este acoplamiento, los resultados quedan **mal escalados en silencio**.
Por eso `explore_one.compute_one()` **no expone materiales** todavía. Si el
usuario pide variar materiales: arreglar primero `compute_bands_eig` (y
`compute_driver.run`, que escribe `"Ct0": CT0` en el `.npz`) para usar
`float(r.vel0[1])` en vez de la constante del módulo, antes de exponer el kwarg.

## Hábito de verificación (úsalo para cualquier cambio numérico)

Antes de optimizar o refactorizar código de cálculo: generar una **línea base
dorada** (correr con el código actual, guardar resultados en `.npz`) y después
del cambio verificar `max|nuevo − base| == 0.0` (o dentro de tolerancia
explícita). Así se verificaron todas las optimizaciones de este repo (G0
Toeplitz + caché, cortocircuito `ALPHA==0`, caché de `T_n`): resultados
bit-idénticos, solo más rápido. No confíes en "se ve bien" para código de física.

## Git

- Rama de trabajo: `claude/code-review-optimization-49vxac`.
- Los commits deben quedar como *Verified* en GitHub: `git config user.email
  noreply@anthropic.com && git config user.name Claude` antes de commitear (hay
  un stop-hook que lo chequea).
- Nunca reescribir commits que no sean míos (p. ej. si el usuario borra archivos
  desde la web de GitHub) — esos quedan "Unverified" con su propia identidad y
  está bien así.

## Estado pendiente / decisiones del usuario

- Hay ~44 métodos en `Bandas_Tools.py` sin referencia interna, agrupados por
  subsistema (bandas 2D/plano, solver "mejorado" alternativo, post-proceso de
  bandas, `coeficiente_dispersion_rigid`+`G0_cached` sueltos). **No se han
  borrado** — el usuario dijo "dejémoslo así" cuando se le preguntó. No asumas
  que están muertos sin volver a preguntar.
- El bug de `real_func`/`imag_func` intercambiados en `coeficiente_dispersion_rigid`
  ya está corregido, pero esa función no la llama el camino del determinante
  (`cond_borde='rigid'` usa `_elastic`, no `_rigid`) — el fix no cambió ninguna
  salida existente.
