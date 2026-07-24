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

- Eje de frecuencia normalizado: **`ωa/2πC_t0`**. Camino de alta simetría:
  `X-Γ-M-X` (cuadrada) / `Γ-M-K-Γ` (triangular). `plot_bands.py` los dibuja con
  **tramos equiespaciados** (mismo ancho cada segmento), como las figuras del
  artículo — no distancia geométrica real en k.
- `cond_borde='rigid'` es un nombre engañoso: internamente selecciona
  `coeficiente_dispersion_elastic` (inclusión con material propio). **Cualquier
  otro valor** (p. ej. `'hollow'`, que es lo que usan `scripts_figs`) selecciona
  `coeficiente_dispersion_hollow` (cavidad + recubrimiento, sin segundo material).
- `coeficiente_dispersion_hollow` **no depende de materiales**: `mu0` se asigna
  pero nunca se usa en el cuerpo de la función. Para ese modelo la física depende
  solo de `ψ`, `r1/a`, `r2/a` — no de densidad ni velocidades.
- La tesis usa `cut=2` (`m ∈ {−2..2}`) para las figuras de bandas publicadas.
  `cut` más alto (p. ej. 7) es válido pero revela más resonancias planas — no
  esperes que se vea igual a la figura publicada sin igualar `cut`.
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
