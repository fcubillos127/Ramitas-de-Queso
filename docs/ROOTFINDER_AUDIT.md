# Root-finder audit (certified G0 baseline)

This note records the numerical baseline used to audit `zeros_longitudinal_fullgrid` after replacing only the lattice-sum layer by `CertifiedRed`.

## Frozen physical/numerical configuration

- square lattice
- `psi = 0`
- hollow inclusion
- `r1/a = 0.45`, `r2/a = 0.50`
- `cut = 4`
- `n_suma = 12`
- `Ct0 = 295`
- normalized search interval: `0.001 <= omega*a/(2*pi*Ct0) <= 1.25`

The representative points are `k/pi = 0, 0.5, 1, 1.5, 2, 2.5, 3`.

## Isolation principle

Both methods below use the **same certified G0**.  Therefore differences isolate the root-finding layer rather than the lattice-sum implementation.

Historical method:

1. uniform frequency grid (`ventanas_por_unidad = 100`),
2. sign changes of `Re(det(TG0-I))`,
3. midpoint seed,
4. two-dimensional `fsolve` with `xtol = 1e-2`,
5. acceptance from `ier == 1` and imaginary-frequency tolerance.

Reference method:

1. retain sign-change intervals for simple roots,
2. add local minima of a dual singular residual,
3. refine on the real axis,
4. certify only if both raw and equilibrated smallest singular values are small,
5. infer local geometric multiplicity from the equilibrated nullspace.

The conservative residual is

`R = max(sigma_min(A), sigma_min(D_r A D_c))`, with `A = TG0-I`.

The diagonal matrices `D_r,D_c` only equilibrate row/column scales; they preserve exact matrix rank.

## Representative result

Using one-to-one frequency matching with tolerance `2e-3` in normalized frequency:

- historical candidates: **30**
- candidates matched to a certified root: **20**
- extra/spurious/duplicate historical candidates: **10**
- certified roots missed by the historical search: **3**

All three missed roots are two-dimensional nullspaces (double degeneracies):

- `k/pi = 0`: `omega*a/(2*pi*Ct0) ~= 0.709996`
- `k/pi = 1` (Gamma): `omega*a/(2*pi*Ct0) ~= 1.064587`
- `k/pi = 3`: `omega*a/(2*pi*Ct0) ~= 0.709996`

For the 20 matched roots, the historical frequency error has approximately

- median: `5.96e-6`
- maximum: `8.98e-5`

in normalized frequency.

## Pole/scaling discrimination

A balanced SVD alone is not a sufficient certificate: near an exact reciprocal-space pole the divergent term can become effectively low rank after equilibration. Conversely, the raw SVD alone can look singular in a strongly scaled low-frequency matrix.

The dual test separates these cases:

- true root: raw small **and** balanced small;
- reciprocal pole: balanced can be small, raw remains finite;
- low-frequency scaling artifact: raw can be small, balanced remains finite.

For example at `k/pi = 1.5`, the reciprocal pole at normalized frequency `0.75` has raw `sigma_min` of order `5e-2` even arbitrarily close to the pole, while the balanced value decreases with distance to the pole.  It is therefore rejected by the dual residual.

## Discovery-grid stability

For the same baseline, the certified root set at the seven representative k points was unchanged when the initial scan was varied between roughly 120 and 350 points over the full normalized interval.  Refinement then moves each accepted root to residuals typically `1e-8` or smaller.

This is a diagnostic baseline, not yet a claim that `n_suma=12` or the residual thresholds are universally converged for every `psi`, lattice, or frequency range.
