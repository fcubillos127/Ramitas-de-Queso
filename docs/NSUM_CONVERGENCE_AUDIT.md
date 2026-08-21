# Reciprocal-truncation convergence audit

This document records the `n_suma` convergence study performed on the certified
MST reference solver.  It is a numerical-audit baseline, not yet a production
parameter recommendation for every lattice, deformation, or frequency range.

## Frozen configuration

The benchmark isolates reciprocal-lattice truncation while keeping the
remaining model fixed:

- lattice: square (`sq`)
- deformation: `psi = 0`
- boundary/scatterer model: `hollow`
- geometry: `r1/a = 0.45`, `r2/a = 0.50`
- angular cutoff: `cut = 4`
- transverse reference speed: `Ct0 = 295`
- normalized frequency range: `0.001 <= omega*a/(2*pi*Ct0) <= 1.25`
- certified fixed-k root finder with dual singular residual
- initial discovery grid: 120 normalized-frequency samples

The reciprocal truncations were

`n_suma = 8, 12, 20, 30, 40, 60`.

A refinement step was accepted only when the two consecutive local spectra had

1. a complete one-to-one root matching,
2. unchanged geometric multiplicities,
3. maximum normalized-frequency drift below
   `max(5e-5, 5e-5*abs(omega_norm))`, and
4. dual singular residual `R <= 1e-6` for every root in both spectra.

Two consecutive accepted refinement steps were required.  An early apparent
stable regime is invalidated if a later refinement fails.

## Seven-point benchmark

| `k/pi` | recommended `n_suma` | final roots at `n=60` | final mode count |
|---:|---:|---:|---:|
| 0.0 | 20 | 2 | 3 |
| 0.5 | 20 | 4 | 4 |
| 1.0 (Gamma) | 20 | 3 | 4 |
| 1.5 | 40 | 4 | 4 |
| 2.0 | 40 | 4 | 4 |
| 2.5 | 20 | 4 | 4 |
| 3.0 | 20 | 2 | 3 |

The mode count exceeds the distinct-root count at the endpoints and at Gamma
because a certified root has geometric multiplicity two.

### Stable degeneracies

At `k/pi = 0` and `3`, the double root converges to approximately

`omega*a/(2*pi*Ct0) = 0.709996794` at `n_suma=60`.

At Gamma (`k/pi = 1`), the double root converges to approximately

`omega*a/(2*pi*Ct0) = 1.064583490`.

The multiplicity remains two throughout the tested truncation sequence.  Thus
these degeneracies are not produced by a post-processing duplication rule.

## Slow-convergence region

The strongest reciprocal-truncation sensitivity in this sample occurs around
`k/pi = 1.5--2.0`, particularly for roots close to normalized frequency one.

At `k/pi = 1.5`, one root evolves as

| `n_suma` | normalized frequency |
|---:|---:|
| 8  | 1.011912723 |
| 12 | 1.011506870 |
| 20 | 1.011381116 |
| 30 | 1.011356408 |
| 40 | 1.011350272 |
| 60 | 1.011347073 |

The maximum spectral drift over the entire local root set is

- `8 -> 12`: `4.059e-4` (fail)
- `12 -> 20`: `1.258e-4` (fail)
- `20 -> 30`: `2.471e-5` (pass)
- `30 -> 40`: `6.135e-6` (pass)
- `40 -> 60`: `3.200e-6` (pass)

At `k/pi = 2.0`, the corresponding maximum drifts are

- `8 -> 12`: `1.425e-4` (fail)
- `12 -> 20`: `5.203e-5` (fail)
- `20 -> 30`: `1.020e-5` (pass)
- `30 -> 40`: `2.594e-6` (pass)
- `40 -> 60`: `1.388e-6` (pass)

This demonstrates that a small global truncation such as `n_suma=5` cannot be
justified as a generally converged setting for this spectral interval.

## Adaptive policy

`adaptive_nsum.py` implements a conservative stopping policy.  With the current
reference defaults:

- an increasing schedule such as `8, 12, 20, 30, 40, 60, 80` is evaluated;
- two consecutive accepted refinement steps establish a candidate minimum
  truncation;
- one additional accepted refinement step is required as confirmation;
- any failed refinement resets the candidate.

Two values are reported deliberately:

- `recommended_n_suma`: the minimum truncation at which the final stable regime
  was established;
- `certification_n_suma`: the larger truncation actually evaluated to confirm
  that regime.

The frequencies returned for scientific use are those at the certification
truncation, not the lower recommended truncation.

## Current conclusion and scope

For the seven representative points above, every local spectrum is converged by
`n_suma=40` under the stated tolerance, and a subsequent `n_suma=60` calculation
confirms the two most sensitive points.  This does **not** yet prove that 40 is
sufficient for every k along the path, for `psi != 0`, for a triangular lattice,
or outside the tested frequency window.  A denser adaptive k scan is the next
validation step before reciprocal-truncation convergence is considered closed.
