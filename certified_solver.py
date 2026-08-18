"""Drop-in validation solver using the certified lattice sum.

CertifiedRed inherits the production root finder unchanged.  Only the G0 layer
is replaced.  This lets us ask whether zeros_longitudinal_fullgrid behaves
better once the secular function itself is numerically well-defined.
"""
from __future__ import annotations

from Bandas_Tools import Red
import g0_certified as g0c


class CertifiedRed(Red):
    def G0(self, f, k, pol, cut, n_suma=None):
        if n_suma is None:
            n_suma = self.n_suma
        return g0c.G0_matrix(self, f, k, pol, cut, int(n_suma))

    def G0_convergente(self, f, k, pol, cut, n_suma_ini=None, tol=1e-6,
                       n_suma_max=200, paso=1, norma="max", verbose=False):
        # 'paso' is accepted for API compatibility.  The certified reference
        # intentionally checks every full truncation n to avoid skipped shells.
        mat, info = g0c.G0_converged(
            self, f, k, pol, cut,
            n_suma_ini=n_suma_ini,
            tol=tol,
            n_suma_max=n_suma_max,
            stable_passes=1,
            norm=norma,
        )
        if verbose:
            print("[G0-certified]", info)
        return mat, info

    def G0_convergente_cached(self, f, k, pol, cut, n_suma_ini=None, tol=1e-6,
                              n_suma_max=200, paso=1, norma="max", verbose=False,
                              use_result_cache=False, stable_passes=2):
        # Correctness reference first; cache can be added after validation.
        mat, info = g0c.G0_converged(
            self, f, k, pol, cut,
            n_suma_ini=n_suma_ini,
            tol=tol,
            n_suma_max=n_suma_max,
            stable_passes=stable_passes,
            norm=norma,
        )
        if verbose:
            print("[G0-certified]", info)
        return mat, info
