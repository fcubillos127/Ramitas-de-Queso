from __future__ import annotations
from typing import Iterable, Literal
import numpy as np
from scipy.integrate import quad
from scipy.special import jv, jvp, hankel1, h1vp

Kind = Literal["F", "V"]
Chan = Literal["b", "c"]

def _k0_from_red(red, freq_norm: float) -> complex:
    w = (freq_norm * (2 * np.pi * red.vel0[1]), 0.0001)
    return red.k0(w, 1)

def _integrand_Fb(n: int, k0: complex, r: float) -> complex:
    k0r = k0 * r
    return jv(n, k0r) * (k0r * jvp(n, k0r) - jv(n, k0r)) / (r**3)

def _integrand_Fc(n: int, k0: complex, r: float) -> complex:
    k0r = k0 * r
    return jv(n, k0r) * (k0r * h1vp(n, k0r) - hankel1(n, k0r)) / (r**3)

def _integrand_Vb(n: int, k0: complex, r: float) -> complex:
    k0r = k0 * r
    return hankel1(-n, k0r) * (k0r * jvp(n, k0r) - jv(n, k0r)) / (r**3)

def _integrand_Vc(n: int, k0: complex, r: float) -> complex:
    k0r = k0 * r
    return hankel1(-n, k0r) * (k0r * h1vp(n, k0r) - hankel1(n, k0r)) / (r**3)

from scipy.integrate import quad

def compute_Fb(red, n: int, freq_norm: float, r1: float, r: np.ndarray,
               epsabs: float = 1e-9, epsrel: float = 1e-9, limit: int = 200) -> np.ndarray:
    """
    Calcula F_b(n, freq_norm, r) usando integración con tolerancias ajustadas.
    """
    k0 = _k0_from_red(red, freq_norm)
    out = np.empty_like(r, dtype=complex)
    for i, ri in enumerate(r):
        real_part = quad(lambda x: np.real(_integrand_Fb(n, k0, x)),
                         r1, ri, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        imag_part = quad(lambda x: np.imag(_integrand_Fb(n, k0, x)),
                         r1, ri, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        out[i] = real_part + 1j * imag_part
    return out

def compute_Fc(red, n: int, freq_norm: float, r1: float, r: np.ndarray,
               epsabs: float = 1e-9, epsrel: float = 1e-9, limit: int = 200) -> np.ndarray:
    """
    Calcula F_c(n, freq_norm, r) con tolerancias ajustadas.
    """
    k0 = _k0_from_red(red, freq_norm)
    out = np.empty_like(r, dtype=complex)
    for i, ri in enumerate(r):
        real_part = quad(lambda x: np.real(_integrand_Fc(n, k0, x)),
                         r1, ri, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        imag_part = quad(lambda x: np.imag(_integrand_Fc(n, k0, x)),
                         r1, ri, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        out[i] = real_part + 1j * imag_part
    return out

def compute_Vb(red, n: int, freq_norm: float, r2: float, r: np.ndarray,
               epsabs: float = 1e-9, epsrel: float = 1e-9, limit: int = 200) -> np.ndarray:
    """
    Calcula V_b(n, freq_norm, r) con tolerancias ajustadas.
    """
    k0 = _k0_from_red(red, freq_norm)
    out = np.empty_like(r, dtype=complex)
    for i, ri in enumerate(r):
        real_part = quad(lambda x: np.real(_integrand_Vb(n, k0, x)),
                         ri, r2, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        imag_part = quad(lambda x: np.imag(_integrand_Vb(n, k0, x)),
                         ri, r2, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        out[i] = real_part + 1j * imag_part
    return out

def compute_Vc(red, n: int, freq_norm: float, r2: float, r: np.ndarray,
               epsabs: float = 1e-9, epsrel: float = 1e-9, limit: int = 200) -> np.ndarray:
    """
    Calcula V_c(n, freq_norm, r) con tolerancias ajustadas.
    """
    k0 = _k0_from_red(red, freq_norm)
    out = np.empty_like(r, dtype=complex)
    for i, ri in enumerate(r):
        real_part = quad(lambda x: np.real(_integrand_Vc(n, k0, x)),
                         ri, r2, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        imag_part = quad(lambda x: np.imag(_integrand_Vc(n, k0, x)),
                         ri, r2, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        out[i] = real_part + 1j * imag_part
    return out


def compute_series(
    red,
    kind: Kind,
    chan: Chan,
    n: int,
    freq_norm: float,
    r1: float,
    r2: float,
    r: np.ndarray,
) -> np.ndarray:
    if kind == "F" and chan == "b":
        return compute_Fb(red, n, freq_norm, r1, r)
    if kind == "F" and chan == "c":
        return compute_Fc(red, n, freq_norm, r1, r)
    if kind == "V" and chan == "b":
        return compute_Vb(red, n, freq_norm, r2, r)
    if kind == "V" and chan == "c":
        return compute_Vc(red, n, freq_norm, r2, r)
    raise ValueError("Par (kind, chan) inválido")
