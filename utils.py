import numpy as np
from matplotlib import pyplot as plt
from scipy.special import jn, yv
from scipy.special import hankel1 as hn
from numpy.linalg import inv, pinv, norm, det
from math import factorial as sfactorial

def A1l(m, k, r, lame, shear):
    kr = k * r
    v1 = -m * jn(m, kr)
    v2 = kr * jn(m - 1, kr)
    val = (v1 + v2) / r
    return val

def A2l(m, k, r, lame, shear):
    kr = k * r
    val = (1j * m * jn(m, kr)) / r
    return val

def A3l(m, k, r, lame, shear):
    v1 = (2*shear*m*(m + 1)-(r**2)*(k**2)*(lame + 2*shear))*jn(m, r*k)
    v2 = -2*shear*k*r*jn(m-1, r*k)

    val = (v1+v2) / (r**2)
    return val

def A4l(m, k, r, lame, shear):
    kr = k * r
    v1 = -(m + 1) * jn(m, kr)
    v2 = kr * jn(m - 1, kr)
    val = (2j * m * shear) * (v1 + v2) / (r ** 2)
    return val

def A1t(m, k, r, lame, shear):
    kr = k * r
    val = (1j * m * jn(m, kr)) / r
    return val

def A2t(m, k, r, lame, shear):
    kr = k * r
    v1 = -kr * jn(m - 1, kr)
    v2 = m * jn(m, kr)
    val = (v1 + v2) / r
    return val

def A3t(m, k, r, lame, shear):
    kr = k * r
    v1 = -(m + 1) * jn(m, kr)
    v2 = kr * jn(m - 1, kr)
    val = (2j * m * shear) * (v1 + v2) / (r ** 2)
    return val

def A4t(m, k, r, lame, shear):
    kr = k * r
    v1 = (-2 * m * (m + 1) + (k ** 2) * (r ** 2)) * jn(m, kr)
    v2 = 2 * kr * jn(m - 1, kr)
    val = shear * (v1 + v2) / (r ** 2)
    return val

def B1l(m, k, r, lame, shear):
    kr = k * r
    v1 = -m * hn(m, kr)
    v2 = kr * hn(m - 1, kr)
    val = (v1 + v2) / r
    return val

def B2l(m, k, r, lame, shear):
    kr = k * r
    val = (1j * m * hn(m, kr)) / r
    return val

def B3l(m, k, r, lame, shear):
    kr = k * r
    v1 = (2 * shear * m * (m + 1) - (kr ** 2) * (lame + 2 * shear)) * hn(m, kr)
    v2 = -2 * shear * k * r * hn(m - 1, kr)
    val = (v1 + v2) / (r ** 2)
    return val

def B4l(m, k, r, lame, shear):
    kr = k * r
    v1 = -(m + 1) * hn(m, kr)
    v2 = kr * hn(m - 1, kr)
    val = (2j * m * shear) * (v1 + v2) / (r ** 2)
    return val

def B1t(m, k, r, lame, shear):
    kr = k * r
    val = (1j * m * hn(m, kr)) / r
    return val

def B2t(m, k, r, lame, shear):
    kr = k * r
    v1 = -kr * hn(m - 1, kr)
    v2 = m * hn(m, kr)
    val = (v1 + v2) / r
    return val

def B3t(m, k, r, lame, shear):
    kr = k * r
    v1 = -(m + 1) * hn(m, kr)
    v2 = kr * hn(m - 1, kr)
    val = (2j * m * shear) * (v1 + v2) / (r ** 2)
    return val

def B4t(m, k, r, lame, shear):
    kr = k * r
    v1 = (-2 * m * (m + 1) + kr ** 2) * hn(m, kr)
    v2 = 2 * kr * hn(m - 1, kr)
    val = shear * (v1 + v2) / (r ** 2)
    return val

def C1l(m, k, r, lame, shear):
    kr = k * r
    v1 = -m * jn(m, kr)
    v2 = kr * jn(m - 1, kr)
    val = (v1 + v2) / r
    return val

def C2l(m, k, r, lame, shear):
    kr = k * r
    val = (1j * m * jn(m, kr)) / r
    return val

def C3l(m, k, r, lame, shear):
    kr = k * r
    v1 = (2 * shear * m * (m + 1) - (kr ** 2) * (lame + 2 * shear)) * jn(m, kr)
    v2 = -2 * shear * k * r * jn(m - 1, kr)
    val = (v1 + v2) / (r ** 2)
    return val

def C4l(m, k, r, lame, shear):
    kr = k * r
    v1 = -(m + 1) * jn(m, kr)
    v2 = kr * jn(m - 1, kr)
    val = (2j * m * shear) * (v1 + v2) / (r ** 2)
    return val

def C1t(m, k, r, lame, shear):
    kr = k * r
    val = (1j * m * jn(m, kr)) / r
    return val

def C2t(m, k, r, lame, shear):
    kr = k * r
    v1 = -kr * jn(m - 1, kr)
    v2 = m * jn(m, kr)
    val = (v1 + v2) / r
    return val

def C3t(m, k, r, lame, shear):
    kr = k * r
    v1 = -(m + 1) * jn(m, kr)
    v2 = kr * jn(m - 1, kr)
    val = (2j * m * shear) * (v1 + v2) / (r ** 2)
    return val

def C4t(m, k, r, lame, shear):
    kr = k * r
    v1 = (-2 * m * (m + 1) + kr ** 2) * jn(m, kr)
    v2 = 2 * kr * jn(m - 1, kr)
    val = shear * (v1 + v2) / (r ** 2)
    return val


