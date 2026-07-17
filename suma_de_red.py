import numpy as np
from scipy.special import jn
from scipy.special import hankel1 as hn
from numpy.linalg import norm

latt = 'hx'

def K(a, k, lattice = latt):
    """ Calcula el vector de Bloch para una red reciproca dada """
    if lattice == 'sq':
        if k > (3 * np.pi / a):
            raise ValueError('k fuera de la zona de Brillouin')
        if k <= (np.pi / a):
            return np.array([np.pi / a - k, np.pi / a - k])
        elif k <= (2 * np.pi / a):
            return np.array([k - np.pi / a, 0])
        else:
            return np.array([np.pi / a, k - 2 * np.pi / a])
    elif lattice == 'hx':
        if k > 2 * np.pi * (1 + 1 / np.sqrt(3)) / a:
            raise ValueError('k fuera de la zona de Brillouin')
        if k <= 2 * np.pi / (3 * a):
            return np.array([k / 2 + np.pi / a, (1 / np.sqrt(3)) * (3 * k / 2 - np.pi / a)])
        elif k <= 2 * np.pi / a:
            return np.array([4 * np.pi / (3 * a) - (k - 2 * np.pi / (3 * a)), 0])
        else:
            return np.array([(np.sqrt(3) / 2) * (k - 2 * np.pi / a), -0.5 * (k - 2 * np.pi / a)])

def Kh(a, k1, k2, lattice = latt):
    """ Entrega el vector de red reciproca que lleva desde la celda central a una celda
    que se encuentra k1 veces en la direccion del primer vector primitivo de red reciproca
    y k2 veces en la direccion del segundo vector primitivo de red reciproca """
    if lattice == 'sq':
        vec_1 = [2 * np.pi / a, 0];
        vec_2 = [0, 2 * np.pi / a];
    elif lattice == 'hx':
        vec_1 = [2*np.pi/a, -(1/np.sqrt(3))*(2*np.pi/a)]
        vec_2 = [2*np.pi/a, (1/np.sqrt(3))*(2*np.pi/a)]
    vec = k1*np.array(vec_1) + k2*np.array(vec_2);
    return np.array(vec)

def S1(M, m, k, n, a, k0_, lattice = latt):
    """ Entrega parte de la suma sobre los vectores de red reciproca necesario para el
    calculo de la suma de red """
    N0 = M - m
    N = abs(N0)
    S = 0
    Kvec = K(a, k, lattice)
    tol = 1e-12
    for i in range(-n, n + 1):
        for l in range(-n, n + 1):
            Qh = Kh(a, i, l, lattice) + Kvec
            Qh_ = norm(Qh)
            if Qh_ < tol:
                continue
            ang = np.angle(Qh[0] + 1j*Qh[1])
            num = jn(N + 1, Qh_*a)*np.exp(1j*N*ang)
            den = Qh_*(Qh_**2 - k0_**2)
            S += num/den
    area = a ** 2 if lattice == 'sq' else (np.sqrt(3) * a ** 2 / 2)
    valor = (S*k0_*4*(1j)**(N+1))/area
    return valor

def S(M, m, k, n, a, k0_, lattice = latt):
    """
    Entrega la suma de red del sistema para un k escalar.

    Esta versión mantiene la implementación original basada en llamadas a
    S1.  Se conserva para compatibilidad, aunque internamente se
    recomienda el uso de las funciones vectorizadas ``precompute_Qh`` y
    ``S_pre`` (véase más abajo) cuando se necesite optimizar el cálculo.
    """
    N0 = M - m
    N = abs(N0)
    krondelta = int(N == 0)
    term1 = ((2j + k0_*a*np.pi*hn(1, k0_*a))/(k0_*np.pi*a))*krondelta
    term2 = S1(M, m, k, n, a, k0_, lattice)
    Sval = -(term1 + term2)/jn(N + 1, k0_ * a)
    if N0 < 0:
        Sval = -np.conj(Sval)
    return Sval

#
#  Funciones vectorizadas para la suma de red
#
#  Para mejorar el rendimiento de las sumas de red se pueden
#  precomputar los vectores de red recíproca y trabajar con arrays
#  vectorizados en lugar de bucles de Python.  Las siguientes
#  funciones permiten reutilizar la misma malla de vectores
#  ``Qh_mod`` (módulo) y ``ang`` (ángulo) para distintas frecuencias
#  k0_ y diferentes valores de M, m.

def precompute_Qh(a, k_vec, n, lattice='hx'):
    """
    Precalcula los módulos y ángulos de los vectores de red recíproca
    desplazados por ``k_vec``.

    Parámetros
    ----------
    a : float
        Constante de red.
    k_vec : array_like, shape (2,)
        Vector bidimensional en el espacio recíproco (k_x, k_y).
    n : int
        Límite superior de la suma doble (se recorrerán i,l en [-n, n]).
    lattice : str
        Tipo de red ('sq' para cuadrada, 'hx' para hexagonal).

    Devuelve
    -------
    Qh_mod : ndarray, shape ((2n+1), (2n+1))
        Módulo de cada vector de red recíproca desplazado.
    ang : ndarray, shape ((2n+1), (2n+1))
        Ángulo del vector (argumento de la parte compleja).
    """
    # Generamos malla de índices
    i_vals = np.arange(-n, n+1)
    l_vals = np.arange(-n, n+1)
    I, L = np.meshgrid(i_vals, l_vals, indexing='ij')
    # Vectores de red recíproca
    # Usamos Kh vectorizado: multiplicamos enteros por los vectores base
    if lattice == 'sq':
        vec1 = np.array([2 * np.pi / a, 0])
        vec2 = np.array([0, 2 * np.pi / a])
    elif lattice == 'hx':
        vec1 = np.array([2 * np.pi / a, -(1/np.sqrt(3)) * (2 * np.pi / a)])
        vec2 = np.array([2 * np.pi / a, (1/np.sqrt(3)) * (2 * np.pi / a)])
    else:
        raise ValueError("Lattice no reconocida.")
    # Construimos todos los Qh como matrices de forma ((2n+1),(2n+1),2)
    Qh = I[..., None] * vec1 + L[..., None] * vec2
    Qh = Qh + k_vec  # desplazamiento
    # Módulo y ángulo de cada vector
    Qh_mod = np.linalg.norm(Qh, axis=2)
    ang = np.angle(Qh[..., 0] + 1j * Qh[..., 1])
    return Qh_mod, ang

def S1_pre(N, k0_, Qh_mod, ang, a, lattice='hx'):
    """
    Parte de la suma de red para un orden de diferencia N=|M-m|.
    Utiliza las matrices de módulos y ángulos precomputadas para
    vectorizar el cálculo.

    Parámetros
    ----------
    N : int
        Valor absoluto de M-m.
    k0_ : complex or float
        Módulo del vector de onda multiplicado por a.
    Qh_mod, ang : ndarray
        Salida de ``precompute_Qh`` (módulos y ángulos).
    a : float
        Constante de red.
    lattice : str
        Tipo de red ('sq' o 'hx').  Sólo afecta al factor de área.

    Devuelve
    -------
    complex
        Resultado de la suma ``S1`` vectorizada.
    """
    # Calculamos J_N+1(Qh_mod * a) de forma vectorizada
    bessel = jn(N + 1, Qh_mod * a)
    # Denominador: Qh * (Qh^2 - k0_^2)
    denom = Qh_mod * (Qh_mod**2 - k0_**2)
    # Prevenir divisiones por cero
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    # Numerador: jn(...) * exp(i N ang)
    num = bessel * np.exp(1j * N * ang)
    S_sum = np.sum(num / denom)
    area = a ** 2 if lattice == 'sq' else (np.sqrt(3) * a ** 2 / 2)
    return (S_sum * k0_ * 4 * (1j) ** (N + 1)) / area

def S_pre(M, m, k0_, Qh_mod, ang, a, lattice='hx'):
    """
    Suma de red total para un par (M,m) usando Qh precomputado.

    Parámetros
    ----------
    M, m : int
        Índices del modo longitudinal.
    k0_ : complex or float
        Módulo del vector de onda (w/Cl0).
    Qh_mod, ang : ndarray
        Salida de ``precompute_Qh``.
    a : float
        Constante de red.
    lattice : str
        Tipo de red.

    Devuelve
    -------
    complex
        Valor de la suma de red para (M,m).
    """
    N0 = M - m
    N = abs(N0)
    # Término diagonal de Hankel (sólo si N==0)
    krondelta = int(N == 0)
    denom1 = k0_ * np.pi * a
    # Prevenir división por cero
    if np.abs(denom1) < 1e-12:
        denom1 = 1e-12
    term1 = ((2j + k0_ * a * np.pi * hn(1, k0_ * a)) / denom1) * krondelta
    # Término de suma de red vectorizado
    term2 = S1_pre(N, k0_, Qh_mod, ang, a, lattice)
    # Denominador de Bessel
    denom_bessel = jn(N + 1, k0_ * a)
    if np.abs(denom_bessel) < 1e-12:
        denom_bessel = 1e-12
    Sval = -(term1 + term2) / denom_bessel
    if N0 < 0:
        Sval = -np.conj(Sval)
    return Sval

# NUEVAS VERSIONES DE S Y S1 QUE ACEPTAN k_vec EN VEZ DE SU MÓDULO

def S1_vec(M, m, k_vec, n, a, k0_, lattice='hx'):
    """
    Parte de la suma de red para k como vector bidimensional (kx, ky).
    """
    N0 = M - m
    N = abs(N0)
    S = 0
    tol = 1e-12

    for i in range(-n, n + 1):
        for l in range(-n, n + 1):
            Qh = Kh(a, i, l, lattice) + k_vec
            Qh_ = norm(Qh)
            if Qh_ < tol:
                continue
            ang = np.angle(Qh[0] + 1j * Qh[1])
            num = jn(N + 1, Qh_ * a) * np.exp(1j * N * ang)
            den = Qh_ * (Qh_**2 - k0_**2)
            if abs(den) < 1e-12 or np.isnan(den):
                den = 1e-12  # prevenir división por cero o NaNs
            S += num / den

    area = a ** 2 if lattice == 'sq' else (np.sqrt(3) * a ** 2 / 2)
    valor = (S * k0_ * 4 * (1j)**(N + 1)) / area
    return valor

def S_vec(M, m, k_vec, n, a, k0_, lattice='hx'):
    """
    Suma de red total para k como vector bidimensional (kx, ky).
    """
    N0 = M - m
    N = abs(N0)
    krondelta = int(N == 0)

    den1 = k0_ * np.pi * a
    if abs(den1) < 1e-12:
        den1 = 1e-12  # prevenir división por cero

    term1 = ((2j + k0_ * a * np.pi * hn(1, k0_ * a)) / den1) * krondelta
    term2 = S1_vec(M, m, k_vec, n, a, k0_, lattice)

    denom_bessel = jn(N + 1, k0_ * a)
    if abs(denom_bessel) < 1e-12:
        denom_bessel = 1e-12  # evitar división por cero

    Sval = -(term1 + term2) / denom_bessel
    if N0 < 0:
        Sval = -np.conj(Sval)
    return Sval

def generar_malla_2D(a, nk=20, lattice='hx'):
    b1 = Kh(a, 1, 0, lattice)
    b2 = Kh(a, 0, 1, lattice)
    u_vals = np.linspace(0, 1, nk, endpoint=False)
    v_vals = np.linspace(0, 1, nk, endpoint=False)
    k_malla = np.zeros((nk, nk, 2))
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            k_malla[i, j, :] = u * b1 + v * b2
    return k_malla, b1, b2

def generar_zona_brillouin_hexagonal(a, nk):
    k_malla_paralelogramo, b1, b2 = generar_malla_2D(a, nk, lattice='hx')
    centro_celda = 0.5 * (b1 + b2)
    k_malla_centrada = k_malla_paralelogramo - centro_celda
    G_vectores = [
        Kh(a, 1, 0, 'hx'), Kh(a, -1, 0, 'hx'),
        Kh(a, 0, 1, 'hx'), Kh(a, 0, -1, 'hx'),
        Kh(a, 1, -1, 'hx'), Kh(a, -1, 1, 'hx')
    ]
    puntos_en_bz = []
    tolerancia = 1e-9
    for i in range(nk):
        for j in range(nk):
            k_vec = k_malla_centrada[i, j]
            esta_dentro = True
            for G in G_vectores:
                if np.dot(k_vec, G) > 0.5 * norm(G)**2 + tolerancia:
                    esta_dentro = False
                    break
            if esta_dentro:
                puntos_en_bz.append(k_vec)
    return np.array(puntos_en_bz)

# ===================================================================
# --- NUEVA FUNCIÓN PARA GRAFICAR ---
# ===================================================================

def generar_zona_brillouin_wigner(a, nk=200):
    """
    Genera puntos (kx, ky) dentro de la 1BZ hexagonal usando el criterio geométrico
    de celdas de Wigner-Seitz (más preciso que el dot-product).
    """
    # Base recíproca
    b1 = Kh(a, 1, 0, 'hx')
    b2 = Kh(a, 0, 1, 'hx')

    # Crear malla cuadrada grande centrada en Gamma
    kx = np.linspace(-5 * np.pi / a, 5 * np.pi / a, nk)
    ky = np.linspace(-5 * np.pi / a, 5 * np.pi / a, nk)
    KX, KY = np.meshgrid(kx, ky)
    puntos = np.stack((KX.flatten(), KY.flatten()), axis=1)

    # Red recíproca de primeros vecinos (excepto el origen)
    vecinos = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            vecinos.append(Kh(a, i, j, 'hx'))
    vecinos = np.array(vecinos)

    puntos_dentro = []
    for k_vec in puntos:
        distancia_origen = np.linalg.norm(k_vec)
        distancias_vecinos = np.linalg.norm(k_vec - vecinos, axis=1)
        if np.all(distancia_origen <= distancias_vecinos):
            puntos_dentro.append(k_vec)

    return np.array(puntos_dentro)
