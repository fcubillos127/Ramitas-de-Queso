from __future__ import annotations
import os
import time
import scipy
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
from numpy.linalg import det
from scipy.special import jvp, h1vp, jv, hankel1
from scipy.optimize import fsolve, linear_sum_assignment, differential_evolution, root_scalar
from scipy.signal import find_peaks
#import Four_Materiales_Tools_prima as SW
from scipy.integrate import quad
import Suma_red_A_prima_prueba as sum
# NUEVOS IMPORTS (fnv)
from fnv.fnv_store import FNVData, build_fnv_grid as _build_fnv_grid, save_fnv as _save_fnv_npz, load_fnv as _load_fnv_npz
from fnv.fnv_plot import (
    plot_Re_fn as _fnv_plot_Re_fn,
    plot_Im_fn as _fnv_plot_Im_fn,
    plot_Re_vn as _fnv_plot_Re_vn,
    plot_Im_vn as _fnv_plot_Im_vn,
)
from pathlib import Path
from project_io import data_path, graphs_path  # NUEVO
# CSV:
from fnv.fnv_csv import (
    export_default_block_csv, load_fnv_from_csv,
    export_default_split_csv, load_fnv_from_csv_split,
)
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator, CubicSpline, UnivariateSpline
from scipy.signal import savgol_filter
from utils import *

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def savefrec(k2, frec, path):
    count = 0
    for i in k2:
        name = path + '/frecuencias' + str(np.around(i,2)) + '.txt'
        np.savetxt(name, frec[count])
        count = count + 1

class Red:
    def __init__(self, comp):
        self.a = 1 # Constante de Red (en metros)
        self.comp = comp # Componentes elásticos (nombre de los materiales)
        self.vel0 = None # Velocidades de propagacion de la onda en la matriz (en metros/segundos)
        self.vels = None # Velocidades de propagacion de la onda en la matriz (en metros/segundos)
        self.dens = None # Densidades de los medios materiales: [matriz, inclusion] (en kilogramos/metros^3)
        self.filling = 0 # Fraccion de llenado: Area_inclusion / Area_celda
        self.cut = 2 # Limite de la expansion de la onda en el espacio de Fourier
        self.nbands = 0 # Numero de (las primeras nbands) bandas buscadas
        self.nk = 0 # Cantidad de puntas buscados sobre el camino que une los puntos de alta simetria en 1D
        self.k_init = 0 # Valor del limite inferior del intervalo de vectores de Bloch k recorridos en el camino 1D
        self.r1 = None # Radio de la inclusion (circular en 2D)
        self.r2 = None # Radio inferior de la segunda inclusion (en forma de aro en 2D)
        self.r3 = None # Radio superior de la segunda inclusion (en forma de aro en 2D)
        self.k = None # Vector con los valores del modulo del vector de Bloch (eje x para la estructura de bandas en 1D)
        self.n_suma = 5 # Cantidad de términos que utilizará la suma de red
        self.kx_malla = None #Vector con los valores de la componente x del vector de Bloch (eje x para la estructura de bandas en 2D)
        self.ky_malla = None #Vector con los valores de la componente y del vector de Bloch (eje y para la estructura de bandas en 2D)
        self.b1 = None # Primer vector base del espacio reciproco
        self.b2 = None # Segundo vector base del espacio reciproco
        self.frec_malla = None # Autofrecuencias para la configuracion en 2D (eje z para la estructura de bandas en 2D)
        self.autovec_malla = None # Autovectores para la configuracion en 2D
        self.shear = None # Parametro de Lame 
        self._lame = None  # Reservado para uso con @property # Parametro de Lame
        self._shear = None # Parametro de Lame
        self.omega_longitudinal = None  # Autofrecuencias para la configuracion en 1D (eje y para la estructura de bandas en 1D)
        self.foldername = None # Nombre de la carpeta donde se guardaran los resultados
        self.frecfolder = None # Nombre de la subcarpeta donde se guardaran los resultados
        self.lattice = 'sq' # Geometría de la red: 'sq' para cuadrada y 'hx' para triangular (o hexagonal)
        self._set_k_end() # Valor del limite superior del intervalo de vectores de Bloch k recorridos en el camino 1D
        # Contenedor central y contexto legacy
        self.fnv_data: FNVData | None = None
        self._legacy_context = {}
        self.cond_borde = None
        self.imag_tol = 1e-1
        self.sol_tol = 1e-1
        # ---------------------------
        # --- edición manual de puntos (borrado/restauración/preview) ---
        self._deleted_stack = []   # historial para undo/restore
        self._mask_sigma = None    # máscara RAM para sigma_min [nk_full, nbands]
        self._mask_fullgrid = None # máscara RAM para fullgrid  [nk_full, nbands]
        # ----------------------------------------------------------------
        self._cache_G0 = {}
        self.epsfcn=None
    @property
    def Fnb(self):
        """DEPRECATED (solo lectura): última serie Fnb usada en un plot."""
        return self._legacy_context.get("Fnb", None)
    
    @property
    def Fnc(self):
        """DEPRECATED (solo lectura): última serie Fnc usada en un plot."""
        return self._legacy_context.get("Fnc", None)
    
    @property
    def Vnb(self):
        """DEPRECATED (solo lectura): última serie Vnb usada en un plot."""
        return self._legacy_context.get("Vnb", None)
    
    @property
    def Vnc(self):
        """DEPRECATED (solo lectura): última serie Vnc usada en un plot."""
        return self._legacy_context.get("Vnc", None)

    def fnv_build(self, n_list, freq_list, r):
        """Llena/actualiza self.fnv_data con eje (n, freq, r) para F/V ⨉ b/c (r arbitrario)."""
        return _build_fnv_grid(self, list(n_list), list(freq_list), np.asarray(r, dtype=float))
    
    def save_fnv(self, path: str):
        if self.fnv_data is None:
            raise ValueError("No hay datos fnv para guardar.")
        _save_fnv_npz(self.fnv_data, path)
    
    def load_fnv(self, path: str):
        self.fnv_data = _load_fnv_npz(path)

        
    def _set_k_end(self):
        """ Calcula el valor del último punto del eje k (vector de Bloch) para el camino recorrido en el camino de alta simetría 1D segun la red """
        if self.lattice == 'sq':
            self.k_end = 3 * np.pi / self.a # Módulo de la longitud del camino recorrido en el camino de alta simetría 1D en la red cuadrada
        elif self.lattice == 'hx':
            self.k_end = (2 * np.pi * (1 + 1/np.sqrt(3))) / self.a # Módulo de la longitud del camino recorrido en el camino de alta simetría 1D en la red hexagonal

    def _calc_r(self):
        """ Calcula el radio de la inclusion central dado un filling ratio según la red """
        if self.lattice == 'sq':
            return np.sqrt((self.filling * self.a**2) / np.pi) # Radio de la inclusión central dado un filling ratio si la red es cuadrada
        elif self.lattice == 'hx':
            return np.sqrt((np.sqrt(3) * self.filling * self.a**2) / (2 * np.pi)) # Radio de la inclusión central dado un filling ratio si la red es hexagonal

    def _rebuild_vel(self, mu, lamb, rho):
        """ Si no se entregan los valores de las velocidades de propagacion longitudinal (vl) y transversal (vt), se calculan en base a los parametros elasticos"""
        vl = np.sqrt((2 * mu + lamb) / rho)
        vt = np.sqrt(mu / rho)
        return [vl, vt]

    @property
    def mu(self):
        """ Calcula el parametro de Lame mu si se dan las velocidades, pero no los parametros elasticos """
        if self.vel0 and self.dens:
            return [self.dens[0] * self.vel0[1]**2, self.dens[1] * self.vels[1]**2]
        elif self.shear:
            return self.shear
        return None

    @property
    def lame(self):
        """ Calcula el parametro de Lame lambda si se dan las velocidades, pero no los parametros elasticos """
        if self.vel0 and self.dens:
            lamb0 = self.dens[0] * (self.vel0[0]**2) - 2 * self.mu[0]
            lambs = self.dens[1] * (self.vels[0]**2) - 2 * self.mu[1]
            return [lamb0, lambs]
        elif self._lame:
            return self._lame
        return None

    def save_fnv_csv_split_default(
        self,
        subdir: str = "fnv/split",
        name_template: str = "{kind}{chan}_n{n}.csv",
        float_fmt: str | None = None,
    ):
        """
        Exporta CSVs separados por canal y modo n al folder data/<subdir>/.
        Un archivo por combinación (kind,chan,n) con columnas: freq, r, real, imag.
        name_template debe incluir {kind}, {chan}, {n}.
        Devuelve la lista de rutas (Path).
        """
        if self.fnv_data is None:
            raise ValueError("No hay datos fnv para guardar (self.fnv_data is None).")
        # Permite "fnv/split" o "fnv\\split"
        parts = subdir.replace("\\", "/").split("/")
        out_dir = data_path(*parts)
        return export_default_split_csv(
            self.fnv_data,
            out_dir,
            name_template=name_template,
            float_fmt=float_fmt,
        )
    
    def load_fnv_csv_split_default(
        self,
        subdir: str = "fnv/split",
        name_pattern: str = r"(?P<kind>[FV])(?P<chan>[bc])_n(?P<n>\d+)\.csv",
    ):
        """
        Carga self.fnv_data desde múltiples CSVs en data/<subdir>/ (formato split).
        Espera archivos que cumplan con name_pattern (por defecto: F/V + b/c + n entero).
        """
        parts = subdir.replace("\\", "/").split("/")
        in_dir = data_path(*parts)  # mismo helper = ruta en data/
        self.fnv_data = load_fnv_from_csv_split(in_dir, name_pattern=name_pattern)
        return in_dir

    def save_fnv_csv_default(self, filename: str = "fnv_data.csv", subdir: str = "fnv"):
        """
        Exporta el BLOQUE por defecto (el último r usado/construido) a CSV 'largo':
        columnas: kind, chan, n, freq, r, real, imag
        Guarda en data/<subdir>/<filename>.
        """
        if self.fnv_data is None:
            raise ValueError("No hay datos fnv para guardar en CSV (self.fnv_data is None).")
        csv_path = data_path(subdir, filename)
        export_default_block_csv(self.fnv_data, str(csv_path))
        return csv_path
    
    def load_fnv_csv_default(self, filename: str = "fnv_data.csv", subdir: str = "fnv"):
        """
        Carga self.fnv_data desde data/<subdir>/<filename> (CSV largo).
        Sobrescribe el contenedor actual con un FNVData de un solo bloque (default).
        """
        csv_path = data_path(subdir, filename)
        self.fnv_data = load_fnv_from_csv(str(csv_path))
        return csv_path

    def save_fnv_default(self, filename: str = "fnv_cache.npz", subdir: str = "fnv") -> Path:
        """
        Guarda self.fnv_data en data/<subdir>/<filename>.
        Devuelve la ruta absoluta donde se guardó.
        """
        if self.fnv_data is None:
            raise ValueError("No hay datos fnv para guardar (self.fnv_data is None).")
        out = data_path(subdir, filename)
        # delega en los helpers ya integrados
        from fnv.fnv_store import save_fnv as _save_fnv_npz
        _save_fnv_npz(self.fnv_data, str(out))
        return out
    
    def load_fnv_default(self, filename: str = "fnv_cache.npz", subdir: str = "fnv") -> Path:
        """
        Carga self.fnv_data desde data/<subdir>/<filename>.
        Devuelve la ruta absoluta usada.
        """
        from fnv.fnv_store import load_fnv as _load_fnv_npz
        src = data_path(subdir, filename)
        self.fnv_data = _load_fnv_npz(str(src))
        return src
    
    def save_figure(self, fig, filename: str, subdir: str = "fnv", dpi: int = 300) -> Path:
        """
        Guarda una figura Matplotlib en graphs/<subdir>/<filename>.
        Devuelve la ruta absoluta donde se guardó.
        """
        out = graphs_path(subdir, filename)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        return out
    
    def save_current_figure(self, filename: str, subdir: str = "fnv", dpi: int = 300) -> Path:
        """
        Guarda la figura activa (plt.gcf()) en graphs/<subdir>/<filename>.
        Útil si no capturaste 'fig' desde plot_*.
        """
        import matplotlib.pyplot as plt
        fig = plt.gcf()
        return self.save_figure(fig, filename, subdir=subdir, dpi=dpi)

    def asign_param(self):
        """ Calcula y asigna los parametros no entregados segun los parametros conocidos.
        Ademas, se crean directorios para guardar los resultados y se genera el un arreglo con los valores del vector de Bloch
        a lo largo del camino por los puntos de alta simetria """
        if self.dens is None:
            raise ValueError("No density inputs")
        if self.vel0 and self.vels:
            self.shear = self.mu
            self._lame = self.lame
        elif self.shear and self._lame:
            self.vel0 = self._rebuild_vel(self.shear[0], self._lame[0], self.dens[0])
            self.vels = self._rebuild_vel(self.shear[1], self._lame[1], self.dens[1])
        else:
            raise ValueError("Material parameters are incomplete")
        self.r1 = self._calc_r()
        self.k = np.linspace(self.k_init, self.k_end, self.nk)
        self.foldername = self._make_path()
        self.frecfolder = os.path.join(self.foldername, f"filling = {self.filling} N={self.nbands}, {self.lattice}, {self.cond_borde} ,psi={self.psi}")

    def _make_path(self):
        """ Genera el nombre de los directorios para guardar los resultados"""
        return os.path.join(os.path.expanduser('~'), 'Documents', 'Metamateriales', 'Maiguel',
                            f'lattice_2try ={self.lattice}')

    def k0(self, f, p):
        """ Calcula el vector de onda en el la matriz (solo longitudinal).
        p corresponde a la polarizacion y es un parametro necesario para el correcto funcionamiento de la suma de red,
        aun asi, en este caso nuestro sistema es acustico, asi que solo usaremos modos longitudinales """
        Cl0, Ct0 = self.vel0
        re_f, im_f = f
        w = (re_f + 1j*im_f)
        val = w/Cl0 if p==0 else w/Ct0
        return val
    
    def ks(self, f, p):
        Cls, Cts = self.vels
        re_f, im_f = f
        w = (re_f + 1j*im_f)
        val = w/Cls if p==0 else w/Cts
        return val
    
    def kls(self, f, p):
        """ Calcula el vector de onda en el la inclusión (solo longitudinal).
        p corresponde a la polarizacion y es un parametro necesario para el correcto funcionamiento de la suma de red,
        aun asi, en este caso nuestro sistema es acustico, asi que solo usaremos modos longitudinales  """
        Cl0 = self.vels[0]
        re_f, im_f = f
        w = (re_f + 1j*im_f)
        val = w/Cl0
        return val

    def G0(self, f, k, pol, cut, n_suma=None):
        """
        Genera la matriz ``G0`` utilizando la suma de red.

        Para mejorar el rendimiento, esta versión utiliza las
        funciones vectorizadas de ``Suma_red_A_prima``.  Se
        precalculan los módulos y ángulos de los vectores de red
        recíproca desplazados por el vector de Bloch ``k_vec`` y se
        reutilizan al calcular los términos ``S_pre``.

        Parámetros
        ----------
        f : tuple-like
            Frecuencia compleja (parte real, parte imaginaria).
        k : float
            Módulo del vector de Bloch en la dirección seleccionada.
        pol : int
            Polarización (actualmente se ignora ya que sólo se
            consideran modos longitudinales).
        cut : int
            Límite superior de la suma en M y m (la matriz tendrá
            tamaño ``2*cut+1``).
        n_suma : int or None
            Número de términos en la suma de red para S; si es
            ``None`` se usa ``self.n_suma``.

        Devuelve
        -------
        mat : ndarray, shape ((2*cut+1), (2*cut+1))
            Matriz compleja de la suma de red.
        """
        if n_suma is None:
            n_suma = self.n_suma
        size = 2 * cut + 1
        a = self.a
        lattice = self.lattice

        # --- Memoización: G0 NO depende de psi ni de los materiales de la
        #     inclusión; solo de (f, k, pol, cut, n_suma, a, lattice, vel0).
        #     Esto evita recalcularla, p.ej. al comparar dos valores de psi
        #     sobre la misma malla de frecuencias. Se usa un caché propio para
        #     no colisionar con el de G0_cached (que guarda G0_convergente).
        cache = self.__dict__.setdefault("_cache_G0_plain", {})
        key = (round(float(f[0]), 12), round(float(f[1]), 12), round(float(k), 12),
               int(pol), int(cut), int(n_suma), float(a), str(lattice),
               float(self.vel0[0]), float(self.vel0[1]))
        cached = cache.get(key)
        if cached is not None:
            return cached

        k0_ = self.k0(f, pol)
        # Convertir k escalar en vector de Bloch (kx, ky)
        k_vec = sum.K(a, k, lattice)
        Qh_mod, ang = sum.precompute_Qh(a, k_vec, int(n_suma), lattice)

        # --- La matriz es de Toeplitz: S_pre(M, m, ...) depende únicamente de
        #     la diferencia d = M - m (con la simetría S(-d) = -conj(S(d))).
        #     Por eso basta calcular 2*cut+1 valores en vez de (2*cut+1)^2.
        Sd = {}
        for d in range(0, 2 * cut + 1):
            Sd[d] = sum.S_pre(d, 0, k0_, Qh_mod, ang, a, lattice)
        for d in range(1, 2 * cut + 1):
            Sd[-d] = -np.conj(Sd[d])

        mat = np.empty((size, size), dtype=complex)
        for i in range(-cut, cut + 1):
            row = i + cut
            for j in range(-cut, cut + 1):
                mat[row, j + cut] = Sd[i - j]

        # Cota de memoria del caché (FIFO simple).
        if len(cache) >= 8192:
            cache.pop(next(iter(cache)))
        cache[key] = mat
        return mat

    def G0_convergente(self,
        f, k, pol, cut,
        n_suma_ini=None, tol=1e-6, n_suma_max=200, paso=1,
        norma='max', verbose=False
    ):
        """
        Suma de red adaptativa incremental (anillos |h|_∞ = n).
        Requiere Suma_red_A_prima_prueba.py con: K, Kh, S1_pre.
    
        Parámetros
        ----------
        f : (float, float)
            Frecuencia compleja como par (Re(w), Im(w)).
        k : float
            Parámetro de Bloch a lo largo del camino 1D (tu self.k).
        pol : int
            Polarización (si aplica).
        cut : int
            Orden multipolar => matriz (2*cut+1) x (2*cut+1).
        n_suma_ini : int or None
            Anillo inicial (si None usa self.n_suma o 3).
        tol : float
            Tolerancia de convergencia relativa.
        n_suma_max : int
            Límite superior del anillo.
        paso : int
            Incremento del anillo por iteración.
        norma : {'max','fro'}
            Norma para el criterio de convergencia.
        verbose : bool
            Logs breves de progreso.
    
        Returns
        -------
        mat : ndarray (2*cut+1, 2*cut+1), complex
        info : dict  {'n_suma': int, 'err_rel': float, 'converged': bool}
        """
        import numpy as np
        import Suma_red_A_prima_prueba as sum
    
        # ---- utilidades internas ----------------------------------------------
        def _ring_pairs(n):
            """Lista de pares (i,j) con max(|i|,|j|)=n; incluye (0,0) sólo si n==0."""
            if n == 0:
                return [(0, 0)]
            pairs = []
            # bordes superior e inferior
            for i in range(-n, n+1):
                pairs.append((i,  n))
                pairs.append((i, -n))
            # bordes izquierdo y derecho (sin repetir esquinas)
            for j in range(-n+1, n):
                pairs.append(( n, j))
                pairs.append((-n, j))
            return pairs
    
        def _Qh_from_ring(a, k_vec, n, lattice):
            """Construye Qh_mod y ang sólo del anillo n."""
            pairs = _ring_pairs(n)
            if len(pairs) == 0:
                # no debería ocurrir
                return np.empty((0,)), np.empty((0,))
            # Vectores Kh + k_vec
            Q = []
            for (i, j) in pairs:
                Q.append(sum.Kh(a, i, j, lattice) + k_vec)
            Q = np.asarray(Q, dtype=float)  # shape (Nr, 2)
            Qh_mod = np.linalg.norm(Q, axis=1)
            ang = np.angle(Q[:, 0] + 1j*Q[:, 1])
            return Qh_mod, ang
    
        # ---- parámetros y base --------------------------------------------------
        if n_suma_ini is None:
            n_suma_ini = getattr(self, 'n_suma', 3)
    
        size = 2*cut + 1
        a = float(self.a)
        lattice = getattr(self, 'lattice', 'sq')
        k0_ = self.k0(f, pol)             # típico: k0_ = w / C_l0 (compat.)
        k_vec = sum.K(a, k, lattice)      # tu generador de Bloch en recíproco
    
        # Precomputo de constantes por diferencia N = |M-m| (matrices base)
        M_idx = np.arange(-cut, cut+1)
        m_idx = np.arange(-cut, cut+1)
        M_grid, m_grid = np.meshgrid(M_idx, m_idx, indexing='ij')
        N0 = M_grid - m_grid                 # puede ser <0
        N  = np.abs(N0)
    
        # término "diagonal Hankel" como en tu S_pre:
        # term1 = ((2j + k0*a*pi*H1^(1)(k0*a)) / (k0*pi*a)) * δ_{N,0}
        from scipy.special import hankel1 as hn, jn
        krondelta = (N == 0).astype(float)
        denom1 = k0_ * np.pi * a
        denom1 = np.where(np.abs(denom1) < 1e-12, 1e-12, denom1)
        term1 = ((2j + k0_ * a * np.pi * hn(1, k0_ * a)) / denom1) * krondelta
    
        # denominador de Bessel: j_{N+1}(k0*a)
        denom_bessel = jn(N + 1, k0_ * a)
        denom_bessel = np.where(np.abs(denom_bessel) < 1e-12, 1e-12, denom_bessel)
    
        # base S = -(term1)/denom_bessel, con simetría N0<0: S -> -conj(S)
        S_base = -(term1) / denom_bessel
        S_base = np.where(N0 < 0, -np.conj(S_base), S_base)
    
        mat = S_base.copy()
        mat_prev = None
        err_rel = np.inf
        converged = False
    
        # ---- lazo incremental por anillos n ------------------------------------
        # si n_suma_ini > 0, podemos incluir el anillo 0 primero (opcional)
        n_start = 0 if n_suma_ini == 0 else n_suma_ini
    
        for n in range(n_start, n_suma_max + 1, paso):
            # construir Qh sólo del anillo n
            Qh_mod_ring, ang_ring = _Qh_from_ring(a, k_vec, n, lattice)
            if verbose:
                print(f"[G0-conv] anillo n={n}  puntos={Qh_mod_ring.size}")
    
            # contribución de red del anillo para cada (M,m):
            # term2_ring = S1_pre(N, k0_, Qh_ring, ang_ring, a, lattice)
            # ΔS = -(term2_ring)/denom_bessel   y aplicar simetría si N0<0
            # vectorizamos por N usando bucle sobre N únicos para ahorrar llamadas
            mat_inc = np.zeros_like(mat, dtype=complex)
            for Nu in np.unique(N.ravel()):
                mask = (N == Nu)
                if not np.any(mask):
                    continue
                # llamar una vez S1_pre para este Nu con la Qh del anillo
                term2_ring = sum.S1_pre(int(Nu), k0_, Qh_mod_ring, ang_ring, a, lattice)
                dS = -(term2_ring) / denom_bessel[mask]
                # simetría para N0<0
                dS = np.where(N0[mask] < 0, -np.conj(dS), dS)
                mat_inc[mask] = dS
    
            # acumular
            mat_new = mat + mat_inc
    
            # criterio de convergencia
            if mat_prev is not None:
                dif = mat_new - mat
                if norma == 'fro':
                    num = np.linalg.norm(dif)
                    den = max(1.0, np.linalg.norm(mat_new))
                else:
                    num = np.max(np.abs(dif))
                    den = max(1.0, np.max(np.abs(mat_new)))
                err_rel = float(num / den)
                if verbose:
                    print(f"[G0-conv] n={n:3d}  err_rel={err_rel:.3e}")
                if err_rel < tol:
                    converged = True
                    if verbose:
                        print(f"[G0-conv] ✅ Convergió: n_suma={n}, tol={tol:g}")
                    return mat_new, {'n_suma': n, 'err_rel': err_rel, 'converged': True}
    
            mat_prev = mat
            mat = mat_new
    
        if verbose:
            print(f"[G0-conv] ⚠️ No convergió (n_suma_max={n_suma_max}), err_rel={err_rel:.3e}")
        return mat, {'n_suma': n, 'err_rel': err_rel, 'converged': False}

    # --- helpers de caché (se crean on-demand) -----------------------------------
    def _ensure_g0_caches(self):
        if not hasattr(self, "_g0_geom_cache"):
            self._g0_geom_cache = {}          # {(k,a,lattice,n): (Qh_mod_ring, ang_ring)}
        if not hasattr(self, "_g0_result_cache"):
            self._g0_result_cache = {}        # key -> (mat, info)
            self._g0_result_cache_order = []  # LRU orden
            self._g0_result_cache_max = 64    # capacidad por defecto
    
    def _ring_pairs(self, n: int):
        """Pares (i,j) con max(|i|,|j|)=n. n==0 -> [(0,0)]."""
        if n == 0:
            return [(0, 0)]
        pairs = []
        for i in range(-n, n+1):
            pairs.append((i,  n)); pairs.append((i, -n))
        for j in range(-n+1, n):
            pairs.append(( n, j)); pairs.append((-n, j))
        return pairs
    
    def _get_ring_geom(self, k: float, n: int):
        """
        Devuelve (Qh_mod_ring, ang_ring) cacheados para el anillo n del k dado.
        Clave del caché: (round(k,12), a, lattice, n).
        """
        self._ensure_g0_caches()
        a = float(self.a)
        lattice = getattr(self, "lattice", "sq")
        key = (round(float(k), 12), a, str(lattice), int(n))
        if key in self._g0_geom_cache:
            return self._g0_geom_cache[key]
    
        import numpy as np
        import Suma_red_A_prima_prueba as sum
    
        k_vec = sum.K(a, k, lattice)
        pairs = self._ring_pairs(n)
        Q = np.array([sum.Kh(a, i, j, lattice) + k_vec for (i, j) in pairs], dtype=float)  # (Nr,2)
        Qh_mod = np.linalg.norm(Q, axis=1)
        ang = np.angle(Q[:, 0] + 1j * Q[:, 1])
    
        self._g0_geom_cache[key] = (Qh_mod, ang)
        return self._g0_geom_cache[key]
    
    def _g0_result_cache_get(self, key):
        self._ensure_g0_caches()
        if key in self._g0_result_cache:
            # mover a reciente
            try:
                self._g0_result_cache_order.remove(key)
            except ValueError:
                pass
            self._g0_result_cache_order.append(key)
            return self._g0_result_cache[key]
        return None
    
    def _g0_result_cache_put(self, key, value):
        self._ensure_g0_caches()
        # insertar
        self._g0_result_cache[key] = value
        self._g0_result_cache_order.append(key)
        # LRU evict
        while len(self._g0_result_cache_order) > self._g0_result_cache_max:
            old = self._g0_result_cache_order.pop(0)
            self._g0_result_cache.pop(old, None)
    
    # --- versión incremental por anillos + caché de anillos ----------------------
    def G0_convergente_cached(
        self, f, k, pol, cut,
        n_suma_ini=None, tol=1e-6, n_suma_max=200, paso=1,
        norma='max', verbose=False,
        use_result_cache=False,
        stable_passes=2     # <-- NUEVO
    ):
        """
        Igual que antes, pero exige 'stable_passes' iteraciones consecutivas
        con err_rel < tol antes de declarar convergencia.
        """
        import numpy as np
        from scipy.special import hankel1 as hn, jn
        import Suma_red_A_prima_prueba as sum
    
        # --- cache resultado (opcional) ---
        a = float(self.a)
        lattice = getattr(self, "lattice", "sq")
        if n_suma_ini is None:
            n_suma_ini = getattr(self, "n_suma", 3)
        res_key = (round(float(k),12), round(float(f[0]),12), round(float(f[1]),12),
                   int(pol), int(cut), int(n_suma_ini), int(n_suma_max), int(paso),
                   float(tol), str(norma), str(lattice), a, int(stable_passes))
        if use_result_cache:
            got = self._g0_result_cache_get(res_key)
            if got is not None:
                if verbose:
                    print("[G0c] cache HIT (resultado completo)")
                return got
    
        # --- tamaños/índices multipolares ---
        size = 2*cut + 1
        k0_ = self.k0(f, pol)
        M_idx = np.arange(-cut, cut+1)
        m_idx = np.arange(-cut, cut+1)
        M_grid, m_grid = np.meshgrid(M_idx, m_idx, indexing='ij')
        N0 = M_grid - m_grid
        N  = np.abs(N0)
    
        # --- término base (como en S_pre) ---
        krondelta = (N == 0).astype(float)
        denom1 = k0_ * np.pi * a
        denom1 = np.where(np.abs(denom1) < 1e-12, 1e-12, denom1)
        term1 = ((2j + k0_ * a * np.pi * hn(1, k0_ * a)) / denom1) * krondelta
        denom_bessel = jn(N + 1, k0_ * a)
        denom_bessel = np.where(np.abs(denom_bessel) < 1e-12, 1e-12, denom_bessel)
    
        S_base = -(term1) / denom_bessel
        S_base = np.where(N0 < 0, -np.conj(S_base), S_base)
    
        mat = S_base.copy()
        mat_prev = None
        err_rel = np.inf
        converged = False
        ok_in_a_row = 0   # <-- NUEVO
    
        # --- lazo incremental por anillos ---
        n_start = 0 if n_suma_ini == 0 else n_suma_ini
        for n in range(n_start, n_suma_max + 1, paso):
            Qh_mod_ring, ang_ring = self._get_ring_geom(k, n)
            if verbose:
                print(f"[G0c] anillo n={n}  puntos={Qh_mod_ring.size}")
    
            mat_inc = np.zeros_like(mat, dtype=complex)
    
            # vectorizar por N únicos
            for Nu in np.unique(N.ravel()):
                mask = (N == Nu)
                if not np.any(mask):
                    continue
                term2_ring = sum.S1_pre(int(Nu), k0_, Qh_mod_ring, ang_ring, a, lattice)
                dS = -(term2_ring) / denom_bessel[mask]
                dS = np.where(N0[mask] < 0, -np.conj(dS), dS)
                mat_inc[mask] = dS
    
            mat_new = mat + mat_inc
    
            # criterio de convergencia con estabilidad
            if mat_prev is not None:
                dif = mat_new - mat
                if norma == 'fro':
                    num = np.linalg.norm(dif)
                    den = max(1.0, np.linalg.norm(mat_new))
                else:
                    num = np.max(np.abs(dif))
                    den = max(1.0, np.max(np.abs(mat_new)))
                err_rel = float(num / den)
    
                if err_rel < tol:
                    ok_in_a_row += 1
                else:
                    ok_in_a_row = 0
    
                if verbose:
                    print(f"[G0c] n={n:3d}  err_rel={err_rel:.3e}  stable {ok_in_a_row}/{stable_passes}")
    
                if ok_in_a_row >= stable_passes:
                    converged = True
                    if verbose:
                        print(f"[G0c] ✅ Convergió: n_suma={n}, tol={tol:g}, stable_passes={stable_passes}")
                    if use_result_cache:
                        self._g0_result_cache_put(res_key, (mat_new, {'n_suma': n, 'err_rel': err_rel, 'converged': True}))
                    return mat_new, {'n_suma': n, 'err_rel': err_rel, 'converged': True}
    
            mat_prev = mat
            mat = mat_new
    
        if verbose:
            print(f"[G0c] ⚠️ No convergió (n_suma_max={n_suma_max}), err_rel={err_rel:.3e}")
        info = {'n_suma': n, 'err_rel': err_rel, 'converged': False}
        if use_result_cache:
            self._g0_result_cache_put(res_key, (mat, info))
        return mat, info


    def determinant_longitudinal(self, f, k, cut,
                                 use_adaptive_g0=False,
                                 g0_tol=1e-6,
                                 g0_n_suma_max=300,
                                 g0_stable_passes=2,
                                 g0_verbose=False,
                                 **g0_extra):
        """ Forma la matriz de transmision y el determinante a calcular """
        cutoff=cut
        size = 2 * cutoff + 1
        if self.cond_borde=='rigid':
            T = np.diag(np.array([self.coeficiente_dispersion_elastic(f, n) for n in np.arange(-cutoff,cutoff+1)],dtype=complex))
        else:
            T = np.diag(np.array([self.coeficiente_dispersion_hollow(f, n) for n in np.arange(-cutoff,cutoff+1)],dtype=complex))
        #G = self.G0(f, k, 1, cutoff, n_suma)
        if use_adaptive_g0:
            G, ginfo = self.G0_convergente_cached(
                f, k, pol=1, cut=cut,
                n_suma_ini=getattr(self, 'n_suma', 3),
                tol=g0_tol, n_suma_max=g0_n_suma_max, paso=1,
                norma='max', verbose=g0_verbose,
                stable_passes=g0_stable_passes
            )
        else:
            G = self.G0(f, k, pol=1, cut=cut)

        M = T @ G
        identidad = np.identity(size)
        determinante = det(M - identidad)

        return [np.real(determinante), np.imag(determinante)]
    
    def Det_longitudinal(self, f, *args):
        k_val, cut, use_adaptive_g0, g0_tol, g0_n_suma_max, g0_stable_passes, g0_verbose = args
        return self.determinant_longitudinal(f, k_val, cut,
                                             use_adaptive_g0=use_adaptive_g0,
                                             g0_tol=g0_tol,
                                             g0_n_suma_max=g0_n_suma_max,
                                             g0_stable_passes=g0_stable_passes,
                                             g0_verbose=g0_verbose) 

    
    def graficar_dispersion_coef(
        self,
        psi: float,
        n: int,
        frequency=None,
        save_csv: bool = True,
        csv_subdir: str = "dispersion_coef",
        filename_template_csv: str = "Tn_psi{psi}_n{n}.csv",
        save_png: bool = True,
        png_subdir: str = "dispersion_coef",
        filename_template_png: str = "Tn_psi{psi}_n{n}.png",
        png_dpi: int = 300,
        show: bool = True,
    ):
        """
        Grafica Re/Im del coeficiente de dispersión T_n, guarda CSV en data/<csv_subdir>/
        y PNG en graphs/<png_subdir>/.
    
        Devuelve
        --------
        (fig, ax, csv_path, png_path)
        """
        import numpy as np
        import matplotlib.pyplot as plt
    
        # Guardar/restaurar self.psi
        psi_original = getattr(self, "psi", None)
        self.psi = psi
    
        # Malla de frecuencias (rad/s)
        if frequency is None:
            frequency = np.linspace(0.0, 1.5, 100)
        frequency = np.asarray(frequency, dtype=float)*2*np.pi*self.vel0[1]/self.a
    
        # Eje normalizado ωa/2πCt0
        Ct0 = self.vel0[1]
        w_norm = frequency*self.a / (2 * np.pi * Ct0)
    
        # Evaluación de T_n(ω)
        if self.cond_borde=='hollow':
            coeficientes = np.array(
                [self.coeficiente_dispersion_hollow([w, 0.001], int(n)) for w in frequency],
                dtype=complex
            )
        else:
            coeficientes = np.array(
                [self.coeficiente_dispersion_elastic([w, 0.001], int(n)) for w in frequency],
                dtype=complex
            )
        real_part = np.real(coeficientes)
        imag_part = np.imag(coeficientes)
    
        # Plot (Re en azul, Im en rojo)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(w_norm, real_part, "b-", label="Parte real")
        ax.plot(w_norm, imag_part, "r--", label="Parte imaginaria")
        ax.set_xlabel(r"$\omega a/2\pi C_{t0}$", fontsize=15)
        ax.set_ylabel(r"$T_n^0$", fontsize=15)
        ax.set_title(f"Coeficiente de dispersión para n={n} | ψ={psi}", fontsize=15)
        ax.legend()
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
    
        # Helpers de rutas (provenientes del proyecto)
        csv_path = None
        png_path = None
    
        def _slug(v):
            s = f"{v}".strip()
            s = s.replace(" ", "")
            s = s.replace("-", "m").replace("+", "")
            s = s.replace(".", "p")
            return s
    
        psi_s = _slug(psi)
        fname_csv = self.cond_borde+filename_template_csv.format(psi=psi_s, n=int(n))
        fname_png = filename_template_png.format(psi=psi_s, n=int(n))
    
        # ---- Guardar CSV en data/<csv_subdir>/ ----
        if save_csv:
            try:
                from project_io import data_path
                out_csv = data_path(csv_subdir, fname_csv)
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                try:
                    import pandas as pd
                    pd.DataFrame({
                        "freq": frequency,
                        "w_norm": w_norm,
                        "real": real_part.astype(float),
                        "imag": imag_part.astype(float),
                        "n": int(n),
                        "psi": float(psi),
                    }).to_csv(out_csv, index=False)
                except Exception:
                    arr = np.column_stack([frequency, w_norm, real_part, imag_part])
                    header = "freq,w_norm,real,imag"
                    np.savetxt(str(out_csv), arr, delimiter=",", header=header, comments="")
                csv_path = str(out_csv)
            except Exception:
                import os
                os.makedirs(csv_subdir, exist_ok=True)
                csv_path = os.path.join(csv_subdir, fname_csv)
                try:
                    import pandas as pd
                    pd.DataFrame({
                        "freq": frequency,
                        "w_norm": w_norm,
                        "real": real_part.astype(float),
                        "imag": imag_part.astype(float),
                        "n": int(n),
                        "psi": float(psi),
                    }).to_csv(csv_path, index=False)
                except Exception:
                    arr = np.column_stack([frequency, w_norm, real_part, imag_part])
                    header = "freq,w_norm,real,imag"
                    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")
    
        # ---- Guardar PNG en graphs/<png_subdir>/ ----
        if save_png:
            saved = False
            # 1) Preferir helper de la clase (usa graphs_path por dentro)
            """try:
                if hasattr(self, "save_figure"):
                    self.save_figure(fig, fname_png, subdir=png_subdir, dpi=png_dpi)
                    from project_io import graphs_path  # solo para obtener ruta absoluta
                    png_path = str(graphs_path(png_subdir, fname_png))
                    saved = True
                    print('hey1')
            except Exception:
                saved = False
                """
            # 2) Fallback usando graphs_path directamente
            if not saved:
                try:
                    from project_io import graphs_path
                    out_png = graphs_path(png_subdir, fname_png)
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(out_png, dpi=png_dpi, bbox_inches="tight")
                    png_path = str(out_png)
                    saved = True
                    print('hey2')
                except Exception:
                    saved = False
    
            # 3) Último fallback local ./graphs/<subdir>
            if not saved:
                import os
                out_dir = os.path.join("graphs", png_subdir)
                os.makedirs(out_dir, exist_ok=True)
                png_path = os.path.join(out_dir, fname_png)
                fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
                print('hey3')
    
        if show:
            plt.show()
    
        # Restaurar psi
        self.psi = psi_original
    
        return fig, ax, csv_path, png_path

    def graficar_dif_dispersion_coef(
        self,
        psi,
        psi2,
        n,
        frequency=None,
        save_csv=True,
        csv_subdir="dispersion_diff",
        name_template="dif_Tn_psi{psi}_psi2{psi2}_n{n}.csv",
        # Guardado automático de figuras (2 archivos: REAL e IMAG)
        save_png=True,
        png_subdir="dispersion_diff",
        png_name_template_real= "dif_Tn_psi{psi}_psi2{psi2}_nset_{nset}_REAL.png",
        png_name_template_imag= "dif_Tn_psi{psi}_psi2{psi2}_nset_{nset}_IMAG.png",
        png_dpi=300,
    ):
        """
        Grafica Re/Im de [T_n(psi) - T_n(psi2)] separadas en DOS figuras:
          - Figura 1: parte Real (color azul)
          - Figura 2: parte Imaginaria (color rojo)
        En ambas, los modos n se diferencian por estilo de línea.
    
        Además:
          - Guarda un CSV por cada n en data/<csv_subdir>/ (freq, w_norm, real, imag, n, psi, psi2).
          - Guarda dos PNG (REAL e IMAG) en graphs/<png_subdir>/.
    
        Parámetros
        ----------
        psi, psi2 : float
        n : int o iterable de int
        frequency : array-like (rad/s) o None -> if None: linspace(0,400,100)
        save_csv : bool
        csv_subdir : str
        name_template : str con {psi}, {psi2}, {n}
        save_png : bool
        png_subdir : str
        png_name_template_real, png_name_template_imag : str con {psi}, {psi2}, {nset}
        png_dpi : int
    
        Devuelve
        --------
        (fig_real, ax_real, fig_imag, ax_imag)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    
        # Helpers de rutas si existen
        data_path = None
        graphs_path = None
        try:
            from project_io import data_path as _data_path, graphs_path as _graphs_path
            data_path = _data_path
            graphs_path = _graphs_path
        except Exception:
            pass
    
        # Normalizar n a lista
        def _to_list(maybe_iter):
            if hasattr(maybe_iter, "__iter__") and not isinstance(maybe_iter, (str, bytes)):
                return list(maybe_iter)
            return [maybe_iter]
        n_list = [int(x) for x in _to_list(n)]
    
        # Malla de frecuencias (rad/s)
        if frequency is None:
            frequency = np.linspace(0, 400, 100)
        frequency = np.asarray(frequency, dtype=float)
    
        # Eje normalizado (como en tu convención)
        Ct0 = self.vel0[1]
        x = frequency / (2 * np.pi * Ct0)
    
        # Estilos por modo n
        style_cycle = [
            "-", "--", ":", "-.", (0, (5, 2, 1, 2)), (0, (3, 1, 1, 1, 1, 1)),
            (0, (1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5))
        ]
    
        col_real = "b"
        col_imag = "r"
    
        # Preservar psi original
        psi_original = getattr(self, "psi", None)
    
        # Figuras separadas
        fig_re, ax_re = plt.subplots(figsize=(8, 5))
        fig_im, ax_im = plt.subplots(figsize=(8, 5))
    
        # Bucket CSV por n
        csv_data_by_n = {}
    
        # Cálculo y ploteo
        for idx, n_val in enumerate(n_list):
            ls = style_cycle[idx % len(style_cycle)]
    
            # T_n(psi)
            self.psi = psi
            if self.cond_borde=='hollow':
                coef1 = np.array([self.coeficiente_dispersion_hollow([w, 0.001], n_val) for w in frequency], dtype=complex)
    
            # T_n(psi2)
                self.psi = psi2
                coef2 = np.array([self.coeficiente_dispersion_hollow([w, 0.001], n_val) for w in frequency], dtype=complex)
            else:
                coef1 = np.array([self.coeficiente_dispersion_elastic([w, 0.001], n_val) for w in frequency], dtype=complex)
    
            # T_n(psi2)
                self.psi = psi2
                coef2 = np.array([self.coeficiente_dispersion_elastic([w, 0.001], n_val) for w in frequency], dtype=complex)
    
            dif = coef1 - coef2
            re = np.real(dif)
            im = np.imag(dif)
    
            # Real: azul; Imag: rojo. Estilos según n.
            ax_re.plot(x, re, color=col_real, linestyle=ls, label=f"Parte Real")
            ax_re.plot(x, im, color=col_imag, linestyle=ls, label=f"Parte Imaginaria")
            ax_im.plot(x, im, color=col_imag, linestyle=ls, label=f"n={n_val}")
    
            # Guardar datos para CSV
            csv_data_by_n[int(n_val)] = dict(
                freq=frequency.copy(),
                w_norm=x.copy(),
                real=re.astype(float),
                imag=im.astype(float),
            )
    
        # Leyendas (solo por n, ya que color es único por parte)
        ax_re.legend(loc="upper right")
        ax_im.legend(loc="upper right", title="Modo n")
    
        # Etiquetas y estilos
        for ax, label, color in [(ax_re, "Re", col_real), (ax_im, "Im", col_imag)]:
            ax.set_xlabel(r'$\omega a/2\pi C_{t0}$', fontsize=15)
            ax.set_ylabel(rf'${label}(T_n^\psi - T_n^{{\psi_2}})$', fontsize=15)
            if len(n_list) == 1:
                ax.set_title(f" n={n_list[0]}  |  ψ={psi}, ψ₂={psi2}", fontsize=15)
            else:
                ax.set_title(f" n ∈ {{{', '.join(map(str, n_list))}}}  |  ψ={psi}, ψ₂={psi2}", fontsize=15)
            ax.tick_params(axis="both", labelsize=13)
            ax.grid(True, alpha=0.25)
        fig_re.tight_layout()
        fig_im.tight_layout()
    
        # --------- Guardar CSV por n ---------
        if save_csv:
            def _slug(v):
                s = f"{v}".strip()
                s = s.replace(" ", "")
                s = s.replace("-", "m").replace("+", "")
                s = s.replace(".", "p")
                return s
            psi_s = _slug(psi)
            psi2_s = _slug(psi2)
    
            for n_val, d in csv_data_by_n.items():
                fname = name_template.format(self.cond_borde,psi=psi_s, psi2=psi2_s, n=int(n_val))
                if data_path is not None:
                    out = data_path(csv_subdir, fname)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    path_str = str(out)
                else:
                    import os
                    os.makedirs(csv_subdir, exist_ok=True)
                    path_str = os.path.join(csv_subdir, fname)
    
                try:
                    import pandas as pd
                    df = pd.DataFrame({
                        "freq": d["freq"],
                        "w_norm": d["w_norm"],
                        "real": d["real"],
                        "imag": d["imag"],
                        "n": int(n_val),
                        "psi": float(psi),
                        "psi2": float(psi2),
                    })
                    df.to_csv(path_str, index=False)
                except Exception:
                    arr = np.column_stack([d["freq"], d["w_norm"], d["real"], d["imag"]])
                    header = "freq,w_norm,real,imag"
                    np.savetxt(path_str, arr, delimiter=",", header=header, comments="")
    
        # --------- Guardar PNGs ---------
        if save_png:
            def _slug(v):
                s = f"{v}".strip()
                s = s.replace(" ", "")
                s = s.replace("-", "m").replace("+", "")
                s = s.replace(".", "p")
                return s
            psi_s = _slug(psi)
            psi2_s = _slug(psi2)
            nset = "_".join(str(int(v)) for v in n_list)
    
            fname_re = png_name_template_real.format(self.cond_borde,psi=psi_s, psi2=psi2_s, nset=nset)
            fname_im = png_name_template_imag.format(self.cond_borde,psi=psi_s, psi2=psi2_s, nset=nset)
    
            saved_ok_re = False
            saved_ok_im = False
            # 1) usar helper save_figure de tu clase si existe
            try:
                if hasattr(self, "save_figure"):
                    self.save_figure(fig_re, fname_re, subdir=png_subdir, dpi=png_dpi)
                    self.save_figure(fig_im, fname_im, subdir=png_subdir, dpi=png_dpi)
                    saved_ok_re = saved_ok_im = True
            except Exception:
                saved_ok_re = saved_ok_im = False
    
            # 2) fallback con graphs_path
            if (not saved_ok_re or not saved_ok_im) and graphs_path is not None:
                out_re = graphs_path(png_subdir, fname_re)
                out_im = graphs_path(png_subdir, fname_im)
                out_re.parent.mkdir(parents=True, exist_ok=True)
                out_im.parent.mkdir(parents=True, exist_ok=True)
                fig_re.savefig(out_re, dpi=png_dpi, bbox_inches="tight")
                fig_im.savefig(out_im, dpi=png_dpi, bbox_inches="tight")
                saved_ok_re = saved_ok_im = True
    
            # 3) último fallback: ./graphs/<subdir>
            if not saved_ok_re or not saved_ok_im:
                import os
                out_dir = os.path.join("graphs", png_subdir)
                os.makedirs(out_dir, exist_ok=True)
                fig_re.savefig(os.path.join(out_dir, fname_re), dpi=png_dpi, bbox_inches="tight")
                fig_im.savefig(os.path.join(out_dir, fname_im), dpi=png_dpi, bbox_inches="tight")
    
        # Restaurar psi original
        self.psi = psi_original
    
        # Retorna ambas figuras y ejes
        return fig_re, ax_re, fig_im, ax_im

    def graficar_determinante(self, psi, k, cut):
        self.psi = psi
        frequency = np.linspace(1000,1600)
        det = [self.determinant_longitudinal([freq, 0.01], k, cut)for freq in frequency]
        det = np.array(det)
        real_part = det[:,0]
        imag_part = det[:,1]

        plt.figure(figsize=(8, 5))
        plt.plot(frequency*self.a/(2*np.pi*self.vel0[1]), real_part, 'b-', label='Parte real')
        plt.plot(frequency*self.a/(2*np.pi*self.vel0[1]), imag_part, 'r--', label='Parte imaginaria')
        
        plt.xlabel(r'$\omega$')
        plt.ylabel('Determinante')
        plt.title(f'Determinante para psi={psi}')
        plt.legend()
        plt.ylim([-0.01,0.01])
        plt.show()

    def graficar_dif_determinante(self, psi, psi2, k, cut):
        self.psi = psi
        frequency = np.linspace(0,2000,100)
        det = [self.determinant_longitudinal([freq, 0.001], k, cut) for freq in frequency]
        self.psi=psi2
        det2 = [self.determinant_longitudinal([freq, 0.001], k, cut) for freq in frequency]
        dif= np.array(det2)-np.array(det)
        real_part = dif[:,0]
        imag_part = dif[:,1]

        plt.figure(figsize=(8, 5))
        plt.plot(frequency/(2*np.pi*self.vel0[1]), real_part, 'b-', label='Parte real')
        plt.plot(frequency/(2*np.pi*self.vel0[1]), imag_part, 'r--', label='Parte imaginaria')
    
        plt.xlabel(r'$\omega$')
        plt.ylabel('Determinante')
        plt.title(f'Determinante para psi={psi}')
        plt.legend()
        plt.ylim([-0.1,0.1])
        plt.show()
        
    def plot_Re_fn(self, n, freq, r1, r2):
        return _fnv_plot_Re_fn(self, n, freq, r1, r2)
    
    def plot_Im_fn(self, n, freq, r1, r2):
        return _fnv_plot_Im_fn(self, n, freq, r1, r2)
    
    def plot_Re_vn(self, n, freq, r1, r2):
        return _fnv_plot_Re_vn(self, n, freq, r1, r2)
    
    def plot_Im_vn(self, n, freq, r1, r2):
        return _fnv_plot_Im_vn(self, n, freq, r1, r2)
        fig, ax = plt.subplots(figsize=(8, 5))
        mark = [None, 'o', '^']
        col = ['k', 'r']
        lines = ['-', '','']
        def _legend_handles_types(ax):
            from matplotlib.lines import Line2D
            return [
                Line2D([0], [0], color=col[0], label=r'$V_n^{(b)}$'),
                Line2D([0], [0], color=col[1], label=r'$V_n^{(c)}$'),
            ]
    
        def _legend_handles_labels(ax):
            from matplotlib.lines import Line2D
            return [
                Line2D([0], [0], color='black', marker=None, linestyle='-', label=r'$\omega$'+f'=0.5'),
                Line2D([0], [0], color='black', marker=mark[1], linestyle='', label=r'$\omega$'+f'=1.0'),
                Line2D([0], [0], color='black', marker=mark[2], linestyle='', label=r'$\omega$'+f'=1.5'),
            ]
        for i, frequency in enumerate(freq):
            f = [frequency*(2*np.pi*self.vel0[1]), 0.0001]
            k0_ = self.k0(f,1)
            def Vnb(r):
                def func(rp): 
                    k0r= k0_*rp
                    return hankel1(-n, k0r)*(k0r* jvp(n, k0r)-jv(n, k0r))/(rp**3)
                def real_func(x):
                    return np.real(func(x))
                def imag_func(x):
                    return np.imag(func(x))
                real_integral = quad(real_func, r, r2)
                imag_integral = quad(imag_func, r, r2)
                return real_integral[0] + 1j*imag_integral[0]
    
            def Vnc(r):
                def func(rp): 
                    k0r= k0_*rp
                    return hankel1(-n, k0r)*(k0r* h1vp(n, k0r)-hankel1(n, k0r))/(rp**3)
                def real_func(x):
                    return np.real(func(x))
                def imag_func(x):
                    return np.imag(func(x))
                real_integral = quad(real_func, r, r2)
                imag_integral = quad(imag_func, r, r2)
                
                return real_integral[0] + 1j*imag_integral[0]
                
                return imag_integral[0] 
            pos = np.linspace(r1, r2, 50)
        
            vnb = [Vnb(r) for r in pos]
            vnc = [Vnc(r) for r in pos]
            
            ax.plot(pos, np.real(vnb), color = col[0], marker=mark[i], linestyle=lines[i], label=r'$V_n^{(b)}$'+r'$\omega$'+f'={frequency}')
            ax.plot(pos, np.real(vnc), color = col[1], marker=mark[i], linestyle=lines[i], label=r'$V_n^{(c)}$'+r'$\omega$'+f'={frequency}')
            
        handles_types = _legend_handles_types(ax)
        leg1 = ax.legend(handles=handles_types, loc='lower right')
        handles_labels = _legend_handles_labels(ax)
        leg2 = ax.legend(handles=handles_labels, loc='upper right')
        ax.add_artist(leg1)
        plt.xlabel(r'r')
        plt.ylabel(r'Re($V_n$)')
        plt.title(f'Modo ={n}')
        return fig, ax
             
    def coeficiente_dispersion_hollow(self, frequency, n):
        """
        Calcula el coeficiente de dispersión T_n = e_n / d_n para el modo angular n
        y frecuencia dada, usando la deduccion exacta del caso con inclusion y recubrimiento elastico.
        """
        # --- Parámetros físicos y geométricos ---
        a = self.r1  # radio interior (inclusión)
        b = self.r2  # radio exterior (recubrimiento)
        psi=self.psi
        mu0 = self.shear[0]  # módulo de cizalla exterior

        k0_ = self.k0(frequency, 1)  # número de onda exterior
        D = psi* a**2/(b**2-a**2)
        ALPHA =  D * b**2 * n / (2 * np.pi)  # constante psi

        # --- Funciones integrales ---
        def Fnb(r):
            def func(rp): 
                k0r= k0_*rp
                J1 = jv(n, k0r)
                return J1*(k0r* jvp(n, k0r)-J1)/(rp**3)
            def real_func(x):
                val=np.real(func(x))
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return val
            def imag_func(x):
                val=np.imag(func(x))
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return val
            real_integral = quad(real_func, a, r, limit=100, epsabs=1e-6, epsrel=1e-8)
            imag_integral = quad(imag_func, a, r, limit=100, epsabs=1e-6, epsrel=1e-8)
            return real_integral[0] + 1j*imag_integral[0]

        def Fnc(r):
            def func(rp): 
                k0r= k0_*rp
                return jv(n, k0r)*(k0r* h1vp(n, k0r)-hankel1(n, k0r))/(rp**3)
            def real_func(x):
                val=np.real(func(x))
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return val
            def imag_func(x):
                val=np.imag(func(x))
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return val
            real_integral = quad(real_func, a, r, limit=100, epsabs=1e-6, epsrel=1e-8)
            imag_integral = quad(imag_func, a, r, limit=100, epsabs=1e-6, epsrel=1e-8)
            return real_integral[0] + 1j*imag_integral[0]

        def Vnb(r):
            def func(rp): 
                k0r= k0_*rp
                return hankel1(-n, k0r)*(k0r* jvp(n, k0r)-jv(n, k0r))/(rp**3)
            def real_func(x):
                val=np.real(func(x))
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return val
            def imag_func(x):
                val=np.imag(func(x))
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return val
            real_integral = quad(real_func, r, b, limit=100, epsabs=1e-6, epsrel=1e-8)
            imag_integral = quad(imag_func, r, b, limit=100, epsabs=1e-6, epsrel=1e-8)
            return real_integral[0] + 1j*imag_integral[0]

        def Vnc(r):
            def func(rp): 
                k0r= k0_*rp
                return hankel1(-n, k0r)*(k0r* h1vp(n, k0r)-hankel1(n, k0r))/(rp**3)
            def real_func(x):
                val=np.real(func(x))
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return val
            def imag_func(x):
                val=np.imag(func(x))
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return val
            real_integral = quad(real_func, r, b, limit=100, epsabs=1e-6, epsrel=1e-8)
            imag_integral = quad(imag_func, r, b, limit=100, epsabs=1e-6, epsrel=1e-8)
            return real_integral[0] + 1j*imag_integral[0]
        
        # --- Derivadas funciones integrales
        def Fnb_p(r):
            k0r= k0_*r
            J1 = jv(n, k0r)
            return J1*(k0r* jvp(n, k0r)-J1)/(r**3)
        
        def Fnc_p(r):
            k0r= k0_*r
            return jv(n, k0r)*(k0r* h1vp(n, k0r)-hankel1(n, k0r))/(r**3)
        
        def Vnb_p(r):
            k0r= k0_*r
            return -hankel1(n, k0r)*(k0r* jvp(n, k0r)-jv(n, k0r))/(r**3)
        
        def Vnc_p(r):
            k0r= k0_*r
            H1 = hankel1(n, k0r)
            return -H1*(k0r* h1vp(n, k0r)-H1)/(r**3)
        

        # --- Construcción de A1, A2 y sus derivadas ---
        k0a = k0_*a
        k0b = k0_*b
        H1 = hankel1(n, k0a)
        J1 = jv(-n, k0a)
        term1 = k0_*jvp(-n, k0a)
        J2 = jv(n, k0b)
        H2 = hankel1(n, k0b)
        J3 = jv(-n, k0b)
        term2 = k0_*h1vp(n, k0b)
        term3 = k0_*jvp(n, k0b)
        if self.psi==0:
            A1p = k0_*jvp(n, k0a) 
            A2p = k0_*h1vp(n, k0a) 
            A2p = A2p if abs(A2p) > 1e-12 else 1e-12
            return -A1p/A2p
        else:
            A1p = k0_*jvp(n, k0a) + ALPHA*(Fnb_p(a)*H1 + Vnb_p(a)*J1 + term1*Vnb(a))
            A2p = k0_*h1vp(n, k0a) + ALPHA*(Fnc_p(a)*H1 + Vnc_p(a)*J1 + term1*Vnc(a))
            B1 = J2 + ALPHA*Fnb(b)*H2
            B2 = H2 + ALPHA*Fnc(b)*H2
            B1p = term3 + ALPHA*(Fnb_p(b)*H2 + Vnb_p(b)*J3 + term2*Fnb(b))
            B2p = term2 + ALPHA*(Fnc_p(b)*H2 + Vnc_p(b)*J3 + term2*Fnc(b))
        B3 = J2
        B4 = H2
        B3p = term3
        B4p = term2
        M = A2p/A1p
        
        term4 = B2p-M*B1p
        term5 = B2-M*B1

        numerador = B3*(term4)-B3p*(term5)
        den = B4p*(term5)-B4*(term4)
        den = den if abs(den) > 1e-12 else 1e-12
        
        Tn = numerador / den

        return Tn
    
    def coeficiente_dispersion_rigid(self, frequency, n):
        """
        Calcula el coeficiente de dispersión T_n = e_n / d_n para el modo angular n
        y frecuencia dada, usando la deduccion exacta del caso con inclusion y recubrimiento elastico.
        """
        # --- Parámetros físicos y geométricos ---
        a = self.r1  # radio interior (inclusión)
        b = self.r2  # radio exterior (recubrimiento)
        psi=self.psi
        mu0, mu = self.shear  # módulo de cizalla exterior

        k0_ = self.k0(frequency, 1)  # número de onda exterior
        k  = self.ks(frequency, 1)  # número de onda en el recubrimiento
        D = psi* a**2/(b**2-a**2)
        ALPHA =  D * b**2 * n / (2 * np.pi)  # constante psi

        # --- Funciones integrales ---
        def Fnb(r):
            def func(rp): 
                k0p = k0_*rp
                J1 = jv(n, k0p)
                return J1*(k0p* jvp(n, k0p)-J1)/(rp**3)
            def real_func(x):
                return np.real(func(x))
            def imag_func(x):
                return np.imag(func(x))
            real_integral = quad(real_func, a, r)
            imag_integral = quad(imag_func, a, r)
            
            return real_integral[0] + 1j*imag_integral[0]

        def Fnc(r):
            def func(rp): 
                k0p = k0_*rp
                return jv(n, k0p)*(k0p* h1vp(n, k0p)-hankel1(n, k0p))/(rp**3)
            def real_func(x):
                return np.real(func(x))
            def imag_func(x):
                return np.imag(func(x))
            real_integral = quad(real_func, a, r)
            imag_integral = quad(imag_func, a, r)
            
            return real_integral[0] + 1j*imag_integral[0]

        def Vnb(r):
            def func(rp): 
                k0p = k0_*rp
                return hankel1(-n, k0p)*(k0p* jvp(n, k0p)-jv(n, k0p))/(rp**3)
            def real_func(x):
                return np.real(func(x))
            def imag_func(x):
                return np.imag(func(x))
            real_integral = quad(real_func, r, b)
            imag_integral = quad(imag_func, r, b)
            
            return real_integral[0] + 1j*imag_integral[0]

        def Vnc(r):
            def func(rp): 
                k0p = k0_*rp
                return hankel1(-n, k0p)*(k0p* h1vp(n, k0p)-hankel1(n, k0p))/(rp**3)
            def real_func(x):
                return np.real(func(x))
            def imag_func(x):
                return np.imag(func(x))
            real_integral = quad(real_func, r, b)
            imag_integral = quad(imag_func, r, b)
            
            return real_integral[0] + 1j*imag_integral[0]
        
        # --- Derivadas funciones integrales
        def Fnb_p(r):
            k0r = k0_*r
            J1 = jv(n, k0r)
            return J1*(k0r* jvp(n, k0r)-J1)/(r**3)
        
        def Fnc_p(r):
            k0r = k0_*r
            return jv(n, k0r)*(k0r* h1vp(n, k0r)-hankel1(n, k0r))/(r**3)
        
        def Vnb_p(r):
            k0r = k0_*r
            return -hankel1(-n, k0r)*(k0r* jvp(n, k0r)-jv(n, k0r))/(r**3)
        
        def Vnc_p(r):
            k0r = k0_*r
            return -hankel1(-n, k0r)*(k0r* h1vp(n, k0r)-hankel1(n, k0r))/(r**3)
        

        # --- Construcción de A1, A2 y sus derivadas ---
        k0a = k0_*a
        k0b = k0_*b
        J2 = jv(-n, k0a)
        term1 = ALPHA*J2
        H1 = hankel1(n, k0a)
        term2 = k0_*jvp(-n, k0a)
        J3 = jv(n, k0b)
        term3 = k0_*h1vp(n, k0b)
        H2 = hankel1(n, k0b)
        term4 = k0_*jvp(n, k0b)
        alph1 = ALPHA*H2
        J4 = jv(-n, k0b)
        
        A1 = jv(n, k0a) + term1*Vnb(a)
        A2 = H1 + term1*Vnc(a)
        A1p = k0_*jvp(n, k0a) - ALPHA*(Fnb_p(a)*H1 - Vnb_p(a)*J2 - term2*Vnb(a))
        A2p = k0_*h1vp(n, k0a) - ALPHA*(Fnc_p(a)*H1 - Vnc_p(a)*J2 - term2*Vnc(a))
        B1 = J3 - alph1*Fnb(b)
        B2 = H2 - alph1*Fnc(b)
        B1p = term4 - ALPHA*(Fnb_p(b)*H2 - Vnb_p(b)*J4 + term3*Fnb(b))
        B2p = term3 - ALPHA*(Fnc_p(b)*H2 - Vnc_p(b)*J4 + term3*Fnc(b))

        ka = k*a
        term5 = mu0*jv(n, ka)
        term6 = mu*k*jvp(n, ka)
        
        M1 = term6*A1 - term5*A1p
        M2 = term5*A2p - term6*A2
        
        numerador = M2*(J3*B1p - term4*B1)/M1 + J3*B2p-term4*B2
        den = M2*(term3*B1 - H2*B1p)/M1 + term3*B2-H2*B2p
        den = den if abs(den) > 1e-12 else 1e-12

        Tn = numerador / den

        return Tn
 
    def coeficiente_dispersion_elastic(self, frequency, n):
        """
        Calcula el coeficiente de dispersión T_n = e_n / d_n para el modo angular n
        y frecuencia dada, usando la deduccion exacta del caso con inclusion y recubrimiento elastico.
        """
        # --- Parámetros físicos y geométricos ---
        a = self.r1  # radio interior (inclusión)
        b = self.r2  # radio exterior (recubrimiento)
        psi=self.psi
        mu0 = self.shear[0]  # módulo de cizalle exterior
        mu1 = self.shear[1]
        k0_ = self.k0(frequency, 1)  # número de onda exterior
        ks_ = self.ks(frequency, 1)
        D = psi* a**2/(b**2-a**2)
        ALPHA =  D * b**2 * n / (2 * np.pi)  # constante psi

        # --- Funciones integrales ---
        # Nota de optimización: todas estas integrales aparecen SIEMPRE
        # multiplicadas por ALPHA. Cuando ALPHA == 0 (esto ocurre si psi == 0
        # o si n == 0) el resultado es 0 exacto, así que se puede evitar la
        # integración numérica (quad) sin cambiar el valor devuelto.
        def Fnb(r):
            if ALPHA == 0:
                return 0.0
            def func(rp):
                k0r= k0_*rp
                J1 = jv(n, k0r)
                return J1*(k0r* jvp(n, k0r)-J1)/(rp**3)
            def real_func(x):
                return np.real(func(x))
            def imag_func(x):
                return np.imag(func(x))
            real_integral = quad(real_func, a, r)
            imag_integral = quad(imag_func, a, r)
            return real_integral[0] + 1j*imag_integral[0]

        def Fnc(r):
            if ALPHA == 0:
                return 0.0
            def func(rp):
                k0r= k0_*rp
                return jv(n, k0r)*(k0r* h1vp(n, k0r)-hankel1(n, k0r))/(rp**3)
            def real_func(x):
                return np.real(func(x))
            def imag_func(x):
                return np.imag(func(x))
            real_integral = quad(real_func, a, r)
            imag_integral = quad(imag_func, a, r)
            return real_integral[0] + 1j*imag_integral[0]

        def Vnb(r):
            if ALPHA == 0:
                return 0.0
            def func(rp):
                k0r= k0_*rp
                return hankel1(-n, k0r)*(k0r* jvp(n, k0r)-jv(n, k0r))/(rp**3)
            def real_func(x):
                return np.real(func(x))
            def imag_func(x):
                return np.imag(func(x))
            real_integral = quad(real_func, r, b)
            imag_integral = quad(imag_func, r, b)
            return real_integral[0] + 1j*imag_integral[0]

        def Vnc(r):
            if ALPHA == 0:
                return 0.0
            def func(rp):
                k0r= k0_*rp
                return hankel1(-n, k0r)*(k0r* h1vp(n, k0r)-hankel1(n, k0r))/(rp**3)
            def real_func(x):
                return np.real(func(x))
            def imag_func(x):
                return np.imag(func(x))
            real_integral = quad(real_func, r, b)
            imag_integral = quad(imag_func, r, b)
            return real_integral[0] + 1j*imag_integral[0]

        # --- Derivadas funciones integrales
        def Fnb_p(r):
            k0r= k0_*r
            J1 = jv(n, k0r)
            return J1*(k0r* jvp(n, k0r)-J1)/(r**3)
        
        def Fnc_p(r):
            k0r= k0_*r
            return jv(n, k0r)*(k0r* h1vp(n, k0r)-hankel1(n, k0r))/(r**3)
        
        def Vnb_p(r):
            k0r= k0_*r
            return -hankel1(n, k0r)*(k0r* jvp(n, k0r)-jv(n, k0r))/(r**3)
        
        def Vnc_p(r):
            k0r= k0_*r
            H1 = hankel1(n, k0r)
            return -H1*(k0r* h1vp(n, k0r)-H1)/(r**3)
        

        # --- Construcción de A1, A2 y sus derivadas ---
        k0a = k0_*a
        k0b = k0_*b
        H1 = hankel1(n, k0a)
        J1 = jv(-n, k0a)
        term1 = k0_*jvp(-n, k0a)
        J2 = jv(n, k0b)
        H2 = hankel1(n, k0b)
        J3 = jv(-n, k0b)
        term2 = k0_*h1vp(n, k0b)
        term3 = k0_*jvp(n, k0b)
        
        A1p = k0_*jvp(n, k0a) + ALPHA*(Fnb_p(a)*H1 + Vnb_p(a)*J1 + term1*Vnb(a))
        A2p = k0_*h1vp(n, k0a) + ALPHA*(Fnc_p(a)*H1 + Vnc_p(a)*J1 + term1*Vnc(a))
        B1 = J2 + ALPHA*Fnb(b)*H2
        B2 = H2 + ALPHA*Fnc(b)*H2
        B1p = term3 + ALPHA*(Fnb_p(b)*H2 + Vnb_p(b)*J3 + term2*Fnb(b))
        B2p = term2 + ALPHA*(Fnc_p(b)*H2 + Vnc_p(b)*J3 + term2*Fnc(b))
        B3 = J2
        B4 = H2
        B3p = term3
        B4p = term2
        
        ksa = ks_*a
        A1 = jv(n, k0a) + ALPHA*Vnb(a)*J1
        A2 = hankel1(n, k0a) + ALPHA*Vnc(a)*J1
        Jpa = jvp(n,ksa)
        M1 = mu1*ks_*Jpa*A1-mu0*jv(n,ksa)*A1p
        M2 = mu1*ks_*Jpa*A2-mu0*jv(n,ksa)*A2p
        M = M1/M2
        
        term4 = B1-M*B2
        term5 = B1p-M*B2p

        numerador = B3p*(term4)-J2*(term5)
        den = H2*(term5)-B4p*(term4)
        den = den if abs(den) > 1e-12 else 1e-12
        
        Tn = numerador / den

        return Tn

    def zeros_longitudinal_fullgrid(self, C_l0, ventanas_por_unidad=100, 
                                            w_norm_max=1.25, buscar_todas=False,
                                            use_external_output=False, out_freq_dir=None,
                                            out_fig_dir=None, w_norm_min=1e-3,eta=1e-2,use_adaptive_g0=False,
                                            g0_tol=1e-6,
                                            g0_n_suma_max=300,
                                            g0_stable_passes=2,
                                            g0_verbose=False,
                                            **g0_extra):
        """
        Calcula la estructura de bandas usando un método híbrido optimizado.
        Combina tres estrategias para un equilibrio óptimo entre velocidad y fiabilidad:
        1. Continuación: Intervalo centrado en soluciones previas (rápido y mejorado).
        2. Cambio de Signo: Busca intervalos con raíces (robusto).
        3. Búsqueda Ciega: Último recurso si todo lo anterior falla.
        """
        if use_external_output:
            if out_freq_dir is not None:
                self.frecfolder = str(out_freq_dir)
            if out_fig_dir is not None:
                self.foldername = str(out_fig_dir)
        
        # En todos los casos, asegúrate de que existan
        Path(self.frecfolder).mkdir(parents=True, exist_ok=True)
        Path(self.foldername).mkdir(parents=True, exist_ok=True)
        print("\n Iniciando método híbrido y robusto de búsqueda para todos los k ...")
        start_time = time.time()

        # --- Parámetros ---
        tolerancia_raices = self.sol_tol
        tolerancia_imaginaria = self.imag_tol

        punto_K = 2 * np.pi / (3 * self.a)

        # --- Preparación de k-points ---
        if self.lattice == 'hx' and not np.any(np.isclose(self.k, punto_K, atol=1e-6)):
            self.k = np.append(self.k, punto_K)
        self.k = np.sort(self.k)
        self.nk = len(self.k)

        # --- Inicialización ---
        nmax = 5 if buscar_todas else self.nbands
        frec = np.full((self.nk, nmax, 2), np.nan)
        # No utilizaremos soluciones_anteriores para la continuidad.
        # Al prescindir de las semillas de iteraciones previas, cada k
        # se procesa de forma independiente basándose únicamente en
        # cambios de signo.

        # --- Funciones auxiliares ---
        def eval_det(omega, k_val):
            try:
                return self.Det_longitudinal([omega, eta], k_val, self.cut, use_adaptive_g0, g0_tol, g0_n_suma_max, g0_stable_passes, g0_verbose)
            except:
                return [np.nan]

        def resolver_con_fsolve(semilla_w, k_val):
            try:
                sol, info, ier, _ = fsolve(
                    self.Det_longitudinal, [semilla_w, eta],
                    args=(k_val, self.cut, use_adaptive_g0, g0_tol, g0_n_suma_max, g0_stable_passes, g0_verbose), xtol=tolerancia_raices,
                    epsfcn=self.epsfcn,
                    full_output=True
                )
                w_real, w_imag = sol
                w_norm = (w_real * self.a) / (2 * np.pi * C_l0)
                if ier == 1 and abs(w_imag) < tolerancia_imaginaria and 0 < w_norm < w_norm_max:
                    return (w_real, w_imag)
            except:
                return None
            return None

        def buscar_cambios_signo(w_min, w_max, k_val, n_puntos=50, max_nivel=2):
            """
            Busca intervalos donde el determinante cambia de signo.

            Se realiza un muestreo inicial uniforme y se detectan los
            intervalos donde `Re(det)` cambia de signo.  Cada
            intervalo se subdivide recursivamente hasta un máximo de
            `max_nivel` niveles para capturar raíces cercanas.
            """
            # Muestreo inicial
            w_array = np.linspace(w_min, w_max, n_puntos)
            det_vals = [eval_det(w, k_val)[0] for w in w_array]
            intervalos = []

            # Definir función interna para subdividir un intervalo si hay más de un cambio
            def subdividir(a, b, nivel):
                c = 0.5 * (a + b)
                try:
                    f_a = eval_det(a, k_val)[0]
                    f_c = eval_det(c, k_val)[0]
                    f_b = eval_det(b, k_val)[0]
                except:
                    return [(a, b)]
                subints = []
                if nivel >= max_nivel:
                    return [(a, b)]
                if not (np.isnan(f_a) or np.isnan(f_c)) and f_a * f_c < 0:
                    subints += subdividir(a, c, nivel + 1)
                if not (np.isnan(f_c) or np.isnan(f_b)) and f_c * f_b < 0:
                    subints += subdividir(c, b, nivel + 1)
                if not subints:
                    subints = [(a, b)]
                return subints

            for i in range(len(det_vals) - 1):
                fa, fb = det_vals[i], det_vals[i + 1]
                if np.isnan(fa) or np.isnan(fb):
                    continue
                if fa * fb < 0:
                    a, b = w_array[i], w_array[i + 1]
                    intervalos += subdividir(a, b, 0)
            return intervalos

        def resolver_en_intervalo_continuacion(w_centro, k_val, ancho_rel=0.025, n_semillas=51):
            """
            Busca raíces alrededor de una solución anterior usando un intervalo relativo.
            """
            soluciones_locales = []
            ancho = ancho_rel * w_centro
            w_array = np.linspace(w_centro - ancho, w_centro + ancho, n_semillas)

            for w in w_array:
                try:
                    det1 = eval_det(w - 1e-5, k_val)[0]
                    det2 = eval_det(w + 1e-5, k_val)[0]
                    cambio_signo = det1 * det2 < 0
                except:
                    cambio_signo = False

                if cambio_signo:
                    resultado = resolver_con_fsolve(w, k_val)
                    if resultado and not any(np.isclose(resultado[0], s[0], rtol=1e-2) for s in soluciones_locales):
                        soluciones_locales.append(resultado)
            return soluciones_locales

        # --- Búsqueda principal ---
        barra = tqdm(range(self.nk), desc="Búsqueda por k", dynamic_ncols=True)
        for idx_k in barra:
            k_val = self.k[idx_k]
            soluciones = []
            es_punto_K = self.lattice == 'hx' and k_val == punto_K

            # Sólo usamos la estrategia de cambio de signo para detectar intervalos con raíces.
            minimi = w_norm_min
            w_min = 2 * np.pi * C_l0 * minimi / self.a
            w_max = 2 * np.pi * C_l0 * w_norm_max / self.a
            n_puntos_busqueda = int(ventanas_por_unidad * (w_norm_max - minimi))
            # Buscar intervalos donde el determinante cambia de signo
            intervalos = buscar_cambios_signo(w_min, w_max, k_val, n_puntos=n_puntos_busqueda)

            for w1, w2 in intervalos:
                sol = resolver_con_fsolve(0.5 * (w1 + w2), k_val)
                if sol:
                    w_real = sol[0]
                    # Normalización de la frecuencia real
                    w_norm_local = (w_real * self.a) / (2 * np.pi * C_l0)
                    # Ignorar raíces con frecuencia real muy pequeña
                    if w_norm_local < 1e-3:
                        continue
                    # Eliminar duplicados: si ya existe una raíz muy cercana, no la añadimos
                    if not any(np.isclose(sol[0], s[0], rtol=1e-4) for s in soluciones):
                        soluciones.append(sol)
            
            # Ordenar soluciones por frecuencia real y recortar si se desean menos bandas
            soluciones = sorted(soluciones, key=lambda x: x[0])
            if not buscar_todas:
                soluciones = soluciones[:self.nbands]

            for b, s in enumerate(soluciones):
                if b < nmax:
                    frec[idx_k, b, :] = s

            # Corrección en el punto K (cono de Dirac)
            if es_punto_K and self.nbands >= 2:
                w1 = frec[idx_k, 0, 0]
                w2 = frec[idx_k, 1, 0]
                if not np.isnan(w1) and not np.isnan(w2) and abs(w2 - w1) > 7:
                    print(f"Cono de Dirac en K: Δf = {abs(w2 - w1):.2f} Hz -> Duplicando banda 1")
                    frec[idx_k, 1, :] = frec[idx_k, 0, :]

            # Guardar resultados de este k en disco
            np.savetxt(os.path.join(self.frecfolder, f"frecuencias{self.k[idx_k]:.4f}.txt"), frec[idx_k])

        self.omega_longitudinal = frec
        # Graficar las bandas calculadas
        self.graficar_bandas_grid()

        # Guardar resultados completos en un archivo CSV
        # Cada fila del CSV contendrá: valor de k, índice de banda (1-based),
        # frecuencia real y frecuencia imaginaria.  Se ignoran entradas NaN.
        try:
            rows = []
            for idx_k, k_val in enumerate(self.k):
                for band_idx in range(nmax):
                    w_real, w_imag = frec[idx_k, band_idx, :]
                    if not np.isnan(w_real):
                        rows.append({
                            'k': float(k_val),
                            'band': int(band_idx + 1),
                            'w_real': float(w_real),
                            'w_imag': float(w_imag),
                        })
            if rows:
                df = pd.DataFrame(rows)
                csv_path = os.path.join(self.frecfolder, 'bandas_longitudinales.csv')
                df.to_csv(csv_path, index=False)
                print(f"Resultados guardados en CSV: {csv_path}")
        except Exception as e:
            # En caso de error al guardar, se informa pero no se interrumpe
            print(f"Advertencia: no se pudo guardar el CSV ({e})")

        print(f"Completado en {time.time() - start_time:.2f} s")

    
    def buscar_bandas_vecindad(self, C_l0, k_inicial, k_final, w_norm_inicial, w_norm_final, ventanas_por_unidad=20, semillas_por_ventana=10):
        print("**** Iniciando búsqueda localizada de bandas ****")
        start_time = time.time()

        dw = 1 / ventanas_por_unidad
        tolerancia_raices = 1e-1
        profundidad_max = min(5, int(np.log2(ventanas_por_unidad)))

        k_array_completo = np.array(self.k)
        k_filtrados = [k for k in k_array_completo if k_inicial <= k <= k_final]
        
        imag_tol = 1e-1
        
        punto_K = 2 * np.pi / (3 * self.a) 
        if not any(np.isclose(punto_K, kf, atol=1e-6) for kf in k_filtrados):
            if k_inicial <= punto_K <= k_final:
                print(" El punto K no estaba incluido. Se agregará manualmente al conjunto de k.")
                k_filtrados.append(punto_K)
            else:
                print(" El intervalo especificado no contiene al punto M. No se incluirá.")

        k_filtrados = sorted(k_filtrados)
        self.nk = len(k_filtrados)

        frec = np.full((self.nk, self.nbands, 2), np.nan)
        omega_anterior = None
        tiempos_por_k = []

        w_norm_max = w_norm_final
        w_norm_min = w_norm_inicial

        kstr1 = f"{k_inicial:.3f}".replace(".", "p")
        kstr2 = f"{k_final:.3f}".replace(".", "p")
        foldername = os.path.join(self.foldername, f"vecindad_k_{kstr1}_a_{kstr2}, para x="+str(self.filling))
        frecfolder = os.path.join(foldername, "frecuencias")
        self.create_folder(foldername)
        self.create_folder(frecfolder)

        barra = tqdm(k_filtrados, desc="Calculando bandas", dynamic_ncols=True, leave=True)

        for idx_k, k_val in enumerate(barra):
            inicio_k = time.time()
            soluciones_validas = []
            def buscar_raices_recursivas(w1, w2, profundidad):
                if profundidad > profundidad_max or (w2 - w1) < 1e-3:
                    return
                semillas = []
                if omega_anterior is not None:
                    for w_prev in omega_anterior:
                        if w1 < w_prev < w2:
                            semillas.extend([
                                w_prev * (1 - 1e-3),
                                w_prev,
                                w_prev * (1 + 1e-3)])
                semillas.extend(np.linspace(w1, w2, semillas_por_ventana))
                semillas = sorted(set(semillas))
                for omega_re in semillas:
                    D = self.determinant_longitudinal([omega_re, 0.1], k_val, self.cut)
                    if np.linalg.norm(D) > 1e-2:
                        continue
                    sol, info, ier, _ = fsolve(
                        self.determinant_longitudinal, [omega_re, 0.1], args=(k_val, self.cut),
                        xtol=1e-12, maxfev=2000, full_output=True)
                    re_w, im_w = sol
                    if ier == 1 and abs(im_w) < imag_tol:
                        if 0 < (re_w * self.a) / (2 * np.pi * C_l0) < w_norm_max:
                            if not any(np.isclose(re_w, s[0], atol=tolerancia_raices) for s in soluciones_validas):
                                soluciones_validas.append((re_w, im_w))
                if len(soluciones_validas) < self.nbands:
                    mid = 0.5 * (w1 + w2)
                    buscar_raices_recursivas(w1, mid, profundidad + 1)
                    buscar_raices_recursivas(mid, w2, profundidad + 1)
            j_min = int(w_norm_min * ventanas_por_unidad)
            j_max = int(w_norm_max * ventanas_por_unidad)
            for j in range(j_min, j_max):
                w1_norm = j * dw
                w2_norm = (j + 1) * dw
                w1 = 2 * np.pi * C_l0 * w1_norm / self.a
                w2 = 2 * np.pi * C_l0 * w2_norm / self.a
                buscar_raices_recursivas(w1, w2, 0)
                if len(soluciones_validas) >= self.nbands:
                    break
            soluciones_validas = sorted(set(soluciones_validas), key=lambda x: x[0])
            while len(soluciones_validas) < self.nbands:
                soluciones_validas.append((np.nan, np.nan))
            frec[idx_k, :, 0] = [s[0] for s in soluciones_validas[:self.nbands]]
            frec[idx_k, :, 1] = [s[1] for s in soluciones_validas[:self.nbands]]
            omega_anterior = [s[0] for s in soluciones_validas[:self.nbands] if not np.isnan(s[0])]
            nombre_archivo = os.path.join(frecfolder, f"frecuencias{k_val:.4f}.txt")
            np.savetxt(nombre_archivo, frec[idx_k])
            tiempos_por_k.append(time.time() - inicio_k)

        # Reordenamiento y verificación
        for i in range(1, self.nk):
            for b in range(self.nbands):
                w_anterior = frec[i - 1, b, 0]
                if np.isnan(w_anterior):
                    continue
                distancias = np.abs(frec[i, :, 0] - w_anterior)
                if np.all(np.isnan(distancias)):
                    continue
                idx_min = np.nanargmin(distancias)
                if idx_min != b:
                    frec[i, [b, idx_min], :] = frec[i, [idx_min, b], :]
                    
        self.omega_longitudinal = frec

        # Graficar bandas
        PLOT_VECINDAD = False
        if PLOT_VECINDAD:
            fig, ax = plt.subplots()
            for b in range(frec.shape[1]):
                ax.plot(k_filtrados, frec[:, b, 0] / (2 * np.pi), label=f"Banda {b + 1}")
            ax.axvline(x=punto_K, color='k', linestyle='--', linewidth=0.8)
            ax.set_xlabel("k")
            ax.set_ylabel("Frecuencia [Hz]")
            ax.set_title("Bandas en vecindad localizada")
            plt.tight_layout()
            plt.savefig(os.path.join(foldername, "bandas_vecindad.png"), dpi=300)
            plt.close()
        
        print("\nTiempo total: {:.2f} s".format(time.time() - start_time))
        print("Tiempo medio por k: {:.2f} s".format(np.mean(tiempos_por_k)))
        print("Máximo: {:.2f} s | Mínimo: {:.2f} s".format(np.max(tiempos_por_k), np.min(tiempos_por_k)))

    def create_folder(self, foldername):
        createFolder(foldername)        

    def graficar_bandas_grid(self, ylim =[None, None], args=[None, None], hlines=[None,None], title=True):
        npsi, omega = args
        if self.omega_longitudinal is None:
            raise ValueError("No se han calculado las bandas longitudinales")
        fig, ax = plt.subplots()
        k = self.k
        w = self.omega_longitudinal[:, :, 0] * self.a / (2 * np.pi * self.vel0[1])
        if npsi==2:
            w2 = omega * self.a / (2 * np.pi * self.vel0[1])
        for i in range(w.shape[1]):
            banda = w[:, i]
            ax.plot(k, banda, '.', label=f"Banda {i+1}", color='black')  # Puntos pequeños como en el original
            if npsi==2:
                banda2 = w2[:,i]
                ax.plot(k, banda2, '.', color='red')
        a = self.a
        if self.lattice == 'sq':
            posiciones_k = [0, np.pi/a, 2*np.pi/a, 3*np.pi/a]
            etiquetas_k = ['X', r'$\Gamma$', 'M', 'X']
        elif self.lattice == 'hx':
            posiciones_k = [0, 2*np.pi/(3*a), 2*np.pi/a, 2*np.pi*(1 + 1/np.sqrt(3))/a]
            etiquetas_k = ['M', 'K', r'$\Gamma$', 'M']
        else:
            posiciones_k = k
            etiquetas_k = [f"{val:.2f}" for val in k]
        if hlines[0]!=None:
            plt.axhline(hlines[0], color='black', linestyle='--')
            if hlines[1]!=None:
                plt.axhline(hlines[1], color='black', linestyle='--')
                plt.fill_between(k, hlines[0], hlines[1], color='gray', alpha=0.3, hatch='//', edgecolor='k', linewidth=0.5)
            
        ax.set_xticks(posiciones_k)
        ax.set_xticklabels(etiquetas_k)
        ax.set_ylabel(r'$\omega a / 2\pi C_{t0}$', fontsize=15)
        ax.set_xlabel(r'$ka$', fontsize=15)
        if title:
            Title = r'$\psi=$' + f'{self.psi}'
            ax.set_title(Title, fontsize=15)
        plt.tick_params(axis='both', labelsize=15)
        if ylim[0]==None and ylim[1]==None: 
            plt.tight_layout()
        else:
            plt.ylim(ylim)          
        nombre_grafico = os.path.join(self.frecfolder, "bandas_longitudinales_grid.png")
        plt.savefig(nombre_grafico, dpi=300)
        plt.show()
        plt.close(fig)
    
    def cargar_frecuencias_grid(self):
        frec = np.full((self.nk, self.nbands, 2), np.nan)
        for idx_k, k_val in enumerate(self.k):
            nombre_archivo = os.path.join(self.frecfolder, f"frecuencias{k_val:.4f}.txt")
            if os.path.exists(nombre_archivo):
                try:
                    datos = np.loadtxt(nombre_archivo)
                    if datos.shape == (self.nbands, 2):
                        frec[idx_k] = datos
                except:
                    print(f"Error al leer {nombre_archivo}")
        self.omega_longitudinal = frec

    # ======= Sigma-min helpers (inside class Red) =======
    def _w_from_norm(self, w_norm, C_l0):
        import numpy as np
        return (float(w_norm) * 2.0 * np.pi * float(C_l0)) / float(self.a)

    def _pick_local_minima(self, x, y, max_peaks):
        import numpy as np
        x = np.asarray(x); y = np.asarray(y)
        idx = []
        for i in range(1, len(x)-1):
            if np.isfinite(y[i]) and y[i] < y[i-1] and y[i] < y[i+1]:
                idx.append(i)
        idx = sorted(idx, key=lambda i: y[i])
        return [(x[i], y[i]) for i in idx[:max_peaks]]

    def _bracket_min(self, k, w0, C_l0, halfwidth, w_norm_max):
        s0 = self._sigma_min_norm(k, w0, C_l0)
        for fmul in (1.0, 1.5, 2.0, 3.0):
            a = max(1e-4, w0 - fmul*halfwidth)
            b = min(w_norm_max, w0 + fmul*halfwidth)
            sa = self._sigma_min_norm(k, a, C_l0)
            sb = self._sigma_min_norm(k, b, C_l0)
            if sa > s0 and sb > s0:
                return a, b
        return None

    def _minimize_brent(self, k, a, b, C_l0, tol=1e-4):
        import numpy as np
        gr = (np.sqrt(5) - 1) / 2
        x1 = b - gr*(b - a)
        x2 = a + gr*(b - a)
        f1 = self._sigma_min_norm(k, x1, C_l0)
        f2 = self._sigma_min_norm(k, x2, C_l0)
        neval = 2
        while abs(b - a) > tol and neval < int(getattr(self,'max_eval_per_refine',20)):
            if f1 > f2:
                a = x1; x1 = x2; f1 = f2
                x2 = a + gr*(b - a); f2 = self._sigma_min_norm(k, x2, C_l0)
            else:
                b = x2; x2 = x1; f2 = f1
                x1 = b - gr*(b - a); f1 = self._sigma_min_norm(k, x1, C_l0)
            neval += 1
        x_star = (a + b) / 2
        f_star = self._sigma_min_norm(k, x_star, C_l0)
        return x_star, f_star

    def _resolve_k_subset(self, k_indices=None, k_values=None, tol=1e-9):
        """
        Devuelve (k_full, idx) donde:
          - k_full: arreglo completo de k tomado de self.kpath (si existe) o self.k
          - idx: índices (ordenados) del subconjunto a evaluar
        Acepta k_indices (índices) y/o k_values (valores). Si ambos se entregan, usa la intersección.
        """
        import numpy as np
    
        k_full = getattr(self, 'kpath', None)
        if k_full is None or len(k_full) == 0:
            k_full = getattr(self, 'k', None)
        if k_full is None or len(k_full) == 0:
            # Si no hay grilla predefinida, la construimos como siempre
            k_full = np.linspace(self.k_init, self.k_end, int(self.nk))
            self.k = k_full
    
        k_full = np.asarray(k_full, float)
        all_idx = set(range(len(k_full)))
    
        # Por índices
        if k_indices is not None:
            all_idx &= set(int(i) for i in k_indices)
    
        # Por valores
        if k_values is not None:
            vals = np.asarray(list(k_values), float)
            hit = set()
            for kv in vals:
                j = int(np.argmin(np.abs(k_full - kv)))
                if abs(k_full[j] - kv) <= tol:
                    hit.add(j)
            all_idx = hit if (k_indices is None) else (all_idx & hit)
    
        if not all_idx:
            raise ValueError("El subconjunto de k quedó vacío (revisa índices/valores/tolerancia).")
    
        idx_sorted = np.array(sorted(all_idx), dtype=int)
        return k_full, idx_sorted

    def zeros_longitudinal_fullgrid_subset(self,
                                           C_l0,
                                           w_norm_max=1.0,
                                           ventanas_por_unidad=100,
                                           subdir="fullgrid",
                                           progress="tqdm",
                                           k_indices=None,
                                           k_values=None,
                                           use_adaptive_g0=False,
                                           g0_tol=1e-6,
                                           g0_n_suma_max=300,
                                           g0_stable_passes=2,
                                           g0_verbose=False,
                                           **g0_extra):
        """
        Ejecuta zeros_longitudinal_fullgrid SOLO sobre un subconjunto de k (por índice o por valor).
        No toca el método original. Escribe/actualiza solo los archivos de esos k.
        """
        import time
    
        # Resolver subconjunto
        k_full, k_idx = self._resolve_k_subset(k_indices=k_indices, k_values=k_values)
        print(f"[fullgrid-subset] k seleccionados: {len(k_idx)}/{len(k_full)}")
    
        # Guardar estado y preparar estado temporal
        prev_k      = getattr(self, 'k', None)
        prev_nk     = getattr(self, 'nk', None)
        prev_kinit  = getattr(self, 'k_init', None)
        prev_kend   = getattr(self, 'k_end', None)
    
        try:
            sub_k = k_full[k_idx]
            self.k      = sub_k
            self.nk     = int(len(sub_k))
            self.k_init = float(sub_k[0])
            self.k_end  = float(sub_k[-1])
    
            t0 = time.time()
            F_small = self.zeros_longitudinal_fullgrid(C_l0=C_l0,
                                                       w_norm_max=w_norm_max,
                                                       ventanas_por_unidad=ventanas_por_unidad,
                                                       use_adaptive_g0=use_adaptive_g0,       # <-- activa cache + incremental
                                                       g0_tol=1e-6,
                                                       g0_n_suma_max=400,          # subimos tope
                                                       g0_stable_passes=2,
                                                       g0_verbose=g0_verbose)
            dt = time.time() - t0
            print(f"[fullgrid-subset] listo en {dt:.2f}s")
    
            # No reconstruimos tensor completo aquí (fullgrid suele apoyarse en archivos)
            self.k_subset_idx = k_idx
            return F_small, k_idx
    
        finally:
            self.k      = prev_k
            self.nk     = prev_nk
            self.k_init = prev_kinit
            self.k_end  = prev_kend

    def _kgrid_full(self):
        import numpy as np
        k_full = getattr(self, 'kpath', None)
        if k_full is None or len(k_full) == 0:
            k_full = getattr(self, 'k', None)
        if k_full is None or len(k_full) == 0:
            raise ValueError("No hay kpath/k definidos.")
        return np.asarray(k_full, float)
    
    def _ensure_tensor(self, attr, nk_full, nb):
        import numpy as np
        arr = getattr(self, attr, None)
        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[:2] == (nk_full, nb) and arr.shape[2] == 2:
            return arr
        new_arr = np.full((nk_full, nb, 2), np.nan, dtype=float)
        setattr(self, attr, new_arr)
        return new_arr
    
    def _ensure_masks(self, nk_full, nb):
        import numpy as np
        if (self._mask_sigma is None) or (self._mask_sigma.shape != (nk_full, nb)):
            self._mask_sigma = np.zeros((nk_full, nb), dtype=bool)
        if (self._mask_fullgrid is None) or (self._mask_fullgrid.shape != (nk_full, nb)):
            self._mask_fullgrid = np.zeros((nk_full, nb), dtype=bool)
    
    def _load_frec_file(self, base_dir, k_val):
        import numpy as np, os
        fname = os.path.join(base_dir, f"frecuencias{k_val:.4f}.txt")
        if not os.path.exists(fname):
            return None, fname
        arr = np.loadtxt(fname); arr = np.atleast_2d(arr)
        return arr, fname
    
    def _save_frec_file(self, fname, arr2d):
        import numpy as np, os
        if arr2d is None or arr2d.size == 0 or arr2d.shape[0] == 0:
            if os.path.exists(fname):
                os.remove(fname)
            return
        np.savetxt(fname, arr2d, fmt="%.10e")

    def delete_point(self, i, n, mode="sigma", subdir=None, sync_disk=False,
                     preview=True, highlight_color="#E74C3C", marker="x", markersize=60, alpha=0.9, ylim=[None, None]):
        """
        Borra ω(k[i])[n] en RAM (NaN + máscara) y opcionalmente en disco (sync_disk=True).
        - mode: 'sigma' | 'fullgrid'
        - subdir: carpeta de datos; si None, infiere por mode.
        - preview: si True, muestra el plot en consola (plt.show()) con el punto resaltado.
        AVISA si el punto elegido ya es NaN en RAM.
        """
        import numpy as np, os
        from pathlib import Path
    
        if mode not in ("sigma", "fullgrid"):
            raise ValueError("mode debe ser 'sigma' o 'fullgrid'.")
    
        k_full = self._kgrid_full()
        nk_full = len(k_full)
        nb = int(self.nbands)
        if not (0 <= i < nk_full) or not (0 <= n < nb):
            raise IndexError("Índices i/n fuera de rango.")
    
        arr_name = 'omega_longitudinal_sigma' if mode == 'sigma' else 'omega_longitudinal'
        F = self._ensure_tensor(arr_name, nk_full, nb)
        self._ensure_masks(nk_full, nb)
    
        # --- AVISO si ya es NaN en RAM ---
        curr = F[i, n, :].copy()
        if (not np.isfinite(curr[0])) and (not np.isfinite(curr[1])):
            print(f"[delete {mode}] AVISO: k={k_full[i]:.4f} (i={i}) band={n} ya es NaN en RAM.")
    
        # backup previo
        old_val = curr
        backup_entry = {
            "mode": mode, "i": int(i), "n": int(n),
            "old_val": old_val,
            "k_val": float(k_full[i]),
            "sync_disk": bool(sync_disk),
            "file": None, "file_before": None,
            "subdir": subdir or ("sigma_min" if mode == "sigma" else "fullgrid")
        }
    
        if preview:
            try:
                self._plot_bands_highlight(mode=mode, highlight=(i, n),
                                           color=highlight_color, marker=marker,
                                           size=markersize, alpha=alpha,
                                           show=True, save=False, ylim=ylim)
            except Exception as _e:
                print(f"[preview] no se pudo mostrar preview: {_e}")
    
        
        # RAM: NaN + máscara
        F[i, n, :] = np.nan
        if mode == "sigma":
            self._mask_sigma[i, n] = True
        else:
            self._mask_fullgrid[i, n] = True
    
        self._deleted_stack.append(backup_entry)
        
        # disco (opcional)
        if sync_disk:
            base_dir = self.frecfolder
            arr_file, fname = self._load_frec_file(base_dir, k_full[i])
            backup_entry["file"] = fname
            if arr_file is not None:
                backup_entry["file_before"] = arr_file.copy()
                if 0 <= n < arr_file.shape[0]:
                    arr_new = np.delete(arr_file, n, axis=0)
                else:
                    arr_new = arr_file
                self._save_frec_file(fname, arr_new)
        
        # preview en consola
        print(f"[delete {mode}] k={k_full[i]:.4f} (i={i}) band={n} -> removed (disk={sync_disk})")
        
        return True

    def restore_deleted(self, mode="auto", i=None, n=None, preview=True,
                        highlight_color="#2E86C1", marker="o", markersize=60, alpha=0.9):
        """
        Restaura el último borrado o el que coincida con (mode, i, n).
        Muestra preview en consola si corresponde.
        """
        import os, numpy as np
        if not self._deleted_stack:
            print("[restore] no hay entradas en el historial.")
            return False
    
        # elegir entrada
        idx = None
        if (i is None) and (n is None) and (mode == "auto"):
            idx = -1
        else:
            for j in range(len(self._deleted_stack)-1, -1, -1):
                ent = self._deleted_stack[j]
                if (mode in ("auto", ent["mode"])) and (i is None or ent["i"] == i) and (n is None or ent["n"] == n):
                    idx = j; break
        if idx is None:
            print("[restore] no encontré una entrada que coincida.")
            return False
    
        ent = self._deleted_stack.pop(idx)
        k_val = ent["k_val"]; i = ent["i"]; n = ent["n"]; mode = ent["mode"]
    
        k_full = self._kgrid_full()
        nk_full = len(k_full)
        nb = int(self.nbands)
    
        arr_name = 'omega_longitudinal_sigma' if mode == 'sigma' else 'omega_longitudinal'
        F = self._ensure_tensor(arr_name, nk_full, nb)
        self._ensure_masks(nk_full, nb)
    
        # RAM
        F[i, n, :] = ent["old_val"]
        if mode == "sigma":
            self._mask_sigma[i, n] = False
        else:
            self._mask_fullgrid[i, n] = False
    
        # disco
        if ent["sync_disk"] and ent["file"] is not None and ent["file_before"] is not None:
            self._save_frec_file(ent["file"], ent["file_before"])
    
        print(f"[restore] {mode} k={k_val:.4f} (i={i}) band={n} -> restored (disk={ent['sync_disk']})")
    
        # preview en consola
        if preview:
            try:
                self._plot_bands_highlight(mode=mode, highlight=(i, n),
                                           color=highlight_color, marker=marker,
                                           size=markersize, alpha=alpha,
                                           show=True, save=False)
            except Exception as _e:
                print(f"[preview] no se pudo mostrar preview: {_e}")
    
        return True
        
    def _plot_bands_highlight(self, mode="sigma", highlight=None,
                              color="#E74C3C", marker="x", size=60, alpha=0.9,
                              show=True, save=False, fname=None, ylim=[None,None]):
        """
        Dibuja el scatter de bandas actuales y resalta 'highlight=(i,n)'.
        - show=True: muestra en consola (plt.show()).
        - save=False: NO guarda a archivo por defecto.
        AVISA si el punto a resaltar es NaN.
        """
        import numpy as np, os
        import matplotlib.pyplot as plt
    
        if mode not in ("sigma", "fullgrid"):
            raise ValueError("mode debe ser 'sigma' o 'fullgrid'.")
    
        k_full = self._kgrid_full()
        nk_full = len(k_full); nb = int(self.nbands)
    
        arr_name = 'omega_longitudinal_sigma' if mode == 'sigma' else 'omega_longitudinal'
        F = self._ensure_tensor(arr_name, nk_full, nb)
    
        wr = F[:, :, 0]
        wi = F[:, :, 1]
        a_val = float(self.a); Cl0 = float(self.vel0[1])
        wnorm = (wr * a_val) / (2.0*np.pi*Cl0)
    
        mask = np.isfinite(wnorm)
        xs, ys = [], []
        for ii in range(nk_full):
            for bb in range(nb):
                if mask[ii, bb]:
                    xs.append(float(k_full[ii]))
                    ys.append(float(wnorm[ii, bb]))
    
        fig = plt.figure(figsize=(8.0, 4.0))
        ax = fig.add_subplot(111)
        if xs:
            ax.scatter(xs, ys, s=16, c='black', alpha=0.85)
    
        # highlight
        if highlight is not None:
            i, n = highlight
            if 0 <= i < nk_full and 0 <= n < nb and np.isfinite(wnorm[i, n]):
                ax.scatter([float(k_full[i])], [float(wnorm[i, n])],
                           s=size, c=color, marker=marker, edgecolors='none', alpha=alpha)
            else:
                print(f"[preview] AVISO: el punto a resaltar (i={i}, n={n}) es NaN o está fuera de rango; no se puede marcar.")
    
        ax.set_xlabel("k (camino)")
        ax.set_ylabel(r"$\bar{\omega}=\omega a / (2\pi C_{l0})$")
        ax.grid(True, ls="--", alpha=0.35)
        if ylim[0] != None:
            plt.ylim(ylim)
        ax.set_title(f"{'σmin' if mode=='sigma' else 'fullgrid'} — preview edición")
    
        if save:
            # Solo si tú lo activas; por defecto no se guarda
            from pathlib import Path
            outdir = Path(self.foldername or self.frecfolder or ".")
            outdir.mkdir(parents=True, exist_ok=True)
            if fname is None and highlight is not None:
                i, n = highlight
                fname = f"preview_{mode}_i{i}_n{n}.png"
            elif fname is None:
                fname = f"preview_{mode}.png"
            path = outdir / fname
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            print(f"[preview] guardado en: {path}")
    
        if show:
            # Muestra en consola (notebook o ventana interactiva según backend)
            plt.show()
    
        plt.close(fig)
        return True

    def order_bands_by_continuity_global(self,
        mode="fullgrid",
        k_indices=None, k_values=None,
        delta_max_norm=0.18,          # <— más laxo por defecto
        penalty_missing=0.10,         # <— un poco más alto
        gap_stay_penalty=0.02,        # <— evita camino todo-GAP
        gap_start_penalty=0.05,       # <— NUEVO: penaliza iniciar en GAP
        merge_tol_norm=3e-4,
        write=False, subdir_out="fullgrid_ordered"):
        """
        Ordena bandas globalmente (min-cost en DAG) sobre el camino en k (o subset).
        - mode: 'fullgrid' usa self.omega_longitudinal; 'sigma' usa self.omega_longitudinal_sigma
        - subset: por índices o por valores de k
        - tolerancias en ω̄ (frecuencia normalizada con C_l0 y 'a')
        - write=True -> guarda archivos reordenados en 'subdir_out' (no pisa los originales)
        Devuelve:
          F_ord: (nk_full, nb, 2) con filas del subset re-etiquetadas por continuidad,
          paths: (nb, nk_sub) con el índice local elegido en cada k o -1 si gap.
        """
        import numpy as np, os
        from pathlib import Path
    
        assert mode in ("fullgrid", "sigma")
        k_full, k_idx = self._resolve_k_subset(k_indices=k_indices, k_values=k_values)
        nk_full = len(k_full)
        nb = int(self.nbands)
        k_sub = k_full[k_idx]
        nk_sub = len(k_sub)
    
        # 1) toma tensor base (no lo pisa)
        base_attr = 'omega_longitudinal' if mode == 'fullgrid' else 'omega_longitudinal_sigma'
        F_base = self._ensure_tensor(base_attr, nk_full, nb)  # (nk_full, nb, 2)
    
        # 2) construir layers por k del subset: lista de picos válidos (wr, wi) y sus ω̄
        def dedup_layer(rows_wrwi, wnorm, tol):
            """Elimina duplicados cercanos en ω̄ dentro de la misma capa."""
            if rows_wrwi.shape[0] <= 1: return rows_wrwi, wnorm
            order = np.argsort(wnorm)
            rows = rows_wrwi[order, :].copy()
            wn = wnorm[order].copy()
            keep = [0]
            for j in range(1, len(wn)):
                if abs(wn[j] - wn[keep[-1]]) > tol:
                    keep.append(j)
            keep = np.array(keep, dtype=int)
            return rows[keep], wn[keep]
    
        a_val = float(self.a); Cl0 = float(self.vel0[1])
        layers = []         # por k_sub: lista de dicts {rows: (m,2), wn:(m,), used: bool[m]}
        # Usamos TODO lo que haya en la fila original (aunque esté desordenado)
        for i in k_idx:
            rows = F_base[i, :, :].astype(float)  # (nb, 2) (wr, wi)
            mask = np.isfinite(rows[:, 0])        # válidos por wr
            rows = rows[mask, :]
            wn = (rows[:, 0] * a_val) / (2.0*np.pi*Cl0)
            rows, wn = dedup_layer(rows, wn, merge_tol_norm)
            layers.append({
                "rows": rows,            # (m_i, 2)
                "wn": wn,                # (m_i,)
                "used": np.zeros(len(wn), dtype=bool)
            })
    
        # 3) DP en DAG para encontrar nb caminos disjuntos (con nodo GAP por capa)
        GAP = -1  # índice especial para gap en cada capa
    
        def shortest_path_once():
            """Devuelve mejor camino global como lista de índices [node_t], node_t∈{0..m_t-1}∪{GAP}."""
            # dp[t][j] = costo mínimo hasta la capa t terminando en j (j∈0..m_t-1 o GAP)
            dp = []
            prv = []
            # capa 0
            m0 = len(layers[0]["wn"])
            dp0 = np.full(m0 + 1, np.inf)   # +1 para GAP
            prv0 = np.full(m0 + 1, -2, int)
            
            # Llegar a un nodo real cuesta 0 (preferimos nodos reales)
            for j in range(m0):
                if not layers[0]["used"][j]:
                    dp0[j] = 0.0
            
            # Iniciar en GAP tiene penalización (evita camino todo-GAP)
            dp0[m0] = float(gap_start_penalty)
            
            dp.append(dp0); prv.append(prv0)

            # transiciones
            for t in range(1, nk_sub):
                mt = len(layers[t]["wn"])
                dp_t = np.full(mt + 1, np.inf)
                prv_t = np.full(mt + 1, -2, int)
    
                # pre-capa
                wn_prev = layers[t-1]["wn"]; mprev = len(wn_prev)
                wn_curr = layers[t]["wn"]
    
                for jprev in range(mprev + 1):                # 0..mprev-1 ó GAP=mprev
                    if jprev < mprev and layers[t-1]["used"][jprev]:
                        continue
                    cost_prev = dp[t-1][jprev]
                    if not np.isfinite(cost_prev): continue
    
                    for jcur in range(mt + 1):                 # 0..mt-1 ó GAP=mt
                        if jcur < mt and layers[t]["used"][jcur]:
                            continue
    
                        # costo de transición
                        if jprev == mprev and jcur == mt:
                            c = gap_stay_penalty
                        elif jprev == mprev and jcur < mt:
                            c = penalty_missing
                        elif jprev < mprev and jcur == mt:
                            c = penalty_missing
                        else:
                            d = abs(wn_curr[jcur] - wn_prev[jprev])
                            if d > float(delta_max_norm):
                                continue  # prohibida
                            c = d
    
                        val = cost_prev + c
                        if val < dp_t[jcur]:
                            dp_t[jcur] = val
                            prv_t[jcur] = jprev
    
                dp.append(dp_t); prv.append(prv_t)
    
            # cierre: elegir mejor final
            last = dp[-1]
            jbest = int(np.nanargmin(last))
            path = [jbest]
            # backtrack
            for t in range(nk_sub-1, 0, -1):
                j = path[-1]
                jprev = prv[t][j]
                if jprev < -1:
                    # no hay camino válido
                    return None
                path.append(jprev)
            path.reverse()
            # mapear GAP índice (mt) a GAP=-1 por consistencia
            final = []
            for t, j in enumerate(path):
                mt = len(layers[t]["wn"])
                final.append(GAP if j == mt else j)
            return final
    
        # encontraremos nb caminos, marcando como 'used' los nodos reales seleccionados
        paths = np.full((nb, nk_sub), GAP, int)
        for b in range(nb):
            path = shortest_path_once()
            if path is None:
                break
            paths[b, :] = path
            # marcar usados
            for t, j in enumerate(path):
                if j != GAP:
                    layers[t]["used"][j] = True
    
        # 4) construir tensor ordenado en RAM (no pisa el original)
        F_ord = self._ensure_tensor(base_attr + "_ordered", nk_full, nb)  # crea si no existía
        F_ord[:] = np.nan
    
        for t, ireal in enumerate(k_idx):
            mt = len(layers[t]["wn"])
            # limpiar fila
            F_ord[ireal, :, :] = np.nan
            for b in range(nb):
                j = paths[b, t]
                if j == GAP:  # hueco
                    continue
                rows = layers[t]["rows"]
                if 0 <= j < mt:
                    F_ord[ireal, b, 0] = rows[j, 0]  # wr
                    F_ord[ireal, b, 1] = rows[j, 1]  # wi
    
        # 5) opcional: escribir archivos re-ordenados en subdir_out (solo subset)
        if write:
            out_dir = Path(self.frecfolder) / str(subdir_out)
            out_dir.mkdir(parents=True, exist_ok=True)
            for t, ireal in enumerate(k_idx):
                kval = k_full[ireal]
                # compactar sacando NaN finales
                row = F_ord[ireal, :, :]
                mask = np.isfinite(row[:, 0])
                arr_out = row[mask, :2].astype(float, copy=False)
                self._save_frec_file(os.path.join(out_dir, f"frecuencias{kval:.4f}.txt"), arr_out)
    
        # publicar atributo ordenado
        if mode == "fullgrid":
            self.omega_longitudinal_ordered = F_ord
        else:
            self.omega_longitudinal_sigma_ordered = F_ord
    
        # métricas de log
        n_assigned = int(np.sum(paths != GAP))
        n_possible = int(nb * nk_sub)
        print(f"[order-global] mode={mode} nk_sub={nk_sub} nb={nb}  assigned={n_assigned}/{n_possible} "
              f"delta_max_norm={delta_max_norm}  penalty_missing={penalty_missing}  merge_tol={merge_tol_norm}")
    
        return F_ord, paths

    # --- dentro de class Red ------------------------------------------------------
    def write_omega_longitudinal(self,
                                 outdir: str = None,
                                 k_idx=None,
                                 columns: str = "complex",
                                 scale: float = None,
                                 drop_nan: bool = True,
                                 overwrite: bool = True,
                                 fmt: str = "%.10e"):
        """
        Escribe al disco las frecuencias longitudinales actuales (self.omega_longitudinal)
        para cada k, en archivos separados:
            <outdir>/frecuencias{self.k[i]:.4f}.txt
    
        Parámetros
        ----------
        outdir : str or None
            Carpeta de salida. Si None, usa self.frecfolder.
        k_idx : None | slice | list[int] | np.ndarray
            Subconjunto de índices k a exportar. None => todos.
        columns : {"real","complex"}
            "real"    => archivo de 1 columna: Re(ω)
            "complex" => archivo de 2 columnas: Re(ω)  Im(ω)
        scale : float or None
            Si se da, multiplica ω por 'scale' ANTES de guardar (aplica a Re e Im).
            Ej: scale = self.a / (2*np.pi*ct0) para ω→ω̄.
        drop_nan : bool
            True: descarta bandas con partes no finitas (NaN/Inf) antes de guardar.
            False: escribe NaN tal cual en el archivo.
        overwrite : bool
            True: sobrescribe si existe. False: si el archivo existe, se salta.
        fmt : str
            Formato de guardado (np.savetxt).
    
        Devuelve
        --------
        stats : dict
            {'written': int, 'skipped': int, 'rows_total': int}
        """
        import os
        import numpy as np
    
        # --- carpeta destino ---
        outdir = outdir or getattr(self, "frecfolder", None)
        if not outdir:
            raise ValueError("No se especificó 'outdir' y 'self.frecfolder' no existe.")
        os.makedirs(outdir, exist_ok=True)
    
        # --- preparar ω en forma compleja ---
        W = self.omega_longitudinal
        if W is None:
            raise ValueError("self.omega_longitudinal es None.")
    
        # Acepta (nk, nb) complejo, o (nk, nb, 2) con [Re, Im]
        if np.iscomplexobj(W):
            w = W.astype(np.complex128, copy=False)
        elif W.ndim == 3 and W.shape[2] >= 2:
            w = W[:, :, 0].astype(np.float64) + 1j * W[:, :, 1].astype(np.float64)
        elif W.ndim == 2:
            # Solo real disponible -> tratar como complejo con Im=0
            w = W.astype(np.float64) + 0j
        else:
            raise ValueError("Formato de self.omega_longitudinal no reconocido.")
    
        nk = w.shape[0]
        # --- subconjunto de k ---
        if k_idx is None:
            k_idx = range(nk)
        elif isinstance(k_idx, (list, tuple, np.ndarray, slice)):
            k_idx = np.arange(nk)[k_idx]
        else:
            raise ValueError("k_idx debe ser None, slice o colección de índices.")
    
        # --- factor de escala (opcional) ---
        if scale is not None:
            w = w * scale
    
        # --- guardar por k ---
        written = 0
        skipped = 0
        rows_total = 0
    
        for i in k_idx:
            wi = w[i, :]
            # Selección de filas válidas
            if drop_nan:
                valid = np.isfinite(wi.real) & np.isfinite(wi.imag)
                wi = wi[valid]
    
            # Si no hay nada que guardar (todas Nan), dejar archivo vacío (0 filas)
            if columns == "real":
                arr = wi.real.reshape(-1, 1)
            elif columns == "complex":
                arr = np.column_stack((wi.real, wi.imag))
            else:
                raise ValueError("columns debe ser 'real' o 'complex'.")
    
            fname = os.path.join(outdir, f"frecuencias{self.k[i]:.4f}.txt")
            if (not overwrite) and os.path.exists(fname):
                skipped += 1
                continue
    
            # Guardar (archivo puede quedar con 0 filas si no hay datos válidos)
            np.savetxt(fname, arr, fmt=fmt)
            written += 1
            rows_total += arr.shape[0]
    
        return {"written": written, "skipped": skipped, "rows_total": rows_total}
    # --- fin de método ------------------------------------------------------------

    def determinantlongitudinal_mejorado(self, f, k, cutoff, nsuma=5):
        """
        Versión mejorada que retorna Re(det), Im(det) y |det|.
        
        Parameters
        ----------
        f : tuple or float
            Frecuencia (omega_real, omega_imag) o solo omega_real
        k : float
            Vector de Bloch
        cutoff : int
            Truncamiento de modos
        nsuma : int
            Puntos de suma en G0
            
        Returns
        -------
        tuple : (Re[det], Im[det], |det|)
        """
        # Asegurar que f sea tupla
        if isinstance(f, (int, float)):
            f = (float(f), 0.0)
        
        size = 2 * cutoff + 1
        
        # Construir matriz T (coeficientes de dispersión)
        if self.condborde == "rigid":
            T = np.diag(np.array([self.coeficientedispersionelastic(f, n) 
                                  for n in np.arange(-cutoff, cutoff+1)], 
                                 dtype=complex))
        else:
            T = np.diag(np.array([self.coeficientedispersionhollow(f, n) 
                                  for n in np.arange(-cutoff, cutoff+1)], 
                                 dtype=complex))
        
        # Suma de red G0
        G = self.G0(f, k, 1, cutoff, nsuma)
        
        # Matriz completa: M = T + G - I
        M = T + G
        identidad = np.identity(size)
        determinante = np.linalg.det(M - identidad)
        
        # Módulo del determinante
        det_modulo = np.abs(determinante)
        
        return np.real(determinante), np.imag(determinante), det_modulo
    
    def verificar_residuo(self, sol, k, cutoff, tol_residuo=1e-6):
        """
        Verifica que la solución sea realmente un cero del determinante.
        
        Parameters
        ----------
        sol : tuple
            (omega_real, omega_imag) solución encontrada
        k : float
            Vector de Bloch
        cutoff : int
            Truncamiento
        tol_residuo : float
            Tolerancia máxima para |det(A)|
            
        Returns
        -------
        es_valido : bool
            True si el residuo es aceptable
        residuo : float
            Valor de |det(A(ω_sol))|
        """
        det_re, det_im, det_modulo = self.determinantlongitudinal_mejorado(
            sol, k, cutoff
        )
        
        # El residuo es la norma del determinante complejo
        residuo = np.sqrt(det_re**2 + det_im**2)
        
        es_valido = (residuo < tol_residuo)
        
        return es_valido, residuo
    
    def buscar_raiz_con_verificacion(self, omega_guess, k, cutoff,
                                      nsuma=5,
                                      tol_residuo=1e-6,
                                      imagtol=1e-3,
                                      xtol=1e-10,
                                      maxfev=2000):
        """
        Busca una raíz con todas las verificaciones.
        
        Parameters
        ----------
        omega_guess : float or tuple
            Estimación inicial
        k : float
            Vector de Bloch
        cutoff : int
            Truncamiento
        nsuma : int
            Puntos en suma de red
        tol_residuo : float
            Tolerancia del residuo
        imagtol : float
            Tolerancia para parte imaginaria
        xtol : float
            Tolerancia de convergencia de fsolve
        maxfev : int
            Máximo número de evaluaciones
            
        Returns
        -------
        dict or None :
            Diccionario con info de la solución si es válida, None si no
        """
        # Asegurar que omega_guess sea tupla
        if isinstance(omega_guess, (int, float)):
            omega_guess = (float(omega_guess), 0.0)
        
        try:
            # Resolver sistema Re[det]=0, Im[det]=0
            sol, info, ier, msg = fsolve(
                self.determinant_longitudinal,
                omega_guess,
                args=(k, cutoff, nsuma),
                xtol=xtol,
                maxfev=maxfev,
                full_output=True
            )
            
            omega_real, omega_imag = sol
            
            # ====== VERIFICACIONES ======
            
            # 1. fsolve convergió exitosamente
            if ier != 1:
                return None
            
            # 2. Residuo suficientemente pequeño
            es_valido, residuo = self.verificar_residuo(
                sol, k, cutoff, tol_residuo
            )
            if not es_valido:
                return None
            
            # 3. Frecuencia real positiva
            if omega_real <= 0:
                return None
            
            # 4. Clasificar tipo de modo según parte imaginaria
            if abs(omega_imag) < imagtol:
                tipo_modo = "propagante"
            elif abs(omega_imag) < 0.1:
                tipo_modo = "evanescente"
            else:
                # Parte imaginaria muy grande, rechazar
                return None
            
            # ====== SOLUCIÓN VÁLIDA ======
            return {
                'omega_real': omega_real,
                'omega_imag': omega_imag,
                'residuo': residuo,
                'tipo': tipo_modo,
                'nfev': info['nfev'],
                'mensaje': msg,
                'omega_guess': omega_guess[0]
            }
            
        except Exception as e:
            # Capturar cualquier error numérico
            return None

    def contar_raices_en_intervalo(self, k, omega_min, omega_max,
                                    cutoff=None,
                                    nsuma=5,
                                    npts=100,
                                    delta_imag=0.01,
                                    verbose=False):
        """
        Cuenta raíces de det(A)=0 usando el Teorema del Argumento.
        
        Este método cuenta el número de ceros del determinante dentro
        de un contorno rectangular en el plano complejo omega.
        
        Parameters
        ----------
        k : float
            Vector de Bloch
        omega_min, omega_max : float
            Intervalo de frecuencias (eje real)
        cutoff : int or None
            Truncamiento (usa self.cut si es None)
        nsuma : int
            Puntos en suma de red
        npts : int
            Número de puntos para discretizar el contorno
        delta_imag : float
            Desplazamiento imaginario del contorno
        verbose : bool
            Si imprimir información de diagnóstico
            
        Returns
        -------
        n_raices : int
            Número estimado de raíces en el intervalo
        info : dict
            Información diagnóstica
        """
        if cutoff is None:
            cutoff = self.cut
        
        # PASO 1: Construir contorno rectangular
        # Lado superior: omega_min + i*delta → omega_max + i*delta
        # Lado inferior: omega_max - i*delta → omega_min - i*delta
        
        omega_real = np.linspace(omega_min, omega_max, npts)
        
        # Evaluar determinante en contorno superior
        contorno_sup = []
        for w in omega_real:
            det_val = self.determinant_longitudinal(
                (w, delta_imag), k, cutoff, nsuma
            )
            contorno_sup.append(complex(det_val[0], det_val[1]))
        
        # Evaluar determinante en contorno inferior (orden inverso)
        contorno_inf = []
        for w in omega_real[::-1]:
            det_val = self.determinant_longitudinal(
                (w, -delta_imag), k, cutoff, nsuma
            )
            contorno_inf.append(complex(det_val[0], det_val[1]))
        
        # PASO 2: Combinar contornos
        valores_contorno = contorno_sup + contorno_inf
        
        # Verificar que el determinante no se anule en el contorno
        min_modulo = min(abs(z) for z in valores_contorno)
        if min_modulo < 1e-8:
            warnings.warn(
                f"Determinante casi cero en contorno (|det|_min={min_modulo:.2e}). "
                f"Raíz muy cerca del contorno. Considere ajustar delta_imag."
            )
        
        # PASO 3: Calcular argumentos (ángulos en plano complejo)
        argumentos = [np.angle(z) for z in valores_contorno]
        
        # PASO 4: Calcular cambio total del argumento
        # (desenrollando discontinuidades de ±2π)
        cambio_total = 0.0
        for i in range(len(argumentos) - 1):
            delta = argumentos[i+1] - argumentos[i]
            
            # Manejar saltos de ±2π
            if delta > np.pi:
                delta -= 2.0 * np.pi
            elif delta < -np.pi:
                delta += 2.0 * np.pi
            
            cambio_total += delta
        
        # PASO 5: Número de raíces = cambio_total / (2π)
        n_raices_float = cambio_total / (2.0 * np.pi)
        n_raices = int(round(n_raices_float))
        
        # Verificar que el resultado sea cercano a un entero
        error_redondeo = abs(n_raices_float - n_raices)
        if error_redondeo > 0.05:
            warnings.warn(
                f"Posible error numérico: n_raices = {n_raices_float:.3f} "
                f"(redondeado a {n_raices}). Considere aumentar npts."
            )
        
        # Información diagnóstica
        info = {
            'k': k,
            'omega_min': omega_min,
            'omega_max': omega_max,
            'n_raices_exacto': n_raices_float,
            'n_raices': n_raices,
            'error_redondeo': error_redondeo,
            'cambio_argumento': cambio_total,
            'min_modulo_contorno': min_modulo,
            'delta_imag': delta_imag,
            'npts': npts
        }
        
        if verbose:
            print(f"Conteo de raíces por Teorema del Argumento:")
            print(f"  Intervalo: [{omega_min:.2f}, {omega_max:.2f}]")
            print(f"  k = {k:.4f}")
            print(f"  Raíces encontradas: {n_raices}")
            print(f"  Valor exacto: {n_raices_float:.4f}")
            print(f"  Error de redondeo: {error_redondeo:.2e}")
        
        return n_raices, info

    def validar_continuidad_banda(self, omega_actual, omega_anterior,
                                   k_actual, k_anterior,
                                   tol_salto_relativo=0.3):
        """
        Valida continuidad de banda entre puntos k consecutivos.
        
        Parameters
        ----------
        omega_actual, omega_anterior : float
            Frecuencias en k actual y anterior
        k_actual, k_anterior : float
            Valores de k
        tol_salto_relativo : float
            Tolerancia para saltos relativos (ej: 0.3 = 30%)
            
        Returns
        -------
        es_continua : bool
            True si la banda es continua
        diagnostico : dict
            Información diagnóstica
        """
        if np.isnan(omega_anterior) or np.isnan(omega_actual):
            return True, {'razon': 'NaN presente, no se puede validar'}
        
        # Salto absoluto y relativo
        salto_abs = abs(omega_actual - omega_anterior)
        salto_rel = salto_abs / omega_anterior if omega_anterior > 0 else np.inf
        
        # Estimación de velocidad de grupo: vg ≈ Δω/Δk
        dk = abs(k_actual - k_anterior)
        if dk > 1e-10:
            velocidad_grupo = salto_abs / dk
        else:
            velocidad_grupo = np.nan
        
        # Criterio de continuidad
        es_continua = (salto_rel <= tol_salto_relativo)
        
        diagnostico = {
            'omega_actual': omega_actual,
            'omega_anterior': omega_anterior,
            'salto_abs': salto_abs,
            'salto_rel': salto_rel,
            'salto_rel_porcentaje': salto_rel * 100,
            'velocidad_grupo': velocidad_grupo,
            'dk': dk,
            'es_continua': es_continua
        }
        
        return es_continua, diagnostico
    
    
    def validar_solucion_fisica(self, omega, k, banda_idx,
                                es_punto_gamma=False,
                                omega_max_fisico=None):
        """
        Valida que la solución satisfaga condiciones físicas básicas.
        
        Parameters
        ----------
        omega : float
            Frecuencia encontrada
        k : float
            Vector de Bloch
        banda_idx : int
            Índice de banda (0-indexed)
        es_punto_gamma : bool
            Si es el punto Γ (k≈0)
        omega_max_fisico : float or None
            Límite superior físico razonable
            
        Returns
        -------
        es_valido : bool
            True si satisface condiciones físicas
        diagnostico : dict
            Información diagnóstica
        """
        diagnostico = {'checks': []}
        es_valido = True
        
        # Check 1: Frecuencia positiva
        if omega <= 0:
            es_valido = False
            diagnostico['checks'].append({
                'nombre': 'frecuencia_positiva',
                'resultado': False,
                'razon': f'ω = {omega:.2e} ≤ 0'
            })
        else:
            diagnostico['checks'].append({
                'nombre': 'frecuencia_positiva',
                'resultado': True
            })
        
        # Check 2: En Γ, rama acústica debe tender a 0
        if es_punto_gamma and abs(k) < 1e-6:
            if banda_idx == 0:  # Primera banda (acústica)
                umbral_acustica = 0.1 * self.vel0[1]  # 10% de velocidad de sonido
                if omega > umbral_acustica:
                    es_valido = False
                    diagnostico['checks'].append({
                        'nombre': 'rama_acustica_gamma',
                        'resultado': False,
                        'razon': f'Rama acústica no tiende a 0 en Γ (ω={omega:.2f})'
                    })
                else:
                    diagnostico['checks'].append({
                        'nombre': 'rama_acustica_gamma',
                        'resultado': True
                    })
        
        # Check 3: Límite superior físico
        if omega_max_fisico is not None:
            if omega > omega_max_fisico:
                es_valido = False
                diagnostico['checks'].append({
                    'nombre': 'limite_superior',
                    'resultado': False,
                    'razon': f'ω = {omega:.2f} > ω_max = {omega_max_fisico:.2f}'
                })
            else:
                diagnostico['checks'].append({
                    'nombre': 'limite_superior',
                    'resultado': True
                })
        
        diagnostico['es_valido'] = es_valido
        
        return es_valido, diagnostico

    def zeros_fullgrid_mejorado(self, Cl0, 
                                ventanasporunidad=100,
                                wnormmax=1.25,
                                tol_residuo=1e-6,
                                imagtol=1e-3,
                                verificar_completitud=True,
                                validar_continuidad=True,
                                verbose=True):
        """
        Versión mejorada de zeros_fullgrid con todas las verificaciones.
        
        Parameters
        ----------
        Cl0 : float
            Velocidad de referencia
        ventanasporunidad : int
            Semillas por unidad de frecuencia normalizada
        wnormmax : float
            Frecuencia normalizada máxima
        tol_residuo : float
            Tolerancia del residuo |det(A)|
        imagtol : float
            Tolerancia para parte imaginaria
        verificar_completitud : bool
            Si verificar conteo de raíces
        validar_continuidad : bool
            Si validar continuidad de bandas
        verbose : bool
            Si mostrar información detallada
            
        Returns
        -------
        frec : ndarray
            Array (nk, nbands, 2) con frecuencias [real, imag]
        diagnosticos : list
            Lista de diccionarios con diagnósticos por punto k
        """
        if verbose:
            print("\n" + "="*70)
            print("BÚSQUEDA MEJORADA DE ESTRUCTURA DE BANDAS")
            print("="*70)
            print(f"Parámetros:")
            print(f"  - Tolerancia residuo: {tol_residuo:.1e}")
            print(f"  - Tolerancia Im(ω): {imagtol:.1e}")
            print(f"  - Verificar completitud: {verificar_completitud}")
            print(f"  - Validar continuidad: {validar_continuidad}")
            print("="*70 + "\n")
        
        starttime = time.time()
        
        # Preparar arrays
        frec = np.full((self.nk, self.nbands, 2), np.nan)
        diagnosticos = []
        
        # Intervalo de búsqueda global
        omega_min_global = 0.0
        omega_max_global = (2 * np.pi * Cl0 / self.a) * wnormmax
        
        # Loop principal sobre puntos k
        iterator = tqdm(range(self.nk), desc="Calculando bandas") if verbose else range(self.nk)
        
        for idx_k in iterator:
            kval = self.k[idx_k]
            
            # Identificar puntos especiales
            es_gamma = (abs(kval) < 1e-6)
            
            # ====== PASO 1: Contar raíces teóricas (opcional) ======
            n_teorico = None
            if verificar_completitud:
                try:
                    n_teorico, info_conteo = self.contar_raices_en_intervalo(
                        kval, omega_min_global, omega_max_global,
                        npts=150, delta_imag=0.02, verbose=False
                    )
                except Exception as e:
                    if verbose:
                        print(f"⚠ Error en conteo de raíces k[{idx_k}]: {e}")
                    n_teorico = None
            
            # ====== PASO 2: Generar semillas iniciales ======
            nsemillas_base = int(ventanasporunidad * wnormmax)
            semillas = np.linspace(omega_min_global, omega_max_global, nsemillas_base)
            
            # Si hay soluciones anteriores, agregar semillas cercanas
            if idx_k > 0:
                for banda in range(self.nbands):
                    omega_prev = frec[idx_k-1, banda, 0]
                    if not np.isnan(omega_prev) and omega_prev > 0:
                        # Agregar semillas alrededor de la solución anterior
                        semillas = np.append(semillas, [
                            omega_prev * 0.90,
                            omega_prev * 0.95,
                            omega_prev,
                            omega_prev * 1.05,
                            omega_prev * 1.10
                        ])
            
            # Eliminar duplicados y ordenar
            semillas = np.unique(semillas)
            semillas = semillas[(semillas >= omega_min_global) & 
                               (semillas <= omega_max_global)]
            
            # ====== PASO 3: Búsqueda de raíces ======
            soluciones = []
            
            for omega_guess in semillas:
                resultado = self.buscar_raiz_con_verificacion(
                    omega_guess, kval, self.cut,
                    nsuma=self.n_suma,
                    tol_residuo=tol_residuo,
                    imagtol=imagtol,
                    xtol=self.sol_tol
                )
                
                if resultado is not None:
                    omega = resultado['omega_real']
                    
                    # Evitar duplicados
                    if not any(abs(omega - s['omega_real']) < 1e-4 
                              for s in soluciones):
                        soluciones.append(resultado)
            
            # ====== PASO 4: Ordenar soluciones por frecuencia ======
            soluciones = sorted(soluciones, key=lambda x: x['omega_real'])
            
            # ====== PASO 5: Verificar completitud ======
            advertencias = []
            if verificar_completitud and n_teorico is not None:
                if len(soluciones) != n_teorico:
                    msg = (f"⚠ k[{idx_k}]={kval:.4f}: Esperadas {n_teorico}, "
                          f"encontradas {len(soluciones)} raíces")
                    advertencias.append(msg)
                    if verbose:
                        print(msg)
            
            # ====== PASO 6: Almacenar resultados ======
            for banda in range(min(len(soluciones), self.nbands)):
                frec[idx_k, banda, 0] = soluciones[banda]['omega_real']
                frec[idx_k, banda, 1] = soluciones[banda]['omega_imag']
            
            # ====== PASO 7: Validar continuidad ======
            continuidad_checks = []
            if validar_continuidad and idx_k > 0:
                for banda in range(self.nbands):
                    if not np.isnan(frec[idx_k, banda, 0]):
                        es_cont, diag = self.validar_continuidad_banda(
                            frec[idx_k, banda, 0],
                            frec[idx_k-1, banda, 0],
                            kval,
                            self.k[idx_k-1]
                        )
                        
                        continuidad_checks.append(diag)
                        
                        if not es_cont:
                            msg = (f"⚠ Discontinuidad banda {banda}, "
                                  f"k[{idx_k}]={kval:.4f}, "
                                  f"salto={diag['salto_rel_porcentaje']:.1f}%")
                            advertencias.append(msg)
                            if verbose:
                                print(msg)
            
            # ====== PASO 8: Diagnóstico ======
            diagnostico = {
                'k_idx': idx_k,
                'k_val': kval,
                'es_gamma': es_gamma,
                'n_encontrado': len(soluciones),
                'n_teorico': n_teorico,
                'soluciones': soluciones,
                'continuidad': continuidad_checks,
                'advertencias': advertencias,
                'residuos': [s['residuo'] for s in soluciones]
            }
            diagnosticos.append(diagnostico)
        
        # ====== GUARDAR RESULTADOS ======
        self.omegalongitudinal = frec
        
        tiempo_total = time.time() - starttime
        
        # ====== REPORTE FINAL ======
        if verbose:
            print("\n" + "="*70)
            print("REPORTE FINAL DE CALIDAD")
            print("="*70)
            print(f"Tiempo total: {tiempo_total:.2f} s")
            print(f"Puntos k procesados: {self.nk}")
            
            # Estadísticas de residuos
            todos_residuos = []
            for d in diagnosticos:
                todos_residuos.extend(d['residuos'])
            
            if todos_residuos:
                print(f"\nResiduos |det(A)|:")
                print(f"  - Máximo: {max(todos_residuos):.2e}")
                print(f"  - Promedio: {np.mean(todos_residuos):.2e}")
                print(f"  - Mediana: {np.median(todos_residuos):.2e}")
            
            # Puntos con problemas
            puntos_incompletos = sum(
                1 for d in diagnosticos
                if d['n_teorico'] and d['n_encontrado'] != d['n_teorico']
            )
            
            puntos_discontinuos = sum(
                1 for d in diagnosticos
                if any(not c['es_continua'] for c in d['continuidad'])
                if d['continuidad']
            )
            
            print(f"\nDiagnóstico:")
            print(f"  - Puntos con raíces faltantes: {puntos_incompletos}/{self.nk}")
            print(f"  - Puntos con discontinuidades: {puntos_discontinuos}/{self.nk}")
            
            # Calidad general
            if puntos_incompletos == 0 and puntos_discontinuos == 0:
                print(f"\n✓ ¡EXCELENTE! Todas las bandas calculadas correctamente")
            elif puntos_incompletos < self.nk * 0.05:
                print(f"\n✓ BUENO: <5% de puntos con problemas")
            else:
                print(f"\n⚠ ADVERTENCIA: Considere revisar parámetros de búsqueda")
            
            print("="*70 + "\n")
        
        return frec, diagnosticos

    # ========================================================================
    # 6. HERRAMIENTAS DE VISUALIZACIÓN Y DIAGNÓSTICO
    # ========================================================================
    
    def plot_determinante_2d(self, k, omega_min, omega_max,
                             npts_omega=200,
                             npts_imag=50,
                             omega_imag_max=0.1,
                             save_path=None):
        """
        Grafica |det(A)| en el plano complejo omega.
        Útil para diagnóstico visual de dónde están los ceros.
        
        Parameters
        ----------
        k : float
            Vector de Bloch
        omega_min, omega_max : float
            Rango de Re(ω)
        npts_omega : int
            Puntos en eje real
        npts_imag : int
            Puntos en eje imaginario
        omega_imag_max : float
            Máximo valor de Im(ω)
        save_path : str or None
            Si no None, guardar figura en esta ruta
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        print(f"Generando mapa de |det(A)| para k={k:.4f}...")
        
        # Malla en plano complejo
        omega_real = np.linspace(omega_min, omega_max, npts_omega)
        omega_imag = np.linspace(-omega_imag_max, omega_imag_max, npts_imag)
        
        OR, OI = np.meshgrid(omega_real, omega_imag)
        det_modulo = np.zeros_like(OR)
        
        # Evaluar |det(A)| en malla
        for i in range(npts_imag):
            for j in range(npts_omega):
                omega_complex = (OR[i, j], OI[i, j])
                det_val = self.determinant_longitudinal(
                    omega_complex, k, self.cut, self.nsuma
                )
                det_modulo[i, j] = np.sqrt(det_val[0]**2 + det_val[1]**2)
        
        # Graficar
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Escala logarítmica para mejor visualización
        det_log = np.log10(det_modulo + 1e-12)
        
        # Mapa de color
        im = ax.contourf(OR, OI, det_log, levels=50, cmap='viridis')
        
        # Contornos de nivel
        contours = ax.contour(OR, OI, det_log, 
                             levels=[-8, -6, -4, -2],
                             colors='white',
                             linewidths=1.5,
                             alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%d')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'$\log_{10}|\det(\mathbf{A})|$', fontsize=14)
        
        # Eje real
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # Etiquetas
        ax.set_xlabel(r'$\mathrm{Re}(\omega)$ [rad/s]', fontsize=14)
        ax.set_ylabel(r'$\mathrm{Im}(\omega)$ [rad/s]', fontsize=14)
        ax.set_title(
            f'Módulo del determinante en plano complejo (k = {k:.4f})',
            fontsize=16
        )
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")
        
        return fig, ax
    
    
    def reporte_calidad_solucion(self, k_idx, soluciones):
        """
        Genera reporte de texto sobre calidad de soluciones.
        
        Parameters
        ----------
        k_idx : int
            Índice de k
        soluciones : list
            Lista de diccionarios de soluciones
            
        Returns
        -------
        str : Reporte formateado
        """
        k_val = self.k[k_idx]
        
        reporte = f"\n{'='*70}\n"
        reporte += f"REPORTE DE CALIDAD - k[{k_idx}] = {k_val:.6f}\n"
        reporte += f"{'='*70}\n\n"
        
        if len(soluciones) == 0:
            reporte += "⚠ No se encontraron soluciones válidas\n"
            return reporte
        
        reporte += f"Soluciones encontradas: {len(soluciones)}\n\n"
        
        for i, sol in enumerate(soluciones):
            reporte += f"Banda {i}:\n"
            reporte += f"  ω_real    = {sol['omega_real']:.6f} rad/s\n"
            reporte += f"  ω_imag    = {sol['omega_imag']:.2e} rad/s\n"
            reporte += f"  |det(A)|  = {sol['residuo']:.2e}\n"
            reporte += f"  Tipo      = {sol['tipo']}\n"
            reporte += f"  Neval     = {sol['nfev']}\n"
            reporte += f"  Semilla   = {sol['omega_guess']:.2f}\n"
            
            # Indicador visual de calidad
            if sol['residuo'] < 1e-8:
                calidad = "★★★ Excelente"
            elif sol['residuo'] < 1e-6:
                calidad = "★★☆ Buena"
            elif sol['residuo'] < 1e-4:
                calidad = "★☆☆ Aceptable"
            else:
                calidad = "☆☆☆ Dudosa"
            
            reporte += f"  Calidad   = {calidad}\n\n"
        
        return reporte
    
    
    def exportar_diagnosticos(self, diagnosticos, filepath):
        """
        Exporta diagnósticos a archivo CSV para análisis posterior.
        
        Parameters
        ----------
        diagnosticos : list
            Lista de diccionarios de diagnóstico
        filepath : str
            Ruta del archivo CSV a crear
        """
        import pandas as pd
        
        # Preparar datos
        data = []
        for d in diagnosticos:
            for i, sol in enumerate(d['soluciones']):
                row = {
                    'k_idx': d['k_idx'],
                    'k_val': d['k_val'],
                    'banda_idx': i,
                    'omega_real': sol['omega_real'],
                    'omega_imag': sol['omega_imag'],
                    'residuo': sol['residuo'],
                    'tipo': sol['tipo'],
                    'nfev': sol['nfev'],
                    'n_teorico': d['n_teorico'],
                    'n_encontrado': d['n_encontrado']
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Diagnósticos exportados a: {filepath}")

    def G0_cached(self, f, k):
        if isinstance(f, (list, tuple, np.ndarray)):
            f_real = float(f[0])
            f_imag = float(f[1])
        else:
            f_real = float(f)
            f_imag = 0.0
        key = (round(f_real, 6), round(f_imag, 6), round(float(k), 4))
        if key not in self._cache_G0:
            freq = [f_real , f_imag]
            self._cache_G0[key] = self.G0_convergente(freq, k, pol=1, 
                                                      cut=self.cut, 
                                                      n_suma_ini=self.n_suma,
                                                      n_suma_max=400
                                                      )[0]
        return self._cache_G0[key]
    
    def generate_adaptive_seeds(self, omega_min, omega_max, k, n0=20, max_levels=1):
        """
        Genera semillas adaptativas para búsqueda de raíces.
        1. Crea n0 semillas uniformes.
        2. En cada nivel, evalúa el determinante en las semillas.
        3. Donde haya cambio de signo en Re[det], inserta semilla intermedia.
        max_levels controla cuántas rondas de refinamiento.
        """
        seeds = np.linspace(omega_min, omega_max, n0).tolist()
        for _ in range(max_levels):
            det_vals = []
            # Evaluar Re(det) en cada semilla
            for w in seeds:
                det_re, _ = self.determinant_longitudinal([w, 0.0001], k, self.cut, self.n_suma)
                det_vals.append(det_re)
            # Generar nuevas semillas en intervalos con cambio de signo
            new_seeds = []
            for i in range(len(seeds)-1):
                if det_vals[i] * det_vals[i+1] < 0:
                    mid = 0.5 * (seeds[i] + seeds[i+1])
                    new_seeds.append(mid)
            seeds += new_seeds
            seeds = sorted(set(seeds))
        return np.array(seeds)
    
    def verify_completeness(self, k, omega_min, omega_max, found_roots):
        expected_count, _ = self.contar_raices_en_intervalo(k, omega_min, omega_max)
        found_count = len(found_roots)
        if found_count < expected_count:
            print(f"⚠️ Faltan {expected_count - found_count} raíces en k={k:.4f}")
            return False
        if found_count > expected_count:
            print(f"⚠️ Sobran {found_count - expected_count} raíces en k={k:.4f}")
            return False
        return True

    def zeros_transv_fullgrid_subset(self,
                                           C_l0,
                                           w_norm_max=1.0,
                                           ventanas_por_unidad=100,   # compatibilidad
                                           subdir="fullgrid",
                                           progress="tqdm",
                                           k_indices=None,
                                           k_values=None,
                                           robust_units="auto",       # "omega" | "wnorm" | "auto"
                                           write=True,
                                           write_nan_rows=True,      # <-- si True escribe NaN como placeholders
                                           w_norm_min=1e-3,
                                           **kwargs):
        """
        Calcula raíces longitudinales sólo en un SUBCONJUNTO de k usando robust_root_finding.
        Devuelve (F_small, k_idx) con shape F_small=(nk_sub, nbands, 2) y NaN donde no hay raíz.
        Actualiza en RAM self.omega_longitudinal (full tensor) rellenando con NaN fuera del subset.
    
        - robust_units: "omega"  => límites en ω; "wnorm" => límites en ω̄; "auto" intenta ω y, si vacío, ω̄.
        - write_nan_rows: si True, escribe filas con NaN (placeholders) en archivos de salida; si False, sólo válidas.
        """
        import os, numpy as np
        from tqdm import tqdm
    
        nb = int(self.nbands)
        # factor ω = ω̄ * (2π C_l0 / a)
        fact = (2.0 * np.pi * float(C_l0)) / float(self.a)
        omega_min = max(1e-9, float(w_norm_min) * fact)
        omega_max = float(w_norm_max) * fact
    
        # -------- resolver subset k ----------
        if hasattr(self, "_resolve_k_subset"):
            k_full, k_idx = self._resolve_k_subset(k_indices=k_indices, k_values=k_values)
        else:
            # Fallback simple
            k_full = np.asarray(self.k, dtype=float)
            if k_indices is not None:
                k_idx = np.arange(k_full.size)[k_indices]
            elif k_values is not None:
                kv = np.asarray(k_values, dtype=float).ravel()
                k_idx = np.array([np.argmin(np.abs(k_full - x)) for x in kv], dtype=int)
            else:
                k_idx = np.arange(k_full.size, dtype=int)
    
        k_sub = k_full[k_idx]
        nk_sub = len(k_sub)
        print(f"[fullgrid-subset/rrf] nk_sub={nk_sub}/{len(k_full)}  nb={nb}  (w̄∈[{w_norm_min},{w_norm_max}])")
    
        # -------- salida subset (prellenar NaN) ----------
        F_small = np.full((nk_sub, nb, 2), np.nan, dtype=float)
    
        # -------- preparar RAM full tensor con NaN ----------
        W = getattr(self, "omega_longitudinal", None)
        nk_tot = len(k_full)
        if not (isinstance(W, np.ndarray) and W.ndim == 3 and W.shape[0] == nk_tot and W.shape[1] == nb and W.shape[2] >= 2):
            W_full = np.full((nk_tot, nb, 2), np.nan, dtype=float)
        else:
            # copiar sólo las dos primeras columnas (Re, Im) si hay más dims
            W_full = np.array(W[:, :, :2], copy=True)
    
        # -------- carpeta salida --------
        out_dir = os.path.join(self.frecfolder, str(subdir))
        if write:
            os.makedirs(out_dir, exist_ok=True)
    
        # -------- ayudantes --------
        def _roots_to_omega(R, assumed="auto"):
            R = np.array(R, dtype=float).reshape(-1, 2)
            if R.size == 0:
                return R
            if assumed == "omega":   # ya en ω
                return R
            if assumed == "wnorm":   # venía en ω̄
                return R * fact
            # "auto": si Re(raíz) es chico (~<5), asumimos ω̄
            vmax = np.nanmax(np.abs(R[:, 0]))
            return R * fact if vmax < 5.0 else R
    
        # -------- lazo principal --------
        it = tqdm(range(nk_sub), desc="fullgrid-subset (robust)", unit="k", leave=False) if progress == "tqdm" else range(nk_sub)
    
        for isub in it:
            k_val = float(k_sub[isub])
    
            roots_omega = []
            if robust_units in ("omega", "auto"):
                try:
                    r = self.robust_root_finding(k_val, omega_min, omega_max)
                    roots_omega = _roots_to_omega(r, assumed="auto")
                except Exception:
                    roots_omega = []
    
            roots = roots_omega
            if (robust_units in ("wnorm", "auto")) and (len(roots) == 0):
                try:
                    r2 = self.robust_root_finding(k_val, float(w_norm_min), float(w_norm_max))
                    roots = _roots_to_omega(r2, assumed="wnorm")
                except Exception:
                    roots = []
    
            # ordenar por Re(ω)
            if len(roots) > 0:
                roots = roots[np.argsort(roots[:, 0])]
    
            # volcar a F_small (resto queda NaN)
            upto = min(nb, len(roots))
            if upto > 0:
                F_small[isub, :upto, 0] = roots[:upto, 0]
                F_small[isub, :upto, 1] = roots[:upto, 1]
    
            # actualizar RAM full en su fila correspondiente
            W_full[k_idx[isub], :, 0:2] = F_small[isub, :, 0:2]
    
            # escribir a disco
            if write:
                row = F_small[isub, :, 0:2]
                if write_nan_rows:
                    arr_out = row  # incluye NaN como placeholders
                else:
                    mask = np.isfinite(row[:, 0])
                    arr_out = row[mask, :]
                fname = os.path.join(out_dir, f"frecuencias{k_val:.4f}.txt")
                np.savetxt(fname, arr_out, fmt="%.10e")
    
        # Commit en RAM
        self.omega_longitudinal = W_full
        print(f"[fullgrid-subset/rrf] actualizado RAM (shape={W_full.shape}), wrote={'yes' if write else 'no'}")
    
        return F_small, k_idx


    def global_root_finding(self, k, omega_min, omega_max):
        def obj(x):
            w=float(x[0])
            re, im = self.determinant_longitudinal([w, 0.001], k, self.cut, self.n_suma)
            return re**2 + im**2
        result = differential_evolution(
            obj, bounds=[(omega_min, omega_max)],
            maxiter=500, polish=True,updating='deferred',
            workers=1,      # o -1 para paralelizar
            strategy='best1bin',
            tol=1e-6
            )
        return [float(result.x[0]), 0.0] if result.fun < 1e-8 else None

    def multi_algorithm_search(self, k, omega_min, omega_max, seeds):
        roots = []
        def det_real(w):
            return self.determinant_longitudinal([w,0.001], k, self.cut, self.n_suma)[0]

        for seed in seeds:
            # fsolve
            try:
                sol, info, ier, _ = fsolve(
                    self.determinant_longitudinal,
                    [seed,0.001],
                    args=(k, self.cut, self.n_suma),
                    xtol=self.sol_tol, maxfev=500, full_output=True
                )
                if ier==1 and omega_min <= sol[0] <= omega_max and abs(sol[1]) <= self.imag_tol:
                    roots.append([sol[0], sol[1]])
            except:
                pass
            # brentq
            a, b = seed-0.1, seed+0.1
            if a<omega_min: a=omega_min
            if b>omega_max: b=omega_max
            try:
                if det_real(a)*det_real(b) <0:
                    sol2 = root_scalar(det_real, bracket=[a,b], method='brentq')
                    roots.append([sol2.root, 0.0])
            except:
                pass

        # unicidad
        unique = []
        for r in roots:
            if not any(abs(r[0]-u[0])<1e-6 for u in unique):
                unique.append(r)
        return unique

    def robust_root_finding(self, k, omega_min, omega_max):
        # 1) intervalos de cambio de signo
        def eval_det(omega, k_val):
            try:
                return self.Det_longitudinal([omega, 0.01], k_val, self.cut)
            except:
                return [np.nan]
        def buscar_cambios_signo(w_min, w_max, k_val, n_puntos=50, max_nivel=2):
            """
            Busca intervalos donde el determinante cambia de signo.

            Se realiza un muestreo inicial uniforme y se detectan los
            intervalos donde `Re(det)` cambia de signo.  Cada
            intervalo se subdivide recursivamente hasta un máximo de
            `max_nivel` niveles para capturar raíces cercanas.
            """
            # Muestreo inicial
            w_array = np.linspace(w_min, w_max, n_puntos)
            det_vals = [eval_det(w, k_val)[0] for w in w_array]
            intervalos = []

            # Definir función interna para subdividir un intervalo si hay más de un cambio
            def subdividir(a, b, nivel):
                c = 0.5 * (a + b)
                try:
                    f_a = eval_det(a, k_val)[0]
                    f_c = eval_det(c, k_val)[0]
                    f_b = eval_det(b, k_val)[0]
                except:
                    return [(a, b)]
                subints = []
                if nivel >= max_nivel:
                    return [(a, b)]
                if not (np.isnan(f_a) or np.isnan(f_c)) and f_a * f_c < 0:
                    subints += subdividir(a, c, nivel + 1)
                if not (np.isnan(f_c) or np.isnan(f_b)) and f_c * f_b < 0:
                    subints += subdividir(c, b, nivel + 1)
                if not subints:
                    subints = [(a, b)]
                return subints

            for i in range(len(det_vals) - 1):
                fa, fb = det_vals[i], det_vals[i + 1]
                if np.isnan(fa) or np.isnan(fb):
                    continue
                if fa * fb < 0:
                    a, b = w_array[i], w_array[i + 1]
                    intervalos += subdividir(a, b, 0)
            return intervalos
        intervals = buscar_cambios_signo(omega_min, omega_max, k, n_puntos=500)
        all_roots = []
        # Si no hay intervalos, buscar mínimos locales
        if len(intervals) == 0:
             grid = np.linspace(omega_min, omega_max, 500)
             det_mag = []
             for w in grid:
                 det_re, det_im = self.determinantlongitudinal([w,0.0], k, self.cut, self.nsuma)
                 det_mag.append(det_re**2 + det_im**2)
             det_mag = np.array(det_mag)
             
             # Encontrar mínimos locales (picos invertidos)
             peaks, _ = find_peaks(-det_mag, prominence=0.1*np.max(-det_mag))
             
             # Crear pseudo-intervalos alrededor de cada mínimo
             for idx in peaks:
                 w_center = grid[idx]
                 w1 = max(omega_min, w_center - 0.05*(omega_max-omega_min))
                 w2 = min(omega_max, w_center + 0.05*(omega_max-omega_min))
                 intervals.append((w1, w2))
        # 2) refinar cada intervalo
        for w1, w2 in intervals:
            seeds = np.linspace(w1, w2, 5)
            roots = self.multi_algorithm_search(k, w1, w2, seeds)
            all_roots.extend(roots)
        # 3) búsqueda global para faltantes
        for _ in range(5):
            gr = self.global_root_finding(k, omega_min, omega_max)
            if gr and abs(gr[1]) <= self.imag_tol and not any(abs(gr[0]-r[0])<1e-6 for r in all_roots):
                all_roots.append(gr)
        # 4) verificación
        # 4) verificación
        expected_count, _ = self.contar_raices_en_intervalo(k, omega_min, omega_max)
        if len(all_roots) < expected_count:
            # Búsqueda local exhaustiva en malla fina para intervalos faltantes
            grid = np.linspace(omega_min, omega_max, 1000)
            dets = np.array([self.determinant_longitudinal([w, 0.001], k, self.cut, self.n_suma)[0]
                             for w in grid])
            # Encuentra ceros aproximados donde det cambia signo
            below = dets[:-1] * dets[1:] < 0
            for idx in np.where(below)[0]:
                w1, w2 = grid[idx], grid[idx+1]
                # Bisección rápida
                a, b = w1, w2
                for _ in range(8):
                    m = 0.5*(a + b)
                    if (self.determinant_longitudinal([a,0.001],k,self.cut,self.n_suma)[0] *
                        self.determinant_longitudinal([m,0.001],k,self.cut,self.n_suma)[0]) <= 0:
                        b = m
                    else:
                        a = m
                guess = 0.5*(a + b)
                sol, info, ier, _ = fsolve(
                    self.determinant_longitudinal,
                    [guess, 0.001],
                    args=(k, self.cut, self.n_suma),
                    xtol=1e-10,
                    maxfev=500,
                    full_output=True
                )
                if ier == 1 and not any(abs(sol[0]-r[0]) < 1e-6 for r in all_roots):
                    if abs(sol[1]) <= self.imag_tol:
                        all_roots.append([sol[0], sol[1]])

        return sorted(all_roots, key=lambda x: x[0])
    

    def zeros_fullgrid_subset(self):
        """
        Sustituye la búsqueda original de ceros longitudinales
        por el método híbrido robust_root_finding.
        Retorna frec[nk, nb, 2] con [f_real, f_imag].
        """
        nk = len(self.k)
        nb = self.nbands
        frec = np.full((nk, nb, 2), np.nan)
    
        # Límites de búsqueda para omega
        omega_min = 0.01
        omega_max = (2 * np.pi * self.vel0[1] / self.a) * 1.4
    
        for ik, kval in enumerate(self.k):
            # Obtiene raíces [f_real, f_imag]
            roots = self.robust_root_finding(kval, omega_min, omega_max)
            # Asigna hasta nb raíces
            for ib, (w_real, w_imag) in enumerate(roots[:nb]):
                frec[ik, ib, 0] = w_real
                frec[ik, ib, 1] = w_imag
    
        return frec

    def D(self, m, f):
        """D(m, f) entrega la matriz de transmision dentro de la celda unidad
        para un modo y frecuencia especificado, donde m es el modo y f la
        frecuencia. Esto solo es valido cuando los scatterers son cilindricos"""

        shear0, shears = self.shear
        lamb0, lambs = self._lame
        k0l = self.k0(f,0)
        k0t = self.k0(f,1)
        ksl = self.ks(f,0)
        kst = self.ks(f,1)
        r = self.r1
        
        # === BLOQUE A ===
        A1l0 = A1l(m, k0l, r, lamb0, shear0)
        A2l0 = A2l(m, k0l, r, lamb0, shear0)
        A3l0 = A3l(m, k0l, r, lamb0, shear0)
        A4l0 = A4l(m, k0l, r, lamb0, shear0)
        A1t0 = A1t(m, k0t, r, lamb0, shear0)
        A2t0 = A2t(m, k0t, r, lamb0, shear0)
        A3t0 = A3t(m, k0t, r, lamb0, shear0)
        A4t0 = A4t(m, k0t, r, lamb0, shear0)
        
        # === BLOQUE B ===
        B1l0 = B1l(m, k0l, r, lamb0, shear0)
        B2l0 = B2l(m, k0l, r, lamb0, shear0)
        B3l0 = B3l(m, k0l, r, lamb0, shear0)
        B4l0 = B4l(m, k0l, r, lamb0, shear0)
        B1t0 = B1t(m, k0t, r, lamb0, shear0)
        B2t0 = B2t(m, k0t, r, lamb0, shear0)
        B3t0 = B3t(m, k0t, r, lamb0, shear0)
        B4t0 = B4t(m, k0t, r, lamb0, shear0)
        
        # === BLOQUE C ===
        C1l0 = C1l(m, ksl, r, lambs, shears)
        C2l0 = C2l(m, ksl, r, lambs, shears)
        C3l0 = C3l(m, ksl, r, lambs, shears)
        C4l0 = C4l(m, ksl, r, lambs, shears)
        C1t0 = C1t(m, kst, r, lambs, shears)
        C2t0 = C2t(m, kst, r, lambs, shears)
        C3t0 = C3t(m, kst, r, lambs, shears)
        C4t0 = C4t(m, kst, r, lambs, shears)
        
        A1 = np.block([[A1l0, A1t0], [A2l0, A2t0]])
        B1 = np.block([[B1l0, B1t0], [B2l0, B2t0]])
        C1 = np.block([[C1l0, C1t0], [C2l0, C2t0]])
        A2 = np.block([[A3l0, A3t0], [A4l0, A4t0]])
        B2 = np.block([[B3l0, B3t0], [B4l0, B4t0]])
        C2 = np.block([[C3l0, C3t0], [C4l0, C4t0]])
        
        m0 = C2 @ inv(C1)
        m1 = inv(B2 - (m0 @ B1))
        m2 = m0 @ A1
        m3 = m2 - A2

        mat = m1 @ m3

        return mat

    def T_plane(self, f, pol1, pol2, cut):

        mat = np.zeros([2*cut+1, 2*cut+1], dtype=complex)

        for i in range(-cut, cut+1):

            mat[i+cut, i+cut] = self.D(i, f)[pol1, pol2]

        return mat
    
    def determinant_plane(self, f, k, cut, n_suma=5):
        """ Forma la matriz de transmision y el determinante a calcular """
        size = 2 * cut + 1
        mat = np.zeros([2*size, 2*size], dtype=complex)
        SlDll = self.T_plane(f, 0, 0, cut) @ self.G0(f, k, 0, cut);
        SlDlt = self.T_plane(f, 0, 1, cut) @ self.G0(f, k, 0, cut);
        StDtl = self.T_plane(f, 1, 0, cut) @ self.G0(f, k, 1, cut);
        StDtt = self.T_plane(f, 1, 1, cut) @ self.G0(f, k, 1, cut);
        mat = np.block([[SlDll,SlDlt],[StDtl,StDtt]])

        M = mat - np.identity(2*size)

        determinante = det(M)

        return [np.real(determinante), np.imag(determinante)]
    
    def Det_plane(self, frequency, *args):
        bloch_k, cutoff = args
        #print(self.determinant_plane(frequency, bloch_k, cutoff, n_suma=self.n_suma) )
        return self.determinant_plane(frequency, bloch_k, cutoff, n_suma=self.n_suma) 

    def zeros_plane_fullgrid(self, C_l0, ventanas_por_unidad=100, 
                                            w_norm_max=1.25, buscar_todas=True,
                                            use_external_output=False, out_freq_dir=None,
                                            out_fig_dir=None):
        """
        Calcula la estructura de bandas usando un método híbrido optimizado.
        Combina tres estrategias para un equilibrio óptimo entre velocidad y fiabilidad:
        1. Continuación: Intervalo centrado en soluciones previas (rápido y mejorado).
        2. Cambio de Signo: Busca intervalos con raíces (robusto).
        3. Búsqueda Ciega: Último recurso si todo lo anterior falla.
        """
        if use_external_output:
            if out_freq_dir is not None:
                self.frecfolder = str(out_freq_dir)
            if out_fig_dir is not None:
                self.foldername = str(out_fig_dir)
        
        # En todos los casos, asegúrate de que existan
        Path(self.frecfolder).mkdir(parents=True, exist_ok=True)
        Path(self.foldername).mkdir(parents=True, exist_ok=True)
        print("\n Iniciando método híbrido y robusto de búsqueda para todos los k ...")
        start_time = time.time()

        # --- Parámetros ---
        tolerancia_raices = self.sol_tol
        tolerancia_imaginaria = self.imag_tol

        punto_K = 2 * np.pi / (3 * self.a)

        # --- Preparación de k-points ---
        if self.lattice == 'hx' and not np.any(np.isclose(self.k, punto_K, atol=1e-6)):
            self.k = np.append(self.k, punto_K)
        self.k = np.sort(self.k)
        self.nk = len(self.k)

        # --- Inicialización ---
        nmax = 5 if buscar_todas else self.nbands
        frec = np.full((self.nk, nmax, 2), np.nan)
        # No utilizaremos soluciones_anteriores para la continuidad.
        # Al prescindir de las semillas de iteraciones previas, cada k
        # se procesa de forma independiente basándose únicamente en
        # cambios de signo.

        # --- Funciones auxiliares ---
        def eval_det(omega, k_val):
            try:
                return self.Det_plane([omega, 0.1], k_val, self.cut)
            except:
                return [np.nan]

        def resolver_con_fsolve(semilla_w, k_val):
            try:
                sol, info, ier, _ = fsolve(
                    self.Det_plane, [semilla_w, 0.1],
                    args=(k_val, self.cut), xtol=tolerancia_raices,
                    full_output=True
                )
                w_real, w_imag = sol
                w_norm = (w_real * self.a) / (2 * np.pi * C_l0)
                if ier == 1 and abs(w_imag) < tolerancia_imaginaria and 0 < w_norm < w_norm_max:
                    return (w_real, w_imag)
            except:
                return None
            return None

        def buscar_cambios_signo(w_min, w_max, k_val, n_puntos=50, max_nivel=2):
            """
            Busca intervalos donde el determinante cambia de signo.

            Se realiza un muestreo inicial uniforme y se detectan los
            intervalos donde `Re(det)` cambia de signo.  Cada
            intervalo se subdivide recursivamente hasta un máximo de
            `max_nivel` niveles para capturar raíces cercanas.
            """
            # Muestreo inicial
            w_array = np.linspace(w_min, w_max, n_puntos)
            det_vals = [eval_det(w, k_val)[0] for w in w_array]
            intervalos = []

            # Definir función interna para subdividir un intervalo si hay más de un cambio
            def subdividir(a, b, nivel):
                c = 0.5 * (a + b)
                try:
                    f_a = eval_det(a, k_val)[0]
                    f_c = eval_det(c, k_val)[0]
                    f_b = eval_det(b, k_val)[0]
                except:
                    return [(a, b)]
                subints = []
                if nivel >= max_nivel:
                    return [(a, b)]
                if not (np.isnan(f_a) or np.isnan(f_c)) and f_a * f_c < 0:
                    subints += subdividir(a, c, nivel + 1)
                if not (np.isnan(f_c) or np.isnan(f_b)) and f_c * f_b < 0:
                    subints += subdividir(c, b, nivel + 1)
                if not subints:
                    subints = [(a, b)]
                return subints

            for i in range(len(det_vals) - 1):
                fa, fb = det_vals[i], det_vals[i + 1]
                if np.isnan(fa) or np.isnan(fb):
                    continue
                if fa * fb < 0:
                    a, b = w_array[i], w_array[i + 1]
                    intervalos += subdividir(a, b, 0)
            return intervalos

        def resolver_en_intervalo_continuacion(w_centro, k_val, ancho_rel=0.025, n_semillas=51):
            """
            Busca raíces alrededor de una solución anterior usando un intervalo relativo.
            """
            soluciones_locales = []
            ancho = ancho_rel * w_centro
            w_array = np.linspace(w_centro - ancho, w_centro + ancho, n_semillas)

            for w in w_array:
                try:
                    det1 = eval_det(w - 1e-5, k_val)[0]
                    det2 = eval_det(w + 1e-5, k_val)[0]
                    cambio_signo = det1 * det2 < 0
                except:
                    cambio_signo = False

                if cambio_signo:
                    resultado = resolver_con_fsolve(w, k_val)
                    if resultado and not any(np.isclose(resultado[0], s[0], rtol=1e-2) for s in soluciones_locales):
                        soluciones_locales.append(resultado)
            return soluciones_locales

        # --- Búsqueda principal ---
        barra = tqdm(range(self.nk), desc="Búsqueda por k", dynamic_ncols=True)
        for idx_k in barra:
            k_val = self.k[idx_k]
            soluciones = []
            es_punto_K = self.lattice == 'hx' and k_val == punto_K

            # Sólo usamos la estrategia de cambio de signo para detectar intervalos con raíces.
            minimi = 1e-3
            w_min = 2 * np.pi * C_l0 * minimi / self.a
            w_max = 2 * np.pi * C_l0 * w_norm_max / self.a
            n_puntos_busqueda = int(ventanas_por_unidad * (w_norm_max - minimi))
            # Buscar intervalos donde el determinante cambia de signo
            intervalos = buscar_cambios_signo(w_min, w_max, k_val, n_puntos=n_puntos_busqueda)
            for w1, w2 in intervalos:
                sol = resolver_con_fsolve(0.5 * (w1 + w2), k_val)
                if sol:
                    w_real = sol[0]
                    # Normalización de la frecuencia real
                    w_norm_local = (w_real * self.a) / (2 * np.pi * C_l0)
                    # Ignorar raíces con frecuencia real muy pequeña
                    if w_norm_local < 1e-3:
                        continue
                    # Eliminar duplicados: si ya existe una raíz muy cercana, no la añadimos
                    if not any(np.isclose(sol[0], s[0], rtol=1e-4) for s in soluciones):
                        soluciones.append(sol)

            # Ordenar soluciones por frecuencia real y recortar si se desean menos bandas
            soluciones = sorted(soluciones, key=lambda x: x[0])
            if not buscar_todas:
                soluciones = soluciones[:self.nbands]

            for b, s in enumerate(soluciones):
                if b < nmax:
                    frec[idx_k, b, :] = s

            # Corrección en el punto K (cono de Dirac)
            if es_punto_K and self.nbands >= 2:
                w1 = frec[idx_k, 0, 0]
                w2 = frec[idx_k, 1, 0]
                if not np.isnan(w1) and not np.isnan(w2) and abs(w2 - w1) > 7:
                    print(f"Cono de Dirac en K: Δf = {abs(w2 - w1):.2f} Hz -> Duplicando banda 1")
                    frec[idx_k, 1, :] = frec[idx_k, 0, :]

            # Guardar resultados de este k en disco
            np.savetxt(os.path.join(self.frecfolder, f"frecuencias{self.k[idx_k]:.4f}.txt"), frec[idx_k])

        self.omega_longitudinal = frec
        # Graficar las bandas calculadas
        self.graficar_bandas_grid()

        # Guardar resultados completos en un archivo CSV
        # Cada fila del CSV contendrá: valor de k, índice de banda (1-based),
        # frecuencia real y frecuencia imaginaria.  Se ignoran entradas NaN.
        try:
            rows = []
            for idx_k, k_val in enumerate(self.k):
                for band_idx in range(nmax):
                    w_real, w_imag = frec[idx_k, band_idx, :]
                    if not np.isnan(w_real):
                        rows.append({
                            'k': float(k_val),
                            'band': int(band_idx + 1),
                            'w_real': float(w_real),
                            'w_imag': float(w_imag),
                        })
            if rows:
                df = pd.DataFrame(rows)
                csv_path = os.path.join(self.frecfolder, 'bandas_longitudinales.csv')
                df.to_csv(csv_path, index=False)
                print(f"Resultados guardados en CSV: {csv_path}")
        except Exception as e:
            # En caso de error al guardar, se informa pero no se interrumpe
            print(f"Advertencia: no se pudo guardar el CSV ({e})")

        print(f"Completado en {time.time() - start_time:.2f} s")

    def swap_omega_longitudinal_rows(self, i, j, indices=None, k_min=None, k_max=None):
        """
        Intercambia las bandas i y j de self.omega_longitudinal solo en un subset de self.k.
    
        Parámetros
        ----------
        i, j : int
            Índices de banda a intercambiar (pueden ser negativos tipo Python).
        indices : iterable[int] | None
            Índices (respecto de self.k) donde aplicar el swap. Si se entrega, tiene prioridad.
        k_min, k_max : float | None
            Rango en k para seleccionar subset si 'indices' es None.
    
        Devuelve
        --------
        idx : np.ndarray[int]
            Índices de k (globales) en los que se aplicó el intercambio.
        """
        import numpy as np
    
        arr = getattr(self, "omega_longitudinal", None)
        if arr is None:
            raise ValueError("omega_longitudinal no existe en el objeto.")
        if arr.ndim not in (2, 3):
            raise ValueError("omega_longitudinal debe ser 2D (nk, nb) o 3D (nk, nb, 2).")
    
        k_full = np.asarray(self.k)
        nk = k_full.shape[0]
        if arr.shape[0] != nk:
            raise ValueError(f"Dimensión incompatible: omega_longitudinal.shape[0]={arr.shape[0]} != len(k)={nk}")
    
        nb = arr.shape[1]
        # normalizar índices de banda a [0, nb)
        ii = i % nb
        jj = j % nb
        if ii == jj:
            return np.array([], dtype=int)  # nada que hacer
    
        # construir subset de k
        if indices is not None:
            idx = np.unique(np.asarray(list(indices), dtype=int))
        else:
            mask = np.ones(nk, dtype=bool)
            if k_min is not None:
                mask &= (k_full >= float(k_min))
            if k_max is not None:
                mask &= (k_full <= float(k_max))
            idx = np.where(mask)[0]
    
        if idx.size == 0:
            return idx
    
        # swap
        if arr.ndim == 3:
            # forma (nk, nb, 2) típico: (Re, Im)
            for kpos in idx:
                arr[kpos, [ii, jj], :] = arr[kpos, [jj, ii], :]
        else:
            # forma (nk, nb)
            for kpos in idx:
                arr[kpos, [ii, jj]] = arr[kpos, [jj, ii]]
    
        self.omega_longitudinal = arr
        return idx

    def smooth_interpolate_longitudinal(
        self,
        method: str = "pchip",              # 'linear' | 'pchip' | 'akima' | 'cubic'
        keep_edges_nan: bool = True,
        smooth: str | None = None,          # None | 'savgol' | 'spline'
        smooth_params: dict | None = None,
        bands=None,                         # None/'all' o iterable de índices de banda
        in_place: bool = True               # True -> actualiza self.omega_longitudinal
    ):
        """
        Rellena NaN internos banda por banda usando smooth_interpolate(k, omega_banda).
        Mantiene el formato (nk, nb, 2) [Re, Im].
    
        Retorna el array rellenado (y si in_place=True, también actualiza self.omega_longitudinal).
        """
        import numpy as np
    
        W = np.asarray(self.omega_longitudinal)
        if W.ndim != 3 or W.shape[2] < 2:
            raise ValueError("omega_longitudinal debe tener shape (nk, nb, 2)")
    
        k = np.asarray(self.k).reshape(-1)
        nk, nb = W.shape[0], W.shape[1]
        if k.size != nk:
            raise ValueError("len(self.k) debe coincidir con omega_longitudinal.shape[0]")
    
        # Selección de bandas
        if bands is None or bands == "all":
            band_idx = range(nb)
        else:
            band_idx = list(bands)
    
        # Convertir a complejo
        Wc = W[:, :, 0] + 1j * W[:, :, 1]
        Wc_filled = Wc.copy()
    
        # Llamar a la función base por banda
        for b in band_idx:
            y = Wc[:, b]
            # Usa tu smooth_interpolate ya definida en el módulo/clase
            y_f = smooth_interpolate(
                k, y,
                method=method,
                keep_edges_nan=keep_edges_nan,
                smooth=smooth,
                smooth_params=smooth_params
            )
            Wc_filled[:, b] = y_f
    
        # Volver a (nk, nb, 2)
        W_out = np.empty_like(W, dtype=float)
        W_out[:, :, 0] = np.real(Wc_filled)
        W_out[:, :, 1] = np.imag(Wc_filled)
    
        if in_place:
            self.omega_longitudinal = W_out
        return W_out

def smooth_interpolate(
    k,
    omega,
    method: str = "pchip",           # 'linear' | 'pchip' | 'akima' | 'cubic'
    keep_edges_nan: bool = True,
    smooth: str | None = None,       # None | 'savgol' | 'spline'
    smooth_params: dict | None = None
):
    """
    Interpola omega(k) rellenando NaN internos y, opcionalmente, aplica un suavizado.
    - Interpola solo dentro del rango [min(k válidos), max(k válidos)].
    - Si keep_edges_nan=True, deja NaN fuera de ese rango.
    - smooth:
        'savgol' -> Savitzky–Golay (params: window, polyorder)
        'spline' -> UnivariateSpline (params: s (float), k (int=3..5))
    """
    k = np.asarray(k).reshape(-1)
    y = np.asarray(omega).reshape(-1)
    if y.size != k.size:
        raise ValueError("len(k) debe coincidir con len(omega)")

    if np.iscomplexobj(y):
        valid = np.isfinite(y.real) & np.isfinite(y.imag)
    else:
        valid = np.isfinite(y)

    if valid.sum() < 2:
        # No hay suficientes puntos para interpolar
        return omega.copy()

    # Orden estable y máscara
    idx = np.argsort(k, kind="mergesort")
    inv = np.empty_like(idx); inv[idx] = np.arange(idx.size)
    ks, ys, vs = k[idx], y[idx], valid[idx]

    x_min, x_max = ks[vs].min(), ks[vs].max()
    inside = (ks >= x_min) & (ks <= x_max)
    to_fill = (~vs) & inside

    def _fit_eval(x, yr, xi, kind):
        if kind == "linear":
            return np.interp(xi, x, yr)
        elif kind == "pchip":
            return PchipInterpolator(x, yr, extrapolate=False)(xi)
        elif kind == "akima":
            return Akima1DInterpolator(x, yr)(xi)
        elif kind == "cubic":
            return CubicSpline(x, yr, bc_type="not-a-knot", extrapolate=False)(xi)
        else:
            raise ValueError("method desconocido")

    ys_filled = ys.copy()
    if to_fill.any():
        if np.iscomplexobj(ys):
            xr = ks[vs]; yr = ys[vs].real; yi = ys[vs].imag
            yr_i = _fit_eval(xr, yr, ks[to_fill], method)
            yi_i = _fit_eval(xr, yi, ks[to_fill], method)
            ys_filled[to_fill] = yr_i + 1j*yi_i
        else:
            ys_filled[to_fill] = _fit_eval(ks[vs], ys[vs].astype(float), ks[to_fill], method)

    # Opcional: suavizado sobre los puntos interiores válidos (post-interpolación)
    if smooth is not None:
        params = smooth_params.copy() if smooth_params else {}
        mask_smooth = inside & np.isfinite(ys_filled if not np.iscomplexobj(ys_filled)
                                           else ys_filled.real)
        xs = ks[mask_smooth]

        def _apply_smooth(vec):
            if vec.size < 5:
                return vec  # demasiado corto para suavizar con sentido
            if smooth == "savgol":
                # parámetros por defecto robustos
                w = int(params.get("window", 7))
                p = int(params.get("polyorder", 2))
                # ventana impar y <= tamaño
                w = max(3, min(w | 1, vec.size - (1 - vec.size % 2)))  # fuerza impar
                if w <= p:
                    p = max(1, min(p, w - 1))
                try:
                    return savgol_filter(vec, window_length=w, polyorder=p, mode="interp")
                except Exception:
                    return vec
            elif smooth == "spline":
                s = float(params.get("s", 0.0))   # 0 => interpola exactamente
                kspl = int(params.get("k", 3))    # grado 3..5 típico
                kspl = max(1, min(5, kspl))
                # Evita fallar si hay puntos repetidos en x
                xu, idx_u = np.unique(xs, return_index=True)
                yu = vec[idx_u]
                if xu.size <= kspl:
                    return vec
                try:
                    spl = UnivariateSpline(xu, yu, s=s, k=kspl, ext=0)  # ext=0 => NaN fuera
                    return spl(xs)
                except Exception:
                    return vec
            else:
                raise ValueError("smooth desconocido")

        if np.iscomplexobj(ys_filled):
            yr = ys_filled[mask_smooth].real
            yi = ys_filled[mask_smooth].imag
            yr_s = _apply_smooth(yr)
            yi_s = _apply_smooth(yi)
            tmp = ys_filled.copy()
            tmp[mask_smooth] = yr_s + 1j*yi_s
            ys_filled = tmp
        else:
            yr = ys_filled[mask_smooth].astype(float)
            yr_s = _apply_smooth(yr)
            tmp = ys_filled.copy()
            tmp[mask_smooth] = yr_s
            ys_filled = tmp

    # Bordes
    if not keep_edges_nan:
        left = ks < x_min; right = ks > x_max
        if left.any():
            ys_filled[left] = ys_filled[inside][0]
        if right.any():
            ys_filled[right] = ys_filled[inside][-1]

    return ys_filled[inv]



def smooth_interpolate_bands(frec):
    """
    Interpola suavemente valores NaN en frec real.
        frec: ndarray shape (nk, nbands, 2) con [f_real, f_imag].
        Devuelve ndarray igual, con:
          - f_real interpolado donde faltaba,
          - f_imag original donde existía, 0 donde se interpoló.
    """
    nk, nbands,_ = frec.shape
    frec_out = np.empty_like(frec)

    for b in range(nbands):
        # Extraer componentes
        real = frec[:, b, 0]
        imag = frec[:, b, 1]

        # Serie de parte real
        serie_real = pd.Series(real)
        real_interp = serie_real.interpolate(
            method='linear', limit=1, limit_direction='both'
        ).values

        # Máscara de valores originales (no NaN en real)
        mask_orig = ~np.isnan(real)

        # Imag original donde mask_orig, 0 donde no
        imag_out = np.where(mask_orig, imag, 0.0)

        frec_out[:, b, 0] = real_interp
        frec_out[:, b, 1] = imag_out

    return frec_out


