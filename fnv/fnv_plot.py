from __future__ import annotations
from typing import Sequence
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from .fnv_store import build_fnv_grid

def _ensure_iter(x):
    return x if isinstance(x, (list, tuple, np.ndarray)) else [x]

def _pick_r(red, r1, r2, r_opt):
    if r_opt is not None:
        return np.asarray(r_opt, dtype=float)
    if getattr(red, "fnv_data", None) and red.fnv_data.default_key:
        blk = red.fnv_data.blocks[red.fnv_data.default_key]
        return blk.r
    return np.linspace(r1, r2, 50, dtype=float)

# Paleta fija por canal: b -> azul, c -> rojo
_COL = {"b": "#1f77b4", "c": "#d62728"}
# Marcadores para frecuencias 2..N (la 1ª es línea sólida)
_MARKERS = ['o', '^', 's', 'd', 'v', '>', '<', 'p', '*', 'h']

def _make_legends(ax, freqs, kind_label):
    """
    Crea dos leyendas:
      - canales (colores): F^(b)/F^(c) o V^(b)/V^(c)
      - frecuencias: ω=f0 (línea), ω=f_i (marcadores)
    """
    # Leyenda de canales (colores)
    if kind_label == "F":
        lab_b = r'$F_n^{(b)}$'
        lab_c = r'$F_n^{(c)}$'
    else:
        lab_b = r'$V_n^{(b)}$'
        lab_c = r'$V_n^{(c)}$'

    chan_handles = [
        Line2D([0], [0], color=_COL["b"], lw=2.0, label=lab_b),
        Line2D([0], [0], color=_COL["c"], lw=2.0, label=lab_c),
    ]
    leg_chan = ax.legend(chan_handles, [lab_b, lab_c], loc='upper left', fontsize=9, frameon=True)
    ax.add_artist(leg_chan)

    # Leyenda de frecuencias (línea para la primera, marcadores para el resto)
    freq_handles = []
    freq_labels = []
    if len(freqs) >= 1:
        f0 = freqs[0]
        freq_handles.append(Line2D([0], [0], color='black', lw=2.0))  # línea sólida
        freq_labels.append(rf'$\omega={f0}$')
    for i, f in enumerate(freqs[1:], start=1):
        mk = _MARKERS[(i-1) % len(_MARKERS)]
        freq_handles.append(Line2D([0], [0], color='black', marker=mk, linestyle='', markersize=6))
        freq_labels.append(rf'$\omega={f}$')

    if freq_handles:
        ax.legend(freq_handles, freq_labels, loc='upper right', fontsize=9, frameon=True, ncol=1)

# ---------------- helpers: un modo n por llamada ----------------
def _plot_F_component_single(red, n: int, freqs: Sequence[float], r: np.ndarray,
                             which: str, ax, r1: float, r2: float):
    """
    Dibuja F_n para un n fijo:
      - ω[0]: línea sólida (canales b/c en colores)
      - ω[1:]: solo marcadores (mismos colores por canal)
    """
    # Construir/usar caché
    build_fnv_grid(red, [n], list(freqs), r)
    blk = red.fnv_data.get_block(r)
    i_n = blk.n_list.index(n)

    for j, f in enumerate(freqs):
        j_f = blk.freq_list.index(f)
        Fb = blk.data[("F","b")][i_n, j_f, :]
        Fc = blk.data[("F","c")][i_n, j_f, :]

        yb = np.real(Fb) if which == "Re" else np.imag(Fb)
        yc = np.real(Fc) if which == "Re" else np.imag(Fc)

        if j == 0:
            # primera frecuencia: línea sólida
            ax.plot(r, yb, color=_COL["b"], linestyle='-', label=None)
            ax.plot(r, yc, color=_COL["c"], linestyle='-', label=None)
        else:
            # resto: solo marcadores
            mk = _MARKERS[(j-1) % len(_MARKERS)]
            ax.plot(r, yb, color=_COL["b"], linestyle='', marker=mk, markersize=4, label=None)
            ax.plot(r, yc, color=_COL["c"], linestyle='', marker=mk, markersize=4, label=None)

        # contexto legacy (última serie de este loop)
        red._legacy_context = {"Fnb": yb, "Fnc": yc, "r": r, "n": n, "freq": f}

def _plot_V_component_single(red, n: int, freqs: Sequence[float], r: np.ndarray,
                             which: str, ax, r1: float, r2: float):
    build_fnv_grid(red, [n], list(freqs), r)
    blk = red.fnv_data.get_block(r)
    i_n = blk.n_list.index(n)

    for j, f in enumerate(freqs):
        j_f = blk.freq_list.index(f)
        Vb = blk.data[("V","b")][i_n, j_f, :]
        Vc = blk.data[("V","c")][i_n, j_f, :]

        yb = np.real(Vb) if which == "Re" else np.imag(Vb)
        yc = np.real(Vc) if which == "Re" else np.imag(Vc)

        if j == 0:
            ax.plot(r, yb, color=_COL["b"], linestyle='-', label=None)
            ax.plot(r, yc, color=_COL["c"], linestyle='-', label=None)
        else:
            mk = _MARKERS[(j-1) % len(_MARKERS)]
            ax.plot(r, yb, color=_COL["b"], linestyle='', marker=mk, markersize=4, label=None)
            ax.plot(r, yc, color=_COL["c"], linestyle='', marker=mk, markersize=4, label=None)

        red._legacy_context.update({"Vnb": yb, "Vnc": yc, "r": r, "n": n, "freq": f})

# ---------------- API compatibles con Red (n escalar o iterable) ----------------
def plot_Re_fn(red, n, freq, r1, r2, r_opt=None):
    n_list = _ensure_iter(n)
    freqs = _ensure_iter(freq)
    r = _pick_r(red, r1, r2, r_opt)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n_i in n_list:
        _plot_F_component_single(red, int(n_i), freqs, r, "Re", ax, r1, r2)

    _make_legends(ax, freqs, kind_label="F")
    ax.set_xlabel(r'r')
    ax.set_ylabel(r'Re($F_n$)')
    ax.set_title('Modos: ' + ', '.join(str(int(nn)) for nn in n_list))
    return fig, ax

def plot_Im_fn(red, n, freq, r1, r2, r_opt=None):
    n_list = _ensure_iter(n)
    freqs = _ensure_iter(freq)
    r = _pick_r(red, r1, r2, r_opt)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n_i in n_list:
        _plot_F_component_single(red, int(n_i), freqs, r, "Im", ax, r1, r2)

    _make_legends(ax, freqs, kind_label="F")
    ax.set_xlabel(r'r')
    ax.set_ylabel(r'Im($F_n$)')
    ax.set_title('Modos: ' + ', '.join(str(int(nn)) for nn in n_list))
    return fig, ax

def plot_Re_vn(red, n, freq, r1, r2, r_opt=None):
    n_list = _ensure_iter(n)
    freqs = _ensure_iter(freq)
    r = _pick_r(red, r1, r2, r_opt)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n_i in n_list:
        _plot_V_component_single(red, int(n_i), freqs, r, "Re", ax, r1, r2)

    _make_legends(ax, freqs, kind_label="V")
    ax.set_xlabel(r'r')
    ax.set_ylabel(r'Re($V_n$)')
    ax.set_title('Modos: ' + ', '.join(str(int(nn)) for nn in n_list))
    return fig, ax

def plot_Im_vn(red, n, freq, r1, r2, r_opt=None):
    n_list = _ensure_iter(n)
    freqs = _ensure_iter(freq)
    r = _pick_r(red, r1, r2, r_opt)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n_i in n_list:
        _plot_V_component_single(red, int(n_i), freqs, r, "Im", ax, r1, r2)

    _make_legends(ax, freqs, kind_label="V")
    ax.set_xlabel(r'r')
    ax.set_ylabel(r'Im($V_n$)')
    ax.set_title('Modos: ' + ', '.join(str(int(nn)) for nn in n_list))
    return fig, ax
