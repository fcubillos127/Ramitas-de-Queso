"""
Puente entre las bandas por autovalores (scripts_figs) y las herramientas de
edicion manual de la clase Red (delete_point, order_bands_by_continuity_global,
smooth_interpolate_longitudinal, graficar_bandas_grid), que operan sobre
self.omega_longitudinal.

Flujo tipico (en VSCode):

    from scripts_figs.bridge_to_omega import eig_to_red, red_to_eig_npz
    red = eig_to_red("data/bands_sq_c7.npz", psi_index=4, imtol=0.12)  # psi=0.8
    red.order_bands_by_continuity_global()      # reordena por continuidad
    red.delete_point(i=30, n=5)                 # borra un espurio (con undo: restore_deleted)
    red.smooth_interpolate_longitudinal()       # rellena huecos internos
    red.graficar_bandas_grid(ylim=[0, 1.4])     # grafica (estilo del codigo)
    red_to_eig_npz(red, "data/bands_sq_edit.npz", psi=0.8)  # exporta lo editado
    # y luego, para la figura con segmentos equiespaciados:
    #   python scripts_figs/plot_bands.py data/bands_sq_edit.npz graphs/bandas_sq_edit
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from bandcalc import build_red, CT0


def eig_to_red(npz, psi_index=0, imtol=0.12, cut=None, n_suma=5,
               imag_tol=0.8, sol_tol=1e-2, cond_borde="hollow",
               r1=0.45, r2=0.5, frecfolder=None):
    """Crea un Red con self.omega_longitudinal poblado desde el .npz de bandas
    (metodo de autovalores). Enmascara los modos de fuga con |Im(mu)|>imtol.
    Deja el objeto listo para las rutinas de edicion de la clase.

    cut=None -> se lee del .npz (guardado por compute_driver.run) si esta
    disponible; si no, usa 2. n_suma, imag_tol, sol_tol, cond_borde, r1, r2 solo
    importan si luego corres ademas el solver original de Miguel sobre este
    mismo objeto (zeros_longitudinal_fullgrid)."""
    d = np.load(npz)
    lattice = str(d["lattice"]); a = float(d["a"]); Ct0 = float(d["Ct0"])
    psi = float(d["psis"][psi_index])
    k = np.asarray(d["k_%d" % psi_index])
    wn = np.array(d["wn_%d" % psi_index]).copy()
    if ("im_%d" % psi_index) in d.files:
        wn[np.array(d["im_%d" % psi_index]) > imtol] = np.nan
    if cut is None:
        cut = int(d["cut"]) if "cut" in d.files else 2

    r = build_red(lattice, psi, cut=cut, nk=len(k), n_suma=n_suma,
                  imag_tol=imag_tol, sol_tol=sol_tol, cond_borde=cond_borde,
                  r1=r1, r2=r2, a=a)
    r.k = k
    r.nk = len(k)
    r.nbands = wn.shape[1]
    # omega_longitudinal guarda w_real (rad/s) e w_imag; graficar_bandas_grid
    # re-normaliza con a/(2*pi*vel0[1]).
    om = np.full((len(k), wn.shape[1], 2), np.nan)
    om[:, :, 0] = wn * 2*np.pi*Ct0 / a
    om[:, :, 1] = 0.0
    r.omega_longitudinal = om
    # carpeta de salida para graficar_bandas_grid / borrados en disco
    r.frecfolder = frecfolder or os.path.join(os.getcwd(), "edit_%s_psi%s" % (lattice, str(psi)))
    os.makedirs(r.frecfolder, exist_ok=True)
    return r


def red_to_eig_npz(red, out, psi=None, imag=None):
    """Exporta red.omega_longitudinal al formato .npz que lee plot_bands.py
    (un solo psi), para replotear lo editado con segmentos equiespaciados."""
    a = red.a; Ct0 = float(red.vel0[1])
    wn = red.omega_longitudinal[:, :, 0] * a / (2*np.pi*Ct0)
    im = np.zeros_like(wn) if imag is None else imag
    data = {
        "lattice": red.lattice, "a": a, "Ct0": Ct0, "wmax": 1.4,
        "psis": np.array([psi if psi is not None else getattr(red, "psi", 0.0)]),
        "k_0": np.asarray(red.k), "wn_0": wn, "im_0": im,
    }
    np.savez(out, **data)
    return out


if __name__ == "__main__":
    # demo/autotest: carga, reordena, borra, suaviza y exporta (sin ventana)
    import matplotlib; matplotlib.use("Agg")
    npz = sys.argv[1] if len(sys.argv) > 1 else "bands_sq_c7.npz"
    r = eig_to_red(npz, psi_index=0)
    print("omega_longitudinal:", r.omega_longitudinal.shape, " nbands:", r.nbands)
    r.order_bands_by_continuity_global()
    print("order_bands_by_continuity_global: OK")
    r.smooth_interpolate_longitudinal()
    print("smooth_interpolate_longitudinal: OK")
