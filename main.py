# main.py
# Script de prueba para la clase Red: genera gráfico del coeficiente de dispersión

from Bandas_Four_Materiales_Tools_prima2_new import Red
import numpy as np
if __name__ == "__main__":
    # ----------------------------------------------------------
    # 1. Crear una instancia de Red con nombres ficticios de materiales
    # ----------------------------------------------------------
    for x in [1]:
        red = Red(comp=["matriz", "inclusion"])
    
        # ----------------------------------------------------------
        # 2. Definir parámetros físicos de prueba
        # ----------------------------------------------------------
        red.dens = [1150, 1250]               # Densidades [kg/m^3]
        red.vel0 = [295, 295]               # Velocidades en la matriz: [Cl0, Ct0] [m/s]
        red.vels = [894, 894]              # Velocidades en la inclusión: [Cls, Cts] [m/s]
        red.filling = 0.5                    # Fracción de llenado
        red.cut = 6       
        red.nbands = 5             # Número de bandas a calcular
        red.nk = 50
        red.n_suma = 5
        red.lattice='sq'
        red.psi=x
        red.a= 0.1
        red._set_k_end()
        red.cond_borde='rigid'
        red.imag_tol = 0.8
        red.sol_tol =1e-2
    
    
                               # Número de puntos en k
    
        # ----------------------------------------------------------
        # 3. Calcular parámetros derivados y vector k
        # ----------------------------------------------------------
        red.asign_param()                    # Esto inicializa self.k, self.a, y carpetas
        red.r1 = 0.45*red.a
        red.r2 = 0.5*red.a
        print('psi=',red.psi,' r1=', red.r1,' r2=', red.r2, ' imag_tol=', red.imag_tol, ' sol_tol=', red.sol_tol)
        #red.graficar_dispersion_coef(0.05, x, frequency=np.linspace(0,1.5,100))
        red.graficar_dif_determinante(0, 0.05, np.pi, 6)
        #red.graficar_dif_dispersion_coef(0, 0.05, [-1,0,1])
        