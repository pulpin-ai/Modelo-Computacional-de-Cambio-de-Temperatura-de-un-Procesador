import numpy as np
import matplotlib.pyplot as plt
from time import time

# -------------------------
# Parámetros físicos / malla
# -------------------------
L = 0.02
Nx, Ny = 101, 101
h = L / (Nx - 1)
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

k = 130.0            # conductividad térmica [W/(m·K)]
P_total = 120.0      # potencia total [W]
factor_tamano = 8    # el núcleo = L/factor_tamano (puedes cambiar)

# -------------------------
# Funciones auxiliares
# -------------------------
def crear_mascara_nucleo(X, Y, L, factor_tamano=8):
    nucleo_L = L / factor_tamano
    x_min = L/2 - nucleo_L/2
    x_max = L/2 + nucleo_L/2
    y_min = L/2 - nucleo_L/2
    y_max = L/2 + nucleo_L/2
    return np.logical_and.reduce((X >= x_min, X <= x_max, Y >= y_min, Y <= y_max))

def calcular_qg_y_C(P_nucleo, L, k, h, factor_tamano=8):
    A_nucleo = (L / factor_tamano)**2
    t_espesor = 1.0           # modelo 2D
    q_nucleo = P_nucleo / (A_nucleo * t_espesor)   # W/m^3
    C = q_nucleo * h**2 / (4 * k)
    return q_nucleo, C

# -------------------------
# Máscara y constantes
# -------------------------
nucleo_mask = crear_mascara_nucleo(X, Y, L, factor_tamano=factor_tamano)
qg_value, C_value = calcular_qg_y_C(P_total, L, k, h, factor_tamano=factor_tamano)

# Construir array C_mask que contiene C_value en el núcleo y 0 fuera
C_mask = np.zeros_like(X)
C_mask[nucleo_mask] = C_value

def solve_poisson_sor(C_mask, omega=1.7, T_bound=300.0, tol=1e-6, max_iter=20000, verbose=False):
    T = np.ones_like(C_mask) * T_bound
    T[0, :] = T_bound;
    T[-1, :] = T_bound;
    T[:, 0] = T_bound;
    T[:, -1] = T_bound
    start = time()
    for iteracion in range(1, max_iter+1):
        T_old = T.copy()
        # in-place SOR
        for i in range(1, T.shape[0]-1):
            for j in range(1, T.shape[1]-1):
                t_new = 0.25*(T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1]) + C_mask[i,j]
                T[i,j] = (1-omega)*T[i,j] + omega * t_new
        err_iter = np.max(np.abs(T - T_old))
        if iteracion % 200 == 0 and verbose:
            print(f"SOR iteracion={iteracion} err_iter={err_iter:.3e}")
        if err_iter < tol:
            if verbose: print("Convergencia (SOR) en", iteracion, "iteraciones")
            break
    elapsed = time() - start
    return T, iteracion, err_iter, elapsed

tol = 1e-6
max_iter = 20000
omega = 1.7

print("Resolviendo con SOR (omega=", omega, ") ...")
T_sor, it_sor, err_sor, t_sor = solve_poisson_sor(C_mask, omega=omega, T_bound=300.0, tol=tol, max_iter=max_iter, verbose=True)
print(f"SOR: iter={it_sor}, err={err_sor:.3e}, tiempo={t_sor:.2f}s")
