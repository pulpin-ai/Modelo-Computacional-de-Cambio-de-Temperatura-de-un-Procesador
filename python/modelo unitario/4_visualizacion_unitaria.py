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

# -------------------------
# Ejecutar solver elegido
# -------------------------
# Recomendación inicial: usar SOR con omega≈1.7 (rápido), si no, Gauss-Seidel.
tol = 1e-6
max_iter = 20000
omega = 1.7

print("Resolviendo con SOR (omega=", omega, ") ...")
T_sor, it_sor, err_sor, t_sor = solve_poisson_sor(C_mask, omega=omega, T_bound=300.0, tol=tol, max_iter=max_iter, verbose=True)
print(f"SOR: iter={it_sor}, err={err_sor:.3e}, tiempo={t_sor:.2f}s")


# -------------------------
# Calcular y mostrar flujo de calor: q = -k * grad(T)
# -------------------------
Gy, Gx = np.gradient(T_sor, h, h)   # Gy = dT/dy, Gx = dT/dx
qx = -k * Gx
qy = -k * Gy

# Figura 1: Mapa de calor
plt.figure(figsize=(7, 6))
im1 = plt.pcolormesh(X, Y, T_sor, shading='auto', cmap='hot')
plt.title('Temperatura T [K]', fontsize=14, fontweight='bold')
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
plt.gca().set_aspect('equal')
plt.colorbar(im1, label='T [K]')
plt.tight_layout()

# Figura 2: Magnitud y dirección del flujo de calor 1
plt.figure(figsize=(8,7))
q_magnitude = np.sqrt(qx**2 + qy**2)  # magnitud del flujo
pcm = plt.pcolormesh(X, Y, q_magnitude, shading='auto', cmap='viridis')
cbar = plt.colorbar(pcm, label='|q| [W/m²]')
plt.streamplot(X, Y, qx, qy, 
               density=1.8,
               color='white',
               linewidth=1.2,
               arrowsize=1.0,
               arrowstyle='->')
plt.title('Magnitud del flujo de calor |q| y dirección', fontsize=14, fontweight='bold')
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
plt.gca().set_aspect('equal')
plt.tight_layout()

# Figura 2: Magnitud y dirección del flujo de calor 2
plt.figure(figsize=(7, 6))
q_magnitude = np.sqrt(qx**2 + qy**2)  # magnitud del flujo
im2 = plt.pcolormesh(X, Y, q_magnitude, shading='auto', cmap='plasma')
plt.streamplot(X, Y, qx, qy, density=1.5, color='white', linewidth=1, arrowsize=0.8)
plt.title('Magnitud y dirección del flujo de calor', fontsize=14, fontweight='bold')
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
plt.gca().set_aspect('equal')
plt.colorbar(im2, label='|q| [W/m²]')
plt.tight_layout()

plt.show()