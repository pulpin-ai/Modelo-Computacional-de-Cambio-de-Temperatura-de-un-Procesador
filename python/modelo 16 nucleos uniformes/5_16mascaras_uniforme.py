import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from time import time

# ============================================================
# PARÁMETROS DEL DOMINIO
# ============================================================
L = 0.02               # Largo del dominio [m]
Nx, Ny = 101, 101      # Número de puntos en x e y
h = L / (Nx - 1)       # Paso espacial
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

k = 130.0              # Conductividad térmica [W/(m·K)]
P_total = 60           # Potencia total [W]
factor_tamano = 16     # Tamaño relativo de cada núcleo (L/factor_tamano)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def crear_mascara_16_nucleos(X, Y, L, factor_tamano=16):
    """Crea una máscara booleana con 16 núcleos distribuidos en 4x4."""
    Nx, Ny = X.shape
    n_side = 4
    nucleo_L = L / factor_tamano
    paso = L / n_side
    mask = np.zeros_like(X, dtype=bool)

    for i in range(n_side):
        for j in range(n_side):
            cx = (i + 0.5) * paso
            cy = (j + 0.5) * paso
            x_min, x_max = cx - nucleo_L/2, cx + nucleo_L/2
            y_min, y_max = cy - nucleo_L/2, cy + nucleo_L/2
            mask |= np.logical_and.reduce((X >= x_min, X <= x_max, Y >= y_min, Y <= y_max))
    return mask

def calcular_qg_y_C(P_nucleo, L, k, h, factor_tamano=8):
    """Calcula densidad volumétrica de calor y constante C para SOR."""
    A_nucleo = (L / factor_tamano)**2
    t_espesor = 1.0
    q_nucleo = P_nucleo / (A_nucleo * t_espesor)
    C = q_nucleo * h**2 / (4 * k)
    return q_nucleo, C

def solve_poisson_sor(C_mask, omega=1.7, T_bound=300.0, tol=1e-6, max_iter=10000, verbose=False):
    """Resuelve la ecuación de Poisson usando SOR."""
    T = np.ones_like(C_mask) * T_bound
    # Condiciones de frontera
    T[0, :] = T_bound
    T[-1, :] = T_bound
    T[:, 0] = T_bound
    T[:, -1] = T_bound

    start = time()
    for iteracion in range(1, max_iter + 1):
        T_old = T.copy()
        # In-place SOR
        for i in range(1, T.shape[0] - 1):
            for j in range(1, T.shape[1] - 1):
                t_new = 0.25 * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1]) + C_mask[i,j]
                T[i,j] = (1 - omega) * T[i,j] + omega * t_new
        err_iter = np.max(np.abs(T - T_old))
        if iteracion % 200 == 0 and verbose:
            print(f"SOR iteración={iteracion} err={err_iter:.3e}")
        if err_iter < tol:
            if verbose: print("Convergencia alcanzada en", iteracion, "iteraciones")
            break
    elapsed = time() - start
    return T, iteracion, err_iter, elapsed

# ============================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================
nucleo_mask = crear_mascara_16_nucleos(X, Y, L, factor_tamano)
P_por_nucleo = P_total / 16
qg_value, C_value = calcular_qg_y_C(P_por_nucleo, L, k, h, factor_tamano)

C_mask = np.zeros_like(X)
C_mask[nucleo_mask] = C_value

# ============================================================
# RESOLUCIÓN DE LA ECUACIÓN DE POISSON
# ============================================================
tol = 1e-6
max_iter = 20000
omega = 1.7

print("="*60)
print("INICIANDO SOLVER SOR...")
print("="*60)
print("Resolviendo con SOR (omega=", omega, ") ...")
T_sor, it_sor, err_sor, t_sor = solve_poisson_sor(
    C_mask, omega=omega, T_bound=300.0, tol=tol, max_iter=max_iter, verbose=True
)
print(f"SOR: iter={it_sor}, err={err_sor:.3e}, tiempo={t_sor:.2f}s")

# ============================================================
# VISUALIZACIÓN DE LA MÁSCARA DE NÚCLEOS
# ============================================================
plt.figure(figsize=(6,6))
plt.pcolormesh(X, Y, nucleo_mask, shading='auto', cmap='hot')
plt.colorbar(label='Máscara núcleo (True=1, False=0)')
plt.title('Máscara de los 16 núcleos')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.gca().set_aspect('equal')
plt.show()

# ============================================================
# VISUALIZACIÓN DE LA TEMPERATURA ESTACIONARIA
# ============================================================
plt.figure(figsize=(6,6))
pcm = plt.pcolormesh(X, Y, T_sor, shading='auto', cmap='hot')
plt.colorbar(pcm, label='T [K]')
plt.title('Campo de temperatura (solución estacionaria)')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.gca().set_aspect('equal')
plt.show()

# ============================================================
# CÁLCULO DEL FLUJO DE CALOR
# ============================================================
Gy, Gx = np.gradient(T_sor, h, h)
qx = -k * Gx
qy = -k * Gy
q_magnitude = np.sqrt(qx**2 + qy**2)

# ============================================================
# FIGURA 3: MAGNITUD Y DIRECCIÓN DEL FLUJO (STREAMPLOT PLASMA)
# ============================================================
plt.figure(figsize=(7, 6))
im2 = plt.pcolormesh(X, Y, q_magnitude, shading='auto', cmap='plasma')
plt.streamplot(
    X, Y, qx, qy,
    density=1.5,
    color='white',
    linewidth=1,
    arrowsize=0.8
)
plt.title('Magnitud y dirección del flujo de calor', fontsize=14, fontweight='bold')
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
plt.gca().set_aspect('equal')
plt.colorbar(im2, label='|q| [W/m²]')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Campo de temperatura con vector de flujo de calor
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 8))
plt.subplots_adjust(bottom=0.15)

def update(val):
    ax.clear()
    step = int(slider.val)

    # Fondo de temperatura
    ax.pcolormesh(X, Y, T_sor, shading='auto', cmap='inferno', alpha=0.9)

    # Normalización para quiver
    mag_sample = q_magnitude[::step, ::step]
    qx_sample = qx[::step, ::step]
    qy_sample = qy[::step, ::step]
    mag_safe = np.where(mag_sample > 0, mag_sample, 1)

    qx_norm = qx_sample / mag_safe
    qy_norm = qy_sample / mag_safe

    # Escala logarítmica
    mag_log = np.log10(mag_sample + 1)
    scale_factor = mag_log / (np.max(mag_log) + 1e-10)

    ax.quiver(
        X[::step, ::step], Y[::step, ::step],
        qx_norm * scale_factor,
        qy_norm * scale_factor,
        mag_sample,
        cmap='viridis',
        scale=15, width=0.003,
        headwidth=4, headlength=5,
        minlength=0.5, pivot='mid'
    )

    ax.set_title('Campo de flujo de calor q = -k ∇T', fontsize=14, fontweight='bold')
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_aspect('equal')
    fig.canvas.draw_idle()

# Deslizador
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Densidad (step)', 1, 20, valinit=6, valstep=1)
slider.on_changed(update)
update(6)
plt.show()