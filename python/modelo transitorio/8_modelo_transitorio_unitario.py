import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import time

# ============================================================
# MODELO TÉRMICO OPTIMIZADO - UN SOLO NÚCLEO
# Resuelve: ρ·cp·∂T/∂t = k·∇²T + q
# Optimización: Operaciones vectorizadas con NumPy
# ============================================================

print("="*60)
print("SIMULACIÓN TÉRMICA DE UN NÚCLEO (OPTIMIZADO)")
print("="*60)

# --------------------------------------------------------
# PARÁMETROS DEL CHIP
# --------------------------------------------------------
L = 0.01                # Tamaño del chip [m] (10 mm)
Nx, Ny = 51, 51        # Resolución de la malla
h = L / (Nx - 1)       # Paso espacial [m]

# Malla espacial
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

# Propiedades del material (Silicio)
k = 130.0              # Conductividad térmica [W/(m·K)]
rho = 2330.0           # Densidad [kg/m³]
cp = 700.0             # Calor específico [J/(kg·K)]
alpha = k / (rho * cp) # Difusividad térmica [m²/s]

# Núcleo en el centro
nuc_L = L / 8          # Lado del núcleo [m]
nuc_cx = L / 2         # Centro x
nuc_cy = L / 2        # Centro y

print(f"\n📊 PARÁMETROS:")
print(f"   • Tamaño del chip: {L*1000:.1f} mm × {L*1000:.1f} mm")
print(f"   • Núcleo: {nuc_L*1000:.1f} mm × {nuc_L*1000:.1f} mm (centro)")
print(f"   • Resolución: {Nx} × {Ny} nodos")
print(f"   • Material: Silicio")

# --------------------------------------------------------
# PARÁMETROS TEMPORALES
# --------------------------------------------------------
dt_max = h**2 / (4 * alpha)
dt = 0.5 * dt_max
t_final = 10.0          # Tiempo total [s]
n_steps = int(t_final / dt)

print(f"\n⏱️  TIEMPO:")
print(f"   • dt: {dt:.3e} s")
print(f"   • Tiempo total: {t_final:.1f} s")
print(f"   • Pasos: {n_steps}")

# --------------------------------------------------------
# POTENCIA Y CONDICIONES
# --------------------------------------------------------
potencia = 20.0        # Potencia del núcleo [W]
T_inicial = 300.0      # Temperatura inicial [K]
T_borde = 300.0        # Temperatura en bordes [K]

print(f"\n⚡ POTENCIA: {potencia:.1f} W")
print(f"🌡️  T inicial: {T_inicial:.1f} K ({T_inicial-273.15:.1f}°C)")

# --------------------------------------------------------
# CREAR MÁSCARA DEL NÚCLEO
# --------------------------------------------------------
mascara = (
    (X >= nuc_cx - nuc_L/2) & (X <= nuc_cx + nuc_L/2) &
    (Y >= nuc_cy - nuc_L/2) & (Y <= nuc_cy + nuc_L/2)
)

# Generación de calor
q_matrix = np.zeros_like(X)
A_nucleo = nuc_L**2
espesor = 1.0
q_val = potencia / (A_nucleo * espesor)
q_matrix[mascara] = q_val

# --------------------------------------------------------
# SOLVER OPTIMIZADO (VECTORIZADO)
# --------------------------------------------------------
def solve_transient_vectorized(q_matrix, dt, t_final):
    """
    Solver optimizado usando operaciones vectorizadas de NumPy
    Speedup: ~10-50x más rápido que bucles Python
    """
    T = np.ones_like(q_matrix) * T_inicial
    n_steps = int(t_final / dt)
    save_interval = max(1, n_steps // 50)
    
    T_history = []
    t_history = []
    
    # Pre-calcular constantes
    r = alpha * dt / h**2
    q_factor = dt / (rho * cp)
    
    print(f"\n🔄 SIMULANDO (Modo Vectorizado)...")
    start = time()
    
    for step in range(n_steps):
        t = step * dt
        
        # Laplaciano usando slicing vectorizado (¡MUCHO más rápido!)
        laplacian = np.zeros_like(T)
        laplacian[1:-1, 1:-1] = (
            T[2:, 1:-1] + T[:-2, 1:-1] +    # Norte + Sur
            T[1:-1, 2:] + T[1:-1, :-2] -    # Este + Oeste
            4 * T[1:-1, 1:-1]               # Centro
        ) / h**2
        
        # Actualización vectorizada (toda la matriz a la vez)
        T[1:-1, 1:-1] += dt * (alpha * laplacian[1:-1, 1:-1] + 
                                q_matrix[1:-1, 1:-1] / (rho * cp))
        
        # Condiciones de borde (vectorizado)
        T[0, :] = T_borde
        T[-1, :] = T_borde
        T[:, 0] = T_borde
        T[:, -1] = T_borde
        
        if step % save_interval == 0:
            T_history.append(T.copy())
            t_history.append(t)
    
    elapsed = time() - start
    print(f"   ✓ Completado en {elapsed:.2f} s")
    print(f"   ⚡ Speedup: ~{n_steps * Nx * Ny / 1e6 / elapsed:.1f}M ops/s")
    
    return np.array(T_history), np.array(t_history), T

# --------------------------------------------------------
# EJECUTAR
# --------------------------------------------------------
T_history, t_history, T_final = solve_transient_vectorized(q_matrix, dt, t_final)

T_max_final = np.max(T_final)
print(f"\n📈 RESULTADOS:")
print(f"   • T máxima final: {T_max_final:.2f} K ({T_max_final-273.15:.2f}°C)")
print(f"   • Incremento: {T_max_final - T_inicial:.2f} K")

# ============================================================
# ANIMACIÓN
# ============================================================
print(f"\n🎬 CREANDO ANIMACIÓN...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(top=0.92)

vmin = T_inicial
vmax = np.max(T_history)

# Temperatura
im1 = ax1.pcolormesh(X*1000, Y*1000, T_history[0], 
                     shading='auto', cmap='hot', vmin=vmin, vmax=vmax)
ax1.set_xlabel('x [mm]')
ax1.set_ylabel('y [mm]')
ax1.set_title('Temperatura', fontweight='bold')
ax1.set_aspect('equal')

# Dibujar núcleo
nuc_mm = nuc_L * 1000
cx_mm = nuc_cx * 1000
cy_mm = nuc_cy * 1000
rect = plt.Rectangle((cx_mm - nuc_mm/2, cy_mm - nuc_mm/2), 
                    nuc_mm, nuc_mm,
                    fill=False, edgecolor='cyan', 
                    linewidth=2, linestyle='--')
ax1.add_patch(rect)
ax1.text(cx_mm, cy_mm, f'{potencia:.1f}W', 
        ha='center', va='center', 
        color='white', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# Evolución temporal
t_max_history = [np.max(T) for T in T_history]
line, = ax2.plot([], [], 'r-', linewidth=2, label='T máxima')
ax2.axhline(y=T_inicial, color='blue', linestyle='--', alpha=0.5, label='T inicial')
ax2.set_xlim(0, t_final)
ax2.set_ylim(T_inicial - 0.05, vmax + 0.05)
ax2.set_xlabel('Tiempo [s]')
ax2.set_ylabel('Temperatura [K]')
ax2.set_title('Evolución de T máxima', fontweight='bold')
ax2.grid(alpha=0.3)
ax2.legend()

time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def init():
    im1.set_array(T_history[0].ravel())
    line.set_data([], [])
    time_text.set_text('')
    return im1, line, time_text

def animate(frame):
    im1.set_array(T_history[frame].ravel())
    line.set_data(t_history[:frame+1], t_max_history[:frame+1])
    
    t = t_history[frame]
    T_max = t_max_history[frame]
    time_text.set_text(f't = {t:.3f} s\nT_max = {T_max:.2f} K\n({T_max-273.15:.2f}°C)')
    
    return im1, line, time_text

anim = FuncAnimation(fig, animate, init_func=init, 
                    frames=len(T_history), interval=50, 
                    blit=True, repeat=True)

fig.suptitle('Simulación Térmica - Un Solo Núcleo', 
             fontsize=14, fontweight='bold')

print("   ✓ Animación lista")
print("\n" + "="*60)

# --------------------------------------------------------
# COMPARACIÓN INICIAL VS FINAL
# --------------------------------------------------------
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

im3 = ax3.pcolormesh(X*1000, Y*1000, T_history[0], 
                     shading='auto', cmap='hot', vmin=vmin, vmax=vmax)
ax3.set_title('Estado Inicial (t = 0 s)', fontweight='bold')
ax3.set_xlabel('x [mm]')
ax3.set_ylabel('y [mm]')
ax3.set_aspect('equal')
plt.colorbar(im3, ax=ax3, label='T [K]')

im4 = ax4.pcolormesh(X*1000, Y*1000, T_final, 
                     shading='auto', cmap='hot', vmin=vmin, vmax=vmax)
ax4.set_title(f'Estado Final (t = {t_final:.1f} s)', fontweight='bold')
ax4.set_xlabel('x [mm]')
ax4.set_ylabel('y [mm]')
ax4.set_aspect('equal')
plt.colorbar(im4, ax=ax4, label='T [K]')



plt.tight_layout()
plt.show()