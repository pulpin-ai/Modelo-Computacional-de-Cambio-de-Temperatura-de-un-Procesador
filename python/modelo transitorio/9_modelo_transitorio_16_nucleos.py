import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import time

# ============================================================
# SIMULACIÓN TÉRMICA OPTIMIZADA - 16 NÚCLEOS CON VARIACIÓN DINÁMICA
# Resuelve: ρ·cp·∂T/∂t = k·∇²T + q(t)
# Núcleos con potencia variable y algunos inactivos
# ============================================================

print("="*70)
print("SIMULACIÓN TÉRMICA DE 16 NÚCLEOS (OPTIMIZADO + DINÁMICO)")
print("="*70)

# --------------------------------------------------------
# PARÁMETROS DEL CHIP
# --------------------------------------------------------
L = 0.02                # Tamaño del chip [m] (20 mm)
Nx, Ny = 101, 101      # Resolución de la malla
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

# Geometría: 16 núcleos en grilla 4×4
nuc_L = L / 16          # Lado de cada núcleo [m]

print(f"\n📊 PARÁMETROS:")
print(f"   • Tamaño del chip: {L*2000:.1f} mm × {L*2000:.1f} mm")
print(f"   • Núcleo: {nuc_L*1000:.1f} mm × {nuc_L*1000:.1f} mm cada uno")
print(f"   • Resolución: {Nx} × {Ny} nodos ({Nx*Ny:,} puntos)")
print(f"   • Material: Silicio")

# --------------------------------------------------------
# PARÁMETROS TEMPORALES
# --------------------------------------------------------
dt_max = h**2 / (4 * alpha)
dt = 0.8 * dt_max
t_final = 5.0          # Tiempo total [s]
n_steps = int(t_final / dt)

print(f"\n⏱️  TIEMPO:")
print(f"   • dt: {dt:.3e} s")
print(f"   • Tiempo total: {t_final:.1f} s")
print(f"   • Pasos: {n_steps:,}")

# --------------------------------------------------------
# CONFIGURACIÓN DE 16 NÚCLEOS
# --------------------------------------------------------
# Potencias base [W] - Distribución realista
# 0 = inactivo, >0 = activo con diferentes cargas
potencias_base = np.array([
    [18.0,  0.0, 25.0,  2.5],  # Fila 1: núcleos 1-4
    [ 8.5, 15.0,  0.0, 15.0],  # Fila 2: núcleos 5-8
    [ 0.5, 28.0, 12.0,  0.0],  # Fila 3: núcleos 9-12
    [ 6.0, 30.0, 20.0,  9.5]   # Fila 4: núcleos 13-16
]).flatten()

# Parámetros de variación temporal (simula carga de trabajo variable)
# Algunos núcleos varían más que otros
variacion_amplitud = np.array([
    [0.3, 0.0, 0.5, 0.2],  # Fila 1
    [0.4, 0.8, 0.0, 0.6],  # Fila 2
    [0.6, 0.4, 0.6, 0.0],  # Fila 3
    [0.2, 0.5, 0.4, 0.3]   # Fila 4
]).flatten()

# Frecuencias de variación (Hz) - cada núcleo tiene su ritmo
frecuencias = np.random.uniform(0.3, 1.5, 16)

T_inicial = 300.0      # Temperatura inicial [K]
T_borde = 300.0        # Temperatura en bordes [K]

n_activos = np.sum(potencias_base > 0)
P_max = np.max(potencias_base)
P_total_base = np.sum(potencias_base)

print(f"\n⚡ CONFIGURACIÓN DE POTENCIA:")
print(f"   • Núcleos activos: {n_activos}/16")
print(f"   • Núcleos inactivos: {16-n_activos}")
print(f"   • Potencia base total: {P_total_base:.1f} W")
print(f"   • Potencia máxima: {P_max:.1f} W (núcleo {np.argmax(potencias_base)+1})")
print(f"   • Con variación temporal dinámica")
print(f"\n🌡️  T inicial: {T_inicial:.1f} K ({T_inicial-273.15:.1f}°C)")

# --------------------------------------------------------
# CREAR MÁSCARAS DE LOS 16 NÚCLEOS
# --------------------------------------------------------
def crear_mascaras_16_nucleos(X, Y, L, nuc_L):
    """Crea máscaras para los 16 núcleos en grilla 4×4"""
    mascaras = []
    grid_pos = np.linspace(nuc_L/2, L - nuc_L/2, 4)
    
    for cy in grid_pos[::-1]:  # De arriba a abajo
        for cx in grid_pos:     # De izquierda a derecha
            mask = (
                (X >= cx - nuc_L/2) & (X <= cx + nuc_L/2) &
                (Y >= cy - nuc_L/2) & (Y <= cy + nuc_L/2)
            )
            mascaras.append(mask)
    
    return mascaras, grid_pos

mascaras, grid_pos = crear_mascaras_16_nucleos(X, Y, L, nuc_L)

# --------------------------------------------------------
# FUNCIÓN DE POTENCIA VARIABLE EN EL TIEMPO
# --------------------------------------------------------
def calcular_potencias_dinamicas(t, potencias_base, amplitud, frecuencias):
    """
    Calcula potencias que varían con el tiempo
    P(t) = P_base * (1 + A * sin(2π*f*t))
    """
    potencias_t = np.zeros_like(potencias_base)
    
    for i in range(len(potencias_base)):
        if potencias_base[i] > 0:  # Solo núcleos activos varían
            variacion = 1 + amplitud[i] * np.sin(2 * np.pi * frecuencias[i] * t)
            potencias_t[i] = potencias_base[i] * max(0.1, variacion)  # Mínimo 10%
    
    return potencias_t

# --------------------------------------------------------
# SOLVER OPTIMIZADO CON POTENCIA DINÁMICA
# --------------------------------------------------------
def solve_transient_dynamic(mascaras, potencias_base, amplitud, frecuencias, 
                           dt, t_final, save_interval=None):
    """
    Solver optimizado con generación de calor variable en el tiempo
    """
    T = np.ones((Ny, Nx)) * T_inicial
    n_steps = int(t_final / dt)
    
    if save_interval is None:
        save_interval = max(1, n_steps // 100)
    
    T_history = []
    t_history = []
    P_history = []  # Guardar evolución de potencias
    
    # Volumen de cada núcleo
    A_nucleo = nuc_L**2
    espesor = 1.0
    V_nucleo = A_nucleo * espesor
    
    print(f"\n🔄 SIMULANDO (Vectorizado + Dinámico)...")
    start = time()
    
    for step in range(n_steps):
        t = step * dt
        
        # Calcular potencias en este instante
        potencias_t = calcular_potencias_dinamicas(t, potencias_base, 
                                                   amplitud, frecuencias)
        
        # Construir matriz de generación de calor q(x,y,t)
        q_matrix = np.zeros_like(T)
        for i, mask in enumerate(mascaras):
            if potencias_t[i] > 0:
                q_matrix[mask] = potencias_t[i] / V_nucleo
        
        # Laplaciano vectorizado
        laplacian = np.zeros_like(T)
        laplacian[1:-1, 1:-1] = (
            T[2:, 1:-1] + T[:-2, 1:-1] +
            T[1:-1, 2:] + T[1:-1, :-2] -
            4 * T[1:-1, 1:-1]
        ) / h**2
        
        # Actualización vectorizada
        T[1:-1, 1:-1] += dt * (alpha * laplacian[1:-1, 1:-1] + 
                                q_matrix[1:-1, 1:-1] / (rho * cp))
        
        # Condiciones de borde
        T[0, :] = T_borde
        T[-1, :] = T_borde
        T[:, 0] = T_borde
        T[:, -1] = T_borde
        
        # Guardar estados
        if step % save_interval == 0:
            T_history.append(T.copy())
            t_history.append(t)
            P_history.append(potencias_t.copy())
            
            if len(T_history) % 20 == 0:
                T_max = np.max(T)
                P_total = np.sum(potencias_t)
                print(f"      t = {t:.3f} s | T_max = {T_max:.2f} K | P_total = {P_total:.1f} W")
    
    elapsed = time() - start
    print(f"   ✓ Completado en {elapsed:.2f} s")
    print(f"   ⚡ Rendimiento: {n_steps * Nx * Ny / 1e6 / elapsed:.1f}M ops/s")
    
    return np.array(T_history), np.array(t_history), np.array(P_history), T

# --------------------------------------------------------
# EJECUTAR SIMULACIÓN
# --------------------------------------------------------
T_history, t_history, P_history, T_final = solve_transient_dynamic(
    mascaras, potencias_base, variacion_amplitud, frecuencias,
    dt, t_final
)

T_max_final = np.max(T_final)
print(f"\n📈 RESULTADOS FINALES:")
print(f"   • Frames guardados: {len(T_history)}")
print(f"   • T máxima final: {T_max_final:.2f} K ({T_max_final-273.15:.2f}°C)")
print(f"   • Incremento máx.: {T_max_final - T_inicial:.2f} K")

# ============================================================
# VISUALIZACIONES SEPARADAS
# ============================================================
print(f"\n🎬 CREANDO VISUALIZACIONES...")

vmin = T_inicial
vmax = np.max(T_history)
nuc_mm = nuc_L * 1000
grid_pos_mm = grid_pos * 1000

def dibujar_nucleos(ax, P_array=None):
    """Dibuja los 16 núcleos con etiquetas de potencia"""
    nucleo_num = 1
    for cy in grid_pos_mm[::-1]:
        for cx in grid_pos_mm:
            rect = plt.Rectangle((cx - nuc_mm/2, cy - nuc_mm/2), 
                                nuc_mm, nuc_mm,
                                fill=False, edgecolor='cyan', 
                                linewidth=1.5, linestyle='--')
            ax.add_patch(rect)
            
            if P_array is not None:
                p = P_array[nucleo_num-1]
                label = f'{nucleo_num}\n{p:.1f}W' if p > 0 else f'{nucleo_num}\nOFF'
            else:
                label = str(nucleo_num)
            
            ax.text(cx, cy, label, 
                    ha='center', va='center', 
                    color='white', fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            nucleo_num += 1

# ============================================================
# 1. MAPAS DE CALOR: INICIAL, INTERMEDIO, FINAL
# ============================================================
idx_medio = len(T_history) // 2

fig1 = plt.figure(figsize=(16, 6))
gs = fig1.add_gridspec(2, 3, height_ratios=[20, 1], hspace=0.3)

ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[0, 2])
cax = fig1.add_subplot(gs[1, :])

# Inicial
im1 = ax1.pcolormesh(X*1000, Y*1000, T_history[0], 
                     shading='auto', cmap='hot', vmin=vmin, vmax=vmax)
ax1.set_title(f'Estado Inicial (t = {t_history[0]:.2f} s)', fontweight='bold', fontsize=12)
ax1.set_xlabel('x [mm]')
ax1.set_ylabel('y [mm]')
ax1.set_aspect('equal')
dibujar_nucleos(ax1, P_history[0])

# Intermedio
im2 = ax2.pcolormesh(X*1000, Y*1000, T_history[idx_medio], 
                     shading='auto', cmap='hot', vmin=vmin, vmax=vmax)
ax2.set_title(f'Estado Intermedio (t = {t_history[idx_medio]:.2f} s)', fontweight='bold', fontsize=12)
ax2.set_xlabel('x [mm]')
ax2.set_ylabel('y [mm]')
ax2.set_aspect('equal')
dibujar_nucleos(ax2, P_history[idx_medio])

# Final
im3 = ax3.pcolormesh(X*1000, Y*1000, T_history[-1], 
                     shading='auto', cmap='hot', vmin=vmin, vmax=vmax)
ax3.set_title(f'Estado Final (t = {t_history[-1]:.2f} s)', fontweight='bold', fontsize=12)
ax3.set_xlabel('x [mm]')
ax3.set_ylabel('y [mm]')
ax3.set_aspect('equal')
dibujar_nucleos(ax3, P_history[-1])

# Barra de color en el eje separado
cbar = fig1.colorbar(im3, cax=cax, orientation='horizontal')
cbar.set_label('T [K]', fontsize=12)

fig1.suptitle('Evolución de Temperatura: Inicial → Intermedio → Final', 
              fontsize=14, fontweight='bold', y=0.98)


# ============================================================
# 2. COMPARACIÓN INICIAL VS FINAL
# ============================================================
fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(13, 5))

# Inicial
im4 = ax4.pcolormesh(X*1000, Y*1000, T_history[0], 
                     shading='auto', cmap='hot', vmin=vmin, vmax=vmax)
ax4.set_title('Estado Inicial', fontweight='bold', fontsize=12)
ax4.set_xlabel('x [mm]')
ax4.set_ylabel('y [mm]')
ax4.set_aspect('equal')
dibujar_nucleos(ax4, P_history[0])

# Final
im5 = ax5.pcolormesh(X*1000, Y*1000, T_history[-1], 
                     shading='auto', cmap='hot', vmin=vmin, vmax=vmax)
ax5.set_title('Estado Final', fontweight='bold', fontsize=12)
ax5.set_xlabel('x [mm]')
ax5.set_ylabel('y [mm]')
ax5.set_aspect('equal')
plt.colorbar(im5, ax=ax5, label='T [K]')
dibujar_nucleos(ax5, P_history[-1])

delta_T = np.max(T_history[-1]) - np.max(T_history[0])
fig2.suptitle(f'Comparación Inicial vs Final (ΔT_max = {delta_T:.2f} K)', 
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
# ============================================================
# 3. ANIMACIÓN DE TEMPERATURA
# ============================================================
fig3, ax6 = plt.subplots(figsize=(8, 7))

im6 = ax6.pcolormesh(X*1000, Y*1000, T_history[0], 
                     shading='auto', cmap='hot', vmin=vmin, vmax=vmax)
plt.colorbar(im6, ax=ax6, label='Temperatura [K]')
ax6.set_xlabel('x [mm]')
ax6.set_ylabel('y [mm]')
ax6.set_title('Animación de Temperatura', fontweight='bold', fontsize=12)
ax6.set_aspect('equal')

nucleos_texts = []
nucleo_num = 1
for cy in grid_pos_mm[::-1]:
    for cx in grid_pos_mm:
        rect = plt.Rectangle((cx - nuc_mm/2, cy - nuc_mm/2), 
                            nuc_mm, nuc_mm,
                            fill=False, edgecolor='cyan', 
                            linewidth=1.5, linestyle='--')
        ax6.add_patch(rect)
        
        text = ax6.text(cx, cy, f'{nucleo_num}\n0W', 
                ha='center', va='center', 
                color='white', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        nucleos_texts.append(text)
        nucleo_num += 1

time_text = ax6.text(0.02, 0.98, '', transform=ax6.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

def init_temp():
    im6.set_array(T_history[0].ravel())
    time_text.set_text('')
    return [im6, time_text] + nucleos_texts

def animate_temp(frame):
    im6.set_array(T_history[frame].ravel())
    
    # Actualizar etiquetas
    potencias_frame = P_history[frame]
    for i, (text, p) in enumerate(zip(nucleos_texts, potencias_frame)):
        if p > 0:
            text.set_text(f'{i+1}\n{p:.1f}W')
        else:
            text.set_text(f'{i+1}\nOFF')
    
    
    return [im6, time_text] + nucleos_texts

anim_temp = FuncAnimation(fig3, animate_temp, init_func=init_temp, 
                         frames=len(T_history), interval=50, 
                         blit=True, repeat=True)

# ============================================================
# 4. BARRAS ANIMADAS DE POTENCIA
# ============================================================
fig4, ax7 = plt.subplots(figsize=(10, 6))

nucleos_idx = np.arange(1, 17)
bars = ax7.bar(nucleos_idx, potencias_base, color='steelblue', alpha=0.7)
ax7.set_xlabel('Núcleo #', fontsize=11)
ax7.set_ylabel('Potencia [W]', fontsize=11)
ax7.set_title('Potencia por Núcleo (Animado)', fontweight='bold', fontsize=13)
ax7.set_ylim(0, 35)
ax7.grid(alpha=0.3, axis='y')
ax7.set_xticks(nucleos_idx)

P_total_history = [np.sum(P) for P in P_history]
power_text = ax7.text(0.98, 0.98, '', transform=ax7.transAxes,
                     fontsize=11, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

def init_bars():
    for bar in bars:
        bar.set_height(0)
    power_text.set_text('')
    return list(bars) + [power_text]

def animate_bars(frame):
    potencias_frame = P_history[frame]
    
    for bar, p in zip(bars, potencias_frame):
        bar.set_height(p)
        # Colorear según nivel
        if p == 0:
            bar.set_color('gray')
        elif p > 20:
            bar.set_color('red')
        elif p > 10:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    t = t_history[frame]
    P_total = P_total_history[frame]
    power_text.set_text(f't = {t:.3f} s\nP_total = {P_total:.1f} W\nActivos: {np.sum(potencias_frame > 0)}/16')
    
    return list(bars) + [power_text]

anim_bars = FuncAnimation(fig4, animate_bars, init_func=init_bars, 
                         frames=len(P_history), interval=50, 
                         blit=True, repeat=True)

print("   ✓ Todas las visualizaciones creadas")
print("\n" + "="*70)
print("VISUALIZACIONES:")
print("  1. Mapas de calor: Inicial → Intermedio → Final")
print("  2. Comparación: Inicial vs Final")
print("  3. Animación de temperatura")
print("  4. Animación de barras de potencia")
print("="*70)

plt.show()
# ============================================================
# 5. ANÁLISIS DE FLUJO DE CALOR (ESTADO FINAL)
# ============================================================
print("\n🔥 CALCULANDO FLUJO DE CALOR...")

# Calcular gradiente de temperatura
dT_dx = np.zeros_like(T_final)
dT_dy = np.zeros_like(T_final)

dT_dx[:, 1:-1] = (T_final[:, 2:] - T_final[:, :-2]) / (2 * h)
dT_dy[1:-1, :] = (T_final[2:, :] - T_final[:-2, :]) / (2 * h)

# Flujo de calor: q = -k * ∇T
qx = -k * dT_dx
qy = -k * dT_dy
q_magnitude = np.sqrt(qx**2 + qy**2)

print(f"   • Flujo máximo: {np.max(q_magnitude):.2e} W/m²")
print(f"   • Flujo promedio: {np.mean(q_magnitude):.2e} W/m²")

# Calcular flujo de calor para todos los frames
print("\n🔥 CALCULANDO FLUJO DE CALOR PARA ANIMACIÓN...")
qx_history = []
qy_history = []
q_mag_history = []

for T_frame in T_history:
    dT_dx_frame = np.zeros_like(T_frame)
    dT_dy_frame = np.zeros_like(T_frame)
    
    dT_dx_frame[:, 1:-1] = (T_frame[:, 2:] - T_frame[:, :-2]) / (2 * h)
    dT_dy_frame[1:-1, :] = (T_frame[2:, :] - T_frame[:-2, :]) / (2 * h)
    
    qx_frame = -k * dT_dx_frame
    qy_frame = -k * dT_dy_frame
    q_mag_frame = np.sqrt(qx_frame**2 + qy_frame**2)
    
    qx_history.append(qx_frame)
    qy_history.append(qy_frame)
    q_mag_history.append(q_mag_frame)

qx_history = np.array(qx_history)
qy_history = np.array(qy_history)
q_mag_history = np.array(q_mag_history)

print(f"   ✓ Flujo calculado para {len(qx_history)} frames")
# ============================================================
# 7. ANIMACIÓN DE TEMPERATURA + CAMPO VECTORIAL DE FLUJO
# ============================================================
fig7, ax10 = plt.subplots(figsize=(9, 8))

# Mapa de calor inicial
im10 = ax10.pcolormesh(X*1000, Y*1000, T_history[0], 
                       shading='auto', cmap='inferno', alpha=0.85,
                       vmin=vmin, vmax=vmax)
plt.colorbar(im10, ax=ax10, label='Temperatura [K]')

# Parámetro de muestreo para vectores
vector_step = 8

# Vectores iniciales
qx_sample_init = qx_history[0][::vector_step, ::vector_step]
qy_sample_init = qy_history[0][::vector_step, ::vector_step]
q_mag_sample_init = q_mag_history[0][::vector_step, ::vector_step]

# Normalizar y escalar
mag_safe_init = np.where(q_mag_sample_init > 0, q_mag_sample_init, 1)
qx_norm_init = qx_sample_init / mag_safe_init
qy_norm_init = qy_sample_init / mag_safe_init
mag_log_init = np.log10(q_mag_sample_init + 1)
scale_factor_init = mag_log_init / (np.max(mag_log_init) + 1e-10)

quiver_obj = ax10.quiver(
    X[::vector_step, ::vector_step]*1000, 
    Y[::vector_step, ::vector_step]*1000,
    qx_norm_init * scale_factor_init,
    qy_norm_init * scale_factor_init,
    q_mag_sample_init,
    cmap='viridis', scale=12, width=0.004,
    headwidth=4, headlength=5, minlength=0.5, 
    pivot='mid', alpha=0.8
)

ax10.set_xlabel('x [mm]', fontsize=11)
ax10.set_ylabel('y [mm]', fontsize=11)
ax10.set_title('Animación: Temperatura + Flujo de Calor', fontweight='bold', fontsize=13)
ax10.set_aspect('equal')

# Dibujar núcleos
nucleos_texts_vec = []
nucleo_num = 1
for cy in grid_pos_mm[::-1]:
    for cx in grid_pos_mm:
        rect = plt.Rectangle((cx - nuc_mm/2, cy - nuc_mm/2), 
                            nuc_mm, nuc_mm,
                            fill=False, edgecolor='cyan', 
                            linewidth=1.5, linestyle='--')
        ax10.add_patch(rect)
        
        text = ax10.text(cx, cy, f'{nucleo_num}\n0W', 
                ha='center', va='center', 
                color='white', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        nucleos_texts_vec.append(text)
        nucleo_num += 1

time_text_vec = ax10.text(0.02, 0.98, '', transform=ax10.transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

def init_temp_vector():
    im10.set_array(T_history[0].ravel())
    time_text_vec.set_text('')
    return [im10, quiver_obj, time_text_vec] + nucleos_texts_vec

def animate_temp_vector(frame):
    # Actualizar temperatura
    im10.set_array(T_history[frame].ravel())
    
    # Actualizar vectores de flujo
    qx_sample = qx_history[frame][::vector_step, ::vector_step]
    qy_sample = qy_history[frame][::vector_step, ::vector_step]
    q_mag_sample = q_mag_history[frame][::vector_step, ::vector_step]
    
    # Normalizar y escalar
    mag_safe = np.where(q_mag_sample > 0, q_mag_sample, 1)
    qx_norm = qx_sample / mag_safe
    qy_norm = qy_sample / mag_safe
    mag_log = np.log10(q_mag_sample + 1)
    scale_factor = mag_log / (np.max(mag_log) + 1e-10)
    
    # Actualizar quiver
    quiver_obj.set_UVC(qx_norm * scale_factor, qy_norm * scale_factor, q_mag_sample)
    
    # Actualizar etiquetas de núcleos
    potencias_frame = P_history[frame]
    for i, (text, p) in enumerate(zip(nucleos_texts_vec, potencias_frame)):
        if p > 0:
            text.set_text(f'{i+1}\n{p:.1f}W')
        else:
            text.set_text(f'{i+1}\nOFF')
    
    # Actualizar tiempo y estadísticas
    t = t_history[frame]
    T_max_frame = np.max(T_history[frame])
    q_max_frame = np.max(q_mag_sample)
    time_text_vec.set_text(f't = {t:.2f} s\nT_max = {T_max_frame:.1f} K\n|q|_max = {q_max_frame:.2e} W/m²')
    
    return [im10, quiver_obj, time_text_vec] + nucleos_texts_vec

anim_temp_vector = FuncAnimation(fig7, animate_temp_vector, init_func=init_temp_vector, 
                                 frames=len(T_history), interval=50, 
                                 blit=True, repeat=True)

print("   ✓ Todas las visualizaciones creadas")
print("\n" + "="*70)
print("VISUALIZACIONES:")
print("  1. Mapas de calor: Inicial → Intermedio → Final")
print("  2. Comparación: Inicial vs Final")
print("  3. Animación de temperatura")
print("  4. Animación de barras de potencia")
print("  5. Magnitud y dirección del flujo de calor (streamplot)")
print("  6. Campo vectorial interactivo (con control de densidad)")
print("  7. Animación fusionada: Temperatura + Campo de flujo vectorial")
print("="*70)

plt.show()