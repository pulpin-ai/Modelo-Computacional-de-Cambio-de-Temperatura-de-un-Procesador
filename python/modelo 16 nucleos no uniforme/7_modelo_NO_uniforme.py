import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from time import time

# --------------------------------------------------------
# PARÁMETROS GENERALES
# --------------------------------------------------------
L = 0.02                # longitud del chip (20 mm)
Nx, Ny = 201, 201       # resolución del mallado
h = L / (Nx - 1)        # paso de malla

x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

k = 130.0               # conductividad térmica del silicio [W/mK]
factor_tamano = 16      # cada núcleo = L/factor

# --------------------------------------------------------
# POTENCIA DE LOS 16 NÚCLEOS
# Puedes cambiar estos valores
# 0 W → núcleo apagado
# --------------------------------------------------------

def generar_y_promediar_matrices(n_matrices=10000, n_elementos=16, suma_objetivo=range(115, 131)):
    matrices = []
    
    # Generar pesos base diferentes para cada posición (esto crea la variabilidad)
    pesos_base = np.random.exponential(scale=2, size=n_elementos)
    
    for _ in range(n_matrices):
        suma_target = np.random.choice(list(suma_objetivo))
        
        # Usar los pesos base con algo de ruido
        valores = pesos_base * np.random.uniform(0.5, 1.5, n_elementos)
        
        # Asegurar al menos un valor no-cero
        if valores.sum() == 0:
            valores[np.random.randint(0, n_elementos)] = 1
        
        # Normalizar para alcanzar el target
        factor = suma_target / valores.sum()
        valores = valores * factor
        
        matrices.append(valores)
    
    matrices = np.array(matrices)
    matriz_promedio = np.mean(matrices, axis=0)   

    n_a_cero = np.random.randint(1, 4)
    indices_a_cero = np.random.choice(n_elementos, size=n_a_cero, replace=False)
    matriz_promedio[indices_a_cero] = 0

    return matriz_promedio, matrices

matriz_promedio, todas = generar_y_promediar_matrices()
potencias = matriz_promedio

# --------------------------------------------------------
# FUNCIÓN: Crear 16 máscaras de núcleos (rejilla 4x4)
# --------------------------------------------------------
def crear_mascara_nucleos_16(X, Y, L, factor_tamano=16):
    nuc_L = L / factor_tamano
    grid_positions = np.linspace(nuc_L/2, L - nuc_L/2, 4)
    mascaras = []

    for cx in grid_positions:
        for cy in grid_positions:
            mask = (
                (X >= cx - nuc_L/2) & (X <= cx + nuc_L/2) &
                (Y >= cy - nuc_L/2) & (Y <= cy + nuc_L/2)
            )
            mascaras.append(mask)

    return mascaras

# --------------------------------------------------------
# FUNCIÓN: Calcular qg y C para un núcleo
# --------------------------------------------------------
def calcular_qg_y_C(P_nucleo, L, k, h, factor_tamano=16):
    nuc_L = L / factor_tamano
    A_nucleo = nuc_L**2
    t_espesor = 1.0
    q_nucleo = P_nucleo / (A_nucleo * t_espesor)
    C = q_nucleo * h**2 / (4 * k)
    return q_nucleo, C

# --------------------------------------------------------
# CREAR LAS 16 MÁSCARAS
# --------------------------------------------------------
mascaras = crear_mascara_nucleos_16(X, Y, L, factor_tamano)

# --------------------------------------------------------
# CONSTRUIR MATRIZ C_mask (fuente de calor)
# --------------------------------------------------------
C_mask = np.zeros_like(X)

for i in range(16):
    Pi = potencias[i]

    if Pi == 0:
        continue  # núcleo apagado

    qg_i, Ci = calcular_qg_y_C(Pi, L, k, h, factor_tamano)
    C_mask[mascaras[i]] = Ci

# --------------------------------------------------------
# SOLVER SOR
# --------------------------------------------------------
def solve_poisson_sor(C_mask, omega=1.7, T_bound=300.0, tol=1e-6, max_iter=10000, verbose=False):
    T = np.ones_like(C_mask) * T_bound
    T[0, :] = T_bound
    T[-1, :] = T_bound
    T[:, 0] = T_bound
    T[:, -1] = T_bound


    start = time()
    for it in range(1, max_iter+1):
        T_old = T.copy()

        for i in range(1, T.shape[0]-1):
            for j in range(1, T.shape[1]-1):
                t_new = 0.25 * (
                    T[i+1, j] + T[i-1, j] +
                    T[i, j+1] + T[i, j-1]
                ) + C_mask[i, j]
                T[i, j] = (1-omega)*T[i, j] + omega * t_new

        err = np.max(np.abs(T - T_old))

        if verbose and it % 200 == 0:
            print(f"Iteración {it} | error = {err:.3e}")

        if err < tol:
            if verbose:
                print("Convergencia alcanzada en:", it)
            break

    elapsed = time() - start
    return T, it, err, elapsed

# --------------------------------------------------------
# EJECUTAR SOLVER
# --------------------------------------------------------
print("="*60)
print("INICIANDO SOLVER SOR...")
print(f"Potencia total del chip: {np.sum(potencias):.2f} W")
print(f"Núcleos activos: {np.sum(potencias > 0)} de 16")
print("="*60)

T_sor, it_sor, err_sor, t_sor = solve_poisson_sor(
    C_mask, omega=1.7, T_bound=300.0, tol=1e-6, max_iter=10000, verbose=True
)

print("="*60)
print(f"SOR: iter={it_sor}, error={err_sor:.3e}, tiempo={t_sor:.2f}s")
print(f"Temperatura máxima: {np.max(T_sor):.2f} K")
print(f"Temperatura mínima: {np.min(T_sor):.2f} K")
print(f"Incremento máximo: {np.max(T_sor) - 300.0:.2f} K")
print("="*60)

# --------------------------------------------------------
# CALCULAR GRADIENTES Y FLUJOS DE CALOR
# --------------------------------------------------------
Gy, Gx = np.gradient(T_sor, h, h)
qx = -k * Gx
qy = -k * Gy
q_magnitude = np.sqrt(qx**2 + qy**2)

# --------------------------------------------------------
# FIGURA 1: MAPA DE CALOR DE TEMPERATURA
# --------------------------------------------------------
plt.figure(figsize=(7, 6))
im1 = plt.pcolormesh(X, Y, T_sor, shading='auto', cmap='hot')
plt.title('Temperatura T [K]', fontsize=14, fontweight='bold')
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
plt.gca().set_aspect('equal')
plt.colorbar(im1, label='T [K]')
plt.tight_layout()

# --------------------------------------------------------
# FIGURA 3: MAGNITUD Y DIRECCIÓN DEL FLUJO (STREAMPLOT PLASMA)
# --------------------------------------------------------
plt.figure(figsize=(7, 6))
im2 = plt.pcolormesh(X, Y, q_magnitude, shading='auto', cmap='plasma')
plt.streamplot(X, Y, qx, qy, density=1.5, color='white', linewidth=1, arrowsize=0.8)
plt.title('Magnitud y dirección del flujo de calor', fontsize=14, fontweight='bold')
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
plt.gca().set_aspect('equal')
plt.colorbar(im2, label='|q| [W/m²]')
plt.tight_layout()

# --------------------------------------------------------
# FIGURA 4: CAMPO VECTORIAL INTERACTIVO CON DESLIZADOR
# --------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 8))
plt.subplots_adjust(bottom=0.15)

def update(val):
    ax.clear()
    step = int(slider.val)

    heat = ax.pcolormesh(
        X, Y, T_sor,
        shading='auto',
        cmap='inferno',
        alpha=0.90
    )

    mag_sample = q_magnitude[::step, ::step]
    qx_sample = qx[::step, ::step]
    qy_sample = qy[::step, ::step]

    mag_sample_safe = np.where(mag_sample > 0, mag_sample, 1)

    qx_norm = qx_sample / mag_sample_safe
    qy_norm = qy_sample / mag_sample_safe

    mag_log = np.log10(mag_sample + 1)
    scale_factor = mag_log / (np.max(mag_log) + 1e-10)

    ax.quiver(
        X[::step, ::step], Y[::step, ::step],
        qx_norm * scale_factor,
        qy_norm * scale_factor,
        q_magnitude[::step, ::step],
        cmap='viridis',
        scale=15,
        width=0.003,
        headwidth=4,
        headlength=5,
        minlength=0.5,
        pivot='mid'
    )

    ax.set_title('Campo de flujo de calor   q = -k ∇T', fontsize=14, fontweight='bold')
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_aspect('equal')

    fig.canvas.draw_idle()

ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Densidad (step)', 1, 20, valinit=6, valstep=1)
slider.on_changed(update)

update(6)

# --------------------------------------------------------
# FIGURA 5: DISTRIBUCIÓN DE POTENCIAS POR NÚCLEO
# --------------------------------------------------------
plt.figure(figsize=(8, 6))
nucleos = np.arange(1, 17)
colores = plt.cm.hot(potencias / np.max(potencias))
bars = plt.bar(nucleos, potencias, color=colores, edgecolor='black', linewidth=1.2)
plt.axhline(y=np.mean(potencias[potencias > 0]), color='cyan', linestyle='--', 
            linewidth=2, label=f'Promedio activos: {np.mean(potencias[potencias > 0]):.1f} W')
plt.title('Distribución de potencias por núcleo', fontsize=14, fontweight='bold')
plt.xlabel('Núcleo', fontsize=12)
plt.ylabel('Potencia [W]', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()

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

# --------------------------------------------------------
# FIGURA 5: MAGNITUD Y DIRECCIÓN DEL FLUJO (STREAMPLOT)
# --------------------------------------------------------
fig5, ax8 = plt.subplots(figsize=(8, 7))

im8 = ax8.pcolormesh(X*1000, Y*1000, q_magnitude, shading='auto', cmap='plasma')
ax8.streamplot(X*1000, Y*1000, qx, qy, density=1.5, color='white', linewidth=1, arrowsize=0.8)
ax8.set_title('Magnitud y Dirección del Flujo de Calor (Estado Final)', 
              fontsize=13, fontweight='bold')
ax8.set_xlabel('x [mm]', fontsize=11)
ax8.set_ylabel('y [mm]', fontsize=11)
ax8.set_aspect('equal')
plt.colorbar(im8, ax=ax8, label='|q| [W/m²]')
dibujar_nucleos(ax8, P_history[-1])
plt.tight_layout()

# --------------------------------------------------------
# FIGURA 6: CAMPO VECTORIAL INTERACTIVO CON DESLIZADOR
# --------------------------------------------------------
from matplotlib.widgets import Slider

fig6, ax9 = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15)

def update_vector_field(val):
    ax9.clear()
    step = int(slider.val)
    
    # Mapa de calor de fondo
    heat = ax9.pcolormesh(X*1000, Y*1000, T_final,
                          shading='auto', cmap='inferno', alpha=0.90)
    
    # Muestreo del campo vectorial
    mag_sample = q_magnitude[::step, ::step]
    qx_sample = qx[::step, ::step]
    qy_sample = qy[::step, ::step]
    
    # Normalizar direcciones
    mag_sample_safe = np.where(mag_sample > 0, mag_sample, 1)
    qx_norm = qx_sample / mag_sample_safe
    qy_norm = qy_sample / mag_sample_safe
    
    # Escalar por magnitud logarítmica
    mag_log = np.log10(mag_sample + 1)
    scale_factor = mag_log / (np.max(mag_log) + 1e-10)
    
    # Dibujar vectores
    q = ax9.quiver(X[::step, ::step]*1000, Y[::step, ::step]*1000,
                   qx_norm * scale_factor, qy_norm * scale_factor,
                   q_magnitude[::step, ::step],
                   cmap='viridis', scale=15, width=0.003,
                   headwidth=4, headlength=5, minlength=0.5, pivot='mid')
    
    ax9.set_title('Campo de Flujo de Calor   q = -k ∇T', 
                  fontsize=13, fontweight='bold')
    ax9.set_xlabel('x [mm]', fontsize=11)
    ax9.set_ylabel('y [mm]', fontsize=11)
    ax9.set_aspect('equal')
    dibujar_nucleos(ax9, P_history[-1])
    
    fig6.canvas.draw_idle()

# Crear deslizador
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Densidad (step)', 1, 20, valinit=6, valstep=1)
slider.on_changed(update_vector_field)

# Dibujar estado inicial
update_vector_field(6)

print("   ✓ Todas las visualizaciones creadas")
print("\n" + "="*70)
print("VISUALIZACIONES:")
print("  1. Mapas de calor: Inicial → Intermedio → Final")
print("  2. Comparación: Inicial vs Final")
print("  3. Animación de temperatura")
print("  4. Animación de barras de potencia")
print("  5. Magnitud y dirección del flujo de calor (streamplot)")
print("  6. Campo vectorial interactivo (con control de densidad)")
print("="*70)

plt.show()