import numpy as np
import matplotlib.pyplot as plt

# Parámetros del chip
L = 0.02  # Longitud total del chip en metros (2 cm)
Nx, Ny = 101, 101  # Número de puntos en las direcciones x e y
h = L / (Nx - 1)  # Espaciado entre puntos en la malla

# Coordenadas
x = np.linspace(0, L, Nx)  # Array de puntos en dirección x
y = np.linspace(0, L, Ny)  # Array de puntos en dirección y
X, Y = np.meshgrid(x, y)  # Malla 2D con las coordenadas

# Función para crear la máscara del núcleo
def crear_mascara_nucleo(X, Y, L, factor_tamaño=16):
    nucleo_L = L / factor_tamaño  # Tamaño del núcleo (1/factor_tamaño del chip total)
    x_min = L/2 - nucleo_L/2  # Límite izquierdo del núcleo (centrado en x)
    x_max = L/2 + nucleo_L/2  # Límite derecho del núcleo (centrado en x)
    y_min = L/2 - nucleo_L/2  # Límite inferior del núcleo (centrado en y)
    y_max = L/2 + nucleo_L/2  # Límite superior del núcleo (centrado en y)
    
    # Crear máscara booleana
    nucleo_mask = np.logical_and.reduce((
        X >= x_min, X <= x_max,
        Y >= y_min, Y <= y_max
    ))
    return nucleo_mask

# Crear la máscara del núcleo usando la función
nucleo_mask = crear_mascara_nucleo(X, Y, L, 16)

# Visualización
plt.figure(figsize=(6,6))
plt.pcolormesh(X, Y, nucleo_mask, shading='auto', cmap='hot')
plt.title('Chip de Silicio con un Solo Núcleo (1/16 del tamaño)')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.gca().set_aspect('equal')
plt.show()