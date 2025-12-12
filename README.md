# Modelado y Simulación de la Distribución de Temperatura en un Procesador

Conjunto de scripts Python que implementan modelos de conducción térmica en un chip de silicio con múltiples núcleos. Incluye análisis estacionarios, transitorios y visualización de campos de temperatura y flujo de calor.

---


## 🎯 Descripción General

Este proyecto implementa simulaciones de transferencia de calor en procesadores multicore usando el **método de diferencias finitas**. Se resuelve la **ecuación de Poisson** para estado estacionario:

$$\nabla^2 T = -\frac{q}{k}$$

donde:
- $T$ = temperatura
- $q$ = generación de calor volumétrica
- $k$ = conductividad térmica

**Características principales:**
- ✅ Modelos estacionarios con 1, 16 núcleos distribuidos uniformemente
- ✅ Modelos con distribuciones de potencia no uniformes
- ✅ Análisis transitorio con evolución temporal
- ✅ Visualización de campos de temperatura y flujo de calor
- ✅ Cálculo del gradiente de temperatura y flujos vectoriales

---


## 🔧 Requisitos

```
pip install numpy matplotlib scipy
```

# 🌡️ Parámetros Físicos Comunes

## 🔲 Geometría del Chip

| Parámetro | Valor | Unidad | Descripción |
|-----------|-------|--------|-------------|
| **L** | 0.02 | m | Lado del chip cuadrado |
| **Nx, Ny** | 101-201 | - | Resolución de la malla (puntos) |
| **h** | L/(Nx-1) | m | Espaciado entre nodos |

## ⚛️ Propiedades del Material (Silicio)

| Parámetro | Valor | Unidad | Descripción |
|-----------|-------|--------|-------------|
| **k** | 130 | W/(m·K) | Conductividad térmica |
| **ρ** | 2330 | kg/m³ | Densidad (para transitorios) |
| **cₚ** | 710 | J/(kg·K) | Calor específico (para transitorios) |
| **α** | k/(ρ·cₚ) | m²/s | Difusividad térmica |

## 🌡️ Condiciones de Frontera

| Parámetro | Valor | Unidad | Descripción |
|-----------|-------|--------|-------------|
| **T_bound** | 300 | K | Temperatura en los bordes |
| **T_inicial** | 300 | K | Temperatura inicial (transitorios) |

## ⚙️ Parámetros de Solver

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **ω** | 1.7 | Factor de sobre-relajación SOR |
| **tol** | 1e-6 | Tolerancia de convergencia |
| **max_iter** | 20000 | Iteraciones máximas |

---

## 📚 Guía de Scripts
Modelo Unitario
Análisis básico con un único núcleo en el centro del chip.

# `1_generacion_de_calor_interna.py`
Objetivo: Calcular los parámetros adimensionales para la ecuación discreta.

Fórmulas:

$$
q_g = \frac{P_nucleo}{A_{nucleo} \cdot t_{espesor}} [W/m^3]
$$
$$
C = \frac{q_g \cdot h^2}{4k} [K]
$$

### parametros ajustables

````
L = 0.02              # Lado del chip [m]
Nx, Ny = 101, 101     # Malla 101×101
k = 130               # Conductividad térmica [W/(m·K)]
P_total = 120         # Potencia total [W]
factor_tamano = 8     # Núcleo = L/8
````

### salida

````
q_g: 6000000.0  W/m³
C:   0.00923077  K
````

# `2_Creacion_de_un_nucleo.py`

**Objetivo:** Visualizar la máscara booleana que define dónde se genera calor en el núcleo del procesador.

## 📋 Descripción

Este script crea y visualiza una máscara booleana 2D que representa la ubicación del núcleo en el chip de silicio. La máscara se utiliza posteriormente para definir las regiones donde se genera calor en la simulación térmica.

## 🧮 Método

### **Definición de la Máscara**
Para un chip cuadrado de lado $L$ y un núcleo centrado de radio $R$, la máscara booleana $M(x,y)$ se define como:

$$
M(x,y) = \begin{cases}
1 & \text{si } \sqrt{(x - x_c)^2 + (y - y_c)^2} \leq R \\
0 & \text{en otro caso}
\end{cases}
$$

donde:
- $(x_c, y_c)$ = centro del chip = $\left(\frac{L}{2}, \frac{L}{2}\right)$
- $R$ = radio del núcleo (típicamente 0.002 m = 2 mm)

### **Generación de la Malla**
Se crea una malla discreta con $N_x \times N_y$ puntos:

$$
x_i = i \cdot h, \quad i = 0, 1, \dots, N_x-1
$$
$$
y_j = j \cdot h, \quad j = 0, 1, \dots, N_y-1
$$

con $h = \frac{L}{N_x-1}$.

### **Cálculo de la Máscara Discreta**
Para cada punto de la malla $(x_i, y_j)$:

$$
M[i,j] = \begin{cases}
1 & \text{si } (x_i - x_c)^2 + (y_j - y_c)^2 \leq R^2 \\
0 & \text{en otro caso}
\end{cases}
$$

## 🔧 Función Clave

```python
def crear_mascara_nucleo(Nx, Ny, L, R_nucleo):
    """
    Crea una máscara booleana para un núcleo circular centrado.
    """
    
    return mascara, X, Y
```

# `3_Calculo_por_mallas.py`

**Objetivo:** Resolver la ecuación de Poisson 2D usando SOR.

## 🧮 Ecuación Discreta

$$
T_{i,j} = \frac{1}{4} \left( T_{i+1,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1} \right) + C_{i,j}
$$

donde:
- $T_{i,j}$ = temperatura en nodo (i,j)
- $C_{i,j} = \frac{h^2}{4} \cdot \frac{q_{g_{i,j}}}{k}$ = término fuente

## 🔧 Algoritmo SOR

**Paso iterativo:**

1. Calcular valor Gauss-Seidel:
   $$T_{i,j}^{GS} = \frac{1}{4} \left( T_{i+1,j}^{n} + T_{i-1,j}^{n+1} + T_{i,j+1}^{n} + T_{i,j-1}^{n+1} \right) + C_{i,j}$$

2. Aplicar sobre-relajación:
   $$T_{i,j}^{n+1} = T_{i,j}^{n} + \omega \left( T_{i,j}^{GS} - T_{i,j}^{n} \right)$$

## 🎯 Condiciones de Borde

**Dirichlet en todos los bordes:**
- $T(0,y) = T(L,y) = T(x,0) = T(x,L) = 300$ K

## 📊 Salida
- Campo de temperatura 2D convergido $T_{i,j}$
- Número de iteraciones realizadas
- Error final

## ⚙️ Parámetros Típicos
```python
omega = 1.7      # Factor de sobre-relajación
tol = 1e-6       # Tolerancia de convergencia
max_iter = 20000 # Iteraciones máximas
T_bound = 300    # Temperatura en bordes (K)
```

# `4_visualizacion_unitaria.py`

**Objetivo:** Visualizar temperatura y flujo de calor en 3 gráficos.

## 📊 Gráficos

1. **Mapa de temperatura** - `pcolormesh` con colormap `'hot'`
2. **Magnitud del flujo** - $\lvert q \rvert = \sqrt{q_x^2 + q_y^2}$
3. **Campo vectorial** - `streamplot` con flechas blancas

## 🧮 Cálculos

**Flujo de calor:**
$$
\vec{q} = -k \nabla T = -k \left( \frac{\partial T}{\partial x}, \frac{\partial T}{\partial y} \right)
$$

**Derivadas discretas:**
- $\frac{\partial T}{\partial x} \approx \frac{T_{i+1,j} - T_{i-1,j}}{2h}$
- $\frac{\partial T}{\partial y} \approx \frac{T_{i,j+1} - T_{i,j-1}}{2h}$

## 🎨 Visualización

- **Temperatura:** colormap `'hot'` (rojo = caliente)
- **Flujo:** colormap `'viridis'` + vectores blancos
- **Submuestreo** para claridad en campo vectorial

# Modelo 16 Núcleos Uniformes

## `5_16mascaras_uniforme.py`

**Objetivo:** Crear 16 máscaras en rejilla 4×4 con potencia uniforme.

**Parámetros:**
- Chip: 0.02 × 0.02 m
- 16 núcleos en rejilla 4×4
- Radio núcleo: 0.002 m
- Potencia uniforme por núcleo

**Salida:**
1. Campo de temperatura con 16 picos
2. Gráfico `quiver` de flujos vectoriales
3. Distribución simétrica de calor


# `6_generar_matrix_no_Uniforme.py`

**Objetivo:** Generar distribuciones de potencia aleatorias para experimentos.

## 🔧 Función Principal

```python
def generar_y_promediar_matrices(n_matrices=10000, n_elementos=16,  
                                 suma_objetivo=range(115, 131)):
```

**Retorna:**
- `matriz_promedio`: Array de 16 potencias (algunos pueden ser 0)
- `todas`: Todas las matrices generadas

## 🔄 Algoritmo

1. **Genera** 10000 matrices con distribuciones aleatorias exponenciales
2. **Normaliza** para que sumen entre 115-130 W
3. **Promedia** y anula 1-3 componentes aleatoriamente

## 🚀 Uso

```python
potencias, _ = generar_y_promediar_matrices()
print(potencias)  # [7.2, 9.8, 22.5, ..., 16.9] W
```

# `7_modelo_NO_uniforme.py`

**Objetivo:** Estudiar efectos de desbalance térmico en el chip.

## ⚙️ Potencias Personalizables

```python
potencias = np.array([  
    7.2, 9.8, 22.5, 1.2,    # Fila 1  
    8.9, 3.6, 12.3, 15.5,   # Fila 2  
    10.1, 15.4, 9.7, 3.2,   # Fila 3  
    4.8, 30.0, 17.1, 16.9   # Fila 4  
])  
# P_total ≈ 168 W
```

## ✨ Características

- Construcción dinámica de `C_mask` con potencias variables
- Resolución **201×201** para precisión
- Visualización completa con `streamplot` y `quiver` interactivo
- Deslizador de densidad (igual que `7_MapaCalor_Flujo.py`)

## 📝 Secciones del Código

1. **Parámetros generales:** Dominio y malla
2. **Generación de potencias:** Array de 16 valores
3. **Máscaras de núcleos:** Rejilla 4×4
4. **Construcción de C_mask:** Suma ponderada de máscaras
5. **Solver SOR:** Convergencia iterativa
6. **Visualización:** 4 gráficos (T, q, streamplot, quiver interactivo)

---

# 🌡️ Modelos Transitorios

### Análisis temporal del cambio de temperatura en el tiempo

# `8_modelo_transitorio_unitario.py`

**Objetivo:** Simular la evolución térmica transitoria de un núcleo único mediante la ecuación de difusión de calor.

## 🧮 Ecuación Resuelta

$$\rho c_p \frac{\partial T}{\partial t} = k \nabla^2 T + q$$

**Donde:**
- $T$ = temperatura [K]
- $\rho$ = densidad del silicio [kg/m³]
- $c_p$ = calor específico [J/(kg·K)]
- $k$ = conductividad térmica [W/(m·K)]
- $q$ = generación de calor volumétrica [W/m³]

## ⏱️ Método Numérico

- **Discretización temporal:** Forward Euler implícito
- **Discretización espacial:** Diferencias finitas centrales
- **Estabilidad:** $r = \alpha \frac{\Delta t}{h^2} < 0.25$ (criterio CFL)

## ⚡ Características

✅ Operaciones vectorizadas con NumPy (10-50x más rápido que bucles Python)
✅ Malla 51×51 nodos para 10 mm × 10 mm
✅ Un núcleo centrado generando 20 W
✅ Condiciones de borde Dirichlet a 300 K
✅ Tiempo simulado: 10 segundos

## 📊 Salidas

1. **Animación:** Evolución de temperatura + gráfico de T_máxima
2. **Comparación:** Estado inicial vs final
3. **Estadísticas:** T_máxima final y incremento total

## 🔧 Parámetros Ajustables

```python
L = 0.01              # Tamaño chip [m]
dt = 0.5 * dt_max     # Paso temporal estable
t_final = 10.0        # Tiempo total [s]
potencia = 20.0       # Potencia núcleo [W]
T_inicial = 300.0     # Temperatura inicial [K]
```

---

# `9_modelo_transitorio_16_nucleos.py`

**Objetivo:** Simular el comportamiento dinámico de 16 núcleos con potencias variables en el tiempo.

## 🧮 Ecuación Resuelta

$$\rho c_p \frac{\partial T}{\partial t} = k \nabla^2 T + q(t)$$

Con generación de calor **variable y dinámica** por núcleo.

## 🌊 Potencia Dinámica

Cada núcleo activo varía su potencia sinusoidalmente:

$$P_i(t) = P_{base,i} \left( 1 + A_i \sin(2\pi f_i t) \right)$$

**Donde:**
- $P_{base,i}$ = potencia nominal del núcleo $i$ [W]
- $A_i$ = amplitud de variación (0-0.8)
- $f_i$ = frecuencia de oscilación (0.3-1.5 Hz)

## ⚡ Características Avanzadas

✅ 16 núcleos en grilla 4×4 con potencias individuales
✅ Algunos núcleos inactivos (potencia = 0)
✅ Variación temporal realista que simula carga de trabajo
✅ Malla 101×101 nodos para 20 mm × 20 mm
✅ Cálculo de flujo de calor vectorial
✅ Tiempo simulado: 5 segundos

## 📊 Visualizaciones Incluidas

1. **Mapas de calor:** Inicial → Intermedio → Final
2. **Comparación:** Estado inicial vs final con ΔT
3. **Animación de temperatura:** Con etiquetas de potencia por núcleo
4. **Gráfico de barras animado:** Potencia dinámica por núcleo
5. **Campo vectorial:** Temperatura + flujo de calor

## 🔧 Parámetros de Configuración

```python
# Potencias base [W] - Distribución realista
potencias_base = np.array([
    [18.0,  0.0, 25.0,  2.5],  # Fila 1
    [ 8.5, 15.0,  0.0, 15.0],  # Fila 2
    [ 0.5, 28.0, 12.0,  0.0],  # Fila 3
    [ 6.0, 30.0, 20.0,  9.5]   # Fila 4
])

# Variación amplitud (fraccional)
variacion_amplitud = np.array([
    [0.3, 0.0, 0.5, 0.2],  # Núcleos fijos tienen amplitud 0
    [0.4, 0.8, 0.0, 0.6],
    [0.6, 0.4, 0.6, 0.0],
    [0.2, 0.5, 0.4, 0.3]
])

t_final = 5.0          # Tiempo total [s]
```

## 📈 Salida de Resultados

- T máxima final y evolución
- Potencia total y por núcleo
- Número de núcleos activos
- Flujo de calor máximo y promedio
- Rendimiento: ~1000M operaciones/s (GPU-grade)



