import numpy as np

objetivo = range(115, 131)  # 110 a 130 inclusive

def generar_y_promediar_matrices(n_matrices=10000, n_elementos=16, suma_objetivo=objetivo):
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

# Probar
matriz_promedio, todas = generar_y_promediar_matrices()
potencias = matriz_promedio
print(potencias)