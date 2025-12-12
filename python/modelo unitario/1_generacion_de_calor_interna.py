
L = 0.02 # Lado del chip [ m ]
Nx , Ny = 101 , 101 # Malla 101 x101
h = L / ( Nx - 1) # Espaciado entre nodos [ m ]
t_espesor = 1.0 # Espesor unitario ( modelo 2 D 
k = 130                     # Conductividad térmica del silicio

def calcular_qg_y_C(P_nucleo, L, k, h, factor_tamano=8):
    A_nucleo = (L / factor_tamano)**2
    t_espesor = 1.0           # modelo 2D
    q_nucleo = P_nucleo / (A_nucleo * t_espesor)   # W/m^3
    C = q_nucleo * h**2 / (4 * k)
    return q_nucleo, C

P_total = 120
P_nucleo = P_total

qg , C = calcular_qg_y_C ( P_nucleo , L , k , h )

print ("q_g: ", qg, " W/mˆ3")
print ("C: ", C, " K ")