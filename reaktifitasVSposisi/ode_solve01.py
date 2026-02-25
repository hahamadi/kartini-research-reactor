import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import reference_function as rf

H = 0.38
rho_max = 1.95

rho_target = 1.95

# Tebakan awal C
C0 = 0.5
C1 = 2.0

x_end = 0.38
h=0.01

x0 = 0
rho0 = 0

tau = 1 #konstanta lag reaktivitas

df = pd.read_excel('data_ref_reactivity.xlsx')
xRef = df['posisi_batang'].values
yRef = df['reactivity'].values
#print(xRef)

xEul, yEul = rf.euler_method("sin2", x0, x_end, rho0, rho_max, h)

xRK4, yRK4 = rf.rk4_method("sin2", x0, x_end, rho0, rho_max, h)

xFung = np.arange(0, x_end, 0.01)
yFung = [rf.reference_function(i) for i in xFung]

xAdam, yAB, yAM = rf.adams_method("sin2", x0, x_end, rho0, rho_max, h)

xShoot, yShoot = rf.shooting_rk4_method("sin2", x0, rho0, rho_max, h, x_end, C0, C1, tol=1e-6, max_iter=50)

xETD, yETD = rf.etd1_method("sin2", x0, rho0, rho_max, h, x_end, tau)

#plt.plot(xRef, yRef, 'k.', ls = '-', label='Polynomial_ref')
plt.plot(xFung, yFung, 'k', ls = '-', label='Polynomial_ref')

plt.plot(xEul, yEul, 'ro', ls = '-', markerfacecolor='r', label='Euler')
plt.plot(xRK4, yRK4, 'gv', ls = '-', markerfacecolor='g', label='Runge-Kutta 4')

plt.plot(xAdam, yAB, 'y^', ls = '-', markerfacecolor='y', label = "Adams–Bashforth 2")
plt.plot(xAdam, yAM, 'c<', ls = '-', markerfacecolor='c',label= "Adams–Moulton")
plt.plot(xShoot, yShoot, 'm>', ls='-', markerfacecolor='m', label= "Shooting–RK4")
plt.plot(xETD, yETD, 'rs', ls='-', markerfacecolor='r', label= "ETD1")

plt.xlabel("Control Rod Position (m)")
plt.ylabel("Reactivity ($)")

plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
plt.savefig("reactivityVsPosition.png", dpi=300, bbox_inches='tight')
# Display the plot
plt.show()
    
    
    