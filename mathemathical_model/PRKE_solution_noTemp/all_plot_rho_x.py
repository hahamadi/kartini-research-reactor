import matplotlib.pyplot as plt
import rho_x_functions as rxf

plt.figure()
plt.plot(rxf.xspan, rxf.rho_polynom, marker = "o", fillstyle='none',label="polynomials")
plt.plot(rxf.xspan, rxf.rho_sin2, marker = "d", fillstyle='none', label="sin2")
plt.plot(rxf.xspan, rxf.rho_gauss, marker = "s", fillstyle='none', label="gaussian")
plt.plot(rxf.xspan, rxf.rho_sigmoid, marker = "v", fillstyle='none', label="sigmoid")
plt.xlabel("Control Rod Position (m)", fontsize=12)
plt.ylabel("reactivity ($)", fontsize=12)
plt.title("Reactivity vs Position")
#plt.plot(valx, valrho)
plt.grid()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('grafik_reactivity_posisi.svg', dpi=300, format='svg', bbox_inches='tight')
plt.show()