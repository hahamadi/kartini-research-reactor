import polynom_solution_rk4V2 as psrk
import numpy as np
import matplotlib.pyplot as plt
from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, Lambda

markers = ['o', 's', 'D', '^', 'v']#, 'P']
dt = [0.010, 0.013, 0.018, 0.019] #[0.005, 0.010, 0.125] #, 0.130, 0.135, 0.150]

plt.figure(figsize=(9, 6))
for i in range(len(dt)):
    print(f"dt {dt[i]}")
    dfrk4 = psrk.run_simulation_prke(H, rho_max, v_percent, pos_x_percent, t_end, dt[i], Lambda, psrk.rho_polynomial)

    nvals = dfrk4["neutron_density_n"].values
    nlog = []
    time = []
    for j in range(len(nvals)):
        if nvals[j] > 0.0:
            nlog.append(np.log(nvals[j]))
            time.append(dfrk4["time_s"].values[j])

    if np.any(nvals <= 0):
        plt.plot(
            time,
            nlog,
            marker=markers[i],
            fillstyle='none',
            markevery=max(1, len(dfrk4) // 40),
            label=f"dt={dt[i]:.3f} s"
        )
    else:
        plt.plot(
            time,
            nlog,
            marker=markers[i],
            fillstyle='none',
            markevery=max(1, len(nvals) // 40),
            label=f"dt={dt[i]:.3f} s"
        )
    
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("log(n(t))", fontsize=12)
plt.title("Log neutron density vs Time RK4 Polynomial")
#plt.savefig("grafik_LogNeutron_time_pos{pos_x_percent}_dtVar_rk4.svg", dpi=300, format='svg', bbox_inches='tight')
plt.grid(True)
plt.tight_layout()
plt.show()