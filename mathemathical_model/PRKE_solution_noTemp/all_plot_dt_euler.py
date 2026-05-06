import polynom_solution_eulerV2 as pse
import numpy as np
import matplotlib.pyplot as plt
from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, Lambda

markers = ['o', 's', 'D', '^', 'v']#, 'P']
dt = [0.005, 0.010, 0.012, 0.013, 0.014]#[0.005, 0.010, 0.125] #, 0.130, 0.135, 0.150]

plt.figure(figsize=(9, 6))
for i in range(len(dt)):
    print(f"dt {dt[i]}")
    dfeul = pse.run_simulation_prke(H, rho_max, v_percent, pos_x_percent, t_end, dt[i], Lambda, pse.rho_polynomial)

    nvals = dfeul["neutron_density_n"].values
    nlog = []
    time = []
    for j in range(len(nvals)):
        if nvals[j] > 0.0:
            nlog.append(np.log(nvals[j]))
            time.append(dfeul["time_s"].values[j])
            
    if np.any(nvals <= 0):
        plt.plot(
            time,
            nlog,
            marker=markers[i],
            fillstyle='none',
            markevery=max(1, len(dfeul) // 40),
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
plt.title("Log neutron density vs Time Euler Polynomial")
#plt.savefig("grafik_LogNeutron_time_pos{pos_x_percent}_dtVar.svg", dpi=300, format='svg', bbox_inches='tight')
plt.grid(True)
plt.tight_layout()
plt.show()