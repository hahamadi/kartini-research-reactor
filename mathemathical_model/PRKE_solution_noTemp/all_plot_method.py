import matplotlib.pyplot as plt
import numpy as np

from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda
from rho_x_functions import rho_polynomial

import polynom_solution_eulerV2 as pse
import polynom_solution_rk4V2 as psrk4
import polynom_solution_lsoda as pslso
import polynom_solution_bdf as psbdf
import polynom_solution_etd1 as psetd1

case = {
    "polynomETD1": [psetd1.run_simulation_prke, rho_polynomial],
    "polynomRK4": [psrk4.run_simulation_prke, rho_polynomial],
    "polynomBDF": [psbdf.run_simulation_prke, rho_polynomial],
    "polynomEuler": [pse.run_simulation_prke, rho_polynomial],
    "polynomLSODA": [pslso.run_simulation_prke, rho_polynomial]
}

allresults = {}
for name, func in case.items():
    dfsol = func[0](H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda, func[1])
    nvals = dfsol["neutron_density_n"].values
    nlog = []
    time = []
    for i in range(len(nvals)):
        if nvals[i] > 0:
            nlog.append(np.log(nvals[i]))
            time.append(dfsol["time_s"].values[i])
    
    allresults[name] = [dfsol, nvals, nlog, time]

plt.figure(figsize=(9, 6))
for nm, res in allresults.items():
    print(f"plotting for {nm}...")
    nlogs = res[2]
    times = res[3]
    plt.plot(
        times,
        nlogs,
        marker='o',
        fillstyle='none',
        markevery=max(1, len(times) // 40),
        label=nm
    )
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("log(n(t))", fontsize=12)
plt.title("Log neutron density vs Time for Polynomial")
#plt.savefig("grafik_LogNeutron_time_pos{pos_x_percent}_BDF.svg", dpi=300, format='svg', bbox_inches='tight')
plt.grid(True)
plt.tight_layout()
plt.show()