import matplotlib.pyplot as plt
import numpy as np

from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda
from rho_x_functions import rho_polynomial , drho_dx_sin2, drho_dx_sigmoid, drho_dx_gauss

import polynom_solution_bdf as psb
import sinkuadrat_solution_bdf as ssb
import sigmoid_solution_bdf as sigsb
import gaussbell_solution_bdf as gbb

case = {
    "polynomBDF": [psb.run_simulation_prke, rho_polynomial],
    "sin2BDF": [ssb.run_simulation_prke, drho_dx_sin2],
    "sigmoidBDF": [sigsb.run_simulation_prke, drho_dx_sigmoid],
    "gaussBDF": [gbb.run_simulation_prke, drho_dx_gauss]
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
plt.title("Log neutron density vs Time RK4 Polynomial")
#plt.savefig("grafik_LogNeutron_time_pos{pos_x_percent}_BDF.svg", dpi=300, format='svg', bbox_inches='tight')
plt.grid(True)
plt.tight_layout()
plt.show()

"""
plt.figure()
plt.plot(dfPolyRk4["time_s"].values, dfPolyRk4["neutron_density_n"].values,
         marker='o', fillstyle='none', label="polynomRK4")
plt.plot(dfSinEk4["time_s"].values, dfSinEk4["neutron_density_n"].values, 
         marker='o', fillstyle='none', label="sin2Ek4")
plt.plot(dfGaussRk4["time_s"].values, dfGaussRk4["neutron_density_n"].values, 
         marker='o', fillstyle='none', label="gaussRk4")
plt.plot(dfSigmoidRk4["time_s"].values, dfSigmoidRk4["neutron_density_n"].values, 
         marker='o', fillstyle='none', label="sigmoidRk4")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("n(t)")
plt.title("Number of neutrons vs Time")
plt.grid()
plt.show()

plt.figure()
plt.plot(dfPolyRk4["time_s"].values, np.log(dfPolyRk4["neutron_density_n"].values),
         marker='o', fillstyle='none', label="polynomRk4")
plt.plot(dfSinEk4["time_s"].values, np.log(dfSinEk4["neutron_density_n"].values), 
         marker='o', fillstyle='none', label="sin2Rk4")
plt.plot(dfGaussRk4["time_s"].values, np.log(dfGaussRk4["neutron_density_n"].values), 
         marker='o', fillstyle='none', label="gaussRk4")
plt.plot(dfSigmoidRk4["time_s"].values, np.log(dfSigmoidRk4["neutron_density_n"].values), 
         marker='o', fillstyle='none', label="sigmoidRk4")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("log(n(t))")
plt.title("Number of neutrons vs Time")
plt.grid()
plt.show()

plt.figure()
plt.plot(dfPolyRk4["time_s"].values, dfPolyRk4["rho_dollar"].values, 
         marker='o', fillstyle='none', label="polynomRk4")
plt.plot(dfSinEk4["time_s"].values, dfSinEk4["rho_dollar"].values, 
         marker='o', fillstyle='none', label="sin2Ek4")
plt.plot(dfGaussRk4["time_s"].values, dfGaussRk4["rho_dollar"].values, 
         marker='o', fillstyle='none', label="gaussRk4")
plt.plot(dfSigmoidRk4["time_s"].values, dfSigmoidRk4["rho_dollar"].values, 
         marker='o', fillstyle='none', label="sigmoidRk4")
plt.xlabel("Time (s)")
plt.ylabel("Reactivity ($)")
plt.title("Reactivity vs Time")
plt.legend()
plt.grid()
plt.show()
"""