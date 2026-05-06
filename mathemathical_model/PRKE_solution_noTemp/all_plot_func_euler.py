import matplotlib.pyplot as plt
import numpy as np

from simulation_config import pos_x_percent, dt
from polynom_solution_eulerV2 import df_out as dfPE
from sinkuadrat_solution_euler import df_out_euler as dfSinEuler
from gaussbell_solution_euler import df_out_euler as dfGaussEuler
from sigmoid_solution_euler import df_out_euler as dfSigmoidEuler


cases = {
    "polynomial": dfPE,
    "sin2": dfSinEuler,
    "gaussian": dfGaussEuler,
    "sigmoid": dfSigmoidEuler,
}

markers = ["o","d","s","v"]
idx = 0
plt.figure(figsize=(9, 6))
for key, df in cases.items():
    nvals = df["neutron_density_n"].values
    nlog = []
    time = []
    for i in range(len(nvals)):
        if nvals[i] > 0:
            nlog.append(np.log(nvals[i]))
            time.append(df["time_s"].values[i])
    
    plt.plot(
        time,
        nlog,
        marker=markers[idx],
        fillstyle='none',
        markevery=max(1, len(time) // 40),
        label=key
    )
    idx += 1

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel("Time (s)")
plt.ylabel("log(n(t))")
plt.title("log Number of neutrons vs Time")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(dfPE["time_s"].values, dfPE["neutron_density_n"].values,
         marker='o', fillstyle='none', label="polynomial")
plt.plot(dfSinEuler["time_s"].values, dfSinEuler["neutron_density_n"].values, 
         marker='o', fillstyle='none', label="sin2Euler")
plt.plot(dfGaussEuler["time_s"].values, dfGaussEuler["neutron_density_n"].values, 
         marker='o', fillstyle='none', label="gaussEuler")
plt.plot(dfSigmoidEuler["time_s"].values, dfSigmoidEuler["neutron_density_n"].values, 
         marker='o', fillstyle='none', label="sigmoidEuler")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("n(t)")
plt.title("Number of neutrons vs Time")
plt.grid()
plt.show()

plt.figure()
plt.plot(dfPolyEuler["time_s"].values, np.log(dfPolyEuler["neutron_density_n"].values),
         marker='o', fillstyle='none', label="polynomEuler")
plt.plot(dfSinEuler["time_s"].values, np.log(dfSinEuler["neutron_density_n"].values), 
         marker='o', fillstyle='none', label="sin2Euler")
plt.plot(dfGaussEuler["time_s"].values, np.log(dfGaussEuler["neutron_density_n"].values), 
         marker='o', fillstyle='none', label="gaussEuler")
plt.plot(dfSigmoidEuler["time_s"].values, np.log(dfSigmoidEuler["neutron_density_n"].values), 
         marker='o', fillstyle='none', label="sigmoidEuler")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("log(n(t))")
plt.title("Number of neutrons vs Time")
plt.grid()
plt.show()

plt.figure()
plt.plot(dfPolyEuler["time_s"].values, dfPolyEuler["rho_dollar"].values, 
         marker='o', fillstyle='none', label="polynomEuler")
plt.plot(dfSinEuler["time_s"].values, dfSinEuler["rho_dollar"].values, 
         marker='o', fillstyle='none', label="sin2Euler")
plt.plot(dfGaussEuler["time_s"].values, dfGaussEuler["rho_dollar"].values, 
         marker='o', fillstyle='none', label="gaussEuler")
plt.plot(dfSigmoidEuler["time_s"].values, dfSigmoidEuler["rho_dollar"].values, 
         marker='o', fillstyle='none', label="sigmoidEuler")
plt.xlabel("Time (s)")
plt.ylabel("Reactivity ($)")
plt.title("Reactivity vs Time")
plt.legend()
plt.grid()
plt.show()
