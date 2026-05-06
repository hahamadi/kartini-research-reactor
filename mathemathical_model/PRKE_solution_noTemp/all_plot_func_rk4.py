import matplotlib.pyplot as plt
import numpy as np

from polynom_solution_rk4V2 import df_out_rk4 as dfPolyRk4
from sinkuadrat_solution_rk4 import df_out_rk4 as dfSinEk4
from gaussbell_solution_rk4 import df_out_rk4 as dfGaussRk4
from sigmoid_solution_rk4 import df_out_rk4 as dfSigmoidRk4

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
