import matplotlib.pyplot as plt
import numpy as np

import simulation_config as config
import polynom_solution_lsoda as psl 
import general_LSDOA_method as glm
import prke_kartini_bdf_references as ref
import rho_x_functions as rxf

cases = {
    f"polynomial-Ref_BDF {1e-5}": ref.reference_simulation_prke_polynomial(rho_polynomial=rxf.rho_polynomial, H=config.H, rho_max=config.rho_max, 
                                 v_percent=config.v_percent, 
                                 pos_x_percent=config.pos_x_percent, 
                                 t_end=config.t_end, dt=config.dt, Lambda=config.Lambda),
    "polynomial": psl.run_simulation_prke(rxf.rho_polynomial, config.H, config.rho_max, 
                                          config.v_percent, 
                                          config.pos_x_percent, 
                                          config.t_end, config.dt, config.Lambda),
    "sin2": glm.run_simulation_prke(rxf.drho_dx_sin2, config.H, config.rho_max,
                                    config.v_percent, config.pos_x_percent, 
                                    config.t_end, config.dt, config.Lambda),
    "gaussian": glm.run_simulation_prke(rxf.drho_dx_gauss, config.H, config.rho_max, 
                                        config.v_percent, config.pos_x_percent, 
                                        config.t_end, config.dt, config.Lambda),
    "sigmoid": glm.run_simulation_prke(rxf.drho_dx_sigmoid, config.H, config.rho_max, 
                                       config.v_percent, config.pos_x_percent, 
                                       config.t_end, config.dt, config.Lambda),
}

markers = ["o","d","s","v",".","x","+"]
colors = ['b','g','r','c','m','y','k']
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
    if key == f"polynomial-Ref_BDF {1e-5}":
        plt.plot(
            time,
            nlog,
            color = colors[idx],
            marker = markers[idx],
            fillstyle='none',
            markevery=max(1, len(time) // 100),
            label=key.replace(" ", "\n")
        )
    else:
        plt.plot(
        time,
        nlog,
        color = colors[idx],
        marker = markers[idx],
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
plt.savefig(f"LogNeutron_At_{config.pos_x_percent}_{config.dt}.svg", dpi=300, format='svg', 
            bbox_inches='tight')

plt.show()
"""
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
"""
