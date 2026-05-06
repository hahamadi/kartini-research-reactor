import matplotlib.pyplot as plt
import numpy as np

import polynomial_solution_etd1_temp as pset

import simulation_config as config

position_variation = np.arange(10, 100, 25)
position_variation = np.append(position_variation, 100)

markers = ["o","d","s","v",".","x","+"]
colors = ['b','g','r','c','m','y','k']
plt.figure(figsize=(9, 6))

idx = 0
for pos_x_percent in position_variation:
    df = pset.prke_solver_temp(config.H, config.rho_max, config.v_percent, 
                               pos_x_percent, config.t_end, config.dt,
                         config.T0)
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
        color=colors[idx],
        marker=markers[idx],
        fillstyle='none',
        markevery=max(1, len(time) // 40),
        label=f"Position: {pos_x_percent}%"
    )
    idx += 1
    if idx >= len(colors):
        idx = 0

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("log(n(t))", fontsize=12)
plt.title("Log neutron density vs Time for different positions", fontsize=10)
#plt.savefig("grafik_LogNeutron_time_pos{pos_x_percent}_BDF.svg", dpi=300, format='svg', bbox_inches='tight')
plt.grid(True)
plt.tight_layout()
plt.show()