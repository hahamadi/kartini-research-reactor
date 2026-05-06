import matplotlib.pyplot as plt
import numpy as np

import polynomial_solution_euler_temp as pset
import polynomial_solution_bdf_temp as psbdt
import polynomial_solution_etd1_temp as psetd
import polynomial_solution_rk4_temp as psrk

case = {
        "polynomialEuler" : pset.prke_solver_temp()
        }

allresults = {}
for name, func in case.items():
    df = func
    nvals = df["neutron_density_n"].values
    nlog = []
    time = []
    for i in range(len(nvals)):
        if nvals[i] > 0:
            nlog.append(np.log(nvals[i]))
            time.append(df["time_s"].values[i])
    
    allresults[name] = [df, nvals, nlog, time]

markers = ["o","d","s","v",".","x","+"]
colors = ['b','g','r','c','m','y','k']
plt.figure(figsize=(9, 6))
idmarker = 0
for nm, res in allresults.items():
    print(f"plotting for {nm}...")
    nlogs = res[2]
    times = res[3]
    plt.plot(
        times,
        nlogs,
        color=colors[idmarker],
        marker=markers[idmarker],
        fillstyle='none',
        markevery=max(1, len(times) // 40),
        label=nm
    )
    idmarker += 1
    
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
,
"polynomialRK4" : psrk.prke_solver_temp(),
"polynomialBDF" : psbdt.prke_solver_temp(),
"polynomialETD1" : psetd.prke_solver_temp()
"""