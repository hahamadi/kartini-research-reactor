import matplotlib.pyplot as plt

from polynom_solution_euler import df_out_euler

from polynom_solution_rk4V2 import df_out_rk4
from polynom_solution_radau import df_out_radau
from polynom_solution_bdf import df_out_bdf
from polynom_solution_etd1 import df_out_etd1

plt.figure()

plt.plot(df_out_euler["time_s"].values, df_out_euler["neutron_density_n"].values, marker='^', label="euler")
plt.plot(df_out_rk4["time_s"].values, df_out_rk4["neutron_density_n"].values, marker='.', label="RK4")
plt.plot(df_out_radau["time_s"].values, df_out_radau["neutron_density_n"].values, marker='<', label="radau")
plt.plot(df_out_bdf["time_s"].values, df_out_bdf["neutron_density_n"].values, marker='>', label="BDF")
plt.plot(df_out_etd1["time_s"].values, df_out_etd1["neutron_density_n"].values, marker='+', label="ETD1")

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("n(t)")
plt.title("Number of neutrons vs Time")
plt.grid()
plt.show()

plt.figure()
plt.plot(df_out_euler["time_s"].values, df_out_euler["rho_dollar"].values, marker='^', label="euler")
plt.plot(df_out_rk4["time_s"].values, df_out_rk4["rho_dollar"].values, marker='.', label="RK4")
plt.plot(df_out_radau["time_s"].values, df_out_radau["rho_dollar"].values, marker='<', label="radau")
plt.plot(df_out_bdf["time_s"].values, df_out_bdf["rho_dollar"].values, marker='>', label="BDF")
plt.plot(df_out_etd1["time_s"].values, df_out_etd1["rho_dollar"].values, marker='+', label="ETD1")
plt.xlabel("Time (s)")
plt.ylabel("Reactivity ($)")
plt.title("Reactivity vs Time")
plt.legend()
plt.grid()
plt.show()