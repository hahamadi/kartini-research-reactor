import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_exp = pd.read_excel("hasil_simulasi_kartini_explicitEuler.xlsx")
df_imp = pd.read_excel("hasil_simulasi_kartini_implicitEuler.xlsx")
df_rk4 = pd.read_excel("hasil_simulasi_kartini_RK4.xlsx")

df_compare = pd.DataFrame()
col = ["neutron_density_n","rod_position_m"]

df_compare["time"] = df_exp["time_s"]

df_compare["time"] = df_exp["time_s"]
df_compare["n_exp"] = df_exp["neutron_density_n"]
df_compare["n_imp"] = df_imp["neutron_density_n"]
df_compare["pos_exp"] = df_exp["rod_position_m"]
df_compare["pos_imp"] = df_imp["rod_position_m"]

df_compare["abs_error"] = abs(df_compare["n_exp"] - df_compare["n_imp"])
df_compare["rel_error_percent"] = (
    abs(df_compare["n_exp"] - df_compare["n_imp"]) /
    df_compare["n_imp"]
) * 100

df_compare["abs_error_pos"] = abs(df_compare["pos_exp"] - df_compare["pos_imp"])
df_compare["rel_error_percent_pos"] = (
    abs(df_compare["pos_exp"] - df_compare["pos_imp"]) /
    df_compare["pos_imp"]
) * 100


L2_error = np.sqrt(np.mean((df_compare["n_exp"] - df_compare["n_imp"])**2))
max_error = np.max(df_compare["abs_error"])
mean_rel_error = np.mean(df_compare["rel_error_percent"])

print("L2 Error :", L2_error)
print("Max Error:", max_error)
print("Mean Relative Error (%):", mean_rel_error)

plt.figure()
plt.plot(df_exp["time_s"], df_exp["neutron_density_n"], label="Explicit Euler")
plt.plot(df_imp["time_s"], df_imp["neutron_density_n"], label="Implicit Euler")
plt.plot(df_rk4["time_s"], df_rk4["neutron_density_n"], label="Runge-Kutta 4")
plt.xlabel("Time (s)")
plt.ylabel("Neutron Density n(t)")
plt.title("Explicit, Implicit Euler and RK4")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(df_exp["time_s"], df_exp["rod_position_m"], label="Explicit Euler")
plt.plot(df_imp["time_s"], df_imp["rod_position_m"], label="Implicit Euler")
plt.xlabel("Time (s)")
plt.ylabel("position (m)")
plt.title("Explicit vs Implicit Euler")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(df_compare["time"], df_compare["rel_error_percent"])
plt.xlabel("Time (s)")
plt.ylabel("Relative Error (%)")
plt.title("Relative Error: Explicit vs Implicit")
plt.grid()
plt.show()