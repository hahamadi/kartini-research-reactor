import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import os
from rho_x_functions import rho_polynomial
from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda

cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd, file_delayed_neutron), index_col=None)

beta_groups = df_fdn["beta"].to_numpy()
lambda_groups = df_fdn["lambda"].to_numpy()
beta = np.sum(beta_groups)

def rod_position(t, v_rod, pos_x):
    return min(v_rod * t, pos_x)

def rho_abs_from_t(t, v_rod, pos_x):
    return rho_polynomial(rod_position(t, v_rod, pos_x)) * beta

def rhs(t, y, v_rod, pos_x):
    n = y[0]
    c = y[1:]

    rho_abs = rho_abs_from_t(t, v_rod, pos_x)

    dn = ((rho_abs - beta) / Lambda) * n + np.sum(lambda_groups * c)
    dc = (beta_groups / Lambda) * n - lambda_groups * c

    return np.concatenate(([dn], dc))

def run_simulation_prke(H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda, rho_polynomial):
    v_rod = (v_percent / 100.0) * H

    pos_x = (pos_x_percent / 100.0) * H

    N = int(np.ceil(t_end / dt)) + 1
    times = np.linspace(0.0, t_end, N)



    # kondisi awal
    n0 = 1.0
    c0 = beta_groups / (Lambda * lambda_groups) * n0
    y0 = np.concatenate(([n0], c0))

    Y = np.zeros((N, len(y0)))
    Y[0, :] = y0

    pos_t = np.zeros(N)
    rho_t = np.zeros(N)
    rho_abs_t = np.zeros(N)

    pos_t[0] = rod_position(times[0], v_rod, pos_x)
    rho_t[0] = rho_polynomial(pos_t[0])
    rho_abs_t[0] = rho_t[0] * beta

    for i in range(1, N):
        t = times[i-1]
        h = times[i] - times[i-1]
        y = Y[i-1, :]

        k1 = rhs(t, y, v_rod, pos_x)
        k2 = rhs(t + 0.5*h, y + 0.5*h*k1, v_rod, pos_x)
        k3 = rhs(t + 0.5*h, y + 0.5*h*k2, v_rod, pos_x)
        k4 = rhs(t + h, y + h*k3, v_rod, pos_x)

        Y[i, :] = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        pos_t[i] = rod_position(times[i], v_rod, pos_x)
        rho_t[i] = rho_polynomial(pos_t[i])
        rho_abs_t[i] = rho_t[i] * beta

    n_t = Y[:, 0]
    c_t = Y[:, 1:]

    df_out_rk4 = pd.DataFrame({
        "time_s" : times,
        "rod_position_m" : pos_t,
        "rod_position_%" : 100 * pos_t / H,
        "rho_dollar" : rho_t,
        "rho_abs" : rho_abs_t,
        "neutron_density_n" : n_t
        })
    return df_out_rk4
    
df_out_rk4 = run_simulation_prke(H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda, rho_polynomial)
"""
v_rod = (v_percent / 100.0) * H

pos_x = (pos_x_percent / 100.0) * H

N = int(np.ceil(t_end / dt)) + 1
times = np.linspace(0.0, t_end, N)



# kondisi awal
n0 = 1.0
c0 = beta_groups / (Lambda * lambda_groups) * n0
y0 = np.concatenate(([n0], c0))

Y = np.zeros((N, len(y0)))
Y[0, :] = y0

pos_t = np.zeros(N)
rho_t = np.zeros(N)
rho_abs_t = np.zeros(N)

pos_t[0] = rod_position(times[0])
rho_t[0] = rho_polynomial(pos_t[0])
rho_abs_t[0] = rho_t[0] * beta

for i in range(1, N):
    t = times[i-1]
    y = Y[i-1, :]

    k1 = rhs(t, y)
    k2 = rhs(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = rhs(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = rhs(t + dt, y + dt*k3)

    Y[i, :] = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    pos_t[i] = rod_position(times[i])
    rho_t[i] = rho_polynomial(pos_t[i])
    rho_abs_t[i] = rho_t[i] * beta

n_t = Y[:, 0]
c_t = Y[:, 1:]

df_out_rk4 = pd.DataFrame({
    "time_s" : times,
    "rod_position_m" : pos_t,
    "rod_position_%" : 100 * pos_t / H,
    "rho_dollar" : rho_t,
    "rho_abs" : rho_abs_t,
    "neutron_density_n" : n_t
    })

plt.figure()
plt.plot(times, rho_t)
plt.xlabel("Time (s)")
plt.ylabel("Reactivity ($)")
plt.title("Reactivity vs Time")
plt.grid()
plt.show()

plt.figure()
plt.plot(times, n_t)
plt.xlabel("Time (s)")
plt.ylabel("n(t)")
plt.title("Number of neutrons vs Time")
plt.grid()
plt.show()
"""