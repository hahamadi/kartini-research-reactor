import numpy as np
import pandas as pd
import os

from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda
from rho_x_functions import drho_dx_sigmoid

cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd, file_delayed_neutron), index_col=None)

beta_group = df_fdn["beta"].to_numpy()
lambda_group = df_fdn["lambda"].to_numpy()

beta = np.sum(beta_group)
m = len(beta_group)

v_rod = (v_percent / 100.0) * H
pos_x = (pos_x_percent / 100.0) * H
t_end_rod = pos_x_percent / v_percent

N = int(np.ceil(t_end / dt)) + 1
times = np.linspace(0.0, t_end, N)

def rod_position(t):
    x = v_rod * t
    return min(x, pos_x)

def rod_velocity(t):
    return v_rod if t < t_end_rod else 0.0

def rhs(t, y):
    """
    y[0]   = rho ($)
    y[1]   = n
    y[2:]  = c_i
    """
    rho = y[0]
    n = y[1]
    c = y[2:]

    x = rod_position(t)
    v = rod_velocity(t)

    # d(rho)/dt = d(rho)/dx * dx/dt
    drho_dt = drho_dx_sigmoid(x, rho, rho_max, H) * v

    rho_abs = rho * beta
    sum_lambda_ci = np.sum(lambda_group * c)

    dn = ((rho_abs - beta) / Lambda) * n + sum_lambda_ci
    dc = (beta_group / Lambda) * n - lambda_group * c

    dydt = np.zeros_like(y)
    dydt[0] = drho_dt
    dydt[1] = dn
    dydt[2:] = dc

    return dydt

# Inisialisasi state
y = np.zeros((N, 2 + m))

# kondisi awal
rho0 = 0.0
n0 = 1.0
c0 = (beta_group / (Lambda * lambda_group)) * n0

y[0, 0] = rho0
y[0, 1] = n0
y[0, 2:] = c0

# RK4 loop
for i in range(N - 1):
    t = times[i]
    h = times[i + 1] - times[i]

    y_i = y[i, :]

    k1 = rhs(t, y_i)
    k2 = rhs(t + 0.5 * h, y_i + 0.5 * h * k1)
    k3 = rhs(t + 0.5 * h, y_i + 0.5 * h * k2)
    k4 = rhs(t + h, y_i + h * k3)

    y[i + 1, :] = y_i + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

rho_t = y[:, 0]
n_t = y[:, 1]
c_t = y[:, 2:]

pos_t = np.array([rod_position(t) for t in times])
rho_abs_t = rho_t * beta

df_out_rk4 = pd.DataFrame({
    "time_s": times,
    "rod_position_m": pos_t,
    "rod_position_%": 100.0 * pos_t / H,
    "rho_dollar": rho_t,
    "rho_abs": rho_abs_t,
    "neutron_density_n": n_t
})

n_t1 = np.log(n_t[~np.isnan(n_t)])