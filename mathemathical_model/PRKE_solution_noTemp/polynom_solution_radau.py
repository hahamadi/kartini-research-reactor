import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import os

cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd, file_delayed_neutron), index_col=None)

Lambda = 4.3e-5
beta_groups = df_fdn["beta"].to_numpy()
lambda_groups = df_fdn["lambda"].to_numpy()
beta = np.sum(beta_groups)

H = 0.38
rho_max = 1.95

v_percent = 1.6579
v_rod = (v_percent / 100.0) * H

pos_x_percent = 35
pos_x = (pos_x_percent / 100.0) * H

t_end = pos_x_percent / v_percent
dt = 0.01

N = int(np.ceil(t_end / dt)) + 1
times = np.arange(0, t_end + dt, dt)

def rod_position(t):
    return min(v_rod * t, pos_x)

def rho_polynomial(x):
    rho = -129.16*x**5 + 279.95*x**4 - 215.04*x**3 + 58.294*x**2 + 1.3702*x
    return min(rho, rho_max)

def rho_abs_from_t(t):
    return rho_polynomial(rod_position(t)) * beta

def rhs(t, y):
    n = y[0]
    c = y[1:]

    rho_abs = rho_abs_from_t(t)

    dn = ((rho_abs - beta) / Lambda) * n + np.sum(lambda_groups * c)
    dc = (beta_groups / Lambda) * n - lambda_groups * c

    return np.concatenate(([dn], dc))

# kondisi awal
n0 = 1.0
c0 = beta_groups / (Lambda * lambda_groups) * n0
y0 = np.concatenate(([n0], c0))

val = solve_ivp(fun = rhs, t_span=(times[0],times[-1]), y0 = y0, method = "Radau", t_eval=times)

#print("success =", val.success)
#print("message =", val.message)

n_t = val.y[0, :]
c_t = val.y[1:, :].T

pos_t = np.array([rod_position(t) for t in val.t])
rho_t = np.array([rho_polynomial(x) for x in pos_t])   # dollar
rho_abs_t = rho_t * beta

df_out_radau = pd.DataFrame({
    "time_s" : times,
    "rod_position_m" : pos_t,
    "rod_position_%" : 100 * pos_t / H,
    "rho_dollar" : rho_t,
    "rho_abs" : rho_abs_t,
    "neutron_density_n" : n_t
    })
