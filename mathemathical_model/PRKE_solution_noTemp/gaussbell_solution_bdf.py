import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import os

from rho_x_functions import drho_dx_gauss
from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda

cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd, file_delayed_neutron), index_col=None)

beta_groups = df_fdn["beta"].to_numpy()
lambda_groups = df_fdn["lambda"].to_numpy()
beta = np.sum(beta_groups)

def run_simulation_prke(H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda, drho_dx_gauss):
    v_rod     = (v_percent / 100.0) * H
    pos_x     = (pos_x_percent / 100.0) * H
    t_end_rod = pos_x / v_rod               # time when rod reaches pos_x
    N         = int(np.ceil(t_end / dt)) + 1
    times     = np.arange(0.0, t_end + dt, dt)

    def rod_position(t):
        x = v_rod * t
        return min(x, pos_x)

    def rod_velocity(t):
        return v_rod if t < t_end_rod else 0.0
    
    def drho_dt(t, rho):
        x = rod_position(t)
        v = rod_velocity(t)
        return drho_dx_gauss(x, rho, rho_max, H) * v
    
    rho_dollar = np.zeros(N)   # reactivity in dollars, shape (N,)
    rho_dollar[0] = 0.0
 
    for i in range(N - 1):
        t_i = times[i]
        h_i = times[i + 1] - times[i]
        r_i = rho_dollar[i]
 
        k1 = drho_dt(t_i, r_i)
        k2 = drho_dt(t_i + 0.5 * h_i, r_i + 0.5 * h_i * k1)
        k3 = drho_dt(t_i + 0.5 * h_i, r_i + 0.5 * h_i * k2)
        k4 = drho_dt(t_i + h_i, r_i + h_i * k3)
 
        rho_dollar[i + 1] = r_i + (h_i / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
 
    rho_abs_arr = rho_dollar * beta  

    rho_abs_interp = CubicSpline(times, rho_abs_arr)

    def pke_rhs(t, y):
        n, c    = y[0], y[1:]
        rho_abs = float(rho_abs_interp(t))
        dn_dt   = ((rho_abs - beta) / Lambda) * n + np.dot(lambda_groups, c)
        dc_dt   = (beta_groups / Lambda) * n - lambda_groups * c
        return np.concatenate(([dn_dt], dc_dt))
    
    n0 = 1.0
    c0 = (beta_groups / (Lambda * lambda_groups)) * n0
    y0 = np.concatenate(([n0], c0))

    sol = solve_ivp(fun=pke_rhs, t_span=(times[0], times[-1]), y0=y0, method="BDF", t_eval=times)
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
 
    n_t = sol.y[0, :]
    c_t = sol.y[1:, :].T
 
    # rod position at each output time
    pos_t = np.array([rod_position(t) for t in sol.t])
 
    precursor_cols = {f"precursor_C{i+1}": c_t[:, i] for i in range(c_t.shape[1])}
 
    return pd.DataFrame({
        "time_s"           : sol.t,
        "rod_position_m"   : pos_t,
        "rod_position_%" : 100.0 * pos_t / H,
        "rho_dollar"       : rho_dollar,     # from RK4 stage
        "rho_abs"          : rho_abs_arr,    # rho_dollar * beta
        "neutron_density_n"  : n_t
    })