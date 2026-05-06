import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
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

def reference_simulation_prke_polynomial(rho_polynomial=rho_polynomial, H=H, rho_max=rho_max, v_percent=v_percent, pos_x_percent=pos_x_percent, 
                        t_end=t_end, dt=dt, Lambda=Lambda):
    v_rod = (v_percent / 100.0) * H

    h = 1e-5
    
    pos_x = (pos_x_percent / 100.0) * H

    times = np.arange(0, t_end + h, h)

    # kondisi awal
    n0 = 1.0
    c0 = beta_groups / (Lambda * lambda_groups) * n0
    y0 = np.concatenate(([n0], c0))

    val = solve_ivp(fun = rhs, t_span=(times[0],times[-1]), y0 = y0, method = "BDF", 
                    t_eval=times, args=(v_rod, pos_x), rtol=1e-10, atol=1e-12)

    n_t = val.y[0, :]
    c_t = val.y[1:, :].T

    pos_t = np.array([rod_position(t, v_rod, pos_x) for t in val.t])
    rho_t = np.array([rho_polynomial(x) for x in pos_t])   # dollar
    rho_abs_t = rho_t * beta

    df_out_bdf = pd.DataFrame({
        "time_s" : times,
        "rod_position_m" : pos_t,
        "rod_position_%" : 100 * pos_t / H,
        "rho_dollar" : rho_t,
        "rho_abs" : rho_abs_t,
        "neutron_density_n" : n_t
        })
    return df_out_bdf