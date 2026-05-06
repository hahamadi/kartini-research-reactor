import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda
from rho_x_functions import rho_polynomial

cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd, file_delayed_neutron), index_col=None)

beta_groups = df_fdn["beta"].to_numpy(dtype=float)
lambda_groups = df_fdn["lambda"].to_numpy(dtype=float)
beta = np.sum(beta_groups)
#group_mem = df_fdn["beta"].to_numpy()

def run_simulation_prke(H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda, rho_function=rho_polynomial):
    v_rod = (v_percent / 100.0) * H
    pos_x = (pos_x_percent / 100.0) * H

    N = int(np.ceil(t_end / dt)) + 1
    times = np.arange(0, t_end + dt, dt)

    def rod_position(t):
        return min(v_rod * t, pos_x)

    def rho_abs_from_t(t):
        return rho_polynomial(rod_position(t)) * beta

    def phi1(z):
        if abs(z) < 1e-8:
            return 1.0 + z/2.0 + z*z/6.0
        return (np.exp(z)-1.0)/z
    
    pos_t = np.zeros_like(times)
    rho_t = np.zeros_like(times)
    rho_abs_t = np.zeros_like(times)

    rho_t[0] = 0.0 #rho_abs
    rho_abs_t[0] = beta * rho_t[0]

    n_t = np.zeros_like(times)
    n_t[0] = 1.0
    c_t = np.zeros((N, len(beta_groups)))

    for ci2 in range(len(beta_groups)):
        beta_i = df_fdn.loc[ci2, "beta"]
        lam_i = df_fdn.loc[ci2, "lambda"]
        c_t[0, ci2] = (beta_i / (Lambda * lam_i)) * n_t[0]
    
    for i in range(1, len(times)):
        delT = times[i] - times[i-1]
        pos_t[i] = pos_t[i-1] + delT * v_rod
        #pos_t[i] = min(H, pos_t[i-1] + delT * v_rod)
        if pos_t[i-1] >= pos_x:
            pos_t[i] = pos_x
            v_rod = 0

        # rho in $ (no linear part -> Euler is ok)
        ch = np.exp(1.0*delT)
        Cworth = (np.pi * rho_max) / (2*H)
        g_rho = Cworth * v_rod * (np.sin(np.pi * pos_t[i-1] / H)**2)
        rho_t[i] = rho_polynomial(pos_t[i])
    
        rho_abs_t[i] = rho_t[i] * beta
        # sum lambda_i c_i
        sum_lambda_ci = float(np.sum(c_t[i-1, :] * lambda_groups))

        # Neutron ETD for n' = Cn*n + sum(lambda c)
        Cn = (rho_abs_t[i-1] - beta) / Lambda
        z = delT * Cn
        expoN = np.exp(z)
        n_t[i] = n_t[i-1]*expoN + delT * phi1(z) * sum_lambda_ci

        # Precursors ETD
        for j in range(len(beta_groups)):
            lam_i = lambda_groups[j]
            beta_i = beta_groups[j]
            expoC = np.exp(-lam_i * delT)
            c_t[i, j] = expoC*c_t[i-1, j] + (1.0 - expoC)/lam_i * ((beta_i/Lambda) * n_t[i-1])
    
    df_out_etd1 = pd.DataFrame({
        "time_s" : times,
        "rod_position_m" : pos_t,
        "rod_position_%" : 100 * pos_t / H,
        "rho_dollar" : rho_t,
        "rho_absolute_t" : rho_abs_t,
        "neutron_density_n" : n_t
        }) 
    return df_out_etd1