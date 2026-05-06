import numpy as np
import pandas as pd

import simulation_config as config
import rho_x_functions as rho_func


def rod_position(t, v_rod, pos_x):
    return min(v_rod * t, pos_x)

def rho_abs_from_t(t, v_rod, pos_x):
    return rho_func.rho_polynomial(rod_position(t, v_rod, pos_x)) * config.beta

def rhs(t, y, v_rod, pos_x):
    T = y[0]
    n = y[1]
    c = y[2:]

    rho_abs_time = rho_abs_from_t(t, v_rod, pos_x)

    dT = -config.beta_T*(T - config.T0) + config.kappa_T * n

    rho_abs = rho_abs_time - config.alpha_T_abs_per_K * (T - config.T0) 

    dn = ((rho_abs - config.beta) / config.Lambda) * n + np.sum(config.lambda_groups * c)
    dc = (config.beta_groups / config.Lambda) * n - config.lambda_groups * c

    return np.concatenate(([dT], [dn], dc))

def prke_solver_temp():
    Temp = np.zeros_like(config.times)
    rho_net_abs = np.zeros_like(config.times)

    Temp[0] = config.T0
    
    pos_t = np.zeros_like(config.times)

    rho_t = np.zeros_like(config.times)
    rho_abs_t = np.zeros_like(config.times)

    n0 = 1.0
    c0 = config.beta_groups / (config.Lambda * config.lambda_groups) * n0
    
    y0 = np.concatenate(([config.T0],[n0], c0))
   
    Y = np.zeros((len(config.times), len(y0)))
    Y[0, :] = y0

    pos_t[0] = rod_position(config.times[0], config.v_rod, config.pos_x)
    rho_t[0] = rho_func.rho_polynomial(pos_t[0])
    rho_abs_t[0] = rho_t[0] * config.beta
    rho_net_abs[0] = rho_abs_t[0] - config.alpha_T_abs_per_K * (config.T0 - config.T0)

    for idx in np.arange(1, len(config.times), 1):
        t = config.times[idx-1]
        h = config.times[idx] - config.times[idx-1]
        y = Y[idx-1, :]

               
        k1 = rhs(t, y, config.v_rod, config.pos_x)
        k2 = rhs(t + 0.5*h, y + 0.5*h*k1, config.v_rod, config.pos_x)
        k3 = rhs(t + 0.5*h, y + 0.5*h*k2, config.v_rod, config.pos_x)
        k4 = rhs(t + h, y + h*k3, config.v_rod, config.pos_x)

        Y[idx, :] = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        pos_t[idx] = rod_position(config.times[idx], config.v_rod, config.pos_x)
        rho_t[idx] = rho_func.rho_polynomial(pos_t[idx])
        rho_abs_t[idx] = rho_t[idx] * config.beta
        Temp[idx] = Y[idx, 0]
        rho_net_abs[idx] = rho_abs_t[idx] - config.alpha_T_abs_per_K * (Temp[idx] - config.T0)
    
    n_t = Y[:, 1]

    df = pd.DataFrame({
        "time_s" : config.times,
        "rod_position_m" : pos_t,
        "rod_position_%" : 100 * pos_t / config.H,
        "rho_dollar" : rho_t,
        "rho_abs" : rho_abs_t,
        "rho_abs_all" : rho_net_abs,
        "temperature_K" : Temp,
        "neutron_density_n" : n_t
        })
    return df


