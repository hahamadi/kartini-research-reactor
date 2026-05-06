import numpy as np
import pandas as pd

import simulation_config as config
import rho_x_functions as rho_func

def dn_dt(n, t, rho_abs_t, beta, lamb, sum_lambda_ci):
    fun = ((rho_abs_t - beta) / lamb) * n + sum_lambda_ci
    return fun

def dc_dt(n, c_i, beta_i, lambda_i, Lambda):
    return (beta_i / Lambda) * n - lambda_i * c_i

def prke_solver_temp():
    Temp = np.zeros_like(config.times)
    rho_net_abs = np.zeros_like(config.times)

    Temp[0] = config.T0
    rho_net_abs[0] = config.rho_abs

    pos_t = np.zeros_like(config.times)

    rho_t = np.zeros_like(config.times)
    rho_abs_t = np.zeros_like(config.times)

    rho_t[0] = rho_abs_t[0] = config.rho_abs
    n_t = np.zeros_like(config.times)
    n_t[0] = 1.0

    c_t = np.zeros((len(config.times), len(config.beta_groups)))

    for i in range(len(config.beta_groups)):
        c_t[0, i] = (config.beta_groups[i] / (config.Lambda * config.lambda_groups[i])) * n_t[0]
    
    for idx in np.arange(1, len(config.times), 1):
        h = config.times[idx] - config.times[idx-1]
        v_now = config.v_rod
        
        if config.t_rod_end < config.times[idx]:
            pos_t[idx] = config.pos_x
            v_now = 0.0
        else:
            pos_t[idx] = min(config.H, pos_t[idx-1] + v_now * h)

        rho_t[idx] = rho_func.rho_polynomial(pos_t[idx])
        
        rho_abs_t[idx] = rho_t[idx] * config.beta
        
        dTdt = config.kappa_T * n_t[idx-1] - config.beta_T * (Temp[idx-1] - config.T0)
        Temp[idx] = Temp[idx-1] + dTdt * h

        rho_net_abs[idx] = rho_abs_t[idx] - config.alpha_T_abs_per_K * (Temp[idx] - config.T0)

        lambda_ci = 0
        for ci in np.arange(len(config.beta_groups)):
            lambda_ci += config.lambda_groups[ci] * c_t[idx-1, ci]
        
        n_t[idx] = n_t[idx-1] + h * dn_dt(n_t[idx-1], config.times[idx-1], 
                                          rho_net_abs[idx], config.beta, config.Lambda, lambda_ci)
        
        for ci2 in np.arange(len(config.beta_groups)):
            c_t[idx, ci2] = c_t[idx-1, ci2] + h * dc_dt(n_t[idx-1], c_t[idx-1, ci2], 
                                                        config.beta_groups[ci2], config.lambda_groups[ci2], 
                                                        config.Lambda)
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


