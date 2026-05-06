import numpy as np
import pandas as pd

import simulation_config as config
import rho_x_functions as rho_func

def fung_temp_update_etd1(T_n, t_n, dt, n_t, T0, kappa_T, beta_T):
    
    ch = np.exp(-beta_T*dt)
    forcing = kappa_T * n_t
    yn =  T0 + (T_n - T0) * ch + (forcing / beta_T) * (1.0 - ch)
       
    return yn

def phi1(z):
    if abs(z) < 1e-8:
        return 1.0 + z/2.0 + z*z/6.0
    return (np.exp(z)-1.0)/z

def prke_solver_temp(H, rho_max, v_percent, pos_x_percent, t_end, dt,
                         T0):
    v_rod     = (v_percent / 100.0) * H
    pos_x     = (pos_x_percent / 100.0) * H

    t_rod_end = pos_x_percent / v_percent

    times = np.arange(0, t_end, dt)

    Temp = np.zeros_like(times)
    rho_net_abs = np.zeros_like(times)

    Temp[0] = T0
    rho_net_abs[0] = 0.0

    pos_t = np.zeros_like(times)

    rho_t = np.zeros_like(times)
    rho_abs_t = np.zeros_like(times)

    rho_t[0] = rho_abs_t[0] = rho_net_abs[0]
    n_t = np.zeros_like(times)
    n_t[0] = 1.0

    c_t = np.zeros((len(times), len(config.beta_groups)))

    for i in range(len(config.beta_groups)):
        c_t[0, i] = (config.beta_groups[i] / (config.Lambda * config.lambda_groups[i])) * \
            n_t[0]
    
    for idx in np.arange(1, len(times), 1):
        h = times[idx] - times[idx-1]
        v_now = config.v_rod
        if t_rod_end < times[idx]:
            pos_t[idx] = pos_x
            v_now = 0.0
        else:
            pos_t[idx] = min(H, pos_t[idx-1] + v_now * h)

        rho_t[idx] = rho_func.rho_polynomial(pos_t[idx])
        
        rho_abs_t[idx] = rho_t[idx] * config.beta
        
        Temp[idx] = fung_temp_update_etd1(Temp[idx-1], times[idx-1], h, n_t[idx-1], 
                                     T0, config.kappa_T, config.beta_T)

        rho_net_abs[idx] = rho_abs_t[idx] - config.alpha_T_abs_per_K * (Temp[idx] - config.T0)

        lambda_ci = 0
        for ci in np.arange(len(config.beta_groups)):
            lambda_ci += config.lambda_groups[ci] * c_t[idx-1, ci]
        
        Cn = (rho_net_abs[idx] - config.beta) / config.Lambda
        z = h * Cn
        expoN = np.exp(z)
        n_t[idx] = n_t[idx-1]*expoN + h * phi1(z) * lambda_ci
       
        for ci2 in np.arange(len(config.beta_groups)):
            expoC = np.exp(-config.lambda_groups[ci2] * h)
            c_t[idx, ci2] = expoC*c_t[idx-1, ci2] + (1.0 - expoC)/config.lambda_groups[ci2] * \
            ((config.beta_groups[ci2]/config.Lambda) * n_t[idx])
    
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