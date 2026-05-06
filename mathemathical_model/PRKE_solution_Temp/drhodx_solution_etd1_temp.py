import numpy as np
import pandas as pd

from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda
from simulation_config import T0, alpha_T_abs_per_K, beta_T, kappa_T
import simulation_config as config

def fung_temp_update_etd1(T_n, t_n, dt, n_t, T0, kappa_T, beta_T):
    
    ch = np.exp(-beta_T*dt)
    forcing = kappa_T * n_t
    yn =  T0 + (T_n - T0) * ch + (forcing / beta_T) * (1.0 - ch)
       
    return yn

def rho_numeric(rho_total, H, drho_dx, dx=1e-4):
    x0 = 0.0
    x_grid = np.arange(x0, H + dx, dx)

    rho_x = np.zeros_like(x_grid)
    for i in range(1, len(x_grid)):
        xi, ri, h = x_grid[i - 1], rho_x[i - 1], dx
        k1 = drho_dx(xi, ri, rho_total, H)
        k2 = drho_dx(xi + h / 2, ri + h / 2 * k1, rho_total, H)
        k3 = drho_dx(xi + h / 2, ri + h / 2 * k2, rho_total, H)
        k4 = drho_dx(xi + h, ri + h * k3, rho_total, H)
        rho_x[i] = ri + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_grid, rho_x

def phi1(z):
    if abs(z) < 1e-8:
        return 1.0 + z/2.0 + z*z/6.0
    return (np.exp(z) - 1.0) / z

def prke_solver_temp_var(H, rho_max, v_percent, pos_x_percent, t_end, dt,
                         T0,  drho_func):
    
    v_rod     = (v_percent / 100.0) * H
    pos_x     = (pos_x_percent / 100.0) * H
    t_end_rod = pos_x / v_rod               # time when rod reaches pos_x
    
    times     = np.arange(0.0, t_end, dt)
    N = len(times)

    x_grid, rho_x = rho_numeric(rho_max, H, drho_dx=drho_func, dx=1e-4)

    pos_t = np.minimum(v_rod * times, pos_x)
    rho_t = np.interp(pos_t, x_grid, rho_x)
    rho_abs_t = rho_t * config.beta

    Temp = np.zeros_like(times)
    rho_net_abs = np.zeros_like(times)
    
    Temp[0] = T0
    rho_net_abs[0] = rho_abs_t[0] - alpha_T_abs_per_K * (Temp[0] - T0)

    n_t = np.zeros(N)
    c_t = np.zeros((N, len(config.beta_groups)))
 
    n_t[0]   = 1.0
    c_t[0, :] = (config.beta_groups / (config.Lambda * config.lambda_groups)) * n_t[0]
    
    for i in range(1,N):
        delT = times[i] - times[i-1]

        Temp[i] = fung_temp_update_etd1(Temp[i-1], times[i-1], delT, n_t[i-1], 
                                     T0, kappa_T, beta_T)
        rho_net_abs[i] = rho_abs_t[i] - alpha_T_abs_per_K * (Temp[i] - T0)
        
        sum_lambda_ci = float(np.dot(c_t[i-1, :], config.lambda_groups))
        Cn = (rho_net_abs[i] - config.beta) / config.Lambda
        z = delT * Cn

        n_t[i] = n_t[i-1]*np.exp(z) + delT * phi1(z) * sum_lambda_ci
        expo_c   = np.exp(-config.lambda_groups * delT)  
        c_t[i, :] = expo_c * c_t[i-1, :] + (1.0 - expo_c) / config.lambda_groups * \
            ((config.beta_groups / config.Lambda) * n_t[i])
    
    df = pd.DataFrame({
        "time_s" : times,
        "rod_position_m" : pos_t,
        "rod_position_%" : 100 * pos_t / H,
        "rho_dollar" : rho_t,
        "rho_abs" : rho_abs_t,
        "rho_abs_all" : rho_net_abs,
        "temperature_K" : Temp,
        "neutron_density_n" : n_t
        })
    return df