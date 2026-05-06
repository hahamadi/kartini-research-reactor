import numpy as np
import pandas as pd

import simulation_config as config
from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, dt
from simulation_config import Lambda, T0, kappa_T, beta_T, alpha_T_abs_per_K
import rho_x_functions as rho_func

from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

def rod_position(t, v_rod, pos_x):
        x = v_rod * t
        return min(x, pos_x)

def rod_velocity(t, v_rod, t_end_rod):
    return v_rod if t < t_end_rod else 0.0

def drho_dt(t, rho, drho_func, rho_max, H, v_rod, pos_x, t_end_rod):
    x = rod_position(t, v_rod, pos_x)
    v = rod_velocity(t, v_rod, t_end_rod)
    return drho_func(x, rho, rho_max, H) * v

def pke_rhs(t, y, rho_abs_interp):
    T, n, c    = y[0], y[1], y[2:]

    dT = -beta_T*(T - config.T0) + kappa_T * n

    rho_abs = rho_abs_interp(t) - alpha_T_abs_per_K * (T - T0)
    
    dn_dt   = ((rho_abs - config.beta) / Lambda) * n + np.dot(config.lambda_groups, c)
    dc_dt   = (config.beta_groups / Lambda) * n - config.lambda_groups * c
    return np.concatenate(([dT], [dn_dt], dc_dt))

def prke_solver_temp_var(H, rho_max, v_percent, pos_x_percent, t_end, dt,
                         T0,  drho_func):
    v_rod     = (v_percent / 100.0) * H
    pos_x     = (pos_x_percent / 100.0) * H
    t_end_rod = pos_x / v_rod

    times = np.arange(0, t_end, dt)
    N = len(times)

    rho_dollar = np.zeros(N)
    rho_dollar[0] = 0.0

    for i in range(N-1):
        ti = times[i]
        hi = times[i+1] - times[i]
        ri = rho_dollar[i]

        k1 = drho_dt(ti, ri, drho_func, rho_max, H, v_rod, pos_x, t_end_rod)
        k2 = drho_dt(ti + hi/2, ri + hi/2 * k1, drho_func, rho_max, H, v_rod, pos_x, t_end_rod)
        k3 = drho_dt(ti + hi/2, ri + hi/2 * k2, drho_func, rho_max, H, v_rod, pos_x, t_end_rod)
        k4 = drho_dt(ti + hi, ri + hi * k3, drho_func, rho_max, H, v_rod, pos_x, t_end_rod)
        rho_dollar[i+1] = ri + (hi/6) * (k1 + 2*k2 + 2*k3 + k4)

    rho_abs_t = rho_dollar * config.beta
    
    rho_net_abs = np.zeros_like(times)

    rho_abs_interp = CubicSpline(times, rho_abs_t)

    n0 = 1.0
    c0 = (config.beta_groups / (Lambda * config.lambda_groups)) * n0
    y0 = np.concatenate(([T0], [n0], c0))
    sol = solve_ivp(
        pke_rhs,
        t_span = (times[0], times[-1]),
        y0 = y0,
        args=(rho_abs_interp,),
        method="BDF",
        t_eval=times,
        rtol=1e-8,
        atol=1e-10
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    
    n_t = sol.y[1,:]
    Temp = sol.y[0,:]

    pos_t = np.array([rod_position(t, v_rod, pos_x) for t in times])

    rho_net_abs = rho_abs_t - alpha_T_abs_per_K * (Temp - T0)
    
    df = pd.DataFrame({
        "time_s" : times,
        "rod_position_m" : pos_t,
        "rod_position_%" : 100 * pos_t / H,
        "rho_dollar" : rho_dollar,
        "rho_abs" : rho_abs_t,
        "rho_abs_all" : rho_net_abs,
        "temperature_K" : Temp,
        "neutron_density_n" : n_t
        })
    return df