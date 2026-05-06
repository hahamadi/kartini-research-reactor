import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import simulation_config as config
import rho_x_functions as rho_func

def rho_net(t, Temp):
    pos = rod_position(t)
    rho_dollar = rho_func.rho_polynomial(pos)
    rho_abs = rho_dollar * config.beta
    rho = rho_abs - config.alpha_T_abs_per_K * (Temp - config.T0)
    return rho

def rod_position(t):
    if t >= config.t_rod_end:
        return config.pos_x
    return min(config.H, config.v_rod * t)

def prke_system(t, y):
    n   = y[0]
    T   = y[1]
    c   = y[2: 2 + len(config.beta_groups)]

    rho = rho_net(t, T)

    sum_lambda_ci = np.dot(config.lambda_groups, c)
    dn = ((rho - config.beta) / config.Lambda) * n + sum_lambda_ci

    dT = config.kappa_T * n - config.beta_T * (T - config.T0)

    dc = (config.beta_groups / config.Lambda) * n - config.lambda_groups * c

    return [dn, dT, *dc]

def prke_solver_temp():
    n0  = 1.0
    T0  = config.T0
    c0  = (config.beta_groups / (config.Lambda * config.lambda_groups)) * n0

    y0 = [n0, T0, *c0]

    # --- Solve ---
    t_span = (config.times[0], config.times[-1])
    t_eval = config.times

    sol = solve_ivp(
        prke_system,
        t_span,
        y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        dense_output=False
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    t_out   = sol.t
    n_out   = sol.y[0]
    T_out   = sol.y[1]
    c_out   = sol.y[2: 2 + len(config.beta_groups)]

    pos_out      = np.array([rod_position(t) for t in t_out])
    rho_dollar   = np.array([rho_func.rho_polynomial(rod_position(t)) for t in t_out])
    rho_abs_out  = rho_dollar * config.beta
    rho_net_out  = np.array([rho_net(t, T) for t, T in zip(t_out, T_out)])

    df = pd.DataFrame({
        "time_s" : config.times,
        "rod_position_m" : pos_out,
        "rod_position_%" : 100 * pos_out / config.H,
        "rho_dollar" : rho_dollar,
        "rho_abs" : rho_abs_out,
        "rho_abs_all" : rho_net_out,
        "temperature_K" : T_out,
        "neutron_density_n" : n_out
        })
    return df