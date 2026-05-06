import os
import numpy as np
import pandas as pd
 
from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda
from rho_x_functions import drho_dx_sin2
 
# ---------------------------------------------------------------------------
# Delayed neutron data
# ---------------------------------------------------------------------------
cwd      = os.getcwd()
main_cwd = os.path.split(cwd)[0]
df_fdn   = pd.read_excel(os.path.join(main_cwd, "fraction_delayed_neutrons_U235.xlsx"), index_col=None)
 
beta_groups   = df_fdn["beta"].to_numpy(dtype=float)
lambda_groups = df_fdn["lambda"].to_numpy(dtype=float)
beta          = beta_groups.sum()
 
# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
 
def run_simulation_prke(H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda, drho_dx_func):
    """
    Parameters
    ----------
    drho_dx_func : callable f(x, rho, rho_total, H) — spatial worth derivative
    """
    v_rod     = (v_percent / 100.0) * H
    pos_x     = (pos_x_percent / 100.0) * H
    t_end_rod = pos_x / v_rod
    N         = int(np.ceil(t_end / dt)) + 1
    times     = np.arange(0.0, t_end + dt, dt)
 
    # ------------------------------------------------------------------
    # Pre-compute rho(x) on a fine grid ONCE using vectorised RK4
    # ------------------------------------------------------------------
    # Instead of calling the Python while-loop for every time step,
    # we integrate drho/dx across [0, H] once on a fine uniform grid,
    # then use np.interp for fast O(log N) lookup at any x.
 
    dx_fine = 1e-4
    x_grid  = np.arange(0.0, H + dx_fine, dx_fine)  # shape (M,)
    rho_grid = np.zeros(len(x_grid))                  # rho_grid[i] = rho(x_grid[i])
 
    for i in range(1, len(x_grid)):
        xi, ri, h = x_grid[i - 1], rho_grid[i - 1], dx_fine
        k1 = drho_dx_func(xi,         ri,               rho_max, H)
        k2 = drho_dx_func(xi + h/2,   ri + h/2 * k1,   rho_max, H)
        k3 = drho_dx_func(xi + h/2,   ri + h/2 * k2,   rho_max, H)
        k4 = drho_dx_func(xi + h,     ri + h * k3,      rho_max, H)
        rho_grid[i] = ri + h / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
 
    # Vectorised lookup: rho in dollars for any array of positions
    def rho_from_pos(x_arr):
        return np.interp(x_arr, x_grid, rho_grid)
 
    # ------------------------------------------------------------------
    # Rod position and reactivity on the full time grid — all at once
    # ------------------------------------------------------------------
    pos_t     = np.minimum(v_rod * times, pos_x)   # shape (N,)
    rho_t     = rho_from_pos(pos_t)                 # dollars, shape (N,)
    rho_abs_t = rho_t * beta                        # dimensionless, shape (N,)
 
    # ------------------------------------------------------------------
    # ETD-1 time stepping
    # ------------------------------------------------------------------
    def phi1(z):
        """phi1(z) = (e^z - 1) / z, stable near z=0 via Taylor expansion."""
        small = np.abs(z) < 1e-8
        safe_z = np.where(small, 1.0, z)              # avoid 0/0 in np.where
        result = np.where(small,
                          1.0 + z/2.0 + z*z/6.0,
                          (np.exp(safe_z) - 1.0) / safe_z)
        return float(result)
 
    n_t = np.zeros(N)
    c_t = np.zeros((N, len(beta_groups)))
 
    n_t[0]   = 1.0
    c_t[0, :] = (beta_groups / (Lambda * lambda_groups)) * n_t[0]
 
    for i in range(1, N):
        delT = times[i] - times[i - 1]
 
        # --- neutron density (ETD-1) ---
        sum_lambda_ci = float(np.dot(lambda_groups, c_t[i - 1]))
        Cn  = (rho_abs_t[i - 1] - beta) / Lambda
        z   = delT * Cn
        n_t[i] = n_t[i - 1] * np.exp(z) + delT * phi1(z) * sum_lambda_ci
 
        # --- precursors (ETD-1), fully vectorised over all 6 groups ---
        expo_c   = np.exp(-lambda_groups * delT)         # shape (6,)
        c_t[i] = (expo_c * c_t[i - 1]
                  + (1.0 - expo_c) / lambda_groups * (beta_groups / Lambda) * n_t[i - 1])
 
    #precursor_cols = {f"precursor_C{j+1}": c_t[:, j] for j in range(c_t.shape[1])}
 
    return pd.DataFrame({
        "time_s"           : times,
        "rod_position_m"   : pos_t,
        "rod_position_pct" : 100.0 * pos_t / H,
        "rho_dollar"       : rho_t,
        "rho_abs"          : rho_abs_t,
        "neutron_density_n"  : n_t
        })