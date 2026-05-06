import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import os

# --- Data delayed neutrons ---
cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd, file_delayed_neutron), index_col=None)

Lambda = 4.3e-5
beta_groups = df_fdn["beta"].to_numpy()
lambda_groups = df_fdn["lambda"].to_numpy()
beta = np.sum(beta_groups)

H = 0.38
rho_total = 1.95  # ρ_total

v_percent = 1.6579
v_rod = (v_percent / 100) * H

pos_x_percent = 35
pos_x = (pos_x_percent / 100) * H

t_end = pos_x_percent / v_percent
dt = 0.01
times = np.arange(0, t_end + dt, dt)

# --- Rod position ---
def rod_position(t):
    return min(v_rod * t, pos_x)

# --- Compute rho from dρ/dx ---
def rho_from_dx(x, rho_total, H):
    """
    Integrate dρ/dx = (π * rho_total)/(2H) * sin^2(π x/H)
    Analytic integral:
      ∫ sin^2(π s/H) ds = s/2 - (H/(4π)) * sin(2π s/H)
    """
    return (rho_total / 2) * (x - H/(2*np.pi) * np.sin(2*np.pi*x/H))

def rho_abs_from_t(t):
    x = rod_position(t)
    return rho_from_dx(x, rho_total, H) * beta

# --- PRKE RHS ---
def rhs(t, y):
    n = y[0]
    c = y[1:]
    
    rho_abs = rho_abs_from_t(t)
    
    dn = ((rho_abs - beta) / Lambda) * n + np.sum(lambda_groups * c)
    dc = (beta_groups / Lambda) * n - lambda_groups * c
    
    return np.concatenate(([dn], dc))

# --- Initial conditions ---
n0 = 1.0
c0 = beta_groups / (Lambda * lambda_groups) * n0
y0 = np.concatenate(([n0], c0))

# --- Solve PRKE ---
val = solve_ivp(rhs, (times[0], times[-1]), y0, method='Radau', t_eval=times)

# --- Extract results ---
n_t = val.y[0, :]
c_t = val.y[1:, :].T
pos_t = np.array([rod_position(t) for t in val.t])
rho_t = np.array([rho_from_dx(x, rho_total, H) for x in pos_t])
rho_abs_t = rho_t * beta

# --- Output DataFrame ---
df_out_sin_radau2 = pd.DataFrame({
    "time_s": times,
    "rod_position_m": pos_t,
    "rod_position_%": 100 * pos_t / H,
    "rho_dollar": rho_t,
    "rho_abs": rho_abs_t,
    "neutron_density_n": n_t
})