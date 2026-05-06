import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# --- Load delayed neutron data ---
cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd, file_delayed_neutron), index_col=None)

# --- Constants ---
Lambda = 4.3e-5
beta_groups = df_fdn["beta"].to_numpy()
lambda_groups = df_fdn["lambda"].to_numpy()
beta = np.sum(beta_groups)

H = 0.38
rho_total = 1.95  # ρ_total in $

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

# --- dρ/dx ---
def drho_dx(x, rho, rho_total, H):
    C = (2 * rho_total) / (H)
    return C * np.sin((np.pi * x) / H)**2

# --- Integrate ρ(x) numerically using RK4 ---
def rho_numeric(x_end, rho_total, H, dx=1e-4):
    rho = 0.0
    x = 0.0
    while x < x_end:
        if x + dx > x_end:
            dx = x_end - x
        k1 = drho_dx(x, rho, rho_total, H)
        k2 = drho_dx(x + dx/2, rho + dx/2 * k1, rho_total, H)
        k3 = drho_dx(x + dx/2, rho + dx/2 * k2, rho_total, H)
        k4 = drho_dx(x + dx, rho + dx * k3, rho_total, H)
        rho += dx / 6 * (k1 + 2*k2 + 2*k3 + k4)
        x += dx
    return rho

# --- Absolute reactivity at time t ---
def rho_abs_from_t(t):
    x = rod_position(t)
    rho = rho_numeric(x, rho_total, H)
    return rho * beta

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

# --- Solve PRKE with Radau ---
val = solve_ivp(rhs, (times[0], times[-1]), y0, method='Radau', t_eval=times)

# --- Extract results ---
n_t = val.y[0, :]
c_t = val.y[1:, :].T
pos_t = np.array([rod_position(t) for t in val.t])
rho_t = np.array([rho_numeric(x, rho_total, H) for x in pos_t])
rho_abs_t = rho_t * beta

# --- Output DataFrame ---
df_out_radau_sin = pd.DataFrame({
    "time_s": times,
    "rod_position_m": pos_t,
    "rod_position_%": 100 * pos_t / H,
    "rho_dollar": rho_t,
    "rho_abs": rho_abs_t,
    "neutron_density_n": n_t
})

"""
# --- Plot results ---
plt.figure()
plt.plot(times, rho_t)
plt.xlabel("Time (s)")
plt.ylabel("Reactivity ($)")
plt.title("Reactivity vs Time (RK4 numeric)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(times, n_t)
plt.xlabel("Time (s)")
plt.ylabel("Neutron density n(t)")
plt.title("Neutron Density vs Time")
plt.grid(True)
plt.show()
"""