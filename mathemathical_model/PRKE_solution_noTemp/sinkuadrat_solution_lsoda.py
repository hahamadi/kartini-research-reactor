import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda
from rho_x_functions import drho_dx_sin2

# =========================
# Baca data delayed neutron
# =========================
cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd, file_delayed_neutron), index_col=None)

beta_group = df_fdn["beta"].to_numpy()
lambda_group = df_fdn["lambda"].to_numpy()

beta = np.sum(beta_group)
m = len(beta_group)

# =========================
# Parameter gerak batang
# =========================
v_rod = (v_percent / 100.0) * H
pos_x = (pos_x_percent / 100.0) * H
t_end_rod = pos_x_percent / v_percent

# grid waktu untuk integrasi rho(t) dengan RK4
N = int(np.ceil(t_end / dt)) + 1
times = np.linspace(0.0, t_end, N)

def rod_position(t):
    x = v_rod * t
    return min(x, pos_x)

def rod_velocity(t):
    return v_rod if t < t_end_rod else 0.0

# =========================
# STEP 1: hitung rho(t) dengan RK4
# =========================
def drho_dt(t, rho):
    x = rod_position(t)
    v = rod_velocity(t)
    return drho_dx_sin2(x, rho, rho_max, H) * v

def rho_abs_t_rk4():
    rho_t = np.zeros(N)
    rho_t[0] = 0.0

    for i in range(N - 1):
        t = times[i]
        h = times[i + 1] - times[i]
        rho_i = rho_t[i]

        k1 = drho_dt(t, rho_i)
        k2 = drho_dt(t + 0.5*h, rho_i + 0.5*h*k1)
        k3 = drho_dt(t + 0.5*h, rho_i + 0.5*h*k2)
        k4 = drho_dt(t + h, rho_i + h*k3)

        rho_t[i + 1] = rho_i + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    rho_abs_t = rho_t * beta
    return rho_abs_t, rho_t
rho_abs_t, rho_t = rho_abs_t_rk4()
pos_t = np.array([rod_position(t) for t in times])

# interpolasi rho_abs(t) agar bisa dipakai LSODA
rho_abs_interp = interp1d(
    times, rho_abs_t,
    kind="linear",
    bounds_error=False,
    fill_value=(rho_abs_t[0], rho_abs_t[-1])
)

# =========================
# STEP 2: PRKE dengan LSODA
# state y = [n, c1, c2, ..., cm]
# =========================
def rhs_prke(t, y):
    n = y[0]
    c = y[1:]

    rho_abs = float(rho_abs_interp(t))

    dn = ((rho_abs - beta) / Lambda) * n + np.sum(lambda_group * c)
    dc = (beta_group / Lambda) * n - lambda_group * c

    return np.concatenate(([dn], dc))

def jac_prke(t, y):
    rho_abs = float(rho_abs_interp(t))

    J = np.zeros((m + 1, m + 1))
    J[0, 0] = (rho_abs - beta) / Lambda
    J[0, 1:] = lambda_group
    J[1:, 0] = beta_group / Lambda
    J[1:, 1:] = -np.diag(lambda_group)

    return J

# kondisi awal
n0 = 1.0
c0 = (beta_group / (Lambda * lambda_group)) * n0
y0 = np.concatenate(([n0], c0))

sol = solve_ivp(
    fun=rhs_prke,
    t_span=(times[0], times[-1]),
    y0=y0,
    method="LSODA",
    jac=jac_prke,
    t_eval=times,
    rtol=1e-8,
    atol=1e-10
)

if not sol.success:
    raise RuntimeError(f"LSODA gagal: {sol.message}")

n_t = sol.y[0]
c_t = sol.y[1:].T

# =========================
# output
# =========================
df_out_lsoda = pd.DataFrame({
    "time_s": sol.t,
    "rod_position_m": pos_t,
    "rod_position_%": 100.0 * pos_t / H,
    "rho_dollar": rho_t,
    "rho_abs": rho_abs_t,
    "neutron_density_n": n_t
})

#print(df_out_lsoda.head())
#print(sol.message)