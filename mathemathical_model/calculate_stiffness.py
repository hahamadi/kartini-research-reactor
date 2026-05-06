import numpy as np
from numpy.linalg import eigvals
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import pandas as pd

df_fdn = pd.read_excel('fraction_delayed_neutrons_U235.xlsx', index_col=None)

# =========================
# PARAMETER REAKTOR Kartini
# =========================

Lambda = 4.0e-5
beta_i = df_fdn["beta"].to_numpy()
beta = np.sum(beta_i)
lam_i = df_fdn["lambda"].to_numpy()

# Parameter termal 
alpha_T = 2.0e-5   # koef. feedback temperatur, sesuaikan tanda sesuai model
kappa   = 0.1      # coupling neutron -> temperatur
gamma_T = 0.05     # laju pendinginan
T_ref   = 300.0
T_cool  = 300.0

H = 0.38 #units (meter)
rho_max = 1.95 # units dollar $

v_percent = 1.49 # units (%/s)
v_rod = (v_percent/100) * H # units m/s


def x_of_t_linear(t, x0=0.0, v=v_rod):
    return x0 + v*t

def rho_polynomial(x_n):
    g = -129.16*(x_n**5) + 279.95*(x_n**4) - 215.04*(x_n**3) + 58.294*(x_n**2) + 1.3702*x_n - 0.0029
    return g

def rho_t_polynomial(t, x0=0.0, v=v_rod):
    x_n = x_of_t_linear(t, x0=x0, v=v)
    return rho_polynomial(x_n)

def prfk_rhs_no_feedback(t, y, rho_func):
    n = y[0]
    C = y[1:]
    
    rho = rho_func(t)
    
    dn_dt = ((rho - beta) / Lambda) * n + np.sum(lam_i * C)
    dC_dt = (beta_i / Lambda) * n - lam_i * C
    
    return np.concatenate(([dn_dt], dC_dt))

def prk_rhs_with_feedback(t, y, rho_func):
    n = y[0]
    C = y[1:-1]
    T = y[-1]
    
    rho_ext = rho_func(t)
    rho = rho_ext - alpha_T * (T - T_ref)
    
    dn_dt = ((rho - beta) / Lambda) * n + np.sum(lam_i * C)
    dC_dt = (beta_i / Lambda) * n - lam_i * C
    dT_dt = kappa * n - gamma_T * (T - T_cool)
    
    return np.concatenate(([dn_dt], dC_dt, [dT_dt]))

def jacobian_no_feedback(rho):
    g = len(beta_i)
    J = np.zeros((g + 1, g + 1))
    
    # baris dn/dt
    J[0, 0] = (rho - beta) / Lambda
    J[0, 1:] = lam_i
    
    # baris dCi/dt
    for i in range(g):
        J[i+1, 0] = beta_i[i] / Lambda
        J[i+1, i+1] = -lam_i[i]
    
    return J

def jacobian_with_feedback(t, y, rho_func):
    n = y[0]
    T = y[-1]
    
    rho_ext = rho_func(t)
    rho = rho_ext - alpha_T * (T - T_ref)
    
    g = len(beta_i)
    J = np.zeros((g + 2, g + 2))
    
    # dn/dt
    J[0, 0] = (rho - beta) / Lambda
    J[0, 1:g+1] = lam_i
    J[0, -1] = -(alpha_T / Lambda) * n
    
    # dCi/dt
    for i in range(g):
        J[i+1, 0] = beta_i[i] / Lambda
        J[i+1, i+1] = -lam_i[i]
    
    # dT/dt
    J[-1, 0] = kappa
    J[-1, -1] = -gamma_T
    
    return J

def timescale_analysis(rho=0.0, include_temperature=False):
    # skala waktu prompt
    tau_prompt = Lambda / max(abs(beta - rho), 1e-16)
    
    # skala waktu precursor
    tau_delayed = 1.0 / lam_i
    
    scales = [tau_prompt] + list(tau_delayed)
    
    if include_temperature:
        tau_T = 1.0 / gamma_T
        scales.append(tau_T)
    
    scales = np.array(scales)
    
    tau_min = np.min(scales)
    tau_max = np.max(scales)
    R_tau = tau_max / tau_min
    
    return {
        "scales": scales,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "R_tau": R_tau
    }

def stiffness_from_jacobian(J, tol=1e-12):
    eigs = eigvals(J)
    reals = np.abs(np.real(eigs))
    
    # buang mode yang hampir nol agar rasio tidak rusak
    reals_nonzero = reals[reals > tol]
    
    if len(reals_nonzero) == 0:
        R_lambda = np.nan
        lam_max = np.nan
        lam_min = np.nan
    else:
        lam_max = np.max(reals_nonzero)
        lam_min = np.min(reals_nonzero)
        R_lambda = lam_max / lam_min
    
    return {
        "eigenvalues": eigs,
        "lam_max": lam_max,
        "lam_min": lam_min,
        "R_lambda": R_lambda
    }

def explicit_step_limits(J, tol=1e-12):
    eigs = eigvals(J)
    reals = np.real(eigs)
    
    neg_reals = np.abs(reals[reals < -tol])
    
    if len(neg_reals) == 0:
        return {
            "h_euler_max": np.nan,
            "h_rk4_max": np.nan
        }
    
    mu_max = np.max(neg_reals)
    
    # Euler: |1 + h*mu| < 1  => h < 2/|mu|
    h_euler = 2.0 / mu_max
    
    # RK4 kira-kira pada sumbu real negatif ~ 2.8
    h_rk4 = 2.8 / mu_max
    
    return {
        "h_euler_max": h_euler,
        "h_rk4_max": h_rk4
    }

def frozen_stiffness_analysis(time_grid, rho_func):
    R_lambda_list = []
    h_euler_list = []
    h_rk4_list = []
    
    for t in time_grid:
        rho = rho_func(t)
        J = jacobian_no_feedback(rho)
        
        eig_info = stiffness_from_jacobian(J)
        step_info = explicit_step_limits(J)
        
        R_lambda_list.append(eig_info["R_lambda"])
        h_euler_list.append(step_info["h_euler_max"])
        h_rk4_list.append(step_info["h_rk4_max"])
    
    return {
        "t": np.array(time_grid),
        "R_lambda": np.array(R_lambda_list),
        "h_euler_max": np.array(h_euler_list),
        "h_rk4_max": np.array(h_rk4_list)
    }

rho0 = 0.0

# Analisis skala waktu
ts_info = timescale_analysis(rho=rho0, include_temperature=False)

print("=== Analisis Skala Waktu ===")
print("Semua skala waktu:", ts_info["scales"])
print("tau_min =", ts_info["tau_min"])
print("tau_max =", ts_info["tau_max"])
print("R_tau   =", ts_info["R_tau"])

# Jacobian
J = jacobian_no_feedback(rho0)

# Analisis eigenvalue
eig_info = stiffness_from_jacobian(J)

print("\n=== Analisis Eigenvalue ===")
print("Eigenvalues =", eig_info["eigenvalues"])
print("lam_max     =", eig_info["lam_max"])
print("lam_min     =", eig_info["lam_min"])
print("R_lambda    =", eig_info["R_lambda"])

step_info = explicit_step_limits(J)
print("\n=== Estimasi Batas Langkah Eksplisit ===")
print("h_euler_max =", step_info["h_euler_max"])
print("h_rk4_max   =", step_info["h_rk4_max"])

t_grid = np.linspace(0, 5, 300)
result = frozen_stiffness_analysis(t_grid, lambda t: rho_t_polynomial(t, x0=0.0, v=v_rod))

plt.figure(figsize=(8,5))
plt.plot(result["t"], result["R_lambda"])
plt.xlabel("t")
plt.ylabel("R_lambda")
plt.title("Rasio Stiffness Lokal (Frozen Jacobian)")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(result["t"], result["h_euler_max"], label="Euler")
plt.plot(result["t"], result["h_rk4_max"], label="RK4")
plt.xlabel("t")
plt.ylabel("Batas langkah stabil")
plt.title("Estimasi h maksimum stabil")
plt.legend()
plt.grid(True)
plt.show()