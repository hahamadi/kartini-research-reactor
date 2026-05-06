import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import os

cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd, file_delayed_neutron), index_col=None)

Lambda = 4.3e-5
beta_groups = df_fdn["beta"].to_numpy()
lambda_groups = df_fdn["lambda"].to_numpy()
beta = np.sum(beta_groups)

H = 0.38
rho_max = 1.95

v_percent = 1.6579
v_rod = (v_percent / 100.0) * H

pos_x_percent = 35
pos_x = (pos_x_percent / 100.0) * H

t_end = pos_x_percent / v_percent
dt = 0.009

N = int(np.ceil(t_end / dt)) + 1
times = np.arange(0, t_end + dt, dt)

def rod_position(t):
    return min(v_rod * t, pos_x)

def rho_polynomial(x):
    rho = -129.16*x**5 + 279.95*x**4 - 215.04*x**3 + 58.294*x**2 + 1.3702*x
    return min(rho, rho_max)

def rho_abs_from_t(t):
    return rho_polynomial(rod_position(t)) * beta

def rhs(t, y):
    n = y[0]
    c = y[1:]

    rho_abs = rho_abs_from_t(t)

    dn = ((rho_abs - beta) / Lambda) * n + np.sum(lambda_groups * c)
    dc = (beta_groups / Lambda) * n - lambda_groups * c

    return np.concatenate(([dn], dc))

def jacobian(t):
    rho_abs = rho_abs_from_t(t)
    G = len(beta_groups)
    
    J = np.zeros((G+1, G+1))
    
    J[0, 0]  =  (rho_abs - beta) / Lambda
    J[0, 1:] = lambda_groups
    
    J[1:, 0] = beta_groups / Lambda
    J[1: , 1:] = np.diag(-lambda_groups)
    return J

def stiffness_ratio(t, tol = 1e-12):
    J = jacobian(t)
    eigs = np.linalg.eigvals(J)
    
    real_parts = np.abs(np.real(eigs))
    real_parts = real_parts[real_parts > tol]
    
    if len(real_parts) < 2:
        return np.inf, eigs
    
    S = np.max(real_parts)/np.min(real_parts)
    return S, eigs
# kondisi awal
n0 = 1.0
c0 = beta_groups / (Lambda * lambda_groups) * n0
y0 = np.concatenate(([n0], c0))

val = solve_ivp(fun = rhs, t_span=(times[0],times[-1]), y0 = y0, method = "LSODA", t_eval=times)

#print("success =", val.success)
#print("message =", val.message)

stiffness_t = []
eigvals_t = []

for t in val.t:
    S, eigs = stiffness_ratio(t)
    stiffness_t.append(S)
    eigvals_t.append(eigs)

stiffness_t = np.array(stiffness_t)

print("Stiffness ratio min =", np.min(stiffness_t))
print("Stiffness ratio max =", np.max(stiffness_t))
print("Stiffness ratio mean =", np.mean(stiffness_t))

idx_max = np.argmax(stiffness_t)
print("Waktu stiffness maksimum =", val.t[idx_max], "s")
print("Nilai stiffness maksimum =", stiffness_t[idx_max])

plt.figure(figsize=(8,5))
plt.semilogy(val.t, stiffness_t, lw=2)
plt.xlabel("Time (s)")
plt.ylabel("Stiffness ratio S(t)")
plt.title("Local stiffness ratio of PRKE system")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.show()