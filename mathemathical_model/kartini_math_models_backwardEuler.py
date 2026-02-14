import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Rod reactivity model: sin^2 differential worth
# dρ/dt = (dρ/dx)*v
# with dρ/dx = C * sin^2(pi x/H), where C chosen by you
# NOTE: This is your original model (not strictly normalized to ρ(H)=ρ_max)
# ---------------------------
def drho_dt_sin2(x, v, rho_max_dollar, H):
    C = (np.pi * rho_max_dollar) / (2.0 * H)  # your chosen constant
    return C * (np.sin(np.pi * x / H) ** 2) * v  # $/s

# ---------------------------
# 6-group Point Kinetics: Backward Euler step
# Unknowns at k+1: n_{k+1}, c_{i,k+1}
# A y_{k+1} = b
# ---------------------------
def backward_euler_6group_step(n_k, c_k, dt, rho_abs_k1, beta_i, lam, beta_total, Lambda, S_k1=0.0):
    G = len(beta_i)
    A = np.zeros((1 + G, 1 + G), dtype=float)
    b = np.zeros(1 + G, dtype=float)

    # neutron equation:
    # n_{k+1} = n_k + dt*( S + ((rho - beta)/Lambda)*n_{k+1} + sum lam_i c_{i,k+1} )
    A[0, 0] = 1.0 - dt * (rho_abs_k1 - beta_total) / Lambda
    A[0, 1:] = -dt * lam
    b[0] = n_k + dt * S_k1

    # precursor equations:
    # c_{i,k+1} = c_{i,k} + dt*( (beta_i/Lambda)*n_{k+1} - lam_i*c_{i,k+1} )
    for i in range(G):
        A[1+i, 0] = -dt * (beta_i[i] / Lambda)
        A[1+i, 1+i] = 1.0 + dt * lam[i]
        b[1+i] = c_k[i]

    y_next = np.linalg.solve(A, b)
    return float(y_next[0]), y_next[1:]


def init_precursors_steady(n0, beta_i, lam, Lambda):
    # steady-state at rho=0, S=0:
    # c_i0 = (beta_i/(Lambda*lam_i)) * n0
    return (beta_i / (Lambda * lam)) * n0


# ===========================
# MAIN SIMULATION
# ===========================
# --- delayed neutron data (U-235) ---
df_fdn = pd.read_excel("fraction_delayed_neutrons_U235.xlsx")
beta_i = df_fdn["beta"].to_numpy(dtype=float)
lam = df_fdn["lambda"].to_numpy(dtype=float)
beta_total = float(beta_i.sum())

# --- kinetics parameters ---
Lambda = 4.0e-5   # s (TRIGA-ish)
S = 0.0          # external source (0 if not modeling startup source)

# --- rod / worth parameters ---
H = 0.38
rho_max_dollar = 1.95

T0 = 300 #initial temperature (K)
alpha_T_abs_per_K = 6e-5   # reaktivitas absolut per K (mulai dari 5e-5 s/d 2e-4)
a_K_per_s_at_n1 = 0.03     # K/s saat n=1 (pemanasan)
b_1_per_s = 0.01           # 1/s (pendinginan), time constant ~100 s

v_percent = 0.666242        # %/s
v_rod = (v_percent / 100.0) * H  # m/s

# simulate moving rod from 0% to 20%
target_percent = 80.0
x_target = (target_percent / 100.0) * H
t_end = target_percent / v_percent  # seconds to reach target at v_percent (%/s)

dt = 0.01
N = int(np.ceil(t_end / dt)) + 1
t = np.linspace(0.0, t_end, N)

# arrays
x = np.zeros(N)              # m
rho_d = np.zeros(N)          # $ (integrated)
rho_abs = np.zeros(N)        # absolute reactivity
n = np.zeros(N)              # neutron density (relative)
c = np.zeros((N, len(beta_i)))

# initial conditions
x[0] = 0.0
rho_d[0] = 0.0
rho_abs[0] = beta_total * rho_d[0]

T = np.zeros(N)
rho_net_abs = np.zeros(N)

T[0] = T0
rho_net_abs[0] = rho_abs[0]  # karena T=T0 di awal

n0 = 1.0                     # IMPORTANT: don't start from 0
n[0] = n0
c[0, :] = init_precursors_steady(n0, beta_i, lam, Lambda)

for k in range(N-1):
    # 1) update rod position (explicit, known)
    x[k+1] = min(H, x[k] + dt * v_rod)
    if x[k+1] > x_target:
        x[k+1] = x_target

    # 2) update reactivity in dollars (Euler integrate dρ/dt)
    rho_d[k+1] = rho_d[k] + dt * drho_dt_sin2(x[k], v_rod, rho_max_dollar, H)

    # 3) convert $ -> absolute (use beta_total)
    rho_abs[k+1] = beta_total * rho_d[k+1]
    
    dTdt = a_K_per_s_at_n1 * n[k] - b_1_per_s * (T[k] - T0)
    T[k+1] = T[k] + dt * dTdt
    
    rho_net_abs[k+1] = rho_abs[k+1] - alpha_T_abs_per_K * (T[k+1] - T0)

    # 4) implicit Euler kinetics step using rho_abs at k+1
    n[k+1], c[k+1, :] = backward_euler_6group_step(
        n_k=n[k],
        c_k=c[k, :],
        dt=dt,
        rho_abs_k1=rho_net_abs[k+1],
        beta_i=beta_i,
        lam=lam,
        beta_total=beta_total,
        Lambda=Lambda,
        S_k1=S
    )

df_out = pd.DataFrame({
    "time_s" : t,
    "rod_position_m" : x,
    "rod_position_%" : 100 * x / H,
    "rho_dollar" : rho_d,
    "rho_abs" : rho_abs,
    "rho_net_abs" : rho_net_abs,
    "neutron_density_n" : n,
    "temperature_K" : T
    })

df_out.to_excel("hasil_simulasi_kartini_implicitEuler.xlsx", index=False)
# ===========================
# PLOTS
# ===========================
plt.figure()
plt.plot(t, 100*x/H)
plt.xlabel("Time (s)")
plt.ylabel("Rod position (%)")
plt.title("Rod Position vs Time")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, rho_d)
plt.xlabel("Time (s)")
plt.ylabel("Reactivity ($)")
plt.title("Reactivity vs Time (sin^2 worth, integrated)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, n)
plt.xlabel("Time (s)")
plt.ylabel("n(t) (relative neutron density)")
plt.title("Neutron Density vs Time (6-group, Implicit Euler)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, T)
plt.xlabel("Time (s)")
plt.ylabel("Fuel temperature T (K)")
plt.title("Temperature vs Time")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, rho_abs, label="rho_rod_abs")
plt.plot(t, rho_net_abs, label="rho_net_abs (with feedback)")
plt.xlabel("Time (s)")
plt.ylabel("Reactivity (absolute)")
plt.title("Rod vs Net Reactivity")
plt.grid(True)
plt.legend()
plt.show()

print("beta_total =", beta_total)
print("rho_d_end ($) =", rho_d[-1])
print("rho_abs_end =", rho_abs[-1])
print("n_end =", n[-1])
