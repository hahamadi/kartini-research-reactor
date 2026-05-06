import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import polynom_solution_lsoda as psl
import sinkuadrat_solution_lsoda as ssl
import gaussbell_solution_lsoda as gsl
import sigmoid_solution_lsoda as sigsl


# =========================================================
# Ambil parameter neutron delayed dari salah satu modul
# =========================================================
Lambda = psl.Lambda
pos_x_percent = psl.pos_x_percent
beta_groups = psl.df_fdn["beta"].to_numpy()
lambda_groups = psl.df_fdn["lambda"].to_numpy()

dt_sim = psl.dt

# =========================================================
# Jacobian PRKE
# y = [n, c1, c2, ..., cm]
# =========================================================
def jac_prke(rho_abs, beta_groups, lambda_groups, Lambda):
    beta = np.sum(beta_groups)
    m = len(beta_groups)

    J = np.zeros((m + 1, m + 1))
    J[0, 0] = (rho_abs - beta) / Lambda
    J[0, 1:] = lambda_groups
    J[1:, 0] = beta_groups / Lambda
    J[1:, 1:] = -np.diag(lambda_groups)
    return J


# =========================================================
# Hitung stiffness ratio dan dt stabil
# =========================================================
def stiffness_metrics(sol, rho_abs_series, beta_groups, lambda_groups, Lambda, eps=1e-12):
    """
    Menghitung:
      S(t)     = stiffness ratio
      hE(t)    = batas dt stabil Euler eksplisit
      hRK4(t)  = batas dt stabil RK4
    berdasarkan eigenvalue Jacobian PRKE.

    Yang dipakai hanya eigenvalue dengan Re(lambda) < -eps
    agar fokus pada mode stabil.
    """
    S = []
    hE = []
    hRK4 = []
    lam_fast_all = []
    lam_slow_all = []

    for tk, yk, rho_abs_k in zip(sol.t, sol.y.T, rho_abs_series):
        J = jac_prke(rho_abs_k, beta_groups, lambda_groups, Lambda)
        eigvals = np.linalg.eigvals(J)

        stable_reals = np.real(eigvals[np.real(eigvals) < -eps])
        stable_abs = np.abs(stable_reals)

        if len(stable_abs) > 0:
            lam_fast = np.max(stable_abs)
            lam_slow = np.min(stable_abs)

            S.append(lam_fast / lam_slow)
            hE.append(2.0 / lam_fast)
            hRK4.append(2.785 / lam_fast)
            lam_fast_all.append(lam_fast)
            lam_slow_all.append(lam_slow)
        else:
            S.append(np.nan)
            hE.append(np.nan)
            hRK4.append(np.nan)
            lam_fast_all.append(np.nan)
            lam_slow_all.append(np.nan)

    return {
        "time": np.array(sol.t),
        "S": np.array(S),
        "hE": np.array(hE),
        "hRK4": np.array(hRK4),
        "lam_fast": np.array(lam_fast_all),
        "lam_slow": np.array(lam_slow_all),
    }


# =========================================================
# Utility untuk ambil object solver dari tiap modul
# polynomial -> val
# lainnya    -> sol
# =========================================================
def get_solution_object(module_obj):
    if hasattr(module_obj, "sol"):
        return module_obj.sol
    if hasattr(module_obj, "val"):
        return module_obj.val
    raise AttributeError("Modul tidak memiliki atribut 'sol' atau 'val'.")


# =========================================================
# Kumpulkan semua data
# =========================================================
cases = {
    "polynomial": psl,
    "sin2": ssl,
    "gaussian": gsl,
    "sigmoid": sigsl,
}

results = {}

for name, mod in cases.items():
    sol_obj = get_solution_object(mod)
    rho_abs_series = mod.df_out_lsoda["rho_abs"].to_numpy()

    metrics = stiffness_metrics(
        sol_obj,
        rho_abs_series,
        beta_groups,
        lambda_groups,
        Lambda
    )

    results[name] = {
        "module": mod,
        "sol": sol_obj,
        "metrics": metrics,
    }


# =========================================================
# Ringkasan numerik
# =========================================================
print("=" * 72)
print("RINGKASAN STIFFNESS DAN BATAS DT STABIL")
print("=" * 72)

#df = pd.DataFrame(columns=[""])
dfres = pd.DataFrame(columns=["name","S(t)_min", "S(t)_max", "h_E_min", \
                              "h_E_max", "h_RK4_min", "h_RK4_max"])
idx = 0
for name, data in results.items():
    S = data["metrics"]["S"]
    hE = data["metrics"]["hE"]
    hRK4 = data["metrics"]["hRK4"]
    dfres.loc[idx] = [name, np.nanmin(S), np.nanmax(S), np.nanmin(hE),
                      np.nanmax(hE), np.nanmin(hRK4), np.nanmax(hRK4)]

    print(f"\nKasus: {name}")
    print(f"  S(t) min   = {np.nanmin(S):.6e}")
    print(f"  S(t) max   = {np.nanmax(S):.6e}")
    print(f"  h_E min    = {np.nanmin(hE):.6e} s")
    print(f"  h_E max    = {np.nanmax(hE):.6e} s")
    print(f"  h_RK4 min  = {np.nanmin(hRK4):.6e} s")
    print(f"  h_RK4 max  = {np.nanmax(hRK4):.6e} s")
    idx += 1
dfres.to_excel(f"output_stiffness_ratio_LSDOA_pos{pos_x_percent}.xlsx", index=False)

# =========================================================
# Plot 1: neutron density n(t)
# =========================================================
plt.figure(figsize=(9, 6))
for name, data in results.items():
    mod = data["module"]
    plt.plot(
        mod.df_out_lsoda["time_s"].values,
        mod.df_out_lsoda["neutron_density_n"].values,
        marker='o',
        fillstyle='none',
        markevery=max(1, len(mod.df_out_lsoda) // 40),
        label=name
    )
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("n(t)")
plt.title("Number of neutrons vs Time")
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================================================
# Plot 2: log neutron density
# =========================================================
markers = ["o","d","s","v"]
idx = 0
plt.figure(figsize=(9, 6))
for name, data in results.items():
    mod = data["module"]
    nvals = mod.df_out_lsoda["neutron_density_n"].values

    # hindari log nilai <= 0
    nlog = np.where(nvals > 0.0, np.log(nvals), np.nan)

    plt.plot(
        mod.df_out_lsoda["time_s"].values,
        nlog,
        marker=markers[idx],
        fillstyle='none',
        markevery=max(1, len(mod.df_out_lsoda) // 40),
        label=name
    )
    idx += 1
    
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("log(n(t))", fontsize=12)
plt.title("Log neutron density vs Time")
plt.savefig("grafik_LogNeutron_time_pos{pos_x_percent}.svg", dpi=300, format='svg', bbox_inches='tight')
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================================================
# Plot 3: stiffness ratio S(t)
# =========================================================
plt.figure(figsize=(9, 6))
for name, data in results.items():
    t = data["metrics"]["time"]
    S = data["metrics"]["S"]
    plt.plot(t, S, label=name)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("S(t)")
plt.title("Stiffness ratio vs Time")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,6))
for name, data in results.items():
    t = data["metrics"]["time"]
    S = data["metrics"]["S"]
    plt.plot(t, S, label=name)

plt.yscale('log')  # <-- ini kuncinya
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("S(t) (log scale)")
plt.title("Stiffness ratio vs Time (log scale)")
plt.grid(True, which="both")
plt.show()

plt.figure(figsize=(9,6))

for name, data in results.items():
    t = data["metrics"]["time"]
    lam_fast = data["metrics"]["lam_fast"]
    lam_slow = data["metrics"]["lam_slow"]

    plt.plot(t, lam_fast, '--', label=f"{name} fast")
    plt.plot(t, lam_slow, '-', label=f"{name} slow")

plt.yscale('log')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("|lambda|")
plt.title("Eigenvalue spectrum (fast vs slow modes)")
plt.grid(True, which="both")
plt.show()

# =========================================================
# Plot 4: dt stabil Euler
# =========================================================
dt_sim = psl.dt

plt.figure(figsize=(9, 6))
for name, data in results.items():
    t = data["metrics"]["time"]
    hE = data["metrics"]["hE"]
    plt.plot(t, hE, label=name)

plt.axhline(dt_sim, linestyle='--', linewidth=2, label=f'dt simulasi = {dt_sim:.4f} s')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("h_E(t) [s]")
plt.title("Stable time step limit for Explicit Euler")
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================================================
# Plot 5: dt stabil RK4
# =========================================================
plt.figure(figsize=(9, 6))
for name, data in results.items():
    t = data["metrics"]["time"]
    hRK4 = data["metrics"]["hRK4"]
    plt.plot(t, hRK4, label=name)

plt.axhline(dt_sim, linestyle='--', linewidth=2, label=f'dt simulasi = {dt_sim:.4f} s')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("h_RK4(t) [s]")
plt.title("Stable time step limit for RK4")
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================================================
# Plot 6: rho(t) untuk semua kasus
# =========================================================
plt.figure(figsize=(9, 6))
for name, data in results.items():
    mod = data["module"]
    plt.plot(
        mod.df_out_lsoda["time_s"].values,
        mod.df_out_lsoda["rho_dollar"].values,
        label=name
    )
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("rho(t) [$]")
plt.title("Reactivity vs Time")
plt.grid(True)
plt.tight_layout()
plt.show()