import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import simulation_config as config
from polynomial_solution_bdf_temp import prke_system
from drhodx_solution_bdf_temp import pke_rhs, prke_solver_temp_var
import rho_x_functions as rho_func
from scipy.interpolate import CubicSpline

import matplotlib.ticker as ticker

def compute_jacobian(func, t, y, eps=1e-7):

    f0 = np.array(func(t, y))
    n = len(y)
    jacobian = np.zeros((n, n))

    for i in range(n):
        y_perturbed = np.copy(y)
        y_perturbed[i] += eps
        f_perturbed = np.array(func(t, y_perturbed))
        jacobian[:, i] = (f_perturbed - f0) / eps
    return jacobian

def compute_stiffness(func, t_points, y_points, verbose=True):
    
    stiffness_values = []
    for i, t in enumerate(t_points):
        y = y_points[:, i]
        J = compute_jacobian(func, t, y)
        eigvals = np.linalg.eigvals(J)
        re_parts = np.real(eigvals)

        neg_mask = re_parts < 0
        if not np.any(neg_mask):
            stiffness = 0.0
            print(f"[t={t:.2f}s] No stable eigenvalues found, skipping.")
            continue
        neg_eigs = np.abs(re_parts[neg_mask])
        lambda_max = np.max(neg_eigs)
        lambda_min = np.min(neg_eigs)

        stiffness = lambda_max / lambda_min if lambda_min > 0 else np.inf
        hEul = 1.0 / lambda_max
        hRK = 2.0 / lambda_max
        stiffness_values.append({
            "t"               : t,
            "lambda_max"      : lambda_max,
            "lambda_min"      : lambda_min,
            "stiffness_ratio" : stiffness,
            "hEul"            : hEul,
            "hRK"             : hRK
        })

    if verbose and stiffness_values:
        header = (f"\n{'t(s)':>8} | {'|λ_max|':>12} | {'|λ_min|':>12} | "
                  f"{'Stiffness Ratio':>16} | {'hEul(s)':>12} | {'hRK(s)':>12}")
        print(header)
        print("-" * 90)
        for r in stiffness_values:
            print(f"{r['t']:>8.2f} | {r['lambda_max']:>12.4e} | {r['lambda_min']:>12.4e} | "
                  f"{r['stiffness_ratio']:>16.4e} | {r['hEul']:>12.4e} | {r['hRK']:>12.4e}")

    return stiffness_values

def run_stiffness_drhodx(drho_func, label="drhodx"):

    v_rod     = (config.v_percent / 100.0) * config.H
    pos_x     = (config.pos_x_percent / 100.0) * config.H
    t_end_rod = pos_x / v_rod
    times     = np.arange(0, config.t_end, config.dt)
    N         = len(times)

    def rod_position_local(t):
        return min(v_rod * t, pos_x)

    def rod_velocity_local(t):
        return v_rod if t < t_end_rod else 0.0

    def drho_dt_local(t, rho_val):
        x = rod_position_local(t)
        v = rod_velocity_local(t)
        return drho_func(x, rho_val, config.rho_max, config.H) * v

    
    rho_dollar = np.zeros(N)
    for i in range(N - 1):
        ti = times[i]
        hi = times[i+1] - times[i]
        ri = rho_dollar[i]
        k1 = drho_dt_local(ti,        ri)
        k2 = drho_dt_local(ti + hi/2, ri + hi/2 * k1)
        k3 = drho_dt_local(ti + hi/2, ri + hi/2 * k2)
        k4 = drho_dt_local(ti + hi,   ri + hi   * k3)
        rho_dollar[i+1] = ri + (hi/6) * (k1 + 2*k2 + 2*k3 + k4)

    rho_abs_t      = rho_dollar * config.beta
    rho_abs_interp = CubicSpline(times, rho_abs_t)

    
    n0 = 1.0
    c0 = (config.beta_groups / (config.Lambda * config.lambda_groups)) * n0
    y0 = np.concatenate(([config.T0], [n0], c0))

    
    def f_wrapped(t, y):
        return pke_rhs(t, y, rho_abs_interp)

    sol = solve_ivp(
        f_wrapped,
        (times[0], times[-1]),
        y0,
        method = "BDF",
        t_eval = times,
        rtol   = 1e-8,
        atol   = 1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    
    sample_times = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 150.0, 199.0]
    indices      = [np.argmin(np.abs(sol.t - t)) for t in sample_times]
    t_selected   = sol.t[indices]
    y_selected   = sol.y[:, indices]

    print(f"\n{'='*90}")
    print(f"  Stiffness Analysis — {label}")
    print(f"{'='*90}")

    
    results = compute_stiffness(f_wrapped, t_selected, y_selected, verbose=True)
    return results

def plot_stiffness(results):
    t_vals  = [r["t"]               for r in results]
    sr_vals = [r["stiffness_ratio"] for r in results]
    hm_vals = [r["hEul"]           for r in results]
    hr_vals = [r["hRK"]            for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Stiffness ratio
    axes[0].semilogy(t_vals, sr_vals, "b-o", markevery=1)
    axes[0].axhline(1e3, color="r", linestyle="--", label="Stiffness threshold (1000)")
    axes[0].set_ylabel("Stiffness Ratio  λ_max / λ_min", fontsize=11)
    axes[0].set_title("Stiffness Analysis of PRKE System", fontsize=12)
    axes[0].legend()
    axes[0].grid(True)

    # h_min and h_RK
    axes[1].semilogy(t_vals, hm_vals, "r-s", label="h_min (Explicit Euler limit)")
    axes[1].semilogy(t_vals, hr_vals, "g-^", label="h_RK  (RK4 limit)")
    axes[1].axhline(config.dt, color="k", linestyle="--", label=f"Current dt = {config.dt} s")
    axes[1].set_ylabel("Step size (s)", fontsize=11)
    axes[1].set_xlabel("Time (s)",      fontsize=11)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def compare_all_models():

    # --- polynomial ---
    n0 = 1.0
    c0 = (config.beta_groups / (config.Lambda * config.lambda_groups)) * n0
    y0 = np.array([n0, config.T0, *c0])

    sol_poly = solve_ivp(
        prke_system,
        (config.times[0], config.times[-1]),
        y0, method="BDF", t_eval=config.times,
        rtol=1e-8, atol=1e-10,
    )
    sample_times = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 150.0, 199.0]
    indices      = [np.argmin(np.abs(sol_poly.t - t)) for t in sample_times]

    all_results = {}
    all_results["polynomial"] = compute_stiffness(
        prke_system,
        sol_poly.t[indices],
        sol_poly.y[:, indices],
        verbose=True
    )

    # --- drhodx models ---
    models = {
        "sin2"    : rho_func.drho_dx_sin2,
        "gauss"   : rho_func.drho_dx_gauss,
        "sigmoid" : rho_func.drho_dx_sigmoid,
    }
    for label, drho_fn in models.items():
        all_results[label] = run_stiffness_drhodx(drho_fn, label=label)

    # --- style ---
    colors = {
        "polynomial" : "k",
        "sin2"       : "b",
        "gauss"      : "g",
        "sigmoid"    : "r"
    }
    markers = {
        "polynomial" : "D",
        "sin2"       : "o",
        "gauss"      : "s",
        "sigmoid"    : "^"
    }

    # --- Figure 1: Stiffness Ratio ---
    plt.figure(figsize=(9, 6))
    for label, results in all_results.items():
        t_vals  = [r["t"]               for r in results]
        sr_vals = [r["stiffness_ratio"] for r in results]
        plt.semilogy(t_vals, sr_vals,
                     color=colors[label],
                     marker=markers[label],
                     fillstyle='none',
                     label=label)
    plt.axhline(1e3, color="gray", linestyle="--", label="Threshold (1000)")
    plt.xlabel("Time (s)",          fontsize=12)
    plt.ylabel("Stiffness Ratio",   fontsize=12)
    plt.title("Stiffness Ratio — All rho Models", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- Figure 2: hEul ---
    plt.figure(figsize=(9, 6))
    for label, results in all_results.items():
        t_vals  = [r["t"]    for r in results]
        hE_vals = [r["hEul"] for r in results]
        plt.semilogy(t_vals, hE_vals,
                     color=colors[label],
                     marker=markers[label],
                     fillstyle='none',
                     label=label)
    plt.axhline(config.dt, color="gray", linestyle="--", label=f"dt = {config.dt} s")
    plt.xlabel("Time (s)",                      fontsize=12)
    plt.ylabel("hEul (s)",                      fontsize=12)
    plt.title("Max Explicit Euler Step — All rho Models", fontsize=12)
    plt.legend()
    # --- Major grid ---
    plt.grid(True, which='major', linestyle='-',  linewidth=0.8, color='gray', alpha=0.7)

    # --- Minor grid ---
    plt.grid(True, which='minor', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)
    plt.tight_layout()

    # --- Figure 3: hRK ---
    plt.figure(figsize=(9, 6))
    for label, results in all_results.items():
        t_vals  = [r["t"]   for r in results]
        hR_vals = [r["hRK"] for r in results]
        plt.semilogy(t_vals, hR_vals,
                     color=colors[label],
                     marker=markers[label],
                     fillstyle='none',
                     label=label)
    plt.axhline(config.dt, color="gray", linestyle="--", label=f"dt = {config.dt} s")
    plt.xlabel("Time (s)",               fontsize=12)
    plt.ylabel("hRK (s)",                fontsize=12)
    plt.title("Max RK4 Step — All rho Models", fontsize=12)
    plt.legend()
    # --- Major grid ---
    plt.grid(True, which='major', linestyle='-',  linewidth=0.8, color='gray', alpha=0.7)

    # --- Minor grid ---
    plt.grid(True, which='minor', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)

    plt.tight_layout()

    plt.show()
    return all_results

if __name__ == "__main__":

    # --- original BDF (polynomial rho) ---
    print("\n" + "="*90)
    print("  Stiffness Analysis — polynomial rho (prke_system)")
    print("="*90)
    n0 = 1.0
    c0 = (config.beta_groups / (config.Lambda * config.lambda_groups)) * n0
    y0 = np.array([n0, config.T0, *c0])

    sol = solve_ivp(
        prke_system,
        (config.times[0], config.times[-1]),
        y0, method="BDF", t_eval=config.times,
        rtol=1e-8, atol=1e-10,
    )

    sample_times = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 150.0, 199.0]
    indices      = [np.argmin(np.abs(sol.t - t)) for t in sample_times]
    results_poly = compute_stiffness(prke_system, sol.t[indices],
                                     sol.y[:, indices], verbose=True)
    plot_stiffness(results_poly)

    # --- drhodx models (sin2, gauss, sigmoid) ---
    compare_all_models()