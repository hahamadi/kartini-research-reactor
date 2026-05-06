import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import rho_x_functions as rho_func
import simulation_config as config
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from polynom_solution_lsoda import run_simulation_prke as run_poly_lsoda
from general_LSDOA_method import run_simulation_prke as run_drhodx_lsoda
import rho_x_functions as rho_func


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

def make_rhs_poly(rho_func_callable, v_rod, pos_x):
    """
    Returns f(t, y) for polynomial rho — no temperature state.
    y = [n, c1, c2, ..., cm]
    """
    def rod_position(t):
        return min(v_rod * t, pos_x)

    def rhs(t, y):
        n = y[0]
        c = y[1:]
        rho_abs = rho_func_callable(rod_position(t)) * config.beta
        dn = ((rho_abs - config.beta) / config.Lambda) * n \
             + np.dot(config.lambda_groups, c)
        dc = (config.beta_groups / config.Lambda) * n \
             - config.lambda_groups * c
        return np.concatenate(([dn], dc))

    return rhs


# ─────────────────────────────────────────────
# RHS WITHOUT TEMPERATURE — drhodx rho
# ─────────────────────────────────────────────
def make_rhs_drhodx(rho_abs_interp):
    """
    Returns f(t, y) for drhodx rho — no temperature state.
    y = [n, c1, c2, ..., cm]
    """
    def rhs(t, y):
        n = y[0]
        c = y[1:]
        rho_abs = float(rho_abs_interp(t))
        dn = ((rho_abs - config.beta) / config.Lambda) * n \
             + np.dot(config.lambda_groups, c)
        dc = (config.beta_groups / config.Lambda) * n \
             - config.lambda_groups * c
        return np.concatenate(([dn], dc))

    return rhs


# ─────────────────────────────────────────────
# BUILD rho_abs TRAJECTORY for drhodx models
# ─────────────────────────────────────────────
def build_rho_abs_interp(drho_func):
    """
    RK4 integration of drho/dt to get rho_abs(t), then interpolate.
    Same logic as inside prke_solver_temp_var.
    """
    from scipy.interpolate import CubicSpline

    v_rod     = (config.v_percent / 100.0) * config.H
    pos_x     = (config.pos_x_percent / 100.0) * config.H
    t_end_rod = pos_x / v_rod
    times     = np.arange(0, config.t_end, config.dt)
    N         = len(times)

    def rod_position(t):
        return min(v_rod * t, pos_x)

    def rod_velocity(t):
        return v_rod if t < t_end_rod else 0.0

    def drho_dt(t, rho_val):
        x = rod_position(t)
        v = rod_velocity(t)
        return drho_func(x, rho_val, config.rho_max, config.H) * v

    rho_dollar = np.zeros(N)
    for i in range(N - 1):
        ti = times[i]
        hi = times[i+1] - times[i]
        ri = rho_dollar[i]
        k1 = drho_dt(ti,        ri)
        k2 = drho_dt(ti + hi/2, ri + hi/2 * k1)
        k3 = drho_dt(ti + hi/2, ri + hi/2 * k2)
        k4 = drho_dt(ti + hi,   ri + hi   * k3)
        rho_dollar[i+1] = ri + (hi/6) * (k1 + 2*k2 + 2*k3 + k4)

    rho_abs_t = rho_dollar * config.beta
    return CubicSpline(times, rho_abs_t), times


# ─────────────────────────────────────────────
# STIFFNESS — no temperature, all models
# ─────────────────────────────────────────────
def run_stiffness_no_temp():
    
    sample_times = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 150.0, 199.0]
    v_rod = (config.v_percent / 100.0) * config.H
    pos_x = (config.pos_x_percent / 100.0) * config.H

    # initial condition — no temperature
    n0 = 1.0
    c0 = (config.beta_groups / (config.Lambda * config.lambda_groups)) * n0
    y0 = np.concatenate(([n0], c0))

    all_results = {}

    # ---------- polynomial ----------
    f_poly = make_rhs_poly(rho_func.rho_polynomial, v_rod, pos_x)

    sol = solve_ivp(f_poly, (config.times[0], config.times[-1]),
                    y0, method="BDF", t_eval=config.times,
                    rtol=1e-8, atol=1e-10)

    indices = [np.argmin(np.abs(sol.t - t)) for t in sample_times]
    print("\n" + "="*90)
    print("  No-Temp Stiffness — polynomial")
    print("="*90)
    all_results["polynomial"] = compute_stiffness(
        f_poly, sol.t[indices], sol.y[:, indices], verbose=True)

    # ---------- drhodx models ----------
    drhodx_models = {
        "sin2"    : rho_func.drho_dx_sin2,
        "gauss"   : rho_func.drho_dx_gauss,
        "sigmoid" : rho_func.drho_dx_sigmoid,
    }

    for label, drho_fn in drhodx_models.items():
        rho_abs_interp, times = build_rho_abs_interp(drho_fn)
        f_drhodx = make_rhs_drhodx(rho_abs_interp)

        sol = solve_ivp(f_drhodx, (times[0], times[-1]),
                        y0, method="BDF", t_eval=times,
                        rtol=1e-8, atol=1e-10)

        indices = [np.argmin(np.abs(sol.t - t)) for t in sample_times]
        print("\n" + "="*90)
        print(f"  No-Temp Stiffness — {label}")
        print("="*90)
        all_results[label] = compute_stiffness(
            f_drhodx, sol.t[indices], sol.y[:, indices], verbose=True)

    return all_results


# ─────────────────────────────────────────────
# PLOT — no temperature
# ─────────────────────────────────────────────
def plot_stiffness_no_temp(all_results):
    colors  = {"polynomial":"g", "sin2":"r", "gauss":"c", "sigmoid":"m"}
    markers = {"polynomial":"d", "sin2":"s", "gauss":"v", "sigmoid":"."}

    import matplotlib.ticker as ticker

    def style_ax():
        plt.grid(True, which='major', linestyle='-',  linewidth=0.8,
                 color='gray', alpha=0.7)
        plt.grid(True, which='minor', linestyle='--', linewidth=0.5,
                 color='gray', alpha=0.4)
        plt.minorticks_on()
        ax = plt.gca()
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(12.5))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10.0, subs=np.arange(2,10)*0.1))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        plt.legend()
        plt.tight_layout()

    # Figure 1 — Stiffness Ratio
    plt.figure(figsize=(9, 6))
    for label, results in all_results.items():
        plt.semilogy([r["t"] for r in results],
                     [r["stiffness_ratio"] for r in results],
                     color=colors[label], marker=markers[label],
                     fillstyle='none', label=label)
    plt.axhline(1e3, color="gray", linestyle="--", label="Threshold (1000)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Stiffness Ratio", fontsize=12)
    plt.title("Time (s) vs Stiffness Ratio ", fontsize=10)
    
    style_ax()
    plt.savefig(f"stiffness_ratio_time_{config.pos_x_percent}.svg", dpi=300, format='svg', bbox_inches='tight')

    # Figure 2 — hEul
    plt.figure(figsize=(9, 6))
    for label, results in all_results.items():
        plt.semilogy([r["t"] for r in results],
                     [r["hEul"] for r in results],
                     color=colors[label], marker=markers[label],
                     fillstyle='none', label=label)
    plt.axhline(config.dt, color="gray", linestyle="--",
                label=f"dt = {config.dt} s")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("hEul (s)", fontsize=12)
    plt.title("Max Explicit Euler Step", fontsize=10)
    style_ax()
    plt.savefig(f"stiffness_hEul_time_{config.pos_x_percent}.svg", dpi=300, format='svg', bbox_inches='tight')
    

    # Figure 3 — hRK
    plt.figure(figsize=(9, 6))
    for label, results in all_results.items():
        plt.semilogy([r["t"] for r in results],
                     [r["hRK"] for r in results],
                     color=colors[label], marker=markers[label],
                     fillstyle='none', label=label)
    plt.axhline(config.dt, color="gray", linestyle="--",
                label=f"dt = {config.dt} s")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("hRK (s)", fontsize=12)
    plt.title("Max RK4 Step", fontsize=10)
    style_ax()
    plt.savefig(f"stiffness_hRK_time_{config.pos_x_percent}.svg", dpi=300, format='svg', bbox_inches='tight')
    

    plt.show()


# ─────────────────────────────────────────────
# UPDATE __main__
# ─────────────────────────────────────────────
if __name__ == "__main__":

    results_no_temp = run_stiffness_no_temp()
    
    plot_stiffness_no_temp(results_no_temp)
    
    rows = []
    for model, res in results_no_temp.items():
        for r in res:
            rows.append({
                "model"           : model,
                "time_s"          : r["t"],
                "S(t)_max"        : np.round(r["lambda_max"], 3),
                "S(t)_min"        : np.round(r["lambda_min"], 3),
                "stiffness_ratio" : np.round(r["stiffness_ratio"], 3),
                "hEul_s"          : np.round(r["hEul"], 4),
                "hRK_s"           : np.round(r["hRK"], 4)
            })

    df = pd.DataFrame(rows)
    filename = f"stiffness_no_temp_results_BDF_pos{config.pos_x_percent}.xlsx"
    df.to_excel(filename, index=False)
    print(f"Stiffness results saved to {filename}")
    print(df.head())