import numpy as np
import matplotlib.pyplot as plt

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
def build_rho_abs_interp(drho_func, pos_x_percent=None):
    pos_x_pct = pos_x_percent if pos_x_percent is not None else config.pos_x_percent

    v_rod     = (config.v_percent / 100.0) * config.H
    pos_x     = (pos_x_pct / 100.0) * config.H          # ✅ uses local
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


# ✅ Then pass pos_x_pct when calling it inside run_stiffness_no_temp
def run_stiffness_no_temp(pos_x_percent=None):
    pos_x_pct = pos_x_percent if pos_x_percent is not None else config.pos_x_percent

    sample_times, t_rod_end = get_sample_times_from_position(
        config.v_percent, pos_x_pct, config.H
    )
    v_rod = (config.v_percent / 100.0) * config.H
    pos_x = (pos_x_pct / 100.0) * config.H              # ✅ fixed

    n0 = 1.0
    c0 = (config.beta_groups / (config.Lambda * config.lambda_groups)) * n0
    y0 = np.concatenate(([n0], c0))

    all_results = {}

    # --- polynomial ---
    f_poly = make_rhs_poly(rho_func.rho_polynomial, v_rod, pos_x)
    sol = solve_ivp(f_poly, (config.times[0], config.times[-1]),
                    y0, method="LSODA", t_eval=config.times,
                    rtol=1e-8, atol=1e-10)
    indices = [np.argmin(np.abs(sol.t - t)) for t in sample_times]
    print("\n" + "="*90)
    print(f"  No-Temp Stiffness — polynomial | pos_x = {pos_x_pct}%")
    print("="*90)
    all_results["polynomial"] = compute_stiffness(
        f_poly, sol.t[indices], sol.y[:, indices], verbose=True)

    # --- drhodx models ---
    drhodx_models = {
        "sin2"    : rho_func.drho_dx_sin2,
        "gauss"   : rho_func.drho_dx_gauss,
        "sigmoid" : rho_func.drho_dx_sigmoid,
    }
    for label, drho_fn in drhodx_models.items():
        rho_abs_interp, times = build_rho_abs_interp(drho_fn, pos_x_pct)  # ✅ pass pos_x_pct
        f_drhodx = make_rhs_drhodx(rho_abs_interp)
        sol = solve_ivp(f_drhodx, (times[0], times[-1]),
                        y0, method="LSODA", t_eval=times,
                        rtol=1e-8, atol=1e-10)
        indices = [np.argmin(np.abs(sol.t - t)) for t in sample_times]
        print("\n" + "="*90)
        print(f"  No-Temp Stiffness — {label} | pos_x = {pos_x_pct}%")
        print("="*90)
        all_results[label] = compute_stiffness(
            f_drhodx, sol.t[indices], sol.y[:, indices], verbose=True)

    return all_results, sample_times, t_rod_end

def get_sample_times_from_position(v_percent, pos_x_percent, H, extra_times=None):
    
    v_rod     = (v_percent / 100.0) * H
    pos_x     = (pos_x_percent / 100.0) * H
    t_rod_end = pos_x / v_rod          # time rod stops moving

    # Key position events
    pos_checkpoints_percent = [10, 25, 50, 75, 90, 100]  # % of pos_x
    t_positions = [
        (p / 100.0) * pos_x / v_rod
        for p in pos_checkpoints_percent
        if (p / 100.0) * pos_x / v_rod <= t_rod_end
    ]

    # After rod stops — observe transient decay
    t_after = [
        t_rod_end + offset
        for offset in [1.0, 5.0, 10.0, 30.0, 60.0, 100.0]
        if t_rod_end + offset < config.t_end
    ]

    sample_times = sorted(set(
        [0.1]            # initial
        + t_positions    # during rod motion
        + [t_rod_end]    # rod stops
        + t_after        # after rod stops
        + (extra_times or [])
    ))

    print(f"\nRod position info:")
    print(f"  v_rod     = {v_rod:.5f} m/s")
    print(f"  pos_x     = {pos_x:.4f} m  ({pos_x_percent}%)")
    print(f"  t_rod_end = {t_rod_end:.2f} s")
    print(f"\nSample times generated: {[round(t, 3) for t in sample_times]}")

    return sample_times, t_rod_end

def run_stiffness_position_variation(position_list=None):
    """
    Run stiffness analysis for multiple rod positions.
    position_list : list of pos_x_percent values to test
    """
    if position_list is None:
        position_list = [10, 25, 50, 75, 100]

    all_position_results = {}

    for pos in position_list:
        print(f"\n{'#'*90}")
        print(f"  POSITION: {pos}%")
        print(f"{'#'*90}")
        results, sample_times, t_rod_end = run_stiffness_no_temp(pos_x_percent=pos)
        all_position_results[pos] = {
            "results"      : results,
            "sample_times" : sample_times,
            "t_rod_end"    : t_rod_end
        }

    return all_position_results
# ─────────────────────────────────────────────
# PLOT — no temperature
# ─────────────────────────────────────────────
def plot_stiffness_no_temp(all_results):

    # ✅ guard — auto-unpack if tuple accidentally passed
    if isinstance(all_results, tuple):
        all_results = all_results[0]

    # ✅ guard — check it is actually a dict now
    if not isinstance(all_results, dict):
        raise TypeError(f"plot_stiffness_no_temp expects dict, got {type(all_results)}")

    colors  = {"polynomial":"k", "sin2":"b", "gauss":"g", "sigmoid":"r"}
    markers = {"polynomial":"D", "sin2":"o", "gauss":"s", "sigmoid":"^"}

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
            ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
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
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Stiffness Ratio", fontsize=12)
    plt.title("Stiffness Ratio — No Temperature Feedback", fontsize=12)
    style_ax()

    # Figure 2 — hEul
    plt.figure(figsize=(9, 6))
    for label, results in all_results.items():
        plt.semilogy([r["t"] for r in results],
                     [r["hEul"] for r in results],
                     color=colors[label], marker=markers[label],
                     fillstyle='none', label=label)
    plt.axhline(config.dt, color="gray", linestyle="--",
                label=f"dt = {config.dt} s")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("hEul (s)", fontsize=12)
    plt.title("Max Explicit Euler Step — No Temperature Feedback", fontsize=12)
    style_ax()

    # Figure 3 — hRK
    plt.figure(figsize=(9, 6))
    for label, results in all_results.items():
        plt.semilogy([r["t"] for r in results],
                     [r["hRK"] for r in results],
                     color=colors[label], marker=markers[label],
                     fillstyle='none', label=label)
    plt.axhline(config.dt, color="gray", linestyle="--",
                label=f"dt = {config.dt} s")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("hRK (s)", fontsize=12)
    plt.title("Max RK4 Step — No Temperature Feedback", fontsize=12)
    style_ax()

    plt.show()


# ─────────────────────────────────────────────
# UPDATE __main__
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ✅ always unpack 3 values
    results_no_temp, sample_times, t_rod_end = run_stiffness_no_temp()
    plot_stiffness_no_temp(results_no_temp)

    pos_variation = [10, 25, 50, 75, 100]
    all_pos = run_stiffness_position_variation(pos_variation)

    print(f"\n{'='*60}")
    print(f"  SUMMARY — Stiffness Ratio at t_rod_end (polynomial)")
    print(f"{'='*60}")
    print(f"{'pos (%)':>10} | {'t_rod_end (s)':>14} | {'Stiffness Ratio':>16}")
    print("-" * 50)
    for pos, data in all_pos.items():
        t_end_rod = data["t_rod_end"]
        res_poly  = data["results"]["polynomial"]
        closest   = min(res_poly, key=lambda r: abs(r["t"] - t_end_rod))
        print(f"{pos:>10} | {t_end_rod:>14.2f} | {closest['stiffness_ratio']:>16.4e}")