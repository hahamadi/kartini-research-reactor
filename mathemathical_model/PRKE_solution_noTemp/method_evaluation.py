"""
evaluate_methods.py
===================
Numerical method evaluation for PRKE (Point Reactor Kinetics Equations).
Metrics: wall-clock time, convergence (error vs dt), work-precision diagram.

USAGE:
    Place this file in the same folder as your other polynom_solution_*.py files,
    alongside simulation_config.py, rho_x_functions.py, and the Excel file.
    Run:  python evaluate_methods.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os

# ── shared physics ──────────────────────────────────────────────────────────
from simulation_config import H, rho_max, v_percent, pos_x_percent, Lambda
from rho_x_functions import rho_polynomial

cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd, file_delayed_neutron), index_col=None)
beta_groups  = df_fdn["beta"].to_numpy(dtype=float)
lambda_groups = df_fdn["lambda"].to_numpy(dtype=float)
beta = float(np.sum(beta_groups))

t_end = pos_x_percent / v_percent  # same as simulation_config

# ── reference solution (BDF with very tight tolerances via scipy) ────────────
from scipy.integrate import solve_ivp

def _rhs_ref(t, y, v_rod, pos_x):
    n   = y[0]
    c   = y[1:]
    x   = min(v_rod * t, pos_x)
    rho_abs = rho_polynomial(x) * beta
    dn  = ((rho_abs - beta) / Lambda) * n + np.dot(lambda_groups, c)
    dc  = (beta_groups / Lambda) * n - lambda_groups * c
    return np.concatenate(([dn], dc))

def compute_reference():
    """High-accuracy BDF solution used as 'truth' for error comparisons."""
    v_rod  = (v_percent / 100.0) * H
    pos_x  = (pos_x_percent / 100.0) * H
    n0 = 1.0
    c0 = beta_groups / (Lambda * lambda_groups) * n0
    y0 = np.concatenate(([n0], c0))
    # very fine time grid for reference
    t_fine = np.linspace(0, t_end, 5001)
    sol = solve_ivp(_rhs_ref, (0, t_end), y0, method="BDF",
                    t_eval=t_fine, args=(v_rod, pos_x),
                    rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[0]   # times, neutron density

print("Computing reference solution …")
t_ref, n_ref = compute_reference()
print(f"  Reference: {len(t_ref)} points, n_final = {n_ref[-1]:.6f}")

# ── self-contained solver implementations ───────────────────────────────────
# (independent of the original module files so this script is self-contained)

def solve_euler(dt_val):
    v_rod = (v_percent / 100.0) * H
    pos_x = (pos_x_percent / 100.0) * H
    N     = int(np.ceil(t_end / dt_val)) + 1
    times = np.linspace(0.0, t_end, N)

    n0 = 1.0
    c0 = beta_groups / (Lambda * lambda_groups) * n0
    n  = n0
    c  = c0.copy()
    n_out = np.empty(N); n_out[0] = n0

    for i in range(1, N):
        dt_i  = times[i] - times[i-1]
        t_now = times[i-1]
        x     = min(v_rod * t_now, pos_x)
        rho_a = rho_polynomial(x) * beta
        dn    = ((rho_a - beta) / Lambda) * n + np.dot(lambda_groups, c)
        dc    = (beta_groups / Lambda) * n - lambda_groups * c
        n     = n  + dt_i * dn
        c     = c  + dt_i * dc
        n_out[i] = n

    return times, n_out


def solve_rk4(dt_val):
    v_rod = (v_percent / 100.0) * H
    pos_x = (pos_x_percent / 100.0) * H
    N     = int(np.ceil(t_end / dt_val)) + 1
    times = np.linspace(0.0, t_end, N)

    n0 = 1.0
    c0 = beta_groups / (Lambda * lambda_groups) * n0
    y  = np.concatenate(([n0], c0))
    Y  = np.empty((N, len(y))); Y[0] = y

    def rhs(t, yv):
        n  = yv[0]; cv = yv[1:]
        x  = min(v_rod * t, pos_x)
        ra = rho_polynomial(x) * beta
        dn = ((ra - beta) / Lambda) * n + np.dot(lambda_groups, cv)
        dc = (beta_groups / Lambda) * n - lambda_groups * cv
        return np.concatenate(([dn], dc))

    for i in range(1, N):
        t = times[i-1]; h = times[i] - t; yv = Y[i-1]
        k1 = rhs(t,           yv)
        k2 = rhs(t + 0.5*h,  yv + 0.5*h*k1)
        k3 = rhs(t + 0.5*h,  yv + 0.5*h*k2)
        k4 = rhs(t + h,       yv + h*k3)
        Y[i] = yv + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    return times, Y[:, 0]


def solve_etd1(dt_val):
    """
    ETD1 for PRKE.
    n'  = [(rho - beta)/Lambda] * n  +  sum(lambda_i * c_i)   [linear-in-n part exact]
    c_i' = (beta_i/Lambda)*n  -  lambda_i*c_i                 [linear-in-c exact]
    Precursor update uses n[i] (consistent) not n[i-1].
    """
    v_rod = (v_percent / 100.0) * H
    pos_x = (pos_x_percent / 100.0) * H
    N     = int(np.ceil(t_end / dt_val)) + 1
    times = np.linspace(0.0, t_end, N)

    def phi1(z):
        return 1.0 + z/2.0 + z**2/6.0 if abs(z) < 1e-8 else (np.exp(z) - 1.0) / z

    n0 = 1.0
    c0 = beta_groups / (Lambda * lambda_groups) * n0
    n_out = np.empty(N); n_out[0] = n0
    c = c0.copy()

    for i in range(1, N):
        h     = times[i] - times[i-1]
        t_now = times[i-1]
        x     = min(v_rod * t_now, pos_x)
        rho_a = rho_polynomial(x) * beta

        # neutron density (ETD1: exact linear operator + phi1 nonhomogeneous)
        Cn = (rho_a - beta) / Lambda
        z  = h * Cn
        src = float(np.dot(lambda_groups, c))
        n_new = n_out[i-1] * np.exp(z) + h * phi1(z) * src
        n_out[i] = n_new

        # precursor groups (exact integrating factor, use n_new for source)
        exp_lam = np.exp(-lambda_groups * h)
        c = exp_lam * c + (1.0 - exp_lam) / lambda_groups * (beta_groups / Lambda) * n_new

    return times, n_out


def solve_bdf(dt_val):
    v_rod = (v_percent / 100.0) * H
    pos_x = (pos_x_percent / 100.0) * H
    N     = int(np.ceil(t_end / dt_val)) + 1
    times = np.linspace(0.0, t_end, N)

    n0 = 1.0
    c0 = beta_groups / (Lambda * lambda_groups) * n0
    y0 = np.concatenate(([n0], c0))

    def rhs(t, y):
        n = y[0]; cv = y[1:]
        x = min(v_rod * t, pos_x)
        ra = rho_polynomial(x) * beta
        dn = ((ra - beta) / Lambda) * n + np.dot(lambda_groups, cv)
        dc = (beta_groups / Lambda) * n - lambda_groups * cv
        return np.concatenate(([dn], dc))

    sol = solve_ivp(rhs, (0, t_end), y0, method="BDF",
                    t_eval=times, rtol=1e-6, atol=1e-9)
    return sol.t, sol.y[0]


def solve_lsoda(dt_val):
    v_rod = (v_percent / 100.0) * H
    pos_x = (pos_x_percent / 100.0) * H
    N     = int(np.ceil(t_end / dt_val)) + 1
    times = np.linspace(0.0, t_end, N)

    n0 = 1.0
    c0 = beta_groups / (Lambda * lambda_groups) * n0
    y0 = np.concatenate(([n0], c0))

    def rhs(t, y):
        n = y[0]; cv = y[1:]
        x = min(v_rod * t, pos_x)
        ra = rho_polynomial(x) * beta
        dn = ((ra - beta) / Lambda) * n + np.dot(lambda_groups, cv)
        dc = (beta_groups / Lambda) * n - lambda_groups * cv
        return np.concatenate(([dn], dc))

    sol = solve_ivp(rhs, (0, t_end), y0, method="LSODA",
                    t_eval=times, rtol=1e-6, atol=1e-9)
    return sol.t, sol.y[0]


# ── evaluation helpers ───────────────────────────────────────────────────────

def interp_ref(t_query):
    """Interpolate reference solution to arbitrary time points."""
    return np.interp(t_query, t_ref, n_ref)


def global_error(t_sol, n_sol):
    """Max absolute error vs reference (L-inf norm)."""
    n_exact = interp_ref(t_sol)
    with np.errstate(invalid='ignore'):
        valid = n_exact > 0
    return float(np.max(np.abs(n_sol[valid] - n_exact[valid])))


def relative_error(t_sol, n_sol):
    """Max relative error vs reference."""
    n_exact = interp_ref(t_sol)
    with np.errstate(invalid='ignore', divide='ignore'):
        rel = np.abs((n_sol - n_exact) / n_exact)
    return float(np.nanmax(rel))


SOLVERS = {
    "Euler":  solve_euler,
    "RK4":    solve_rk4,
    "ETD1":   solve_etd1,
    "BDF":    solve_bdf,
    "LSODA":  solve_lsoda,
}
COLORS = {
    "Euler": "#D85A30",
    "RK4":   "#185FA5",
    "ETD1":  "#7F77DD",
    "BDF":   "#1D9E75",
    "LSODA": "#BA7517",
}
MARKERS = {"Euler":"o","RK4":"s","ETD1":"^","BDF":"D","LSODA":"v"}

# ── 1. Timing benchmark ──────────────────────────────────────────────────────
print("\n── Timing benchmark (dt = 0.01, 10 runs each) ──")
dt_bench  = 0.01
N_REPEATS = 10
timing = {}

for name, solver in SOLVERS.items():
    times_list = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        solver(dt_bench)
        times_list.append(time.perf_counter() - t0)
    timing[name] = np.median(times_list)
    print(f"  {name:8s}  {timing[name]*1e3:.2f} ms  (median of {N_REPEATS})")

# ── 2. Convergence study ─────────────────────────────────────────────────────
print("\n── Convergence study ──")
dt_list = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001]
conv = {name: [] for name in SOLVERS}

for dt_val in dt_list:
    for name, solver in SOLVERS.items():
        try:
            t_s, n_s = solver(dt_val)
            err = global_error(t_s, n_s)
        except Exception as e:
            err = np.nan
        conv[name].append(err)
        print(f"  {name:8s}  dt={dt_val:.3f}  err={err:.3e}")

# ── 3. Work-precision diagram ─────────────────────────────────────────────────
print("\n── Work–precision study ──")
# nfe = cost per step * number of steps
STAGES = {"Euler":1, "RK4":4, "ETD1":1, "BDF":5, "LSODA":5}
# BDF/LSODA adaptive internally; we estimate NFE from dt_list
wp = {name: {"nfe":[], "err":[]} for name in SOLVERS}

for dt_val in dt_list:
    N_steps = int(np.ceil(t_end / dt_val))
    for name, solver in SOLVERS.items():
        try:
            t_s, n_s = solver(dt_val)
            err = global_error(t_s, n_s)
        except Exception:
            err = np.nan
        nfe = N_steps * STAGES.get(name, 1)
        wp[name]["nfe"].append(nfe)
        wp[name]["err"].append(err)

# ── 4. Solution comparison at dt = 0.01 ──────────────────────────────────────
print("\n── Computing solutions at dt=0.01 for plot ──")
solutions = {}
for name, solver in SOLVERS.items():
    t_s, n_s = solver(0.01)
    solutions[name] = (t_s, n_s)

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

ax1 = fig.add_subplot(gs[0, :2])   # solution comparison  (wide)
ax2 = fig.add_subplot(gs[0, 2])    # timing bar chart
ax3 = fig.add_subplot(gs[1, 0])    # convergence log-log
ax4 = fig.add_subplot(gs[1, 1])    # work-precision
ax5 = fig.add_subplot(gs[1, 2])    # error at fixed dt=0.01

# ── ax1: solution comparison ─────────────────────────────────────────────────
ax1.plot(t_ref, np.log(np.maximum(n_ref, 1e-30)), color='black',
         linewidth=2, linestyle='-', label='Reference (BDF tol=1e-10)', zorder=5)
for name, (t_s, n_s) in solutions.items():
    valid = n_s > 0
    ax1.plot(t_s[valid], np.log(n_s[valid]),
             color=COLORS[name], linewidth=1.2, linestyle='--',
             marker=MARKERS[name], markersize=3,
             markevery=max(1, valid.sum()//30), label=name)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("log n(t)")
ax1.set_title("Solution comparison — log neutron density  (dt = 0.01 s)")
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.4)

# ── ax2: timing bar chart ─────────────────────────────────────────────────────
names = list(timing.keys())
vals  = [timing[n]*1e3 for n in names]
bars  = ax2.bar(names, vals, color=[COLORS[n] for n in names], edgecolor='white', linewidth=0.5)
for bar, v in zip(bars, vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{v:.1f}", ha='center', va='bottom', fontsize=8)
ax2.set_ylabel("Wall time (ms)  — median of 10 runs")
ax2.set_title(f"Execution time  (dt = {dt_bench})")
ax2.grid(axis='y', alpha=0.4)

# ── ax3: convergence ─────────────────────────────────────────────────────────
for name in SOLVERS:
    errs = conv[name]
    valid = [(d, e) for d, e in zip(dt_list, errs) if np.isfinite(e) and e > 0]
    if not valid: continue
    dx, dy = zip(*valid)
    ax3.loglog(dx, dy, color=COLORS[name], marker=MARKERS[name],
               markersize=5, linewidth=1.5, label=name)

# reference slope lines
d0 = np.array([dt_list[0], dt_list[-1]])
ax3.loglog(d0, 5*d0**1,   'k--', linewidth=0.8, alpha=0.5, label='slope 1')
ax3.loglog(d0, 2*d0**4,   'k:',  linewidth=0.8, alpha=0.5, label='slope 4')
ax3.set_xlabel("Step size dt (s)")
ax3.set_ylabel("Max absolute error  |n − n_ref|")
ax3.set_title("Convergence (error vs step size)")
ax3.legend(fontsize=7)
ax3.grid(True, which='both', alpha=0.3)

# ── ax4: work-precision ───────────────────────────────────────────────────────
for name in SOLVERS:
    nfe_l = wp[name]["nfe"]
    err_l = wp[name]["err"]
    valid = [(n, e) for n, e in zip(nfe_l, err_l) if np.isfinite(e) and e > 0]
    if not valid: continue
    nx, ey = zip(*valid)
    ax4.loglog(nx, ey, color=COLORS[name], marker=MARKERS[name],
               markersize=5, linewidth=1.5, label=name)
ax4.set_xlabel("Function evaluations (NFE)")
ax4.set_ylabel("Max absolute error")
ax4.set_title("Work–precision diagram\n(lower-left = better)")
ax4.legend(fontsize=7)
ax4.grid(True, which='both', alpha=0.3)

# ── ax5: per-method relative error at dt=0.01 ────────────────────────────────
rel_errs = {}
for name, (t_s, n_s) in solutions.items():
    rel_errs[name] = relative_error(t_s, n_s)

names2  = list(rel_errs.keys())
vals2   = [rel_errs[n] for n in names2]
bars2   = ax5.bar(names2, vals2, color=[COLORS[n] for n in names2],
                  edgecolor='white', linewidth=0.5)
for bar, v in zip(bars2, vals2):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.05,
             f"{v:.2e}", ha='center', va='bottom', fontsize=7, rotation=15)
ax5.set_yscale('log')
ax5.set_ylabel("Max relative error  |n − n_ref| / |n_ref|")
ax5.set_title(f"Relative error at dt = 0.01 s")
ax5.grid(axis='y', alpha=0.4)

fig.suptitle("PRKE Numerical Method Evaluation — Polynomial Rod Worth", fontsize=13, fontweight='bold')
#plt.savefig("prke_method_evaluation.png", dpi=150, bbox_inches='tight')
print("\nSaved: prke_method_evaluation.png")
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════════════════════")
print(f"{'Method':<10} {'Time(ms)':>10} {'Rel. Error':>14} {'n_final':>12}")
print("──────────────────────────────────────────────────────────────")
for name in SOLVERS:
    t_s, n_s = solutions[name]
    print(f"{name:<10} {timing[name]*1e3:>10.2f} {rel_errs[name]:>14.3e} {n_s[-1]:>12.6f}")
print("══════════════════════════════════════════════════════════════")
print(f"{'Reference':<10} {'—':>10} {'—':>14} {n_ref[-1]:>12.6f}")