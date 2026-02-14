import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_rod_steps_point_kinetics_with_feedback(
    schedule=[(20.0, 30.0), (30.0, 60.0), (40.0, 50.0), (50.0, 60.0)],  # [(target%, hold_s), ...]
    H=0.38,                      # m
    rod_speed_percent=0.666242,  # %/s
    Lambda=4.0e-5,               # s
    dt=0.01,                     # s
    t_end=900.0,                 # s
    dn_table_path="fraction_delayed_neutrons_U235.xlsx",
    worth_model="cosine_bell",   # "polynomial" or "cosine_bell"
    rho_total_dollar=1.98,       # used if cosine_bell
    n0=1.0,                      # initial neutron density (relative)
    x0_percent=0.0,              # initial rod position %
    source_S=0.0,                # external source term (often 0)
    # --- thermal feedback (lumped) ---
    T0=300.0,                    # K
    alpha_T_abs_per_K=6.0e-5,    # absolute reactivity per K (positive number; subtracted => negative feedback)
    a_K_per_s_at_n1=0.03,        # K/s when n=1 (heating strength)
    b_1_per_s=0.01,              # 1/s cooling (time constant ~1/b)
    # --- plotting / events ---
    plot=True,
    event_tol_pct=1e-3           # tolerance in percent for reaching targets
):
    """
    6-group point kinetics (Implicit Euler) + rod worth + thermal feedback:
      rho_net_abs(t) = rho_rod_abs(t) - alpha_T*(T(t)-T0)

    Outputs:
      df_out   : time series (t, rod%, x, rho_rod$, rho_net$, rho_abs_rod, rho_abs_net, n, T)
      df_events: event summary table (reach targets, mean during holds, etc.)
    """

    # ---------------------------
    # Read delayed neutron table
    # ---------------------------
    df = pd.read_excel(dn_table_path)
    cols = {c.lower().strip(): c for c in df.columns}
    beta_col = cols.get("beta", None)
    lam_col  = cols.get("lambda", None) or cols.get("lam", None) or cols.get("lamb", None)

    if beta_col is None or lam_col is None:
        raise ValueError(f"Excel must contain columns 'beta' and 'lambda' (or 'lam'). Found: {list(df.columns)}")

    beta_i = df[beta_col].to_numpy(dtype=float)
    lam = df[lam_col].to_numpy(dtype=float)
    beta = float(beta_i.sum())

    # ---------------------------
    # Worth curve rho($) vs x (m)
    # ---------------------------
    def rho_dollar_polynomial(x):
        return (-129.16*x**5 + 279.95*x**4 - 215.04*x**3
                + 58.294*x**2 + 1.3702*x - 0.0029)

    def rho_dollar_cosine_bell(x):
        x = np.clip(x, 0.0, H)
        return rho_total_dollar * (x / H - (1.0 / (2.0*np.pi)) * np.sin(2.0*np.pi*x / H))

    worth_model_l = worth_model.lower()
    if worth_model_l == "polynomial":
        rho_func = rho_dollar_polynomial
    elif worth_model_l == "cosine_bell":
        rho_func = rho_dollar_cosine_bell
    else:
        raise ValueError("worth_model must be 'polynomial' or 'cosine_bell'")

    # ---------------------------
    # Implicit Euler step (solve (1+G)x(1+G))
    # ---------------------------
    G = len(beta_i)

    def backward_euler_step(n_k, c_k, rho_abs_k1):
        A = np.zeros((1+G, 1+G), dtype=float)
        b = np.zeros(1+G, dtype=float)

        # n_{k+1} = n_k + dt*( S + ((rho-beta)/Lambda)*n_{k+1} + sum lam_i c_{i,k+1} )
        A[0, 0] = 1.0 - dt * (rho_abs_k1 - beta) / Lambda
        A[0, 1:] = -dt * lam
        b[0] = n_k + dt * source_S

        # c_{i,k+1} = c_{i,k} + dt*( beta_i/Lambda*n_{k+1} - lam_i*c_{i,k+1} )
        for i in range(G):
            A[1+i, 0] = -dt * (beta_i[i] / Lambda)
            A[1+i, 1+i] = 1.0 + dt * lam[i]
            b[1+i] = c_k[i]

        y_next = np.linalg.solve(A, b)
        return float(y_next[0]), y_next[1:].astype(float)

    def initial_precursors_steady(n_init):
        return (beta_i / (Lambda * lam)) * n_init

    # ---------------------------
    # Simulation arrays
    # ---------------------------
    N = int(np.ceil(t_end / dt)) + 1
    t = np.linspace(0.0, t_end, N)

    x = np.zeros(N)
    rod_pct = np.zeros(N)

    rho_rod_d = np.zeros(N)     # $ from rod worth
    rho_net_d = np.zeros(N)     # $ after feedback (for plotting)
    rho_rod_abs = np.zeros(N)   # absolute
    rho_net_abs = np.zeros(N)   # absolute

    n = np.zeros(N)
    C = np.zeros((N, G))
    T = np.zeros(N)

    # initial conditions
    x[0] = np.clip((x0_percent/100.0)*H, 0.0, H)
    rod_pct[0] = 100.0 * x[0] / H

    rho_rod_d[0] = rho_func(x[0])
    rho_rod_abs[0] = beta * rho_rod_d[0]

    T[0] = float(T0)

    rho_net_abs[0] = rho_rod_abs[0] - alpha_T_abs_per_K * (T[0] - T0)
    rho_net_d[0] = rho_net_abs[0] / beta if beta != 0 else 0.0

    n[0] = float(n0)
    C[0, :] = initial_precursors_steady(n[0])

    v = (rod_speed_percent / 100.0) * H  # m/s

    # ---------------------------
    # Schedule state + events
    # ---------------------------
    schedule = list(schedule)
    if len(schedule) == 0:
        raise ValueError("schedule must have at least one segment")

    seg_idx = 0
    target_pct, hold_time = schedule[seg_idx]
    target_x = np.clip((target_pct/100.0)*H, 0.0, H)
    hold_remaining = 0.0
    holding_now = False

    events = {}
    hold_n_values = []
    hold_t_values = []

    def pct_close(a, b, tol=event_tol_pct):
        return abs(a - b) <= tol

    # ---------------------------
    # Main loop
    # ---------------------------
    for k in range(N-1):
        # --- rod motion (piecewise target + hold) ---
        if hold_remaining > 0.0:
            x_next = x[k]
            hold_remaining = max(0.0, hold_remaining - dt)
            holding_now = True
        else:
            holding_now = False
            if abs(x[k] - target_x) < 1e-12:
                hold_remaining = float(hold_time)
                x_next = x[k]
                holding_now = hold_remaining > 0.0
            else:
                direction = np.sign(target_x - x[k])
                x_next = x[k] + direction * v * dt
                if (direction > 0 and x_next > target_x) or (direction < 0 and x_next < target_x):
                    x_next = target_x

        x[k+1] = np.clip(x_next, 0.0, H)
        rod_pct[k+1] = 100.0 * x[k+1] / H

        # --- rod reactivity ($ and abs) at k+1 ---
        rho_rod_d[k+1] = rho_func(x[k+1])
        rho_rod_abs[k+1] = beta * rho_rod_d[k+1]

        # --- thermal state update (explicit Euler is fine here) ---
        # dT/dt = a*n - b*(T-T0)
        dTdt = a_K_per_s_at_n1 * n[k] - b_1_per_s * (T[k] - T0)
        T[k+1] = T[k] + dt * dTdt

        # --- net reactivity with feedback (abs) ---
        rho_net_abs[k+1] = rho_rod_abs[k+1] - alpha_T_abs_per_K * (T[k+1] - T0)
        rho_net_d[k+1] = rho_net_abs[k+1] / beta if beta != 0 else 0.0

        # --- kinetics implicit Euler using rho_net_abs at k+1 ---
        n[k+1], C[k+1, :] = backward_euler_step(n[k], C[k, :], rho_net_abs[k+1])

        # -----------------------
        # Event logging
        # -----------------------
        # reach target
        key_reach = f"reach_{int(round(target_pct))}"
        if key_reach not in events and pct_close(rod_pct[k+1], target_pct):
            events[key_reach] = {
                "time_s": t[k+1],
                "rod_percent": rod_pct[k+1],
                "n": n[k+1],
                "rho_rod_$": rho_rod_d[k+1],
                "rho_net_$": rho_net_d[k+1],
                "T_K": T[k+1],
            }

        # collect during hold at this target
        if holding_now and pct_close(rod_pct[k+1], target_pct):
            hold_n_values.append(n[k+1])
            hold_t_values.append(t[k+1])

        # finalize mean when hold ends (we detect: collected values exist, and hold_remaining==0 while at target)
        if (len(hold_n_values) > 0) and (hold_remaining == 0.0) and pct_close(rod_pct[k+1], target_pct):
            key_mean = f"mean_hold_{int(round(target_pct))}"
            if key_mean not in events:
                events[key_mean] = {
                    "time_s_end": t[k+1],
                    "rod_percent": rod_pct[k+1],
                    "n_mean": float(np.mean(hold_n_values)),
                    "n_min": float(np.min(hold_n_values)),
                    "n_max": float(np.max(hold_n_values)),
                    "T_end_K": float(T[k+1]),
                    "rho_net_$_end": float(rho_net_d[k+1]),
                }
                hold_n_values = []
                hold_t_values = []

                # log start moving to next, then advance segment
                if seg_idx + 1 < len(schedule):
                    next_pct, _ = schedule[seg_idx + 1]
                    events[f"start_move_to_{int(round(next_pct))}"] = {
                        "time_s": t[k+1],
                        "from_percent": target_pct,
                        "to_percent": next_pct,
                        "n": n[k+1],
                        "rho_net_$": rho_net_d[k+1],
                        "T_K": T[k+1],
                    }
                    seg_idx += 1
                    target_pct, hold_time = schedule[seg_idx]
                    target_x = np.clip((target_pct/100.0)*H, 0.0, H)
                    hold_remaining = 0.0
                    holding_now = False

    # ---------------------------
    # Outputs
    # ---------------------------
    df_out = pd.DataFrame({
        "time_s": t,
        "rod_percent": rod_pct,
        "x_m": x,
        "rho_rod_dollar": rho_rod_d,
        "rho_net_dollar": rho_net_d,
        "rho_rod_abs": rho_rod_abs,
        "rho_net_abs": rho_net_abs,
        "n": n,
        "T_K": T
    })

    # Event table
    def sort_key(k):
        if k.startswith("reach_"):
            return (0, int(k.split("_")[1]))
        if k.startswith("mean_hold_"):
            return (1, int(k.split("_")[2]))
        if k.startswith("start_move_to_"):
            return (2, int(k.split("_")[3]))
        return (9, 0)

    rows = [{"event": k, **events[k]} for k in sorted(events.keys(), key=sort_key)]
    df_events = pd.DataFrame(rows)

    # ---------------------------
    # Plots
    # ---------------------------
    if plot:
        plt.figure()
        plt.plot(df_out["time_s"], df_out["rod_percent"])
        plt.xlabel("Time (s)")
        plt.ylabel("Rod position (%)")
        plt.title("Rod Position vs Time (Piecewise Targets)")
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(df_out["time_s"], df_out["rho_rod_dollar"], label="rho_rod ($)")
        plt.plot(df_out["time_s"], df_out["rho_net_dollar"], label="rho_net ($) with feedback")
        plt.xlabel("Time (s)")
        plt.ylabel("Reactivity ($)")
        plt.title(f"Reactivity vs Time (worth_model={worth_model}, thermal feedback)")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(df_out["time_s"], df_out["n"])
        plt.xlabel("Time (s)")
        plt.ylabel("Neutron density / relative power n(t)")
        plt.title("6-group Point Kinetics (Implicit Euler) + Thermal Feedback")
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(df_out["time_s"], df_out["T_K"])
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature T (K)")
        plt.title("Lumped Temperature Model")
        plt.grid(True)
        plt.show()
        
        means = df_events[df_events["event"].str.startswith("mean_hold_")].copy()
        means["pos_percent"] = means["event"].str.replace("mean_hold_", "").astype(float)

    plt.figure()
    plt.plot(means["pos_percent"], means["n_mean"], marker="o")
    plt.xlabel("Rod position (%)")
    plt.ylabel("Mean neutron density during hold (n_mean)")
    plt.title("Mean n during hold vs Rod Position")
    plt.grid(True)
    plt.show()


    print("\n=== EVENT SUMMARY (table) ===")
    if len(df_events) == 0:
        print("No events recorded (check dt / speed / tolerance).")
    else:
        print(df_events.to_string(index=False))

    return df_out, df_events


# ---------------------------
# Example usage (copy/paste)
# ---------------------------
df_out, df_events = simulate_rod_steps_point_kinetics_with_feedback(
     schedule=[(20.0, 30.0), (30.0, 60.0), (40.0, 50.0), (50.0, 60.0), (60.0, 60.0)],
     H=0.38,
     rod_speed_percent=0.666242,
     Lambda=4.0e-5,
     dt=0.01,
     t_end=3600.0,
     dn_table_path="fraction_delayed_neutrons_U235.xlsx",
     worth_model="cosine_bell",
     rho_total_dollar=1.98,
     n0=0.0,
     x0_percent=0.0,
     # feedback knobs (tune these):
     T0=300.0,
     alpha_T_abs_per_K=6e-5,
     a_K_per_s_at_n1=0.03,
     b_1_per_s=0.01,
     plot=True
 )
