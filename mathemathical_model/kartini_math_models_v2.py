import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_rod_steps_point_kinetics(
    schedule=[(20.0, 20.0), (30.0, 30.0)],     # [(target_percent, hold_seconds), ...]
    H=0.38,                                    # m, stroke (0..H)
    rod_speed_percent=0.666242,                # %/s (your measured speed)
    Lambda=4.0e-5,                             # s (prompt generation time)
    dt=0.01,                                   # s
    t_end=120.0,                               # s
    dn_table_path="fraction_delayed_neutrons_U235.xlsx",
    worth_model="polynomial",                  # "polynomial" or "cosine_bell"
    rho_total_dollar=1.95,                     # used if worth_model="cosine_bell"
    n0=1.0,                                    # initial neutron density (relative)
    x0_percent=0.0,                            # initial rod position in %
    source_S=0.0,                              # external source term S(t), often 0
    plot=True
):
    """
    One-block simulation:
      - piecewise rod motion: 0% -> target1 (hold) -> target2 (hold) -> ...
      - rho(t) from rod position via worth_model
      - 6-group point kinetics solved with Backward Euler (implicit Euler)
      - produces plots + event summary table

    Outputs:
      - df_out: time series dataframe (t, rod%, x, rho$, rho_abs, n)
      - df_events: event summary dataframe
    """

    # ---------------------------
    # Read delayed neutron table
    # ---------------------------
    df = pd.read_excel(dn_table_path)

    # Flexible column name handling
    cols = {c.lower().strip(): c for c in df.columns}
    beta_col = cols.get("beta", None)
    lam_col  = cols.get("lambda", None) or cols.get("lam", None) or cols.get("lamb", None)

    if beta_col is None or lam_col is None:
        raise ValueError(
            f"Excel must contain columns 'beta' and 'lambda' (or 'lam'). Found: {list(df.columns)}"
        )

    beta_i = df[beta_col].to_numpy(dtype=float)
    lam = df[lam_col].to_numpy(dtype=float)
    if len(beta_i) != 6 or len(lam) != 6:
        # Still allow, but warn
        print(f"Warning: expected 6 groups; got {len(beta_i)} groups.")

    beta = float(beta_i.sum())

    # ---------------------------
    # Worth curve rho($) vs x (m)
    # ---------------------------
    def rho_dollar_polynomial(x):
        # Fig.8 polynomial you used previously (x in meters)
        return (-129.16*x**5 + 279.95*x**4 - 215.04*x**3
                + 58.294*x**2 + 1.3702*x - 0.0029)

    def rho_dollar_cosine_bell(x):
        # Integral worth (0..H) normalized: rho(0)=0, rho(H)=rho_total_dollar
        x = np.clip(x, 0.0, H)
        return rho_total_dollar * (x / H - (1.0 / (2.0*np.pi)) * np.sin(2.0*np.pi*x / H))

    if worth_model.lower() == "polynomial":
        rho_dollar = rho_dollar_polynomial
    elif worth_model.lower() == "cosine_bell":
        rho_dollar = rho_dollar_cosine_bell
    else:
        raise ValueError("worth_model must be 'polynomial' or 'cosine_bell'")

    # ---------------------------
    # Implicit Euler step (7x7)
    # ---------------------------
    def backward_euler_step(n_k, c_k, rho_abs_k1):
        A = np.zeros((1 + len(beta_i), 1 + len(beta_i)), dtype=float)
        b = np.zeros(1 + len(beta_i), dtype=float)

        # neutron equation:
        # n_{k+1} = n_k + dt*( S + ((rho-beta)/Lambda)*n_{k+1} + sum lam_i c_{i,k+1} )
        A[0, 0] = 1.0 - dt * (rho_abs_k1 - beta) / Lambda
        A[0, 1:] = -dt * lam
        b[0] = n_k + dt * source_S  # S(t_{k+1}) assumed constant; adapt if needed

        # precursor equations:
        # c_{i,k+1} = c_{i,k} + dt*( beta_i/Lambda*n_{k+1} - lam_i*c_{i,k+1} )
        for i in range(len(beta_i)):
            A[1+i, 0] = -dt * (beta_i[i] / Lambda)
            A[1+i, 1+i] = 1.0 + dt * lam[i]
            b[1+i] = c_k[i]

        y_next = np.linalg.solve(A, b)
        return float(y_next[0]), y_next[1:].astype(float)

    def initial_precursors_steady(n_init):
        # Steady-state for rho=0, S=0: c_i = (beta_i/(Lambda*lam_i))*n
        return (beta_i / (Lambda * lam)) * n_init

    # ---------------------------
    # Simulation arrays
    # ---------------------------
    N = int(np.ceil(t_end / dt)) + 1
    t = np.linspace(0.0, t_end, N)

    x = np.zeros(N)              # m
    rod_pct = np.zeros(N)        # %
    rho_dol = np.zeros(N)          # $
    rho_abs = np.zeros(N)        # dimensionless
    n = np.zeros(N)              # neutron density / relative power
    C = np.zeros((N, len(beta_i)))

    # initial conditions
    x[0] = np.clip((x0_percent/100.0)*H, 0.0, H)
    rod_pct[0] = 100.0 * x[0] / H
    rho_dol[0] = rho_dollar(x[0])
    rho_abs[0] = beta * rho_dol[0]          # $ -> absolute reactivity
    n[0] = n0
    C[0, :] = initial_precursors_steady(n0)

    v = (rod_speed_percent / 100.0) * H   # m/s

    # ---------------------------
    # Schedule state + event logging
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

    def pct_close(a, b, tol_pct=1e-6):
        return abs(a - b) <= tol_pct

    # ---------------------------
    # Main loop
    # ---------------------------
    for k in range(N-1):
        # Determine next x based on holding/moving
        if hold_remaining > 0.0:
            # holding position
            x_next = x[k]
            hold_remaining = max(0.0, hold_remaining - dt)
            holding_now = True
        else:
            holding_now = False
            # move toward target
            if abs(x[k] - target_x) < 1e-12:
                # reached target -> start holding
                hold_remaining = float(hold_time)
                x_next = x[k]
                holding_now = hold_remaining > 0.0
            else:
                direction = np.sign(target_x - x[k])
                x_next = x[k] + direction * v * dt
                # prevent overshoot
                if (direction > 0 and x_next > target_x) or (direction < 0 and x_next < target_x):
                    x_next = target_x

        x[k+1] = np.clip(x_next, 0.0, H)
        rod_pct[k+1] = 100.0 * x[k+1] / H

        # worth -> rho
        rho_dol[k+1] = rho_dollar(x[k+1])
        rho_abs[k+1] = beta * rho_dol[k+1]

        # kinetics (implicit Euler uses rho at k+1)
        n[k+1], C[k+1, :] = backward_euler_step(n[k], C[k, :], rho_abs[k+1])

        # -----------------------
        # Event logging (generic)
        # -----------------------
        # Reach target event (first time per target)
        key_reach = f"reach_{int(round(target_pct))}"
        if key_reach not in events and pct_close(rod_pct[k+1], target_pct, tol_pct=1e-3):
            events[key_reach] = {"time_s": t[k+1], "rod_percent": rod_pct[k+1], "n": n[k+1], "rho_$": rho_dol[k+1]}

        # Hold mean logging
        if holding_now and pct_close(rod_pct[k+1], target_pct, tol_pct=1e-3):
            # start collecting when hold starts
            hold_n_values.append(n[k+1])

        # When holding just finished (transition out of hold): record mean for this target
        # We detect by: was holding at step k (hold_remaining before update >0) and now hold_remaining == 0
        # A simpler: if we have collected values and hold_remaining == 0 and we're at target, finalize.
        if (len(hold_n_values) > 0) and (hold_remaining == 0.0) and pct_close(rod_pct[k+1], target_pct, tol_pct=1e-3):
            key_mean = f"mean_hold_{int(round(target_pct))}"
            if key_mean not in events:
                events[key_mean] = {"time_s_end": t[k+1], "rod_percent": rod_pct[k+1], "n_mean": float(np.mean(hold_n_values))}
                hold_n_values = []  # reset after logging mean

                # Also log "start_move_to_next" if there is a next segment
                if seg_idx + 1 < len(schedule):
                    next_pct, _ = schedule[seg_idx + 1]
                    events[f"start_move_to_{int(round(next_pct))}"] = {
                        "time_s": t[k+1], "from_percent": target_pct, "to_percent": next_pct,
                        "n": n[k+1], "rho_$": rho_dol[k+1]
                    }

                    # advance to next segment
                    seg_idx += 1
                    target_pct, hold_time = schedule[seg_idx]
                    target_x = np.clip((target_pct/100.0)*H, 0.0, H)
                    hold_remaining = 0.0
                    holding_now = False

    # ---------------------------
    # Build outputs
    # ---------------------------
    df_out = pd.DataFrame({
        "time_s": t,
        "rod_percent": rod_pct,
        "x_m": x,
        "rho_dollar": rho_dol,
        "rho_abs": rho_abs,
        "n": n
    })

    # Event summary table
    rows = []
    # For readability: sort keys in a logical-ish order
    def sort_key(k):
        # reach_20, mean_hold_20, start_move_to_30, reach_30, mean_hold_30, ...
        if k.startswith("reach_"):
            return (0, int(k.split("_")[1]))
        if k.startswith("mean_hold_"):
            return (1, int(k.split("_")[2]))
        if k.startswith("start_move_to_"):
            return (2, int(k.split("_")[3]))
        return (9, 0)

    for k in sorted(events.keys(), key=sort_key):
        rows.append({"event": k, **events[k]})

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
        plt.plot(df_out["time_s"], df_out["rho_dollar"])
        plt.xlabel("Time (s)")
        plt.ylabel("Reactivity ($)")
        plt.title(f"Reactivity vs Time (worth_model={worth_model})")
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(df_out["time_s"], df_out["n"])
        plt.xlabel("Time (s)")
        plt.ylabel("Neutron density / relative power n(t)")
        plt.title("6-group Point Kinetics (Implicit Euler)")
        plt.grid(True)
        plt.show()

    print("\n=== EVENT SUMMARY (table) ===")
    if len(df_events) == 0:
        print("No events recorded (check dt / speed / tolerance).")
    else:
        print(df_events.to_string(index=False))

    return df_out, df_events


# ---------------------------
# Example usage (uncomment to run)
# ---------------------------
df_out, df_events = simulate_rod_steps_point_kinetics(
     schedule=[(20.0, 30.0), (30.0, 60.0), (40.0, 50.0), (50.0, 60.0)],
     H=0.38,
     rod_speed_percent=0.666242,
     Lambda=4.0e-5,
     dt=0.01,
     t_end=900.0,
     dn_table_path="fraction_delayed_neutrons_U235.xlsx",
     worth_model="cosine_bell",    # or "cosine_bell"
     rho_total_dollar=1.98,
     n0=1.0,
     x0_percent=0.0,
     plot=True
)
