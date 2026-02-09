import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATA_DIR = os.path.join(os.getcwd(),"data_operasi_reaktor")  # ganti ke folder Anda jika perlu
PATTERN = os.path.join(DATA_DIR, "data_download_practice1_*.xlsx")

ROD_COL = "Regulator Rod [%]"
PWR_COL_CANDIDATES = ["Power NP1000 [kW]", "Power NLW2 [kW]"]  # pilih yang tersedia
TIME_COL = "Time"

# threshold untuk mendeteksi "power response" (untuk delay), dalam unit dP/dt
# kita pakai rule robust: > max( percentile( |dP/dt|, 90 ) * 0.2, angka_min )
MIN_DPDt = 1e-6

# =========================
# HELPERS
# =========================
def extract_date_label_from_filename(path):
    """
    Ambil label tanggal dari nama file: data_download_practice1_DD_MM_YYYY.xlsx
    """
    m = re.search(r"practice1_(\d{2})_(\d{2})_(\d{4})", os.path.basename(path))
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}"
    return os.path.splitext(os.path.basename(path))[0]

def load_one_file(path):
    df = pd.read_excel(path, sheet_name="Download Transaksi", header=1)
    # parse time
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    # pilih power column yang ada
    pwr_col = None
    for c in PWR_COL_CANDIDATES:
        if c in df.columns:
            pwr_col = c
            break
    if pwr_col is None:
        raise ValueError(f"Power column tidak ditemukan di {path}")

    # numeric conversion
    df[ROD_COL] = pd.to_numeric(df[ROD_COL], errors="coerce")
    df[pwr_col] = pd.to_numeric(df[pwr_col], errors="coerce")
    df = df.dropna(subset=[ROD_COL, pwr_col]).reset_index(drop=True)

    # dt & derivatives
    dt_s = df[TIME_COL].diff().dt.total_seconds()
    drod = df[ROD_COL].diff()
    dpwr = df[pwr_col].diff()

    df["dt_s"] = dt_s
    df["rod_speed"] = drod / dt_s           # %/s
    df["dpdt"] = dpwr / dt_s                # kW/s

    return df, pwr_col

def find_motion_segments(df, min_step=0.0):
    """
    Segment rod motion periods where |ΔRod| > min_step
    Returns list of (start_idx, end_idx) inclusive indices.
    """
    moving = df[ROD_COL].diff().abs() > min_step
    idx = np.where(moving.fillna(False).to_numpy())[0]

    if len(idx) == 0:
        return []

    segs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segs.append((start, prev))
            start = i
            prev = i
    segs.append((start, prev))
    return segs

def segment_metrics(df, pwr_col, seg):
    s, e = seg
    # gunakan interval dari s-1 -> e untuk durasi (karena diff berada di i)
    t0 = df.loc[s-1, TIME_COL] if s-1 >= 0 else df.loc[s, TIME_COL]
    t1 = df.loc[e, TIME_COL]
    duration = (t1 - t0).total_seconds()
    if duration <= 0:
        return None

    # rod movement magnitude
    rod0 = df.loc[s-1, ROD_COL] if s-1 >= 0 else df.loc[s, ROD_COL]
    rod1 = df.loc[e, ROD_COL]
    drod = rod1 - rod0

    # average & max speed during segment
    speeds = df.loc[s:e, "rod_speed"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(speeds) == 0:
        return None
    v_mean = speeds.abs().mean()
    v_max = speeds.abs().max()

    # power ramp rate across segment
    p0 = df.loc[s-1, pwr_col] if s-1 >= 0 else df.loc[s, pwr_col]
    p1 = df.loc[e, pwr_col]
    ramp = (p1 - p0) / duration  # kW/s

    direction = "withdrawal" if drod > 0 else "insertion" if drod < 0 else "zero"

    return {
        "t_start": t0,
        "t_end": t1,
        "duration_s": duration,
        "drod_pct": drod,
        "v_mean_pct_s": v_mean,
        "v_max_pct_s": v_max,
        "ramp_kw_s": ramp,
        "direction": direction
    }

def estimate_delay(df, pwr_col, seg):
    """
    Delay: from rod motion start (t0) to first significant dp/dt excursion after t0.
    """
    s, e = seg
    t0 = df.loc[s-1, TIME_COL] if s-1 >= 0 else df.loc[s, TIME_COL]

    # compute threshold from data after t0
    after = df[df[TIME_COL] >= t0].copy()
    dpdt = after["dpdt"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(dpdt) < 10:
        return np.nan

    thr = max(np.percentile(np.abs(dpdt), 90) * 0.2, MIN_DPDt)

    hit = after.index[(after["dpdt"].abs() >= thr)]
    if len(hit) == 0:
        return np.nan

    t_hit = df.loc[hit[0], TIME_COL]
    return (t_hit - t0).total_seconds()

# =========================
# LOAD ALL FILES
# =========================
files = sorted(glob.glob(PATTERN))
if not files:
    raise FileNotFoundError(f"Tidak ada file cocok pattern: {PATTERN}")

all_days = []
seg_rows = []

for f in files:
    label = extract_date_label_from_filename(f)
    df, pwr_col = load_one_file(f)
    df["day"] = label
    df["pwr_col_used"] = pwr_col

    segs = find_motion_segments(df, min_step=0.0)  # deteksi setiap perubahan rod
    for seg in segs:
        m = segment_metrics(df, pwr_col, seg)
        if m is None:
            continue
        m["day"] = label
        m["file"] = os.path.basename(f)
        m["delay_s"] = estimate_delay(df, pwr_col, seg)
        seg_rows.append(m)

    all_days.append((label, df, pwr_col))

seg_df = pd.DataFrame(seg_rows)

# =========================
# FIGURE 1: timelines (2–3 days representative)
# =========================
rep = all_days[:3]  # ambil 3 hari pertama; bisa diganti manual
plt.figure()
for i, (day, df, pwr_col) in enumerate(rep):
    ax = plt.gca() if i == 0 else plt.gca()
    # plot power
    plt.plot(df[TIME_COL], df[pwr_col], label=f"{day} power")

plt.title("Figure 1A: Reactor power timeline (representative days)")
plt.xlabel("Time")
plt.ylabel("Power [kW]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
for (day, df, pwr_col) in rep:
    plt.plot(df[TIME_COL], df[ROD_COL], label=f"{day} rod")
plt.title("Figure 1B: Regulating rod position timeline (representative days)")
plt.xlabel("Time")
plt.ylabel("Regulating Rod [%]")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# FIGURE 2: instantaneous rod speed (one day)
# =========================
day, df, pwr_col = rep[0]
plt.figure()
plt.plot(df[TIME_COL], df["rod_speed"])
plt.title(f"Figure 2: Instantaneous rod speed ({day})")
plt.xlabel("Time")
plt.ylabel("Rod speed [%/s]")
plt.tight_layout()
plt.show()

# =========================
# FIGURE 3: distribution of rod speed across days (boxplot)
# =========================
plt.figure()
data = []
labels = []
for day, df, pwr_col in all_days:
    speeds = df["rod_speed"].replace([np.inf, -np.inf], np.nan).dropna().abs()
    speeds = speeds[speeds > 0]  # only moving
    if len(speeds) > 0:
        data.append(speeds.to_numpy())
        labels.append(day)

plt.boxplot(data, labels=labels, showfliers=False)
plt.title("Figure 3: Distribution of regulating rod speed across datasets")
plt.xlabel("Operational day")
plt.ylabel("Rod speed |%/s| (moving only)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# =========================
# FIGURE 4 (optional): mean & max rod speed per day
# =========================
plt.figure()
day_stats = seg_df.groupby("day").agg(
    mean_speed=("v_mean_pct_s", "mean"),
    max_speed=("v_max_pct_s", "max"),
    std_speed=("v_mean_pct_s", "std"),
).reset_index()

x = np.arange(len(day_stats))
plt.bar(x - 0.2, day_stats["mean_speed"], width=0.4, label="mean speed")
plt.bar(x + 0.2, day_stats["max_speed"], width=0.4, label="max speed")
plt.title("Figure 4 (optional): Mean and maximum rod speed by day")
plt.xlabel("Operational day")
plt.ylabel("Rod speed [%/s]")
plt.xticks(x, day_stats["day"], rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# FIGURE 5: power transient comparison (2 cases, aligned to start)
# =========================
# pilih 2 segmen dari hari berbeda: segmen dengan durasi cukup
cand = seg_df[seg_df["duration_s"] > 10].copy().sort_values("v_mean_pct_s", ascending=False)
pick = cand.groupby("day").head(1).head(2)

plt.figure()
for _, row in pick.iterrows():
    day = row["day"]
    df = [d for (dy, d, pc) in all_days if dy == day][0]
    pwr_col = [pc for (dy, d, pc) in all_days if dy == day][0]

    t0 = row["t_start"]
    t1 = row["t_end"]
    window = df[(df[TIME_COL] >= t0) & (df[TIME_COL] <= t1 + pd.Timedelta(seconds=60))].copy()
    window["t_rel_s"] = (window[TIME_COL] - t0).dt.total_seconds()

    plt.plot(window["t_rel_s"], window[pwr_col], label=f"{day}")

plt.title("Figure 5: Power transients aligned to rod motion start (selected cases)")
plt.xlabel("Time from rod motion start [s]")
plt.ylabel("Power [kW]")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# FIGURE 6: rod speed vs power ramp rate (scatter)
# =========================
plt.figure()
plt.scatter(seg_df["v_mean_pct_s"], seg_df["ramp_kw_s"])
plt.title("Figure 6: Rod speed vs power ramp rate across segments")
plt.xlabel("Mean rod speed [%/s]")
plt.ylabel("Power ramp rate [kW/s]")
plt.tight_layout()
plt.show()

# =========================
# FIGURE 7: delay distribution by day (boxplot)
# =========================
plt.figure()
delay_data = []
delay_labels = []
for day in sorted(seg_df["day"].unique()):
    vals = seg_df.loc[seg_df["day"] == day, "delay_s"].dropna()
    if len(vals) > 0:
        delay_data.append(vals.to_numpy())
        delay_labels.append(day)

plt.boxplot(delay_data, labels=delay_labels, showfliers=False)
plt.title("Figure 7: Delay time between rod motion onset and power response")
plt.xlabel("Operational day")
plt.ylabel("Delay [s]")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# =========================
# FIGURE 8 (optional): variability index (CV) by day
# =========================
plt.figure()
cv = seg_df.groupby("day")["v_mean_pct_s"].apply(lambda x: x.std() / x.mean() if x.mean() != 0 else np.nan)
cv = cv.reset_index(name="cv_speed")

plt.bar(np.arange(len(cv)), cv["cv_speed"])
plt.title("Figure 8 (optional): Coefficient of variation (CV) of rod speed by day")
plt.xlabel("Operational day")
plt.ylabel("CV of mean rod speed [-]")
plt.xticks(np.arange(len(cv)), cv["day"], rotation=45, ha="right")
plt.tight_layout()
plt.show()
