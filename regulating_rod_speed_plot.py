import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
res_name = os.path.join(os.getcwd(),"data_operasi_reaktor2")
path = os.path.join(res_name,"Result_data_v_V2.xlsx")

df = pd.read_excel(path, sheet_name=0)

# Parse tanggal
df["Tanggal"] = pd.to_datetime(df["File"].astype(str), format="%d_%m_%Y", errors="coerce")
if df["Tanggal"].isna().any():
    df["Tanggal"] = pd.to_datetime(df["File"].astype(str), dayfirst=True, errors="coerce")

df = df.dropna(subset=["Tanggal", "Tipe", "Kecepatan_%_s"]).copy()

# Rata-rata per tanggal per tipe
agg = (
    df.groupby(["Tanggal", "Tipe"], as_index=False)["Kecepatan_%_s"]
    .mean()
    .sort_values("Tanggal")
)

pivot = agg.pivot(index="Tanggal", columns="Tipe", values="Kecepatan_%_s").fillna(0)

# Turun dibuat negatif
if "Turun" in pivot.columns:
    pivot["Turun"] = pivot["Turun"]

dates = pivot.index
x = np.arange(len(dates))

fig, ax = plt.subplots(figsize=(14,6))

# Plot bar pada posisi X yang sama
if "Naik" in pivot.columns:
    ax.bar(x, pivot["Naik"], width=0.6, label="Naik")

if "Turun" in pivot.columns:
    ax.bar(x, pivot["Turun"], width=0.6, label="Turun")

# Pindahkan sumbu X ke y=0
ax.spines['bottom'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')

ax.set_xticks(x)
ax.set_xticklabels(dates.strftime("%d-%m-%Y"), rotation=90)

ax.set_xlabel("Tanggal")
ax.set_ylabel("Kecepatan (%/s)")
ax.set_title("Kecepatan Naik & Turun per Tanggal")
ax.legend()

plt.tight_layout()

#out_path = "/mnt/data/grafik_bar_xlabel_di_y0.png"
#plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()

#out_path