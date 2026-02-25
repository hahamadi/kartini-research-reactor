import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

current_folder = os.getcwd()
excel_list = os.path.join(os.getcwd(),"data_operasi_reaktor2")
filename = 'Result_data_v.xlsx'

method = "diff"  # "diff" / "forward" / "central"

dfp = pd.read_excel(os.path.join(excel_list,filename))
dfp["date"] = pd.to_datetime(dfp["date"], dayfirst=True)

# ambil kolom up/down (per tanggal)
dfp["up"] = dfp[f"v_up_mean_{method}"].astype(float)
dfp["down"] = dfp[f"v_down_mean_{method}"].astype(float)

# buat kolom bulan (Period)
dfp["month"] = dfp["date"].dt.to_period("M")

# agregasi per bulan: up -> max, down -> min
g = (dfp.groupby("month")
        .agg(up_max=("up", "max"),
             down_min=("down", "min"))
        .reset_index())

# label bulan
labels = g["month"].dt.strftime("%b %Y").to_list()
x = np.arange(len(g))

down_min = g["down_min"].to_numpy(float)
up_max   = g["up_max"].to_numpy(float)

bottom = down_min
height = up_max - down_min  # karena down_min negatif, tinggi jadi besar

plt.figure(figsize=(16, 6))

plt.bar(x, height, bottom=bottom, width=0.75, color="orange", edgecolor="black", linewidth=0.8, alpha=0.85)

ax = plt.gca()

# pindahkan sumbu-x ke y=0
ax.spines['bottom'].set_position(('data', 0))

# hilangkan garis atas & kanan
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# pastikan ticks tetap di bawah garis tersebut
ax.xaxis.set_ticks_position('bottom')
# garis 0
plt.axhline(0, color="black", linewidth=1)

# (opsional) garis rata-rata global up dan down (berdasarkan data harian)
mean_up = dfp["up"].mean()
mean_down = dfp["down"].mean()
plt.axhline(mean_up, linestyle="--", linewidth=1.5, label=f"Mean Up = {mean_up:.3f}")
plt.axhline(mean_down, linestyle="--", linewidth=1.5, label=f"Mean Down = {mean_down:.3f}")

plt.xticks(x, labels, rotation=45, ha="right")
plt.xlabel("Month")
plt.ylabel("Control Rod Speed (m/s)")
plt.title(f"Monthly Min–Max Range of Regulating Rod Speed ({method})")
plt.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig("Regulating Speed_month.png", dpi=300, bbox_inches='tight')
plt.show()