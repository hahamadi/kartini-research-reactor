import re
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Dapatkan folder saat ini
current_folder = os.getcwd()
excel_list = os.path.join(os.getcwd(),"data_operasi_reaktor")

TIME_COL = "Time"
ROD_COL = "Regulator Rod [%]"
PWR_COL_CANDIDATES = ["Power NP1000 [kW]", "Power NLW2 [kW]"]
# List semua file Excel
excel_files = []
for file in os.listdir(excel_list):
    if file.lower().endswith(('.xlsx', '.xls')):
        excel_files.append(file)

print(f"Jumlah file Excel di folder '{current_folder}': {len(excel_files)}")

# Urutkan berdasarkan tanggal jika formatnya sesuai
data_excel_list = []
if excel_files:
    # Filter hanya file dengan format tanggal DD_MM_YYYY
    dated_files = []
    for file in excel_files:
        if re.search(r'\d{2}_\d{2}_\d{4}', file):
            dated_files.append(file)
    
    if dated_files:
        dated_files.sort(key=lambda x: re.search(r'(\d{2})_(\d{2})_(\d{4})', x).groups()[::-1])
        for file in dated_files:
            excel_name = os.path.join(excel_list,file)
            data_excel_list.append(excel_name) 

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

    df["t_sec"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()
    df["dt_s"] = dt_s
    df["rod_speed"] = drod / dt_s           # %/s
    df["dpdt"] = dpwr / dt_s                # kW/s

    return df, pwr_col

all_df = []
all_pwr_col = []
for i in np.arange(0,len(data_excel_list),1):
    excel_name = data_excel_list[i]
    df_val, pwe_col = load_one_file(excel_name)
    all_df.append(df_val)
    all_pwr_col.append(pwe_col)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(all_df[0][TIME_COL], all_df[0]['Regulator Rod [%]'], label='Posisi Regulator Rod', color='blue')
plt.title('Posisi Regulator Rod')
plt.ylabel('Posisi (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Subplot 2: Kecepatan Rod
plt.subplot(2, 1, 2)
plt.plot(all_df[0]["t_sec"], all_df[0]["rod_speed"], label='Kecepatan Gerak', color='red')
plt.plot(all_df[1]["t_sec"], all_df[1]["rod_speed"], label='Kecepatan Gerak', color='blue')
plt.plot(all_df[2]["t_sec"], all_df[2]["rod_speed"], label='Kecepatan Gerak', color='black')
plt.plot(all_df[3]["t_sec"], all_df[3]["rod_speed"], label='Kecepatan Gerak', color='green')
plt.title('Kecepatan Gerak Regulator Rod')
plt.ylabel('Kecepatan (%/detik)')
plt.xlabel('Waktu')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()