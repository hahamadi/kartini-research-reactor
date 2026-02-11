import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

current_folder = os.getcwd()
excel_list = os.path.join(os.getcwd(),"data_operasi_reaktor")

TIME_COL = "Time"
ROD_COL = "Regulator Rod [%]"

# List semua file Excel
excel_files = []
for file in os.listdir(excel_list):
    if file.lower().endswith(('.xlsx', '.xls')):
        excel_files.append(file)

print(f"Jumlah file Excel di folder '{current_folder}': {len(excel_files)}")

data_excel_list = []
excel_files_short = []
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
            excel_files_short.append(file)
            data_excel_list.append(excel_name)

print(f"nama file pertama '{excel_files_short[0]}', nama file terakhir '{excel_files_short[-1]}'.")
        
def load_one_file(path):
    df = pd.read_excel(path, sheet_name="Download Transaksi", header=1)
    head, filename = os.path.split(path)
    
    date_part = filename.replace(".xlsx", "").split("_")[-3:]

    date_str = "-".join(date_part)
    # parse time
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)


    # numeric conversion
    df[ROD_COL] = pd.to_numeric(df[ROD_COL], errors="coerce")

    # dt & derivatives
    dt_s = df[TIME_COL].diff().dt.total_seconds()
    drod = df[ROD_COL].diff()

    df["Time_sec"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()
    df["dt_s"] = dt_s
    df["v_reg_diff"] = drod / dt_s
    
    v_up = [i for i in df["v_reg_diff"] if i > 0]
    v_down = [i for i in df["v_reg_diff"] if i < 0]
    v_up_mean = np.mean(v_up)
    v_down_mean = np.mean(v_down)
    
    v_up_std = np.std(v_up)#/np.sqrt(len(v_up))
    v_down_std = np.std(v_down)#/np.sqrt(len(v_down))
    return df, [date_str, v_up_mean, v_up_std, v_down_mean, v_down_std]

def forward_central_diff(path):
    head, filename = os.path.split(path)
    
    date_part = filename.replace(".xlsx", "").split("_")[-3:]

    date_str = "-".join(date_part)
    
    df = pd.read_excel(path, sheet_name="Download Transaksi", header=1)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    df[ROD_COL] = pd.to_numeric(df[ROD_COL], errors="coerce")
    x = df[ROD_COL].values
    
    df["Time_sec"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()
    t = df["Time_sec"].values
    
    v_forward = np.empty_like(x, dtype=float)
    v_forward[0] = 0
    v_forward[1:] = (x[1:] - x[:-1]) / (t[1:] - t[:-1])
    df["v_reg_forward"] = v_forward
    v_up_forward = [i for i in df["v_reg_forward"] if i > 0]
    v_down_forward = [i for i in df["v_reg_forward"] if i < 0]
    v_up_mean_forward = np.mean(v_up_forward)
    v_down_mean_forward = np.mean(v_down_forward)
    
    v_up_std_forward = np.std(v_up_forward)#/np.sqrt(len(v_up_forward))
    v_down_std_forward = np.std(v_down_forward)#/np.sqrt(len(v_down_forward))
    val_forward = [date_str, v_up_mean_forward, v_up_std_forward, 
                   v_down_mean_forward, v_down_std_forward]
    
    v_central = np.empty_like(x, dtype=float)
    v_central[:] = 0
    v_central[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
    df["v_reg_central"] = v_central
    
    v_up_central = [i for i in df["v_reg_central"] if i > 0]
    v_down_central = [i for i in df["v_reg_central"] if i < 0]
    
    v_up_mean_central = np.mean(v_up_central)
    v_down_mean_central = np.mean(v_down_central)
    
    v_up_std_central = np.std(v_up_central)#/np.sqrt(len(v_up_central))
    v_down_std_central = np.std(v_down_central)#/np.sqrt(len(v_down_central))
    val_central = [date_str, v_up_mean_central, v_up_std_central, 
                   v_down_mean_central, v_down_std_central]
    return df, val_forward, val_central

def plot_range_with_errors(df, method, ax=None, title=None):
    """
    method: "diff" / "forward" / "central"
    Kolom yang dibutuhkan:
      v_down_mean_{method}, v_down_std_{method},
      v_up_mean_{method},   v_up_std_{method}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 10))

    # pastikan date rapi
    dff = df.copy()
    dff["date"] = pd.to_datetime(dff["date"], dayfirst=True)
    dff = dff.sort_values("date")

    down_mean = dff[f"v_down_mean_{method}"].to_numpy(float)
    up_mean   = dff[f"v_up_mean_{method}"].to_numpy(float)
    down_std  = dff[f"v_down_std_{method}"].to_numpy(float)
    up_std    = dff[f"v_up_std_{method}"].to_numpy(float)

    x = np.arange(len(dff))
    height = up_mean - down_mean  # tinggi batang
    labels = dff["date"].dt.strftime("%d_%m_%Y").to_list()

    # batang range (dari down_mean ke up_mean)
    ax.bar(x, height, bottom=down_mean, width=0.7)

    # errorbar bawah (di titik down_mean)
    ax.errorbar(
    x, down_mean, yerr=down_std,
    fmt="none", capsize=5,
    ecolor="black", elinewidth=1.2)

    # errorbar atas (di titik up_mean)
    ax.errorbar(
        x, up_mean, yerr=up_std,
        fmt="none", capsize=5,
        ecolor="black", elinewidth=1.2
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Date")
    ax.set_ylabel("Kecepatan (%/detik)")
    ax.set_title(title or f"Variabilitas Kecepatan Regulator Rod ({method})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    return ax

def running_v(data, index=5):
    col_df_val = ["date", "v_up_mean_diff", "v_up_std_diff", "v_down_mean_diff",
                  "v_down_std_diff", "v_up_mean_forward","v_up_std_forward",
                  "v_down_mean_forward", "v_down_std_forward", 
                  "v_up_mean_central", "v_up_std_central", "v_down_mean_central", 
                  "v_down_std_central"]
    df = pd.DataFrame(columns=col_df_val)
    for i in range(index):
        df1, val1 = load_one_file(data[i])
        df2, val_forward, val_central = forward_central_diff(data[i])
        val = val1 + val_forward[1:] + val_central[1:]
        if len(col_df_val) == len(val):
            for j in range(len(col_df_val)):
                df.loc[i,col_df_val[j]] = val[j]
    return df
    
df = running_v(data_excel_list, index=len(data_excel_list))
folder_results = os.path.join(os.getcwd(),"data_operasi_reaktor2")
df.to_excel(os.path.join(folder_results,'Result_data_v.xlsx'), index=False)

fig, axes = plt.subplots(1, 1, figsize=(18, 10), sharey=True)

plot_range_with_errors(df, "diff", ax=axes, title="Metode diff")
#plot_range_with_errors(df, "forward", ax=axes[1], title="Metode forward")
#plot_range_with_errors(df, "central", ax=axes[2], title="Metode central")

plt.tight_layout()
plt.show()
"""
plt.figure()
plt.plot(df1["Time_sec"], df1["Regulator Rod [%]"])
plt.xlabel("Time (s)")
plt.ylabel("Regulator position (%)")
plt.title("Regulating Rod Position vs Time")
plt.grid()
plt.show()

plt.figure()
plt.plot(df1["Time_sec"], df1["v_reg_diff"])
plt.plot(df1["Time_sec"], df2["v_reg_forward"])
plt.plot(df1["Time_sec"], df2["v_reg_central"])
plt.xlabel("Time (s)")
plt.ylabel("Rod speed (%/s)")
plt.title("Regulating Rod Speed vs Time")
plt.grid()
plt.show()
"""