import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def first_big_jump_speed(t, x, jump_th=10.0):
    """
    Mencari lonjakan pertama yang signifikan.
    jump_th = ambang lonjakan (misal 10%)
    """

    t = np.asarray(t, float)
    x = np.asarray(x, float)

    for i in range(len(x)-1):
        dx = x[i+1] - x[i]

        if dx > jump_th:
            # titik awal lonjakan
            i_start = i

            # cari plateau atas (nilai maksimum lokal setelah lonjakan)
            # ambil nilai maksimum dalam window kecil berikutnya
            search_window = 10
            i_end = i+1
            xmax = x[i+1]

            for j in range(i+1, min(len(x), i+1+search_window)):
                if x[j] > xmax:
                    xmax = x[j]
                    i_end = j

            t0 = t[i_start]
            t1 = t[i_end]
            x0 = x[i_start]
            x1 = x[i_end]

            v = (x1 - x0) / (t1 - t0)

            return {
                "i_start": i_start,
                "i_end": i_end,
                "t0": t0,
                "t1": t1,
                "x0": x0,
                "x1": x1,
                "v": v
            }

    return None

# ======================
# CONTOH PAKAI
# ======================
current_folder = os.getcwd()
excel_list = os.path.join(os.getcwd(),"data_operasi_reaktor")

file_path = os.path.join(excel_list,"data_download_practice1_03_01_2022.xlsx")

df = pd.read_excel(file_path, sheet_name="Download Transaksi", header=1)

df.columns = df.columns.str.strip()

df["Time"] = pd.to_datetime(df["Time"], dayfirst=True)
df["time_s"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()

t = df["time_s"].to_numpy(float)
x = df["Regulator Rod [%]"].to_numpy(float)

result = first_big_jump_speed(t, x, jump_th=10)

print(result)