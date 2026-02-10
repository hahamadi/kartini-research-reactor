import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_folder = os.getcwd()
excel_list = os.path.join(os.getcwd(),"data_operasi_reaktor")

excel_files = []
for file in os.listdir(excel_list):
    if file.lower().endswith(('.xlsx', '.xls')):
        excel_files.append(file)

path = os.path.join(excel_list,"data_download_practice1_03_12_2025.xlsx")  
print(path)
df = pd.read_excel(path, sheet_name="Download Transaksi", header=1)

df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df["Time_s"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()

t = df["Time_s"].to_numpy()
x = df["Regulator Rod [%]"].to_numpy()
#df["time_s"] = (pd.to_datetime(df["Time"]) - pd.to_datetime(df["Time"]).iloc[0]).dt.total_seconds()

v_forward = np.empty_like(x, dtype=float)
v_forward[0] = np.nan
v_forward[1:] = (x[1:] - x[:-1]) / (t[1:] - t[:-1])

v_central = np.empty_like(x, dtype=float)
v_central[:] = np.nan
v_central[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])

df["v_reg_forward"] = v_forward
df["v_reg_central"] = v_central

v_reg_forward = [x for x in v_forward if np.isnan(x) == False]
v_reg_central = [x for x in v_central if np.isnan(x) == False]


print(max(v_reg_forward),min(v_reg_forward))
print(max(v_reg_central),min(v_reg_central))

plt.figure()
plt.plot(df["Time_s"], df["Regulator Rod [%]"])
plt.xlabel("Time (s)")
plt.ylabel("Regulator position (%)")
plt.title("Regulating Rod Position vs Time")
plt.grid()
plt.show()

# 2. Kecepatan vs waktu
plt.figure()
plt.plot(df["Time_s"], df["v_reg_central"], df["Time_s"], df["v_reg_forward"])
plt.xlabel("Time (s)")
plt.ylabel("Rod speed (%/s)")
plt.title("Regulating Rod Speed vs Time (Baseline)")
plt.grid()
plt.show()


# 3. Histogram kecepatan (ambil nilai saat bergerak)
v_move = df["v_reg_central"].abs()
v_move = v_move[v_move > 0.01]

plt.figure()
plt.hist(v_move, bins=40)
plt.xlabel("Rod speed (%/s)")
plt.ylabel("Frequency")
plt.title("Distribution of Regulating Rod Speed")
plt.grid()
plt.show()