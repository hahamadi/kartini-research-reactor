import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

current_folder = os.getcwd()
excel_list = os.path.join(os.getcwd(),"data_operasi_reaktor2")
filename = 'Result_data_v.xlsx'

df = pd.read_excel(os.path.join(excel_list,filename))

# pastikan kolom date dalam format datetime
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

plt.figure(figsize=(12,6))

plt.plot(df["date"], df["v_up_mean_diff"], label="Diff Method", linewidth=1.5)
plt.plot(df["date"], df["v_up_mean_forward"], label="Forward Method", linewidth=1.5)
plt.plot(df["date"], df["v_up_mean_central"], label="Central Method", linewidth=1.5)

plt.xlabel("Date")
plt.ylabel("Control Rod Speed (cm/s)")
plt.title("Average Upward Control Rod Speed vs Date")

# format tanggal agar tidak bertumpuk
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()