import pandas as pd
import os
import matplotlib.pyplot as plt

current_folder = os.getcwd()
excel_list = os.path.join(os.getcwd(),"data_operasi_reaktor")
filename = 'Result_data_v.xlsx'

df = pd.read_excel(os.path.join(excel_list,filename))
coldf = df.columns.values

plt.figure()
plt.plot(df["date"], df["v_up_mean_diff"], ".-")
plt.xlabel("date")
plt.ylabel("v up")
plt.title("V Up vs date")
plt.grid()
plt.show()
