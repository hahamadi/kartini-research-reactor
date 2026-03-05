import pandas as pd
import numpy as np
from scipy.stats import linregress
import os
import re

current_folder = os.getcwd()
excel_list = os.path.join(os.getcwd(),"data_operasi_reaktor")

excel_files = []
for file in os.listdir(excel_list):
    if file.lower().endswith(('.xlsx', '.xls')):
        excel_files.append(file)

print(f"Jumlah file Excel di folder '{current_folder}': {len(excel_files)}")
# Urutkan berdasarkan tanggal jika formatnya sesuai
data_excel_list = []
date_list = []
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
            
print(data_excel_list[0], data_excel_list[-1])

def process_reactor_file(file_path, original_name):
    # Load data
    df = pd.read_excel(file_path, sheet_name="Download Transaksi", header=1)
    
    # Check if necessary column exists
    if 'Regulator Rod [%]' not in df.columns or 'Time' not in df.columns:
        return None
    
    # Preprocessing
    df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['Time', 'Regulator Rod [%]'])
    df['Seconds'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()
    
    # Segment Detection
    df['diff_pos'] = df['Regulator Rod [%]'].diff()
    df_moving = df[df['diff_pos'] != 0].copy()
    if df_moving.empty:
        return None
        
    df_moving['segment_id'] = (df_moving.index.to_series().diff() > 1).cumsum()
    
    summary_list = []
    for seg_id, group in df_moving.groupby('segment_id'):
        if len(group) >= 5: # Minimal 5 points
            slope, intercept, r_val, p_val, std_err = linregress(group['Seconds'], group['Regulator Rod [%]'])
            duration = group['Seconds'].iloc[-1] - group['Seconds'].iloc[0]
            velo = (group['Regulator Rod [%]'].iloc[-1] - group['Regulator Rod [%]'].iloc[0])/(round(group['Seconds'].iloc[-1], 2) - round(group['Seconds'].iloc[0], 2))
            if duration > 0:
                summary_list.append({
                    'File': original_name,
                    'Tipe': 'Naik' if slope > 0 else 'Turun',
                    'Waktu_Mulai_s': round(group['Seconds'].iloc[0], 2),
                    'Waktu_Akhir_s': round(group['Seconds'].iloc[-1], 2),
                    'Posisi_Awal_%': group['Regulator Rod [%]'].iloc[0],
                    'Posisi_Akhir_%': group['Regulator Rod [%]'].iloc[-1],
                    'Kecepatan_%_s': velo,
                    'R_Squared': round(r_val**2, 5),
                    'Jumlah_Titik': len(group)
                })
    
    if not summary_list:
        return None
        
    df_summary = pd.DataFrame(summary_list)
    
    # Pick best Up and best Down based on R2 and point count
    # Criteria: Filter R2 > 0.9, then sort by points
    best_segments = []
    
    for t in ['Naik', 'Turun']:
        subset = df_summary[df_summary['Tipe'] == t]
        if not subset.empty:
            # Sort by R2 descending, then by number of points descending
            best = subset.sort_values(by=['R_Squared', 'Jumlah_Titik'], ascending=False).head(1)
            best_segments.append(best)
            
    if best_segments:
        return pd.concat(best_segments)
    return None



files = [(os.path.split(i)) for i in data_excel_list]

all_results = []
for path, name in files:
    path2 = os.path.join(path,name)
    name2 = re.search(r'\d{2}_\d{2}_\d{4}', name)
    name2 = name2.group()
    res = process_reactor_file(path2, name2)
    if res is not None:
        all_results.append(res)

if all_results:
    final_df = pd.concat(all_results)
    res_name = os.path.join(os.getcwd(),"data_operasi_reaktor2")
    val_name = os.path.join(res_name,"Result_data_v_V2.xlsx")
    final_df.to_excel(val_name, index=False)
    print(final_df)
else:
    print("No valid movement data found.")