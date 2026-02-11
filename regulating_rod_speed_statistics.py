import pandas as pd
import os
import numpy as np

current_folder = os.getcwd()
excel_list = os.path.join(os.getcwd(),"data_operasi_reaktor2")
filename = 'Result_data_v.xlsx'

df = pd.read_excel(os.path.join(excel_list,filename))
coldf = df.columns.values

methods = ["diff", "forward", "central"]

def minimasi_variabilitas(df, methods):
    results = []
    
    for m in methods:
        v_up = df[f"v_up_mean_{m}"].to_numpy()
        sigma_up = df[f"v_up_std_{m}"].to_numpy()
        
        v_down = df[f"v_down_mean_{m}"].to_numpy()
        sigma_down = df[f"v_down_std_{m}"].to_numpy()
        
        v_up_all = np.mean(v_up)
        v_down_all = np.mean(v_down)
        sigma_up_med = np.mean(sigma_up)
        sigma_down_med = np.mean(sigma_down)
        
        R_up = sigma_up_med/ abs(v_up_all)
        R_down = sigma_down_med/ abs(v_down_all)
        
        results.append([m, v_up_all, sigma_up_med, R_up, v_down_all, sigma_down_med, R_down])
        
    R_df = pd.DataFrame(results,
                columns= ["method",
                          "v_up_all",
                          "sigma_up",
                          "R_up",
                          "v_down_all",
                          "sigma_down",
                          "R_down"
                    ])
    return R_df

def minimasi_variabilitas_robust(df, methods):
    results = []
    
    for m in methods:
        v_up = df[f"v_up_mean_{m}"].to_numpy()
        
        v_down = df[f"v_down_mean_{m}"].to_numpy()
        
        v_up_all = np.median(v_up)
        v_down_all = np.median(v_down)
        
        mad_up = np.median(np.abs(v_up - v_up_all))
        mad_down = np.median(np.abs(v_down - v_down_all))
        
        robust_up = 1.4826 * mad_up
        robust_down = 1.4826 * mad_down
        
        R_up = robust_up/ abs(v_up_all)
        R_down = robust_down/ abs(v_down_all)
        
        results.append([m, v_up_all, robust_up, R_up, v_down_all, robust_down, R_down])
        
    R_df = pd.DataFrame(results,
                columns= ["method",
                          "v_up_all",
                          "robust_up",
                          "R_up",
                          "v_down_all",
                          "robust_down",
                          "R_down"
                    ])
    return R_df

print(minimasi_variabilitas(df,methods))
print(minimasi_variabilitas_robust(df,methods))