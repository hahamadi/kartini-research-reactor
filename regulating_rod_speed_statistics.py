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
        
        results.append([m, v_up_all, robust_up, R_up, v_down_all, robust_down,
                        R_down])
        
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

def multi_criteria_decission(df, methods):
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
        
        var_up = np.var(v_up)
        var_down = np.var(v_down)
        
        stab_rat_up = robust_up/ abs(v_up_all)
        stab_rat_down = robust_down/ abs(v_down_all)
        
        w1, w2, w3 = 0.4, 0.4, 0.2
        J_up = w1 * robust_up + w2 * stab_rat_up + w3 * var_up
        J_down = w1 * robust_down + w2 * stab_rat_down + w3 * var_down
        res = [m, v_up_all, robust_up, stab_rat_up, var_up, J_up,
               v_down_all, robust_down, stab_rat_down, var_down, J_down]
        results.append(res)
        
    R_df = pd.DataFrame(results,
                columns= ["method",
                          "v_up_all",
                          "robust_up",
                          "Ratio_up",
                          "Variance_up",
                          "J_up",
                          "v_down_all",
                          "robust_down",
                          "Ratio_down",
                          "Variance_down",
                          "J_down"
                    ])
    return R_df
print(minimasi_variabilitas(df,methods))
print(minimasi_variabilitas_robust(df,methods))
print(multi_criteria_decission(df,methods))