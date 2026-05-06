import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import os
from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda
from rho_x_functions import drho_dx_gauss

cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd,file_delayed_neutron), 
                       index_col=None)

beta = np.sum(df_fdn["beta"].to_numpy())
group_mem = df_fdn["beta"].to_numpy()


v_rod = (v_percent/100) * H # units m/s

rho_abs = rho_max * beta          # absolut

pos_x = (pos_x_percent/100) * H

t_end_rod = pos_x_percent/v_percent

N = int(np.ceil(t_end / dt)) + 1
times = np.linspace(0.0, t_end, N)

def drho_dt_sin2(t, x, v, rho):
    drho = drho_dx_gauss(x, rho, rho_max, H) * v
    return drho

def dn_dt(n, t, rho_abs_t, beta, lamb, sum_lambda_ci):
    fun = ((rho_abs_t - beta) / lamb) * n + sum_lambda_ci
    return fun

def dc_dt(n, c_i, beta_i, lambda_i, Lambda):
    return (beta_i / Lambda) * n - lambda_i * c_i

#times = np.arange(0,t_end,dt)
pos_t = np.zeros_like(times)

rho_t = np.zeros_like(times)
rho_abs_t = np.zeros_like(times)

rho_t[0] = 0.0
rho_abs_t[0] = rho_t[0] * beta
n_t = np.zeros_like(times)

n_t[0] = 1.0

c_t = np.zeros((N, len(group_mem)))

for ci2 in range(len(group_mem)):
    beta_i = df_fdn.loc[ci2, "beta"]
    lam_i = df_fdn.loc[ci2, "lambda"]
    c_t[0, ci2] = (beta_i / (Lambda * lam_i)) * n_t[0]

for i in np.arange(1,len(times),1):
    delT = times[i]-times[i-1]
    pos_t[i] = pos_t[i-1] + delT * v_rod
       
    if times[i-1] < t_end_rod:
        v_now = v_rod
    else:
        v_now = 0.0
        pos_t[i] = pos_x
    #print(pos_t[i])
    rho_t[i] = rho_t[i-1] + delT * drho_dt_sin2(times[i-1], pos_t[i-1], v_now, rho_t[i-1])
    
    rho_abs_t[i] = rho_t[i] * beta
    
        
    sum_lambda_ci = 0
    for ci in np.arange(0,len(group_mem), 1):
        sum_lambda_ci += c_t[i-1, ci]*df_fdn.loc[ci,"lambda"]
    

        
    n_t[i] = n_t[i-1] + delT * dn_dt(n_t[i-1], times[i-1], rho_abs_t[i-1], beta, Lambda, sum_lambda_ci)
    
    for ci2 in np.arange(0,len(group_mem), 1):
        beta_i = df_fdn.loc[ci2,"beta"]
        lam_i = df_fdn.loc[ci2,"lambda"]
        c_t[i, ci2] = c_t[i-1, ci2] + delT * dc_dt(n_t[i-1], c_t[i-1, ci2], beta_i, lam_i, Lambda)

df_out_euler = pd.DataFrame({
    "time_s" : times,
    "rod_position_m" : pos_t,
    "rod_position_%" : 100 * pos_t / H,
    "rho_dollar" : rho_t,
    "rho_abs" : rho_abs_t,
    "neutron_density_n" : n_t
    })

#df_out.to_excel("hasil_simulasi_kartini_explicitEuler_origin.xlsx", index=False)
n_t1 = np.log(np.array(n_t)[~np.isnan(n_t)])