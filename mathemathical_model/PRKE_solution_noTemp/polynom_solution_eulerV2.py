import numpy as np

import pandas as pd

from simulation_config import H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda
from simulation_config import beta_groups, lambda_groups, beta
from rho_x_functions import rho_polynomial


def rho_t_polynomial(t,x,v):
    drho = rho_polynomial(x) * v
    return drho

def dn_dt(n, t, rho_abs_t, beta, lamb, sum_lambda_ci):
    fun = ((rho_abs_t - beta) / lamb) * n + sum_lambda_ci
    return fun

def dc_dt(n, c_i, beta_i, lambda_i, Lambda):
    return (beta_i / Lambda) * n - lambda_i * c_i

def run_simulation_prke(rho_polynomial, H, rho_max, v_percent, pos_x_percent, t_end, dt, 
                        Lambda):
    v_rod = (v_percent/100) * H
    rho_abs = rho_max * beta
    pos_x = (pos_x_percent/100) * H
    
    t_end_rod = pos_x_percent/v_percent
    
    times = np.arange(0.0, t_end + dt, dt)
    N = len(times)

    pos_t = np.zeros_like(times)
    
    rho_t = np.zeros_like(times)
    rho_abs_t = np.zeros_like(times)
    
    rho_t[0] = 0.0
    rho_abs_t[0] = rho_t[0] * beta
    n_t = np.zeros_like(times)
    n_t[0] = 1.0
    c_t = np.zeros((N, len(beta_groups)))
    
    for ci2 in range(len(beta_groups)):
        beta_i = beta_groups[ci2]
        lam_i = lambda_groups[ci2]
        c_t[0, ci2] = (beta_i / (Lambda * lam_i)) * n_t[0]
    
    for i in np.arange(1,len(times),1):
        delT = times[i]-times[i-1]
        v_now = v_rod if times[i-1] < t_end_rod else 0.0           
                  
        
        pos_t[i] = pos_t[i-1] + delT * v_now
        if pos_t[i] > pos_x:
            pos_t[i] = pos_x
        #print(pos_t[i])
        rho_t[i] = rho_polynomial(pos_t[i])
        
        rho_abs_t[i] = rho_t[i] * beta
        
            
        sum_lambda_ci = 0
        for ci in np.arange(0,len(beta_groups), 1):
            sum_lambda_ci += c_t[i-1, ci]*lambda_groups[ci]
        

            
        n_t[i] = n_t[i-1] + delT * dn_dt(n_t[i-1], times[i-1], rho_abs_t[i-1], beta, Lambda, sum_lambda_ci)
        
        for ci2 in np.arange(0,len(beta_groups), 1):
            beta_i = beta_groups[ci2]
            lam_i = lambda_groups[ci2]
            c_t[i, ci2] = c_t[i-1, ci2] + delT * dc_dt(n_t[i-1], c_t[i-1, ci2], beta_i, lam_i, Lambda)
    df_out_euler = pd.DataFrame({
        "time_s" : times,
        "rod_position_m" : pos_t,
        "rod_position_%" : 100 * pos_t / H,
        "rho_dollar" : rho_t,
        "rho_abs" : rho_abs_t,
        "neutron_density_n" : n_t
        })
    return df_out_euler

df_out = run_simulation_prke(rho_polynomial, H, rho_max, v_percent, pos_x_percent, t_end, dt, Lambda)
"""
v_rod = (v_percent/100) * H # units m/s

rho_abs = rho_max * beta          # absolut

pos_x = (pos_x_percent/100) * H

t_end_rod = pos_x_percent/v_percent

N = int(np.ceil(t_end / dt)) + 1
times = np.linspace(0.0, t_end, N)




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
    rho_t[i] = rho_polynomial(pos_t[i])
    
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


#print(n_t)
plt.figure()
plt.plot(times, rho_t)
plt.xlabel("Time (s)")
plt.ylabel("Reactivity ($)")
plt.title("Reactivity vs Time")
plt.grid()
#plt.savefig("reactivityVsTime_explicitEuler.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(times, n_t)
plt.xlabel("Time (s)")
plt.ylabel("n(t)")
plt.title("Number of neutrons vs Time")
plt.grid()
#plt.savefig("neutronVsTime_explicitEuler.png", dpi=300, bbox_inches='tight')
plt.show()
"""