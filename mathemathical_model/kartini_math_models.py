import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fung_rho_sin2(t, rho, x, v, rho_max, H):
    C = (np.pi * rho_max) / (2*H)
    
    rho_t = C * (np.sin((np.pi * x) / H)**2) * v
    return rho_t

def euler_method_fung(t,n, rho_abs_t, beta, lamb, sum_lambda_ci):
    fun = ((rho_abs_t - beta) * n/lamb) + sum_lambda_ci
    return fun
    

df_fdn = pd.read_excel('fraction_delayed_neutrons_U235.xlsx', index_col=None)
Lambda = 4.0e-5
beta = np.sum(df_fdn["beta"].to_numpy())
group_mem = df_fdn["beta"].to_numpy()

H = 0.38 #units (meter)
rho_max = 1.95 # units dollar $

v_percent = 0.666242 # units (%/s)
v_rod = (v_percent/100) * H # units m/s

#beta = 0.007
rho_abs = rho_max * beta

pos_x_percent = 20 # units in %
pos_x = (pos_x_percent/100) * H

t_end = pos_x_percent/v_percent
dt = 0.01        
 
N = int(np.ceil(t_end / dt)) + 1
times = np.linspace(0.0, t_end, N)

rho_t0 = 0.0

#times = np.arange(0,t_end,dt)
pos_t = np.zeros_like(times)

rho_t = np.zeros_like(times)
rho_abs_t = np.zeros_like(times)
rho_t[0] = rho_abs_t[0] = rho_t0
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
    #print(pos_t[i])
    rho_t[i] = rho_t[i-1] + delT * fung_rho_sin2(times[i-1], rho_t[i-1], pos_t[i-1], v_rod, rho_max, H)
    rho_abs_t[i] = rho_t[i] * beta
    sum_lambda_ci = 0
    for ci in np.arange(0,len(group_mem), 1):
        sum_lambda_ci += c_t[i-1, ci]*df_fdn.loc[ci,"lambda"]
        
    n_t[i] = n_t[i-1] + delT * euler_method_fung(times[i], n_t[i], rho_abs_t[i], beta, Lambda, sum_lambda_ci)
    
    for ci2 in np.arange(0,len(group_mem), 1):
        beta_i = df_fdn.loc[ci2,"beta"]
        lam_i = df_fdn.loc[ci2,"lambda"]
        c_t[i, ci2] = c_t[i-1, ci2] + delT * ((beta_i/lam_i) * n_t[i-1] - lam_i*(c_t[i-1, ci2]*df_fdn.loc[ci2,"lambda"]))
        
        
    
print(n_t)
plt.figure()
plt.plot(times, rho_t)
plt.xlabel("Time (s)")
plt.ylabel("dollar")
plt.title("Regulating Rod Position vs Time")
plt.grid()
plt.show()

plt.figure()
plt.plot(times, n_t)
plt.xlabel("Time (s)")
plt.ylabel("dollar")
plt.title("Regulating Rod Position vs Time")
plt.grid()
plt.show()  