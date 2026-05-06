import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd,file_delayed_neutron), 
                       index_col=None)
Lambda = 4.3e-5

beta = np.sum(df_fdn["beta"].to_numpy())
group_mem = df_fdn["beta"].to_numpy()

H = 0.38 #units (meter)
rho_max = 1.95 # units dollar $

v_percent = 1.6579 # units (%/s)
v_rod = (v_percent/100) * H # units m/s

rho_abs = rho_max * beta          # absolut

pos_x_percent = 35 # units in %
pos_x = (pos_x_percent/100) * H

def drho_dx_polynomial(x):
    drho = -129.16*5*x**4 + 279.95*4*x**3 - 215.04*3*x**2 + 58.294*2*x + 1.3702
    return drho

def drho_dt_polynomial(t,x,v):
    drho = drho_dx_polynomial(x) * v
    return drho

def rho_t_rk4(h, t, x, v):
    
    k1 = drho_dt_polynomial(t, x, v)
    k2 = drho_dt_polynomial(t + 0.5*h, x + 0.5*h*k1, v)
    k3 = drho_dt_polynomial(t + 0.5*h, x + 0.5*h*k2, v)
    k4 = drho_dt_polynomial(t + h, x + h*k3, v)
    
    xn = x + (h/6)*(k1 + 2*k2 + 2*k3 +k4)
    return xn

def dn_dt(n, t, rho_abs_t, beta, lamb, sum_lambda_ci):
    fun = ((rho_abs_t - beta) / lamb) * n + sum_lambda_ci
    return fun

def n_t_rk4(h, n, t, rho_abs_t, beta, lamb, sum_lambda_ci):
    
    k1 = dn_dt(n, t, rho_abs_t, beta, lamb, sum_lambda_ci)
    k2 = dn_dt(n + 0.5*h*k1 , t + 0.5*h , rho_abs_t, beta, lamb, sum_lambda_ci)
    k3 = dn_dt(n + 0.5*h*k2 , t + 0.5*h , rho_abs_t, beta, lamb, sum_lambda_ci)
    k4 = dn_dt(n + h*k3 , t + h , rho_abs_t, beta, lamb, sum_lambda_ci)
    
    n1 = n + (h/6)*(k1 + 2*k2 + 2*k3 +k4) 
    return n1

def dc_dt(t, n, c_i, beta_i, lambda_i, Lambda):
    return (beta_i / Lambda) * n - lambda_i * c_i

def c_t_rk4(h, t, n, c_i, beta_i, lambda_i, Lambda):
    
    k1 = dc_dt(t, n, c_i, beta_i, lambda_i, Lambda)
    k2 = dc_dt(t + 0.5*h, n, c_i + 0.5*h*k1, beta_i, lambda_i, Lambda)
    k3 = dc_dt(t + 0.5*h, n, c_i + 0.5*h*k2, beta_i, lambda_i, Lambda)
    k4 = dc_dt(t + h, n, c_i + h*k3, beta_i, lambda_i, Lambda)
    
    c1 = c_i + (h/6)*(k1 + 2*k2 + 2*k3 +k4)
    return c1

t_end = pos_x_percent/v_percent #6
t_end_rod = pos_x_percent/v_percent

dt = 0.01       
 
N = int(np.ceil(t_end / dt)) + 1
times = np.linspace(0.0, t_end, N)

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
    rho_t[i] = rho_t_rk4(delT, times[i-1], pos_t[i-1], v_now)
    
    rho_abs_t[i] = rho_t[i] * beta
    
        
    sum_lambda_ci = 0
    for ci in np.arange(0,len(group_mem), 1):
        sum_lambda_ci += c_t[i-1, ci]*df_fdn.loc[ci,"lambda"]
    
        
    n_t[i] = n_t_rk4(delT, n_t[i-1], times[i-1], rho_abs_t[i-1], beta, Lambda, sum_lambda_ci)
    
    for ci2 in np.arange(0,len(group_mem), 1):
        beta_i = df_fdn.loc[ci2,"beta"]
        lam_i = df_fdn.loc[ci2,"lambda"]
        c_t[i, ci2] = c_t_rk4(delT, times[i-1], n_t[i-1], c_t[i-1, ci2], beta_i, lam_i, Lambda)

df_out = pd.DataFrame({
    "time_s" : times,
    "rod_position_m" : pos_t,
    "rod_position_%" : 100 * pos_t / H,
    "rho_dollar" : rho_t,
    "rho_abs" : rho_abs_t,
    "neutron_density_n" : n_t
    })

#df_out.to_excel("hasil_simulasi_kartini_explicitEuler_origin.xlsx", index=False)
n_t1 = [np.log(y) for y in n_t if y != np.nan]
 
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