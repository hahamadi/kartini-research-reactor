import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fung_rho(rho_n, t_n, x_n):
    g = -129.16*(x_n**5) + 279.95*(x_n**4) - 215.04*(x_n**3) + 58.294*(x_n**2) + 1.3702*x_n - 0.0029
    return g

def fung_rho_update_RK4(rho_n, t_n, dt, x_n, v, rho_max, H):
    
    k1 = fung_rho(rho_n, t_n, x_n)
    k2 = fung_rho(rho_n + (dt/2)*k1, t_n + dt/2, x_n)
    k3 = fung_rho(rho_n + (dt/2)*k2, t_n + dt/2, x_n)
    k4 = fung_rho(rho_n + dt*k3, t_n + dt, x_n)
    
    y_n = rho_n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y_n

def phi1(z):
    if abs(z) < 1e-8:
        return 1.0 + z/2.0 + z*z/6.0
    return (np.exp(z)-1.0)/z


def phi2(z):
    # stabil untuk z kecil
    if abs(z) < 1e-12:
        return 1.0
    return (1.0 - np.exp(z)) / z

def fung_temp(T_n, t_n, n_t, T0, a, b):
    return a*n_t + b*T0

def fung_temp_update_etd1(T_n, t_n, dt, n_t, T0, a, bet):
    
    ch = np.exp(-bet*dt)
    yn = (T_n*ch) + (fung_temp(T_n, t_n, n_t, T0, a, bet)*(ch - 1)/-bet)
    
    return yn

def fung_n_etd(N, t, dt, Ci, Li):
    val = np.sum(Ci*Li)
    return val

def n_update_etd1(n, t, dt, rho_t, beta, Lambda, Ci, Li):
    C = (rho_t - beta)/Lambda
    expo = np.exp(C*dt)
    phi = (expo-1)/C
    
    n1 = (n*expo) + (phi * fung_n_etd(n, t, dt, Ci, Li))
    return n1

df_fdn = pd.read_excel('fraction_delayed_neutrons_U235.xlsx', index_col=None)
Lambda = 4.0e-5
beta = np.sum(df_fdn["beta"].to_numpy())
group_mem = df_fdn["beta"].to_numpy()

H = 0.38 #units (meter)
rho_max = 1.95 # units dollar $

v_percent = 1.49 # units (%/s)
v_rod = (v_percent/100) * H # units m/s

#beta = 0.007
rho_abs = rho_max * beta          # absolut

pos_x_percent = 30 # units in %
pos_x = (pos_x_percent/100) * H

t_end = 1000 #pos_x_percent/v_percent
#print(t_end)
dt = 0.01      
 
N = int(np.ceil(t_end / dt)) + 1
times = np.linspace(0.0, t_end, N)

T0 = 300 #initial temperature (K)
alpha_T_abs_per_K = 4e-5   # reaktivitas absolut per K (mulai dari 5e-5 s/d 2e-4)
a_K_per_s_at_n1 = 0.03     # K/s saat n=1 (pemanasan)
b_1_per_s = 0.01

T = np.zeros(N)
T[0] = T0

#times = np.arange(0,t_end,dt)
pos_t = np.zeros_like(times)

rho_t = np.zeros_like(times)
rho_abs_t = np.zeros_like(times)
rho_net_abs = np.zeros(N)

rho_t[0] = 0.0 #rho_abs
rho_abs_t[0] = beta * rho_t[0]
rho_net_abs[0] = rho_abs_t[0]

n_t = np.zeros_like(times)
n_t[0] = 1.0

c_t = np.zeros((N, len(group_mem)))

for ci2 in range(len(group_mem)):
    beta_i = df_fdn.loc[ci2, "beta"]
    lam_i = df_fdn.loc[ci2, "lambda"]
    c_t[0, ci2] = (beta_i / (Lambda * lam_i)) * n_t[0]

lam_vec = df_fdn["lambda"].to_numpy(dtype=float)
beta_vec = df_fdn["beta"].to_numpy(dtype=float)


for i in range(1, len(times)):
    delT = times[i] - times[i-1]
    pos_t[i] = pos_t[i-1] + delT * v_rod
    #pos_t[i] = min(H, pos_t[i-1] + delT * v_rod)
    if pos_t[i-1] >= pos_x:
        pos_t[i] = pos_x
        v_rod = 0

    # rho in $ (no linear part -> Euler is ok)
    #rho_t[i] = fung_rho_update_RK4(rho_t[i-1], times[i-1], delT, pos_t[i-1], v_rod, rho_max, H)
    rho_t[i] = fung_rho(rho_t[i-1], times[i-1], pos_t[i-1])
        
    rho_abs_t[i] = rho_t[i] * beta
    
    T[i] = fung_temp_update_etd1(T[i-1], times[i-1], delT, n_t[i-1], T0, a_K_per_s_at_n1, b_1_per_s)
    expoT = np.exp(-b_1_per_s*delT)
    #T[i] = T0 + expoT*(T[i-1] - T0) + (1-expoT)/b_1_per_s * (a_K_per_s_at_n1*n_t[i-1])
    
    rho_net_abs[i] = rho_abs_t[i] - alpha_T_abs_per_K * (T[i]-T0)
    # sum lambda_i c_i
    sum_lambda_ci = float(np.sum(c_t[i-1, :] * lam_vec))

    # Neutron ETD for n' = Cn*n + sum(lambda c)
    Cn = (rho_net_abs[i-1] - beta) / Lambda
    z = delT * Cn
    expoN = np.exp(z)
    n_t[i] = n_t[i-1]*expoN + delT * phi1(z) * sum_lambda_ci

    # Precursors ETD
    for j in range(len(beta_vec)):
        lam_i = lam_vec[j]
        beta_i = beta_vec[j]
        expoC = np.exp(-lam_i * delT)
        c_t[i, j] = expoC*c_t[i-1, j] + (1.0 - expoC)/lam_i * ((beta_i/Lambda) * n_t[i-1])

df_out = pd.DataFrame({
    "time_s" : times,
    "rod_position_m" : pos_t,
    "rod_position_%" : 100 * pos_t / H,
    "rho_dollar" : rho_t,
    "rho_absolute_t" : rho_abs_t,
    "neutron_density_n" : n_t,
    })

#df_out.to_excel(f"hasil_simulasi_kartini_ETD1_h{dt}.xlsx", index=False)

plt.figure()
plt.plot(times, rho_t)
plt.xlabel("Time (s)")
plt.ylabel("Reactivity ($)")
plt.title("Reactivity vs Time")
plt.grid()
#plt.savefig(f"reactivityVsTime_ETD1_feedback_h{dt}.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(times, T)
plt.xlabel("Time (s)")
plt.ylabel("Fuel temperature T (K)")
plt.title("Temperature vs Time")
plt.grid(True)
#plt.savefig(f"TemperatureVsTime_ETD1_feedback_h{dt}.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(times, rho_abs_t, label="rho_rod_abs")
plt.plot(times, rho_net_abs, label="rho_net_abs (with feedback)")
plt.xlabel("Time (s)")
plt.ylabel("Reactivity (absolute)")
plt.title("Rod vs Net Reactivity")
plt.grid(True)
plt.legend()
#plt.savefig(f"ReactivityAllVsTime_ETD1_feedback_h{dt}.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(times, n_t)
plt.xlabel("Time (s)")
plt.ylabel("n(t)")
plt.title("Number of neutrons vs Time")
plt.grid()
#plt.savefig(f"neutronVsTime_ETD1_feedback_h{dt}.png", dpi=300, bbox_inches='tight')
plt.show()
    
    