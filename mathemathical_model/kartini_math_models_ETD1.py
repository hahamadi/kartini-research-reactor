import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def rho_update_etd1(rho_n, t_n, dt, x_n, v, rho_max, H):
    C = (np.pi * rho_max) / (2*H)
    g = C * v * (np.sin(np.pi * x_n / H)**2) - (1.0*rho_n)
    return g

def phi1(z):
    if abs(z) < 1e-8:
        return 1.0 + z/2.0 + z*z/6.0
    return (np.exp(z)-1.0)/z


def phi2(z):
    # stabil untuk z kecil
    if abs(z) < 1e-12:
        return 1.0
    return (1.0 - np.exp(z)) / z

def fung_temp_etd(T, t, n, alfa):
    return alfa * n

def temp_update_etd1(T, t, dt, T0, alfa, b, n):
    C = -1.0*b
    expo = np.exp(C*dt)
    phi = (expo - 1)/C
    #T1 = T0 + expo*(T - T0) + (1-expo)/b * (a*n)
    T1 = T0 + expo * (T - T0) + (phi * fung_temp_etd(T, t, n, alfa))
    return T1

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

pos_x_percent = 50 # units in %
pos_x = (pos_x_percent/100) * H

t_end = 50#pos_x_percent/v_percent
#print(t_end)
dt = 0.01      
 
N = int(np.ceil(t_end / dt)) + 1
times = np.linspace(0.0, t_end, N)

#times = np.arange(0,t_end,dt)
pos_t = np.zeros_like(times)

rho_t = np.zeros_like(times)
rho_abs_t = np.zeros_like(times)

rho_t[0] = 0.0 #rho_abs
rho_abs_t[0] = beta * rho_t[0]
#rho_net_abs[0] = rho_abs_t[0]

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
    ch = np.exp(1.0*delT)
    Cworth = (np.pi * rho_max) / (2*H)
    g_rho = Cworth * v_rod * (np.sin(np.pi * pos_t[i-1] / H)**2)
    rho_t[i] = (rho_t[i-1]*ch) + rho_update_etd1(rho_t[i-1], times[i-1], delT, pos_t[i-1], v_rod, rho_max, H)*(ch - 1)/1.0
    
    rho_abs_t[i] = rho_t[i] * beta
    # sum lambda_i c_i
    sum_lambda_ci = float(np.sum(c_t[i-1, :] * lam_vec))

    # Neutron ETD for n' = Cn*n + sum(lambda c)
    Cn = (rho_abs_t[i-1] - beta) / Lambda
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
plt.plot(times, n_t)
plt.xlabel("Time (s)")
plt.ylabel("n(t)")
plt.title("Number of neutrons vs Time")
plt.grid()
#plt.savefig(f"neutronVsTime_ETD1_feedback_h{dt}.png", dpi=300, bbox_inches='tight')
plt.show()
    
    