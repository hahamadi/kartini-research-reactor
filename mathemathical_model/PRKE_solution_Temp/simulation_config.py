import os
import numpy as np
import pandas as pd


cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd, file_delayed_neutron), index_col=None)
beta_groups  = df_fdn["beta"].to_numpy(dtype=float)
lambda_groups = df_fdn["lambda"].to_numpy(dtype=float)
beta = float(np.sum(beta_groups))

H = 0.38 #units (meter)
rho_max = 1.95 # units dollar $

v_percent = 1.6579 # units (%/s)
v_rod = v_percent * H / 100 # units in meter/second

pos_x_percent = 50 # units in %
pos_x = (pos_x_percent/100) * H # units in meter

dt = 0.010 # units in seconds
t_end = 200 #  #units in seconds
t_rod_end = pos_x_percent/v_percent
times = np.arange(0, t_end, dt)

Lambda = 4.3e-5

T0 = 300 # units in Kelvin
alpha_T_abs_per_K = 5.49e-5 # units in 1/Kelvin
beta_T = 0.015 # units 1/second
kappa_T = 1.37e-5 # units in 1/Kelvin

rho_abs = 0.0

print(f"""Coeficient used in Point Reactor Kinetics Equation (PRKE) of Kartini Reactor:  
    H = {H} m,
    rho_max = {rho_max} $,
    v_percent = {v_percent} %/s,
    pos_x_percent = {pos_x_percent} %,
    pos_x = {pos_x:.3f} m,
    t_end = {t_end} s,
    dt = {dt} s,
    Lambda = {Lambda},
    T0 = {T0} K,
    alpha_T_abs_per_K = {alpha_T_abs_per_K} 1/K,
    beta_T = {beta_T} 1/s,
    kappa_T = {kappa_T} 1/K
    """)