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

pos_x_percent = 40 # units in %
pos_x = (pos_x_percent/100) * H

dt = 0.010
t_end = 26 #200 #pos_x_percent/v_percent #22.0 #pos_x_percent/v_percent

times = np.arange(0, t_end, dt)

print(f"t end {t_end} second")
print(f"dt end {dt}")
Lambda = 4.3e-5

print(f"""H = {H} m
      rho_max = {rho_max} $/m
      v_percent = {v_percent} %/s
      pos_x_percent = {pos_x_percent} %
      pos_x = {pos_x:.3f} m
      dt = {dt} s
      t_end = {t_end} s
      Lambda = {Lambda} 1/s""")