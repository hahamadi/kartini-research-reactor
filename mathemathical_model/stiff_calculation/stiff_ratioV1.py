import numpy as np
from numpy.linalg import eigvals
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import pandas as pd

cwd = os.getcwd()
main_cwd = os.path.split(cwd)[0]
file_delayed_neutron = 'fraction_delayed_neutrons_U235.xlsx'
df_fdn = pd.read_excel(os.path.join(main_cwd,file_delayed_neutron), 
                       index_col=None)

beta_i = df_fdn["beta"].to_numpy()
beta = np.sum(beta_i)
lam_i = df_fdn["lambda"].to_numpy()

print()
Lambda = 6.0e-5

def x_of_t_linear(t, x0=0.0, v):
    return x0 + v*t

def rho_polynomial(x):
    g = -129.16*(x**5) + 279.95*(x**4) - 215.04*(x**3) + 58.294*(x**2) + 1.3702*x - 0.0029
    return g

def rho_diff_worth(x, rho_tot, H):
    v = (rho_tot/4)*((np.pi*x/H) - (0.5*np.sin(2*np.pi*x/H)))
    return val