import numpy as np
#import pandas as pd
#import os
#import matplotlib.pyplot as plt
from scipy.special import erf
import simulation_config as config

H = config.H
rho_max = config.rho_max

dx = 0.01

xspan = np.arange(0, H, dx)

def fung_temp(T_n, t_n, n_t, T0, kappa_T, beta_T):
    return kappa_T*n_t + beta_T*T0

def rho_polynomial(x):
    rho = -129.16*x**5 + 279.95*x**4 - 215.04*x**3 + 58.294*x**2 + 1.3702*x -0.0029
    return rho

def drho_dx_sin2(x, rho, rho_total, H):
    C = (2 * rho_max) / (H)
    fx = C * np.sin((np.pi * x) / H)**2
    return fx

def drho_dx_gauss(x, rho, rho_total, H):
    beta_gauss = 0.38
    alfa_gauss = 0.38
    x0 = H*alfa_gauss
    sigma = (beta_gauss*H)
    Wmax = rho_total / (H*beta_gauss*np.sqrt(0.5*np.pi)*(erf((1-alfa_gauss)/(np.sqrt(2)*beta_gauss)) - erf((-alfa_gauss)/(np.sqrt(2)*beta_gauss))))
    fx = Wmax * np.exp(-((x - x0)**2)/(2 * sigma**2))
    return fx

def drho_dx_sigmoid(x, rho, rho_total, H):
    rho0 = 0.0
    alfa_sigmoid = 0.385
    x0 = alfa_sigmoid*H
    k_sigmoid = 35
    fx = (k_sigmoid * (rho_total - rho0)*np.exp(-1.0*k_sigmoid*(x - x0)))/((1 + np.exp(-1.0*k_sigmoid*(x - x0)))**2)
    return fx

def runge_kutta(fung, x, dx, rho_max, H):

    rho = np.zeros(len(x))
    rho[0] = 0.0
    
    for i in np.arange(1, len(x)):
        k1 = fung(x[i-1], rho[i-1], rho_max, H)
        k2 = fung(x[i-1] + dx/2, rho[i-1] + dx/2 * k1, rho_max, H)
        k3 = fung(x[i-1] + dx/2, rho[i-1] + dx/2 * k2, rho_max, H)
        k4 = fung(x[i-1] + dx, rho[i-1] + dx*k3, rho_max, H)
        rho[i] = rho[i-1] + dx / 6 * (k1 + 2*k2 + 2*k3 + k4)
    return rho

def rk4_method_2(function, rho0, x0, x_end, rho_max, H, x_steps):
    valx = []
    valrho = []
    valx.append(x0)
    valrho.append(rho0)
    h = x_steps
    fung = function
    
    while x0 <= x_end:
        x1 = x0 + h
        k1 = h * fung(x0, rho0, rho_max, H)
        k2 = h * fung(x0 + 0.5*h, rho0 + 0.5*k1, rho_max, H)
        k3 = h * fung(x0 + 0.5*h, rho0 + 0.5*k2, rho_max, H)
        k4 = h * fung(x0 + h, rho0 + k3, rho_max, H)
        
        rho1 = rho0 + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        x0 = x1
        rho0 = rho1
        valx.append(x0)
        valrho.append(rho0)
    
    return valx, valrho


rho_polynom = rho_polynomial(xspan)
rho_sin2 = runge_kutta(drho_dx_sin2, xspan, dx, rho_max, H)
rho_gauss = runge_kutta(drho_dx_gauss, xspan, dx, rho_max, H)
rho_sigmoid = runge_kutta(drho_dx_sigmoid, xspan, dx, rho_max, H)

"""
plt.figure()
plt.plot(xspan, rho_polynom, marker = "o", fillstyle='none',label="poly")
plt.plot(xspan, rho_sin2, marker = "o", fillstyle='none', label="sin2")
plt.plot(xspan, rho_gauss, marker = "o", fillstyle='none', label="gauss")
plt.plot(xspan, rho_sigmoid, marker = "o", fillstyle='none', label="sigmoid")
#plt.plot(valx, valrho)
plt.grid()
plt.legend()
plt.show()
"""