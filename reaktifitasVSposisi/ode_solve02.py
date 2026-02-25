import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import reference_function as rf

W = 5.7
x_0 = 0.1
sigma = 0.3

x_end = 0.38
h=0.01

x0 = 0
rho0 = 0

def euler_method_2(function, rho0, x0, x_end, x_0, sigma, W, x_steps):
    valx = []
    valrho = []
    h = x_steps
    valx.append(x0)
    valrho.append(rho0)
    fung = function
    
    while x0 <= x_end:
        x1 = x0 + h
        rho1 = rho0 +  h*fung(x0, rho0, x_0, sigma, W)
        x0 = x1
        rho0 = rho1
        valx.append(x0)
        valrho.append(rho0)
    
    return valx, valrho

def rk4_method_2(function, rho0, x0, x_end, x_0, sigma, W, x_steps):
    valx = []
    valrho = []
    valx.append(x0)
    valrho.append(rho0)
    h = x_steps
    fung = function
    
    while x0 <= x_end:
        x1 = x0 + h
        k1 = h * fung(x0, rho0, x_0, sigma, W)
        k2 = h * fung(x0 + 0.5*h, rho0 + 0.5*k1, x_0, sigma, W)
        k3 = h * fung(x0 + 0.5*h, rho0 + 0.5*k2, x_0, sigma, W)
        k4 = h * fung(x0 + h, rho0 + k3, x_0, sigma, W)
        
        rho1 = rho0 + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        x0 = x1
        rho0 = rho1
        valx.append(x0)
        valrho.append(rho0)
    
    return valx, valrho
def adams_method_2(function, rho0, x0, x_end, x_0, sigma, W, x_steps):
    h = x_steps
    x = [i for i in np.arange(x0, x_end+h, h)]
    
    fung = function
    #mencari rho1 dengan methode RK4
    k1 = h * fung(x[0], rho0, x_0, sigma, W)
    k2 = h * fung(x[0] + 0.5*h, rho0 + 0.5*k1, x_0, sigma, W)
    k3 = h * fung(x[0] + 0.5*h, rho0 + 0.5*k2, x_0, sigma, W)
    k4 = h * fung(x[0] + h, rho0 + k3, x_0, sigma, W)
    rho1 = rho0 + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    #Adams–Bashforth 2 langkah
    valrhoAB = np.zeros_like(x)
    valrhoAB[0] = rho0
    valrhoAB[1] = rho1
    for i in np.arange(2,len(x),1):
        valrhoAB[i] = valrhoAB[i-1] + h*((1.5*fung(x[i-1],valrhoAB[i-1], x_0, sigma, W)) - 
                                         (0.5*fung(x[i-2],valrhoAB[i-2], x_0, sigma, W)))
    #Adams–Moulton
    valrhoAM = np.zeros_like(x)
    valrhoAM[0] = rho0
    for i in np.arange(1,len(x),1):
        valrhoAM[i] = valrhoAM[i-1] + (h/2)*(fung(x[i],valrhoAM[i], x_0, sigma, W) + 
                                            fung(x[i-1],valrhoAM[i-1], x_0, sigma, W)) 
    return x, valrhoAB, valrhoAM

df = pd.read_excel('data_ref_reactivity.xlsx')
xRef = df['posisi_batang'].values
yRef = df['reactivity'].values

xFung = np.arange(0, x_end, 0.01)
yFung = [rf.reference_function(i) for i in xFung]

xEul, yEul = euler_method_2(rf.fung_gaussian_bell, rho0, x0, x_end, x_0, sigma, W, h)
xRK4, yRK4 = rk4_method_2(rf.fung_gaussian_bell, rho0, x0, x_end, x_0, sigma, W, h)

xAdam, yAB, yAM = adams_method_2(rf.fung_gaussian_bell, rho0, x0, x_end, x_0, sigma, W, h)

plt.plot(xRef, yRef, 'k.', ls = '-')
plt.plot(xFung, yFung, 'k', ls = '-')

plt.plot(xEul, yEul, 'ro', ls = '-', markerfacecolor='r', label='Euler')
plt.plot(xRK4, yRK4, 'gv', ls = '-', markerfacecolor='g', label='Runge-Kutta 4')
plt.plot(xAdam, yAB, 'y^', ls = '-', markerfacecolor='y', label = "Adams–Bashforth 2")
plt.plot(xAdam, yAM, 'c<', ls = '-', markerfacecolor='c',label= "Adams–Moulton")

plt.xlabel("Control Rod Position (m)")
plt.ylabel("Reactivity ($)")

plt.grid()
plt.legend()
# Display the plot
plt.show()