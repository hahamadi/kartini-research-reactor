import numpy as np

def reference_function(x):
    y = -129.16*x**5 + 279.95*x**4 - 215.04*x**3 + 58.294*x**2 + 1.3702*x - 0.0029
    return y

def fung_sin2(x, rho, rho_max, H):
    C = (2 * rho_max) / H
    
    fx = C * np.sin((np.pi * x) / H)**2
    return fx

def fung_shooting_sin2(x, rho, C_opt, H):
    return C_opt * np.sin(np.pi * x / H)**2

def fung_gaussian_bell(x, rho, x_0, sigma, W):
    fx = W * np.exp(-((x - x_0)**2)/(2*sigma**2))
    return fx

def euler_method(function, x0, x_end, rho0, rho_max, x_steps):
    valx = []
    valrho = []
    h = x_steps
    H = x_end
    valx.append(x0)
    valrho.append(rho0)
    if function.lower() == "sin2":
        fung = fung_sin2
    
    while x0 <= x_end:
        x1 = x0 + h
        rho1 = rho0 +  h*fung(x0,rho0,rho_max,H)
        x0 = x1
        rho0 = rho1
        valx.append(x0)
        valrho.append(rho0)
    
    return valx, valrho

def rk4_method(function, x0, x_end, rho0, rho_max, x_steps):
    valx = []
    valrho = []
    valx.append(x0)
    valrho.append(rho0)
    h = x_steps
    H = x_end
    if function.lower() == "sin2":
        fung = fung_sin2
    elif function.lower() == "shooting_sin2":
        fung = fung_shooting_sin2
    
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

def adams_method(function, x0, x_end, rho0, rho_max, x_steps):
    h = x_steps
    H = x_end
    x = [i for i in np.arange(x0, x_end+h, h)]
    
    if function.lower() == "sin2":
        fung = fung_sin2
    #mencari rho1 dengan methode RK4
    k1 = h * fung(x[0], rho0, rho_max, H)
    k2 = h * fung(x[0] + 0.5*h, rho0 + 0.5*k1, rho_max, H)
    k3 = h * fung(x[0] + 0.5*h, rho0 + 0.5*k2, rho_max, H)
    k4 = h * fung(x[0] + h, rho0 + k3, rho_max, H)
    rho1 = rho0 + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    #Adams–Bashforth 2 langkah
    valrhoAB = np.zeros_like(x)
    valrhoAB[0] = rho0
    valrhoAB[1] = rho1
    for i in np.arange(2,len(x),1):
        valrhoAB[i] = valrhoAB[i-1] + h*((1.5*fung(x[i-1],valrhoAB[i-1], rho_max, H)) - 
                                         (0.5*fung(x[i-2],valrhoAB[i-2], rho_max, H)))
    #Adams–Moulton
    valrhoAM = np.zeros_like(x)
    valrhoAM[0] = rho0
    for i in np.arange(1,len(x),1):
        valrhoAM[i] = valrhoAM[i-1] + (h/2)*(fung(x[i],valrhoAM[i], rho_max, H) + 
                                            fung(x[i-1],valrhoAM[i-1], rho_max, H)) 
    return x, valrhoAB, valrhoAM

def fung_sin2_etd(x, rho, rho_max, H, tau):
    C = (2 * rho_max) / H
    
    fx = (C * np.sin((np.pi * x) / H)**2) - (tau*rho)
    return fx

def etd1_method(function, x0, rho0, rho_max, x_steps, x_end, tau):
    h = x_steps
    H = x_end
    x = [i for i in np.arange(x0, x_end+h, h)]
    valrhoETD = np.zeros_like(x)
    if function.lower() == "sin2":
        fung_etd = fung_sin2_etd
        
    exp_fac = np.exp(tau*h)
    valrhoETD[0] = rho0
    for i in np.arange(1, len(x), 1):
        valrhoETD[i] = ((exp_fac*valrhoETD[i-1]) + 
        (((exp_fac-1)/tau)*fung_etd(x[i-1], valrhoETD[i-1], rho_max, H, tau)))
        
    return x, valrhoETD

def rk4_sin2_shooting(C, x0, rho0, h, x_end):
    x = x0
    rho = rho0
    H = x_end
    
    while x < x_end:
        k1 = h * (C * np.sin(np.pi * x / H)**2)
        k2 = h * (C * np.sin(np.pi * (x + 0.5*h) / H)**2)
        k3 = h * (C * np.sin(np.pi * (x + 0.5*h) / H)**2)
        k4 = h * (C * np.sin(np.pi * (x + h) / H)**2)
        
        rho = rho + (k1 + 2*k2 + 2*k3 + k4) / 6
        x = x + h
        
    return rho


def shooting_method(function, x0, rho0, rho_target, C0, C1, x_steps, x_end, tol, max_iter):
    h = x_steps
    
    if function.lower() == "sin2":
        R0 = rk4_sin2_shooting(C0, x0, rho0, h, x_end) - rho_target
        R1 = rk4_sin2_shooting(C1, x0, rho0, h, x_end) - rho_target
    
    for i in range(max_iter):
        if abs(R1) < tol:
            print(f"Converged in {i} iterations")
            return C1
        
        # Secant update
        C2 = C1 - R1 * (C1 - C0) / (R1 - R0)
        
        C0, R0 = C1, R1
        C1 = C2
        R1 = rk4_sin2_shooting(C1, x0, rho0, h, x_end) - rho_target
    
    raise RuntimeError("Shooting method did not converge")

def shooting_rk4_method(function, x0, rho0, rho_max, x_steps, x_end, C0, C1, tol, max_iter):
    
    rho_target = rho_max
    C_opt = shooting_method(function, x0, rho0, rho_target, C0, C1, x_steps, x_end, tol, max_iter)
    if function == "sin2":
        return rk4_method("shooting_sin2", x0, x_end, rho0, C_opt, x_steps)