import matplotlib.pyplot as plt
import numpy as np

import polynom_solution_lsoda as psl
import sinkuadrat_solution_lsoda as ssl
import gaussbell_solution_lsoda as gsl
import sigmoid_solution_lsoda as sigsl

Lambda = psl.Lambda
beta_groups = psl.df_fdn["beta"].to_numpy()
lambda_groups = psl.df_fdn["lambda"].to_numpy()

def jac_prke(t, y, beta_groups, lambda_groups, Lambda, rho_abs):
    beta = np.sum(beta_groups)
    m = len(beta_groups)

    J = np.zeros((m + 1, m + 1))
    J[0, 0] = (rho_abs - beta) / Lambda
    J[0, 1:] = lambda_groups
    J[1:, 0] = beta_groups / Lambda
    J[1:, 1:] = -np.diag(lambda_groups)
    return J

def stiffness_from_solution(sol, rho_abs_series, beta_groups, lambda_groups, Lambda):
    S = []
    hE = []
    hRK4 = []

    for tk, yk, rho_abs_k in zip(sol.t, sol.y.T, rho_abs_series):
        J = jac_prke(tk, yk, beta_groups, lambda_groups, Lambda, rho_abs_k)
        eigvals = np.linalg.eigvals(J)

        re_stable = np.abs(np.real(eigvals[np.real(eigvals) < -1e-12]))

        if len(re_stable) > 0:
            lam_fast = np.max(re_stable)
            lam_slow = np.min(re_stable)

            S.append(lam_fast / lam_slow)
            hE.append(2.0 / lam_fast)
            hRK4.append(2.785 / lam_fast)
        else:
            S.append(np.nan)
            hE.append(np.nan)
            hRK4.append(np.nan)

    return np.array(S), np.array(hE), np.array(hRK4)

S_sin2, hE_sin2, hRK4_sin2 = stiffness_from_solution(
    ssl.sol,
    ssl.df_out_lsoda["rho_abs"].to_numpy(),
    beta_groups, lambda_groups, Lambda
)

S_gauss, hE_gauss, hRK4_gauss = stiffness_from_solution(
    gsl.sol,
    gsl.df_out_lsoda["rho_abs"].to_numpy(),
    beta_groups, lambda_groups, Lambda
)

S_sigmoid, hE_sigmoid, hRK4_sigmoid = stiffness_from_solution(
    sigsl.sol,
    sigsl.df_out_lsoda["rho_abs"].to_numpy(),
    beta_groups, lambda_groups, Lambda
)

S_poly, hE_poly, hRK4_poly = stiffness_from_solution(
    psl.val,
    psl.df_out_lsoda["rho_abs"].to_numpy(),
    beta_groups, lambda_groups, Lambda
)

plt.figure()
plt.plot(psl.df_out_lsoda["time_s"].values, psl.df_out_lsoda["neutron_density_n"].values,
         marker='o', fillstyle='none', label="polynomial")
plt.plot(ssl.df_out_lsoda["time_s"].values, ssl.df_out_lsoda["neutron_density_n"].values, 
         marker='o', fillstyle='none', label="sin2")
plt.plot(gsl.df_out_lsoda["time_s"].values, gsl.df_out_lsoda["neutron_density_n"].values, 
         marker='o', fillstyle='none', label="gaussian")
plt.plot(sigsl.df_out_lsoda["time_s"].values, sigsl.df_out_lsoda["neutron_density_n"].values, 
         marker='o', fillstyle='none', label="sigmoid")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("n(t)")
plt.title("Number of neutrons vs Time")
plt.grid()
plt.show()

plt.figure()
plt.plot(psl.df_out_lsoda["time_s"].values,np.log(psl.df_out_lsoda["neutron_density_n"].values),
         marker='o', fillstyle='none', label="polynomial")
plt.plot(ssl.df_out_lsoda["time_s"].values, np.log(ssl.df_out_lsoda["neutron_density_n"].values), 
         marker='o', fillstyle='none', label="sin2")
plt.plot(gsl.df_out_lsoda["time_s"].values, np.log(gsl.df_out_lsoda["neutron_density_n"].values), 
         marker='o', fillstyle='none', label="gaussian")
plt.plot(sigsl.df_out_lsoda["time_s"].values, np.log(sigsl.df_out_lsoda["neutron_density_n"].values), 
         marker='o', fillstyle='none', label="sigmoid")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("log(n(t))")
plt.title("Number of neutrons vs Time")
plt.grid()
plt.show()

plt.figure()
plt.plot(ssl.sol.t, S_sin2, label="sin2")
plt.plot(gsl.sol.t, S_gauss, label="gaussian")
plt.plot(sigsl.sol.t, S_sigmoid, label="sigmoid")
plt.plot(psl.val.t, S_poly, label="polynomial")
plt.xlabel("Time (s)")
plt.ylabel("Stiffness ratio")
plt.title("Stiffness ratio vs Time")
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(ssl.sol.t, hE_sin2, label="Euler sin2")
plt.plot(ssl.sol.t, hE_gauss, label="Euler gauss")
plt.plot(ssl.sol.t, hRK4_sin2, label="RK4 sin2")
plt.xlabel("Time (s)")
plt.ylabel("Stable time step (s)")
plt.title("Stability limit for sin2")
plt.grid()
plt.legend()
plt.show()