import numpy as np
from polynom_solution_rk4V2 import df_fdn, Lambda
#from polynom_solution_eulerV2 import df_out_euler as dfPolynomEuler
from polynom_solution_rk4V2 import df_out_rk4 as dfPolynomRk
#from sinkuadrat_solution_euler import df_out_euler as dfSinEuler
from sinkuadrat_solution_rk4 import df_out_rk4 as dfSinRk
from gaussbell_solution_rk4 import df_out_rk4 as dfGaussRk
from sigmoid_solution_rk4 import df_out_rk4 as dfSigmoidRk


def prke_jacobian(rho_abs, beta_groups, lambda_groups, Lambda):

    beta = np.sum(beta_groups)
    m = len(beta_groups)

    J =  np.zeros((m+1, m+1))
    J[0,0] = (rho_abs - beta) / Lambda
    J[0, 1:] = lambda_groups
    J[1:, 0] = beta_groups / Lambda
    J[1:, 1:] = -np.diag(lambda_groups)
    return J

def euler_stability_function(z):
    return 1.0 + z

def rk4_stability_function(z):
    return 1.0 + z + 0.5 * z**2 + (1.0 / 6.0) * z**3 + (1.0 / 24.0) * z**4

def max_stable_step_for_method(eigvals, stability_function, h_upper=1.0, tol=1e-12, max_expand=60):
    eigvals = np.asarray(eigvals, dtype=complex)

    if eigvals.size == 0:
        return np.nan

    if np.any(np.real(eigvals) >= 0.0):
        return 0.0

    def is_stable(h):
        z = h * eigvals
        return np.all(np.abs(stability_function(z)) <= 1.0 + 1e-12)

    if not is_stable(0.0):
        return 0.0

    lo = 0.0
    hi = h_upper

    expand_count = 0
    while is_stable(hi) and expand_count < max_expand:
        lo = hi
        hi *= 2.0
        expand_count += 1

    if expand_count == max_expand and is_stable(hi):
        return hi

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if is_stable(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, lo):
            break

    return lo

def stiffness_metrics(rho_abs_series, beta_groups, lambda_groups, Lambda, eps=1e-10):
    S_list = []
    hE_list = []
    hRK4_list = []
    eigs_all = []

    for rho_abs in rho_abs_series:
        J = prke_jacobian(rho_abs, beta_groups, lambda_groups, Lambda)
        eigvals = np.linalg.eigvals(J)
        eigs_all.append(eigvals)

        stable_eigs = eigvals[np.real(eigvals) < -eps]

        if stable_eigs.size == 0:
            S_list.append(np.nan)
            hE_list.append(np.nan)
            hRK4_list.append(np.nan)
            continue

        magnitudes = np.abs(stable_eigs)
        lam_fast = np.max(magnitudes)
        lam_slow = np.min(magnitudes)

        S_list.append(lam_fast / lam_slow)
        hE_list.append(max_stable_step_for_method(stable_eigs, euler_stability_function))
        hRK4_list.append(max_stable_step_for_method(stable_eigs, rk4_stability_function))

    return np.array(S_list), np.array(hE_list), np.array(hRK4_list), eigs_all
alldf = [['polynomial', dfPolynomRk], 
        ['Sin2', dfSinRk],
        ['Gaussian Bell', dfGaussRk],
        ['Sigmoid', dfSigmoidRk]]

beta_group = df_fdn["beta"].to_numpy()
lambda_group = df_fdn["lambda"].to_numpy()
for i in range(len(alldf)):
    dfname = alldf[i][0]
    dfData = alldf[i][1]
    rho_abs_t = dfData["rho_abs"].values
     
    S, hE, hRK4, eigs = stiffness_metrics(rho_abs_t, beta_group, lambda_group, Lambda)
    print(dfname)
    print("Stiffness ratio min/max:", np.nanmin(S), np.nanmax(S))
    print("Euler dt global max    :", np.nanmin(hE))
    print("RK4 dt global max      :", np.nanmin(hRK4))
#print(dfPolynomEuler.columns.values)