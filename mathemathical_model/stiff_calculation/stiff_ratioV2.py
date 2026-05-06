import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import eigvals

# ============================================================
# 1. PARAMETER REAKTOR KARTINI
# ============================================================

# Data dari dokumen
beta_eff = 0.007          # fraksi neutron kasip efektif
Lambda = 4.3e-5           # s, dari Tabel V-6
alpha_T = -1.2e-4         # dk/k per degC, koefisien suhu negatif TRIGA

P_nominal = 100e3         # W, daya nominal Kartini
T_ref = 25.0              # degC, temperatur referensi

# ------------------------------------------------------------
# Data 6 grup delayed neutron:
# tidak diberikan di file, jadi diasumsikan memakai set standar U-235
# lalu diskalakan agar jumlahnya tepat = beta_eff Kartini
# ------------------------------------------------------------
lambda_i = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])  # 1/s
beta_base = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
beta_i = beta_base * (beta_eff / beta_base.sum())

# Kondisi tunak kritis awal pada rho=0
def initial_conditions(n0=1.0):
    C0 = beta_i / (Lambda * lambda_i) * n0
    y0 = np.concatenate(([n0], C0))
    return y0

# ============================================================
# 2. DEFINISI FUNGSI REAKTIVITAS
# ============================================================

# Konversi dollar ke dk/k
def dollar_to_rho(rho_dollar):
    return beta_eff * rho_dollar

# Gerak batang kendali linear
def rod_position(t, x0=0.0, v=1.6666666667):
    # x dalam persen, misal 0 sampai 100
    return x0 + v * t

# -------------------------
# A. Polynomial rod-worth
# r(x) dalam dollar
# dari dokumen: r = 6E-10 x5 - 1E-07 x4 - 7E-08 x3 + 0.0007 x2 + 0.0168 x
# -------------------------
def r_poly_dollar(x):
    return 6e-10*x**5 - 1e-7*x**4 - 7e-8*x**3 + 7e-4*x**2 + 1.68e-2*x

def rho_poly(t, scale=0.2, x0=0.0, v=1.6666666667):
    # scale dipakai supaya amplitudo tidak terlalu besar
    x = rod_position(t, x0, v)
    return scale * dollar_to_rho(r_poly_dollar(x))

# -------------------------
# B. d(rho_dollar)/dx = A sin^2(omega x + phi)
# rho_dollar(x) = rho0 + A/2 x - A/(4 omega) sin(2 omega x + 2 phi) + const
# -------------------------
def rho_sinsq(t, A=0.0372, omega=np.pi/50.0, phi=0.0, rho0_dollar=0.0, x0=0.0, v=1.6666666667):
    x = rod_position(t, x0, v)
    # pilih konstanta agar rho(x0) = rho0_dollar
    const = rho0_dollar - (A/2)*x0 + (A/(4*omega))*np.sin(2*omega*x0 + 2*phi)
    rho_dollar = const + (A/2)*x - (A/(4*omega))*np.sin(2*omega*x + 2*phi)
    return dollar_to_rho(rho_dollar)

# -------------------------
# C. Gaussian dalam dollar
# -------------------------
def rho_gaussian(t, rho_peak_dollar=0.5, xc=50.0, sigma=10.0, x0=0.0, v=1.6666666667):
    x = rod_position(t, x0, v)
    rho_dollar = rho_peak_dollar * np.exp(-0.5*((x - xc)/sigma)**2)
    return dollar_to_rho(rho_dollar)

# -------------------------
# D. Sigmoid dalam dollar
# -------------------------
def rho_sigmoid(t, delta_rho_dollar=0.8, k=0.2, x_mid=50.0, x0=0.0, v=1.6666666667):
    x = rod_position(t, x0, v)
    rho_dollar = delta_rho_dollar / (1.0 + np.exp(-k*(x - x_mid)))
    return dollar_to_rho(rho_dollar)

# ============================================================
# 3. OPSIONAL: FEEDBACK SUHU SEDERHANA
# ============================================================
# Ini BUKAN model termohidrolika detail Kartini.
# Hanya model lumped 1-node supaya efek alpha_T dapat dilihat.

def dTdt_simple(T, P, tau_th=20.0, T_cool=39.0):
    # Model sederhana:
    # dT/dt = ((P/P_nominal)*Qref - (T - T_cool))/tau_th
    # Dipilih agar T menuju T_cool + Qref*(P/P_nominal)
    # Jika Qref=125, maka pada 100 kW T ~ 39 + 125 = 164 C
    Qref = 125.0
    return ((P / P_nominal) * Qref - (T - T_cool)) / tau_th

# ============================================================
# 4. MODEL PRKE
# ============================================================

def prke_rhs(t, y, rho_func, with_temp_feedback=False):
    if with_temp_feedback:
        n = y[0]
        C = y[1:7]
        T = y[7]
    else:
        n = y[0]
        C = y[1:7]

    # daya ter-normalisasi terhadap daya nominal
    P = n * P_nominal

    rho_ext = rho_func(t)

    if with_temp_feedback:
        rho_fb = alpha_T * (T - T_ref)
        rho = rho_ext + rho_fb
    else:
        rho = rho_ext

    dn_dt = ((rho - beta_eff) / Lambda) * n + np.dot(lambda_i, C)
    dC_dt = beta_i / Lambda * n - lambda_i * C

    if with_temp_feedback:
        dT_dt = dTdt_simple(T, P)
        return np.concatenate(([dn_dt], dC_dt, [dT_dt]))
    else:
        return np.concatenate(([dn_dt], dC_dt))

# ============================================================
# 5. JACOBIAN DAN STIFFNESS RATIO
# ============================================================

def jacobian_prke(rho):
    J = np.zeros((7, 7))
    J[0, 0] = (rho - beta_eff) / Lambda
    J[0, 1:] = lambda_i
    for i in range(6):
        J[i+1, 0] = beta_i[i] / Lambda
        J[i+1, i+1] = -lambda_i[i]
    return J

def stiffness_ratio(rho):
    J = jacobian_prke(rho)
    ev = eigvals(J)
    real_parts = np.abs(np.real(ev))

    # hindari pembagian nol jika ada eigenvalue yang sangat kecil
    eps = 1e-14
    real_parts = np.where(real_parts < eps, eps, real_parts)

    S = real_parts.max() / real_parts.min()
    return S, ev

def scan_stiffness_over_time(rho_func, t_span=(0.0, 60.0), npts=400):
    t_grid = np.linspace(t_span[0], t_span[1], npts)
    S_grid = []
    rho_grid = []

    for t in t_grid:
        rho = rho_func(t)
        S, _ = stiffness_ratio(rho)
        S_grid.append(S)
        rho_grid.append(rho)

    S_grid = np.array(S_grid)
    rho_grid = np.array(rho_grid)

    idx = np.argmax(S_grid)
    return {
        "t_grid": t_grid,
        "rho_grid": rho_grid,
        "S_grid": S_grid,
        "S_max": S_grid[idx],
        "t_at_Smax": t_grid[idx],
        "rho_at_Smax": rho_grid[idx],
    }

# ============================================================
# 6. SIMULASI PRKE
# ============================================================

def simulate_prke(rho_func, t_end=60.0, with_temp_feedback=False):
    if with_temp_feedback:
        y0 = np.concatenate((initial_conditions(1.0), [39.0]))  # T awal ~ pendingin
        nvar = 8
    else:
        y0 = initial_conditions(1.0)
        nvar = 7

    sol = solve_ivp(
        fun=lambda t, y: prke_rhs(t, y, rho_func, with_temp_feedback=with_temp_feedback),
        t_span=(0.0, t_end),
        y0=y0,
        method="BDF",       # solver stiff
        atol=1e-10,
        rtol=1e-8,
        dense_output=True,
        max_step=0.2
    )

    return sol

# ============================================================
# 7. CONTOH PEMAKAIAN
# ============================================================

if __name__ == "__main__":
    cases = {
        "Polynomial": lambda t: rho_poly(t, scale=0.2, x0=0.0, v=1.6666666667),
        "SinSq d(rho)/dx": lambda t: rho_sinsq(t, A=0.0372, omega=np.pi/50.0, phi=0.0, rho0_dollar=0.0, x0=0.0, v=1.6666666667),
        "Gaussian": lambda t: rho_gaussian(t, rho_peak_dollar=0.5, xc=50.0, sigma=10.0, x0=0.0, v=1.6666666667),
        "Sigmoid": lambda t: rho_sigmoid(t, delta_rho_dollar=0.8, k=0.2, x_mid=50.0, x0=0.0, v=1.6666666667),
    }

    print("=== ANALISIS STIFFNESS PRKE KARTINI ===")
    print(f"beta_eff = {beta_eff:.6f}")
    print(f"Lambda   = {Lambda:.3e} s")
    print(f"alpha_T  = {alpha_T:.3e} dk/k per degC")
    print()

    for name, rho_func in cases.items():
        result = scan_stiffness_over_time(rho_func, t_span=(0.0, 60.0), npts=400)
        print(f"[{name}]")
        print(f"  S_max       = {result['S_max']:.4e}")
        print(f"  t(S_max)    = {result['t_at_Smax']:.4f} s")
        print(f"  rho(S_max)  = {result['rho_at_Smax']:.6e} dk/k")
        print()

    # Simulasi salah satu kasus, misalnya sigmoid
    print("=== SIMULASI TRANSIEN SIGMOID TANPA FEEDBACK SUHU ===")
    sol1 = simulate_prke(cases["Sigmoid"], t_end=60.0, with_temp_feedback=False)
    print("success:", sol1.success)
    print("message:", sol1.message)
    print("n(t_end) =", sol1.y[0, -1])

    print("\n=== SIMULASI TRANSIEN SIGMOID DENGAN FEEDBACK SUHU SEDERHANA ===")
    sol2 = simulate_prke(cases["Sigmoid"], t_end=60.0, with_temp_feedback=True)
    print("success:", sol2.success)
    print("message:", sol2.message)
    print("n(t_end) =", sol2.y[0, -1])
    print("T(t_end) =", sol2.y[7, -1], "degC")