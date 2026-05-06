import numpy as np
import pandas as pd

df_fdn = pd.read_excel('fraction_delayed_neutrons_U235.xlsx', index_col=None)
# 1. Parameter Spesifik Reaktor (Contoh Pendekatan untuk Kartini/TRIGA)
# Anda bisa menyesuaikan nilai beta dan lambda ini dengan data teknis Kartini
beta_i = df_fdn["beta"].to_numpy()
lambda_i = df_fdn["lambda"].to_numpy()
beta_total = np.sum(beta_i)
Lambda = 4.0e-5  # Prompt neutron generation time (s)
rho = 0.0        # Kondisi kritis (t=0)

def calculate_stiffness(rho, beta_i, lambda_i, Lambda):
    # 2. Inisialisasi Matriks Jacobian (7x7 untuk 6 kelompok neutron kasip)
    # Orde: [n, C1, C2, C3, C4, C5, C6]
    J = np.zeros((7, 7))
    
    # Baris pertama (dn/dt)
    J[0, 0] = (rho - np.sum(beta_i)) / Lambda
    for i in range(6):
        J[0, i+1] = lambda_i[i]
        
    # Baris untuk dCi/dt
    for i in range(6):
        J[i+1, 0] = beta_i[i] / Lambda
        J[i+1, i+1] = -lambda_i[i]
        
    # 3. Menghitung Nilai Eigen
    eigenvalues = np.linalg.eigvals(J)
    
    # 4. Analisis Stiffness
    max_eig = np.max(np.abs(eigenvalues))
    min_eig = np.min(np.abs(eigenvalues))
    stiffness_ratio = max_eig / min_eig
    
    # 5. Batas Langkah Waktu (Stability Limit untuk Metode Eksplisit/RK4)
    dt_crit = 2.0 / max_eig
    
    return J, eigenvalues, stiffness_ratio, dt_crit

# Eksekusi
jacobian, eig, ratio, dt_limit = calculate_stiffness(rho, beta_i, lambda_i, Lambda)

print("--- ANALISIS FORMAL STIFFNESS REAKTOR KARTINI ---")
print(f"Eigenvalues: \n{eig}\n")
print(f"Eigenvalue Max (Abs): {np.max(np.abs(eig)):.4e}")
print(f"Eigenvalue Min (Abs): {np.min(np.abs(eig)):.4e}")
print(f"STIFFNESS RATIO (L): {ratio:.4e}")
print(f"Batas dt Kritis (Metode Eksplisit): {dt_limit:.6e} detik")

if ratio > 1000:
    print("\nKESIMPULAN: Sistem SANGAT STIFF. Metode Eksplisit (RK/Euler) akan butuh dt sangat kecil.")