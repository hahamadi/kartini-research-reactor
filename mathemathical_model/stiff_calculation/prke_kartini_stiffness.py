"""
PRKE + rod motion based stiffness-ratio calculator for Reaktor Kartini.

Catatan:
1) Dokumen Kartini yang diunggah memberi parameter penting:
   - beta_eff ~ 0.007
   - panjang lintasan batang = 38 cm
   - kecepatan maksimum penarikan = 0.63 cm/s
   - umur neutron pada bagian desain nuklir = 4.3e-5 s
   - ada juga "waktu hidup efektif neutron cepat" = 6e-5 s di tabel umum
2) Teks polinom batang kendali yang diekstrak OCR dari gambar tampak tidak sepenuhnya konsisten.
   Karena itu, ganti koefisien di bawah dengan hasil fit kalibrasi yang Anda percaya.
3) Jika x dimodelkan sebagai state dengan xdot = konstan, akan muncul satu eigenvalue nol.
   Untuk stiffness ratio fisik PRKE, kita hitung hanya pada subsistem neutron + precursor,
   dengan x dianggap parameter waktu (bukan mode dinamik yang dihitung stiffness-nya).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Callable, Tuple

# =========================
# 1) Parameter Kartini
# =========================

BETA_EFF = 0.007          # dari dokumen
LAMBDA = 4.3e-5           # s, dari tabel desain nuklir
STROKE_CM = 38.0          # cm
ROD_SPEED_CM_S = 0.63     # cm/s
ROD_SPEED_PERCENT_S = 100.0 * ROD_SPEED_CM_S / STROKE_CM

# Delayed neutron 6-group tipikal TRIGA.
# Kalau Anda punya data beta_i dan lambda_i khusus Kartini, ganti di sini.
BETA_I = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273], dtype=float)
LAMBDA_I = np.array([0.0124, 0.0305, 0.1110, 0.3010, 1.1400, 3.0100], dtype=float)

# =========================
# 2) Fit polinom rho_$(x)
#    x dalam persen posisi batang: 0..100
#    rho_$ dalam satuan dollar
# =========================
#
# Koefisien ditulis urut menurun: [a_n, ..., a_1, a_0]
# Anda bebas mengganti sesuai hasil kalibrasi yang benar.
#
# Di bawah ini saya masukkan contoh yang berasal dari OCR dokumen,
# tetapi kemungkinan masih perlu koreksi.
#
ROD_POLY: Dict[str, np.poly1d] = {
    # Kemungkinan dari Gambar V-31/V-32/V-33 hasil OCR
    "rod_A": np.poly1d([6e-10, -1e-7, -7e-8, 7e-4, 1.68e-2, 0.0]),  # orde 5
    "rod_B": np.poly1d([5e-8, -2e-5, 1.2e-3, 1.24e-2, 0.0]),         # orde 4
    "rod_C": np.poly1d([4e-8, -1e-5, 7e-4, 6.2e-3, 0.0]),            # orde 4
}

def rho_dollar(rod_name: str, x_percent: float) -> float:
    """Integral rod worth rho_$(x), x dalam persen 0..100."""
    return float(ROD_POLY[rod_name](x_percent))

def drho_dollar_dx(rod_name: str, x_percent: float) -> float:
    """Turunan d(rho_$)/dx dengan x dalam persen."""
    dp = np.polyder(ROD_POLY[rod_name])
    return float(dp(x_percent))

def rho_abs(rod_name: str, x_percent: float) -> float:
    """Reaktivitas absolut delta-k/k."""
    return BETA_EFF * rho_dollar(rod_name, x_percent)

def drho_abs_dt(rod_name: str, x_percent: float, speed_percent_s: float = ROD_SPEED_PERCENT_S) -> float:
    """d(rho)/dt dalam delta-k/k per s."""
    return BETA_EFF * drho_dollar_dx(rod_name, x_percent) * speed_percent_s

def drho_dollar_dt(rod_name: str, x_percent: float, speed_percent_s: float = ROD_SPEED_PERCENT_S) -> float:
    """d(rho_$)/dt dalam $/s."""
    return drho_dollar_dx(rod_name, x_percent) * speed_percent_s

# =========================
# 3) PRKE 6-group
# =========================
#
# dn/dt   = ((rho - beta)/Lambda) n + sum(lambda_i C_i)
# dC_i/dt = (beta_i/Lambda) n - lambda_i C_i
#
# Jika x(t) diketahui dari gerak batang:
#   dx/dt = v_x = konstan
# maka rho(t) = rho(x(t))
#
# Jacobian yang dipakai untuk stiffness ratio adalah Jacobian subsistem
# [n, C1..C6], dengan x dianggap parameter.
# =========================

def prke_jacobian(rho_abs_value: float,
                  beta_i: np.ndarray = BETA_I,
                  lambda_i: np.ndarray = LAMBDA_I,
                  gen_time: float = LAMBDA) -> np.ndarray:
    """Jacobian 7x7 untuk subsistem PRKE [n, C1..C6]."""
    beta_total = float(np.sum(beta_i))
    J = np.zeros((1 + len(beta_i), 1 + len(beta_i)), dtype=float)
    J[0, 0] = (rho_abs_value - beta_total) / gen_time
    J[0, 1:] = lambda_i
    J[1:, 0] = beta_i / gen_time
    J[1:, 1:] = -np.diag(lambda_i)
    return J

def stiffness_ratio_from_jacobian(J: np.ndarray,
                                  tol: float = 1e-12) -> Tuple[float, np.ndarray]:
    """
    Rasio stiffness = max(|Re(lambda)|) / min(|Re(lambda)|)
    memakai eigenvalue Jacobian.
    """
    eig = np.linalg.eigvals(J)
    real_abs = np.abs(np.real(eig))
    real_abs = real_abs[real_abs > tol]
    if real_abs.size == 0:
        return np.inf, eig
    return float(np.max(real_abs) / np.min(real_abs)), eig

def stiffness_ratio_at_position(rod_name: str,
                                x_percent: float,
                                beta_i: np.ndarray = BETA_I,
                                lambda_i: np.ndarray = LAMBDA_I,
                                gen_time: float = LAMBDA) -> Tuple[float, np.ndarray]:
    """Rasio stiffness PRKE pada posisi batang tertentu."""
    rho = rho_abs(rod_name, x_percent)
    J = prke_jacobian(rho, beta_i=beta_i, lambda_i=lambda_i, gen_time=gen_time)
    return stiffness_ratio_from_jacobian(J)

# =========================
# 4) Profil waktu batang kendali
# =========================

def x_of_t(t: float, x0_percent: float = 0.0, speed_percent_s: float = ROD_SPEED_PERCENT_S,
           xmin: float = 0.0, xmax: float = 100.0) -> float:
    """Posisi batang sebagai fungsi waktu, dibatasi 0..100%."""
    x = x0_percent + speed_percent_s * t
    return float(np.clip(x, xmin, xmax))

def stiffness_ratio_vs_time(rod_name: str,
                            t_grid: np.ndarray,
                            x0_percent: float = 0.0,
                            speed_percent_s: float = ROD_SPEED_PERCENT_S) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Menghasilkan:
      x(t), rho_$(t), kappa(t)
    """
    x_vals = np.array([x_of_t(t, x0_percent=x0_percent, speed_percent_s=speed_percent_s) for t in t_grid], dtype=float)
    rho_d_vals = np.array([rho_dollar(rod_name, x) for x in x_vals], dtype=float)
    kappa_vals = np.array([stiffness_ratio_at_position(rod_name, x)[0] for x in x_vals], dtype=float)
    return x_vals, rho_d_vals, kappa_vals

# =========================
# 5) Ringkasan cepat
# =========================

def summary_table(rod_name: str, x_points=(0.0, 25.0, 50.0, 75.0, 100.0)) -> None:
    print("=" * 78)
    print(f"ROD: {rod_name}")
    print(f"Rod speed = {ROD_SPEED_CM_S:.4f} cm/s = {ROD_SPEED_PERCENT_S:.6f} %/s")
    print(f"beta_eff = {BETA_EFF:.6f}, Lambda = {LAMBDA:.2e} s")
    print("-" * 78)
    print(f"{'x(%)':>8} {'rho($)':>14} {'drho/dx ($/%)':>18} {'drho/dt ($/s)':>18} {'kappa':>14}")
    for x in x_points:
        rho_d = rho_dollar(rod_name, x)
        drdx = drho_dollar_dx(rod_name, x)
        drdt = drho_dollar_dt(rod_name, x)
        kappa, _ = stiffness_ratio_at_position(rod_name, x)
        print(f"{x:8.2f} {rho_d:14.6f} {drdx:18.6f} {drdt:18.6f} {kappa:14.6f}")
    print()

if __name__ == "__main__":
    print("PRKE stiffness-ratio calculator for Kartini")
    print()
    for rod in ROD_POLY:
        summary_table(rod)

    # Contoh profil waktu 0 sampai 60 s
    t = np.linspace(0.0, 60.0, 121)
    rod = "rod_A"
    x_vals, rho_vals, kappa_vals = stiffness_ratio_vs_time(rod, t, x0_percent=0.0)

    print("=" * 78)
    print(f"Contoh transient untuk {rod}")
    print(f"x(t=0) = {x_vals[0]:.3f} %, x(t=60s) = {x_vals[-1]:.3f} %")
    print(f"rho_$(t=0) = {rho_vals[0]:.6f}, rho_$(t=60s) = {rho_vals[-1]:.6f}")
    print(f"kappa_min = {np.min(kappa_vals):.6f}, kappa_max = {np.max(kappa_vals):.6f}")
