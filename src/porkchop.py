import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
!pip install poliastro

from astropy import units as u
from poliastro.iod import izzo

# =========================================
# LOAD EPHEMERIS
# =========================================
df = pd.read_csv("data/veridian_ephemeris.csv")
time = df["MJD"].values

def build_interp(body):
    r = interp1d(time, df[[f"{body}_x", f"{body}_y"]].values, axis=0, kind='cubic')
    v = interp1d(time, df[[f"{body}_vx", f"{body}_vy"]].values, axis=0, kind='cubic')
    return r, v

rC, vC = build_interp("Caelus")
rV, vV = build_interp("Ventus")
rG, vG = build_interp("Glacia")

# =========================================
# CONSTANTS
# =========================================
mu_star = 1.393e11 * u.km**3 / u.s**2

# Caelus parking orbit
mu_caelus = 3.39e4
R_caelus = 7200
r_orbit = R_caelus + 500

v_circ = np.sqrt(mu_caelus / r_orbit)
v_esc = np.sqrt(2 * mu_caelus / r_orbit)

# =========================================
# LAMBERT SOLVER
# =========================================
def lambert_solver(r1, r2, tof_days):
    tof = tof_days * 86400 * u.s

    r1_q = np.array([r1[0], r1[1], 0.0]) * u.km
    r2_q = np.array([r2[0], r2[1], 0.0]) * u.km

    sols = list(izzo.lambert(mu_star, r1_q, r2_q, tof))
    if not sols:
        return None

    v1, v2 = sols[0]
    return v1.value[:2], v2.value[:2]

# =========================================
# GRIDS
# =========================================
t1_values = np.arange(60580, 60700, 1)     # Caelus departure
t2_values = np.arange(60750, 60950, 1)     # Ventus departure

tof_values = np.arange(200, 801, 5)

DV1 = np.full((len(tof_values), len(t1_values)), np.nan)
DV2 = np.full((len(tof_values), len(t2_values)), np.nan)

# =========================================
# PORKCHOP 1: Caelus → Ventus
# =========================================
for i, tof in enumerate(tof_values):
    for j, t1 in enumerate(t1_values):

        t2 = t1 + tof
        if t2 > time[-1]:
            continue

        r1 = rC(t1)
        v1 = vC(t1)

        r2 = rV(t2)
        v2 = vV(t2)

        try:
            sol = lambert_solver(r1, r2, tof)
            if sol is None:
                continue

            v1_t, _ = sol

            v_inf = np.linalg.norm(v1_t - v1)
            dv = np.sqrt(v_inf**2 + v_esc**2) - v_circ

            DV1[i, j] = dv

        except:
            continue

# =========================================
# PORKCHOP 2: Ventus → Glacia
# =========================================
for i, tof in enumerate(tof_values):
    for j, t2 in enumerate(t2_values):

        t3 = t2 + tof
        if t3 > time[-1]:
            continue

        r1 = rV(t2)
        v1 = vV(t2)

        r2 = rG(t3)
        v2 = vG(t3)

        try:
            sol = lambert_solver(r1, r2, tof)
            if sol is None:
                continue

            v1_t, _ = sol

            dv = np.linalg.norm(v1_t - v1)

            DV2[i, j] = dv

        except:
            continue

# =========================================
# PLOTTING
# =========================================
X1, Y1 = np.meshgrid(t1_values, tof_values)
X2, Y2 = np.meshgrid(t2_values, tof_values)

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot 1 ---
cp1 = axs[0].contourf(X1, Y1, DV1, levels=30)
fig.colorbar(cp1, ax=axs[0], label="ΔV (km/s)")
axs[0].set_title("Caelus → Ventus")
axs[0].set_xlabel("Departure Date t1 (MJD)")
axs[0].set_ylabel("Time of Flight (days)")

# --- Plot 2 ---
cp2 = axs[1].contourf(X2, Y2, DV2, levels=30)
fig.colorbar(cp2, ax=axs[1], label="ΔV (km/s)")
axs[1].set_title("Ventus → Glacia")
axs[1].set_xlabel("Departure Date t2 (MJD)")
axs[1].set_ylabel("Time of Flight (days)")

plt.tight_layout()
plt.show()