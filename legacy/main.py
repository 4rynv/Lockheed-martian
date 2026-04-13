import heapq
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from astropy import units as u
from poliastro.iod import izzo
from scipy.interpolate import interp1d

# =========================================
# LOAD DATA
# =========================================
df = pd.read_csv("veridian_ephemeris.csv")
time = df["MJD"].values


def build_interp(body):
    r = interp1d(time, df[[f"{body}_x", f"{body}_y"]].values, axis=0, kind="cubic")
    v = interp1d(time, df[[f"{body}_vx", f"{body}_vy"]].values, axis=0, kind="cubic")
    return r, v


rC, vC = build_interp("Caelus")
rV, vV = build_interp("Ventus")
rG, vG = build_interp("Glacia")

# =========================================
# CONSTANTS
# =========================================
mu_star = 1.393e11 * u.km**3 / u.s**2

# Assignment values (Table 2)
mu_caelus = 3.986e5
mu_ventus = 1.266e8
mu_glacia = 1.267e7

R_caelus = 7200
R_ventus = 65000
R_glacia = 30000

r_orbit = R_caelus + 500
v_circ = np.sqrt(mu_caelus / r_orbit)
v_esc = np.sqrt(2 * mu_caelus / r_orbit)

r_glacia_orbit = R_glacia + 500
v_circ_glacia = np.sqrt(mu_glacia / r_glacia_orbit)

AU = 1.496e8
MATCH_TOL = 0.1  # km/s
ALLOW_TERMINAL_DSM = True
COUNT_GLACIA_CAPTURE_IN_TOTAL = False  # coordinator clarified rendezvous-only scoring
MAX_TOTAL_DV = 25.0

rp_values = np.arange(R_ventus + 2000, R_ventus + 20001, 2000)

# =========================================
# LAMBERT
# =========================================
def _extract_lambert_velocities(raw_solutions):
    # poliastro can return [v1, v2] or [(v1, v2), ...] depending on version
    if (
        len(raw_solutions) == 2
        and hasattr(raw_solutions[0], "unit")
        and np.asarray(raw_solutions[0]).shape == (3,)
    ):
        return raw_solutions[0], raw_solutions[1]

    first = raw_solutions[0]
    if isinstance(first, (tuple, list)) and len(first) >= 2:
        return first[0], first[1]
    if hasattr(first, "v1") and hasattr(first, "v2"):
        return first.v1, first.v2

    return None, None


def lambert_solver(r1, r2, tof_days):
    tof = float(tof_days) * 86400 * u.s

    r1_q = np.array([r1[0], r1[1], 0.0]) * u.km
    r2_q = np.array([r2[0], r2[1], 0.0]) * u.km

    try:
        raw_solutions = list(izzo.lambert(mu_star, r1_q, r2_q, tof))
    except Exception:
        return None

    if not raw_solutions:
        return None

    v1_q, v2_q = _extract_lambert_velocities(raw_solutions)
    if v1_q is None:
        return None

    v1 = np.asarray(v1_q.to_value(u.km / u.s), dtype=float)
    v2 = np.asarray(v2_q.to_value(u.km / u.s), dtype=float)
    return v1[:2], v2[:2]


# =========================================
# FLYBY / DSM HELPERS
# =========================================
def rotate(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def apply_terminal_dsm(v_inf_vec, tol):
    v_inf_mag = np.linalg.norm(v_inf_vec)
    if v_inf_mag <= tol:
        return 0.0, v_inf_vec.copy()

    # Minimum impulse to get inside tolerance ball
    v_inf_post = v_inf_vec * (tol / v_inf_mag)
    dsm_vec = v_inf_post - v_inf_vec
    return np.linalg.norm(dsm_vec), v_inf_post


def glacia_capture_dv(v_inf_arrival):
    return np.sqrt(v_inf_arrival**2 + (2 * mu_glacia / r_glacia_orbit)) - v_circ_glacia


# =========================================
# SEARCH SPACE
# =========================================
tof1_values = np.arange(200, 350, 5)
tof2_values = np.arange(300, 700, 5)


# =========================================
# WORKER FUNCTION
# =========================================
def evaluate_t1(t1):
    best_dv = 1e9
    best_sol = None
    min_arrival_pre_dsm = 1e9
    min_arrival_v_after = None

    top_solutions = []  # heap for top 10

    for tof1 in tof1_values:
        t2 = t1 + tof1
        if t2 > time[-1]:
            continue

        r1 = rC(t1)
        v1 = vC(t1)

        if np.linalg.norm(r1) < 0.4 * AU:
            continue

        r2 = rV(t2)
        v2 = vV(t2)

        if np.linalg.norm(r2) < 0.4 * AU:
            continue

        sol = lambert_solver(r1, r2, tof1)
        if sol is None:
            continue

        v1_t, v2_t = sol

        # Departure DV from Caelus parking orbit
        v_inf_depart = np.linalg.norm(v1_t - v1)
        dv_depart = np.sqrt(v_inf_depart**2 + v_esc**2) - v_circ

        # Flyby
        v_inf_in = v2_t - v2
        v_inf_mag = np.linalg.norm(v_inf_in)

        if v_inf_mag == 0:
            continue

        for rp in rp_values:
            arg = 1 / (1 + (rp * v_inf_mag**2) / mu_ventus)
            if abs(arg) > 1:
                continue

            delta = 2 * np.arcsin(arg)

            for sign in (+1, -1):
                v_inf_out = rotate(v_inf_in, sign * delta)
                v_after = v2 + v_inf_out

                # Keep flyby branches that increase heliocentric energy
                if np.linalg.norm(v_after) <= np.linalg.norm(v2 + v_inf_in):
                    continue

                for tof2 in tof2_values:
                    t3 = t2 + tof2

                    if (t3 - t1) > 2922 or t3 > time[-1]:
                        continue

                    r3 = rG(t3)
                    v3 = vG(t3)

                    if np.linalg.norm(r3) < 0.4 * AU:
                        continue

                    sol2 = lambert_solver(r2, r3, tof2)
                    if sol2 is None:
                        continue

                    v1_2, v2_2 = sol2

                    # Burn at Ventus encounter to connect flyby exit to transfer leg
                    dv_corr = np.linalg.norm(v1_2 - v_after)

                    # Relative arrival speed at Glacia before any final trim burn
                    v_inf_glacia_vec_pre = v2_2 - v3
                    v_inf_glacia_mag_pre = np.linalg.norm(v_inf_glacia_vec_pre)

                    if v_inf_glacia_mag_pre < min_arrival_pre_dsm:
                        min_arrival_pre_dsm = v_inf_glacia_mag_pre
                        min_arrival_v_after = v_after.copy()

                    if ALLOW_TERMINAL_DSM:
                        dv_match, v_inf_glacia_vec_post = apply_terminal_dsm(
                            v_inf_glacia_vec_pre, MATCH_TOL
                        )
                        v_inf_glacia_mag_post = np.linalg.norm(v_inf_glacia_vec_post)
                    else:
                        if v_inf_glacia_mag_pre > MATCH_TOL:
                            continue
                        dv_match = 0.0
                        v_inf_glacia_vec_post = v_inf_glacia_vec_pre
                        v_inf_glacia_mag_post = v_inf_glacia_mag_pre

                    dv_capture = glacia_capture_dv(v_inf_glacia_mag_post)
                    dv_arrival_scored = dv_match
                    if COUNT_GLACIA_CAPTURE_IN_TOTAL:
                        dv_arrival_scored += dv_capture

                    total_dv = dv_depart + dv_corr + dv_arrival_scored
                    if total_dv > MAX_TOTAL_DV:
                        continue

                    # Store for top 10
                    sol_tuple = (
                        total_dv,
                        t1,
                        t2,
                        t3,
                        rp,
                        dv_depart,
                        dv_corr,
                        dv_match,
                        dv_arrival_scored,
                        dv_capture,
                        v_inf_glacia_mag_pre,
                        v_inf_glacia_mag_post,
                        v_inf_glacia_vec_pre,
                        v_inf_glacia_vec_post,
                    )

                    if len(top_solutions) < 10:
                        heapq.heappush(top_solutions, (-total_dv, sol_tuple))
                    elif total_dv < -top_solutions[0][0]:
                        heapq.heapreplace(top_solutions, (-total_dv, sol_tuple))

                    # Best solution
                    if total_dv < best_dv:
                        best_dv = total_dv
                        best_sol = (
                            t1,
                            t2,
                            t3,
                            rp,
                            dv_depart,
                            dv_corr,
                            dv_match,
                            dv_arrival_scored,
                            dv_capture,
                            total_dv,
                            v_inf_glacia_mag_pre,
                            v_inf_glacia_mag_post,
                            v_inf_glacia_vec_pre,
                            v_inf_glacia_vec_post,
                        )

    return best_dv, best_sol, min_arrival_pre_dsm, min_arrival_v_after, top_solutions


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    # You can widen this to assignment range once the logic is validated:
    # np.arange(60000, 61096, 10)
    t1_values = np.arange(60610, 60620, 1)

    print("Using", cpu_count(), "cores")
    print(f"Arrival tolerance target: <= {MATCH_TOL:.3f} km/s")
    print("Terminal DSM enabled:", ALLOW_TERMINAL_DSM)
    print("Glacia capture counted in score:", COUNT_GLACIA_CAPTURE_IN_TOTAL)

    with Pool(cpu_count()) as pool:
        results = pool.map(evaluate_t1, t1_values)

    best_dv = 1e9
    best_solution = None
    global_min_arrival_pre_dsm = 1e9
    global_min_v_after = None
    global_top = []

    for dv, sol, min_arr_pre, v_after, local_top in results:
        if min_arr_pre < global_min_arrival_pre_dsm:
            global_min_arrival_pre_dsm = min_arr_pre
            global_min_v_after = v_after

        if sol is not None and dv < best_dv:
            best_dv = dv
            best_solution = sol

        # Merge heaps
        for item in local_top:
            if len(global_top) < 10:
                heapq.heappush(global_top, item)
            elif item[0] > global_top[0][0]:
                heapq.heapreplace(global_top, item)

    print("\n===== BEST TRAJECTORY =====")

    if best_solution:
        (
            t1,
            t2,
            t3,
            rp,
            dv_dep,
            dv_corr,
            dv_match,
            dv_arr_scored,
            dv_capture,
            dv_tot,
            v_inf_mag_pre,
            v_inf_mag_post,
            v_inf_vec_pre,
            v_inf_vec_post,
        ) = best_solution

        print(f"t1: {t1}, t2: {t2}, t3: {t3}, rp: {rp}")
        print(f"Departure ΔV        : {dv_dep:.3f} km/s")
        print(f"Correction ΔV       : {dv_corr:.3f} km/s")
        print(f"Terminal DSM ΔV     : {dv_match:.3f} km/s")
        print(f"Arrival ΔV (scored) : {dv_arr_scored:.3f} km/s")
        print(f"Total ΔV (scored)   : {dv_tot:.3f} km/s")
        print(f"Glacia capture ΔV   : {dv_capture:.3f} km/s (reference only)")

        print("\n--- Arrival at Glacia ---")
        print(f"Approach |v_inf| before DSM: {v_inf_mag_pre:.3f} km/s")
        print(f"Vector before DSM          : [{v_inf_vec_pre[0]:.3f}, {v_inf_vec_pre[1]:.3f}] km/s")
        print(f"Approach |v_inf| after DSM : {v_inf_mag_post:.3f} km/s")
        print(f"Vector after DSM           : [{v_inf_vec_post[0]:.3f}, {v_inf_vec_post[1]:.3f}] km/s")
        print(f"Tolerance satisfied        : {v_inf_mag_post <= MATCH_TOL + 1e-12}")
    else:
        print("No valid trajectory found")

    print(f"\nMinimum raw arrival |v_inf| before DSM: {global_min_arrival_pre_dsm:.6f} km/s")
    if global_min_v_after is not None:
        print("v_after at raw-min case:", global_min_v_after)

    print("\n===== TOP 10 TRAJECTORIES =====")
    top_sorted = sorted(global_top, key=lambda x: -x[0])

    for i, (neg_dv, sol) in enumerate(top_sorted, 1):
        (
            total_dv,
            t1,
            t2,
            t3,
            rp,
            dv_dep,
            dv_corr,
            dv_match,
            dv_arr_scored,
            dv_capture,
            v_inf_mag_pre,
            v_inf_mag_post,
            v_inf_vec_pre,
            v_inf_vec_post,
        ) = sol

        print(f"\n--- Rank {i} ---")
        print(f"t1: {t1}, t2: {t2}, t3: {t3}, rp: {rp}")
        print(f"Departure ΔV        : {dv_dep:.3f} km/s")
        print(f"Correction ΔV       : {dv_corr:.3f} km/s")
        print(f"Terminal DSM ΔV     : {dv_match:.3f} km/s")
        print(f"Arrival ΔV (scored) : {dv_arr_scored:.3f} km/s")
        print(f"Total ΔV (scored)   : {total_dv:.3f} km/s")
        print(f"Glacia capture ΔV   : {dv_capture:.3f} km/s (reference only)")
        print(f"Approach before DSM : {v_inf_mag_pre:.3f} km/s")
        print(f"Approach after DSM  : {v_inf_mag_post:.3f} km/s")
        print(f"Vector before DSM   : [{v_inf_vec_pre[0]:.3f}, {v_inf_vec_pre[1]:.3f}] km/s")
        print(f"Vector after DSM    : [{v_inf_vec_post[0]:.3f}, {v_inf_vec_post[1]:.3f}] km/s")
