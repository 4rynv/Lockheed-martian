import argparse
import heapq
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy import units as u
from poliastro.iod import izzo
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# =========================================================
# CONSTANTS (assignment + coordinator clarification)
# =========================================================
DAY_SEC = 86400.0
AU_KM = 1.496e8
THERMAL_LIMIT_AU = 0.4
MAX_MISSION_DAYS = 2922.0

MU_STAR = 1.393e11
MU_CAELUS = 3.986e5
MU_VENTUS = 1.266e8

R_CAELUS = 7200.0
R_VENTUS = 65000.0
R_CAELUS_PARK = R_CAELUS + 500.0

V_CIRC_CAELUS = np.sqrt(MU_CAELUS / R_CAELUS_PARK)
V_ESC_CAELUS = np.sqrt(2.0 * MU_CAELUS / R_CAELUS_PARK)

MATCH_TOL_KMS = 0.1
POSITION_TOL_KM = 1.0e4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT  # kept for compatibility with animation script imports
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

EPHEMERIS_PATH = DATA_DIR / "veridian_ephemeris.csv"
OUTPUT_PATH = OUTPUT_DIR / "optimal_trajectory.csv"
SPACECRAFT_EPHEMERIS_OUTPUT_PATH = OUTPUT_DIR / "spacecraft_ephemeris_5day.csv"


# =========================================================
# TASK 1: Ephemeris load
# =========================================================
df = pd.read_csv(EPHEMERIS_PATH)
time_grid = df["MJD"].values


def build_interp(body: str):
    r_interp = interp1d(time_grid, df[[f"{body}_x", f"{body}_y"]].values, axis=0, kind="cubic")
    v_interp = interp1d(time_grid, df[[f"{body}_vx", f"{body}_vy"]].values, axis=0, kind="cubic")
    return r_interp, v_interp


rC, vC = build_interp("Caelus")
rV, vV = build_interp("Ventus")
rG, vG = build_interp("Glacia")


@dataclass
class Candidate:
    total_dv: float
    departure_mjd: float
    ventus_mjd: float
    glacia_mjd: float
    tof_ventus: float
    tof_glacia: float
    flyby_altitude: float
    flyby_radius: float
    flyby_sign: int
    dv_depart: float
    dv_corr: float
    dv_match: float
    v_preflyby_helio: np.ndarray
    v_postflyby_precorr_helio: np.ndarray
    v_postflyby_postcorr_helio: np.ndarray
    v_inf_pre_mag: float
    v_inf_post_mag: float
    v_inf_pre_vec: np.ndarray
    v_inf_post_vec: np.ndarray
    position_error_km: float


# =========================================================
# TASK 2: Lambert solver
# [v1, v2] = lambert_solver(r1, r2, tof, mu)
# =========================================================
def _extract_lambert_velocities(raw_solutions):
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


def lambert_solver(r1: np.ndarray, r2: np.ndarray, tof_seconds: float, mu: float):
    r1_q = np.array([r1[0], r1[1], 0.0]) * u.km
    r2_q = np.array([r2[0], r2[1], 0.0]) * u.km
    mu_q = mu * u.km**3 / u.s**2

    try:
        raw_solutions = list(izzo.lambert(mu_q, r1_q, r2_q, float(tof_seconds) * u.s))
    except Exception:
        return None

    if not raw_solutions:
        return None

    v1_q, v2_q = _extract_lambert_velocities(raw_solutions)
    if v1_q is None:
        return None

    v1 = np.asarray(v1_q.to_value(u.km / u.s), dtype=float)[:2]
    v2 = np.asarray(v2_q.to_value(u.km / u.s), dtype=float)[:2]
    return v1, v2


# =========================================================
# TASK 3: Gravity assist model
# [v_out, delta] = gravity_assist(v_inf_in, v_p, r_p, mu_p)
# =========================================================
def rotate_2d(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def gravity_assist(v_inf_in: np.ndarray, v_p: np.ndarray, r_p: float, mu_p: float, sign: int = 1):
    v_inf = np.linalg.norm(v_inf_in)
    if v_inf == 0.0:
        return None

    arg = 1.0 / (1.0 + (r_p * v_inf**2) / mu_p)
    if abs(arg) > 1.0:
        return None

    delta = 2.0 * np.arcsin(arg)
    v_inf_out = rotate_2d(v_inf_in, sign * delta)
    v_out = v_p + v_inf_out
    return v_out, delta


# =========================================================
# TASK 5: Delta-V formulas for this mission definition
# (rendezvous-only score, no Glacia circular orbit insertion)
# =========================================================
def dv_departure(v_inf_depart_mag: float) -> float:
    return np.sqrt(v_inf_depart_mag**2 + V_ESC_CAELUS**2) - V_CIRC_CAELUS


def terminal_match_dv(v_inf_vec: np.ndarray, tol_kms: float):
    v_inf_mag = np.linalg.norm(v_inf_vec)
    if v_inf_mag <= tol_kms:
        return 0.0, v_inf_vec.copy()

    v_inf_post = v_inf_vec * (tol_kms / v_inf_mag)
    burn_vec = v_inf_post - v_inf_vec
    return np.linalg.norm(burn_vec), v_inf_post


def thermal_safe(r_vec: np.ndarray) -> bool:
    return np.linalg.norm(r_vec) >= THERMAL_LIMIT_AU * AU_KM


def direct_hohmann_baseline():
    # Circular heliocentric baseline between Caelus (0.87 AU) and Glacia (2.75 AU)
    a1 = 0.87 * AU_KM
    a2 = 2.75 * AU_KM
    v1 = np.sqrt(MU_STAR / a1)
    v2 = np.sqrt(MU_STAR / a2)
    dv1 = v1 * (np.sqrt((2.0 * a2) / (a1 + a2)) - 1.0)
    dv2 = v2 * (1.0 - np.sqrt((2.0 * a1) / (a1 + a2)))
    tof_days = np.pi * np.sqrt(((a1 + a2) * 0.5) ** 3 / MU_STAR) / DAY_SEC
    return abs(dv1) + abs(dv2), tof_days


def two_body_ode(_, y):
    x, y_pos, vx, vy = y
    r2 = x * x + y_pos * y_pos
    r3 = r2 * np.sqrt(r2)
    ax = -MU_STAR * x / r3
    ay = -MU_STAR * y_pos / r3
    return [vx, vy, ax, ay]


def propagate_heliocentric_segment(r0: np.ndarray, v0: np.ndarray, rel_days: np.ndarray):
    rel_days = np.asarray(rel_days, dtype=float)
    if rel_days.size == 0:
        return np.empty((0, 2))
    if np.any(rel_days < -1e-10):
        raise ValueError("Propagation received negative relative times.")

    t_eval = rel_days * DAY_SEC
    t_end = float(np.max(t_eval))
    if t_end == 0.0:
        return np.repeat(r0.reshape(1, 2), len(rel_days), axis=0)

    y0 = [r0[0], r0[1], v0[0], v0[1]]
    sol = solve_ivp(
        two_body_ode,
        (0.0, t_end),
        y0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-9,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(f"State propagation failed: {sol.message}")
    return sol.y[:2].T


def build_search_grid(quick: bool):
    if quick:
        # Fast debug grid
        departure_dates = np.arange(60610, 60620, 1)
        tof_to_ventus = np.arange(200, 350, 5)
        tof_to_glacia = np.arange(300, 700, 5)
        flyby_altitudes = np.arange(2000, 20001, 2000)
    else:
        # Assignment grid (Task 4)
        departure_dates = np.arange(60000, 61096, 10)
        tof_to_ventus = np.arange(200, 801, 10)
        tof_to_glacia = np.arange(200, 801, 10)
        flyby_altitudes = np.arange(2000, 20001, 1000)

    return departure_dates, tof_to_ventus, tof_to_glacia, flyby_altitudes


def _push_top_k(heap_store, cand: Candidate, top_k: int):
    item = (
        -cand.total_dv,
        cand.departure_mjd,
        cand.tof_ventus,
        cand.tof_glacia,
        cand.flyby_altitude,
        cand.flyby_sign,
        cand,
    )
    if len(heap_store) < top_k:
        heapq.heappush(heap_store, item)
    elif item[0] > heap_store[0][0]:
        heapq.heapreplace(heap_store, item)


def _sorted_candidates(heap_store):
    return [item[-1] for item in sorted(heap_store, key=lambda x: -x[0])]


def evaluate_departure(payload):
    (
        t1,
        tof_to_ventus,
        tof_to_glacia,
        flyby_altitudes,
        allow_terminal_dsm,
        max_total_dv,
        top_k,
    ) = payload

    best_candidate: Optional[Candidate] = None
    local_top = []
    local_min_raw_arrival = float("inf")

    r1 = np.asarray(rC(t1), dtype=float)
    v1 = np.asarray(vC(t1), dtype=float)
    if not thermal_safe(r1):
        return best_candidate, local_min_raw_arrival, []

    for tof1 in tof_to_ventus:
        t2 = t1 + tof1
        if t2 > time_grid[-1]:
            continue

        r2 = np.asarray(rV(t2), dtype=float)
        v2 = np.asarray(vV(t2), dtype=float)
        if not thermal_safe(r2):
            continue

        leg1 = lambert_solver(r1, r2, tof1 * DAY_SEC, MU_STAR)
        if leg1 is None:
            continue
        v1_leg1, v2_leg1 = leg1

        v_inf_depart_vec = v1_leg1 - v1
        dv_dep = dv_departure(np.linalg.norm(v_inf_depart_vec))

        v_inf_in = v2_leg1 - v2
        if np.linalg.norm(v_inf_in) == 0.0:
            continue

        incoming_helio_speed = np.linalg.norm(v2 + v_inf_in)

        # Precompute leg-2 once for this t2
        leg2_options = []
        for tof2 in tof_to_glacia:
            t3 = t2 + tof2
            if t3 > time_grid[-1] or (t3 - t1) > MAX_MISSION_DAYS:
                continue

            r3 = np.asarray(rG(t3), dtype=float)
            v3 = np.asarray(vG(t3), dtype=float)
            if not thermal_safe(r3):
                continue

            leg2 = lambert_solver(r2, r3, tof2 * DAY_SEC, MU_STAR)
            if leg2 is None:
                continue
            v1_leg2, v2_leg2 = leg2

            # In this patched-conic setup, Lambert endpoint matches planet position
            r3_propagated = propagate_heliocentric_segment(r2, v1_leg2, np.array([float(tof2)]))[0]
            position_error = np.linalg.norm(r3_propagated - r3)
            if position_error > POSITION_TOL_KM:
                continue

            v_inf_pre_vec = v2_leg2 - v3
            v_inf_pre_mag = np.linalg.norm(v_inf_pre_vec)
            local_min_raw_arrival = min(local_min_raw_arrival, v_inf_pre_mag)

            if allow_terminal_dsm:
                dv_match, v_inf_post_vec = terminal_match_dv(v_inf_pre_vec, MATCH_TOL_KMS)
                v_inf_post_mag = np.linalg.norm(v_inf_post_vec)
            else:
                if v_inf_pre_mag > MATCH_TOL_KMS:
                    continue
                dv_match = 0.0
                v_inf_post_vec = v_inf_pre_vec
                v_inf_post_mag = v_inf_pre_mag

            leg2_options.append(
                (
                    tof2,
                    t3,
                    v1_leg2,
                    dv_match,
                    v_inf_pre_mag,
                    v_inf_post_mag,
                    v_inf_pre_vec,
                    v_inf_post_vec,
                    position_error,
                )
            )

        if not leg2_options:
            continue

        for altitude in flyby_altitudes:
            rp = R_VENTUS + altitude
            if rp < 67000.0:          # enforce minimum periapsis distance from Ventus centre
                continue

            v_inf_in_mag = np.linalg.norm(v_inf_in)
            if v_inf_in_mag == 0.0:
                continue

            arg = 1.0 / (1.0 + (rp * v_inf_in_mag**2) / MU_VENTUS)
            if abs(arg) > 1.0:
                continue
            delta_max = 2.0 * np.arcsin(arg)

            for sign in (+1, -1):
                for (
                    tof2,
                    t3,
                    v1_leg2,
                    dv_match,
                    v_inf_pre_mag,
                    v_inf_post_mag,
                    v_inf_pre_vec,
                    v_inf_post_vec,
                    position_error,
                ) in leg2_options:

                    v_inf_out_req = v1_leg2 - v2
                    v_inf_out_req_mag = np.linalg.norm(v_inf_out_req)
                    if v_inf_out_req_mag == 0.0:
                        continue

                    cos_delta_req = np.dot(v_inf_in, v_inf_out_req) / (v_inf_in_mag * v_inf_out_req_mag)
                    cos_delta_req = np.clip(cos_delta_req, -1.0, 1.0)
                    delta_req = np.arccos(cos_delta_req)

                    if delta_req <= delta_max:
                        dv_corr = 0.0
                        v_after = v1_leg2
                    else:
                        dv_corr = v_inf_in_mag * (delta_req - delta_max)
                        v_after = v1_leg2

                    total_dv = dv_dep + dv_corr + dv_match
                    if total_dv > max_total_dv:
                        continue

                    cand = Candidate(
                        total_dv=float(total_dv),
                        departure_mjd=float(t1),
                        ventus_mjd=float(t2),
                        glacia_mjd=float(t3),
                        tof_ventus=float(tof1),
                        tof_glacia=float(tof2),
                        flyby_altitude=float(altitude),
                        flyby_radius=float(rp),
                        flyby_sign=int(sign),
                        dv_depart=float(dv_dep),
                        dv_corr=float(dv_corr),
                        dv_match=float(dv_match),
                        v_preflyby_helio=np.asarray(v2_leg1, dtype=float),
                        v_postflyby_precorr_helio=np.asarray(v_after, dtype=float),
                        v_postflyby_postcorr_helio=np.asarray(v1_leg2, dtype=float),
                        v_inf_pre_mag=float(v_inf_pre_mag),
                        v_inf_post_mag=float(v_inf_post_mag),
                        v_inf_pre_vec=np.asarray(v_inf_pre_vec, dtype=float),
                        v_inf_post_vec=np.asarray(v_inf_post_vec, dtype=float),
                        position_error_km=float(position_error),
                    )

                    _push_top_k(local_top, cand, top_k)
                    if best_candidate is None or cand.total_dv < best_candidate.total_dv:
                        best_candidate = cand

    return best_candidate, local_min_raw_arrival, _sorted_candidates(local_top)


def save_optimal_csv(best: Candidate, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "departure_mjd": best.departure_mjd,
        "tof_ventus": best.tof_ventus,
        "altitude": best.flyby_altitude,
        "tof_glacia": best.tof_glacia,
        "deltaV_total": best.total_dv,
    }
    pd.DataFrame([row]).to_csv(output_path, index=False)


def save_spacecraft_ephemeris_5day(best: Candidate, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t1 = best.departure_mjd
    t2 = best.ventus_mjd
    t3 = best.glacia_mjd

    r1 = np.asarray(rC(t1), dtype=float)
    r2 = np.asarray(rV(t2), dtype=float)
    r3 = np.asarray(rG(t3), dtype=float)

    leg1 = lambert_solver(r1, r2, best.tof_ventus * DAY_SEC, MU_STAR)
    leg2 = lambert_solver(r2, r3, best.tof_glacia * DAY_SEC, MU_STAR)
    if leg1 is None or leg2 is None:
        raise RuntimeError("Could not reconstruct Lambert legs for spacecraft ephemeris export.")

    v1_leg1, _ = leg1
    v1_leg2, _ = leg2

    times = np.arange(t1, t3 + 1e-9, 5.0)
    if not np.isclose(times[-1], t3, atol=1e-9):
        times = np.append(times, t3)
    times = np.unique(np.round(times, 8))

    pos = np.zeros((len(times), 2), dtype=float)
    on_leg1 = times <= (t2 + 1e-9)
    if np.any(on_leg1):
        rel1 = times[on_leg1] - t1
        pos[on_leg1] = propagate_heliocentric_segment(r1, v1_leg1, rel1)

    on_leg2 = ~on_leg1
    if np.any(on_leg2):
        rel2 = times[on_leg2] - t2
        pos[on_leg2] = propagate_heliocentric_segment(r2, v1_leg2, rel2)

    ephem = pd.DataFrame(
        {
            "MJD": times,
            "spacecraft_x": pos[:, 0],
            "spacecraft_y": pos[:, 1],
            "spacecraft_z": np.zeros(len(times)),
        }
    )
    ephem.to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Mission Veridian structured solver (rendezvous mission)")
    parser.add_argument("--quick", action="store_true", help="Use reduced grid for fast debug runs")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of worker processes")
    parser.add_argument("--top-k", type=int, default=10, help="How many best trajectories to keep")
    parser.add_argument(
        "--max-total-dv",
        type=float,
        default=25.0,
        help="Maximum allowed scored mission delta-v (km/s)",
    )
    parser.add_argument(
        "--no-terminal-dsm",
        action="store_true",
        help="Disable terminal DSM; only natural arrivals within tolerance are accepted",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output CSV path")
    parser.add_argument(
        "--ephemeris-output",
        type=Path,
        default=SPACECRAFT_EPHEMERIS_OUTPUT_PATH,
        help="Output CSV path for spacecraft x,y,z every 5 days",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    allow_terminal_dsm = not args.no_terminal_dsm

    departures, tof_to_ventus, tof_to_glacia, flyby_altitudes = build_search_grid(args.quick)
    hohmann_dv, hohmann_tof = direct_hohmann_baseline()

    print("Using", args.workers, "workers")
    print("Quick mode:", args.quick)
    print("Arrival tolerance target:", MATCH_TOL_KMS, "km/s")
    print("Terminal DSM enabled:", allow_terminal_dsm)
    print("Grid sizes:", len(departures), len(tof_to_ventus), len(tof_to_glacia), len(flyby_altitudes))
    print(
        "Direct Hohmann baseline (heliocentric circular):",
        f"ΔV={hohmann_dv:.3f} km/s, TOF={hohmann_tof:.1f} days",
    )

    payloads = [
        (
            float(t1),
            tof_to_ventus,
            tof_to_glacia,
            flyby_altitudes,
            allow_terminal_dsm,
            args.max_total_dv,
            args.top_k,
        )
        for t1 in departures
    ]

    with Pool(args.workers) as pool:
        results = pool.map(evaluate_departure, payloads)

    best_global: Optional[Candidate] = None
    top_global = []
    min_raw_arrival = float("inf")

    for best_local, min_arr_local, top_local in results:
        if min_arr_local < min_raw_arrival:
            min_raw_arrival = min_arr_local

        if best_local is not None and (best_global is None or best_local.total_dv < best_global.total_dv):
            best_global = best_local

        for cand in top_local:
            _push_top_k(top_global, cand, args.top_k)

    print("\n===== BEST TRAJECTORY =====")
    if best_global is None:
        print("No valid trajectory found on the selected grid.")
        return

    print(
        f"t1: {best_global.departure_mjd:.0f}, "
        f"t2: {best_global.ventus_mjd:.0f}, "
        f"t3: {best_global.glacia_mjd:.0f}, "
        f"rp: {best_global.flyby_radius:.0f} km, "
        f"altitude: {best_global.flyby_altitude:.0f} km, "
        f"sign: {best_global.flyby_sign:+d}"
    )
    print(f"Departure ΔV        : {best_global.dv_depart:.3f} km/s")
    print(f"Correction ΔV       : {best_global.dv_corr:.3f} km/s")
    print(f"Terminal DSM ΔV     : {best_global.dv_match:.3f} km/s")
    print(f"Arrival ΔV (scored) : {best_global.dv_match:.3f} km/s")
    print(f"Total ΔV (scored)   : {best_global.total_dv:.3f} km/s")
    
    G0 = 9.80665e-3   # km/s²
    M0 = 2500.0        # kg
    ISP = 300.0        # s

    m_final = M0 * np.exp(-best_global.total_dv / (ISP * G0))
    propellant_used = M0 - m_final

    print(f"\n--- Mass budget ---")
    print(f"Initial mass        : {M0:.1f} kg")
    print(f"Total ΔV            : {best_global.total_dv:.3f} km/s")
    print(f"Propellant used     : {propellant_used:.1f} kg")
    print(f"Final mass          : {m_final:.1f} kg")

    print("\n--- Ventus flyby velocities (heliocentric) ---")
    print(
        "Pre-flyby velocity            : "
        f"[{best_global.v_preflyby_helio[0]:.6f}, {best_global.v_preflyby_helio[1]:.6f}] km/s"
    )
    print(
        "Post-flyby pre-correction vel : "
        f"[{best_global.v_postflyby_precorr_helio[0]:.6f}, {best_global.v_postflyby_precorr_helio[1]:.6f}] km/s"
    )
    print(
        "Post-flyby post-correction vel: "
        f"[{best_global.v_postflyby_postcorr_helio[0]:.6f}, {best_global.v_postflyby_postcorr_helio[1]:.6f}] km/s"
    )

    print("\n--- Arrival at Glacia ---")
    print(f"Approach |v_inf| before DSM: {best_global.v_inf_pre_mag:.3f} km/s")
    print(f"Vector before DSM          : [{best_global.v_inf_pre_vec[0]:.3f}, {best_global.v_inf_pre_vec[1]:.3f}] km/s")
    print(f"Approach |v_inf| after DSM : {best_global.v_inf_post_mag:.3f} km/s")
    print(f"Vector after DSM           : [{best_global.v_inf_post_vec[0]:.3f}, {best_global.v_inf_post_vec[1]:.3f}] km/s")
    print(f"Velocity tolerance met     : {best_global.v_inf_post_mag <= MATCH_TOL_KMS + 1e-12}")
    print(f"Position tolerance met     : {best_global.position_error_km <= POSITION_TOL_KM}")

    if np.isfinite(min_raw_arrival):
        print(f"\nMinimum raw arrival |v_inf| before DSM on grid: {min_raw_arrival:.6f} km/s")

    top_sorted = _sorted_candidates(top_global)
    print("\n===== TOP TRAJECTORIES =====")
    for rank, cand in enumerate(top_sorted, start=1):
        print(
            f"Rank {rank}: "
            f"t1={cand.departure_mjd:.0f}, "
            f"tof1={cand.tof_ventus:.0f}, "
            f"alt={cand.flyby_altitude:.0f}, "
            f"tof2={cand.tof_glacia:.0f}, "
            f"ΔV={cand.total_dv:.3f}, "
            f"|v_inf_post|={cand.v_inf_post_mag:.3f}"
        )

    save_optimal_csv(best_global, args.output)
    print("\nWrote optimal trajectory row to:", args.output)
    save_spacecraft_ephemeris_5day(best_global, args.ephemeris_output)
    print("Wrote spacecraft 5-day ephemeris to:", args.ephemeris_output)


if __name__ == "__main__":
    main()