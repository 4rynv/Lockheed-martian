import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

from veridian_structured import (
    BASE_DIR,
    DAY_SEC,
    MATCH_TOL_KMS,
    MU_STAR,
    MU_VENTUS,
    OUTPUT_PATH,
    R_VENTUS,
    build_interp,
    dv_departure,
    gravity_assist,
    lambert_solver,
    terminal_match_dv,
)


@dataclass
class MissionTrajectory:
    t1: float
    tof1: float
    tof2: float
    altitude: float
    t2: float
    t3: float
    sign: int
    r1: np.ndarray
    r2: np.ndarray
    r3: np.ndarray
    v1_leg1: np.ndarray
    v2_leg1: np.ndarray
    v1_leg2: np.ndarray
    v2_leg2: np.ndarray
    dv_depart: float
    dv_corr: float
    dv_match: float


@dataclass
class BurnEvent:
    name: str
    time_mjd: float
    position_heliocentric: np.ndarray
    delta_v: float


def parse_args():
    parser = argparse.ArgumentParser(description="Create Veridian mission animations")
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        default=OUTPUT_PATH,
        help="CSV containing departure_mjd,tof_ventus,altitude,tof_glacia",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "outputs" / "animations",
        help="Directory for animation files",
    )
    parser.add_argument("--fps", type=int, default=20, help="Animation FPS")
    parser.add_argument("--dpi", type=int, default=140, help="Animation DPI")
    parser.add_argument("--main-step", type=float, default=4.0, help="Days per frame for main animation")
    parser.add_argument(
        "--main-flyby-window",
        type=float,
        default=8.0,
        help="Days around Ventus encounter rendered with finer sampling in main animation",
    )
    parser.add_argument(
        "--main-flyby-step",
        type=float,
        default=0.2,
        help="Fine days-per-frame used near Ventus in main animation",
    )
    parser.add_argument("--flyby-window", type=float, default=120.0, help="Days before/after Ventus encounter")
    parser.add_argument("--flyby-step", type=float, default=0.02, help="Days per frame for flyby zoom")
    parser.add_argument(
        "--flyby-max-radius",
        type=float,
        default=2.0e6,
        help="Max Ventus-relative radius (km) shown for hyperbola construction",
    )
    parser.add_argument(
        "--flyby-samples",
        type=int,
        default=2400,
        help="Dense samples used to construct Ventus flyby hyperbola",
    )
    parser.add_argument("--arrival-window", type=float, default=180.0, help="Days before Glacia arrival")
    parser.add_argument("--arrival-step", type=float, default=1.0, help="Days per frame for arrival zoom")
    return parser.parse_args()


def load_best_trajectory_row(csv_path: Path):
    data = pd.read_csv(csv_path)
    if data.empty:
        raise ValueError(f"No rows found in {csv_path}")
    row = data.iloc[0]
    return float(row["departure_mjd"]), float(row["tof_ventus"]), float(row["altitude"]), float(row["tof_glacia"])


def resolve_nominal_trajectory(t1: float, tof1: float, tof2: float, altitude: float):
    rC, vC = build_interp("Caelus")
    rV, vV = build_interp("Ventus")
    rG, vG = build_interp("Glacia")

    t2 = t1 + tof1
    t3 = t2 + tof2

    r1 = np.asarray(rC(t1), dtype=float)
    r2 = np.asarray(rV(t2), dtype=float)
    r3 = np.asarray(rG(t3), dtype=float)

    leg1 = lambert_solver(r1, r2, tof1 * DAY_SEC, MU_STAR)
    leg2 = lambert_solver(r2, r3, tof2 * DAY_SEC, MU_STAR)
    if leg1 is None or leg2 is None:
        raise RuntimeError("Lambert solve failed for the selected best trajectory.")

    v1_leg1, v2_leg1 = leg1
    v1_leg2, v2_leg2 = leg2

    vc1 = np.asarray(vC(t1), dtype=float)
    vp = np.asarray(vV(t2), dtype=float)
    vg3 = np.asarray(vG(t3), dtype=float)

    dv_dep = dv_departure(np.linalg.norm(v1_leg1 - vc1))
    v_inf_in = v2_leg1 - vp
    incoming_helio_speed = np.linalg.norm(vp + v_inf_in)
    rp = R_VENTUS + altitude

    best_sign = +1
    best_corr = float("inf")
    best_v_after = None
    for sign in (+1, -1):
        ga = gravity_assist(v_inf_in, vp, rp, MU_VENTUS, sign=sign)
        if ga is None:
            continue
        v_after, _ = ga
        if np.linalg.norm(v_after) <= incoming_helio_speed:
            continue
        dv_corr = np.linalg.norm(v1_leg2 - v_after)
        if dv_corr < best_corr:
            best_corr = dv_corr
            best_sign = sign
            best_v_after = v_after

    if best_v_after is None:
        raise RuntimeError("Could not resolve a valid flyby branch for Ventus.")

    dv_match, _ = terminal_match_dv(v2_leg2 - vg3, MATCH_TOL_KMS)

    return MissionTrajectory(
        t1=t1,
        tof1=tof1,
        tof2=tof2,
        altitude=altitude,
        t2=t2,
        t3=t3,
        sign=best_sign,
        r1=r1,
        r2=r2,
        r3=r3,
        v1_leg1=v1_leg1,
        v2_leg1=v2_leg1,
        v1_leg2=v1_leg2,
        v2_leg2=v2_leg2,
        dv_depart=float(dv_dep),
        dv_corr=float(best_corr),
        dv_match=float(dv_match),
    )


def two_body_ode(_, y):
    x, y_pos, vx, vy = y
    r2 = x * x + y_pos * y_pos
    r3 = r2 * np.sqrt(r2)
    ax = -MU_STAR * x / r3
    ay = -MU_STAR * y_pos / r3
    return [vx, vy, ax, ay]


def propagate_segment(r0: np.ndarray, v0: np.ndarray, rel_days: np.ndarray):
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


def make_time_grid(start: float, end: float, step: float, must_include: list[float]):
    raw = np.arange(start, end + 0.5 * step, step)
    combined = np.concatenate([raw, np.asarray(must_include, dtype=float)])
    combined = combined[(combined >= start) & (combined <= end)]
    grid = np.unique(np.round(combined, 8))
    return np.sort(grid)


def build_burn_events(traj: MissionTrajectory):
    return [
        BurnEvent(
            name="Departure burn",
            time_mjd=traj.t1,
            position_heliocentric=traj.r1.copy(),
            delta_v=traj.dv_depart,
        ),
        BurnEvent(
            name="Ventus correction burn",
            time_mjd=traj.t2,
            position_heliocentric=traj.r2.copy(),
            delta_v=traj.dv_corr,
        ),
        BurnEvent(
            name="Glacia terminal DSM",
            time_mjd=traj.t3,
            position_heliocentric=traj.r3.copy(),
            delta_v=traj.dv_match,
        ),
    ]


def spacecraft_positions(times_mjd: np.ndarray, traj: MissionTrajectory):
    times_mjd = np.asarray(times_mjd, dtype=float)
    pos = np.zeros((len(times_mjd), 2), dtype=float)

    on_leg1 = times_mjd <= (traj.t2 + 1e-9)
    if np.any(on_leg1):
        rel1 = times_mjd[on_leg1] - traj.t1
        pos[on_leg1] = propagate_segment(traj.r1, traj.v1_leg1, rel1)

    on_leg2 = ~on_leg1
    if np.any(on_leg2):
        rel2 = times_mjd[on_leg2] - traj.t2
        pos[on_leg2] = propagate_segment(traj.r2, traj.v1_leg2, rel2)

    return pos


def _rotation_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def build_ventus_hyperbola_track(
    traj: MissionTrajectory,
    window_days: float,
    step_days: float,
    n_samples: int,
    max_radius_km: float,
):
    rV, vV = build_interp("Ventus")
    vp = np.asarray(vV(traj.t2), dtype=float)

    rp = R_VENTUS + traj.altitude
    v_inf_in = traj.v2_leg1 - vp
    v_inf_mag = np.linalg.norm(v_inf_in)
    if v_inf_mag <= 0.0:
        raise RuntimeError("Invalid flyby: zero incoming v_inf at Ventus.")

    ga = gravity_assist(v_inf_in, vp, rp, MU_VENTUS, sign=traj.sign)
    if ga is None:
        raise RuntimeError("Invalid gravity-assist geometry for Ventus flyby.")
    v_after, _ = ga
    v_inf_out = v_after - vp

    e = 1.0 + (rp * v_inf_mag**2) / MU_VENTUS
    p = rp * (1.0 + e)
    theta_inf = np.arccos(-1.0 / e)

    # Limit far-field radius so the hyperbola shape is visible in zoom.
    max_radius_km = max(float(max_radius_km), 3.0 * rp)
    cos_theta_limit = (p / max_radius_km - 1.0) / e
    if cos_theta_limit <= -1.0:
        theta_max = theta_inf - 1e-3
    else:
        theta_max = min(theta_inf - 1e-3, np.arccos(np.clip(cos_theta_limit, -1.0, 1.0)))
    theta_max = max(theta_max, 0.4)

    theta = np.linspace(-theta_max, theta_max, max(200, int(n_samples)))
    r = p / (1.0 + e * np.cos(theta))
    pos_pf = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    vel_pf = np.sqrt(MU_VENTUS / p) * np.column_stack((-np.sin(theta), e + np.cos(theta)))

    u_in_des = v_inf_in / np.linalg.norm(v_inf_in)
    u_out_des = v_inf_out / np.linalg.norm(v_inf_out)

    def orient_candidate(flip_y: bool):
        pos = pos_pf.copy()
        vel = vel_pf.copy()
        if flip_y:
            pos[:, 1] *= -1.0
            vel[:, 1] *= -1.0

        u_in_model = vel[0] / np.linalg.norm(vel[0])
        ang = np.arctan2(u_in_des[1], u_in_des[0]) - np.arctan2(u_in_model[1], u_in_model[0])
        rot = _rotation_matrix(ang)
        pos = pos @ rot.T
        vel = vel @ rot.T

        u_out_model = vel[-1] / np.linalg.norm(vel[-1])
        score = float(np.dot(u_out_model, u_out_des))
        return score, pos

    score_a, rel_pos_a = orient_candidate(flip_y=False)
    score_b, rel_pos_b = orient_candidate(flip_y=True)
    rel_pos = rel_pos_a if score_a >= score_b else rel_pos_b

    # True anomaly -> hyperbolic anomaly -> physical time from periapsis
    a = -MU_VENTUS / (v_inf_mag**2)
    q = np.sqrt((e - 1.0) / (e + 1.0)) * np.tan(theta / 2.0)
    q = np.clip(q, -1.0 + 1e-12, 1.0 - 1e-12)
    F = 2.0 * np.arctanh(q)
    M = e * np.sinh(F) - F
    t_rel_sec = np.sqrt(((-a) ** 3) / MU_VENTUS) * M
    times = traj.t2 + t_rel_sec / DAY_SEC

    # Keep requested temporal window around encounter.
    mask = np.abs(times - traj.t2) <= float(window_days)
    if np.count_nonzero(mask) >= 3:
        times = times[mask]
        rel_pos = rel_pos[mask]

    order = np.argsort(times)
    times = times[order]
    rel_pos = rel_pos[order]

    # Resample to uniform frame step (in days).
    if step_days > 0 and len(times) >= 2:
        frame_times = np.arange(times[0], times[-1] + 0.5 * step_days, step_days)
        frame_times = np.unique(np.concatenate([frame_times, np.array([traj.t2])]))
        x = np.interp(frame_times, times, rel_pos[:, 0])
        y = np.interp(frame_times, times, rel_pos[:, 1])
        times = frame_times
        rel_pos = np.column_stack((x, y))

    return times, rel_pos


def spacecraft_positions_with_flyby_patch(
    times_mjd: np.ndarray,
    traj: MissionTrajectory,
    flyby_window_days: float,
    flyby_step_days: float,
    flyby_samples: int,
    flyby_max_radius_km: float,
):
    times_mjd = np.asarray(times_mjd, dtype=float)
    base = spacecraft_positions(times_mjd, traj)

    try:
        hyper_t, hyper_rel = build_ventus_hyperbola_track(
            traj,
            window_days=flyby_window_days,
            step_days=max(0.01, min(flyby_step_days, 0.1)),
            n_samples=flyby_samples,
            max_radius_km=flyby_max_radius_km,
        )
    except Exception:
        return base

    if len(hyper_t) < 3:
        return base

    rV, _ = build_interp("Ventus")
    hyper_abs = np.asarray(rV(hyper_t), dtype=float) + hyper_rel

    patch_start, patch_end = float(hyper_t[0]), float(hyper_t[-1])
    x_interp = np.interp(times_mjd, hyper_t, hyper_abs[:, 0], left=np.nan, right=np.nan)
    y_interp = np.interp(times_mjd, hyper_t, hyper_abs[:, 1], left=np.nan, right=np.nan)

    mask_core = (times_mjd >= patch_start) & (times_mjd <= patch_end)
    if np.any(mask_core):
        base[mask_core, 0] = x_interp[mask_core]
        base[mask_core, 1] = y_interp[mask_core]

    # Smoothly blend at patch edges to avoid visible kinks.
    blend_days = max(0.3, 2.0 * flyby_step_days)
    pre_mask = (times_mjd >= patch_start - blend_days) & (times_mjd < patch_start)
    post_mask = (times_mjd > patch_end) & (times_mjd <= patch_end + blend_days)

    if np.any(pre_mask):
        alpha = (times_mjd[pre_mask] - (patch_start - blend_days)) / blend_days
        hyper_pre = np.column_stack((x_interp[pre_mask], y_interp[pre_mask]))
        valid = np.isfinite(hyper_pre).all(axis=1)
        base_pre = base[pre_mask].copy()
        base_pre[valid] = (1.0 - alpha[valid, None]) * base_pre[valid] + alpha[valid, None] * hyper_pre[valid]
        base[pre_mask] = base_pre

    if np.any(post_mask):
        alpha = 1.0 - (times_mjd[post_mask] - patch_end) / blend_days
        hyper_post = np.column_stack((x_interp[post_mask], y_interp[post_mask]))
        valid = np.isfinite(hyper_post).all(axis=1)
        base_post = base[post_mask].copy()
        base_post[valid] = (1.0 - alpha[valid, None]) * base_post[valid] + alpha[valid, None] * hyper_post[valid]
        base[post_mask] = base_post

    return base


def get_planet_tracks(times_mjd: np.ndarray):
    tracks = {}
    for body in ["Aether", "Caelus", "Ventus", "Glacia"]:
        r_body, _ = build_interp(body)
        tracks[body] = np.asarray(r_body(times_mjd), dtype=float)
    return tracks


def save_animation(anim: FuncAnimation, output_stem: Path, fps: int, dpi: int):
    mp4_path = output_stem.with_suffix(".mp4")
    gif_path = output_stem.with_suffix(".gif")
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(mp4_path, writer=writer, dpi=dpi)
        return mp4_path
    except Exception:
        writer = PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=dpi)
        return gif_path


def animate_heliocentric(
    times,
    sc_track,
    planet_tracks,
    burn_events,
    output_stem,
    fps,
    dpi,
):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")

    body_colors = {
        "Aether": "#f39c12",
        "Caelus": "#00b894",
        "Ventus": "#74b9ff",
        "Glacia": "#a29bfe",
    }

    star = ax.scatter([0], [0], color="gold", marker="*", s=160, label="Veridian")
    _ = star

    lines = {}
    points = {}
    for body, color in body_colors.items():
        line, = ax.plot([], [], color=color, lw=1.0, alpha=0.45)
        point = ax.scatter([], [], color=color, s=22, label=body)
        lines[body] = line
        points[body] = point

    sc_line, = ax.plot([], [], color="white", lw=1.8, label="Spacecraft")
    sc_point = ax.scatter([], [], color="#00ffff", s=42, zorder=6)

    burn_static = []
    if burn_events:
        for ev in burn_events:
            marker = ax.scatter(
                [ev.position_heliocentric[0]],
                [ev.position_heliocentric[1]],
                color="red",
                marker="x",
                s=70,
                linewidths=2.0,
                zorder=7,
            )
            burn_static.append(marker)

    burn_flash = ax.scatter([], [], color="red", s=110, marker="o", edgecolors="white", linewidths=0.8, zorder=8)
    burn_flash.set_visible(False)

    all_points = [sc_track] + [planet_tracks[b] for b in body_colors]
    lim = 1.1 * np.max(np.abs(np.vstack(all_points)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Mission Veridian: Heliocentric Trajectory")
    ax.legend(loc="upper right", fontsize=8)

    mjd_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10)
    burn_text = ax.text(0.02, 0.91, "", transform=ax.transAxes, fontsize=10, color="#ff7675")

    def update(i):
        sc_line.set_data(sc_track[: i + 1, 0], sc_track[: i + 1, 1])
        sc_point.set_offsets(sc_track[i])

        for body in body_colors:
            track = planet_tracks[body]
            lines[body].set_data(track[: i + 1, 0], track[: i + 1, 1])
            points[body].set_offsets(track[i])

        active_event = None
        for ev in burn_events:
            if np.isclose(times[i], ev.time_mjd, atol=1e-6):
                active_event = ev
                break

        if active_event is None:
            burn_flash.set_visible(False)
            burn_text.set_text("")
        else:
            burn_flash.set_offsets(active_event.position_heliocentric)
            burn_flash.set_visible(True)
            burn_text.set_text(f"{active_event.name} | ΔV {active_event.delta_v:.3f} km/s")

        mjd_text.set_text(f"MJD {times[i]:.3f}")
        return [
            sc_line,
            sc_point,
            mjd_text,
            burn_text,
            burn_flash,
            *burn_static,
            *lines.values(),
            *points.values(),
        ]

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / fps, blit=False)
    out_path = save_animation(anim, output_stem, fps=fps, dpi=dpi)
    plt.close(fig)
    return out_path


def animate_planet_stationary(
    times,
    sc_track_abs,
    planet_track_abs,
    planet_name,
    event_time,
    burn_label,
    burn_delta_v,
    output_stem,
    fps,
    dpi,
):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")

    rel_sc = sc_track_abs - planet_track_abs
    event_idx = int(np.argmin(np.abs(times - event_time)))

    pre_line, = ax.plot([], [], color="#55efc4", lw=1.8, label="Inbound")
    post_line, = ax.plot([], [], color="#ff7675", lw=1.8, label="Outbound")
    sc_point = ax.scatter([], [], color="white", s=45, zorder=6)
    ax.scatter([0], [0], color="#74b9ff", s=80, label=f"{planet_name} (stationary)")
    burn_marker = ax.scatter([rel_sc[event_idx, 0]], [rel_sc[event_idx, 1]], color="red", marker="x", s=85, linewidths=2.0, label="Burn")
    burn_flash = ax.scatter([], [], color="red", s=110, marker="o", edgecolors="white", linewidths=0.8, zorder=7)
    burn_flash.set_visible(False)

    lim = 1.1 * np.max(np.abs(rel_sc))
    lim = max(lim, 2.0e6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"x relative to {planet_name} (km)")
    ax.set_ylabel(f"y relative to {planet_name} (km)")
    ax.set_title(f"{planet_name}-Centered Encounter")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    mjd_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10)
    burn_text = ax.text(0.02, 0.91, "", transform=ax.transAxes, fontsize=10, color="#ff7675")

    def update(i):
        if i <= event_idx:
            pre_line.set_data(rel_sc[: i + 1, 0], rel_sc[: i + 1, 1])
            post_line.set_data([], [])
        else:
            pre_line.set_data(rel_sc[: event_idx + 1, 0], rel_sc[: event_idx + 1, 1])
            post_line.set_data(rel_sc[event_idx : i + 1, 0], rel_sc[event_idx : i + 1, 1])

        sc_point.set_offsets(rel_sc[i])

        if i == event_idx:
            burn_flash.set_offsets(rel_sc[event_idx])
            burn_flash.set_visible(True)
            burn_text.set_text(f"{burn_label} | ΔV {burn_delta_v:.3f} km/s")
        else:
            burn_flash.set_visible(False)
            burn_text.set_text("")

        mjd_text.set_text(f"MJD {times[i]:.3f}")
        return [pre_line, post_line, sc_point, burn_marker, burn_flash, mjd_text, burn_text]

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / fps, blit=False)
    out_path = save_animation(anim, output_stem, fps=fps, dpi=dpi)
    plt.close(fig)
    return out_path


def animate_relative_stationary(
    times,
    rel_sc,
    planet_name,
    event_time,
    burn_label,
    burn_delta_v,
    output_stem,
    fps,
    dpi,
):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")

    event_idx = int(np.argmin(np.abs(times - event_time)))

    pre_line, = ax.plot([], [], color="#55efc4", lw=2.0, label="Inbound hyperbola")
    post_line, = ax.plot([], [], color="#ff7675", lw=2.0, label="Outbound hyperbola")
    sc_point = ax.scatter([], [], color="white", s=45, zorder=6)
    ax.scatter([0], [0], color="#74b9ff", s=90, label=f"{planet_name} (stationary)")
    burn_marker = ax.scatter([rel_sc[event_idx, 0]], [rel_sc[event_idx, 1]], color="red", marker="x", s=90, linewidths=2.0, label="Burn")
    burn_flash = ax.scatter([], [], color="red", s=120, marker="o", edgecolors="white", linewidths=0.8, zorder=7)
    burn_flash.set_visible(False)

    lim = 1.1 * np.max(np.abs(rel_sc))
    lim = max(lim, 4.0e5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"x relative to {planet_name} (km)")
    ax.set_ylabel(f"y relative to {planet_name} (km)")
    ax.set_title(f"{planet_name}-Centered Hyperbolic Flyby")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    mjd_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10)
    burn_text = ax.text(0.02, 0.91, "", transform=ax.transAxes, fontsize=10, color="#ff7675")

    def update(i):
        if i <= event_idx:
            pre_line.set_data(rel_sc[: i + 1, 0], rel_sc[: i + 1, 1])
            post_line.set_data([], [])
        else:
            pre_line.set_data(rel_sc[: event_idx + 1, 0], rel_sc[: event_idx + 1, 1])
            post_line.set_data(rel_sc[event_idx : i + 1, 0], rel_sc[event_idx : i + 1, 1])

        sc_point.set_offsets(rel_sc[i])

        if i == event_idx:
            burn_flash.set_offsets(rel_sc[event_idx])
            burn_flash.set_visible(True)
            burn_text.set_text(f"{burn_label} | ΔV {burn_delta_v:.3f} km/s")
        else:
            burn_flash.set_visible(False)
            burn_text.set_text("")

        mjd_text.set_text(f"MJD {times[i]:.3f}")
        return [pre_line, post_line, sc_point, burn_marker, burn_flash, mjd_text, burn_text]

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / fps, blit=False)
    out_path = save_animation(anim, output_stem, fps=fps, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t1, tof1, altitude, tof2 = load_best_trajectory_row(args.trajectory_csv)
    traj = resolve_nominal_trajectory(t1=t1, tof1=tof1, tof2=tof2, altitude=altitude)
    burn_events = build_burn_events(traj)

    print(
        "Loaded trajectory:",
        f"t1={traj.t1:.0f}, t2={traj.t2:.0f}, t3={traj.t3:.0f},",
        f"tof1={traj.tof1:.0f}, tof2={traj.tof2:.0f},",
        f"altitude={traj.altitude:.0f} km, sign={traj.sign:+d}",
    )
    print(
        "Burns:",
        f"Departure={traj.dv_depart:.3f} km/s,",
        f"Correction={traj.dv_corr:.3f} km/s,",
        f"Terminal DSM={traj.dv_match:.3f} km/s",
    )

    # 1) Full heliocentric animation with all planets
    times_main_coarse = make_time_grid(traj.t1, traj.t3, args.main_step, [traj.t2, traj.t3])
    flyby_focus_start = max(traj.t1, traj.t2 - args.main_flyby_window)
    flyby_focus_end = min(traj.t3, traj.t2 + args.main_flyby_window)
    times_main_fine = make_time_grid(
        flyby_focus_start, flyby_focus_end, args.main_flyby_step, [traj.t2]
    )
    times_main = np.unique(np.concatenate([times_main_coarse, times_main_fine]))
    sc_main = spacecraft_positions_with_flyby_patch(
        times_main,
        traj,
        flyby_window_days=args.main_flyby_window,
        flyby_step_days=args.main_flyby_step,
        flyby_samples=args.flyby_samples,
        flyby_max_radius_km=args.flyby_max_radius,
    )
    planets_main = get_planet_tracks(times_main)
    out_main = animate_heliocentric(
        times_main,
        sc_main,
        planets_main,
        burn_events,
        args.output_dir / "01_heliocentric_trajectory",
        fps=args.fps,
        dpi=args.dpi,
    )
    print("Saved:", out_main)

    # 2) Ventus-centered flyby zoom (hyperbolic patch in Ventus frame)
    times_flyby, rel_sc_flyby = build_ventus_hyperbola_track(
        traj,
        window_days=args.flyby_window,
        step_days=args.flyby_step,
        n_samples=args.flyby_samples,
        max_radius_km=args.flyby_max_radius,
    )
    out_flyby = animate_relative_stationary(
        times_flyby,
        rel_sc_flyby,
        planet_name="Ventus",
        event_time=traj.t2,
        burn_label="Ventus correction burn",
        burn_delta_v=traj.dv_corr,
        output_stem=args.output_dir / "02_ventus_flyby_stationary",
        fps=args.fps,
        dpi=args.dpi,
    )
    print("Saved:", out_flyby)

    # 3) Glacia-centered arrival zoom
    arrival_start = max(traj.t1, traj.t3 - args.arrival_window)
    arrival_end = traj.t3
    times_arrival = make_time_grid(arrival_start, arrival_end, args.arrival_step, [traj.t3])
    sc_arrival = spacecraft_positions(times_arrival, traj)
    rG, _ = build_interp("Glacia")
    glacia_track = np.asarray(rG(times_arrival), dtype=float)
    out_arrival = animate_planet_stationary(
        times_arrival,
        sc_arrival,
        glacia_track,
        planet_name="Glacia",
        event_time=traj.t3,
        burn_label="Glacia terminal DSM",
        burn_delta_v=traj.dv_match,
        output_stem=args.output_dir / "03_glacia_arrival_stationary",
        fps=args.fps,
        dpi=args.dpi,
    )
    print("Saved:", out_arrival)


if __name__ == "__main__":
    main()
