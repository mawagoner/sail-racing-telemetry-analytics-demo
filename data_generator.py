"""Generate SailGP-style sample telemetry for dashboard demos."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "boat_id",
    "team_name",
    "leg_id",
    "leg_mode",
    "speed_knots",
    "vmg",
    "true_wind_speed",
    "true_wind_angle",
    "true_wind_direction",
    "gust_knots",
    "heading",
    "heel_angle",
    "foil_cant",
    "x_position",
    "y_position",
    "maneuver_type",
    "course_side",
    "sea_state",
]

TEAM_NAMES = [
    "Apex Wave Racing",
    "Harbor Velocity Team",
    "Northern Reach Sailing",
    "Blue Horizon GP",
    "Straitline Performance",
    "Pacific Foil Works",
    "Coastal Quantum Racing",
    "Tidal Edge United",
]


def _build_timestamps(start_time: datetime, points: int, freq_seconds: int) -> list[datetime]:
    return [start_time + timedelta(seconds=freq_seconds * idx) for idx in range(points)]


def _generate_course_weather(total_points: int, rng: np.random.Generator) -> pd.DataFrame:
    """Build shared course-level weather that all boats experience."""
    phase = np.linspace(0.0, 3.4 * np.pi, total_points)
    shift_component = np.cumsum(rng.normal(0.0, 0.12, total_points))

    true_wind_speed = 13.8 + 1.7 * np.sin(phase) + rng.normal(0.0, 0.35, total_points)
    true_wind_direction = (212.0 + 11.0 * np.sin(phase * 0.55 + 0.8) + shift_component) % 360
    gust_knots = true_wind_speed + np.maximum(0.3, rng.normal(1.3, 0.6, total_points))
    sea_state = np.clip(1.2 + (true_wind_speed - 10.0) * 0.22 + rng.normal(0.0, 0.1, total_points), 1.0, 5.0)

    weather = pd.DataFrame(
        {
            "true_wind_speed": np.round(true_wind_speed, 3),
            "true_wind_direction": np.round(true_wind_direction, 3),
            "gust_knots": np.round(gust_knots, 3),
            "sea_state": np.round(sea_state, 3),
        }
    )
    return weather


def generate_sample_telemetry(
    num_boats: int = 4,
    leg_count: int = 6,
    points_per_leg: int = 120,
    freq_seconds: int = 2,
    seed: int = 42,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """Create realistic multi-boat telemetry including race coordinates and weather."""
    if num_boats < 1:
        raise ValueError("num_boats must be >= 1")
    if leg_count < 1:
        raise ValueError("leg_count must be >= 1")
    if points_per_leg < 20:
        raise ValueError("points_per_leg must be >= 20")
    if freq_seconds < 1:
        raise ValueError("freq_seconds must be >= 1")

    base_time = start_time or datetime.utcnow().replace(microsecond=0)
    total_points = leg_count * points_per_leg
    timestamps = _build_timestamps(base_time, total_points, freq_seconds)

    rng = np.random.default_rng(seed)
    weather = _generate_course_weather(total_points, rng)
    frames: list[pd.DataFrame] = []

    course_length_m = 2200.0
    course_width_m = 520.0

    for boat_idx in range(num_boats):
        boat_id = f"Boat-{boat_idx + 1}"
        team_name = TEAM_NAMES[boat_idx % len(TEAM_NAMES)]

        performance_factor = 1.0 + rng.normal(0.0, 0.035)
        current_sign = int(rng.choice([-1, 1]))

        current_x = rng.uniform(-90.0, 90.0) + (boat_idx - (num_boats - 1) / 2.0) * 18.0
        current_y = 0.0

        for leg_number in range(1, leg_count + 1):
            global_start = (leg_number - 1) * points_per_leg
            global_end = global_start + points_per_leg

            is_upwind = leg_number % 2 == 1
            leg_mode = "Upwind" if is_upwind else "Downwind"
            maneuver_label = "tack" if is_upwind else "gybe"

            if is_upwind:
                twa_base = current_sign * rng.uniform(34.0, 47.0)
                speed_base = rng.uniform(20.0, 24.3) * performance_factor
                heel_base = rng.uniform(11.0, 17.0)
                foil_base = rng.uniform(66.0, 80.0)
            else:
                twa_base = current_sign * rng.uniform(136.0, 162.0)
                speed_base = rng.uniform(23.4, 30.2) * performance_factor
                heel_base = rng.uniform(5.5, 10.5)
                foil_base = rng.uniform(56.0, 71.0)

            sample_index = np.arange(points_per_leg)
            phase = np.linspace(0.0, 2.0 * np.pi, points_per_leg)

            segment_weather = weather.iloc[global_start:global_end].reset_index(drop=True)
            wind_speed = segment_weather["true_wind_speed"].to_numpy() + rng.normal(0.0, 0.25, points_per_leg)
            wind_direction = segment_weather["true_wind_direction"].to_numpy() + rng.normal(
                0.0, 1.1, points_per_leg
            )
            gust_knots = np.maximum(
                wind_speed,
                segment_weather["gust_knots"].to_numpy() + rng.normal(0.0, 0.3, points_per_leg),
            )
            sea_state = np.clip(
                segment_weather["sea_state"].to_numpy() + rng.normal(0.0, 0.06, points_per_leg),
                1.0,
                5.0,
            )

            true_wind_angle = twa_base + rng.normal(0.0, 3.4, points_per_leg)

            maneuver_center = int(points_per_leg * rng.uniform(0.33, 0.67))
            maneuver_width = max(4, points_per_leg // 18)
            transition_start = max(1, maneuver_center - maneuver_width // 2)
            transition_end = min(points_per_leg - 2, maneuver_center + maneuver_width // 2)

            pre_angle = true_wind_angle[transition_start - 1]
            post_angle = -np.sign(pre_angle) * abs(pre_angle) * rng.uniform(0.92, 1.08)
            transition = np.linspace(pre_angle, post_angle, transition_end - transition_start + 1)
            transition += rng.normal(0.0, 1.2, transition.size)
            true_wind_angle[transition_start : transition_end + 1] = transition
            if transition_end + 1 < points_per_leg:
                true_wind_angle[transition_end + 1 :] = (
                    np.abs(true_wind_angle[transition_end + 1 :]) * np.sign(post_angle)
                )

            wind_boost = 0.24 * (wind_speed - np.nanmean(wind_speed))
            speed = (
                speed_base
                + wind_boost
                + 1.35 * np.sin(phase + rng.uniform(-0.4, 0.4))
                + rng.normal(0.0, 0.45, points_per_leg)
            )

            dip_magnitude = rng.uniform(4.1, 7.1)
            dip_scale = max(1.0, maneuver_width / 1.8)
            dip_profile = np.exp(-0.5 * ((sample_index - maneuver_center) / dip_scale) ** 2)
            speed = np.clip(speed - dip_magnitude * dip_profile, 2.0, None)

            target_y = course_length_m if is_upwind else 0.0
            side_bias = (1.0 if current_sign > 0 else -1.0) * rng.uniform(130.0, 260.0)
            end_x = side_bias + (boat_idx - (num_boats - 1) / 2.0) * 22.0 + rng.normal(0.0, 15.0)

            x_track = np.linspace(current_x, end_x, points_per_leg)
            y_track = np.linspace(current_y, target_y, points_per_leg)

            weave_amplitude = rng.uniform(14.0, 40.0) * (1.15 if is_upwind else 0.8)
            x_track += weave_amplitude * np.sin(phase + rng.uniform(-0.3, 0.3))
            y_track += rng.normal(0.0, 7.0, points_per_leg)

            lane_shift = np.linspace(0.0, current_sign * rng.uniform(35.0, 80.0), points_per_leg - maneuver_center)
            x_track[maneuver_center:] += lane_shift

            x_track = np.clip(x_track, -course_width_m, course_width_m)

            dx = np.gradient(x_track)
            dy = np.gradient(y_track)
            heading = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

            abs_twa = np.abs(true_wind_angle)
            heel_angle = (
                heel_base
                + 0.08 * (abs_twa if is_upwind else np.maximum(0.0, 170.0 - abs_twa))
                + rng.normal(0.0, 1.0, points_per_leg)
                + dip_profile * (2.0 if is_upwind else 1.2)
            )
            foil_cant = (
                foil_base
                + rng.normal(0.0, 1.7, points_per_leg)
                + dip_profile * 1.3
                + 0.15 * (wind_speed - np.nanmean(wind_speed))
            )

            heel_angle = np.clip(heel_angle, 0.0, 36.0)
            foil_cant = np.clip(foil_cant, 40.0, 95.0)
            wind_speed = np.clip(wind_speed, 6.0, None)

            vmg = speed * np.cos(np.deg2rad(true_wind_angle))
            maneuver_type = np.full(points_per_leg, "none", dtype=object)
            maneuver_type[maneuver_center] = maneuver_label
            course_side = np.where(true_wind_angle >= 0.0, "starboard", "port")

            leg_df = pd.DataFrame(
                {
                    "timestamp": timestamps[global_start:global_end],
                    "boat_id": boat_id,
                    "team_name": team_name,
                    "leg_id": str(leg_number),
                    "leg_mode": leg_mode,
                    "speed_knots": np.round(speed, 3),
                    "vmg": np.round(vmg, 3),
                    "true_wind_speed": np.round(wind_speed, 3),
                    "true_wind_angle": np.round(true_wind_angle, 3),
                    "true_wind_direction": np.round(wind_direction % 360.0, 3),
                    "gust_knots": np.round(gust_knots, 3),
                    "heading": np.round(heading, 3),
                    "heel_angle": np.round(heel_angle, 3),
                    "foil_cant": np.round(foil_cant, 3),
                    "x_position": np.round(x_track, 3),
                    "y_position": np.round(y_track, 3),
                    "maneuver_type": maneuver_type,
                    "course_side": course_side,
                    "sea_state": np.round(sea_state, 3),
                }
            )
            frames.append(leg_df)

            current_x = float(x_track[-1])
            current_y = float(y_track[-1])
            current_sign *= -1

    telemetry = pd.concat(frames, ignore_index=True)
    telemetry = telemetry[REQUIRED_COLUMNS].sort_values(["timestamp", "boat_id"]).reset_index(drop=True)
    return telemetry
