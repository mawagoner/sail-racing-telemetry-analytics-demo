"""Generate sample sail racing telemetry for dashboard demos."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "boat_id",
    "speed_knots",
    "true_wind_speed",
    "true_wind_angle",
    "heading",
    "heel_angle",
    "foil_cant",
    "vmg",
    "maneuver_type",
    "leg_id",
]


def _build_timestamps(start_time: datetime, points: int, freq_seconds: int) -> list[datetime]:
    return [start_time + timedelta(seconds=freq_seconds * idx) for idx in range(points)]


def generate_sample_telemetry(
    num_boats: int = 3,
    leg_count: int = 6,
    points_per_leg: int = 120,
    freq_seconds: int = 2,
    seed: int = 42,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """Create realistic sample telemetry for multiple boats and race legs."""
    if num_boats < 1:
        raise ValueError("num_boats must be >= 1")
    if leg_count < 1:
        raise ValueError("leg_count must be >= 1")
    if points_per_leg < 20:
        raise ValueError("points_per_leg must be >= 20")
    if freq_seconds < 1:
        raise ValueError("freq_seconds must be >= 1")

    base_time = start_time or datetime.utcnow().replace(microsecond=0)
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []

    for boat_idx in range(num_boats):
        boat_id = f"Boat-{boat_idx + 1}"
        performance_factor = 1.0 + rng.normal(0.0, 0.035)
        current_sign = int(rng.choice([-1, 1]))
        sample_cursor = 0

        for leg_number in range(1, leg_count + 1):
            is_upwind = leg_number % 2 == 1
            maneuver_label = "tack" if is_upwind else "gybe"

            if is_upwind:
                twa_base = current_sign * rng.uniform(34.0, 46.0)
                speed_base = rng.uniform(19.5, 24.0) * performance_factor
                heel_base = rng.uniform(11.0, 17.0)
                foil_base = rng.uniform(68.0, 80.0)
            else:
                twa_base = current_sign * rng.uniform(136.0, 162.0)
                speed_base = rng.uniform(23.0, 30.0) * performance_factor
                heel_base = rng.uniform(6.0, 11.0)
                foil_base = rng.uniform(58.0, 72.0)

            sample_index = np.arange(points_per_leg)
            phase = np.linspace(0.0, 2.0 * np.pi, points_per_leg)

            speed = (
                speed_base
                + 1.3 * np.sin(phase + rng.uniform(-0.35, 0.35))
                + rng.normal(0.0, 0.45, points_per_leg)
            )
            true_wind_speed = (
                rng.normal(14.2, 1.3, points_per_leg)
                + 0.45 * np.sin(np.linspace(0.0, np.pi, points_per_leg))
            )
            true_wind_angle = twa_base + rng.normal(0.0, 3.8, points_per_leg)

            heading_origin = rng.uniform(0.0, 360.0)
            heading_delta = rng.normal(0.0, 0.9, points_per_leg)
            heading = (heading_origin + np.cumsum(heading_delta + 0.08 * np.sign(true_wind_angle))) % 360

            heel_angle = heel_base + rng.normal(0.0, 1.2, points_per_leg)
            foil_cant = foil_base + rng.normal(0.0, 2.0, points_per_leg)

            maneuver_center = int(points_per_leg * rng.uniform(0.35, 0.65))
            maneuver_width = max(4, points_per_leg // 18)
            transition_start = max(1, maneuver_center - maneuver_width // 2)
            transition_end = min(points_per_leg - 2, maneuver_center + maneuver_width // 2)

            pre_angle = true_wind_angle[transition_start - 1]
            post_angle = -np.sign(pre_angle) * abs(pre_angle) * rng.uniform(0.92, 1.08)
            transition = np.linspace(pre_angle, post_angle, transition_end - transition_start + 1)
            transition += rng.normal(0.0, 1.5, transition.size)
            true_wind_angle[transition_start : transition_end + 1] = transition

            if transition_end + 1 < points_per_leg:
                true_wind_angle[transition_end + 1 :] = (
                    np.abs(true_wind_angle[transition_end + 1 :]) * np.sign(post_angle)
                )

            dip_magnitude = rng.uniform(4.0, 7.0)
            dip_scale = max(1.0, maneuver_width / 1.8)
            dip_profile = np.exp(-0.5 * ((sample_index - maneuver_center) / dip_scale) ** 2)
            speed = np.clip(speed - dip_magnitude * dip_profile, 2.0, None)
            heel_angle = np.clip(heel_angle + dip_profile * (2.0 if is_upwind else 1.2), 0.0, 35.0)
            foil_cant = np.clip(foil_cant + dip_profile * 1.3, 40.0, 95.0)
            true_wind_speed = np.clip(true_wind_speed, 6.0, None)

            maneuver_type = np.full(points_per_leg, "none", dtype=object)
            maneuver_type[maneuver_center] = maneuver_label

            leg_start_time = base_time + timedelta(seconds=freq_seconds * sample_cursor)
            timestamp = _build_timestamps(leg_start_time, points_per_leg, freq_seconds)
            vmg = speed * np.cos(np.deg2rad(true_wind_angle))

            leg_df = pd.DataFrame(
                {
                    "timestamp": timestamp,
                    "boat_id": boat_id,
                    "speed_knots": np.round(speed, 3),
                    "true_wind_speed": np.round(true_wind_speed, 3),
                    "true_wind_angle": np.round(true_wind_angle, 3),
                    "heading": np.round(heading, 3),
                    "heel_angle": np.round(heel_angle, 3),
                    "foil_cant": np.round(foil_cant, 3),
                    "vmg": np.round(vmg, 3),
                    "maneuver_type": maneuver_type,
                    "leg_id": str(leg_number),
                }
            )
            frames.append(leg_df)

            sample_cursor += points_per_leg
            current_sign *= -1

    telemetry = pd.concat(frames, ignore_index=True)
    telemetry = telemetry[REQUIRED_COLUMNS].sort_values(["timestamp", "boat_id"]).reset_index(drop=True)
    return telemetry
