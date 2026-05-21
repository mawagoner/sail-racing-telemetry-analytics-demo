"""Sample telemetry generator for high-performance sailing demos.

The generator can optionally inject quality issues to simulate real-world feeds
with missing values, out-of-order packets, and duplicates.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .schema import normalize_telemetry_dataframe

KNOTS_PER_MPS = 1.943844

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


def _build_timestamps(start: datetime, count: int, step_seconds: int) -> list[datetime]:
    return [start + timedelta(seconds=idx * step_seconds) for idx in range(count)]


def _generate_course_weather(total_points: int, rng: np.random.Generator) -> pd.DataFrame:
    phase = np.linspace(0.0, 3.3 * np.pi, total_points)
    wind_shift = np.cumsum(rng.normal(0.0, 0.12, total_points))

    true_wind_speed = 13.6 + 1.9 * np.sin(phase) + rng.normal(0.0, 0.35, total_points)
    true_wind_direction = (212.0 + 10.0 * np.sin(phase * 0.55 + 0.7) + wind_shift) % 360.0
    gust_knots = true_wind_speed + np.maximum(0.3, rng.normal(1.3, 0.55, total_points))
    sea_state = np.clip(1.1 + (true_wind_speed - 10.0) * 0.22 + rng.normal(0.0, 0.08, total_points), 1.0, 5.0)

    return pd.DataFrame(
        {
            "true_wind_speed": true_wind_speed,
            "true_wind_direction": true_wind_direction,
            "gust_knots": gust_knots,
            "sea_state": sea_state,
        }
    )


def _inject_data_issues(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    issue_rate: float,
) -> pd.DataFrame:
    """Inject missing, duplicate, and out-of-order rows for ingestion testing."""
    dirty = frame.copy()
    if dirty.empty:
        return dirty

    issue_rate = float(max(0.0, min(issue_rate, 0.5)))
    issue_size = max(1, int(len(dirty) * issue_rate))

    numeric_candidates = [
        "speed_knots",
        "vmg",
        "true_wind_speed",
        "true_wind_angle",
        "heading",
        "heel_angle",
        "foil_cant",
    ]
    numeric_candidates = [column for column in numeric_candidates if column in dirty.columns]
    missing_per_column = max(1, issue_size // max(2, len(numeric_candidates)))

    for column in numeric_candidates:
        idx = rng.choice(dirty.index.to_numpy(), size=min(missing_per_column, len(dirty)), replace=False)
        dirty.loc[idx, column] = np.nan

    if "maneuver_type" in dirty.columns:
        idx = rng.choice(dirty.index.to_numpy(), size=min(max(1, issue_size // 6), len(dirty)), replace=False)
        dirty.loc[idx, "maneuver_type"] = "NONE "
    if "course_side" in dirty.columns:
        idx = rng.choice(dirty.index.to_numpy(), size=min(max(1, issue_size // 6), len(dirty)), replace=False)
        dirty.loc[idx, "course_side"] = "STARBOARD "
    if "team_name" in dirty.columns:
        idx = rng.choice(dirty.index.to_numpy(), size=min(max(1, issue_size // 6), len(dirty)), replace=False)
        dirty.loc[idx, "team_name"] = ""

    if "timestamp" in dirty.columns:
        idx = rng.choice(dirty.index.to_numpy(), size=min(max(1, issue_size // 5), len(dirty)), replace=False)
        offsets = rng.integers(-8, 9, size=len(idx))
        dirty.loc[idx, "timestamp"] = pd.to_datetime(dirty.loc[idx, "timestamp"], utc=True) + pd.to_timedelta(
            offsets,
            unit="s",
        )

    duplicate_count = min(max(1, issue_size // 5), len(dirty))
    duplicate_idx = rng.choice(dirty.index.to_numpy(), size=duplicate_count, replace=False)
    duplicates = dirty.loc[duplicate_idx].copy()
    dirty = pd.concat([dirty, duplicates], ignore_index=True)

    shuffle_count = min(max(2, issue_size // 3), len(dirty))
    shuffle_idx = rng.choice(dirty.index.to_numpy(), size=shuffle_count, replace=False)
    shuffled_block = dirty.loc[shuffle_idx].sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
    dirty.loc[shuffle_idx] = shuffled_block.to_numpy()
    return dirty.reset_index(drop=True)


def generate_sample_telemetry(
    num_boats: int = 5,
    legs_per_boat: int = 6,
    samples_per_leg: int = 120,
    sampling_interval_seconds: int = 2,
    seed: int = 42,
    inject_data_issues: bool = False,
    issue_rate: float = 0.02,
    normalize: bool = True,
) -> pd.DataFrame:
    """Generate simulated race telemetry aligned to the canonical schema."""
    if num_boats < 1:
        raise ValueError("num_boats must be >= 1")
    if legs_per_boat < 1:
        raise ValueError("legs_per_boat must be >= 1")
    if samples_per_leg < 20:
        raise ValueError("samples_per_leg must be >= 20")
    if sampling_interval_seconds < 1:
        raise ValueError("sampling_interval_seconds must be >= 1")

    rng = np.random.default_rng(seed)
    total_points = legs_per_boat * samples_per_leg
    base_time = datetime.now(tz=timezone.utc).replace(microsecond=0)
    timestamps = _build_timestamps(base_time, total_points, sampling_interval_seconds)

    weather = _generate_course_weather(total_points, rng)
    course_length = 2200.0
    course_width = 560.0

    frames: list[pd.DataFrame] = []

    for boat_idx in range(num_boats):
        boat_id = f"Boat-{boat_idx + 1}"
        team_name = TEAM_NAMES[boat_idx % len(TEAM_NAMES)]

        performance_factor = 1.0 + rng.normal(0.0, 0.04)
        current_sign = int(rng.choice([-1, 1]))
        current_x = rng.uniform(-80.0, 80.0) + (boat_idx - (num_boats - 1) / 2.0) * 17.0
        current_y = 0.0

        for leg_number in range(1, legs_per_boat + 1):
            global_start = (leg_number - 1) * samples_per_leg
            global_end = global_start + samples_per_leg

            is_upwind = leg_number % 2 == 1
            leg_mode = "Upwind" if is_upwind else "Downwind"
            maneuver_label = "tack" if is_upwind else "gybe"

            if is_upwind:
                twa_base = current_sign * rng.uniform(34.0, 47.0)
                speed_base = rng.uniform(20.1, 24.5) * performance_factor
                heel_base = rng.uniform(10.5, 17.5)
                foil_base = rng.uniform(66.0, 80.0)
            else:
                twa_base = current_sign * rng.uniform(136.0, 162.0)
                speed_base = rng.uniform(23.0, 30.0) * performance_factor
                heel_base = rng.uniform(5.5, 11.0)
                foil_base = rng.uniform(56.0, 72.0)

            sample_index = np.arange(samples_per_leg)
            phase = np.linspace(0.0, 2.0 * np.pi, samples_per_leg)

            segment_weather = weather.iloc[global_start:global_end].reset_index(drop=True)
            wind_speed = segment_weather["true_wind_speed"].to_numpy() + rng.normal(0.0, 0.25, samples_per_leg)
            wind_direction = segment_weather["true_wind_direction"].to_numpy() + rng.normal(0.0, 1.1, samples_per_leg)
            gust_knots = np.maximum(
                wind_speed,
                segment_weather["gust_knots"].to_numpy() + rng.normal(0.0, 0.3, samples_per_leg),
            )
            sea_state = np.clip(
                segment_weather["sea_state"].to_numpy() + rng.normal(0.0, 0.05, samples_per_leg),
                1.0,
                5.0,
            )

            true_wind_angle = twa_base + rng.normal(0.0, 3.3, samples_per_leg)

            maneuver_center = int(samples_per_leg * rng.uniform(0.34, 0.66))
            maneuver_width = max(4, samples_per_leg // 18)
            transition_start = max(1, maneuver_center - maneuver_width // 2)
            transition_end = min(samples_per_leg - 2, maneuver_center + maneuver_width // 2)

            pre_angle = true_wind_angle[transition_start - 1]
            post_angle = -np.sign(pre_angle) * abs(pre_angle) * rng.uniform(0.93, 1.07)
            transition = np.linspace(pre_angle, post_angle, transition_end - transition_start + 1)
            transition += rng.normal(0.0, 1.2, transition.size)
            true_wind_angle[transition_start : transition_end + 1] = transition

            if transition_end + 1 < samples_per_leg:
                true_wind_angle[transition_end + 1 :] = (
                    np.abs(true_wind_angle[transition_end + 1 :]) * np.sign(post_angle)
                )

            wind_boost = 0.24 * (wind_speed - np.mean(wind_speed))
            speed = (
                speed_base
                + wind_boost
                + 1.3 * np.sin(phase + rng.uniform(-0.35, 0.35))
                + rng.normal(0.0, 0.46, samples_per_leg)
            )

            dip_magnitude = rng.uniform(4.0, 7.0)
            dip_scale = max(1.0, maneuver_width / 1.8)
            dip_profile = np.exp(-0.5 * ((sample_index - maneuver_center) / dip_scale) ** 2)
            speed = np.clip(speed - dip_magnitude * dip_profile, 2.0, None)

            target_y = course_length if is_upwind else 0.0
            side_bias = (1.0 if current_sign > 0 else -1.0) * rng.uniform(130.0, 260.0)
            end_x = side_bias + (boat_idx - (num_boats - 1) / 2.0) * 22.0 + rng.normal(0.0, 15.0)

            x_track = np.linspace(current_x, end_x, samples_per_leg)
            y_track = np.linspace(current_y, target_y, samples_per_leg)

            weave_amp = rng.uniform(14.0, 40.0) * (1.15 if is_upwind else 0.8)
            x_track += weave_amp * np.sin(phase + rng.uniform(-0.3, 0.3))
            y_track += rng.normal(0.0, 7.0, samples_per_leg)

            lane_shift = np.linspace(0.0, current_sign * rng.uniform(35.0, 80.0), samples_per_leg - maneuver_center)
            x_track[maneuver_center:] += lane_shift
            x_track = np.clip(x_track, -course_width, course_width)

            dx = np.gradient(x_track)
            dy = np.gradient(y_track)
            heading = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

            abs_twa = np.abs(true_wind_angle)
            heel_angle = (
                heel_base
                + 0.08 * (abs_twa if is_upwind else np.maximum(0.0, 170.0 - abs_twa))
                + rng.normal(0.0, 1.0, samples_per_leg)
                + dip_profile * (2.0 if is_upwind else 1.2)
            )
            foil_cant = (
                foil_base
                + rng.normal(0.0, 1.7, samples_per_leg)
                + dip_profile * 1.3
                + 0.15 * (wind_speed - np.mean(wind_speed))
            )

            heel_angle = np.clip(heel_angle, 0.0, 36.0)
            foil_cant = np.clip(foil_cant, 40.0, 95.0)
            wind_speed = np.clip(wind_speed, 6.0, None)

            vmg = speed * np.cos(np.deg2rad(true_wind_angle))
            maneuver_type = np.full(samples_per_leg, "none", dtype=object)
            maneuver_type[maneuver_center] = maneuver_label
            course_side = np.where(true_wind_angle >= 0.0, "starboard", "port")

            frames.append(
                pd.DataFrame(
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
            )

            current_x = float(x_track[-1])
            current_y = float(y_track[-1])
            current_sign *= -1

    telemetry = pd.concat(frames, ignore_index=True)
    if inject_data_issues:
        telemetry = _inject_data_issues(telemetry, rng=rng, issue_rate=issue_rate)

    if normalize:
        return normalize_telemetry_dataframe(telemetry)
    return telemetry.reset_index(drop=True)
