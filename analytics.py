"""Analytics utilities for sail racing telemetry dashboards."""

from __future__ import annotations

from typing import Any

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

MANDATORY_COLUMNS = [
    "timestamp",
    "boat_id",
    "speed_knots",
    "true_wind_speed",
    "true_wind_angle",
    "heading",
    "heel_angle",
    "foil_cant",
    "leg_id",
]

NUMERIC_COLUMNS = [
    "speed_knots",
    "true_wind_speed",
    "true_wind_angle",
    "heading",
    "heel_angle",
    "foil_cant",
    "vmg",
]


def validate_and_normalize_telemetry(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Validate required telemetry fields and coerce stable types."""
    if telemetry is None or telemetry.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    missing_columns = [col for col in MANDATORY_COLUMNS if col not in telemetry.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required telemetry columns: {missing_text}")

    df = telemetry.copy()

    if "maneuver_type" not in df.columns:
        df["maneuver_type"] = "none"

    if "vmg" not in df.columns:
        df["vmg"] = np.nan

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    df["boat_id"] = df["boat_id"].astype(str).str.strip()
    df["leg_id"] = df["leg_id"].astype(str).str.strip()
    df["maneuver_type"] = (
        df["maneuver_type"].fillna("none").astype(str).str.strip().str.lower()
    )

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    missing_vmg = df["vmg"].isna()
    df.loc[missing_vmg, "vmg"] = df.loc[missing_vmg, "speed_knots"] * np.cos(
        np.deg2rad(df.loc[missing_vmg, "true_wind_angle"])
    )

    required_for_rows = MANDATORY_COLUMNS + ["vmg"]
    df = df.dropna(subset=required_for_rows)
    df = df.sort_values(["boat_id", "timestamp"]).reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return df[REQUIRED_COLUMNS]


def _estimate_maneuver_loss(
    speed_values: np.ndarray,
    event_index: int,
    baseline_points: int = 8,
    event_points: int = 4,
) -> tuple[float, float, float]:
    pre_slice = speed_values[max(0, event_index - baseline_points) : event_index]
    post_slice = speed_values[event_index + 1 : event_index + 1 + baseline_points]
    event_slice = speed_values[
        max(0, event_index - event_points) : event_index + event_points + 1
    ]

    if pre_slice.size == 0 or post_slice.size == 0 or event_slice.size == 0:
        return 0.0, 0.0, 0.0

    baseline_reference = np.concatenate([pre_slice[-3:], post_slice[:3]])
    baseline_speed = float(np.nanmean(baseline_reference))
    min_speed = float(np.nanmin(event_slice))
    speed_loss = max(0.0, baseline_speed - min_speed)
    return speed_loss, baseline_speed, min_speed


def detect_maneuvers(
    telemetry: pd.DataFrame,
    cooldown_points: int = 8,
    min_mean_abs_twa: float = 18.0,
) -> pd.DataFrame:
    """Detect tacks and gybes from TWA sign changes and estimate speed loss."""
    if telemetry.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "boat_id",
                "leg_id",
                "maneuver_type",
                "speed_loss_knots",
                "baseline_speed_knots",
                "min_speed_knots",
            ]
        )

    events: list[dict[str, Any]] = []

    for boat_id, group in telemetry.groupby("boat_id", sort=False):
        boat_df = group.sort_values("timestamp").reset_index(drop=True)
        smooth_twa = boat_df["true_wind_angle"].rolling(window=5, center=True, min_periods=1).mean()
        twa_sign = np.sign(smooth_twa)
        twa_sign = twa_sign.replace(0, np.nan).ffill().bfill()
        sign_change = twa_sign.ne(twa_sign.shift(1))
        candidate_positions = np.flatnonzero(sign_change.to_numpy())

        last_event = -cooldown_points
        speed_values = boat_df["speed_knots"].to_numpy()

        for pos in candidate_positions:
            if pos == 0:
                continue
            if (pos - last_event) < cooldown_points:
                continue

            window_start = max(0, pos - 6)
            window_end = min(len(boat_df), pos + 7)
            pre_window = smooth_twa.iloc[window_start:pos]
            post_window = smooth_twa.iloc[pos:window_end]
            if pre_window.empty or post_window.empty:
                continue

            pre_abs = float(pre_window.abs().mean())
            post_abs = float(post_window.abs().mean())
            if min(pre_abs, post_abs) < min_mean_abs_twa:
                continue

            mean_abs_angle = (pre_abs + post_abs) / 2.0
            maneuver_type = "gybe" if mean_abs_angle >= 90.0 else "tack"

            speed_loss, baseline_speed, min_speed = _estimate_maneuver_loss(speed_values, pos)
            event_row = boat_df.iloc[pos]
            events.append(
                {
                    "timestamp": event_row["timestamp"],
                    "boat_id": boat_id,
                    "leg_id": event_row["leg_id"],
                    "maneuver_type": maneuver_type,
                    "speed_loss_knots": round(speed_loss, 3),
                    "baseline_speed_knots": round(baseline_speed, 3),
                    "min_speed_knots": round(min_speed, 3),
                }
            )
            last_event = pos

    if not events:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "boat_id",
                "leg_id",
                "maneuver_type",
                "speed_loss_knots",
                "baseline_speed_knots",
                "min_speed_knots",
            ]
        )

    detected = pd.DataFrame(events).sort_values(["timestamp", "boat_id"]).reset_index(drop=True)
    return detected


def compute_kpis(
    telemetry: pd.DataFrame,
    maneuvers: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute high-level dashboard KPIs."""
    if telemetry.empty:
        return {
            "max_speed": 0.0,
            "avg_speed": 0.0,
            "best_vmg": 0.0,
            "avg_maneuver_loss": 0.0,
        }

    maneuver_df = maneuvers if maneuvers is not None else detect_maneuvers(telemetry)
    average_loss = (
        float(maneuver_df["speed_loss_knots"].mean()) if not maneuver_df.empty else 0.0
    )

    return {
        "max_speed": float(telemetry["speed_knots"].max()),
        "avg_speed": float(telemetry["speed_knots"].mean()),
        "best_vmg": float(telemetry["vmg"].max()),
        "avg_maneuver_loss": average_loss,
    }


def build_leg_comparison(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-leg metrics for boat-to-boat comparisons."""
    if telemetry.empty:
        return pd.DataFrame(
            columns=[
                "boat_id",
                "leg_id",
                "leg_mode",
                "duration_sec",
                "samples",
                "avg_speed_knots",
                "max_speed_knots",
                "avg_vmg",
                "best_vmg",
                "avg_true_wind_speed",
                "avg_heel_angle",
            ]
        )

    grouped = (
        telemetry.groupby(["boat_id", "leg_id"], as_index=False)
        .agg(
            leg_start=("timestamp", "min"),
            leg_end=("timestamp", "max"),
            samples=("timestamp", "count"),
            avg_speed_knots=("speed_knots", "mean"),
            max_speed_knots=("speed_knots", "max"),
            avg_vmg=("vmg", "mean"),
            best_vmg=("vmg", "max"),
            avg_true_wind_speed=("true_wind_speed", "mean"),
            avg_heel_angle=("heel_angle", "mean"),
            mean_abs_twa=("true_wind_angle", lambda values: float(np.abs(values).mean())),
        )
        .reset_index(drop=True)
    )

    grouped["duration_sec"] = (grouped["leg_end"] - grouped["leg_start"]).dt.total_seconds()
    grouped["leg_mode"] = np.where(grouped["mean_abs_twa"] < 90.0, "Upwind", "Downwind")

    grouped["_leg_sort"] = pd.to_numeric(grouped["leg_id"], errors="coerce")
    grouped = grouped.sort_values(["boat_id", "_leg_sort", "leg_id"]).drop(columns=["_leg_sort"])

    display_columns = [
        "boat_id",
        "leg_id",
        "leg_mode",
        "duration_sec",
        "samples",
        "avg_speed_knots",
        "max_speed_knots",
        "avg_vmg",
        "best_vmg",
        "avg_true_wind_speed",
        "avg_heel_angle",
    ]

    rounded = grouped[display_columns].copy()
    numeric_cols = [
        "duration_sec",
        "avg_speed_knots",
        "max_speed_knots",
        "avg_vmg",
        "best_vmg",
        "avg_true_wind_speed",
        "avg_heel_angle",
    ]
    rounded[numeric_cols] = rounded[numeric_cols].round(2)
    return rounded.reset_index(drop=True)


def build_maneuver_summary(maneuvers: pd.DataFrame) -> dict[str, float]:
    """Summarize tack/gybe counts and loss metrics."""
    if maneuvers.empty:
        return {
            "tack_count": 0,
            "gybe_count": 0,
            "avg_loss": 0.0,
            "max_loss": 0.0,
        }

    return {
        "tack_count": int((maneuvers["maneuver_type"] == "tack").sum()),
        "gybe_count": int((maneuvers["maneuver_type"] == "gybe").sum()),
        "avg_loss": float(maneuvers["speed_loss_knots"].mean()),
        "max_loss": float(maneuvers["speed_loss_knots"].max()),
    }
