"""Canonical telemetry schema and normalization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
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

REQUIRED_COLUMNS = [
    "timestamp",
    "boat_id",
    "leg_id",
    "speed_knots",
    "true_wind_speed",
    "true_wind_angle",
    "heading",
    "heel_angle",
    "foil_cant",
    "x_position",
    "y_position",
]

OPTIONAL_COLUMNS = [column for column in CANONICAL_COLUMNS if column not in REQUIRED_COLUMNS]

NUMERIC_COLUMNS = [
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
    "sea_state",
]

STRING_COLUMNS = [
    "boat_id",
    "team_name",
    "leg_id",
    "leg_mode",
    "maneuver_type",
    "course_side",
]


def get_required_columns() -> list[str]:
    """Return required schema columns."""
    return REQUIRED_COLUMNS.copy()


def get_optional_columns() -> list[str]:
    """Return optional schema columns."""
    return OPTIONAL_COLUMNS.copy()


def validate_telemetry_dataframe(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate telemetry shape and values, returning success and user-friendly errors."""
    errors: list[str] = []

    if not isinstance(df, pd.DataFrame):
        return False, ["Input must be a pandas DataFrame."]

    if df.empty:
        return False, ["Telemetry dataframe is empty."]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return False, errors

    parsed_timestamp = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if parsed_timestamp.isna().all():
        errors.append("Column 'timestamp' cannot be parsed as datetime.")

    for column in [
        "speed_knots",
        "true_wind_speed",
        "true_wind_angle",
        "heading",
        "heel_angle",
        "foil_cant",
        "x_position",
        "y_position",
    ]:
        coerced = pd.to_numeric(df[column], errors="coerce")
        if coerced.isna().all():
            errors.append(f"Column '{column}' cannot be coerced to numeric values.")

    if df["boat_id"].astype(str).str.strip().eq("").all():
        errors.append("Column 'boat_id' cannot be empty for all rows.")

    return len(errors) == 0, errors


def _normalize_strings(frame: pd.DataFrame) -> pd.DataFrame:
    for column in STRING_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()

    for column in ["boat_id", "team_name", "leg_id", "leg_mode", "course_side"]:
        frame.loc[frame[column].str.lower().isin(["", "nan", "none", "nat"]), column] = np.nan

    frame.loc[frame["maneuver_type"].str.lower().isin(["", "nan", "nat"]), "maneuver_type"] = np.nan
    frame["maneuver_type"] = frame["maneuver_type"].fillna("none")
    frame["maneuver_type"] = frame["maneuver_type"].str.lower()

    default_course_side = pd.Series(
        np.where(frame["true_wind_angle"] >= 0.0, "starboard", "port"),
        index=frame.index,
    )
    frame["course_side"] = frame["course_side"].fillna(default_course_side)
    frame["course_side"] = frame["course_side"].str.lower()

    return frame


def _normalize_optional_values(frame: pd.DataFrame) -> pd.DataFrame:
    frame["team_name"] = frame["team_name"].fillna(frame["boat_id"].map(lambda value: f"{value} Team"))

    frame["leg_mode"] = frame["leg_mode"].replace("", np.nan)
    default_leg_mode = pd.Series(
        np.where(frame["true_wind_angle"].abs() < 90.0, "Upwind", "Downwind"),
        index=frame.index,
    )
    frame["leg_mode"] = frame["leg_mode"].fillna(default_leg_mode)
    frame["leg_mode"] = frame["leg_mode"].map(
        lambda value: "Upwind" if str(value).strip().lower() == "upwind" else "Downwind"
    )

    frame["vmg"] = frame["vmg"].fillna(frame["speed_knots"] * np.cos(np.deg2rad(frame["true_wind_angle"])))
    frame["true_wind_direction"] = frame["true_wind_direction"].fillna(
        (frame["heading"] + frame["true_wind_angle"]) % 360.0
    )
    frame["gust_knots"] = frame["gust_knots"].fillna(frame["true_wind_speed"] + 1.0)
    frame["sea_state"] = frame["sea_state"].fillna(
        np.clip(1.2 + (frame["true_wind_speed"] - 10.0) * 0.2, 1.0, 5.0)
    )
    return frame


def normalize_telemetry_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize telemetry data to canonical schema and stable dtypes.

    The function fills optional values with deterministic defaults and drops rows
    missing required values after coercion.
    """

    is_valid, errors = validate_telemetry_dataframe(df)
    if not is_valid:
        raise ValueError("Telemetry validation failed: " + " | ".join(errors))

    frame = df.copy()
    for column in CANONICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = _normalize_strings(frame)
    frame = _normalize_optional_values(frame)

    required_subset = REQUIRED_COLUMNS + ["timestamp"]
    frame = frame.dropna(subset=required_subset)

    if frame.empty:
        raise ValueError(
            "No valid telemetry rows remain after normalization. "
            "Check timestamps and required numeric fields."
        )

    frame = frame.sort_values(["timestamp", "boat_id"]).reset_index(drop=True)
    return frame[CANONICAL_COLUMNS]


def describe_schema() -> dict[str, Any]:
    """Return schema metadata for docs and debugging."""
    return {
        "canonical_columns": CANONICAL_COLUMNS.copy(),
        "required_columns": REQUIRED_COLUMNS.copy(),
        "optional_columns": OPTIONAL_COLUMNS.copy(),
        "numeric_columns": NUMERIC_COLUMNS.copy(),
    }
