"""Weather overlay analytics from telemetry signals."""

from __future__ import annotations

import pandas as pd

from ..utils import contiguous_periods, wrapped_angle_delta


def summarize_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate course-level weather at each timestamp."""
    if df.empty:
        return pd.DataFrame(
            columns=["timestamp", "true_wind_speed", "true_wind_direction", "gust_knots", "sea_state"]
        )

    weather = (
        df.groupby("timestamp", as_index=False)
        .agg(
            true_wind_speed=("true_wind_speed", "mean"),
            true_wind_direction=("true_wind_direction", "mean"),
            gust_knots=("gust_knots", "mean"),
            sea_state=("sea_state", "mean"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return weather


def detect_gust_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Detect high-gust periods from weather summary."""
    weather = summarize_weather(df)
    if weather.empty:
        return pd.DataFrame(columns=["period_type", "start", "end", "points", "max_gust_knots"])

    gust_threshold = float(weather["gust_knots"].quantile(0.90))
    gust_mask = weather["gust_knots"] >= gust_threshold
    periods = contiguous_periods(weather["timestamp"], gust_mask, "high_gust")

    if periods.empty:
        periods["max_gust_knots"] = []
        return periods

    max_gusts = []
    for _, row in periods.iterrows():
        window = weather[(weather["timestamp"] >= row["start"]) & (weather["timestamp"] <= row["end"])]
        max_gusts.append(float(window["gust_knots"].max()))
    periods["max_gust_knots"] = max_gusts
    return periods


def detect_wind_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """Detect wind-shift periods from wrapped direction deltas."""
    weather = summarize_weather(df)
    if weather.empty:
        return pd.DataFrame(columns=["period_type", "start", "end", "points", "max_shift_deg"])

    direction_delta = wrapped_angle_delta(weather["true_wind_direction"]).abs().fillna(0.0)
    shift_threshold = max(float(direction_delta.quantile(0.92)), 4.5)
    shift_mask = direction_delta >= shift_threshold
    periods = contiguous_periods(weather["timestamp"], shift_mask, "wind_shift")

    if periods.empty:
        periods["max_shift_deg"] = []
        return periods

    max_shift = []
    for _, row in periods.iterrows():
        window_mask = (weather["timestamp"] >= row["start"]) & (weather["timestamp"] <= row["end"])
        max_shift.append(float(direction_delta[window_mask].max()))
    periods["max_shift_deg"] = max_shift
    return periods


def prepare_weather_overlay(df: pd.DataFrame) -> dict[str, pd.DataFrame | float]:
    """Bundle weather summary and detected periods for plotting overlays."""
    weather = summarize_weather(df)
    gust_periods = detect_gust_periods(df)
    shift_periods = detect_wind_shifts(df)

    gust_threshold = float(weather["gust_knots"].quantile(0.90)) if not weather.empty else 0.0
    direction_delta = wrapped_angle_delta(weather["true_wind_direction"]).abs().fillna(0.0) if not weather.empty else pd.Series(dtype=float)
    shift_threshold = max(float(direction_delta.quantile(0.92)), 4.5) if not direction_delta.empty else 0.0

    return {
        "weather": weather,
        "gust_periods": gust_periods,
        "shift_periods": shift_periods,
        "gust_threshold": gust_threshold,
        "shift_threshold": shift_threshold,
    }
